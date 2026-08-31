# Reusable dataset builder: sample selection, metadata artifacts, manifest-based benchmark registration

Extract the sample-selection logic that currently lives scattered across
`src/datamodule.py` + `src/dataset.py` into a reusable, training-independent layer, so that a
short "filters only" spec can produce a dataset **metadata + manifest artifact** on disk — the
same shape training already logs to MLflow. That artifact then becomes the input to benchmark
registration (and later, to back-prediction of any ad-hoc dataset), instead of
`register_benchmark.py` globbing the filesystem and querying MongoDB itself.

Motivating use case: build the curated benchmark dataset for
`docs/plans/quality-tracking.md` (samples never in any model's training set) as a **frozen,
reviewable, version-hashed artifact** rather than a re-globbed side effect.

## Core decisions

1. **Sample selection is not a `Dataset`.** Building a sample list needs only glob → parse →
   group by `(plate, well, field)` → drop incomplete channel sets → attach labels → limit
   wells/label → balance samples/label. It needs no `__getitem__`, no transforms, no tiling, and
   **no `LabelEncoder`**. So it becomes plain functions, not a class. Instantiating a `Dataset`
   just to read `.samples` would force passing a `label_encoder` that's irrelevant here and drag
   in torch.
2. **The `LabelEncoder` is training-only and stays that way.** Labels are MongoDB treatment
   **strings** end to end: `t_label`/`p_label` are `VARCHAR(255)`, `predicted_class` is resolved
   to a string inside `predictor.predict()` via the model's own `class_names`, and quality
   compares strings (`AVG((p_label = t_label)::int)`). The encoder exists only to produce integer
   targets for the loss, and its `classes` list is logged as `class_names` so the predictor can
   map `argmax -> string`. Nothing in the builder, `register_benchmark`, or `compute_benchmark`
   encodes a label, so no index-alignment concern exists.
3. **Label-set coverage is still worth *warning* about (not failing on).** If a benchmark
   `t_label` string isn't in the model's `class_names` (treatment renamed in Mongo, stray
   whitespace, a treatment the model never trained on), the model cannot emit that string, so
   those samples score 0% by construction and deflate the baseline for reasons unrelated to model
   quality. Likewise, if the benchmark covers 3 treatments and the live window covers 4, the two
   macro-F1 numbers `quality_report` compares aren't strictly apples-to-apples. So: the builder
   records the observed label set in metadata, and `compute_benchmark`/`run_quality_report` print
   a warning on mismatch. A warning explains a weird number; a hard fail would block legitimate
   partial-coverage benchmarks.
4. **Scan the directory once.** Today it's globbed **three times** per training run — once in the
   label provider (purely to discover which `(plate, well)` pairs exist), then again in the train
   dataset, then again in the val dataset. Over drvfs on a network drive with thousands of files
   this is the slowest part of `setup()`. The scan becomes an explicit first step whose result is
   reused by everything downstream.
5. **Label providers shrink to pure lookups.** `get_labels(root_dir, exclude_wells)` conflates
   "discover wells from disk" with "ask MongoDB about wells". Split into
   `wells_from_index(...)` (disk) + `resolve_labels_for_wells(wells)` (MongoDB/dummy). This is
   already the shape of the third copy of this logic,
   `monitoring/backfill_labels.py::resolve_labels(wells)`.
6. **When consolidating the three copies, the `backfill_labels` version wins.** It validates the
   `Treatment` column exists and filters `None`/`""`/`NaN`. `MongoDBLabelsProvider.get_labels`
   does a bare `dict(zip(...))`, so a well with a null Treatment in Mongo puts `None`/`NaN` into
   `labels_dict`, and `LabelEncoder.fit`'s `sorted(set(...))` then raises
   `TypeError: '<' not supported between instances of 'NoneType' and 'str'` — a crash that points
   at the encoder instead of at the actual cause (bad Mongo data). Fixing this is part of the
   consolidation, not a separate task.
7. **Shared label code lives in `utils/`, selection code in `src/`.** `utils/` is already the
   established home for helpers shared across images (`utils/filename_parser.py` is used by both
   api and monitoring), and `monitoring/backfill_labels.py` must be able to import the label
   lookup from the lightweight monitoring image (no torch). Sample selection is dataset-domain
   logic used by training and the standalone builder, neither of which runs in the monitoring
   image, so it belongs in `src/`.
8. **Everything shared must be torch-free.** `src/dataset.py` imports `torch` and `cv2` at module
   level, so anything importable from the monitoring image cannot live there. `utils/labels.py`
   and `src/sample_selection.py` use only stdlib + pandas (+ Pillow when reading shapes).
9. **The builder script is standalone, not a monitoring job.** It's a dev/analysis tool run
   independently, so it may freely use hydra and `src/`. Its output is committed under
   `data/<name>/`, and monitoring only ever consumes the resulting JSON — exactly how
   `compute_reference.py` consumes `val_samples` from a run's `dataset_manifest.json`.
10. **Benchmark registration becomes manifest-based, which makes it dependency-free.** With
    labels and shapes recorded in the manifest at build time, `register_benchmark.py` needs no
    filename parsing (`utils`), no MongoDB (`pymongo`), and no image reads (`pillow` + O-drive
    mount) — only the JSON file and Postgres. This also **fixes a current bug** (see Part D).
11. **Datasets accept injected samples.** Both dataset classes gain an optional `samples=`
    argument; when provided they skip building entirely and become pure access/tile/transform
    layers. This is what lets the standalone builder and training share one code path with no
    `Dataset` instance and no torch in the builder.
12. **Per-split balancing semantics are preserved exactly.** `max_samples_per_label` currently
    applies *within each split* (train balanced to N/label, val balanced to N/label
    independently). The refactor calls `build_samples` once per split, so this is unchanged.
13. **Shape reading is a mode, not a fixed behavior.** The tiled dataset reads the *first*
    sample's image and applies that size to all (fast, relies on "all images same size"); the
    builder must read *per sample* because `register_benchmark` writes `shape_x`/`shape_y` per
    image row and a benchmark may span plates with different sizes. `shape_mode="first"` vs
    `"per_sample"` rather than inheriting one behavior.
14. **`max_wells_per_label` cannot express "never in any training set".** It takes the first N
    sorted wells per label. The benchmark spec therefore uses an explicit `include_wells`
    whitelist as the source of truth. (A future option: subtract wells found in prior runs'
    `dataset_manifest.json`; out of scope here.)
15. **Selection must be reproducible.** `_balance_samples` calls `random.shuffle` with **no
    seed** — masked in training by the global `seed_everything(42)`, but a standalone utility
    would pick a different subset every run. `build_samples` takes an explicit `seed`, and more
    importantly the manifest is written once and frozen, so registration never re-selects.

## Part A — `utils/labels.py` (new, torch-free, shared across images)

One implementation of "wells -> treatment labels", replacing three.

1. `resolve_labels_for_wells(wells, source="mongodb", seed=42) -> dict[tuple[str, str], str]`:
   - `source="mongodb"`: builds the `Plate`/`Well` DataFrame, calls
     `tools.loading.getCategories(df, collection="tags")`, reads `Treatment`. Keeps
     `backfill_labels`' defensive behavior: missing `Treatment` column -> warn + return `{}`;
     `None`/`""`/`NaN` treatments filtered out (fixes decision 6).
   - `source="dummy"`: the existing seeded pseudo-random assignment over `sorted(wells)`, so
     determinism is unchanged.
2. `limit_wells_per_label(labels, n, verbose=False) -> dict` — moved verbatim from
   `MultiChannelDataModule._limit_wells_per_label` (sort wells within each label, take first N).

Callers updated:
- `monitoring/backfill_labels.py` imports it and deletes its local `resolve_labels`.
- `MongoDBLabelsProvider`/`DummyLabelsProvider` keep their current
  `get_labels(root_dir, exclude_wells)` signature as **thin back-compat wrappers**
  (`wells_from_index(scan_directory(root_dir), exclude=...)` + `resolve_labels_for_wells`), so
  nothing outside the datamodule breaks.
- `docker/jobs/monitoring/Dockerfile` must `COPY utils/ ./utils/` for `label-backfill` to import
  it.

## Part B — `src/sample_selection.py` (new, torch-free)

The shared selection core plus the short-form spec.

1. `scan_directory(root_dir) -> dict[tuple[str, str, int], dict[int, Path]]`
   One glob per supported extension, one `FILENAME_PATTERN` match per file, grouped by
   `(plate, well, field)` -> `{channel: path}`. The single place the filesystem is touched.
   Uses `utils/filename_parser.py`'s pattern so the third copy of that regex (currently duplicated
   in both dataset classes) goes away.
2. `wells_from_index(index, exclude_wells=None, include_wells=None) -> set[tuple[str, str]]`
   Pure in-memory filtering. `include_wells` is a whitelist (benchmark); `exclude_wells` is the
   existing corrupted-well blacklist. Applying both is an error worth rejecting loudly.
3. `build_samples(index, labels_dict, channels, max_samples_per_label=None, seed=42,
   shape_mode=None, verbose=False) -> list[dict]`
   Keeps only samples whose `(plate, well)` is in `labels_dict` and which have **all** requested
   channels; attaches `label`; balances per label with an explicitly seeded RNG; optionally
   attaches `shape` per decision 13. Returns dicts in the **same shape** the dataset classes
   currently produce (`plate`, `well`, `field`, `label`, `channel_files`, and `image_size`/`shape`).
4. `@dataclass DatasetSpec` — the short form, exactly the filters and nothing else:
   `root_dir`, `channels`, `use_mongodb`, `exclude_wells`, `include_wells`,
   `max_wells_per_label`, `max_samples_per_label`, `seed`, `verbose`.
   Plus `DatasetSpec.from_datamodule_cfg(cfg.datamodule)` so the standalone builder consumes the
   identical YAML shape training already uses (`configs/datamodule/tiled.yaml`), and so "rebuild
   the dataset this run trained on" is possible later.

Training does **not** have to construct a `DatasetSpec` — it keeps passing its own config fields
to the same functions. The dataclass is for the standalone entry point.

## Part C — rewire training onto the core (behavior-preserving)

1. `src/datamodule.py::setup()` becomes the pipeline it already is, with one scan:
   ```
   index  = scan_directory(self.root_dir)
   wells  = wells_from_index(index, self.exclude_wells)
   labels = resolve_labels_for_wells(wells, source="mongodb" if self.use_mongodb else "dummy")
   labels = limit_wells_per_label(labels, self.max_wells_per_label)
   # unchanged from here: LabelEncoder.fit, stratified well-level train/val split
   ```
   then `_setup_datasets` calls `build_samples(index, train_labels, ...)` /
   `build_samples(index, val_labels, ...)` and injects the results.
   `_limit_wells_per_label` is deleted (moved to `utils/labels.py`).
2. `src/dataset.py`: both classes take `samples=None`; `_build_sample_list` and
   `_balance_samples` are deleted from both (the duplicated pair at lines ~113-158 and ~301-392),
   replaced by a call to `build_samples` when `samples` isn't injected. `TiledMultiChannelDataset`
   keeps `_build_tile_index`, the LRU cache, and `__getitem__` — its actual job.
   `MultiChannelImageDataset` gains balancing/`verbose`/shape for free (the functionality it
   currently lacks).
3. Verification: this part must not change training behavior. Re-run the same 1-epoch smoke
   command used for the monitoring smoke test and diff `dataset_metadata.json` /
   `dataset_manifest.json` against a pre-refactor run — with a fixed seed the selected samples
   should be **identical**.

## Part D — metadata/manifest artifacts + the standalone builder

1. `src/dataset_versioning.py` — add sample-list-based variants next to the existing ones:
   - `create_metadata_from_samples(samples, spec, name) -> dict`: reuses
     `compute_dataset_version` / `compute_config_hash` (hashing the **spec** dict), plus
     `class_names` (`sorted(set(labels))` — same rule as `LabelEncoder`, so it stays comparable to
     a model's), `num_classes`, counts, `wells_used`, per-label distribution.
   - `create_manifest_from_samples(samples, output_path) -> str`: flat `{"samples": [...]}`.
   Per-sample entries keep the **existing** shape (`index`, `plate`, `well`, `field`, `label`,
   `root_path`, `channel_files: {channel: filename}`) plus `shape`, so
   `compute_reference.load_validation_data` and the notebooks keep working unchanged.
   The existing `create_dataset_metadata`/`create_dataset_manifest` become thin wrappers so
   `train.py` is untouched.
2. `scripts/build_dataset.py` (new) — standalone hydra CLI: reads a spec, runs
   scan -> wells -> labels -> `build_samples`, writes
   `data/<name>/{dataset_metadata.json,dataset_manifest.json}`. Prints the label distribution and
   selected wells so the artifact is reviewable before it's trusted.
3. `configs/dataset_spec/benchmark.yaml` (new) — the short form for the benchmark set:
   `root_dir`, `channels`, `use_mongodb: true`, `include_wells` (whitelist), `max_samples_per_label`,
   `seed`.
4. `monitoring/register_benchmark.py` — add `--manifest data/<name>/dataset_manifest.json` as the
   primary path: read JSON, insert `image_metadata` rows (`shape` comes from the manifest),
   `benchmark_dataset` rows (`t_label` comes from the manifest), and
   `benchmark_dataset_member` rows. Idempotency via `get_benchmark_samples()` is unchanged. The
   existing glob mode stays as a fallback.
   **This fixes a current bug:** `register_benchmark.py` imports
   `utils.filename_parser`, but `docker/jobs/monitoring/Dockerfile` copies only `database/` and
   `monitoring/` and the compose service mounts no `utils/` — so `make register-benchmark-run`
   currently dies with `ModuleNotFoundError` before doing anything. The manifest path removes the
   import entirely; `COPY utils/` (Part A) covers the fallback path.
5. `docker/jobs/monitoring/docker-compose.yaml` — on the manifest path `register-benchmark` no
   longer needs `${TOOLS_LIB_PATH}` or `${O_DRIVE_PATH}`; it needs the `data/` directory mounted
   (or the manifest baked in). Its shape collapses to the same image + DB as `drift-report`.

## Part E — follow-on (not in this change)

`compute_reference.load_validation_data` is already a generic "manifest -> predict()-shaped
metadata" loader. Generalizing it into `monitoring/predict_manifest.py --manifest X --mode
live|reference|benchmark` gives back-prediction of any built dataset for almost no new code. Left
out here to keep this change reviewable.

## Files touched

- `utils/labels.py` (new) — one `resolve_labels_for_wells` + `limit_wells_per_label`.
- `src/sample_selection.py` (new) — `scan_directory`, `wells_from_index`, `build_samples`,
  `DatasetSpec`.
- `src/dataset.py` — providers become wrappers; both classes take `samples=`; duplicated
  `_build_sample_list`/`_balance_samples`/`FILENAME_PATTERN` removed.
- `src/datamodule.py` — `setup()` uses the core, one scan; `_limit_wells_per_label` removed.
- `src/dataset_versioning.py` — `create_metadata_from_samples`, `create_manifest_from_samples`;
  existing functions become wrappers.
- `monitoring/backfill_labels.py` — local `resolve_labels` deleted, imports `utils/labels.py`.
- `monitoring/register_benchmark.py` — `--manifest` path (primary), glob path kept as fallback.
- `scripts/build_dataset.py` (new), `configs/dataset_spec/benchmark.yaml` (new).
- `docker/jobs/monitoring/Dockerfile` — `COPY utils/ ./utils/`.
- `docker/jobs/monitoring/docker-compose.yaml` — `register-benchmark` mounts `data/`, drops
  tools/O-drive on the manifest path.
- `data/<name>/` — built artifacts, committed (frozen benchmark definition).

## Tests

Extend rather than duplicate — `tests/monitoring/` is already planned in
`docs/plans/quality-tracking.md`.

1. `tests/test_sample_selection.py` — on a tmp_path of synthetic filenames: `scan_directory`
   grouping; incomplete-channel-set exclusion; `exclude_wells`/`include_wells`;
   `limit_wells_per_label`; `build_samples` balancing **determinism under a fixed seed**;
   `shape_mode` behavior.
2. `tests/test_labels.py` — `resolve_labels_for_wells` with a stubbed `getCategories`: missing
   `Treatment` column -> `{}` + warning; `None`/`""`/`NaN` filtered (the regression from decision 6).
3. `tests/monitoring/test_register_benchmark.py` — manifest path against real Postgres:
   inserts `image_metadata`/`benchmark_dataset`/`benchmark_dataset_member` correctly, is
   idempotent on rerun, and needs neither Mongo nor image files.
4. Regression guard for Part C: build samples from a fixed synthetic tree via both the injected
   path and the dataset-internal path and assert the sample lists are equal.
