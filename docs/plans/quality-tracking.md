# Supervised quality tracking: label backfill + benchmark + Evidently quality reports

Add three complementary pieces on top of the unsupervised drift detection in
`docs/plans/drift-detection.md`:

1. A MongoDB-backed ground-truth **backfill** job (`t_label` for production wells).
2. A model-independent **benchmark dataset** (curated samples never seen in any training set)
   plus a per-run job that scores it, giving every model a comparable quality baseline.
3. An Evidently-based **supervised quality report** (accuracy/F1/confusion matrix) comparing the
   current production window against that model's benchmark score.

Tests and CI for **all** of `monitoring/` (drift + backfill + benchmark + quality) are covered at
the end of this doc, done once everything is implemented.

## Core decisions

1. **Label backfill is global by well, not scoped to a run_id.** `t_label` is a property of the
   physical sample `(plate, well)`, not of which model predicted it — once MongoDB resolves a
   label, it backfills every matching `image_prediction`/`tile_prediction` row regardless of
   `run_id`.
2. **MongoDB connection info is hardcoded inside the external `tools.loading` module** (shared
   server, used by everyone on the team) — nothing to configure on our side. The backfill job
   just needs `tools` + its deps installed and network access to that server, same trust level
   as e.g. the `O_DRIVE_PATH` mount already assumes reachability to shared infra. `tools` itself
   isn't pip-installable from a registry — it's accessed the same way training code already
   reaches it: a read-only bind mount + editable install at container start, e.g.
   `docker/jobs/train/docker-compose.yaml`:
   ```yaml
   volumes:
     - ${TOOLS_LIB_PATH}:/workspace/tools:ro # access to tools library
   ```
   paired with `docker/jobs/train/Dockerfile`'s entrypoint (`pip install -e /workspace/tools`
   before `exec "$@"`). The monitoring image uses the same pattern — the `label-backfill`
   service mounts `${TOOLS_LIB_PATH}:/app/tools:ro` and the Dockerfile's entrypoint does a
   conditional `pip install -e /app/tools` that's a no-op when the volume isn't mounted
   (drift/quality/benchmark runs).
3. **`reference` and `benchmark` are different concepts and are stored differently.**
   - **reference** = a model's own validation split, only meaningful relative to a `run_id`, so
     it stays as the `is_reference` flag on `image_prediction`/`tile_prediction`. Its job is the
     **drift** baseline (input/prediction distribution). No dedicated table.
   - **benchmark** = a fixed, curated set of samples that were **never** in any model's training
     set, existing independently of any model (it's meaningful even with zero models trained).
     It gets its own tables (`benchmark_dataset`, `benchmark_dataset_member`). Its job is the
     **quality** baseline, scored per `run_id`.
4. **The quality baseline is the benchmark, not the validation reference.** For a fair
   cross-model accuracy/F1 comparison, every model is scored on the *same* held-out inputs. The
   current production window's supervised quality is compared against that model's benchmark
   score (same `run_id`).
5. **A prediction row is tagged benchmark via a nullable FK, not a boolean.** Add
   `benchmark_id INTEGER DEFAULT NULL REFERENCES benchmark_dataset(id)` to
   `image_prediction`/`tile_prediction`. `NULL` = ordinary prediction; set = a benchmark
   prediction, and the FK additionally records *which* benchmark sample it scored. Benchmark
   predictions for a model are `WHERE run_id = %s AND benchmark_id IS NOT NULL`. This is richer
   than mirroring the `is_reference` boolean.
6. **`predict()` stays as-is for now** (no `log_full_prediction` refactor). `benchmark_id` rides
   in `image_metadata` exactly like `label`/`is_reference` already do
   (`image_metadata.get("benchmark_id")`), so it's just one more value appended to the existing
   `image_prediction`/`tile_prediction` insert tuples — no new branching, minimal diff. (A later
   cleanup can move persistence into `DBLogger.log_full_prediction`; explicitly out of scope
   here.)
7. **Quality metrics reuse Evidently's classic `ClassificationPreset`** (already pinned at
   `0.6.7` for drift), not hand-rolled scikit-learn — same `Report`/`as_dict()` pattern as
   `run_drift_report.py`, one less new dependency to manage.
8. **Confusion matrix stays in the HTML/JSON report file** on disk (`monitoring/reports/...`),
   not a queryable table — `quality_report` only stores top-line accuracy/F1/sample counts,
   mirroring how `drift_report` doesn't have a raw-value table either.
9. **Quality report is image-level only for the MVP.** Tile-level accuracy against a well-level
   label is less meaningful without re-aggregating by majority vote; can be added later.
10. **If a current window has zero samples with `t_label` populated yet, skip the report/insert
    entirely** (same pattern `run_drift_report.py` already uses when a window has no live rows).
    The backfill job and the quality-report job are deliberately decoupled — the report just
    works with whatever fraction of the window happens to be labeled already, rather than
    requiring backfill to have run first.

## Part A — `monitoring/backfill_labels.py` (new)

Finds production predictions whose true label has since become available in MongoDB and writes
it back.

1. `DBLogger.fetch_unlabeled_wells() -> list[tuple[str, str]]`:
   `SELECT DISTINCT plate, well FROM image_prediction WHERE is_reference = FALSE AND benchmark_id IS NULL AND t_label IS NULL`
   (excludes reference and benchmark rows — those already carry known labels).
2. `DBLogger.update_t_label(plate: str, well: str, t_label: str) -> tuple[int, int]`: single
   method doing both updates so they can't drift out of sync:
   - `UPDATE image_prediction SET t_label = %s WHERE plate = %s AND well = %s AND is_reference = FALSE AND benchmark_id IS NULL AND t_label IS NULL RETURNING id`
   - `UPDATE tile_prediction SET t_label = %s WHERE image_pred_id = ANY(%s::int[])` using the ids
     just returned.
   - Returns `(n_image_rows_updated, n_tile_rows_updated)`.
3. `backfill_labels.py main()`:
   - `MonitoringSettings()` for `db_uri` (reused as-is — this job doesn't call the API, so
     `base_url` is irrelevant here).
   - `wells = db_logger.fetch_unlabeled_wells()`; if empty, print and exit.
   - Build a `Plate`/`Well` DataFrame from `wells`, call
     `tools.loading.getCategories(df, collection="tags")` (same call shape as
     `MongoDBLabelsProvider.get_labels` in `src/dataset.py`), get back a `Treatment` column.
   - For every `(plate, well)` that resolved to a non-null `Treatment`, call
     `db_logger.update_t_label(plate, well, treatment)`.
   - Print a summary: N wells checked, M resolved, X image rows / Y tile rows updated.
4. Runs in the **same** `docker/jobs/monitoring/` image as drift/quality/benchmark — the `tools`
   package is lightweight (only needs `pymongo` + `joblib` beyond what `monitoring_req.txt`
   already pins). No separate Dockerfile/job directory:
   - `requirements/monitoring_req.txt`: add `pymongo>=4.6` and `joblib>=1.3` (the only
     transitive deps `tools.loading.getCategories` actually imports at runtime).
   - `docker/jobs/monitoring/Dockerfile`: entrypoint script that does
     `pip install -e /app/tools 2>/dev/null` then `exec "$@"` — mirrors
     `docker/jobs/train/Dockerfile`. No-op when the tools volume isn't mounted, so the same
     image works for every service.
   - `docker/jobs/monitoring/docker-compose.yaml`: `label-backfill` service (same build/image,
     own `container_name`), with extra volume `${TOOLS_LIB_PATH}:/app/tools:ro`. Uses a `.env`
     alongside `.env.monitoring` for `TOOLS_LIB_PATH` substitution.
   - `docker/makefile`: `label-backfill-run` target, same `$(CMD)`-passthrough style as
     `drift-report-run`, defaulting to `python -m monitoring.backfill_labels`.

## Part B — Benchmark dataset: registration + per-run scoring

The benchmark is a fixed set of curated `(plate, well, field)` samples with known `t_label` that
were never in any model's training set. Registered once, scored once per model.

### B1 — `monitoring/register_benchmark.py` (new, one-time / additive)

Registers benchmark samples into the DB. Model-independent — no MLflow, no inference, so it
runs in the **lightweight monitoring image** alongside `label-backfill` (not the api image) —
just needs `pillow`/`pillow-jxl-plugin` added to `monitoring_req.txt` for reading image shapes,
plus the same `tools` mount as `label-backfill` (for the MongoDB label lookup) and a read-only
`${O_DRIVE_PATH}` mount to reach the benchmark image files.

- **Input** (exact source TBD, decided at implementation time): most likely a list of absolute
  file paths (e.g. the output of a `glob`). From each path, parse `plate`, `well`, `field`,
  `channel`, `root_path`, `file_name` using the same `utils/filename_parser` helpers the API
  uses, so parsing stays consistent with production.
- For each sample:
  - Insert its single-channel image files into `image_metadata` (reusing existing rows via the
    `UNIQUE(file_name)` constraint — benchmark files may already be present).
  - Insert one `benchmark_dataset` row `(plate, well, field, t_label)`.
  - Insert `benchmark_dataset_member` rows linking the `benchmark_id` to each `image_id` with its
    `channel_index` (mirrors `tile_stack_member`'s channel-position semantics).
- `t_label` source at registration: TBD (likely the same `tools.loading.getCategories`
  MongoDB lookup as backfill, or provided alongside the path list). Not blocking the schema.
- Idempotent: re-running with the same paths must not duplicate `benchmark_dataset` rows (guard
  on `UNIQUE(plate, well, field)`).
- New `DBLogger` methods: `log_benchmark_sample(...)` / `log_benchmark_members(...)` (or one
  combined transactional method), plus `get_benchmark_samples()` for resumability.

### B2 — `monitoring/compute_benchmark.py` (new, per run_id)

Scores the registered benchmark set with a specific model. Same shape as `compute_reference.py`
(loads the model via `TilePredictor`, runs each sample through `predict()`), differing only in
where samples come from and how rows are tagged:

- Load benchmark samples from `benchmark_dataset` + `benchmark_dataset_member` +
  `image_metadata` (not from a MLflow `dataset_manifest.json` — the benchmark is model-independent
  and lives in the DB). Load each sample's channel images from their `root_path`/`file_name`.
- Build `image_metadata` dicts carrying `label` = the benchmark `t_label` **and**
  `benchmark_id` = the `benchmark_dataset.id`. `is_reference` stays `False`.
- Run `predictor.predict(sample_image, image_metadata)` — unchanged pipeline; the only new thing
  is `benchmark_id` flowing through into the prediction insert tuples (see Part C schema/predictor
  changes).
- Resumable: skip benchmark samples that already have a prediction row for this `run_id` (via a
  new `DBLogger.get_benchmark_predictions(run_id)`, analogous to `get_reference_samples`).
- Reuses the **api image** (needs torch/mlflow/timm for real inference), same as
  `compute_reference.py` — a `compute-benchmark` job/target mirroring `compute-reference`, not the
  lightweight monitoring image.

### Predictor change (small)

- `api/predictor.py::predict()`: append `image_metadata.get("benchmark_id")` to the
  `image_prediction` and `tile_prediction` insert tuples (right alongside the existing
  `label`/`is_reference` values). No refactor, no behavior change when `benchmark_id` is absent
  (defaults to `NULL`). `DBLogger.log_image_prediction`/`log_tile_prediction` insert statements
  gain the `benchmark_id` column.

## Part C — `monitoring/run_quality_report.py` (new)

1. Refactor shared logic out of `run_drift_report.py` into `monitoring/config.py` (small, focused
   module already holding `MonitoringSettings`) so both report scripts share it instead of
   duplicating:
   - `resolve_run_id(run_id: str | None, api_url: str) -> str | None` (moved as-is from
     `run_drift_report.py`).
   - `resolve_window(window_start, window_end, window_days) -> tuple[datetime, datetime]`,
     factored out of `run_drift_report.py`'s inline window-parsing logic.
   - Update `run_drift_report.py` to import both from `monitoring.config` instead of defining
     them locally.
2. New `DBLogger` methods:
   - `fetch_benchmark_quality(run_id) -> list[dict]`:
     `SELECT p_label, t_label FROM image_prediction WHERE run_id = %s AND benchmark_id IS NOT NULL AND t_label IS NOT NULL`
     — the model's benchmark score, the quality baseline.
   - `fetch_current_quality(run_id, window_start, window_end) -> list[dict]`: production window
     rows (`is_reference = FALSE AND benchmark_id IS NULL`), additionally filtered to
     `t_label IS NOT NULL`, keeping the existing reference-well `NOT EXISTS` exclusion.
3. `run_quality_report.py main()`:
   - Same CLI shape as `run_drift_report.py` (`--run-id`, `--api-url`, `--window-days`/
     `--window-start`/`--window-end`, `--reports-dir`), reusing `resolve_run_id`/`resolve_window`.
   - Pull benchmark (baseline) + current-window rows via the two fetch methods.
   - If benchmark is empty: error out (run `compute-benchmark` for this run_id first).
   - If current is empty (no labeled current-window samples yet): print and exit cleanly, no DB
     insert — the expected common case before backfill has caught up.
   - `ColumnMapping(target="t_label", prediction="p_label")`,
     `Report([ClassificationPreset()]).run(reference_data=benchmark_df, current_data=cur_df, column_mapping=mapping)`
     (Evidently's `reference_data` = our benchmark baseline).
   - `report.save_html(...)` under `monitoring/reports/quality_<run>_<timestamp>/report.html`.
   - Extract from `as_dict()`'s `ClassificationQualityMetric` result: `result["reference"]` /
     `result["current"]` each with `accuracy`/`f1` (anything else stays only in the HTML file, per
     the confusion-matrix decision above).
   - `DBLogger.log_quality_report((run_id, window_start, window_end, n_benchmark_samples,
     n_current_samples, benchmark_accuracy, benchmark_f1, current_accuracy, current_f1,
     report_path))`.
4. Runs in the **same** `docker/jobs/monitoring/` image (same evidently/pandas/psycopg footprint
   as drift, no extra deps). A `quality-report` service in
   `docker/jobs/monitoring/docker-compose.yaml` (same build/image, own `container_name`), and a
   `quality-report-run` target in `docker/makefile` defaulting to
   `python -m monitoring.run_quality_report`.

## Schema

### `database/init/03_benchmark.sql` (new)

```sql
-- Model-independent held-out set: curated samples never in any training set,
-- meaningful even with zero models trained. Scored per model by
-- compute_benchmark.py to give every run a comparable quality baseline.
CREATE TABLE IF NOT EXISTS benchmark_dataset (
    id SERIAL PRIMARY KEY,
    plate VARCHAR(255) NOT NULL,
    well VARCHAR(255) NOT NULL,
    field INTEGER NOT NULL,
    t_label VARCHAR(255) NOT NULL,            -- known ground truth
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (plate, well, field)
);

-- Which single-channel image files make up a benchmark sample (mirrors
-- tile_stack_member's channel-position semantics).
CREATE TABLE IF NOT EXISTS benchmark_dataset_member (
    id SERIAL PRIMARY KEY,
    benchmark_id INTEGER REFERENCES benchmark_dataset(id),
    image_id INTEGER REFERENCES image_metadata(id),
    channel_index INTEGER NOT NULL,
    UNIQUE (benchmark_id, image_id)
);

-- Tag a prediction as a benchmark score via a nullable FK (NULL = ordinary
-- prediction). Richer than a boolean: records which benchmark sample was scored.
ALTER TABLE image_prediction ADD COLUMN IF NOT EXISTS benchmark_id INTEGER DEFAULT NULL REFERENCES benchmark_dataset(id);
ALTER TABLE tile_prediction  ADD COLUMN IF NOT EXISTS benchmark_id INTEGER DEFAULT NULL REFERENCES benchmark_dataset(id);

CREATE INDEX IF NOT EXISTS idx_image_prediction_run_benchmark ON image_prediction(run_id, benchmark_id);
CREATE INDEX IF NOT EXISTS idx_tile_prediction_run_benchmark  ON tile_prediction(run_id, benchmark_id);
```

### `database/init/04_quality.sql` (new)

```sql
CREATE TABLE IF NOT EXISTS quality_report (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    n_benchmark_samples INTEGER NOT NULL,
    n_current_samples INTEGER NOT NULL,
    benchmark_accuracy FLOAT NOT NULL,
    benchmark_f1 FLOAT NOT NULL,
    current_accuracy FLOAT NOT NULL,
    current_f1 FLOAT NOT NULL,
    report_path VARCHAR(512) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quality_report_run_id ON quality_report(run_id, computed_at);
```

## New/changed files summary

- `database/init/03_benchmark.sql` (new) — benchmark tables + `benchmark_id` columns/indexes.
- `database/init/04_quality.sql` (new) — `quality_report`.
- `api/predictor.py` — append `benchmark_id` to `image_prediction`/`tile_prediction` insert tuples.
- `database/dblogger.py` — add `log_benchmark_sample`/`log_benchmark_members`,
  `get_benchmark_samples`, `get_benchmark_predictions`, `fetch_unlabeled_wells`, `update_t_label`,
  `fetch_benchmark_quality`, `fetch_current_quality`, `log_quality_report`; update
  `log_image_prediction`/`log_tile_prediction` inserts for the new `benchmark_id` column.
- `monitoring/config.py` — add `resolve_run_id`, `resolve_window`.
- `monitoring/run_drift_report.py` — import the two helpers instead of defining them.
- `monitoring/backfill_labels.py` (new).
- `monitoring/register_benchmark.py` (new).
- `monitoring/compute_benchmark.py` (new).
- `monitoring/run_quality_report.py` (new).
- `requirements/monitoring_req.txt` — add `pymongo>=4.6`, `joblib>=1.3`, `pillow>=10.0.0`,
  `pillow-jxl-plugin>=0.10.3`.
- `docker/jobs/monitoring/Dockerfile` — entrypoint script for optional tools install; image
  renamed `image_classifier_monitoring` (now backs 4 services, not just drift).
- `docker/jobs/monitoring/docker-compose.yaml` — add `quality-report`, `label-backfill`, and
  `register-benchmark` services (the latter two mount `${TOOLS_LIB_PATH}:/app/tools:ro`;
  `register-benchmark` additionally mounts `${O_DRIVE_PATH}:/mnt/O:ro`).
- `docker/jobs/monitoring/.env` + `.env.example` (new) — `TOOLS_LIB_PATH`, `O_DRIVE_PATH`.
- `docker/jobs/compute/docker-compose.yaml` — merged `compute-reference` (was its own dir) and
  a new `compute-benchmark` service into one file, sharing `.env`/`.env.api`; both reuse the api
  image (real inference needed), same pattern as `docker/jobs/monitoring`'s multi-service file.
- `docker/makefile` — add `quality-report-run`, `label-backfill-run`, `register-benchmark-run`,
  `compute-benchmark` targets.

## Tests & CI (covers drift + backfill + benchmark + quality together, done last)

- `tests/monitoring/` covering: `compute_reference.py`, `run_drift_report.py`,
  `register_benchmark.py`, `compute_benchmark.py`, `backfill_labels.py`, `run_quality_report.py`,
  and all the new `DBLogger` methods, against a real Postgres with `01_prediction.sql` +
  `02_reference.sql` + `03_benchmark.sql` + `04_quality.sql` applied — matching
  `tests/db/test_dblogger.py`'s setup/teardown conventions.
- One `.github/workflows/ci_monitoring.yml` mirroring `ci_db.yml`'s Postgres-service pattern,
  scoped to `database/**`, `monitoring/**`, `tests/monitoring/**`, `requirements/monitoring_req.txt`,
  `docker/jobs/monitoring/**`.

## Verification

- `py_compile` / import-check every new/changed Python file.
- `docker compose config` validation for new/changed compose files.
- Manual smoke test (needs a real Postgres + MongoDB + running API) deferred to the user, same as
  the drift-report work.
- Full pytest/CI pass deferred to the Tests & CI section above, done once for all of `monitoring/`.

## Open risks

- `tools.loading.getCategories` groups by well already (matches training-time label granularity),
  but its actual MongoDB query/error behavior (missing well, network failure, etc.) can't be
  inspected since `tools` isn't in this repo — `backfill_labels.py` (and benchmark label lookup,
  if it uses the same path) must handle "well not found in Mongo yet" (skip, not an error) vs.
  real connection failures (raise) defensively.
- `register_benchmark.py`'s exact input format (glob of paths vs. manifest vs. Mongo query) and
  its `t_label` source are decided at implementation time; the schema above is designed to not
  depend on that choice.
