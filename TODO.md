# Project TODO

## Serving: model/API dependency version mismatch (unresolved)

**Problem:**
Models are logged via `mlflow.pytorch.log_model(..., code_paths=["src"])`
(<ref_snippet file="/d:/personal_project/image_classifier/src/callbacks.py" lines="114-122" />),
which bundles the model class source code with the artifact, but does **not**
pin the runtime dependency versions used during training (e.g. `timm`,
`torch`). The serving API (`api/predictor.py`) loads models in-process via
`mlflow.pytorch.load_model()` inside a single container built from
`requirements/api_req.txt`
(<ref_file file="/d:/personal_project/image_classifier/docker/services/api/Dockerfile" />).

If different developers train models with different dependency versions
(e.g. `timm==1.0.26` vs `timm==1.0.28`), one fixed API container cannot
correctly serve all of them. We hit this already and worked around it by
pinning `timm==1.0.26` in `api_req.txt`, but that's not a real fix — it just
makes the container match whichever model was trained most recently.

**Options discussed (see chat history for full breakdown, brainstormed on 2026-08-11):**
1. Log `pip_requirements` / `extra_pip_requirements` with the model at log-model
   time, and switch serving to `mlflow.pyfunc.load_model()` (supports env
   isolation) — requires refactoring how `predictor.py` calls the model.
2. Build a dedicated Docker image per model/run, using the model's logged
   `conda.yaml` to generate exact requirements — no `predictor.py` changes,
   but adds a build/deploy pipeline step.
3. Dynamically `pip install` the model's exact requirements at API container
   startup (before loading the model) — single container, adapts per model,
   but slow startup and can't serve two conflicting-version models at once.
4. Pin exact versions across `training_req.txt` and `api_req.txt` and enforce
   everyone uses the same environment — zero code changes, simplest, but
   doesn't scale with multiple developers/experiments.

**Recommendation (not yet implemented):** Do (1) regardless — always log
`pip_requirements` with the model, it's cheap and gives us metadata/options
later. Combine with (3) short-term for flexibility, revisit (2) if startup
latency becomes a problem or models need to run concurrently with
conflicting dependency versions.

**Status:** Not started. Revisit before onboarding more developers to
training or before relying on this for production serving.

## Drift detection: parallelisation approach (decided, shelved)

**Context:**
`TilePredictor.predict()` already batches tiles per image (~100 tiles
per image → one `model(batch_of_100)` call). The GPU is already
well-utilized per image, so cross-image batching gives negligible gain.

**Decision: shelve `batch_predict`**
A `batch_predict()` method was designed (see
`docs/plans/batch-predict.md`) but is shelved because `predict()`
already batches tiles at the GPU level. Adding cross-image batching
would add complexity for no meaningful speedup.

**Why threads don't work here:**
- Psycopg 3 connections are not thread-safe. A shared `DBLogger`
  cannot be used across threads — parallel `predict()` calls sharing
  `self.db_logger` would corrupt connection state and return wrong IDs.
- Multiple threads calling `self.model(...)` on one GPU serialize at
  the hardware level (CUDA queues kernels sequentially) — no speedup.
- Python GIL limits CPU-side parallelism for preprocessing/tiling/DB
  logging to ~1.3-1.5x, not true N×.

**Future approach: multi-process with `--shard` and `--device`**
True parallelism comes from **one process per hardware resource**,
each with its own `DBLogger` connection and `TilePredictor` instance,
processing disjoint shards of the samples. No locks, no shared state.

```
# Multi-GPU (true N× speedup):
python compute_reference.py --shard 0/2 --device cuda:0
python compute_reference.py --shard 1/2 --device cuda:1

# CPU-only multi-process (true N× on CPU):
python compute_reference.py --shard 0/4 --device cpu
python compute_reference.py --shard 1/4 --device cpu
python compute_reference.py --shard 2/4 --device cpu
python compute_reference.py --shard 3/4 --device cpu

# Mixed GPU + CPU (GPU gets most samples, CPU offloads a few):
python compute_reference.py --shard 0/3 --device cuda:0   # GPU, ~70% of samples
python compute_reference.py --shard 1/3 --device cpu      # CPU, ~15% of samples
python compute_reference.py --shard 2/3 --device cpu      # CPU, ~15% of samples
```

**Why multiple processes/pods are safe but threads are not:**
- Each process/pod has its **own Psycopg connection**. PostgreSQL uses
  MVCC and handles hundreds of concurrent connections natively.
- A single Psycopg connection shared across threads corrupts because
  the connection has internal protocol state (current query, transaction,
  result buffer) that can't be interleaved.
- The rule: **one worker = one process = one connection.**

**Mixed CPU+GPU caveat:**
CPU inference is 10-50x slower than GPU for typical CNN models. The
CPU processes become the bottleneck unless:
- The model is small (CPU only 5-10x slower).
- There are many CPU cores (8+) and thousands of samples.
- The GPU process gets the majority of samples.
For hundreds of reference samples, single-GPU `predict()` in a loop
is sufficient. Mixed CPU+GPU is only worth it for very large jobs.

**What NOT to do:**
- Do not `ThreadPoolExecutor` parallel `predict()` with a shared
  `DBLogger` — Psycopg 3 will break.
- Do not run multiple processes on the **same GPU** — CUDA serializes
  kernels across processes, so you get the same throughput with extra
  memory overhead (two model copies on GPU) and context-switch cost.
  The only exception is NVIDIA MPS, which is complex and not worth it
  here.
- Do not `multiprocessing` with a shared `DBLogger` — connections are
  not picklable across processes.

**Status:** Decision made. `batch_predict` shelved. Multi-process
`--shard`/`--device` approach documented for future implementation
when `compute_reference.py` needs to scale beyond a single GPU.
Kubernetes horizontal scaling (step 7) uses the same pattern at
larger scale with a job queue.

## Monitoring smoke test (manual, real data)

End-to-end verification of the full DB → API → monitoring pipeline
against real data, real Postgres, real MLflow, and real MongoDB.
Run in order; each step depends on the previous one.

### Prerequisites
- [ ] Start Postgres + apply all 4 schemas (`01_prediction`,
      `02_reference`, `03_benchmark`, `04_quality`) to a clean DB.
- [ ] Start MLflow.
- [ ] Confirm external networks `ml_platform` and `pg_network` exist.
- [ ] Confirm Docker containers can reach MongoDB at
      `192.168.2.127:27017` (hardcoded in `tools/loading.py`) — may
      need `extra_hosts` or a bridge route if the Docker network is
      isolated. **If this is unreachable, label-backfill silently
      no-ops and quality-report skips.**
- [ ] Confirm the O-drive mount (`/mnt/O`) contains the benchmark and
      production images at the paths you'll feed `register-benchmark`.

### 1. Train & save a model
- [ ] Train a model; ensure `dataset_metadata.json["class_names"]`
      matches the sorted `LabelEncoder` classes.
- [ ] Register in MLflow under a stable name
      (e.g. `TransferLearningClassifier/<version>`).
- [ ] **Verify:** model is loadable by `run_id`, metadata `class_names`
      are correct, and the API `/model` endpoint resolves the run.

### 2. Register the benchmark dataset
- [ ] Run `register-benchmark-run` with real benchmark image paths and
      known `t_label` values.
- [ ] **Verify:** rows exist in `image_metadata`, `benchmark_dataset`,
      and `benchmark_dataset_member`.
- [ ] **Verify:** `t_label` values from the manifest match the model's
      `class_names` vocabulary — otherwise accuracy/F1 will be garbage.
- [ ] Re-run `register-benchmark` once to confirm idempotency
      (no duplicate rows, no errors).

### 3. Compute benchmark + reference predictions
- [ ] **3a.** Run `compute-benchmark` for the run.
      - Produces `image_prediction` + `tile_prediction` rows with
        `benchmark_id IS NOT NULL`, `is_reference = FALSE`.
      - **Verify:** counts match `benchmark_dataset_member`.
      - **Verify:** resume doesn't duplicate rows.
- [ ] **3b.** Run `compute-reference` for the run.
      - Produces `image_prediction` + `tile_prediction` rows with
        `is_reference = TRUE`, `benchmark_id IS NULL`.
      - **Verify:** input pixel / channel stats are logged
        (drift-report needs them, not just predictions).

### 4. Generate live production predictions
- [ ] Run the API and send real images through it (one-off, or a
      scheduled script every few hours).
      - Rows must be `is_reference = FALSE`, `benchmark_id IS NULL`,
        `t_label = NULL`.
      - **Verify:** `image_prediction`, `tile_prediction`, and
        `tile_channel_stats` rows are all being written.
- [ ] Spread predictions across a wide enough time window that the
      drift/quality window resolution actually includes them.

### 5. Backfill labels from MongoDB
- [ ] Run `label-backfill-run`.
      - Resolves `(plate, well)` via `tools.loading.getCategories`.
      - Updates `t_label` on `image_prediction` + `tile_prediction`
        for matching wells, globally by well (not per `run_id`).
- [ ] **Verify:** returned update counts are non-zero.
- [ ] **Verify:** returned labels match `class_names`.
- [ ] **Verify:** wells missing from Mongo are skipped without
      aborting the whole run.

### 6. Reports
- [ ] **6a. Drift report:** run `drift-report-run`.
      - Needs reference (3b) + live (4) + channel stats.
      - **Verify:** HTML/JSON report exists under
        `monitoring/reports/` and a `drift_report` row was inserted.
- [ ] **6b. Quality report:** run `quality-report-run`.
      - Needs benchmark predictions (3a) + labeled live (5).
      - **If zero labeled current samples, it skips silently** —
        confirm backfill actually labeled some before expecting a
        report.
      - **Verify:** `quality_report` row + report artifact exist.
      - **Inspect:** benchmark vs current accuracy/F1.

### 7. Final cross-check
- [ ] Query each table (`image_prediction`, `tile_prediction`,
      `benchmark_dataset*`, `quality_report`, `drift_report`) and
      confirm relations/flags are consistent across the whole run.

### Optional
- [ ] Repeat `compute-benchmark` + `quality-report` for a second
      `run_id` to confirm benchmark quality is tracked per-model,
      not globally.
