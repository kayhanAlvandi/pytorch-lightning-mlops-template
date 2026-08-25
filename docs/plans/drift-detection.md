---
agent: devin-local
session: flowery-william
created: 2026-08-13T17:28:02Z
---
# Drift detection (step 3): reference computation + Evidently drift reports

Add a background reference-computation pipeline (per served model, run once against its
validation set) and a drift-report job that compares live `image_prediction`/`tile_prediction`
traffic against that reference using Evidently, grouped by `p_label`, with everything (report
metadata + queryable metrics) stored in Postgres.

## Core decisions

1. **Reference is computed as a standalone job, not wired into the API.** It loads the model
   itself (reusing `TilePredictor`'s loading code) and writes directly to Postgres. Invoked
   manually/via CI-CD after training, before promoting a model to serve.
2. **Reference rows live in the same tables as production, distinguished by `is_reference`.**
   A validation-set prediction is structurally identical to a production one (same
   `image_metadata` → `tile_stack` → `tile_stack_member` → `image_prediction` →
   `tile_prediction` pipeline), so no mirrored `reference_*` tables — just a boolean flag on
   the existing `image_prediction`/`tile_prediction` tables. This also means
   `monitoring/compute_reference.py` can reuse `DBLogger`'s existing methods (with the flag
   added) instead of a parallel set of methods, and no separate `model_reference` table is
   needed for idempotency — that becomes `SELECT EXISTS(SELECT 1 FROM image_prediction WHERE
   run_id=%s AND is_reference)`.
3. **Stored raw (per-well, per-tile), not pre-aggregated**, so real distributional drift tests
   (KS, PSI, chi-square) and correct percentile math are possible. `reference_image_summary`
   (a view, grouped by `p_label`) provides the "meaningful summary" on top, derived not
   duplicated.
4. **Comparison grouping key is `p_label`, not `t_label`.** Production traffic has no ground
   truth, so reference and current must both be grouped by predicted label for a fair
   comparison. `t_label` is stored on reference rows too (free, since validation data is fully
   labeled) — used for a bonus `accuracy` column in `reference_image_summary`, but not part of
   the drift comparison itself.
5. **Input drift**: per-channel pixel stats (mean, std, p1, p5, p95, p99) stored in one
   dedicated table, keyed by `tile_stack_member.id` — not by `tile_prediction`/reference
   prediction row. Pixel stats are a deterministic function of the raw pixel data, not of
   which run/model predicted the tile, so they're computed once per physical tile-channel and
   shared by every prediction (production or reference, any run_id) that reuses the same
   `tile_stack`. This is a genuine 1:1 extension table (own table, not columns bolted onto
   `tile_stack_member` itself) so the relationship table keeps its single responsibility and
   this feature table can evolve independently later.
6. **Drift report output goes to Postgres + file storage, not MLflow.** MLflow's per-run
   artifact model isn't a good fit for a recurring (daily/weekly) report — it would grow one
   run's artifact folder forever with no retention story, and it's a separate system from the
   rest of monitoring's data (all of which lives in Postgres). Instead: the full Evidently
   HTML/JSON report is written to a file (local disk under a mounted `monitoring/reports/`
   directory for now; an S3/GCS URL later per step 6 is just a path-format change), and
   `drift_report`/`drift_report_column` tables store the summary + per-column results,
   directly SQL-queryable/joinable against `image_prediction` — this is also what step 5's
   retraining trigger will query against later.
7. **Constraint honored**: `database/init/01_prediction.sql` is not modified directly. All
   additions (the `is_reference` columns, indexes, views, and new tables) live in
   `database/init/02_refrence.sql` via `ALTER TABLE`/`CREATE TABLE`/`CREATE VIEW`.

## Finalized DB migration (`database/init/02_refrence.sql`) — implemented

```sql
-- Reference vs. live: same tables, one flag
ALTER TABLE image_prediction ADD COLUMN IF NOT EXISTS is_reference BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE tile_prediction ADD COLUMN IF NOT EXISTS is_reference BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_image_prediction_run_ref ON image_prediction(run_id, is_reference);
CREATE INDEX IF NOT EXISTS idx_tile_prediction_run_ref ON tile_prediction(run_id, is_reference);
CREATE INDEX IF NOT EXISTS idx_image_prediction_run_ref_created ON image_prediction(run_id, is_reference, created_at);
CREATE INDEX IF NOT EXISTS idx_tile_prediction_run_ref_created ON tile_prediction(run_id, is_reference, created_at);

-- "production only" views -- everything computing live metrics should query
-- these, not the raw tables, so a missing is_reference filter can't
-- silently mix validation data into production numbers
CREATE OR REPLACE VIEW live_image_prediction AS
    SELECT * FROM image_prediction WHERE is_reference = FALSE;
CREATE OR REPLACE VIEW live_tile_prediction AS
    SELECT * FROM tile_prediction WHERE is_reference = FALSE;

-- symmetric reference-only views
CREATE OR REPLACE VIEW reference_image_prediction AS
    SELECT * FROM image_prediction WHERE is_reference = TRUE;
CREATE OR REPLACE VIEW reference_tile_prediction AS
    SELECT * FROM tile_prediction WHERE is_reference = TRUE;

-- reference summary grouped by p_label (drift comparison key), with a free
-- accuracy bonus since t_label is known for reference rows
CREATE OR REPLACE VIEW reference_image_summary AS
    SELECT run_id, p_label,
           COUNT(*) AS n_wells,
           AVG(vote_fraction) AS avg_vote_fraction,
           AVG(avg_confidence) AS avg_confidence,
           AVG((p_label = t_label)::int)::float AS accuracy
    FROM image_prediction
    WHERE is_reference = TRUE
    GROUP BY run_id, p_label;

-- input drift: one row per tile_stack_member (deterministic given pixels,
-- shared across every run/prediction of the same physical tile-channel)
CREATE TABLE IF NOT EXISTS tile_channel_stats (
    id SERIAL PRIMARY KEY,
    tile_stack_member_id INTEGER REFERENCES tile_stack_member(id) UNIQUE NOT NULL,
    mean FLOAT NOT NULL,
    std FLOAT NOT NULL,
    p1 FLOAT NOT NULL,
    p5 FLOAT NOT NULL,
    p95 FLOAT NOT NULL,
    p99 FLOAT NOT NULL
);

-- drift reports: one row per drift-check execution, plus per-column detail
CREATE TABLE IF NOT EXISTS drift_report (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dataset_drift BOOLEAN NOT NULL,
    n_columns_drifted INTEGER NOT NULL,
    n_columns_total INTEGER NOT NULL,
    report_path VARCHAR(512) NOT NULL
);

CREATE TABLE IF NOT EXISTS drift_report_column (
    id SERIAL PRIMARY KEY,
    drift_report_id INTEGER REFERENCES drift_report(id) NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    column_group VARCHAR(50) NOT NULL,         -- 'image_level' | 'tile_level' | 'channel_stats'
    drift_score FLOAT NOT NULL,
    drifted BOOLEAN NOT NULL,
    stat_test VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_drift_report_run_id ON drift_report(run_id, computed_at);
CREATE INDEX IF NOT EXISTS idx_drift_report_column_report_id ON drift_report_column(drift_report_id);
```

## Manifest prerequisite — implemented

`src/dataset_versioning.py::create_dataset_manifest` now adds:
- A top-level `root_dir` field (single value — one shared root per datamodule run).
- Per-sample `channel_files`: `{channel_number: filename}` — **filenames only**, not full
  paths, combined with `root_dir` at read time (`Path(root_dir) / filename`), so the manifest
  stays portable across machines/containers.

Fixed a bug in the user's initial patch: `sample['channel_files']` values are `Path` objects
(built via `root_dir.glob(...)` in `TiledMultiChannelDataset._build_sample_list`), and
`json.dump(manifest, f, indent=2)` had no `default=str` — this would have raised
`TypeError: Object of type WindowsPath is not JSON serializable` the first time training ran
to completion. Fixed by converting to `path.name` (filename only) at manifest-build time,
which also matches the "filename only, portable" design instead of storing full
training-machine-specific absolute paths.

## Remaining implementation steps (not yet done)

1. **`monitoring/` package** (new top-level dir, not `scripts/`): `compute_reference.py`,
   `run_drift_report.py`, `__init__.py`. Follows this repo's per-domain convention
   (`api/`, `database/`, `src/` each get their own dir + `tests/<domain>/` +
   `requirements/<domain>_req.txt` + `ci_<domain>.yml`); also gives step 4 (accuracy/F1) and
   step 7 (orchestration hooks) a natural home later.
2. **`DBLogger` additions** (`database/dblogger.py`) — partially done already:
   - DONE: `close()` (fixes the `AttributeError` from `api/main.py`'s shutdown path calling
     `db_logger.close()` on a class that had no such method).
   - DONE: `log_tile_channel_stats(...)` with `ON CONFLICT (tile_stack_member_id) DO UPDATE`.
     `DO UPDATE` (not `DO NOTHING`) is correct here: with `DO NOTHING`, conflicting rows return
     no row, so `cursor.results()` would yield fewer ids than input tuples and silently break
     positional alignment with the input list.
   - DONE: `is_reference` parameter added to `log_image_prediction`/`log_tile_prediction`
     (extends the existing tuple shape / SQL; fixed a column-name typo along the way, see
     below).
   - DONE: `log_tile_stack_member` now does `ON CONFLICT (tile_stack_id, image_id) DO UPDATE
     SET tile_stack_id = EXCLUDED.tile_stack_id, image_id = EXCLUDED.image_id RETURNING id`
     (previously had no `RETURNING id` at all; `DO UPDATE` rather than `DO NOTHING` for the
     same reason as `log_tile_channel_stats` above — guarantees a row comes back on conflict
     so ids stay positionally aligned with the input list).
   - DONE: `get_reference_samples(run_id) -> list[tuple]` (renamed from the originally-planned
     coarse `has_reference(run_id) -> bool`, and fixed a `run_id: int` type-hint bug — `run_id`
     is `VARCHAR(64)`). Returns every `(plate, well, field)` already logged as reference for
     that run, so `compute_reference.py` can diff against the manifest's `val_samples` and
     process only what's missing — a single boolean would have skipped the entire remaining
     computation after a crash, since reference spans many samples and a crash partway through
     would still leave *some* reference rows present.
   - Also fixed during review: `is_refrence` typo in `log_image_prediction`/
     `log_tile_prediction`'s SQL (didn't match the actual `is_reference` column from
     `02_refrence.sql` — would have raised `UndefinedColumn` at runtime).
   - **Decided against** adding `UNIQUE (plate, well, field, run_id)` to `image_prediction` (or
     `UNIQUE (image_pred_id, tile_stack_id)` to `tile_prediction`). Reasons: `image_prediction`
     is an append-only event log that drift's `window_start`/`window_end` filtering depends on —
     a uniqueness constraint would mean the same well re-predicted months later overwrites (or
     is rejected) instead of adding a second row, destroying the time series. Also
     `(plate, well, field)` is coarser than real image identity (filenames also carry `T`/`L`/
     `A`/`Z`, which `image_prediction` doesn't store), so genuinely distinct acquisitions could
     collide. And it would contradict the existing `test_each_insert_creates_new_row`
     expectation in `tests/db/test_dblogger.py`. Resumability is handled by the query above
     instead.
3. **Production-side channel-stats logging**: extend `TilePredictor`/`api/main.py`'s call into
   `DBLogger` so every live `/predict` call also computes per-tile per-channel
   mean/std/p1/p5/p95/p99 and logs them via `log_tile_channel_stats`, using the ids returned
   from `log_tile_stack_member`.
4. **DONE**: `monitoring/compute_reference.py` — instantiates `TilePredictor` directly
   (tracking_uri + model_name/run_name via `api/config.py`'s `Settings`, same source as
   `api/main.py`), reads `dataset_manifest.json` from `predictor.model_info["artifact_dir"]`
   (already downloaded by `TilePredictor._load_run_config`), resolves val-sample file paths via
   `root_path` + `channel_files`, runs each sample through the existing `predictor.predict()`
   pipeline (preprocess → tile → predict_tiles → majority vote, plus channel-stats logging) with
   `is_reference=True` and the known `label` set in `image_metadata`. Skips samples that already
   have reference rows (via `DBLogger.get_reference_samples`, step 2's resumability query), so an
   interrupted run resumes instead of restarting or silently no-op'ing.
5. **Docker/Makefile wiring**: `compute-reference` target in `docker/makefile`, running
   `python -m monitoring.compute_reference` inside the existing `api` image (already has
   `mlflow`/`torch`/`timm`/`psycopg` — no new image needed), mirroring `train-run`'s
   `$(CMD)`-passthrough pattern.
6. **DONE**: `monitoring/run_drift_report.py` — pulls reference rows
   (`reference_image_prediction`/`reference_tile_prediction` views) and current-window rows
   (`live_image_prediction`/`live_tile_prediction` + `tile_channel_stats`, with the reference-well
   `NOT EXISTS` exclusion below) via new `DBLogger` fetch methods, and runs Evidently
   `DataDriftPreset` for **three separate groups** (they have different row cardinalities —
   images vs. tiles vs. tile-channels — so they can't share one DataFrame):
   `image_level` (`p_label` categorical + `vote_fraction`/`avg_confidence` numeric),
   `tile_level` (`p_label` categorical + `confidence` numeric, deduped per tile), and
   `channel_stats` (per-channel pixel stats pivoted long→wide to `channel_<n>_<stat>`, numeric).
   Results are extracted from Evidently's `as_dict()` `drift_by_columns`; each group's HTML plus a
   combined `metrics.json` are written under `monitoring/reports/drift_<run>_<timestamp>/`, and one
   `drift_report` row (`dataset_drift`/`n_columns_drifted`/`n_columns_total` aggregated across
   groups, `report_path` = that dir) + its per-column `drift_report_column` rows
   (`column_group` = the group name) are inserted via `DBLogger.log_drift_report`/
   `log_drift_report_column`. `run_id` is taken from `--run-id`, or (when omitted) read off the
   running API's `/model` endpoint via stdlib `urllib` (`--api-url` / `$API_BASE_URL`, default
   `http://localhost:8000`) — so the drift job needs **no mlflow / tracking server**, and always
   targets whatever model is actually being served. Window is `--window-days` (default 7) or
   explicit `--window-start`/`--window-end`.
   Evidently is pinned to `0.6.7` (last release of the classic
   `evidently.report.Report`/`metric_preset`/`as_dict()` API; 0.7.x is an incompatible rewrite).

   **Important: the live-window query must exclude validation samples.** A validation well can
   legitimately be re-imaged and predicted through the API in production — that prediction is a
   real production event and should stay in the log, but including it in the drift window would
   compare the reference set partly against *itself*, understating drift. Since we deliberately
   did not add uniqueness constraints to prevent those rows from being written, this is handled
   at query time:
   ```sql
   -- image-level current window, excluding wells that are part of this model's reference set
   SELECT l.plate, l.well, l.field, l.p_label, l.vote_fraction, l.avg_confidence, l.created_at
   FROM live_image_prediction l
   WHERE l.run_id = %(run_id)s
     AND l.created_at >= %(window_start)s
     AND l.created_at <  %(window_end)s
     AND NOT EXISTS (
         SELECT 1
         FROM reference_image_prediction r
         WHERE r.run_id = l.run_id
           AND r.plate  = l.plate
           AND r.well   = l.well
           AND r.field  = l.field
     );
   ```
   ```sql
   -- tile-level current window (+ per-channel input stats), same exclusion applied via the
   -- parent image_prediction row; channel stats join through tile_stack_member so they are
   -- shared/deduped per physical tile-channel rather than per prediction event
   SELECT t.p_label,
          t.confidence,
          im.channel        AS channel,
          s.mean, s.std, s.p1, s.p5, s.p95, s.p99
   FROM live_tile_prediction t
   JOIN live_image_prediction l   ON l.id = t.image_pred_id
   JOIN tile_stack_member tsm     ON tsm.tile_stack_id = t.tile_stack_id
   JOIN image_metadata im         ON im.id = tsm.image_id
   LEFT JOIN tile_channel_stats s ON s.tile_stack_member_id = tsm.id
   WHERE t.run_id = %(run_id)s
     AND t.created_at >= %(window_start)s
     AND t.created_at <  %(window_end)s
     AND NOT EXISTS (
         SELECT 1
         FROM reference_image_prediction r
         WHERE r.run_id = l.run_id
           AND r.plate  = l.plate
           AND r.well   = l.well
           AND r.field  = l.field
     );
   ```
   The reference side uses the same two shapes without the window/`NOT EXISTS` clauses, reading
   from `reference_image_prediction`/`reference_tile_prediction` filtered by `run_id`. Both
   sides then get pivoted (channel stats long → wide: `channel_1_mean`, `channel_1_p95`, ...)
   and handed to Evidently as reference vs. current DataFrames.
7. **DONE**: `docker/jobs/monitoring/` — lightweight `Dockerfile` (`python:3.11-slim` +
   `requirements/monitoring_req.txt` only; `db_req.txt` folded in since monitoring_req already
   pins `psycopg` — no torch/mlflow/timm) with `database/` + `monitoring/` code baked in (COPY,
   not mounted — this is a deployable/scheduled job). `docker-compose.yaml` is an ephemeral
   `run --rm` job on **both** `pg_network` (Postgres) and `ml-platform` (to reach the `api`
   service's `/model` endpoint for run_id resolution — the plan's original "pg_network only" was
   revised because run_id now comes from the API, not mlflow), bind-mounting
   `monitoring/reports/` for output persistence. Env via `.env.monitoring` (`API_DB_URI` +
   `API_BASE_URL=http://api:8000`), with a committed `.env.monitoring.example`. Added
   `drift-report-run` target to `docker/makefile` (mirrors `compute-reference`'s
   `$(CMD)`-passthrough, defaulting to `python -m monitoring.run_drift_report`). This is
   deliberately the shape a cron/Airflow schedule will later wrap — same image, same command,
   just a different trigger.
8. **DONE**: `requirements/monitoring_req.txt` with `evidently==0.6.7`, `pandas`,
   `psycopg[binary,pool]`, `pydantic-settings` (needed by `monitoring.config.MonitoringSettings`,
   a dedicated settings class with just `db_uri` + API `base_url` — not `api.config.Settings`,
   which carries irrelevant model-serving fields); no `mlflow` — run_id is read off the API's
   `/model` endpoint via stdlib `urllib`.
9. **Tests** (`tests/monitoring/`, new): reference-builder integration tests against a real
   Postgres (matching `tests/db/test_dblogger.py` conventions, small synthetic manifest + fake
   images), and a fixture-based drift-script test with synthetic shifted vs. unshifted data
   (assert `drift_report`/`drift_report_column` rows correctly flag the shifted column and not
   the stable one).
10. **CI**: new `.github/workflows/ci_monitoring.yml` mirroring `ci_db.yml`'s Postgres-service
    pattern (spin up Postgres, apply `01_prediction.sql` + `02_refrence.sql`, run
    `pytest tests/monitoring/`), scoped to `database/**`, `monitoring/**`, `tests/monitoring/**`,
    `requirements/monitoring_req.txt`, `docker/jobs/monitoring/**`.

## Framing note (for README/docs later)

Drift detection here is an **unsupervised early-warning trigger, not a performance metric** —
`p_label`/input drift alone cannot prove the model is wrong (a shift could reflect a real
change in what's being imaged, not a bug). It's most informative combined with the
`avg_confidence`/`vote_fraction` trend and the input (channel-stat) drift signal together:
input drift + no prediction drift, or prediction drift + no input drift, are the more
actionable/suspicious patterns than either alone. This should be stated explicitly wherever
drift results are surfaced, so it doesn't get misread as "the model is now X% accurate."

## Verification

- `pytest tests/db -v` and new `pytest tests/monitoring -v` against a local/CI Postgres with
  both `01_prediction.sql` and `02_refrence.sql` applied.
- Manual smoke test: point the API at a real trained run, run
  `make compute-reference RUN_ID=...` to populate reference rows, then run
  `make drift-report-run RUN_ID=...` and confirm a `drift_report` row + report file are
  created, with no drift flagged on a freshly computed reference vs. itself.
- `ruff check` on all new/changed files, matching existing CI lint steps.
- `docker compose -f docker/jobs/monitoring/docker-compose.yaml build` succeeds and stays
  meaningfully smaller/faster to build than the `api` image (no torch/mlflow/timm).
