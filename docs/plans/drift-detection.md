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
4. **`monitoring/compute_reference.py`**: given a `--run-id`, instantiate `TilePredictor`
   directly (tracking_uri + model_name/run_name, same as `api/config.py`/`main.py`), download
   `dataset_manifest.json` + `hydra_config.yaml` (reusing `TilePredictor._download_hydra_config`/
   `_load_run_config`), resolve val-sample file paths via `root_dir` + `channel_files`, run
   each sample through `preprocess_image` → `tile_image` → `predict_tiles` (reusing the loaded
   model), compute channel stats per tile, and insert via the now-shared `DBLogger` methods
   with `is_reference=True` and the known `t_label`. Skips samples that already have reference
   rows (per-sample resumability query in step 2), so an interrupted run resumes instead of
   restarting or silently no-op'ing.
5. **Docker/Makefile wiring**: `compute-reference` target in `docker/makefile`, running
   `python -m monitoring.compute_reference` inside the existing `api` image (already has
   `mlflow`/`torch`/`timm`/`psycopg` — no new image needed), mirroring `train-run`'s
   `$(CMD)`-passthrough pattern.
6. **`monitoring/run_drift_report.py`**: pull reference rows (`reference_image_summary` view /
   raw `reference_image_prediction`/`reference_tile_prediction` views) and current-window rows
   from `live_image_prediction`/`live_tile_prediction` + `tile_channel_stats`, both grouped by
   `p_label`; build an Evidently `Report` (categorical drift on label distribution, numeric
   drift on `vote_fraction`/`avg_confidence`/`confidence`/channel stats — channel stats pivoted
   from long to wide, one column per channel×stat, before feeding Evidently); write the
   HTML/JSON report to `monitoring/reports/`; insert one `drift_report` row + its
   `drift_report_column` rows via `DBLogger`.

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
7. **`docker/jobs/monitoring/`** (new): `Dockerfile` (lightweight — `db_req.txt` + new
   `requirements/monitoring_req.txt`, no torch/mlflow/timm) and `docker-compose.yaml`
   (ephemeral `run --rm`, `pg_network` only), mirroring `docker/jobs/train/`'s shape. This is
   deliberately the shape step 7 will later wrap in a cron/Airflow schedule — same image, same
   command, just a different trigger. Add `drift-report-run` target to `docker/makefile`.
8. **Dependencies**: new `requirements/monitoring_req.txt` with `evidently`, `pandas`,
   `psycopg[binary,pool]`.
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
