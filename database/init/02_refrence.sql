-- SCHEMA: monitoring additions (reference data, input drift, drift reports)
-- Extends 01_prediction.sql's tables via ALTER TABLE rather than duplicating
-- them, since a validation-set prediction is structurally identical to a
-- production one (same image_metadata -> tile_stack -> tile_stack_member ->
-- image_prediction -> tile_prediction pipeline).

-- ══════════════════════════════════════════════════════════════════════════
-- REFERENCE vs. LIVE: same tables, one flag
-- ══════════════════════════════════════════════════════════════════════════

-- Reference rows come from running a served model's validation set through
-- the exact same tiling/prediction pipeline as production, once, right
-- before that model goes live. is_reference distinguishes the two without
-- needing a parallel set of mirrored tables.
ALTER TABLE image_prediction ADD COLUMN IF NOT EXISTS is_reference BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE tile_prediction ADD COLUMN IF NOT EXISTS is_reference BOOLEAN NOT NULL DEFAULT FALSE;

-- Every monitoring query filters by (run_id, is_reference), and live-window
-- queries additionally filter by created_at.
CREATE INDEX IF NOT EXISTS idx_image_prediction_run_ref ON image_prediction(run_id, is_reference);
CREATE INDEX IF NOT EXISTS idx_tile_prediction_run_ref ON tile_prediction(run_id, is_reference);
CREATE INDEX IF NOT EXISTS idx_image_prediction_run_ref_created ON image_prediction(run_id, is_reference, created_at);
CREATE INDEX IF NOT EXISTS idx_tile_prediction_run_ref_created ON tile_prediction(run_id, is_reference, created_at);

-- Canonical "production only" views: anything that computes live metrics
-- (dashboards, step 5's retrain trigger) should query these, not the raw
-- tables, so a missing `is_reference` filter can't silently mix validation
-- data into production numbers.
CREATE OR REPLACE VIEW live_image_prediction AS
    SELECT * FROM image_prediction WHERE is_reference = FALSE;

CREATE OR REPLACE VIEW live_tile_prediction AS
    SELECT * FROM tile_prediction WHERE is_reference = FALSE;

-- Symmetric convenience views onto the reference-only rows.
CREATE OR REPLACE VIEW reference_image_prediction AS
    SELECT * FROM image_prediction WHERE is_reference = TRUE;

CREATE OR REPLACE VIEW reference_tile_prediction AS
    SELECT * FROM tile_prediction WHERE is_reference = TRUE;

-- Reference summary grouped by p_label (not t_label): production traffic
-- has no ground truth, so drift comparisons must key on predicted label on
-- both sides to be apples-to-apples. t_label is known for reference rows
-- (it's validation data), so accuracy is included here as a free bonus, but
-- it is NOT part of the drift comparison itself.
CREATE OR REPLACE VIEW reference_image_summary AS
    SELECT run_id,
           p_label,
           COUNT(*) AS n_wells,
           AVG(vote_fraction) AS avg_vote_fraction,
           AVG(avg_confidence) AS avg_confidence,
           AVG((p_label = t_label)::int)::float AS accuracy
    FROM image_prediction
    WHERE is_reference = TRUE
    GROUP BY run_id, p_label;

-- ══════════════════════════════════════════════════════════════════════════
-- INPUT DRIFT: per-channel pixel statistics
-- ══════════════════════════════════════════════════════════════════════════

-- One row per (tile_stack, channel-image) pair, i.e. one row per
-- tile_stack_member. Pixel statistics are a deterministic function of the
-- raw pixel data, not of which run/model predicted the tile -- so they are
-- computed once per physical tile-channel and shared by every
-- image_prediction/tile_prediction (production or reference) that reuses
-- the same tile_stack. This is a dedicated 1:1 extension table (not columns
-- bolted onto tile_stack_member itself) so the relationship table
-- (tile_stack_member) keeps its single responsibility.
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

-- ══════════════════════════════════════════════════════════════════════════
-- DRIFT REPORTS
-- ══════════════════════════════════════════════════════════════════════════

-- One row per drift-check execution (manual for now, scheduled in step 7).
-- The full Evidently HTML/JSON report is written to report_path (local disk
-- now, s3://... URL later -- swapping storage backend doesn't touch this
-- schema). No MLflow involvement: this data is queried/joined against
-- image_prediction/tile_prediction directly, and MLflow's per-run artifact
-- model isn't a good fit for a recurring report.
CREATE TABLE IF NOT EXISTS drift_report (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,               -- target serving model this check is about
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dataset_drift BOOLEAN NOT NULL,
    n_columns_drifted INTEGER NOT NULL,
    n_columns_total INTEGER NOT NULL,
    report_path VARCHAR(512) NOT NULL
);

-- Per-column drift detail, one row per column compared (e.g. p_label,
-- vote_fraction, avg_confidence, confidence, channel_1_mean, channel_1_p95,
-- ...) -- directly SQL-queryable/joinable, e.g. for step 5's retrain trigger.
CREATE TABLE IF NOT EXISTS drift_report_column (
    id SERIAL PRIMARY KEY,
    drift_report_id INTEGER REFERENCES drift_report(id) NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    column_group VARCHAR(50) NOT NULL,         -- 'image_level' | 'tile_level' | 'channel_stats'
    drift_score FLOAT NOT NULL,
    drifted BOOLEAN NOT NULL,
    stat_test VARCHAR(50)                      -- e.g. 'ks', 'psi', 'chi2'
);

CREATE INDEX IF NOT EXISTS idx_drift_report_run_id ON drift_report(run_id, computed_at);
CREATE INDEX IF NOT EXISTS idx_drift_report_column_report_id ON drift_report_column(drift_report_id);
