-- SCHEMA: benchmark dataset (model-independent quality baseline)
-- A fixed, curated set of samples that were never in any model's training set.
-- It exists independently of any model (meaningful even with zero models
-- trained) and is scored once per model by compute_benchmark.py, giving every
-- run a comparable supervised-quality baseline. Distinct from `reference`
-- (a model's own validation split, tracked via the is_reference flag in
-- 02_reference.sql), which is the *drift* baseline.

-- ══════════════════════════════════════════════════════════════════════════
-- BENCHMARK DATASET
-- ══════════════════════════════════════════════════════════════════════════

-- One row per curated benchmark sample (a physical field of view with known
-- ground truth). Registered once by register_benchmark.py; independent of runs.
CREATE TABLE IF NOT EXISTS benchmark_dataset (
    id SERIAL PRIMARY KEY,
    plate VARCHAR(255) NOT NULL,
    well VARCHAR(255) NOT NULL,
    field INTEGER NOT NULL,
    t_label VARCHAR(255) NOT NULL,            -- known ground truth
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (plate, well, field)
);

-- Which single-channel image files make up a benchmark sample. Mirrors
-- tile_stack_member's channel-position semantics (channel_index = the model's
-- input channel-axis position, ascending by channel number).
CREATE TABLE IF NOT EXISTS benchmark_dataset_member (
    id SERIAL PRIMARY KEY,
    benchmark_id INTEGER REFERENCES benchmark_dataset(id),
    image_id INTEGER REFERENCES image_metadata(id),
    channel_index INTEGER NOT NULL,
    UNIQUE (benchmark_id, image_id)
);

-- ══════════════════════════════════════════════════════════════════════════
-- BENCHMARK PREDICTIONS: same tables, one nullable FK
-- ══════════════════════════════════════════════════════════════════════════

-- Tag a prediction as a benchmark score via a nullable FK (NULL = ordinary
-- production/reference prediction). Richer than a boolean flag: it also records
-- which benchmark sample the row scored. A model's benchmark score is
-- WHERE run_id = %s AND benchmark_id IS NOT NULL.
ALTER TABLE image_prediction ADD COLUMN IF NOT EXISTS benchmark_id INTEGER DEFAULT NULL REFERENCES benchmark_dataset(id);
ALTER TABLE tile_prediction  ADD COLUMN IF NOT EXISTS benchmark_id INTEGER DEFAULT NULL REFERENCES benchmark_dataset(id);

CREATE INDEX IF NOT EXISTS idx_image_prediction_run_benchmark ON image_prediction(run_id, benchmark_id);
CREATE INDEX IF NOT EXISTS idx_tile_prediction_run_benchmark  ON tile_prediction(run_id, benchmark_id);

-- 02_reference.sql's live_*/reference_* views were defined as `SELECT *` before
-- benchmark_id existed -- Postgres expands `*` at CREATE VIEW time, so ALTER
-- TABLE alone does not add the new column to them. Recreate to pick it up
-- (CREATE OR REPLACE VIEW allows appending columns at the end).
CREATE OR REPLACE VIEW live_image_prediction AS
    SELECT * FROM image_prediction WHERE is_reference = FALSE;

CREATE OR REPLACE VIEW live_tile_prediction AS
    SELECT * FROM tile_prediction WHERE is_reference = FALSE;

CREATE OR REPLACE VIEW reference_image_prediction AS
    SELECT * FROM image_prediction WHERE is_reference = TRUE;

CREATE OR REPLACE VIEW reference_tile_prediction AS
    SELECT * FROM tile_prediction WHERE is_reference = TRUE;
