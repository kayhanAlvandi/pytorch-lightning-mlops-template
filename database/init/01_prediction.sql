
-- SCHEMA: prediction database

-- ── Single-channel image file ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS image_metadata (
    id SERIAL PRIMARY KEY,
    plate VARCHAR(255) NOT NULL,
    well VARCHAR(255) NOT NULL,
    field INTEGER NOT NULL,
    channel INTEGER NOT NULL,
    root_path VARCHAR(255) NOT NULL,
    file_name VARCHAR(255) NOT NULL UNIQUE,
    shape_x INTEGER NOT NULL,
    shape_y INTEGER NOT NULL
);

-- ══════════════════════════════════════════════════════════════════════════
-- STACKS (multi-channel groupings that predictions run on)
-- ══════════════════════════════════════════════════════════════════════════

-- ── Multi-channel tile stack ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tile_stack (
    id SERIAL PRIMARY KEY,
    stack_hash VARCHAR(64) NOT NULL UNIQUE,   -- hash of hash(sorted(image_ids) + (x_left, y_top, crop_size))
    row_ind INTEGER NOT NULL,
    col_ind INTEGER NOT NULL,
    x_left INTEGER NOT NULL,
    y_top INTEGER NOT NULL,
    crop_size INTEGER NOT NULL
);

-- ── Which single-channel tiles make up a tile stack ──────────────────────
CREATE TABLE IF NOT EXISTS tile_stack_member (
    id SERIAL PRIMARY KEY,
    tile_stack_id INTEGER REFERENCES tile_stack(id),
    image_id INTEGER REFERENCES image_metadata(id),
    UNIQUE (tile_stack_id, image_id)
);

-- ══════════════════════════════════════════════════════════════════════════
-- PREDICTIONS
-- ══════════════════════════════════════════════════════════════════════════

-- ── Image-level prediction (majority vote result) ────────────────────────
CREATE TABLE IF NOT EXISTS image_prediction (
    id SERIAL PRIMARY KEY,
    plate VARCHAR(255) NOT NULL,              -- denormalized for easy access
    well VARCHAR(255) NOT NULL,               -- denormalized for easy access
    field INTEGER NOT NULL,                   -- denormalized for easy access
    run_id VARCHAR(64) NOT NULL,              -- MLflow run_id, query MLflow for model details
    p_label VARCHAR(255) NOT NULL,
    t_label VARCHAR(255) DEFAULT NULL,
    total_tiles INTEGER NOT NULL,
    vote_fraction FLOAT NOT NULL,
    avg_confidence FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Tile-level prediction ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tile_prediction (
    id SERIAL PRIMARY KEY,
    image_pred_id INTEGER REFERENCES image_prediction(id),
    tile_stack_id INTEGER REFERENCES tile_stack(id),
    run_id VARCHAR(64) NOT NULL,              -- MLflow run_id, query MLflow for model details
    p_label VARCHAR(255) NOT NULL,
    t_label VARCHAR(255) DEFAULT NULL,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);