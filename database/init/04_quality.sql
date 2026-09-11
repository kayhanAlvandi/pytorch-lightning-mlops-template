-- SCHEMA: supervised quality reports
-- One row per quality-report execution (manual for now, scheduled later). The
-- full Evidently HTML/JSON report (including the confusion matrix) is written
-- to report_path; only top-line accuracy/F1 + sample counts are stored here for
-- SQL querying, mirroring how drift_report keeps raw detail out of Postgres.
--
-- The baseline is the model's benchmark score (03_benchmark.sql), not its
-- validation reference: benchmark_* columns are that run's benchmark quality,
-- current_* columns are the labeled production window's quality.
CREATE TABLE IF NOT EXISTS quality_report (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,               -- target serving model this check is about
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
