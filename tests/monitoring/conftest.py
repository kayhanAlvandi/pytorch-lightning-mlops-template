"""Shared fixtures for monitoring tests.

Unit tests under tests/monitoring/unit/ use the fakes defined here to avoid
real database, API, MLflow, or MongoDB dependencies. DB integration tests
under tests/monitoring/db/ have their own conftest that applies the full
schema set against a real PostgreSQL instance.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Manifest helpers
# ─────────────────────────────────────────────────────────────────────────────

PLATE = "MIG-Exp03-CP-40X-bin1X1"
WELL = "K07"
ROOT_PATH = "/data/benchmark"
SHAPE = [2048, 2048]


def _channel_files(field: int, well: str = WELL, plate: str = PLATE) -> dict:
    """Build a {channel_number_str: filename} dict for a 5-channel sample."""
    return {
        str(ch): f"{plate}_{well}_T0001F{field:03d}L01A01Z01C0{ch}.jxl"
        for ch in range(1, 6)
    }


@pytest.fixture
def flat_manifest(tmp_path: Path) -> Path:
    """A flat {"samples": [...]} manifest with 3 benchmark samples."""
    samples = []
    for i, well in enumerate(["K07", "K08", "K09"], start=1):
        samples.append({
            "plate": PLATE,
            "well": well,
            "field": 1,
            "label": f"Class{chr(64 + i)}",
            "root_path": ROOT_PATH,
            "shape": SHAPE,
            "channel_files": _channel_files(1, well=well),
        })
    manifest = {"samples": samples}
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def train_val_manifest(tmp_path: Path) -> Path:
    """A training-style manifest with train_samples + val_samples."""
    def _sample(well: str, label: str):
        return {
            "plate": PLATE,
            "well": well,
            "field": 1,
            "label": label,
            "root_path": ROOT_PATH,
            "shape": SHAPE,
            "channel_files": _channel_files(1, well=well),
        }
    manifest = {
        "train_samples": [_sample("K07", "ClassA"), _sample("K08", "ClassB")],
        "val_samples": [_sample("K09", "ClassC")],
    }
    path = tmp_path / "train_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def manifest_missing_shape(tmp_path: Path) -> Path:
    """A manifest entry without the 'shape' key (should raise on parse)."""
    manifest = {"samples": [{
        "plate": PLATE, "well": WELL, "field": 1, "label": "ClassA",
        "root_path": ROOT_PATH,
        "channel_files": _channel_files(1),
    }]}
    path = tmp_path / "no_shape.json"
    path.write_text(json.dumps(manifest))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Fake DB logger for unit tests
# ─────────────────────────────────────────────────────────────────────────────

class FakeDBLogger:
    """Minimal in-memory stand-in for DBLogger used by monitoring unit tests.

    Only the methods the monitoring scripts call are implemented; each stores
    its inputs so tests can assert on what was logged without a real database.
    """

    def __init__(self):
        self.connected = False
        self.closed = False
        self._next_id = 1
        self._id_counter = iter(range(1, 100_000))

        # Benchmark registration
        self.benchmark_samples: list[tuple] = []          # (plate, well, field)
        self.logged_image_metadata: list[list[tuple]] = []
        self.logged_benchmark_samples: list[tuple] = []   # (plate, well, field, t_label)
        self.logged_benchmark_members: list[list[tuple]] = []

        # Label backfill
        self.unlabeled_wells: list[tuple] = []             # (plate, well)
        self.updated_labels: list[tuple] = []              # (plate, well, t_label)

        # Reference / benchmark scoring
        self.reference_samples: list[tuple] = []           # (plate, well, field)
        self.benchmark_predictions: list[int] = []         # benchmark_ids already scored
        self.benchmark_member_rows: list[dict] = []

        # Drift / quality fetches
        self.reference_image_level: list[dict] = []
        self.current_image_level: list[dict] = []
        self.reference_tile_level: list[dict] = []
        self.current_tile_level: list[dict] = []
        self.benchmark_quality: list[dict] = []
        self.current_quality: list[dict] = []

        # Report logging
        self.drift_reports: list[tuple] = []
        self.drift_report_columns: list[list[tuple]] = []
        self.quality_reports: list[tuple] = []

    def _next(self) -> int:
        return next(self._id_counter)

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True

    # ── Benchmark registration ──
    def get_benchmark_samples(self):
        return list(self.benchmark_samples)

    def log_image_metadata(self, rows):
        self.logged_image_metadata.append(list(rows))
        return [self._next() for _ in rows]

    def log_benchmark_sample(self, sample):
        self.logged_benchmark_samples.append(sample)
        return self._next()

    def log_benchmark_members(self, members):
        self.logged_benchmark_members.append(list(members))

    # ── Label backfill ──
    def fetch_unlabeled_wells(self):
        return list(self.unlabeled_wells)

    def update_t_label(self, plate, well, t_label):
        self.updated_labels.append((plate, well, t_label))
        return (1, 4)  # pretend 1 image row, 4 tile rows updated

    # ── Reference / benchmark scoring ──
    def get_reference_samples(self, run_id):
        return list(self.reference_samples)

    def get_benchmark_predictions(self, run_id):
        return list(self.benchmark_predictions)

    def fetch_benchmark_members(self):
        return list(self.benchmark_member_rows)

    # ── Drift / quality ──
    def fetch_reference_image_level(self, run_id):
        return list(self.reference_image_level)

    def fetch_current_image_level(self, run_id, window_start, window_end):
        return list(self.current_image_level)

    def fetch_reference_tile_level(self, run_id):
        return list(self.reference_tile_level)

    def fetch_current_tile_level(self, run_id, window_start, window_end):
        return list(self.current_tile_level)

    def fetch_benchmark_quality(self, run_id):
        return list(self.benchmark_quality)

    def fetch_current_quality(self, run_id, window_start, window_end):
        return list(self.current_quality)

    def log_drift_report(self, report):
        self.drift_reports.append(report)
        return self._next()

    def log_drift_report_column(self, columns):
        self.drift_report_columns.append(list(columns))
        return len(columns)

    def log_quality_report(self, report):
        self.quality_reports.append(report)
        return self._next()


@pytest.fixture
def fake_db():
    return FakeDBLogger()
