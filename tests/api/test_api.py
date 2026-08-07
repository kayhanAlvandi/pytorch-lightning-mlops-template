"""API tests using FastAPI's TestClient.

The real model loading (MLflow) is bypassed:
  - startup is forced into the "no model source" branch, so no network / MLflow
  - a FakePredictor is injected for the success-path tests
"""
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main


class FakeDBLogger:
    """In-memory stand-in for DBLogger that records all calls without touching a real DB."""

    def __init__(self):
        self.image_metadata_calls: list[list[tuple]] = []
        self.tile_stack_calls: list[list[tuple]] = []
        self.tile_stack_member_calls: list[list[tuple]] = []
        self.image_prediction_calls: list[tuple] = []
        self.tile_prediction_calls: list[list[tuple]] = []

        self._next_img_id = 1
        self._next_tile_stack_id = 1
        self._next_img_pred_id = 1

    def log_image_metadata(self, image_metadata: list[tuple]) -> list[int]:
        self.image_metadata_calls.append(image_metadata)
        ids = list(range(self._next_img_id, self._next_img_id + len(image_metadata)))
        self._next_img_id += len(image_metadata)
        return ids

    def log_tile_stack(self, tile_stack_metadata: list[tuple]) -> list[int]:
        self.tile_stack_calls.append(tile_stack_metadata)
        ids = list(range(self._next_tile_stack_id, self._next_tile_stack_id + len(tile_stack_metadata)))
        self._next_tile_stack_id += len(tile_stack_metadata)
        return ids

    def log_tile_stack_member(self, tile_stack_members: list[tuple]) -> None:
        self.tile_stack_member_calls.append(tile_stack_members)

    def log_image_prediction(self, image_prediction: tuple) -> int:
        self.image_prediction_calls.append(image_prediction)
        pred_id = self._next_img_pred_id
        self._next_img_pred_id += 1
        return pred_id

    def log_tile_prediction(self, tile_predictions: list[tuple]) -> None:
        self.tile_prediction_calls.append(tile_predictions)

    def close(self) -> None:
        """No-op close to satisfy the lifespan cleanup."""


class FakePredictor:
    """Minimal stand-in for TilePredictor with the attributes the API reads."""

    def __init__(self, db_logger=None):
        self.crop_size = 32
        self.stride = 32
        self.in_channels = 3
        self.class_names = ["ClassA", "ClassB"]
        self.device = "cpu"
        self.db_logger = db_logger
        self.model_info = {
            "source": "fake",
            "model_class": "CNNClassifier",
            "backbone": None,
            "run_id": "deadbeef",
            "num_classes": 2,
            "crop_size": 32,
            "in_channels": 3,
        }

    def predict(self, image, image_metadata):
        if self.db_logger:
            from utils.filename_parser import clean_image_metadata, clean_tiles_metadata
            cleaned = clean_image_metadata(image_metadata)
            img_ids = self.db_logger.log_image_metadata(cleaned)
            fake_tiles = [{"row": 0, "col": 0, "x": 0, "y": 0, "crop_size": self.crop_size}]
            tile_stack_metadata = clean_tiles_metadata(fake_tiles, img_ids)
            tile_stack_ids = self.db_logger.log_tile_stack(tile_stack_metadata)
            members = [(ts_id, img_id) for ts_id in tile_stack_ids for img_id in img_ids]
            self.db_logger.log_tile_stack_member(members)
            img_pred_id = self.db_logger.log_image_prediction(
                ("plate", "well", 1, self.model_info["run_id"], "ClassA", None, 1, 1.0, 0.95)
            )
            self.db_logger.log_tile_prediction(
                [(img_pred_id, tile_stack_ids[0], self.model_info["run_id"], "ClassA", None, 0.95)]
            )
        return {
            "predicted_class": "ClassA",
            "total_tiles": 1,
            "vote_fraction": 1.0,
            "confidence": 0.95,
        }


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch):
    """Force the startup lifespan into the no-model branch so tests never hit MLflow."""
    monkeypatch.setattr(type(api_main.settings), "has_model_source", property(lambda self: False))


@pytest.fixture
def client():
    with TestClient(api_main.app) as c:
        yield c


def _npy_upload(shape=(3, 32, 32)):
    arr = np.zeros(shape, dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return {"files": ("img.npy", buf, "application/octet-stream")}


def _tif_file(filename: str, size=(32, 32)) -> tuple:
    """Create a single-channel .tif upload tuple with the given filename."""
    arr = np.zeros(size, dtype=np.uint16)
    ok, encoded = cv2.imencode(".tif", arr)
    assert ok, f"Failed to encode tif for {filename}"
    buf = io.BytesIO(encoded.tobytes())
    return ("files", (filename, buf, "image/tif"))


def _multi_channel_tif_files(n_channels=3, size=(32, 32)) -> list[tuple]:
    """Create multi-channel .tif uploads with microscopy filenames (C01..CN)."""
    files = []
    for ch in range(1, n_channels + 1):
        fname = f"PLATE1_A01_T0001F001L01A01Z01C0{ch}.tif"
        files.append(_tif_file(fname, size))
    return files


def test_health_reports_no_model(client):
    api_main.predictor = None
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_model_endpoint_503_without_model(client):
    api_main.predictor = None
    r = client.get("/model")
    assert r.status_code == 503


def test_predict_503_without_model(client):
    api_main.predictor = None
    r = client.post("/predict", files=_npy_upload(), data={"root_path": "/tmp"})
    assert r.status_code == 503


def test_predict_success_with_mock(client):
    api_main.predictor = FakePredictor()
    r = client.post("/predict", files=_npy_upload(), data={"root_path": "/tmp"})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == "ClassA"
    assert body["total_tiles"] == 1


def test_predict_rejects_bad_npy_shape(client):
    api_main.predictor = FakePredictor()
    # 2D array is invalid: endpoint expects (C, H, W)
    r = client.post("/predict", files=_npy_upload(shape=(32, 32)), data={"root_path": "/tmp"})
    assert r.status_code == 400


def test_predict_calls_db_logger(client):
    """When a FakeDBLogger is attached, all five log_* methods are called once per request."""
    fake_db = FakeDBLogger()
    api_main.predictor = FakePredictor(db_logger=fake_db)
    r = client.post("/predict", files=_npy_upload(), data={"root_path": "/tmp"})
    assert r.status_code == 200
    assert len(fake_db.image_metadata_calls) == 1, "log_image_metadata should be called once"
    assert len(fake_db.tile_stack_calls) == 1, "log_tile_stack should be called once"
    assert len(fake_db.tile_stack_member_calls) == 1, "log_tile_stack_member should be called once"
    assert len(fake_db.image_prediction_calls) == 1, "log_image_prediction should be called once"
    assert len(fake_db.tile_prediction_calls) == 1, "log_tile_prediction should be called once"


def test_model_info_with_mock(client):
    api_main.predictor = FakePredictor()
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["num_classes"] == 2
    assert body["class_names"] == ["ClassA", "ClassB"]


# ── Multi-channel upload tests ──────────────────────────────────────────────


def test_predict_multi_channel_success(client):
    """Upload multiple .tif files (one per channel) and verify the API stacks them."""
    api_main.predictor = FakePredictor()
    files = _multi_channel_tif_files(n_channels=3)
    r = client.post("/predict", files=files, data={"root_path": "/tmp"})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == "ClassA"
    assert body["total_tiles"] == 1


def test_predict_multi_channel_rejects_mismatched_shapes(client):
    """Channels with different dimensions should return 400."""
    api_main.predictor = FakePredictor()
    files = [
        _tif_file("PLATE1_A01_T0001F001L01A01Z01C01.tif", size=(32, 32)),
        _tif_file("PLATE1_A01_T0001F001L01A01Z01C02.tif", size=(64, 64)),
    ]
    r = client.post("/predict", files=files, data={"root_path": "/tmp"})
    assert r.status_code == 400


def test_predict_multi_channel_logs_per_channel_metadata(client):
    """Multi-channel upload should log image_metadata with correct channel numbers."""
    fake_db = FakeDBLogger()
    api_main.predictor = FakePredictor(db_logger=fake_db)
    files = _multi_channel_tif_files(n_channels=3)
    r = client.post("/predict", files=files, data={"root_path": "/data/images"})
    assert r.status_code == 200
    assert len(fake_db.image_metadata_calls) == 1
    metadata = fake_db.image_metadata_calls[0]
    assert len(metadata) == 3, "Should log one image_metadata row per channel"
    # Each tuple: (plate, well, field, channel, root_path, filename, shape_x, shape_y)
    channels = [row[3] for row in metadata]
    assert channels == [1, 2, 3]
    plates = {row[0] for row in metadata}
    assert plates == {"PLATE1"}
    wells = {row[1] for row in metadata}
    assert wells == {"A01"}
    root_paths = {row[4] for row in metadata}
    assert root_paths == {"/data/images"}


# ── Database connection tests ───────────────────────────────────────────────


def test_db_endpoint_503_without_db(client):
    """GET /db returns 503 when no database connection is configured."""
    api_main.db_logger = None
    r = client.get("/db")
    assert r.status_code == 503


def test_db_endpoint_with_db(client):
    """GET /db returns connection info when a db_logger is set."""
    api_main.db_logger = FakeDBLogger()
    r = client.get("/db")
    assert r.status_code == 200
    body = r.json()
    assert body["connected"] is True
    assert "uri" in body


def test_health_reports_database_connected(client):
    """GET /health includes database_connected=True when db_logger is set."""
    api_main.predictor = None
    api_main.db_logger = FakeDBLogger()
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["database_connected"] is True


def test_health_reports_database_disconnected(client):
    """GET /health includes database_connected=False when db_logger is None."""
    api_main.predictor = None
    api_main.db_logger = None
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["database_connected"] is False
