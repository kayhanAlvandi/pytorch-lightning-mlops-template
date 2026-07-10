"""API tests using FastAPI's TestClient.

The real model loading (MLflow) is bypassed:
  - startup is forced into the "no model source" branch, so no network / MLflow
  - a FakePredictor is injected for the success-path tests
"""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main


class FakePredictor:
    """Minimal stand-in for TilePredictor with the attributes the API reads."""

    def __init__(self):
        self.crop_size = 32
        self.stride = 32
        self.in_channels = 3
        self.class_names = ["ClassA", "ClassB"]
        self.device = "cpu"
        self.model_info = {
            "source": "fake",
            "model_class": "CNNClassifier",
            "backbone": None,
            "run_id": "deadbeef",
            "num_classes": 2,
            "crop_size": 32,
            "in_channels": 3,
        }

    def predict(self, image):
        return {
            "prediction": "ClassA",
            "num_tiles": 1,
            "tiles": [{"row": 0, "col": 0, "prediction": "ClassA"}],
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
    r = client.post("/predict", files=_npy_upload())
    assert r.status_code == 503


def test_predict_success_with_mock(client):
    api_main.predictor = FakePredictor()
    r = client.post("/predict", files=_npy_upload())
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "ClassA"
    assert body["num_tiles"] == 1


def test_predict_rejects_bad_npy_shape(client):
    api_main.predictor = FakePredictor()
    # 2D array is invalid: endpoint expects (C, H, W)
    r = client.post("/predict", files=_npy_upload(shape=(32, 32)))
    assert r.status_code == 400


def test_model_info_with_mock(client):
    api_main.predictor = FakePredictor()
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["num_classes"] == 2
    assert body["class_names"] == ["ClassA", "ClassB"]
