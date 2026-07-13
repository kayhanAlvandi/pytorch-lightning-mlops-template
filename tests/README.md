# Tests

Automated test suite for the image classifier. Tests are organized by type and
selected with pytest markers, so CI and local runs can target subsets.

## Layout

- `tests/unit/` — fast, isolated tests (transforms, dataset, model, datamodule, config).
- `tests/integration/` — end-to-end tests (training smoke, checkpoint round-trip, MLflow).
- `tests/api/` — FastAPI endpoint tests via `TestClient`.
- `tests/diagnostics/` — old exploratory scripts, **excluded** from collection (see `norecursedirs` in `pytest.ini`).
- `tests/conftest.py` — shared fixtures (synthetic images, `tiled_datamodule`, etc.).

## Markers

Registered in `pytest.ini`:

- `slow` — heavier tests.
- `gpu` — require a CUDA device (auto-skipped when unavailable).

## Running

Requires `pytest`, `pytest-cov`, and `httpx` (for the API `TestClient`):

```bash
pip install pytest pytest-cov httpx
```

Common invocations (also available as `make` targets from `tests/`):

| Command | Make target | What runs |
| --- | --- | --- |
| `pytest tests/unit -q` | `make test-fast` | unit tests (no slow/GPU) |
| `pytest tests/integration -q` | `make test-integration` | integration on CPU |
| `pytest -m "not gpu" -q` | `make test-all` | everything except GPU |
| `pytest -m "gpu" -q` | `make gpu-test` | GPU tests (CUDA machine) |

## CI

Two separate workflows run on push / PR to `main`:

| Workflow | File | Scope |
| --- | --- | --- |
| **Tests-Training** | `ci_training.yml` | `src/`, `tests/unit/`, `tests/integration/`, `train.py` — ruff lint + unit tests (push); dispatch: training / integration / all |
| **Tests-Serving** | `ci_serving.yml` | `api/`, `tests/api/` — ruff lint + API tests |

Both use a **reusable composite action** (`.github/actions/setup-env/`) that accepts a `requirements_file` input, sets up Python 3.11, caches pip, and installs dependencies.

GitHub-hosted runners have **no GPU**, so **GPU tests are a local pre-merge check**.
Run them on your CUDA machine before merging:

```bash
pytest -m "gpu" -q   # or: make gpu-test
```

On a machine without CUDA these tests are skipped automatically.
