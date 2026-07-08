# Tests

Automated test suite for the image classifier. Tests are organized by type and
selected with pytest markers, so CI and local runs can target subsets.

## Layout

- `tests/unit/` — fast, isolated tests (transforms, dataset, model, datamodule, config).
- `tests/integration/` — end-to-end tests (training smoke, checkpoint round-trip, MLflow, GPU).
- `tests/api/` — FastAPI endpoint tests via `TestClient`.
- `tests/diagnostics/` — old exploratory scripts, **excluded** from collection (see `norecursedirs` in `pytest.ini`).
- `tests/conftest.py` — shared fixtures (synthetic images, `tiled_datamodule`, etc.).

## Markers

Registered in `pytest.ini`:

- `slow` — heavier tests (training, checkpoints, MLflow).
- `integration` — end-to-end tests.
- `gpu` — require a CUDA device (auto-skipped when unavailable).

## Running

Requires `pytest`, `pytest-cov`, and `httpx` (for the API `TestClient`):

```bash
pip install pytest pytest-cov httpx
```

Common invocations (also available as `make` targets):

| Command | Make target | What runs |
| --- | --- | --- |
| `pytest -m "not slow" -q` | `make test-fast` | unit + API (the CI push suite) |
| `pytest -m "integration and not gpu" -q` | `make test-integration` | integration on CPU |
| `pytest -m "not gpu" -q` | `make test-all` | everything except GPU |
| `pytest -m "gpu" -q` | `make gpu-test` | GPU tests (CUDA machine) |

## CI vs. GPU

`.github/workflows/ci.yml` runs on GitHub-hosted (CPU) runners:

- **push / pull_request** → `pytest -m "not slow"` (fast suite).
- **Manual "Run workflow"** → choose `fast`, `integration`, or `all` (all CPU-only; GPU excluded).

GitHub-hosted runners have **no GPU**, and self-hosted runners are unsafe on a
public repo, so **GPU tests are a local pre-merge check**. Run them on your
CUDA machine before merging:

```bash
pytest -m "gpu" -q   # or: make gpu-test
```

On a machine without CUDA these tests are skipped automatically.
