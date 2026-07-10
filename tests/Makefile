# Test runner shortcuts. Run from the tests/ directory, e.g. `make test-fast`.
# Commands run from the repository root so pytest picks up ../pytest.ini.
ROOT := ..

.PHONY: test test-fast test-integration test-all gpu-test

# Default: fast CPU suite (matches CI push job)
test: test-fast

# Fast unit + API tests (no slow/integration/GPU)
test-fast:
	cd $(ROOT) && pytest -m "not slow" -q

# Integration tests on CPU (training smoke, checkpoint round-trip, MLflow)
test-integration:
	cd $(ROOT) && pytest tests/integration -q

# Everything except GPU (mirrors the CI "all" dispatch option)
test-all:
	cd $(ROOT) && pytest -m "not gpu" -q

# GPU tests — run this locally on a CUDA machine before merging.
# On a machine without CUDA these are skipped automatically.
gpu-test:
	cd $(ROOT) && pytest -m "gpu" -q
