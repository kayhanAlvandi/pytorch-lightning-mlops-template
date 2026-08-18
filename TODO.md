# Project TODO

## Serving: model/API dependency version mismatch (unresolved)

**Problem:**
Models are logged via `mlflow.pytorch.log_model(..., code_paths=["src"])`
(<ref_snippet file="/d:/personal_project/image_classifier/src/callbacks.py" lines="114-122" />),
which bundles the model class source code with the artifact, but does **not**
pin the runtime dependency versions used during training (e.g. `timm`,
`torch`). The serving API (`api/predictor.py`) loads models in-process via
`mlflow.pytorch.load_model()` inside a single container built from
`requirements/api_req.txt`
(<ref_file file="/d:/personal_project/image_classifier/docker/services/api/Dockerfile" />).

If different developers train models with different dependency versions
(e.g. `timm==1.0.26` vs `timm==1.0.28`), one fixed API container cannot
correctly serve all of them. We hit this already and worked around it by
pinning `timm==1.0.26` in `api_req.txt`, but that's not a real fix — it just
makes the container match whichever model was trained most recently.

**Options discussed (see chat history for full breakdown, brainstormed on 2026-08-11):**
1. Log `pip_requirements` / `extra_pip_requirements` with the model at log-model
   time, and switch serving to `mlflow.pyfunc.load_model()` (supports env
   isolation) — requires refactoring how `predictor.py` calls the model.
2. Build a dedicated Docker image per model/run, using the model's logged
   `conda.yaml` to generate exact requirements — no `predictor.py` changes,
   but adds a build/deploy pipeline step.
3. Dynamically `pip install` the model's exact requirements at API container
   startup (before loading the model) — single container, adapts per model,
   but slow startup and can't serve two conflicting-version models at once.
4. Pin exact versions across `training_req.txt` and `api_req.txt` and enforce
   everyone uses the same environment — zero code changes, simplest, but
   doesn't scale with multiple developers/experiments.

**Recommendation (not yet implemented):** Do (1) regardless — always log
`pip_requirements` with the model, it's cheap and gives us metadata/options
later. Combine with (3) short-term for flexibility, revisit (2) if startup
latency becomes a problem or models need to run concurrently with
conflicting dependency versions.

**Status:** Not started. Revisit before onboarding more developers to
training or before relying on this for production serving.
