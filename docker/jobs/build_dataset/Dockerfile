FROM python:3.11-slim

LABEL maintainer="Kayhan Alvandi <keyhanalvandi96@gmail.com>"
LABEL description="Lightweight dataset-building job (scripts/build_dataset.py) -- no torch/timm/mlflow"

WORKDIR /workspace

# Deliberately separate from the training image: build_dataset.py and the
# src modules it uses (sample_selection.py, dataset_versioning.py) are
# torch-free, so running them doesn't need the ~12GB CUDA training image.
COPY requirements/build_dataset_req.txt .
RUN pip install --no-cache-dir -r build_dataset_req.txt

# Code baked in for registry/K8s use; the dev compose still bind-mounts these
# same subdirectories for local iteration without a rebuild.
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY src/ ./src/
COPY utils/ ./utils/

# install external tools lib at container startup (label resolution via
# tools.loading.getCategories, same pattern as the monitoring/training images)
RUN printf '#!/bin/bash\npip install -e /workspace/tools 2>/dev/null\nexec "$@"\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["python", "scripts/build_dataset.py"]
