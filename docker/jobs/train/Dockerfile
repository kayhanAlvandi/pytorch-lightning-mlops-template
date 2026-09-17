FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

LABEL maintainer="Kayhan Alvandi <keyhanalvandi96@gmail.com>"
LABEL description="Training environment for image classification with PyTorch Lightning"

WORKDIR /workspace

COPY requirements/training_req.txt .
RUN pip install --no-cache-dir --break-system-packages -r training_req.txt

# Code baked in for registry/K8s use (train.py only needs src/, configs/
# (hydra config_path="configs") and utils/ -- database/ and scripts/ are not
# imported anywhere on this path). The dev compose still bind-mounts the whole
# repo root over /workspace, so local edits take effect without a rebuild.
COPY src/ ./src/
COPY configs/ ./configs/
COPY utils/ ./utils/
COPY train.py .

# install tools in container startup
RUN echo '#!/bin/bash\npip install -e /workspace/tools 2>/dev/null\nexec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["python", "train.py"]
