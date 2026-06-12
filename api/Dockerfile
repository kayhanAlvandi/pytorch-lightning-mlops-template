FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

LABEL maintainer="Kayhan Alvandi <kayhan.alvandi@tum.de>"
LABEL description="FastAPI service for image classification with PyTorch Lightning"


WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
