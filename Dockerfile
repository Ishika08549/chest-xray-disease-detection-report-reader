FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

COPY outputs/models/best_resnet50.pth ./outputs/models/best_resnet50.pth

COPY data/processed/image_report_dataset.csv \
     ./data/processed/image_report_dataset.csv

ENV MODEL_PATH=/app/outputs/models/best_resnet50.pth
ENV REPORT_DATASET_PATH=/app/data/processed/image_report_dataset.csv

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
