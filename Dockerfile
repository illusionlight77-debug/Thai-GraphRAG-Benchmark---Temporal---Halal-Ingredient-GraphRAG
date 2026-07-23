FROM python:3.11-slim

WORKDIR /app

# curl is only here so the compose healthcheck can probe /health.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: embeddings run in the dedicated `embeddings` (TEI) container, not here —
# this image stays a thin HTTP client. The source CSVs are bind-mounted at runtime
# rather than copied in, which keeps the image ~150MB instead of ~300MB.
COPY thaigraphrag/ ./thaigraphrag/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY pyproject.toml ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8000

CMD ["uvicorn", "thaigraphrag.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
