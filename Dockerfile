# FaceVerification 3.0 — backend image (CPU inference).
# Build from the repository root:  docker build -t faceverification-api .
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/app/models DEEPFACE_HOME=/data

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements_backend.txt backend/requirements_mediapipe.txt backend/
RUN pip install -r backend/requirements_backend.txt \
 && pip install --no-deps -r backend/requirements_mediapipe.txt

COPY backend/ backend/
COPY scripts/fetch_models.py scripts/
RUN python scripts/fetch_models.py && rm -rf frontend

RUN useradd --create-home --uid 10001 app && mkdir -p /data /app/backups && chown -R app:app /data /app/backups
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fsS http://127.0.0.1:8000/api/v1/health/ready || exit 1

# One worker per container: the SSE event bus and the inference pool are per process.
# --proxy-headers trusts X-Forwarded-For only from FORWARDED_ALLOW_IPS (the reverse proxy).
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers --forwarded-allow-ips=${FORWARDED_ALLOW_IPS:-127.0.0.1}"]
