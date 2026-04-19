FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── API target ─────────────────────────────────────────────────────────────────
FROM base AS api
EXPOSE 8004
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8004"]

# ── Celery worker target (runs SFT / reward model / DPO training jobs) ─────────
FROM base AS worker
CMD ["celery", "-A", "worker.celery_app", "worker", \
     "--loglevel=info", "--concurrency=1", "--queues=training,default"]

# ── Inference server target ────────────────────────────────────────────────────
FROM base AS serving
EXPOSE 8080
CMD ["python", "-m", "serving.inference"]
