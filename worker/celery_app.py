"""
Celery application for the RLHF platform training worker.
"""

from __future__ import annotations

import os
from celery import Celery

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "rlhf_platform",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "worker.tasks.launch_training_run": {"queue": "training"},
    },
    task_time_limit=86400,     # 24 h — training can take a long time
    task_soft_time_limit=82800,
)
