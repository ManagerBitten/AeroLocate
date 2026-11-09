from __future__ import annotations

from pathlib import Path

from celery import Celery

from backend.core.config import get_settings
from backend.workers.pipeline import run_analysis
from backend.workers.retention import purge_expired_analyses

settings = get_settings()
celery_app = Celery("aerolocate", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=settings.analysis_timeout_s,
    task_time_limit=settings.analysis_timeout_s + 60,
    worker_max_tasks_per_child=20,
    worker_max_memory_per_child=6_000_000,
    beat_schedule={"purge-expired": {"task": "aerolocate.purge_expired", "schedule": 3600.0}},
)


@celery_app.task(
    name="aerolocate.analyze",
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    max_retries=3,
)
def analyze_task(analysis_id: str, source_path: str) -> None:
    run_analysis(analysis_id, settings, Path(source_path))


@celery_app.task(name="aerolocate.purge_expired")
def purge_expired_task() -> int:
    return purge_expired_analyses(settings)
