from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.core.config import Settings, get_settings
from backend.workers.pipeline import run_analysis

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aerolocate-job")


def enqueue_analysis(analysis_id: str, source_path: Path, settings: Settings | None = None) -> None:
    active = settings or get_settings()
    if active.job_backend == "celery":
        from backend.workers.celery_app import analyze_task

        analyze_task.delay(analysis_id, str(source_path))
        return
    _executor.submit(run_analysis, analysis_id, active, source_path)
