from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from backend.core.config import Settings
from backend.flight_data.providers.base import ResponseCache
from backend.models.database import session_scope
from backend.models.entities import Analysis, AuditLog
from backend.workers.pipeline import delete_artifacts


def purge_expired_analyses(settings: Settings) -> int:
    now = datetime.now(timezone.utc)
    removed = 0
    with session_scope() as session:
        expired = session.scalars(select(Analysis).where(Analysis.expires_at < now)).all()
        for analysis in expired:
            if analysis.state in ("queued", "running"):
                continue
            delete_artifacts(analysis.id, settings)
            session.add(AuditLog(action="analysis.expired", target=analysis.id, detail={}))
            session.delete(analysis)
            removed += 1
    ttl = timedelta(hours=settings.flight_cache_ttl_hours).total_seconds()
    cache_root = settings.data_dir / "flight_cache"
    if cache_root.exists():
        for directory in cache_root.iterdir():
            ResponseCache(directory, ttl).purge_expired()
    return removed
