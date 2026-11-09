from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from sqlalchemy import Engine, create_engine, event, func, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.core.config import get_settings
from backend.models.entities import Base


SPATIAL_INDEXES = (
    ("analyses", "search_region"),
    ("analyses", "estimate"),
    ("flight_candidates", "track"),
    ("candidate_regions", "shape"),
)


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url
    if url.startswith("sqlite"):
        options: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url or url == "sqlite://":
            options["poolclass"] = StaticPool
        engine = create_engine(url, **options)

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(connection: Any, _: Any) -> None:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def init_database() -> None:
    engine = get_engine()
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            for table, column in SPATIAL_INDEXES:
                connection.execute(
                    text(f"CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table} USING GIST ({column})")
                )


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    with session_scope() as session:
        yield session


def geometry_value(session: Session, wkt: str | None) -> Any:
    if wkt is None:
        return None
    if session.get_bind().dialect.name == "postgresql":
        return func.ST_GeomFromText(wkt, 4326)
    return wkt
