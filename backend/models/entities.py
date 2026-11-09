from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Dialect,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.types import UserDefinedType
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class PostgisGeometry(UserDefinedType[Any]):
    cache_ok = True

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def get_col_spec(self, **_: Any) -> str:
        return f"geometry({self.kind},4326)"


class SpatialColumn(TypeDecorator[Any]):
    impl = Text
    cache_ok = True

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PostgisGeometry(self.kind))
        return dialect.type_descriptor(Text())


def spatial(kind: str) -> Any:
    return SpatialColumn(kind)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    extension: Mapped[str] = mapped_column(String(8))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    authorization_basis: Mapped[str] = mapped_column(String(32))
    authorization_note: Mapped[str] = mapped_column(Text, default="")
    request: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    media: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    acoustic: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    localization_status: Mapped[str | None] = mapped_column(String(24), nullable=True, index=True)
    search_region = mapped_column(spatial("POLYGON"), nullable=True, deferred=True)
    estimate = mapped_column(spatial("POINT"), nullable=True, deferred=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    doppler_points: Mapped[list["DopplerObservation"]] = relationship(cascade="all, delete-orphan")
    flights: Mapped[list["FlightCandidateRow"]] = relationship(cascade="all, delete-orphan")
    regions: Mapped[list["RegionRow"]] = relationship(cascade="all, delete-orphan")


class DopplerObservation(Base):
    __tablename__ = "doppler_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    event_id: Mapped[str] = mapped_column(String(8))
    time_s: Mapped[float] = mapped_column(Float)
    frequency_hz: Mapped[float] = mapped_column(Float)
    sigma_hz: Mapped[float] = mapped_column(Float)


class FlightCandidateRow(Base):
    __tablename__ = "flight_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    flight_key: Mapped[str] = mapped_column(String(32))
    identity: Mapped[dict[str, Any]] = mapped_column(JSON)
    providers: Mapped[list[str]] = mapped_column(JSON)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON)
    track = mapped_column(spatial("LINESTRINGZ"), nullable=True, deferred=True)
    observations: Mapped[list["ProviderObservation"]] = relationship(cascade="all, delete-orphan")


class ProviderObservation(Base):
    __tablename__ = "provider_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("flight_candidates.id", ondelete="CASCADE"), index=True)
    timestamp: Mapped[float] = mapped_column(Float)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    altitude_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    altitude_kind: Mapped[str] = mapped_column(String(16))
    ground_speed_mps: Mapped[float | None] = mapped_column(Float, nullable=True)
    track_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    vertical_rate_mps: Mapped[float | None] = mapped_column(Float, nullable=True)
    provider: Mapped[str] = mapped_column(String(32))
    source_type: Mapped[str] = mapped_column(String(32))
    origin: Mapped[str] = mapped_column(String(16))


class RegionRow(Base):
    __tablename__ = "candidate_regions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    area_km2: Mapped[float] = mapped_column(Float)
    shape = mapped_column(spatial("MULTIPOLYGON"), nullable=True, deferred=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(64))
    target: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    client: Mapped[str | None] = mapped_column(String(64), nullable=True)


Index("ix_analysis_owner_created", Analysis.owner_id, Analysis.created_at)
