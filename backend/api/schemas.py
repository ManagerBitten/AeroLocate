from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AuthorizationBasis(str, Enum):
    OWN_RECORDING = "own_recording"
    CONSENT_OF_RECORDER = "consent_of_recorder"
    ORGANIZATIONAL_AUTHORITY = "organizational_authority"


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=320, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=12, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RegionInput(BaseModel):
    south: float = Field(ge=-90, le=90)
    west: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def ordered(self) -> "RegionInput":
        if self.south >= self.north or self.west >= self.east:
            raise ValueError("region bounds are not ordered")
        return self


class AnalysisRequest(BaseModel):
    region: RegionInput
    start_earliest_utc: float
    start_latest_utc: float
    authorization_basis: AuthorizationBasis
    authorization_note: str = Field(default="", max_length=2000)
    providers: list[str] = Field(default_factory=list, max_length=8)
    preprocessing: str = Field(default="none", pattern="^(none|wind|broadband)$")
    temperature_c: float | None = Field(default=None, ge=-60, le=60)
    wind_east_mps: float | None = Field(default=None, ge=-60, le=60)
    wind_north_mps: float | None = Field(default=None, ge=-60, le=60)
    ground_elevation_m: float | None = Field(default=None, ge=-450, le=9000)

    @model_validator(mode="after")
    def ordered(self) -> "AnalysisRequest":
        if self.start_latest_utc < self.start_earliest_utc:
            raise ValueError("the time window ends before it starts")
        return self


class WorkloadRequest(BaseModel):
    region: RegionInput
    start_earliest_utc: float
    start_latest_utc: float
    duration_s: float = Field(default=300.0, gt=0)
    providers: list[str] = Field(default_factory=list)


class WorkloadEstimate(BaseModel):
    region_area_km2: float
    time_window_s: float
    provider_requests: dict[str, int]
    initial_search_tiles: int
    feasible: bool
    advice: list[str]


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    state: str
    stage: str
    progress: float
    error: str | None
    localization_status: str | None
    created_at: datetime
    finished_at: datetime | None
    expires_at: datetime


class AnalysisDetail(AnalysisSummary):
    request: dict[str, Any]
    media: dict[str, Any] | None
    acoustic: dict[str, Any] | None
    result: dict[str, Any] | None
    authorization_basis: str
