from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ValueOrigin(str, Enum):
    MEASURED = "measured"
    REPORTED = "reported"
    INTERPOLATED = "interpolated"
    INFERRED = "inferred"


class AltitudeKind(str, Enum):
    BAROMETRIC = "barometric"
    GEOMETRIC = "geometric"
    UNKNOWN = "unknown"


class ProviderCapability(str, Enum):
    HISTORIC_AREA_SEARCH = "historic_area_search"
    HISTORIC_TRACK_BY_ID = "historic_track_by_id"
    LIVE_AREA_SEARCH = "live_area_search"
    AIRCRAFT_METADATA = "aircraft_metadata"


@dataclass(frozen=True)
class BoundingBox:
    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        if not (-90 <= self.south < self.north <= 90):
            raise ValueError("invalid latitude bounds")
        if not (-180 <= self.west < self.east <= 180):
            raise ValueError("invalid longitude bounds; antimeridian-crossing boxes must be split")

    @property
    def center(self) -> tuple[float, float]:
        return (self.south + self.north) / 2.0, (self.west + self.east) / 2.0


@dataclass(frozen=True)
class TimeWindow:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("time window ends before it starts")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class AircraftIdentity:
    icao24: str | None = None
    callsign: str | None = None
    flight_number: str | None = None
    registration: str | None = None
    aircraft_type: str | None = None
    provider_flight_id: str | None = None


@dataclass(frozen=True)
class PositionObservation:
    timestamp: float
    latitude: float
    longitude: float
    altitude_m: float | None
    altitude_kind: AltitudeKind
    ground_speed_mps: float | None
    track_deg: float | None
    vertical_rate_mps: float | None
    on_ground: bool
    provider: str
    source_type: str
    origin: ValueOrigin = ValueOrigin.REPORTED
    stale: bool = False
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["altitude_kind"] = self.altitude_kind.value
        data["origin"] = self.origin.value
        return data


@dataclass
class FlightRecord:
    identity: AircraftIdentity
    observations: list[PositionObservation]
    providers: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    configured: bool
    available: bool
    capabilities: tuple[ProviderCapability, ...]
    detail: str
    credits_remaining: float | None = None


@dataclass(frozen=True)
class DataQualityReport:
    provider: str
    requested_window: TimeWindow
    records: int
    observations: int
    median_report_interval_s: float | None
    truncated: bool
    notes: tuple[str, ...] = ()
