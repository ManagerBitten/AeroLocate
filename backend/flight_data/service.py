from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.core.config import Settings
from backend.core.geo import LocalFrame
from backend.flight_data.normalization.fusion import FusedFlight, fuse_records
from backend.flight_data.providers.adsbexchange import ADSBExchangeProvider
from backend.flight_data.providers.base import (
    FlightDataProvider,
    HttpSession,
    ProviderCapabilityError,
    ProviderError,
    ResponseCache,
)
from backend.flight_data.providers.flightradar24 import Flightradar24Provider
from backend.flight_data.providers.local_dataset import LocalFlightDatasetProvider
from backend.flight_data.providers.opensky import OpenSkyProvider
from backend.flight_data.reconstruction.trajectory import HermiteTrajectory, reconstruct
from backend.flight_data.types import (
    BoundingBox,
    DataQualityReport,
    FlightRecord,
    ProviderCapability,
    TimeWindow,
)

MAX_TRACK_LOOKUPS = 60
METERS_PER_DEGREE_LATITUDE = 111_320.0


@dataclass
class CandidateFlight:
    key: str
    fused: FusedFlight
    trajectory: HermiteTrajectory


@dataclass
class RetrievalOutcome:
    candidates: list[CandidateFlight] = field(default_factory=list)
    quality: list[DataQualityReport] = field(default_factory=list)
    provider_errors: dict[str, str] = field(default_factory=dict)
    area_search_available: bool = False
    requests_estimated: int = 0


def build_providers(settings: Settings, selection: list[str] | None = None) -> list[FlightDataProvider]:
    cache_dir = settings.data_dir / "flight_cache"
    ttl = settings.flight_cache_ttl_hours * 3600.0

    def session(name: str) -> HttpSession:
        return HttpSession(name, cache=ResponseCache(cache_dir / name, ttl))

    available: dict[str, FlightDataProvider] = {
        "flightradar24": Flightradar24Provider(
            settings.fr24_api_key.get_secret_value(),
            settings.fr24_base_url,
            session("flightradar24"),
            settings.fr24_snapshot_step_s,
        ),
        "adsbexchange": ADSBExchangeProvider(
            settings.adsbx_api_key.get_secret_value(), settings.adsbx_base_url, session("adsbexchange")
        ),
        "opensky": OpenSkyProvider(
            settings.opensky_client_id,
            settings.opensky_client_secret.get_secret_value(),
            settings.opensky_base_url,
            settings.opensky_token_url,
            session("opensky"),
        ),
        "local_dataset": LocalFlightDatasetProvider(settings.local_flight_dataset_dir),
    }
    names = selection if selection else list(available)
    return [available[name] for name in names if name in available]


def pad_box(box: BoundingBox, margin_m: float) -> BoundingBox:
    latitude_pad = margin_m / METERS_PER_DEGREE_LATITUDE
    cosine = max(np.cos(np.radians(box.center[0])), 0.05)
    longitude_pad = latitude_pad / cosine
    return BoundingBox(
        max(box.south - latitude_pad, -90.0),
        max(box.west - longitude_pad, -180.0),
        min(box.north + latitude_pad, 90.0),
        min(box.east + longitude_pad, 180.0),
    )


def _median_interval(records: list[FlightRecord]) -> float | None:
    intervals: list[float] = []
    for record in records:
        times = sorted(item.timestamp for item in record.observations)
        intervals.extend(np.diff(times).tolist())
    return float(np.median(intervals)) if intervals else None


def retrieve_candidates(
    providers: list[FlightDataProvider],
    box: BoundingBox,
    window: TimeWindow,
    frame: LocalFrame,
) -> RetrievalOutcome:
    outcome = RetrievalOutcome()
    collected: list[FlightRecord] = []
    for provider in providers:
        if not provider.is_configured() or not provider.supports(ProviderCapability.HISTORIC_AREA_SEARCH):
            continue
        outcome.requests_estimated += provider.estimate_requests(box, window)
        try:
            records = provider.search_area(box, window)
        except (ProviderError, ProviderCapabilityError) as error:
            outcome.provider_errors[provider.name] = str(error)
            continue
        outcome.area_search_available = True
        collected.extend(records)
        truncated = getattr(provider, "truncated_snapshots", 0) > 0
        outcome.quality.append(
            DataQualityReport(
                provider=provider.name,
                requested_window=window,
                records=len(records),
                observations=sum(len(record.observations) for record in records),
                median_report_interval_s=_median_interval(records),
                truncated=truncated,
            )
        )
    tracks = _refine_tracks(providers, collected, window, outcome)
    fused = fuse_records(collected + tracks)
    for index, flight in enumerate(fused):
        trajectory = reconstruct(flight.observations, frame)
        if trajectory is None:
            continue
        key = flight.identity.icao24 or flight.identity.callsign or f"unidentified-{index}"
        outcome.candidates.append(CandidateFlight(key, flight, trajectory))
    return outcome


def _refine_tracks(
    providers: list[FlightDataProvider],
    collected: list[FlightRecord],
    window: TimeWindow,
    outcome: RetrievalOutcome,
) -> list[FlightRecord]:
    refined: list[FlightRecord] = []
    seen: set[tuple[str, str]] = set()
    for record in collected[:MAX_TRACK_LOOKUPS]:
        for provider in providers:
            if not provider.is_configured() or not provider.supports(ProviderCapability.HISTORIC_TRACK_BY_ID):
                continue
            if provider.name == "local_dataset":
                continue
            marker = (provider.name, record.identity.icao24 or record.identity.provider_flight_id or "")
            if marker in seen:
                continue
            seen.add(marker)
            try:
                track = provider.get_track(record.identity, window)
            except ProviderError as error:
                outcome.provider_errors.setdefault(provider.name, str(error))
                continue
            if track is not None:
                refined.append(track)
    return refined
