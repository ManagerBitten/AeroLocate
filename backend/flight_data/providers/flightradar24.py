from __future__ import annotations

from typing import Any

from backend.flight_data.normalization.units import (
    feet_to_m,
    fpm_to_mps,
    knots_to_mps,
    normalize_callsign,
    normalize_icao24,
    parse_utc,
)
from backend.flight_data.providers.base import (
    FlightDataProvider,
    HttpSession,
    ProviderError,
    ProviderNotConfigured,
)
from backend.flight_data.types import (
    AircraftIdentity,
    AltitudeKind,
    BoundingBox,
    FlightRecord,
    PositionObservation,
    ProviderCapability,
    ProviderStatus,
    TimeWindow,
    ValueOrigin,
)

PROVIDER_NAME = "flightradar24"
SNAPSHOT_LIMIT = 1000
ORIGIN_BY_SOURCE = {
    "ADSB": ValueOrigin.REPORTED,
    "MLAT": ValueOrigin.MEASURED,
    "ESTIMATED": ValueOrigin.INFERRED,
}


class Flightradar24Provider(FlightDataProvider):
    name = PROVIDER_NAME
    capabilities = (
        ProviderCapability.HISTORIC_AREA_SEARCH,
        ProviderCapability.HISTORIC_TRACK_BY_ID,
        ProviderCapability.AIRCRAFT_METADATA,
    )

    def __init__(
        self,
        api_key: str,
        base_url: str,
        session: HttpSession | None = None,
        snapshot_step_s: int = 60,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or HttpSession(PROVIDER_NAME)
        self.snapshot_step_s = snapshot_step_s
        self.truncated_snapshots = 0

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderNotConfigured("AEROLOCATE_FR24_API_KEY is not set")
        return {
            "Accept": "application/json",
            "Accept-Version": "v1",
            "Authorization": f"Bearer {self.api_key}",
        }

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def check_availability(self) -> ProviderStatus:
        if not self.api_key:
            return ProviderStatus(self.name, False, False, self.capabilities, "API key not configured")
        try:
            payload = self.session.get_json(f"{self.base_url}/usage", headers=self._headers(), params={"period": "24h"})
        except ProviderError as error:
            return ProviderStatus(self.name, True, False, self.capabilities, str(error))
        detail = "reachable; historic depth and credits depend on the subscription plan"
        return ProviderStatus(self.name, True, payload is not None, self.capabilities, detail)

    def estimate_requests(self, box: BoundingBox, window: TimeWindow) -> int:
        return int(window.duration // self.snapshot_step_s) + 2

    def search_area(self, box: BoundingBox, window: TimeWindow) -> list[FlightRecord]:
        bounds = f"{box.north:.3f},{box.south:.3f},{box.west:.3f},{box.east:.3f}"
        records: dict[str, FlightRecord] = {}
        timestamp = int(window.start)
        last = int(window.end) + self.snapshot_step_s
        while timestamp <= last:
            payload = self.session.get_json(
                f"{self.base_url}/historic/flight-positions/full",
                params={"timestamp": timestamp, "bounds": bounds, "limit": SNAPSHOT_LIMIT},
                headers=self._headers(),
            )
            rows = (payload or {}).get("data", [])
            if len(rows) >= SNAPSHOT_LIMIT:
                self.truncated_snapshots += 1
            for row in rows:
                key = row.get("fr24_id") or row.get("hex")
                if not key:
                    continue
                record = records.setdefault(key, FlightRecord(_identity(row), [], [self.name]))
                record.observations.append(_observation(row))
            timestamp += self.snapshot_step_s
        return list(records.values())

    def get_track(self, identity: AircraftIdentity, window: TimeWindow) -> FlightRecord | None:
        if not identity.provider_flight_id:
            return None
        payload = self.session.get_json(
            f"{self.base_url}/flight-tracks",
            params={"flight_id": identity.provider_flight_id},
            headers=self._headers(),
        )
        if not payload:
            return None
        entries = payload if isinstance(payload, list) else payload.get("data", [])
        points: list[PositionObservation] = []
        for entry in entries:
            points.extend(_observation(point) for point in entry.get("tracks", []))
        if not points:
            return None
        return FlightRecord(identity, points, [self.name])


def _identity(row: dict[str, Any]) -> AircraftIdentity:
    return AircraftIdentity(
        icao24=normalize_icao24(row.get("hex")),
        callsign=normalize_callsign(row.get("callsign")),
        flight_number=normalize_callsign(row.get("flight")),
        registration=row.get("reg") or None,
        aircraft_type=row.get("type") or None,
        provider_flight_id=row.get("fr24_id") or None,
    )


def _observation(row: dict[str, Any]) -> PositionObservation:
    source = str(row.get("source") or "UNKNOWN").upper()
    altitude = feet_to_m(row.get("alt"))
    return PositionObservation(
        timestamp=parse_utc(row["timestamp"]),
        latitude=float(row["lat"]),
        longitude=float(row["lon"]),
        altitude_m=altitude,
        altitude_kind=AltitudeKind.BAROMETRIC,
        ground_speed_mps=knots_to_mps(row.get("gspeed")),
        track_deg=None if row.get("track") is None else float(row["track"]),
        vertical_rate_mps=fpm_to_mps(row.get("vspeed")),
        on_ground=altitude is not None and altitude <= 0.0,
        provider=PROVIDER_NAME,
        source_type=source,
        origin=ORIGIN_BY_SOURCE.get(source, ValueOrigin.REPORTED),
    )
