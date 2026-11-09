from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from backend.flight_data.normalization.units import (
    feet_to_m,
    fpm_to_mps,
    knots_to_mps,
    normalize_callsign,
    normalize_icao24,
)
from backend.flight_data.providers.base import (
    FlightDataProvider,
    HttpSession,
    ProviderCapabilityError,
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

PROVIDER_NAME = "adsbexchange"
FLAG_STALE = 1
FLAG_GEOMETRIC_VERTICAL_RATE = 4
FLAG_GEOMETRIC_ALTITUDE = 8


def parse_trace(payload: dict[str, Any], provider: str = PROVIDER_NAME) -> FlightRecord:
    base = float(payload["timestamp"])
    observations: list[PositionObservation] = []
    callsign: str | None = None
    for row in payload.get("trace", []):
        offset, latitude, longitude, altitude, ground_speed, track, flags = row[:7]
        flags = int(flags or 0)
        vertical_rate = row[7] if len(row) > 7 else None
        details = row[8] if len(row) > 8 else None
        source = row[9] if len(row) > 9 else None
        geometric = row[10] if len(row) > 10 else None
        if isinstance(details, dict) and details.get("flight"):
            callsign = normalize_callsign(details["flight"])
        on_ground = altitude == "ground"
        if geometric is not None:
            altitude_m, kind = feet_to_m(geometric), AltitudeKind.GEOMETRIC
        elif on_ground or altitude is None:
            altitude_m, kind = (0.0 if on_ground else None), AltitudeKind.UNKNOWN
        else:
            kind = AltitudeKind.GEOMETRIC if flags & FLAG_GEOMETRIC_ALTITUDE else AltitudeKind.BAROMETRIC
            altitude_m = feet_to_m(altitude)
        source_label = str(source) if source else "ADSB"
        multilaterated = "mlat" in source_label.lower()
        observations.append(
            PositionObservation(
                timestamp=base + float(offset),
                latitude=float(latitude),
                longitude=float(longitude),
                altitude_m=altitude_m,
                altitude_kind=kind,
                ground_speed_mps=knots_to_mps(ground_speed),
                track_deg=None if track is None else float(track),
                vertical_rate_mps=fpm_to_mps(vertical_rate),
                on_ground=on_ground,
                provider=provider,
                source_type=source_label,
                origin=ValueOrigin.MEASURED if multilaterated else ValueOrigin.REPORTED,
                stale=bool(flags & FLAG_STALE),
            )
        )
    identity = AircraftIdentity(
        icao24=normalize_icao24(payload.get("icao")),
        callsign=callsign,
        registration=payload.get("r") or None,
        aircraft_type=payload.get("t") or None,
    )
    return FlightRecord(identity, observations, [provider])


class ADSBExchangeProvider(FlightDataProvider):
    name = PROVIDER_NAME
    capabilities = (ProviderCapability.HISTORIC_TRACK_BY_ID, ProviderCapability.LIVE_AREA_SEARCH)

    def __init__(self, api_key: str, base_url: str, session: HttpSession | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or HttpSession(PROVIDER_NAME)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderNotConfigured("AEROLOCATE_ADSBX_API_KEY is not set")
        return {"x-api-key": self.api_key, "Accept-Encoding": "gzip", "Accept": "application/json"}

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def check_availability(self) -> ProviderStatus:
        if not self.api_key:
            return ProviderStatus(self.name, False, False, self.capabilities, "API key not configured")
        try:
            self.session.get_json(f"{self.base_url}/total-aircraft", headers=self._headers())
        except ProviderError as error:
            return ProviderStatus(self.name, True, False, self.capabilities, str(error))
        detail = "reachable; historical traces require the matching plan and are queried per aircraft"
        return ProviderStatus(self.name, True, True, self.capabilities, detail)

    def estimate_requests(self, box: BoundingBox, window: TimeWindow) -> int:
        return 0

    def search_area(self, box: BoundingBox, window: TimeWindow) -> list[FlightRecord]:
        raise ProviderCapabilityError(
            "ADS-B Exchange exposes historical traces per aircraft; a historical area search needs candidates from another provider"
        )

    def get_track(self, identity: AircraftIdentity, window: TimeWindow) -> FlightRecord | None:
        hex_code = normalize_icao24(identity.icao24)
        if not hex_code:
            return None
        merged: FlightRecord | None = None
        for day in _utc_days(window):
            url = (
                f"{self.base_url}/traces-hist/{day.year:04d}/{day.month:02d}/{day.day:02d}"
                f"/traces/{hex_code[-2:]}/trace_full_{hex_code}.json"
            )
            payload = self.session.get_json(url, headers=self._headers())
            if not payload:
                continue
            record = parse_trace(payload)
            if merged is None:
                merged = record
            else:
                merged.observations.extend(record.observations)
        if merged is None:
            return None
        merged.identity = AircraftIdentity(
            icao24=merged.identity.icao24,
            callsign=merged.identity.callsign or identity.callsign,
            flight_number=identity.flight_number,
            registration=merged.identity.registration or identity.registration,
            aircraft_type=merged.identity.aircraft_type or identity.aircraft_type,
        )
        return merged


def _utc_days(window: TimeWindow) -> list[datetime]:
    current = datetime.fromtimestamp(window.start, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    last = datetime.fromtimestamp(window.end, tz=timezone.utc)
    days = []
    while current <= last:
        days.append(current)
        current += timedelta(days=1)
    return days
