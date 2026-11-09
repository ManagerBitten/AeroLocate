from __future__ import annotations

import time
from typing import Any, Callable

import httpx

from backend.flight_data.normalization.units import normalize_callsign, normalize_icao24
from backend.flight_data.providers.base import (
    FlightDataProvider,
    HttpSession,
    ProviderAccessDenied,
    ProviderCapabilityError,
    ProviderError,
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

PROVIDER_NAME = "opensky"
STATE_HISTORY_LIMIT_S = 3600.0
TRACK_HISTORY_LIMIT_S = 30 * 86400.0
STATE_STEP_S = 30
TOKEN_REFRESH_MARGIN_S = 60.0
POSITION_SOURCES = {0: "ADSB", 1: "ASTERIX", 2: "MLAT", 3: "FLARM"}


class OpenSkyProvider(FlightDataProvider):
    name = PROVIDER_NAME
    capabilities = (
        ProviderCapability.HISTORIC_AREA_SEARCH,
        ProviderCapability.HISTORIC_TRACK_BY_ID,
        ProviderCapability.LIVE_AREA_SEARCH,
    )

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        base_url: str,
        token_url: str,
        session: HttpSession | None = None,
        token_client: httpx.Client | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")
        self.token_url = token_url
        self.session = session or HttpSession(PROVIDER_NAME)
        self.token_client = token_client or httpx.Client(timeout=15.0)
        self.clock = clock
        self._token: str | None = None
        self._token_expiry = 0.0

    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _bearer(self) -> dict[str, str]:
        if not self.is_configured():
            return {}
        if self._token is None or self.clock() >= self._token_expiry - TOKEN_REFRESH_MARGIN_S:
            response = self.token_client.post(
                self.token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if response.status_code != 200:
                raise ProviderAccessDenied(f"opensky: token request failed with HTTP {response.status_code}")
            payload = response.json()
            self._token = payload["access_token"]
            self._token_expiry = self.clock() + float(payload.get("expires_in", 1800))
        return {"Authorization": f"Bearer {self._token}"}

    def check_availability(self) -> ProviderStatus:
        if not self.is_configured():
            return ProviderStatus(
                self.name, False, False, self.capabilities, "client credentials not configured; history needs OAuth2"
            )
        try:
            self._bearer()
        except ProviderError as error:
            return ProviderStatus(self.name, True, False, self.capabilities, str(error))
        return ProviderStatus(
            self.name,
            True,
            True,
            self.capabilities,
            "state vectors reach back one hour; the experimental tracks endpoint reaches back 30 days",
        )

    def estimate_requests(self, box: BoundingBox, window: TimeWindow) -> int:
        return int(window.duration // STATE_STEP_S) + 2

    def search_area(self, box: BoundingBox, window: TimeWindow) -> list[FlightRecord]:
        if not self.is_configured():
            raise ProviderCapabilityError("opensky: historic state vectors need OAuth2 client credentials")
        if self.clock() - window.start > STATE_HISTORY_LIMIT_S:
            raise ProviderCapabilityError("opensky: state vectors are only available for the last hour")
        records: dict[str, FlightRecord] = {}
        timestamp = int(window.start)
        while timestamp <= int(window.end):
            payload = self.session.get_json(
                f"{self.base_url}/states/all",
                params={
                    "time": timestamp,
                    "lamin": box.south,
                    "lomin": box.west,
                    "lamax": box.north,
                    "lomax": box.east,
                },
                headers=self._bearer(),
                retry_after_header="X-Rate-Limit-Retry-After-Seconds",
            )
            for row in (payload or {}).get("states") or []:
                observation = _state_observation(row)
                if observation is None:
                    continue
                icao24 = normalize_icao24(row[0])
                record = records.setdefault(
                    icao24,
                    FlightRecord(AircraftIdentity(icao24=icao24, callsign=normalize_callsign(row[1])), [], [self.name]),
                )
                record.observations.append(observation)
            timestamp += STATE_STEP_S
        return list(records.values())

    def get_track(self, identity: AircraftIdentity, window: TimeWindow) -> FlightRecord | None:
        hex_code = normalize_icao24(identity.icao24)
        if not hex_code or self.clock() - window.start > TRACK_HISTORY_LIMIT_S:
            return None
        payload = self.session.get_json(
            f"{self.base_url}/tracks/all",
            params={"icao24": hex_code, "time": int(window.start)},
            headers=self._bearer(),
            retry_after_header="X-Rate-Limit-Retry-After-Seconds",
        )
        if not payload or not payload.get("path"):
            return None
        points = [_track_observation(point) for point in payload["path"]]
        return FlightRecord(identity, points, [self.name])


def _state_observation(row: list[Any]) -> PositionObservation | None:
    time_position, longitude, latitude = row[3], row[5], row[6]
    if time_position is None or latitude is None or longitude is None:
        return None
    geometric = row[13]
    barometric = row[7]
    source = POSITION_SOURCES.get(row[16] if len(row) > 16 else 0, "UNKNOWN")
    return PositionObservation(
        timestamp=float(time_position),
        latitude=float(latitude),
        longitude=float(longitude),
        altitude_m=float(geometric) if geometric is not None else (None if barometric is None else float(barometric)),
        altitude_kind=AltitudeKind.GEOMETRIC if geometric is not None else AltitudeKind.BAROMETRIC,
        ground_speed_mps=None if row[9] is None else float(row[9]),
        track_deg=None if row[10] is None else float(row[10]),
        vertical_rate_mps=None if row[11] is None else float(row[11]),
        on_ground=bool(row[8]),
        provider=PROVIDER_NAME,
        source_type=source,
        origin=ValueOrigin.MEASURED if source == "MLAT" else ValueOrigin.REPORTED,
    )


def _track_observation(point: list[Any]) -> PositionObservation:
    return PositionObservation(
        timestamp=float(point[0]),
        latitude=float(point[1]),
        longitude=float(point[2]),
        altitude_m=None if point[3] is None else float(point[3]),
        altitude_kind=AltitudeKind.BAROMETRIC,
        ground_speed_mps=None,
        track_deg=None if point[4] is None else float(point[4]),
        vertical_rate_mps=None,
        on_ground=bool(point[5]),
        provider=PROVIDER_NAME,
        source_type="OPENSKY_TRACK",
    )
