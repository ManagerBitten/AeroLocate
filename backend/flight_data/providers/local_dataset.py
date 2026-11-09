from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

from backend.flight_data.normalization.units import normalize_callsign, normalize_icao24, parse_utc
from backend.flight_data.providers.adsbexchange import parse_trace
from backend.flight_data.providers.base import FlightDataProvider
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

PROVIDER_NAME = "local_dataset"


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _row_observation(row: dict[str, Any], provider: str) -> PositionObservation:
    return PositionObservation(
        timestamp=parse_utc(_numeric_or_text(row["timestamp"])),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        altitude_m=_optional_float(row.get("altitude_m")),
        altitude_kind=AltitudeKind(row.get("altitude_kind") or "unknown"),
        ground_speed_mps=_optional_float(row.get("ground_speed_mps")),
        track_deg=_optional_float(row.get("track_deg")),
        vertical_rate_mps=_optional_float(row.get("vertical_rate_mps")),
        on_ground=str(row.get("on_ground", "false")).lower() == "true",
        provider=provider,
        source_type=str(row.get("source_type") or "LOCAL"),
        origin=ValueOrigin(row.get("origin") or "reported"),
    )


def _numeric_or_text(value: Any) -> str | float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _open_text(path: Path) -> Any:
    with path.open("rb") as probe:
        gzipped = probe.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt", encoding="utf-8") if gzipped else path.open("r", encoding="utf-8")


class LocalFlightDatasetProvider(FlightDataProvider):
    name = PROVIDER_NAME
    capabilities = (ProviderCapability.HISTORIC_AREA_SEARCH, ProviderCapability.HISTORIC_TRACK_BY_ID)

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _dataset_files(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return [path for path in sorted(self.directory.iterdir()) if path.is_file() and not path.name.startswith(".")]

    def is_configured(self) -> bool:
        return bool(self._dataset_files())

    def check_availability(self) -> ProviderStatus:
        configured = self.is_configured()
        detail = f"{self.directory} is ready" if configured else f"no files in {self.directory}"
        return ProviderStatus(self.name, configured, configured, self.capabilities, detail)

    def _load(self) -> Iterable[FlightRecord]:
        for path in self._dataset_files():
            suffixes = "".join(path.suffixes).lower()
            if ".csv" in suffixes or ".jsonl" in suffixes:
                yield from self._load_rows(path, ".csv" in suffixes)
            else:
                with _open_text(path) as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict) and "trace" in payload:
                    yield parse_trace(payload, self.name)

    def _load_rows(self, path: Path, is_csv: bool) -> Iterable[FlightRecord]:
        grouped: dict[str, FlightRecord] = {}
        with _open_text(path) as handle:
            rows = csv.DictReader(handle) if is_csv else (json.loads(line) for line in handle if line.strip())
            for row in rows:
                key = normalize_icao24(row.get("icao24")) or normalize_callsign(row.get("callsign"))
                if key is None:
                    continue
                record = grouped.setdefault(
                    key,
                    FlightRecord(
                        AircraftIdentity(
                            icao24=normalize_icao24(row.get("icao24")),
                            callsign=normalize_callsign(row.get("callsign")),
                            flight_number=row.get("flight_number") or None,
                            registration=row.get("registration") or None,
                            aircraft_type=row.get("aircraft_type") or None,
                        ),
                        [],
                        [self.name],
                    ),
                )
                record.observations.append(_row_observation(row, self.name))
        yield from grouped.values()

    def search_area(self, box: BoundingBox, window: TimeWindow) -> list[FlightRecord]:
        matches = []
        for record in self._load():
            touches = any(
                window.start <= o.timestamp <= window.end
                and box.south <= o.latitude <= box.north
                and box.west <= o.longitude <= box.east
                for o in record.observations
            )
            if touches:
                matches.append(record)
        return matches

    def get_track(self, identity: AircraftIdentity, window: TimeWindow) -> FlightRecord | None:
        for record in self._load():
            if identity.icao24 and record.identity.icao24 == identity.icao24:
                return record
            if identity.callsign and record.identity.callsign == identity.callsign:
                return record
        return None


def write_records_jsonl(path: Path, records: Iterable[FlightRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            for observation in record.observations:
                row = observation.to_dict()
                row.pop("raw", None)
                row.pop("stale", None)
                row.pop("provider", None)
                row.update(
                    icao24=record.identity.icao24,
                    callsign=record.identity.callsign,
                    flight_number=record.identity.flight_number,
                    registration=record.identity.registration,
                    aircraft_type=record.identity.aircraft_type,
                )
                handle.write(json.dumps(row) + "\n")
