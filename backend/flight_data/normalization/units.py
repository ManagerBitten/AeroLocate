from __future__ import annotations

from datetime import datetime, timezone

FEET_TO_M = 0.3048
KNOTS_TO_MPS = 0.514444
FPM_TO_MPS = 0.00508


def feet_to_m(value: float | None) -> float | None:
    return None if value is None else float(value) * FEET_TO_M


def knots_to_mps(value: float | None) -> float | None:
    return None if value is None else float(value) * KNOTS_TO_MPS


def fpm_to_mps(value: float | None) -> float | None:
    return None if value is None else float(value) * FPM_TO_MPS


def parse_utc(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {value}")
    return parsed.astimezone(timezone.utc).timestamp()


def normalize_icao24(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().lower().lstrip("~")
    return cleaned or None


def normalize_callsign(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().upper()
    return cleaned or None
