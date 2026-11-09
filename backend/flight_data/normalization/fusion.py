from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.core.geo import haversine_m
from backend.flight_data.normalization.units import normalize_callsign, normalize_icao24
from backend.flight_data.types import (
    AircraftIdentity,
    AltitudeKind,
    FlightRecord,
    PositionObservation,
    ValueOrigin,
)

DUPLICATE_WINDOW_S = 1.0
TRAJECTORY_MATCH_M = 5000.0
TRAJECTORY_MATCH_TIME_S = 30.0
ALTITUDE_CONFLICT_M = 150.0
ADS_B_FAMILY = "adsb"


@dataclass
class FusedFlight:
    identity: AircraftIdentity
    observations: list[PositionObservation]
    providers: list[str]
    conflicts: list[str] = field(default_factory=list)
    independent_signal_sources: int = 0
    provider_agreement_is_independent: bool = False
    provenance_counts: dict[str, int] = field(default_factory=dict)


def _same_callsign(a: FlightRecord, b: FlightRecord) -> bool:
    first = normalize_callsign(a.identity.callsign)
    second = normalize_callsign(b.identity.callsign)
    return first is not None and first == second


def _trajectories_coincide(a: FlightRecord, b: FlightRecord) -> bool:
    if not a.observations or not b.observations:
        return False
    times_b = np.array([o.timestamp for o in b.observations])
    order = np.argsort(times_b)
    times_b = times_b[order]
    comparisons = 0
    agreements = 0
    for observation in a.observations[:: max(1, len(a.observations) // 20)]:
        index = int(np.clip(np.searchsorted(times_b, observation.timestamp), 0, len(times_b) - 1))
        if abs(times_b[index] - observation.timestamp) > TRAJECTORY_MATCH_TIME_S:
            continue
        other = b.observations[int(order[index])]
        comparisons += 1
        distance = float(haversine_m(observation.latitude, observation.longitude, other.latitude, other.longitude))
        agreements += distance < TRAJECTORY_MATCH_M
    return comparisons > 0 and agreements / comparisons >= 0.8


def _belongs_together(a: FlightRecord, b: FlightRecord) -> bool:
    hex_a = normalize_icao24(a.identity.icao24)
    hex_b = normalize_icao24(b.identity.icao24)
    if hex_a and hex_b:
        return hex_a == hex_b
    if a.identity.registration and a.identity.registration == b.identity.registration:
        return True
    return _same_callsign(a, b) and _trajectories_coincide(a, b)


def _observation_rank(observation: PositionObservation) -> tuple[int, int, int]:
    origin_rank = {
        ValueOrigin.MEASURED: 2,
        ValueOrigin.REPORTED: 2,
        ValueOrigin.INTERPOLATED: 1,
        ValueOrigin.INFERRED: 0,
    }[observation.origin]
    has_velocity = int(observation.ground_speed_mps is not None and observation.track_deg is not None)
    geometric = int(observation.altitude_kind is AltitudeKind.GEOMETRIC)
    return origin_rank, has_velocity, geometric


def _signal_family(observation: PositionObservation) -> str:
    source = observation.source_type.upper()
    if "MLAT" in source:
        return "mlat"
    if "ASTERIX" in source or "RADAR" in source:
        return "radar"
    if "FLARM" in source:
        return "flarm"
    if "ESTIMATED" in source:
        return "estimated"
    return ADS_B_FAMILY


def _merge_identity(records: list[FlightRecord]) -> AircraftIdentity:
    def first(attribute: str) -> str | None:
        for record in records:
            value = getattr(record.identity, attribute)
            if value:
                return value
        return None

    return AircraftIdentity(
        icao24=normalize_icao24(first("icao24")),
        callsign=normalize_callsign(first("callsign")),
        flight_number=first("flight_number"),
        registration=first("registration"),
        aircraft_type=first("aircraft_type"),
        provider_flight_id=first("provider_flight_id"),
    )


def _merge_observations(records: list[FlightRecord]) -> tuple[list[PositionObservation], list[str]]:
    ordered = sorted(
        (observation for record in records for observation in record.observations),
        key=lambda item: item.timestamp,
    )
    merged: list[PositionObservation] = []
    conflicts: list[str] = []
    for observation in ordered:
        if merged and observation.timestamp - merged[-1].timestamp <= DUPLICATE_WINDOW_S:
            current = merged[-1]
            if (
                current.provider != observation.provider
                and current.altitude_kind == observation.altitude_kind
                and current.altitude_m is not None
                and observation.altitude_m is not None
                and abs(current.altitude_m - observation.altitude_m) > ALTITUDE_CONFLICT_M
            ):
                conflicts.append(
                    f"altitude differs by {abs(current.altitude_m - observation.altitude_m):.0f} m "
                    f"between {current.provider} and {observation.provider} at t={observation.timestamp:.0f}"
                )
            if _observation_rank(observation) > _observation_rank(current):
                merged[-1] = observation
            continue
        merged.append(observation)
    return merged, conflicts


def fuse_records(records: list[FlightRecord]) -> list[FusedFlight]:
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if _belongs_together(records[i], records[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[FlightRecord]] = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)
    fused: list[FusedFlight] = []
    for group in groups.values():
        observations, conflicts = _merge_observations(group)
        providers = sorted({provider for record in group for provider in record.providers})
        families = {_signal_family(item) for record in group for item in record.observations}
        families.discard("estimated")
        counts: dict[str, int] = {}
        for observation in observations:
            counts[observation.provider] = counts.get(observation.provider, 0) + 1
        fused.append(
            FusedFlight(
                identity=_merge_identity(group),
                observations=observations,
                providers=providers,
                conflicts=conflicts,
                independent_signal_sources=len(families),
                provider_agreement_is_independent=len(families) > 1,
                provenance_counts=counts,
            )
        )
    return fused
