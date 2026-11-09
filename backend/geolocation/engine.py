from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

import numpy as np

from backend.core.atmosphere import Atmosphere
from backend.core.geo import LocalFrame
from backend.doppler.curve_fit import CurveClass
from backend.doppler.engine import DopplerCurve
from backend.flight_data.types import BoundingBox
from backend.geolocation.likelihood import EventData, prepare_event
from backend.geolocation.regions import CandidateRegion, RegionSummary, summarize_regions
from backend.geolocation.search import (
    PairRejection,
    Posterior,
    SearchConfig,
    TrackedFlight,
    search_posterior,
)
from backend.optimization.refine import RefinedSolution, monte_carlo_sensitivity, refine_solution
from backend.propagation.simulator import predict_doppler

CANDIDATE_THRESHOLD = 0.5
LOCATED_THRESHOLD = 0.9
DOMINANT_REGION_MASS = 0.8
ALTERNATIVE_REGION_MASS = 0.1
MAX_LOCALIZED_RADIUS_M = 2500.0
MIN_RESPONSIBILITY = 0.5
CONSTRAINTS_PER_PASS = 2
POSITION_UNKNOWNS = 2
MIN_SURPLUS_CONSTRAINTS = 2
MAX_LOCALIZING_SLANT_M = 6000.0
CLOCK_FREE_WINDOW_S = 5.0
HORIZONTAL_POSITION_SIGMA_M = 30.0
GROUND_ELEVATION_SIGMA_M = 15.0
MAX_BLUR_GEOMETRY_GAIN = 2.0
SAME_SOURCE_SIGMAS = 3.0
CALIBRATION_NOTE = (
    "Probabilities are model posteriors conditional on the stated assumptions. "
    "They have been checked against synthetic recordings only and are not calibrated on real recordings."
)


class LocalizationStatus(str, Enum):
    LOCALIZED = "localized"
    REGION_ONLY = "region_only"
    AMBIGUOUS = "ambiguous"
    NOT_LOCALIZABLE = "not_localizable"


@dataclass(frozen=True)
class LocalizationRequest:
    region: BoundingBox
    start_earliest: float
    start_latest: float
    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    microphone_elevation_m: float = 0.0


@dataclass
class FlightMatch:
    flight_key: str
    event_id: str
    probability: float
    predicted_times: list[float]
    predicted_frequency_hz: list[float]
    observed_frequency_hz: list[float]
    source_frequency_hz: float | None
    closest_range_m: float


@dataclass
class ComparisonRow:
    label: str
    events: list[str]
    status: str
    credible_area_km2: float
    region_count: int
    located_probability: float


@dataclass
class LocalizationResult:
    status: LocalizationStatus
    reasons: list[str]
    located_probability: float
    regions: tuple[CandidateRegion, ...]
    latitude: float | None
    longitude: float | None
    coordinate_decimals: int | None
    uncertainty_radius_m: float | None
    matches: list[FlightMatch]
    untracked_probability: dict[str, float]
    associations: dict[str, dict[str, float]]
    rejected_pairs: list[PairRejection]
    recording_start_utc: float | None
    recording_start_sigma_s: float | None
    refined: RefinedSolution | None
    sensitivity_sigma_m: float | None
    comparison: list[ComparisonRow]
    diagnostics: dict[str, Any]
    calibration_note: str = CALIBRATION_NOTE


def select_independent_curves(curves: Sequence[DopplerCurve]) -> tuple[list[DopplerCurve], list[str]]:
    usable = sorted((curve for curve in curves if curve.usable), key=lambda item: item.quality, reverse=True)
    chosen: list[DopplerCurve] = []
    dropped: list[str] = []
    for curve in usable:
        duplicate = next((other for other in chosen if _same_source(curve, other)), None)
        if duplicate is None:
            chosen.append(curve)
        else:
            dropped.append(
                f"{curve.event_id} repeats the geometry of {duplicate.event_id} and was excluded to avoid double counting"
            )
    return chosen, dropped


def _same_source(a: DopplerCurve, b: DopplerCurve) -> bool:
    if a.fit is None or b.fit is None:
        return False
    time_sigma = math.hypot(a.fit.closest_time_sigma * a.noise_inflation, b.fit.closest_time_sigma * b.noise_inflation)
    speed_sigma = math.hypot(a.fit.speed_sigma * a.noise_inflation, b.fit.speed_sigma * b.noise_inflation)
    distance_sigma = math.hypot(a.fit.distance_sigma * a.noise_inflation, b.fit.distance_sigma * b.noise_inflation)
    return (
        abs(a.fit.closest_time - b.fit.closest_time) < SAME_SOURCE_SIGMAS * max(time_sigma, 0.3)
        and abs(a.fit.speed_mps - b.fit.speed_mps) < SAME_SOURCE_SIGMAS * max(speed_sigma, 2.0)
        and abs(a.fit.distance_m - b.fit.distance_m) < SAME_SOURCE_SIGMAS * max(distance_sigma, 30.0)
    )


def _blur_sigma(posterior: Posterior, flights: Sequence[TrackedFlight], events: Sequence[EventData]) -> float:
    terms = []
    by_key = {flight.key: flight for flight in flights}
    for event in events:
        options = posterior.responsibilities.get(event.event_id, {})
        if not options:
            continue
        key = max(options, key=lambda name: options[name])
        trajectory = by_key[key].trajectory
        fit = event.curve.fit
        height = float(np.median(trajectory.positions[:, 2]))
        sigma_h = math.hypot(float(np.median(trajectory.altitude_sigma_m)), GROUND_ELEVATION_SIGMA_M)
        if fit is None:
            gain = MAX_BLUR_GEOMETRY_GAIN
        else:
            horizontal = math.sqrt(max(fit.distance_m**2 - height**2, 1.0))
            gain = min(abs(height) / horizontal, MAX_BLUR_GEOMETRY_GAIN)
        terms.append(1.0 / (HORIZONTAL_POSITION_SIGMA_M**2 + (gain * sigma_h) ** 2))
    if not terms:
        return HORIZONTAL_POSITION_SIGMA_M
    return float(math.sqrt(1.0 / sum(terms)))


def _decimals_for(radius_m: float) -> int:
    meters_per_decimal = [111_000.0, 11_100.0, 1_110.0, 111.0]
    for decimals, size in enumerate(meters_per_decimal):
        if size <= radius_m:
            return decimals
    return len(meters_per_decimal) - 1


def _run(
    events: Sequence[EventData],
    flights: Sequence[TrackedFlight],
    request: LocalizationRequest,
    frame: LocalFrame,
    progress: Callable[[float], None] | None,
) -> tuple[Posterior, RegionSummary]:
    config = SearchConfig(
        region=request.region,
        start_earliest=request.start_earliest,
        start_latest=request.start_latest,
        atmosphere=request.atmosphere,
        microphone_elevation_m=request.microphone_elevation_m,
    )
    posterior = search_posterior(events, flights, config, frame, progress)
    return posterior, summarize_regions(posterior, _blur_sigma(posterior, flights, events))


def _supporting_flights(posterior: Posterior, events: Sequence[EventData]) -> int:
    flights: set[str] = set()
    for event in events:
        fit = event.curve.fit
        if fit is None or fit.distance_m > MAX_LOCALIZING_SLANT_M:
            continue
        options = posterior.responsibilities.get(event.event_id, {})
        if options:
            key = max(options, key=lambda name: options[name])
            if options[key] >= MIN_RESPONSIBILITY:
                flights.add(key)
    return len(flights)


def _surplus_constraints(supporting: int, clock_is_free: bool) -> int:
    return CONSTRAINTS_PER_PASS * supporting - POSITION_UNKNOWNS - int(clock_is_free)


def _classify(
    posterior: Posterior,
    summary: RegionSummary,
    events: Sequence[EventData],
    clock_is_free: bool,
) -> tuple[LocalizationStatus, list[str]]:
    reasons: list[str] = []
    probability = posterior.located_probability
    if not summary.regions or probability < CANDIDATE_THRESHOLD:
        reasons.append(
            f"The evidence does not favour any tracked flight over an untracked source (P={probability:.2f})."
        )
        return LocalizationStatus.NOT_LOCALIZABLE, reasons
    if probability < LOCATED_THRESHOLD:
        reasons.append(
            f"A tracked flight is only moderately preferred over an untracked source (P={probability:.2f}), "
            "so these regions are candidates rather than a location."
        )
        return LocalizationStatus.AMBIGUOUS, reasons
    leading = summary.regions[0]
    alternatives = [region for region in summary.regions[1:] if region.probability >= ALTERNATIVE_REGION_MASS]
    if alternatives or leading.probability < DOMINANT_REGION_MASS:
        reasons.append(
            f"{1 + len(alternatives)} separate regions remain plausible; the leading one holds {leading.probability:.0%}."
        )
        return LocalizationStatus.AMBIGUOUS, reasons
    if leading.radius_m > MAX_LOCALIZED_RADIUS_M or posterior.resolution_limited:
        reasons.append(f"The credible region is broad (equivalent radius {leading.radius_m:.0f} m).")
        return LocalizationStatus.REGION_ONLY, reasons
    supporting = _supporting_flights(posterior, events)
    if _surplus_constraints(supporting, clock_is_free) < MIN_SURPLUS_CONSTRAINTS:
        needed = "three" if clock_is_free else "two"
        reasons.append(
            f"{supporting} matched flight(s) within {MAX_LOCALIZING_SLANT_M / 1000:.0f} km support this region, and "
            f"{needed} are needed to confirm a location. Curves can agree with the wrong flights by chance, "
            "so this region stays a candidate."
        )
        return LocalizationStatus.AMBIGUOUS, reasons
    reasons.append(f"One region holds {leading.probability:.0%} of the posterior.")
    return LocalizationStatus.LOCALIZED, reasons


def _clock_is_free(request: LocalizationRequest) -> bool:
    return request.start_latest - request.start_earliest > CLOCK_FREE_WINDOW_S


def _not_localizable(reason: str, diagnostics: dict[str, Any] | None = None) -> LocalizationResult:
    return LocalizationResult(
        status=LocalizationStatus.NOT_LOCALIZABLE,
        reasons=[reason],
        located_probability=0.0,
        regions=(),
        latitude=None,
        longitude=None,
        coordinate_decimals=None,
        uncertainty_radius_m=None,
        matches=[],
        untracked_probability={},
        associations={},
        rejected_pairs=[],
        recording_start_utc=None,
        recording_start_sigma_s=None,
        refined=None,
        sensitivity_sigma_m=None,
        comparison=[],
        diagnostics=diagnostics or {},
    )


def _associated(
    events: Sequence[EventData], flights: Sequence[TrackedFlight], posterior: Posterior
) -> list[tuple[EventData, TrackedFlight]]:
    by_key = {flight.key: flight for flight in flights}
    pairs = []
    for event in events:
        options = posterior.responsibilities.get(event.event_id, {})
        if not options:
            continue
        key = max(options, key=lambda name: options[name])
        if options[key] >= MIN_RESPONSIBILITY:
            pairs.append((event, by_key[key]))
    return pairs


def _flight_matches(
    pairs: Sequence[tuple[EventData, TrackedFlight]],
    posterior: Posterior,
    solution: RefinedSolution,
    request: LocalizationRequest,
) -> list[FlightMatch]:
    matches = []
    microphone = np.array([solution.x_m, solution.y_m, request.microphone_elevation_m])
    for event, flight in pairs:
        curve = event.curve
        prediction = predict_doppler(flight.trajectory, solution.start + curve.times, microphone, request.atmosphere)
        source = solution.source_frequencies_hz.get(event.event_id)
        scale = float(np.exp(np.median(curve.log_frequency - np.log(prediction.doppler_factor))))
        matches.append(
            FlightMatch(
                flight_key=flight.key,
                event_id=event.event_id,
                probability=posterior.responsibilities[event.event_id][flight.key],
                predicted_times=curve.times.tolist(),
                predicted_frequency_hz=(scale * prediction.doppler_factor).tolist(),
                observed_frequency_hz=np.exp(curve.log_frequency).tolist(),
                source_frequency_hz=source,
                closest_range_m=float(prediction.distance_m.min()),
            )
        )
    return matches


def _comparison(
    events: Sequence[EventData],
    flights: Sequence[TrackedFlight],
    request: LocalizationRequest,
    frame: LocalFrame,
    joint: tuple[Posterior, RegionSummary],
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    if len(events) < 2:
        return rows
    for event in events:
        posterior, summary = _run([event], flights, request, frame, None)
        status, _ = _classify(posterior, summary, [event], _clock_is_free(request))
        rows.append(
            ComparisonRow(
                label=f"{event.event_id} alone",
                events=[event.event_id],
                status=status.value,
                credible_area_km2=summary.total_area_km2,
                region_count=len(summary.regions),
                located_probability=posterior.located_probability,
            )
        )
    posterior, summary = joint
    status, _ = _classify(posterior, summary, events, _clock_is_free(request))
    rows.append(
        ComparisonRow(
            label=f"all {len(events)} events",
            events=[event.event_id for event in events],
            status=status.value,
            credible_area_km2=summary.total_area_km2,
            region_count=len(summary.regions),
            located_probability=posterior.located_probability,
        )
    )
    return rows


def localize(
    curves: Sequence[DopplerCurve],
    flights: Sequence[TrackedFlight],
    request: LocalizationRequest,
    progress: Callable[[float], None] | None = None,
    compare: bool = True,
) -> LocalizationResult:
    chosen, dropped = select_independent_curves(curves)
    counts = {item.value: sum(curve.curve_class is item for curve in curves) for item in CurveClass}
    diagnostics: dict[str, Any] = {"curve_classes": counts, "excluded_events": dropped, "flights_considered": len(flights)}
    if not chosen:
        return _not_localizable("No aircraft event produced a measurable Doppler curve.", diagnostics)
    if not flights:
        return _not_localizable(
            "No flight trajectories were available for the requested region and time window.", diagnostics
        )
    frame = LocalFrame(*request.region.center, 0.0)
    events = [prepare_event(curve) for curve in chosen]
    posterior, summary = _run(events, flights, request, frame, progress)
    status, reasons = _classify(posterior, summary, events, _clock_is_free(request))
    reasons.extend(dropped)
    if status is LocalizationStatus.NOT_LOCALIZABLE:
        result = _not_localizable(reasons[0], diagnostics)
        result.reasons = reasons
        result.located_probability = posterior.located_probability
        result.untracked_probability = posterior.untracked_probability
        result.associations = posterior.responsibilities
        result.rejected_pairs = posterior.rejections
        return result
    start_known = request.start_latest - request.start_earliest <= posterior.time_step_s
    pairs = _associated(events, flights, posterior)
    refined: RefinedSolution | None = None
    sensitivity: float | None = None
    matches: list[FlightMatch] = []
    leading = summary.regions[0]
    if pairs:
        refined = refine_solution(
            pairs,
            leading.x_m,
            leading.y_m,
            summary.start_mean,
            request.microphone_elevation_m,
            request.atmosphere,
            start_known,
        )
        sensitivity = monte_carlo_sensitivity(
            pairs, refined, request.microphone_elevation_m, request.atmosphere, start_known
        )
        matches = _flight_matches(pairs, posterior, refined, request)
    radius = leading.radius_m if sensitivity is None else math.hypot(leading.radius_m, 2.45 * sensitivity)
    latitude = longitude = None
    decimals = None
    if status is LocalizationStatus.LOCALIZED:
        decimals = _decimals_for(radius)
        latitude = round(leading.latitude, decimals)
        longitude = round(leading.longitude, decimals)
    if len(events) == 1:
        reasons.append("Only one aircraft event was usable, so the solution rests on a single pass-by.")
    if posterior.resolution_limited:
        reasons.append(f"Refinement stopped at {posterior.cell_m:.0f} m cells because the posterior stayed broad.")
    if not start_known and summary.start_sigma > 5.0:
        reasons.append(f"The recording time is only constrained to ±{summary.start_sigma:.0f} s.")
    diagnostics.update(
        cell_m=posterior.cell_m,
        time_step_s=posterior.time_step_s,
        tiles_evaluated=posterior.tiles_evaluated,
        pairs_evaluated=posterior.pairs_evaluated,
        search_levels=posterior.levels,
        blur_sigma_m=summary.blur_sigma_m,
    )
    return LocalizationResult(
        status=status,
        reasons=reasons,
        located_probability=posterior.located_probability,
        regions=summary.regions,
        latitude=latitude,
        longitude=longitude,
        coordinate_decimals=decimals,
        uncertainty_radius_m=radius,
        matches=matches,
        untracked_probability=posterior.untracked_probability,
        associations=posterior.responsibilities,
        rejected_pairs=posterior.rejections,
        recording_start_utc=summary.start_mean,
        recording_start_sigma_s=summary.start_sigma,
        refined=refined,
        sensitivity_sigma_m=sensitivity,
        comparison=_comparison(events, flights, request, frame, (posterior, summary)) if compare else [],
        diagnostics=diagnostics,
    )
