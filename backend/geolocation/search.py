from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from backend.core.atmosphere import Atmosphere
from backend.core.geo import LocalFrame
from backend.doppler.curve_fit import CurveClass
from backend.flight_data.reconstruction.trajectory import HermiteTrajectory
from backend.flight_data.types import BoundingBox
from backend.geolocation.likelihood import (
    EventData,
    event_log_likelihood,
    tempering_sigma,
    track_log_sigma,
    untracked_log_likelihood,
)

FloatArray = NDArray[np.float64]
MAX_HORIZONTAL_RANGE_M = 25000.0
SPEED_MARGIN_FRACTION = 0.2
ALTITUDE_MARGIN_M = 400.0


class SearchCancelled(Exception):
    pass


@dataclass(frozen=True)
class SearchConfig:
    region: BoundingBox
    start_earliest: float
    start_latest: float
    atmosphere: Atmosphere
    microphone_elevation_m: float = 0.0
    initial_cell_m: float = 1600.0
    final_cell_m: float = 50.0
    initial_time_step_s: float = 8.0
    final_time_step_s: float = 0.25
    keep_margin: float = 14.0
    max_tiles: int = 400_000
    untracked_prior: float = 0.15


@dataclass
class TrackedFlight:
    key: str
    trajectory: HermiteTrajectory


@dataclass
class PairRejection:
    event_id: str
    flight_key: str
    reason: str


@dataclass
class Posterior:
    frame: LocalFrame
    cell_m: float
    time_step_s: float
    x: FloatArray
    y: FloatArray
    start: FloatArray
    log_weight: FloatArray
    log_evidence_located: float
    log_evidence_unlocated: float
    responsibilities: dict[str, dict[str, float]]
    untracked_probability: dict[str, float]
    rejections: list[PairRejection]
    pairs_evaluated: int
    levels: int
    tiles_evaluated: int
    resolution_limited: bool
    workload_note: str = ""

    @property
    def located_probability(self) -> float:
        total = np.logaddexp(self.log_evidence_located, self.log_evidence_unlocated)
        return float(np.exp(self.log_evidence_located - total))


@dataclass
class _Pair:
    event_index: int
    flight_index: int


def _speed_and_height_profile(
    trajectory: HermiteTrajectory, start: float, end: float, elevation: float
) -> tuple[float, float, float]:
    times = np.linspace(max(start, trajectory.t_start), min(end, trajectory.t_end), 60)
    position, velocity = trajectory.state(times)
    speed = np.linalg.norm(velocity, axis=1)
    return float(speed.min()), float(speed.max()), float((position[:, 2] - elevation).min())


def feasible_pairs(
    events: Sequence[EventData],
    flights: Sequence[TrackedFlight],
    config: SearchConfig,
) -> tuple[list[_Pair], list[PairRejection]]:
    pairs: list[_Pair] = []
    rejected: list[PairRejection] = []
    for event_index, event in enumerate(events):
        fit = event.curve.fit
        start = config.start_earliest + event.curve.start
        end = config.start_latest + event.curve.end
        for flight_index, flight in enumerate(flights):
            if flight.trajectory.t_end < start - 60.0 or flight.trajectory.t_start > end:
                rejected.append(PairRejection(event.event_id, flight.key, "track does not cover the time window"))
                continue
            if fit is not None and event.curve.curve_class is CurveClass.DOPPLER_CONSISTENT:
                low, high, height = _speed_and_height_profile(
                    flight.trajectory, start, end, config.microphone_elevation_m
                )
                tolerance = max(3.0 * fit.speed_sigma * event.curve.noise_inflation, SPEED_MARGIN_FRACTION * fit.speed_mps)
                if high < fit.speed_mps - tolerance or low > fit.speed_mps + tolerance:
                    rejected.append(
                        PairRejection(
                            event.event_id,
                            flight.key,
                            f"track speed {low:.0f}-{high:.0f} m/s is incompatible with the fitted {fit.speed_mps:.0f} m/s",
                        )
                    )
                    continue
                reach = fit.distance_m + 3.0 * fit.distance_sigma * event.curve.noise_inflation + ALTITUDE_MARGIN_M
                if height > reach:
                    rejected.append(
                        PairRejection(
                            event.event_id,
                            flight.key,
                            f"lowest height {height:.0f} m exceeds the fitted slant range {fit.distance_m:.0f} m",
                        )
                    )
                    continue
            pairs.append(_Pair(event_index, flight_index))
    return pairs, rejected


def _initial_tiles(config: SearchConfig, frame: LocalFrame) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    corners = frame.to_enu(
        [config.region.south, config.region.south, config.region.north, config.region.north],
        [config.region.west, config.region.east, config.region.west, config.region.east],
        [0.0, 0.0, 0.0, 0.0],
    )
    x_min, x_max = corners[:, 0].min(), corners[:, 0].max()
    y_min, y_max = corners[:, 1].min(), corners[:, 1].max()
    cell = config.initial_cell_m
    xs = np.arange(x_min + cell / 2, x_max + cell / 2, cell)
    ys = np.arange(y_min + cell / 2, y_max + cell / 2, cell)
    span = config.start_latest - config.start_earliest
    if span <= config.final_time_step_s:
        starts = np.array([0.5 * (config.start_earliest + config.start_latest)])
        step = config.final_time_step_s
    else:
        step = min(config.initial_time_step_s, span)
        starts = np.arange(config.start_earliest + step / 2, config.start_latest + step / 2, step)
    grid_x, grid_y, grid_t = np.meshgrid(xs, ys, starts, indexing="ij")
    return grid_x.ravel(), grid_y.ravel(), grid_t.ravel(), step


def _gate(
    event: EventData,
    trajectory: HermiteTrajectory,
    x: FloatArray,
    y: FloatArray,
    start: FloatArray,
    cell_m: float,
    time_step_s: float,
) -> NDArray[np.bool_]:
    fit = event.curve.fit
    if fit is not None and event.curve.curve_class is CurveClass.DOPPLER_CONSISTENT:
        emission = start + fit.closest_time - fit.distance_m / 340.0
        reach = fit.distance_m + 4.0 * fit.distance_sigma * event.curve.noise_inflation
    else:
        emission = start + float(event.curve.times.mean())
        reach = MAX_HORIZONTAL_RANGE_M
    position, velocity = trajectory.state(emission)
    distance = np.hypot(position[:, 0] - x, position[:, 1] - y)
    speed = np.linalg.norm(velocity, axis=1)
    tolerance = 1.5 * cell_m + speed * (2.0 * time_step_s + 4.0 * max(fit.closest_time_sigma * event.curve.noise_inflation, 1.0) if fit else time_step_s)
    return distance <= reach + tolerance


def _shared_path_sigma(
    events: Sequence[EventData], flights: Sequence[TrackedFlight], pairs: Sequence[_Pair], event_index: int
) -> float:
    event = events[event_index]
    values = [
        track_log_sigma(flights[pair.flight_index].trajectory, event)
        for pair in pairs
        if pair.event_index == event_index
    ]
    return float(np.median(values)) if values else 0.0


def _evaluate(
    events: Sequence[EventData],
    flights: Sequence[TrackedFlight],
    pairs: Sequence[_Pair],
    config: SearchConfig,
    x: FloatArray,
    y: FloatArray,
    start: FloatArray,
    cell_m: float,
    time_step_s: float,
) -> tuple[FloatArray, list[tuple[_Pair, NDArray[np.int64], FloatArray]]]:
    tiles = len(x)
    flight_prior = float(np.log((1.0 - config.untracked_prior) / max(len(flights), 1)))
    unlocated = np.log(config.untracked_prior)
    per_event = np.empty((len(events), tiles))
    clutter_terms = np.empty(len(events))
    pair_logs: list[tuple[_Pair, NDArray[np.int64], FloatArray]] = []
    for event_index, event in enumerate(events):
        temper = tempering_sigma(event, cell_m, time_step_s)
        clutter = unlocated + untracked_log_likelihood(
            event, temper, _shared_path_sigma(events, flights, pairs, event_index)
        )
        clutter_terms[event_index] = clutter
        terms = [np.full(tiles, clutter)]
        for pair in (item for item in pairs if item.event_index == event_index):
            trajectory = flights[pair.flight_index].trajectory
            selected = np.flatnonzero(_gate(event, trajectory, x, y, start, cell_m, time_step_s))
            if len(selected) == 0:
                continue
            values = event_log_likelihood(
                event,
                trajectory,
                np.column_stack([x[selected], y[selected]]),
                config.microphone_elevation_m,
                start[selected],
                config.atmosphere,
                np.hypot(temper, track_log_sigma(trajectory, event)),
            )
            full = np.full(tiles, -np.inf)
            full[selected] = flight_prior + values
            terms.append(full)
            pair_logs.append((pair, selected, values + flight_prior))
        per_event[event_index] = logsumexp(np.vstack(terms), axis=0)
    joint = per_event.sum(axis=0)
    baseline = float(clutter_terms.sum())
    excess = baseline - joint
    has_association = excess < -1e-9
    located = np.full(tiles, -np.inf)
    located[has_association] = joint[has_association] + np.log(-np.expm1(excess[has_association]))
    return located, pair_logs


def search_posterior(
    events: Sequence[EventData],
    flights: Sequence[TrackedFlight],
    config: SearchConfig,
    frame: LocalFrame,
    progress: Callable[[float], None] | None = None,
) -> Posterior:
    pairs, rejections = feasible_pairs(events, flights, config)
    x, y, start, time_step = _initial_tiles(config, frame)
    cell = config.initial_cell_m
    region_area = float(np.ptp(x) + cell) * float(np.ptp(y) + cell)
    time_span = max(config.start_latest - config.start_earliest, config.final_time_step_s)
    levels = int(np.ceil(np.log2(config.initial_cell_m / config.final_cell_m))) + 1
    tiles_evaluated = 0
    resolution_limited = False
    level = 0
    located = np.full(len(x), -np.inf)
    pair_logs: list[tuple[_Pair, NDArray[np.int64], FloatArray]] = []
    while True:
        located, pair_logs = _evaluate(events, flights, pairs, config, x, y, start, cell, time_step)
        tiles_evaluated += len(x)
        level += 1
        if progress:
            progress(min(level / levels, 1.0))
        finite = np.isfinite(located)
        if not finite.any():
            break
        peak = located[finite].max()
        keep = located >= peak - config.keep_margin
        finished_space = cell <= config.final_cell_m * 1.01
        finished_time = time_step <= config.final_time_step_s * 1.01
        if finished_space and finished_time:
            break
        split_space = not finished_space
        split_time = not finished_time
        factor = (4 if split_space else 1) * (2 if split_time else 1)
        if int(keep.sum()) * factor > config.max_tiles:
            resolution_limited = True
            break
        x, y, start, cell, time_step = _subdivide(x[keep], y[keep], start[keep], cell, time_step, split_space, split_time)
    log_area = np.log(cell * cell / region_area)
    log_time = np.log(time_step / time_span) if time_span > config.final_time_step_s else 0.0
    log_weight = located + log_area + log_time
    evidence = float(logsumexp(log_weight[np.isfinite(log_weight)])) if np.isfinite(log_weight).any() else -np.inf
    temper_terms = [
        np.log(config.untracked_prior)
        + untracked_log_likelihood(
            event, tempering_sigma(event, cell, time_step), _shared_path_sigma(events, flights, pairs, index)
        )
        for index, event in enumerate(events)
    ]
    responsibilities, untracked = _associations(events, flights, pair_logs, log_weight, temper_terms)
    return Posterior(
        frame=frame,
        cell_m=cell,
        time_step_s=time_step,
        x=x,
        y=y,
        start=start,
        log_weight=log_weight,
        log_evidence_located=evidence,
        log_evidence_unlocated=float(np.sum(temper_terms)),
        responsibilities=responsibilities,
        untracked_probability=untracked,
        rejections=rejections,
        pairs_evaluated=len(pairs),
        levels=level,
        tiles_evaluated=tiles_evaluated,
        resolution_limited=resolution_limited,
    )


def _subdivide(
    x: FloatArray,
    y: FloatArray,
    start: FloatArray,
    cell: float,
    time_step: float,
    split_space: bool,
    split_time: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:
    if split_space:
        quarter = cell / 4.0
        x = np.concatenate([x - quarter, x + quarter, x - quarter, x + quarter])
        y = np.concatenate([y - quarter, y - quarter, y + quarter, y + quarter])
        start = np.tile(start, 4)
        cell /= 2.0
    if split_time:
        quarter = time_step / 4.0
        x = np.tile(x, 2)
        y = np.tile(y, 2)
        start = np.concatenate([start - quarter, start + quarter])
        time_step /= 2.0
    return x, y, start, cell, time_step


def _associations(
    events: Sequence[EventData],
    flights: Sequence[TrackedFlight],
    pair_logs: list[tuple[_Pair, NDArray[np.int64], FloatArray]],
    log_weight: FloatArray,
    clutter_terms: list[float],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    finite = np.isfinite(log_weight)
    responsibilities: dict[str, dict[str, float]] = {event.event_id: {} for event in events}
    untracked: dict[str, float] = {}
    if not finite.any():
        return responsibilities, {event.event_id: 1.0 for event in events}
    normalized = np.exp(log_weight - logsumexp(log_weight[finite]))
    normalized[~finite] = 0.0
    tiles = len(log_weight)
    for event_index, event in enumerate(events):
        entries = [item for item in pair_logs if item[0].event_index == event_index]
        terms = [np.full(tiles, clutter_terms[event_index])]
        names = [None]
        for pair, selected, values in entries:
            full = np.full(tiles, -np.inf)
            full[selected] = values
            terms.append(full)
            names.append(flights[pair.flight_index].key)
        stacked = np.vstack(terms)
        posterior = np.exp(stacked - logsumexp(stacked, axis=0))
        mass = posterior @ normalized
        untracked[event.event_id] = float(mass[0])
        for name, value in zip(names[1:], mass[1:]):
            responsibilities[event.event_id][name] = float(value)
    return responsibilities, untracked
