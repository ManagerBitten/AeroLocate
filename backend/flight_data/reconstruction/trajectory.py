from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from backend.core.geo import LocalFrame
from backend.flight_data.types import AltitudeKind, PositionObservation, ValueOrigin

FloatArray = NDArray[np.float64]
MAX_PLAUSIBLE_SPEED_MPS = 420.0
MAX_PLAUSIBLE_CLIMB_MPS = 80.0
UNRELIABLE_GAP_S = 45.0
MANEUVER_ACCELERATION_MPS2 = 1.5
BAROMETRIC_ALTITUDE_SIGMA_M = 120.0
GEOMETRIC_ALTITUDE_SIGMA_M = 25.0
HORIZONTAL_POSITION_SIGMA_M = 30.0
UNREPORTED_VELOCITY_SIGMA_MPS = 3.0
REPORTED_VELOCITY_SIGMA_MPS = 1.0
SMOOTHING_HALF_WINDOW_S = 20.0
VELOCITY_SIGMA_FLOOR_MPS = 0.15
MIN_POINTS_FOR_SMOOTHING = 5


@dataclass(frozen=True)
class TrajectoryGap:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class HermiteTrajectory:
    times: FloatArray
    positions: FloatArray
    velocities: FloatArray
    velocity_measured: NDArray[np.bool_]
    altitude_sigma_m: FloatArray
    gaps: list[TrajectoryGap] = field(default_factory=list)
    velocity_sigma_mps: float = VELOCITY_SIGMA_FLOOR_MPS

    @property
    def t_start(self) -> float:
        return float(self.times[0])

    @property
    def t_end(self) -> float:
        return float(self.times[-1])

    def state(self, t: FloatArray) -> tuple[FloatArray, FloatArray]:
        query = np.asarray(t, dtype=float)
        index = np.clip(np.searchsorted(self.times, query, side="right") - 1, 0, len(self.times) - 2)
        t0 = self.times[index]
        h = self.times[index + 1] - t0
        s = ((query - t0) / h)[..., None]
        scale = h[..., None]
        p0, p1 = self.positions[index], self.positions[index + 1]
        m0, m1 = self.velocities[index] * scale, self.velocities[index + 1] * scale
        s2 = s * s
        s3 = s2 * s
        position = (
            (2 * s3 - 3 * s2 + 1) * p0
            + (s3 - 2 * s2 + s) * m0
            + (-2 * s3 + 3 * s2) * p1
            + (s3 - s2) * m1
        )
        velocity = (
            (6 * s2 - 6 * s) * p0
            + (3 * s2 - 4 * s + 1) * m0
            + (-6 * s2 + 6 * s) * p1
            + (3 * s2 - 2 * s) * m1
        ) / scale
        return position, velocity

    def interpolation_sigma_m(self, t: FloatArray) -> FloatArray:
        query = np.asarray(t, dtype=float)
        index = np.clip(np.searchsorted(self.times, query, side="right") - 1, 0, len(self.times) - 2)
        to_prev = np.abs(query - self.times[index])
        to_next = np.abs(self.times[index + 1] - query)
        gap_offset = np.minimum(to_prev, to_next)
        maneuver = 0.5 * MANEUVER_ACCELERATION_MPS2 * gap_offset**2
        outside = (query < self.t_start) | (query > self.t_end)
        return np.where(
            outside,
            np.inf,
            np.sqrt(HORIZONTAL_POSITION_SIGMA_M**2 + maneuver**2 + self.altitude_sigma_m[index] ** 2),
        )

    def covers(self, start: float, end: float) -> bool:
        return self.t_start <= start and self.t_end >= end

    def has_gap_within(self, start: float, end: float) -> bool:
        return any(gap.start < end and gap.end > start for gap in self.gaps)

    def completeness(self, start: float, end: float) -> float:
        span = max(end - start, 1e-9)
        covered = max(0.0, min(end, self.t_end) - max(start, self.t_start))
        lost = sum(
            max(0.0, min(end, gap.end) - max(start, gap.start)) for gap in self.gaps
        )
        return float(np.clip((covered - lost) / span, 0.0, 1.0))


def _reject_implausible(
    times: FloatArray, enu: FloatArray
) -> NDArray[np.bool_]:
    keep = np.ones(len(times), dtype=bool)
    last = 0
    for i in range(1, len(times)):
        dt = times[i] - times[last]
        if dt <= 0:
            keep[i] = False
            continue
        delta = enu[i] - enu[last]
        horizontal = float(np.hypot(delta[0], delta[1])) / dt
        vertical = abs(float(delta[2])) / dt
        tolerance = 2.0 * HORIZONTAL_POSITION_SIGMA_M / dt
        if horizontal > MAX_PLAUSIBLE_SPEED_MPS + tolerance or vertical > MAX_PLAUSIBLE_CLIMB_MPS + tolerance:
            keep[i] = False
            continue
        last = i
    return keep


def _local_kinematic_fit(
    times: FloatArray,
    positions: FloatArray,
    velocities: FloatArray,
    velocity_known: NDArray[np.bool_],
    position_sigma: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    fitted_positions = positions.copy()
    fitted_velocities = velocities.copy()
    fitted_sigma = np.full(len(times), UNREPORTED_VELOCITY_SIGMA_MPS)
    for node, centre in enumerate(times):
        members = np.flatnonzero(np.abs(times - centre) <= SMOOTHING_HALF_WINDOW_S)
        if len(members) < MIN_POINTS_FOR_SMOOTHING:
            continue
        dt = times[members] - centre
        taper = (1.0 - (np.abs(dt) / (SMOOTHING_HALF_WINDOW_S * 1.001)) ** 3) ** 3
        known = velocity_known[members]
        position_rows = np.column_stack([np.ones_like(dt), dt, dt**2 / 2.0, dt**3 / 6.0])
        velocity_rows = np.column_stack([np.zeros_like(dt), np.ones_like(dt), dt, dt**2 / 2.0])[known]
        design = np.vstack([position_rows, velocity_rows])
        for axis in range(3):
            position_weight = np.sqrt(taper) / position_sigma[axis]
            velocity_weight = np.sqrt(taper[known]) / REPORTED_VELOCITY_SIGMA_MPS
            weights = np.concatenate([position_weight, velocity_weight])
            targets = np.concatenate([positions[members, axis], velocities[members[known], axis]])
            weighted = design * weights[:, None]
            normal = weighted.T @ weighted + np.diag([0.0, 0.0, 1e-6, 1e-6])
            solution = np.linalg.solve(normal, weighted.T @ (targets * weights))
            fitted_positions[node, axis] = solution[0]
            fitted_velocities[node, axis] = solution[1]
            if axis == 0:
                fitted_sigma[node] = float(np.sqrt(np.linalg.inv(normal)[1, 1]))
    return fitted_positions, fitted_velocities, fitted_sigma


def reconstruct(observations: Sequence[PositionObservation], frame: LocalFrame) -> HermiteTrajectory | None:
    usable = sorted(
        (
            o
            for o in observations
            if o.altitude_m is not None
            and not o.on_ground
            and not o.stale
            and o.origin is not ValueOrigin.INFERRED
        ),
        key=lambda o: o.timestamp,
    )
    if len(usable) < 2:
        return None
    times = np.array([o.timestamp for o in usable], dtype=float)
    enu = frame.to_enu(
        [o.latitude for o in usable], [o.longitude for o in usable], [o.altitude_m for o in usable]
    )
    keep = _reject_implausible(times, enu)
    usable = [o for o, k in zip(usable, keep) if k]
    if len(usable) < 2:
        return None
    times = times[keep]
    enu = enu[keep]
    finite_difference = np.gradient(enu, times, axis=0)
    velocities = finite_difference.copy()
    measured = np.zeros(len(times), dtype=bool)
    for i, observation in enumerate(usable):
        if observation.ground_speed_mps is not None and observation.track_deg is not None:
            bearing = np.radians(observation.track_deg)
            velocities[i, 0] = observation.ground_speed_mps * np.sin(bearing)
            velocities[i, 1] = observation.ground_speed_mps * np.cos(bearing)
            measured[i] = True
        if observation.vertical_rate_mps is not None:
            velocities[i, 2] = observation.vertical_rate_mps
    altitude_sigma = np.array(
        [
            GEOMETRIC_ALTITUDE_SIGMA_M
            if o.altitude_kind is AltitudeKind.GEOMETRIC
            else BAROMETRIC_ALTITUDE_SIGMA_M
            for o in usable
        ]
    )
    steps = np.diff(times)
    gaps = [
        TrajectoryGap(float(times[i]), float(times[i + 1]))
        for i in np.flatnonzero(steps > UNRELIABLE_GAP_S)
    ]
    position_sigma = np.array(
        [HORIZONTAL_POSITION_SIGMA_M, HORIZONTAL_POSITION_SIGMA_M, float(np.median(altitude_sigma))]
    )
    smoothed_positions, smoothed_velocities, velocity_sigma = _local_kinematic_fit(
        times, enu, velocities, measured, position_sigma
    )
    return HermiteTrajectory(
        times,
        smoothed_positions,
        smoothed_velocities,
        measured,
        altitude_sigma,
        gaps,
        max(float(np.median(velocity_sigma)), VELOCITY_SIGMA_FLOOR_MPS),
    )
