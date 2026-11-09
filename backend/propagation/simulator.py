from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from backend.core.atmosphere import Atmosphere

FloatArray = NDArray[np.float64]
NEWTON_ITERATIONS = 12
NEWTON_TOLERANCE_S = 1e-6


class KinematicPath(Protocol):
    t_start: float
    t_end: float

    def state(self, t: FloatArray) -> tuple[FloatArray, FloatArray]: ...


@dataclass(frozen=True)
class DopplerPrediction:
    emission_time: FloatArray
    distance_m: FloatArray
    delay_s: FloatArray
    radial_velocity_mps: FloatArray
    doppler_factor: FloatArray
    log_factor_sigma: FloatArray
    valid: NDArray[np.bool_]


def solve_emission_time(
    path: KinematicPath,
    observer_time: FloatArray,
    microphone_enu: FloatArray,
    atmosphere: Atmosphere,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    guess = np.clip(observer_time - 5.0, path.t_start, path.t_end)
    valid = np.ones(np.shape(observer_time), dtype=bool)
    for _ in range(NEWTON_ITERATIONS):
        position, velocity = path.state(guess)
        offset = microphone_enu - position
        distance = np.maximum(np.linalg.norm(offset, axis=-1), 1e-3)
        unit = offset / distance[..., None]
        c = atmosphere.path_sound_speed(position[..., 2], microphone_enu[..., 2])
        wind = np.array([atmosphere.wind_east_mps, atmosphere.wind_north_mps, 0.0])
        c_eff = c + unit @ wind
        residual = guess + distance / c_eff - observer_time
        slope = 1.0 - np.sum(velocity * unit, axis=-1) / c_eff
        supersonic = slope <= 0.02
        step = np.where(supersonic, residual, residual / np.where(supersonic, 1.0, slope))
        updated = np.clip(guess - step, path.t_start, path.t_end)
        converged = np.abs(updated - guess) < NEWTON_TOLERANCE_S
        guess = updated
        if np.all(converged):
            break
    position, _ = path.state(guess)
    offset = microphone_enu - position
    distance = np.linalg.norm(offset, axis=-1)
    unit = offset / np.maximum(distance, 1e-3)[..., None]
    wind = np.array([atmosphere.wind_east_mps, atmosphere.wind_north_mps, 0.0])
    c_eff = atmosphere.path_sound_speed(position[..., 2], microphone_enu[..., 2]) + unit @ wind
    closure = np.abs(guess + distance / c_eff - observer_time)
    valid &= closure < 1e-3
    return guess, valid


def predict_doppler(
    path: KinematicPath,
    observer_time: FloatArray,
    microphone_enu: FloatArray,
    atmosphere: Atmosphere,
) -> DopplerPrediction:
    observer_time = np.asarray(observer_time, dtype=float)
    microphone_enu = np.broadcast_to(
        np.asarray(microphone_enu, dtype=float), observer_time.shape + (3,)
    )
    emission, valid = solve_emission_time(path, observer_time, microphone_enu, atmosphere)
    position, velocity = path.state(emission)
    offset = microphone_enu - position
    distance = np.maximum(np.linalg.norm(offset, axis=-1), 1e-3)
    unit = offset / distance[..., None]
    wind = np.array([atmosphere.wind_east_mps, atmosphere.wind_north_mps, 0.0])
    along_wind = unit @ wind
    c_source = atmosphere.sound_speed(position[..., 2]) + along_wind
    c_path = atmosphere.path_sound_speed(position[..., 2], microphone_enu[..., 2]) + along_wind
    radial = np.sum(velocity * unit, axis=-1)
    denominator = c_source - radial
    subsonic = denominator > 0.02 * c_source
    factor = c_source / np.where(subsonic, denominator, c_source)
    sigma_c = atmosphere.sound_speed_sigma(position[..., 2])
    sigma_wind = atmosphere.wind_sigma_mps * np.hypot(unit[..., 0], unit[..., 1])
    sigma_medium = np.hypot(sigma_c, sigma_wind)
    sigma_log = np.abs(radial) / (c_source * denominator.clip(1.0)) * sigma_medium
    return DopplerPrediction(
        emission_time=emission,
        distance_m=distance,
        delay_s=distance / c_path,
        radial_velocity_mps=radial,
        doppler_factor=factor,
        log_factor_sigma=sigma_log,
        valid=valid & subsonic,
    )


@dataclass(frozen=True)
class ClosestApproach:
    observer_time: float
    emission_time: float
    slant_distance_m: float
    horizontal_distance_m: float
    altitude_above_microphone_m: float


def closest_approach(
    path: KinematicPath,
    microphone_enu: FloatArray,
    atmosphere: Atmosphere,
    step_s: float = 0.25,
) -> ClosestApproach:
    times = np.arange(path.t_start, path.t_end + step_s, step_s)
    position, _ = path.state(times)
    distance = np.linalg.norm(position - microphone_enu, axis=-1)
    index = int(np.argmin(distance))
    emission = float(times[index])
    best = position[index]
    delay = float(
        distance[index] / atmosphere.path_sound_speed(best[2], microphone_enu[2])
    )
    return ClosestApproach(
        observer_time=emission + delay,
        emission_time=emission,
        slant_distance_m=float(distance[index]),
        horizontal_distance_m=float(np.hypot(*(best[:2] - microphone_enu[:2]))),
        altitude_above_microphone_m=float(best[2] - microphone_enu[2]),
    )
