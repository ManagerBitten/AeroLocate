from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from backend.core.atmosphere import Atmosphere
from backend.doppler.curve_fit import CurveClass, passby_log_factor
from backend.doppler.engine import DopplerCurve
from backend.flight_data.reconstruction.trajectory import HermiteTrajectory
from backend.propagation.simulator import predict_doppler

FloatArray = NDArray[np.float64]
DRIFT_PRIOR_SIGMA_PER_S = 0.0006
MIN_TRACK_COVERAGE = 0.85
INVALID_LOG_LIKELIHOOD = -np.inf
CHUNK_POINTS = 2_000_000
DEFAULT_SPEED_MPS = 120.0
TRAJECTORY_SPEED_SCALE_SIGMA = 0.012
REFERENCE_SOUND_SPEED_MPS = 340.0
SPEED_MATCH_SIGMA = 0.03
TRACK_ERROR_CORRELATION_S = 20.0
DEFAULT_DISTANCE_M = 1500.0
SPEED_PRIOR_LOG_RANGE = float(np.log(260.0 / 40.0))
DISTANCE_PRIOR_LOG_RANGE = float(np.log(25000.0 / 50.0))


@dataclass(frozen=True)
class EventData:
    curve: DopplerCurve
    centred_times: FloatArray
    log_slope: FloatArray
    speed: float
    distance: float
    speed_ratio: float

    @property
    def event_id(self) -> str:
        return self.curve.event_id


def prepare_event(curve: DopplerCurve) -> EventData:
    fit = curve.fit
    speed = fit.speed_mps if fit else DEFAULT_SPEED_MPS
    distance = fit.distance_m if fit else DEFAULT_DISTANCE_M
    if fit:
        model = passby_log_factor(curve.times, fit.closest_time, speed, distance)
        slope = np.abs(np.gradient(model, curve.times))
    else:
        slope = np.abs(np.gradient(curve.log_frequency, curve.times))
    return EventData(
        curve=curve,
        centred_times=curve.times - curve.times.mean(),
        log_slope=slope,
        speed=speed,
        distance=distance,
        speed_ratio=speed / 340.0,
    )


def tempering_sigma(event: EventData, cell_m: float, time_step_s: float) -> FloatArray:
    time_blur = 0.5 * time_step_s + 0.5 * cell_m / max(event.speed, 10.0)
    cross_blur = 0.5 * event.speed_ratio * min(1.0, cell_m / max(event.distance, 1.0))
    return np.hypot(event.log_slope * time_blur, cross_blur)


def event_log_likelihood(
    event: EventData,
    trajectory: HermiteTrajectory,
    microphone_xy: FloatArray,
    microphone_z: float,
    recording_start: FloatArray,
    atmosphere: Atmosphere,
    temper: FloatArray,
) -> FloatArray:
    total = len(recording_start)
    result = np.full(total, INVALID_LOG_LIKELIHOOD)
    if total == 0:
        return result
    samples = len(event.curve.times)
    chunk = max(1, CHUNK_POINTS // samples)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        result[start:stop] = _chunk_likelihood(
            event, trajectory, microphone_xy[start:stop], microphone_z, recording_start[start:stop], atmosphere, temper
        )
    return result


def _chunk_likelihood(
    event: EventData,
    trajectory: HermiteTrajectory,
    microphone_xy: FloatArray,
    microphone_z: float,
    recording_start: FloatArray,
    atmosphere: Atmosphere,
    temper: FloatArray,
) -> FloatArray:
    curve = event.curve
    observer = recording_start[:, None] + curve.times[None, :]
    inside = (observer > trajectory.t_start) & (observer < trajectory.t_end)
    coverage = inside.mean(axis=1)
    microphone = np.zeros(observer.shape + (3,))
    microphone[..., 0] = microphone_xy[:, 0, None]
    microphone[..., 1] = microphone_xy[:, 1, None]
    microphone[..., 2] = microphone_z
    prediction = predict_doppler(trajectory, observer, microphone, atmosphere)
    usable = inside & prediction.valid
    usable_fraction = usable.mean(axis=1)
    model = np.log(prediction.doppler_factor)
    residual = curve.log_frequency[None, :] - model
    variance = curve.effective_sigma[None, :] ** 2 + temper[None, :] ** 2
    weights = np.where(usable, 1.0 / variance, 0.0)
    scale_sigma = amplitude_scale_sigma(prediction.log_factor_sigma, model)
    chi2 = profiled_chi2(residual, weights, event.centred_times[None, :], model, scale_sigma)
    log_likelihood = -0.5 * chi2 + speed_match_log_factor(event, trajectory, recording_start)
    log_likelihood[(coverage < MIN_TRACK_COVERAGE) | (usable_fraction < MIN_TRACK_COVERAGE)] = INVALID_LOG_LIKELIHOOD
    return log_likelihood


def speed_match_log_factor(
    event: EventData, trajectory: HermiteTrajectory, recording_start: FloatArray
) -> FloatArray | float:
    curve = event.curve
    fit = curve.fit
    if fit is None or curve.curve_class is not CurveClass.DOPPLER_CONSISTENT:
        return 0.0
    emission = recording_start + fit.closest_time - fit.distance_m / REFERENCE_SOUND_SPEED_MPS
    _, velocity = trajectory.state(emission)
    track_speed = np.maximum(np.linalg.norm(velocity, axis=-1), 1.0)
    measured_sigma = _fitted_log_speed_sigma(event)
    total_sigma = float(np.hypot(measured_sigma, SPEED_MATCH_SIGMA))
    difference = np.log(fit.speed_mps / track_speed)
    return -0.5 * (difference / total_sigma) ** 2 + float(np.log(measured_sigma / total_sigma))


def _fitted_log_speed_sigma(event: EventData) -> float:
    curve = event.curve
    fit = curve.fit
    assert fit is not None
    return max(fit.speed_sigma / fit.speed_mps * curve.noise_inflation, 1e-3)


def amplitude_scale_sigma(log_factor_sigma: FloatArray, model: FloatArray) -> FloatArray:
    amplitude = np.maximum(np.max(np.abs(model), axis=-1), 1e-6)
    medium = np.max(log_factor_sigma, axis=-1) / amplitude
    return np.hypot(medium, TRAJECTORY_SPEED_SCALE_SIGMA)


def profiled_chi2(
    residual: FloatArray,
    weights: FloatArray,
    times: FloatArray,
    model: FloatArray,
    scale_sigma: FloatArray,
) -> FloatArray:
    weight_sum = np.maximum(weights.sum(axis=-1, keepdims=True), 1e-30)
    r = residual - np.sum(weights * residual, axis=-1, keepdims=True) / weight_sum
    t = times - np.sum(weights * times, axis=-1, keepdims=True) / weight_sum
    m = model - np.sum(weights * model, axis=-1, keepdims=True) / weight_sum
    tt = np.sum(weights * t * t, axis=-1) + 1.0 / DRIFT_PRIOR_SIGMA_PER_S**2
    mm = np.sum(weights * m * m, axis=-1) + 1.0 / np.maximum(scale_sigma, 1e-9) ** 2
    tm = np.sum(weights * t * m, axis=-1)
    rt = np.sum(weights * r * t, axis=-1)
    rm = np.sum(weights * r * m, axis=-1)
    determinant = tt * mm - tm * tm
    drift = (mm * rt - tm * rm) / determinant
    scale = (tt * rm - tm * rt) / determinant
    return np.sum(weights * r * r, axis=-1) - drift * rt - scale * rm


def track_log_sigma(trajectory: HermiteTrajectory, event: EventData) -> float:
    spacing = float(np.median(np.diff(event.curve.times)))
    correlated_samples = max(1.0, TRACK_ERROR_CORRELATION_S / max(spacing, 1e-3))
    return float(np.sqrt(correlated_samples)) * trajectory.velocity_sigma_mps / REFERENCE_SOUND_SPEED_MPS


def untracked_log_likelihood(event: EventData, temper: FloatArray, path_sigma: float = 0.0) -> float:
    curve = event.curve
    fit = curve.fit
    variance = curve.effective_sigma**2 + temper**2 + path_sigma**2
    expected_path_chi2 = float(np.sum(path_sigma**2 / variance))
    duration = max(float(curve.times[-1] - curve.times[0]), 1.0)
    if fit is None:
        weights = 1.0 / variance
        mean = float(np.sum(weights * curve.log_frequency) / np.sum(weights))
        return -0.5 * (float(np.sum(weights * (curve.log_frequency - mean) ** 2)) + expected_path_chi2)
    chi2 = float(np.sum(fit.residuals**2 / variance)) + expected_path_chi2
    scale = float(np.sqrt(np.mean(variance / curve.sigma_log**2)))
    sigma_time = max(fit.closest_time_sigma * scale, 0.02)
    sigma_log_speed = max(_fitted_log_speed_sigma(event), 1e-3)
    sigma_log_distance = max(fit.distance_sigma / fit.distance_m * scale, 1e-3)
    posterior_volume = (2.0 * np.pi) ** 1.5 * sigma_time * sigma_log_speed * sigma_log_distance
    prior_volume = duration * SPEED_PRIOR_LOG_RANGE * DISTANCE_PRIOR_LOG_RANGE
    occam = min(0.0, float(np.log(posterior_volume / prior_volume)))
    return -0.5 * chi2 + occam
