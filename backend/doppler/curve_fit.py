from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

FloatArray = NDArray[np.float64]
REFERENCE_SOUND_SPEED = 340.0
MIN_SPEED_MPS = 15.0
MAX_SPEED_MPS = 320.0
MIN_DISTANCE_M = 30.0
MAX_DISTANCE_M = 40000.0
MIN_BIC_ADVANTAGE = 10.0
MAX_MAINS_HARMONIC = 12
MAX_RELATIVE_FIT_ERROR = 0.015


class CurveClass(str, Enum):
    DOPPLER_CONSISTENT = "doppler_consistent"
    PARTIAL_DOPPLER = "partial_doppler"
    NO_MEASURABLE_SHIFT = "no_measurable_shift"
    RISING_FREQUENCY = "rising_frequency"
    MAINS_HUM = "mains_hum"
    CROSSING_ARTIFACT = "crossing_artifact"
    INCONSISTENT = "inconsistent"


def passby_log_factor(
    t: FloatArray, closest_time: float, speed: float, distance: float, c: float = REFERENCE_SOUND_SPEED
) -> FloatArray:
    u = t - closest_time + distance / c
    c2 = c * c
    v2 = speed * speed
    discriminant = c2 * c2 * u * u - (c2 - v2) * (c2 * u * u - distance * distance)
    s = (c2 * u - np.sqrt(np.maximum(discriminant, 0.0))) / (c2 - v2)
    radial = -v2 * s / np.sqrt(distance * distance + v2 * s * s)
    return -np.log1p(-radial / c)


@dataclass(frozen=True)
class PassbyFit:
    closest_time: float
    speed_mps: float
    distance_m: float
    log_source_frequency: float
    closest_time_sigma: float
    speed_sigma: float
    distance_sigma: float
    chi2: float
    dof: int
    residuals: FloatArray
    bic_constant: float
    bic_linear: float
    bic_passby: float

    @property
    def source_frequency_hz(self) -> float:
        return float(np.exp(self.log_source_frequency))

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / max(self.dof, 1)


def _weighted_mean(values: FloatArray, weights: FloatArray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _bic(chi2: float, parameters: int, samples: int) -> float:
    return chi2 + parameters * float(np.log(samples))


def fit_passby(times: FloatArray, log_frequency: FloatArray, sigma_log: FloatArray) -> PassbyFit | None:
    if len(times) < 8:
        return None
    weights = 1.0 / sigma_log**2
    constant_mean = _weighted_mean(log_frequency, weights)
    chi2_constant = float(np.sum(weights * (log_frequency - constant_mean) ** 2))
    design = np.vstack([np.ones_like(times), times - times.mean()]).T
    root = np.sqrt(weights)
    linear, *_ = np.linalg.lstsq(design * root[:, None], log_frequency * root, rcond=None)
    chi2_linear = float(np.sum(weights * (log_frequency - design @ linear) ** 2))

    def residual(theta: FloatArray) -> FloatArray:
        closest, log_speed, log_distance = theta
        model = passby_log_factor(times, closest, float(np.exp(log_speed)), float(np.exp(log_distance)))
        offset = _weighted_mean(log_frequency - model, weights)
        return (log_frequency - model - offset) / sigma_log

    slope = np.gradient(log_frequency, times)
    span = float(times[-1] - times[0])
    seeds_time = {float(times[int(np.argmin(slope))]), float(times.mean()), float(times[0] + 0.25 * span), float(times[0] + 0.75 * span)}
    lower = np.array([times[0] - span, np.log(MIN_SPEED_MPS), np.log(MIN_DISTANCE_M)])
    upper = np.array([times[-1] + span, np.log(MAX_SPEED_MPS), np.log(MAX_DISTANCE_M)])
    best = None
    for seed_time in seeds_time:
        for seed_distance in (300.0, 1200.0, 4000.0, 10000.0):
            for seed_speed in (70.0, 180.0):
                start = np.array([seed_time, np.log(seed_speed), np.log(seed_distance)])
                result = least_squares(residual, start, bounds=(lower, upper), loss="cauchy", f_scale=3.0, max_nfev=60)
                if best is None or result.cost < best.cost:
                    best = result
    assert best is not None
    closest, log_speed, log_distance = best.x
    speed, distance = float(np.exp(log_speed)), float(np.exp(log_distance))
    plain = residual(best.x)
    chi2 = float(np.sum(plain**2))
    jacobian = best.jac
    information = jacobian.T @ jacobian
    try:
        covariance = np.linalg.pinv(information + 1e-9 * np.eye(3))
    except np.linalg.LinAlgError:
        covariance = np.full((3, 3), np.inf)
    deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    model = passby_log_factor(times, closest, speed, distance)
    samples = len(times)
    return PassbyFit(
        closest_time=float(closest),
        speed_mps=speed,
        distance_m=distance,
        log_source_frequency=_weighted_mean(log_frequency - model, weights),
        closest_time_sigma=float(deviations[0]),
        speed_sigma=float(speed * deviations[1]),
        distance_sigma=float(distance * deviations[2]),
        chi2=chi2,
        dof=max(samples - 4, 1),
        residuals=plain * sigma_log,
        bic_constant=_bic(chi2_constant, 1, samples),
        bic_linear=_bic(chi2_linear, 2, samples),
        bic_passby=_bic(chi2, 4, samples),
    )


def is_mains_frequency(frequency_hz: float, tolerance: float = 0.004) -> bool:
    for base in (50.0, 60.0):
        ratio = frequency_hz / base
        nearest = round(ratio)
        if 1 <= nearest <= MAX_MAINS_HARMONIC and abs(ratio - nearest) / nearest < tolerance:
            return True
    return False


def fit_at_bounds(fit: PassbyFit) -> bool:
    return (
        fit.speed_mps <= MIN_SPEED_MPS * 1.02
        or fit.speed_mps >= MAX_SPEED_MPS * 0.98
        or fit.distance_m <= MIN_DISTANCE_M * 1.02
        or fit.distance_m >= MAX_DISTANCE_M * 0.98
    )


def classify_curve(
    times: FloatArray,
    log_frequency: FloatArray,
    sigma_log: FloatArray,
    fit: PassbyFit | None,
) -> CurveClass:
    if fit is None:
        return CurveClass.NO_MEASURABLE_SHIFT
    weights = 1.0 / sigma_log**2
    centred = times - np.sum(times * weights) / np.sum(weights)
    slope = float(np.sum(weights * centred * log_frequency) / np.sum(weights * centred**2))
    slope_sigma = float(1.0 / np.sqrt(np.sum(weights * centred**2)))
    span = float(np.ptp(log_frequency))
    typical_sigma = float(np.median(sigma_log))
    flat = fit.bic_constant <= min(fit.bic_linear, fit.bic_passby) or span < 4.0 * typical_sigma
    if flat:
        if is_mains_frequency(float(np.exp(np.median(log_frequency)))):
            return CurveClass.MAINS_HUM
        return CurveClass.NO_MEASURABLE_SHIFT
    if slope > 3.0 * slope_sigma:
        return CurveClass.RISING_FREQUENCY
    if fit_at_bounds(fit):
        return CurveClass.INCONSISTENT
    passby_wins = fit.bic_passby < fit.bic_linear - MIN_BIC_ADVANTAGE
    rms_residual = float(np.sqrt(np.mean(fit.residuals**2)))
    good_fit = fit.reduced_chi2 < 6.0 or rms_residual < MAX_RELATIVE_FIT_ERROR * span
    inside = times[0] <= fit.closest_time <= times[-1]
    if passby_wins and good_fit and inside:
        return CurveClass.DOPPLER_CONSISTENT
    if good_fit and slope < -3.0 * slope_sigma:
        return CurveClass.PARTIAL_DOPPLER
    return CurveClass.INCONSISTENT


def curve_quality(
    times: FloatArray,
    log_frequency: FloatArray,
    sigma_log: FloatArray,
    fit: PassbyFit | None,
    curve_class: CurveClass,
) -> float:
    if fit is None or curve_class not in (CurveClass.DOPPLER_CONSISTENT, CurveClass.PARTIAL_DOPPLER):
        return 0.0
    span = float(np.ptp(log_frequency))
    contrast = span / max(float(np.median(sigma_log)), 1e-9)
    before = float(np.mean(times < fit.closest_time))
    balance = 1.0 - 2.0 * abs(before - 0.5)
    closest_precision = 1.0 / (1.0 + fit.closest_time_sigma)
    distance_precision = 1.0 / (1.0 + fit.distance_sigma / max(fit.distance_m, 1.0) * 4.0)
    rms_residual = float(np.sqrt(np.mean(fit.residuals**2)))
    fit_term = 1.0 / (1.0 + 40.0 * rms_residual / max(span, 1e-9))
    return float(
        np.clip(
            (1.0 - np.exp(-contrast / 30.0)) * (0.3 + 0.7 * balance) * closest_precision ** 0.5 * distance_precision ** 0.5 * fit_term,
            0.0,
            1.0,
        )
    )
