from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from backend.core.atmosphere import Atmosphere
from backend.geolocation.likelihood import (
    DRIFT_PRIOR_SIGMA_PER_S,
    EventData,
    amplitude_scale_sigma,
)
from backend.geolocation.search import TrackedFlight
from backend.propagation.simulator import predict_doppler

FloatArray = NDArray[np.float64]
MONTE_CARLO_DRAWS = 24


@dataclass(frozen=True)
class RefinedSolution:
    x_m: float
    y_m: float
    start: float
    sigma_x_m: float
    sigma_y_m: float
    sigma_start_s: float
    source_frequencies_hz: dict[str, float]
    reduced_chi2: float
    converged: bool


def _residuals(
    theta: FloatArray,
    associations: Sequence[tuple[EventData, TrackedFlight]],
    elevation: float,
    atmosphere: Atmosphere,
    fixed_start: float | None,
    start_reference: float,
) -> FloatArray:
    x, y = theta[0], theta[1]
    start = fixed_start if fixed_start is not None else start_reference + theta[2]
    output: list[FloatArray] = []
    for event, flight in associations:
        curve = event.curve
        observer = start + curve.times
        prediction = predict_doppler(flight.trajectory, observer, np.array([x, y, elevation]), atmosphere)
        model = np.log(prediction.doppler_factor)
        residual = curve.log_frequency - model
        sigma = curve.effective_sigma
        weights = 1.0 / sigma**2
        scale_sigma = float(amplitude_scale_sigma(prediction.log_factor_sigma, model))
        basis = np.column_stack([np.ones_like(residual), event.centred_times, model - model.mean()])
        priors = np.diag([0.0, 1.0 / DRIFT_PRIOR_SIGMA_PER_S**2, 1.0 / scale_sigma**2])
        normal = basis.T @ (weights[:, None] * basis) + priors
        coefficients = np.linalg.solve(normal, basis.T @ (weights * residual))
        output.append((residual - basis @ coefficients) / sigma)
    return np.concatenate(output)


def refine_solution(
    associations: Sequence[tuple[EventData, TrackedFlight]],
    x0: float,
    y0: float,
    start0: float,
    elevation: float,
    atmosphere: Atmosphere,
    start_is_known: bool,
) -> RefinedSolution:
    fixed = start0 if start_is_known else None
    theta0 = np.array([x0, y0] if start_is_known else [x0, y0, 0.0])
    result = least_squares(
        _residuals,
        theta0,
        args=(associations, elevation, atmosphere, fixed, start0),
        loss="soft_l1",
        f_scale=2.0,
        x_scale=np.array([100.0, 100.0] if start_is_known else [100.0, 100.0, 1.0]),
        max_nfev=40,
    )
    plain = _residuals(result.x, associations, elevation, atmosphere, fixed, start0)
    information = result.jac.T @ result.jac
    covariance = np.linalg.pinv(information + 1e-9 * np.eye(len(theta0)))
    deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    frequencies: dict[str, float] = {}
    start = start0 if start_is_known else float(start0 + result.x[2])
    for event, flight in associations:
        curve = event.curve
        prediction = predict_doppler(
            flight.trajectory, start + curve.times, np.array([result.x[0], result.x[1], elevation]), atmosphere
        )
        frequencies[event.event_id] = float(
            np.exp(np.median(curve.log_frequency - np.log(prediction.doppler_factor)))
        )
    dof = max(len(plain) - len(theta0) - len(associations), 1)
    return RefinedSolution(
        x_m=float(result.x[0]),
        y_m=float(result.x[1]),
        start=start,
        sigma_x_m=float(deviations[0]),
        sigma_y_m=float(deviations[1]),
        sigma_start_s=0.0 if start_is_known else float(deviations[2]),
        source_frequencies_hz=frequencies,
        reduced_chi2=float(np.sum(plain**2) / dof),
        converged=bool(result.success),
    )


def monte_carlo_sensitivity(
    associations: Sequence[tuple[EventData, TrackedFlight]],
    solution: RefinedSolution,
    elevation: float,
    atmosphere: Atmosphere,
    start_is_known: bool,
    seed: int = 0,
) -> float:
    generator = np.random.default_rng(seed)
    scatter: list[tuple[float, float]] = []
    for _ in range(MONTE_CARLO_DRAWS):
        perturbed_atmosphere = replace(
            atmosphere,
            surface_temperature_c=atmosphere.surface_temperature_c
            + generator.normal(0.0, atmosphere.temperature_sigma_k),
            wind_east_mps=atmosphere.wind_east_mps + generator.normal(0.0, atmosphere.wind_sigma_mps),
            wind_north_mps=atmosphere.wind_north_mps + generator.normal(0.0, atmosphere.wind_sigma_mps),
        )
        perturbed: list[tuple[EventData, TrackedFlight]] = []
        for event, flight in associations:
            trajectory = flight.trajectory
            shifted = replace(
                trajectory,
                positions=trajectory.positions
                + generator.normal(0.0, 1.0, 3) * np.array([30.0, 30.0, float(np.median(trajectory.altitude_sigma_m))]),
            )
            perturbed.append((event, TrackedFlight(flight.key, shifted)))
        draw = refine_solution(
            perturbed,
            solution.x_m,
            solution.y_m,
            solution.start,
            elevation + generator.normal(0.0, 15.0),
            perturbed_atmosphere,
            start_is_known,
        )
        scatter.append((draw.x_m - solution.x_m, draw.y_m - solution.y_m))
    values = np.array(scatter)
    return float(np.sqrt(np.mean(np.sum(values**2, axis=1)) / 2.0))
