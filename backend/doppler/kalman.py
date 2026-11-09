from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
GATE_CHI2 = 11.0
HAMPEL_WINDOW = 7
HAMPEL_SIGMAS = 4.0


@dataclass(frozen=True)
class SmoothedTrack:
    times: FloatArray
    log_frequency: FloatArray
    sigma_log: FloatArray
    inlier: NDArray[np.bool_]


def hampel_inliers(values: FloatArray, sigma: FloatArray) -> NDArray[np.bool_]:
    keep = np.ones(len(values), dtype=bool)
    half = HAMPEL_WINDOW // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        window = values[lo:hi]
        median = np.median(window)
        spread = 1.4826 * np.median(np.abs(window - median))
        if abs(values[i] - median) > HAMPEL_SIGMAS * max(spread, 2.0 * sigma[i]):
            keep[i] = False
    return keep


def smooth_track(
    times: FloatArray,
    log_frequency: FloatArray,
    sigma_log: FloatArray,
    acceleration_sigma: float = 0.04,
) -> SmoothedTrack:
    count = len(times)
    inlier = hampel_inliers(log_frequency, sigma_log)
    first = int(np.argmax(inlier))
    state = np.array([log_frequency[first], 0.0])
    cov = np.diag([sigma_log[first] ** 2, 0.2**2])
    measurement = np.array([[1.0, 0.0]])
    filtered_state = np.zeros((count, 2))
    filtered_cov = np.zeros((count, 2, 2))
    predicted_state = np.zeros((count, 2))
    predicted_cov = np.zeros((count, 2, 2))
    transitions = np.zeros((count, 2, 2))
    previous = times[0]
    for k in range(count):
        dt = times[k] - previous
        previous = times[k]
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        process = acceleration_sigma**2 * np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]]
        )
        state = transition @ state
        cov = transition @ cov @ transition.T + process
        transitions[k] = transition
        predicted_state[k] = state
        predicted_cov[k] = cov
        if inlier[k]:
            innovation = log_frequency[k] - state[0]
            innovation_var = cov[0, 0] + sigma_log[k] ** 2
            if innovation**2 / innovation_var > GATE_CHI2:
                inlier[k] = False
            else:
                gain = cov[:, 0] / innovation_var
                state = state + gain * innovation
                cov = (np.eye(2) - np.outer(gain, measurement)) @ cov
        filtered_state[k] = state
        filtered_cov[k] = cov
    smoothed_state = filtered_state.copy()
    smoothed_cov = filtered_cov.copy()
    for k in range(count - 2, -1, -1):
        gain = filtered_cov[k] @ transitions[k + 1].T @ np.linalg.inv(predicted_cov[k + 1])
        smoothed_state[k] = filtered_state[k] + gain @ (smoothed_state[k + 1] - predicted_state[k + 1])
        smoothed_cov[k] = (
            filtered_cov[k] + gain @ (smoothed_cov[k + 1] - predicted_cov[k + 1]) @ gain.T
        )
    return SmoothedTrack(
        times=times,
        log_frequency=smoothed_state[:, 0],
        sigma_log=np.sqrt(np.maximum(smoothed_cov[:, 0, 0], 1e-14)),
        inlier=inlier,
    )
