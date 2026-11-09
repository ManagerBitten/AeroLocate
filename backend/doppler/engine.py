from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from backend.doppler.curve_fit import (
    CurveClass,
    PassbyFit,
    classify_curve,
    curve_quality,
    fit_passby,
)
from backend.doppler.kalman import smooth_track
from backend.doppler.ridge import RidgeTrack, extract_ridges, whitened_spectrogram

FloatArray = NDArray[np.float64]
ANALYSIS_RATE_HZ = 8000
WINDOWS_S = (0.35, 0.7, 1.4)
HOP_FRACTION = 0.25
FMIN_HZ = 25.0
FMAX_HZ = 3500.0
MIN_OVERLAP_S = 3.0
COMMON_FATE_TOLERANCE = 0.0035
DUPLICATE_LOG_TOLERANCE = 0.01
DUPLICATE_RESOLUTIONS = 3.0
SIDELOBE_RESOLUTIONS = 8.0
SIDELOBE_SNR_GAP_DB = 12.0
MIN_USABLE_QUALITY = 0.03
FRAME_AGREEMENT_LOG = 0.006
TURBULENCE_LOG_SIGMA = 2e-4
TRIM_PASSES = 3
MAX_CROSSING_HARMONIC = 16.0
CROSSING_LOG_TOLERANCE = 0.006
CROSSING_MIN_FRACTION = 0.5
TRIM_SCALES = 4.0
TRIM_SWING_FRACTION = 0.03
MIN_TRIM_FRACTION = 0.6
HARMONIC_LIMIT = 40


@dataclass(frozen=True)
class DopplerCurve:
    event_id: str
    times: FloatArray
    log_frequency: FloatArray
    sigma_log: FloatArray
    harmonic_numbers: tuple[int, ...]
    component_frequencies_hz: tuple[float, ...]
    window_s: float
    median_snr_db: float
    curve_class: CurveClass
    quality: float
    fit: PassbyFit | None
    noise_inflation: float
    max_log_slope: float
    fit_residuals: FloatArray = field(repr=False, default_factory=lambda: np.zeros(0))
    raw_tracks: tuple[RidgeTrack, ...] = field(repr=False, default=())

    @property
    def usable(self) -> bool:
        return self.quality >= MIN_USABLE_QUALITY and self.curve_class in (
            CurveClass.DOPPLER_CONSISTENT,
            CurveClass.PARTIAL_DOPPLER,
        )

    @property
    def effective_sigma(self) -> FloatArray:
        return np.hypot(self.sigma_log * self.noise_inflation, TURBULENCE_LOG_SIGMA)

    @property
    def start(self) -> float:
        return float(self.times[0])

    @property
    def end(self) -> float:
        return float(self.times[-1])


def to_analysis_rate(audio: FloatArray, sample_rate: int) -> tuple[FloatArray, int]:
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    if sample_rate == ANALYSIS_RATE_HZ:
        return mono.astype(np.float64), sample_rate
    divisor = np.gcd(int(sample_rate), ANALYSIS_RATE_HZ)
    if sample_rate < ANALYSIS_RATE_HZ:
        return mono.astype(np.float64), sample_rate
    resampled = signal.resample_poly(mono, ANALYSIS_RATE_HZ // divisor, int(sample_rate) // divisor)
    return resampled.astype(np.float64), ANALYSIS_RATE_HZ


def _overlap(a: RidgeTrack, b: RidgeTrack) -> tuple[FloatArray, FloatArray, FloatArray] | None:
    start, end = max(a.start, b.start), min(a.end, b.end)
    if end - start < MIN_OVERLAP_S:
        return None
    grid = np.arange(start, end, min(a.times[1] - a.times[0], b.times[1] - b.times[0]))
    return grid, np.interp(grid, a.times, a.frequency_hz), np.interp(grid, b.times, b.frequency_hz)


def _deduplicate(tracks: list[RidgeTrack]) -> list[RidgeTrack]:
    kept: list[RidgeTrack] = []
    for track in sorted(tracks, key=lambda item: item.energy, reverse=True):
        duplicate = False
        for other in kept:
            shared = _overlap(track, other)
            if shared is None:
                continue
            _, f_a, f_b = shared
            resolution = max(1.0 / track.window_s, 1.0 / other.window_s)
            tolerance = max(
                DUPLICATE_LOG_TOLERANCE,
                DUPLICATE_RESOLUTIONS * resolution / float(np.median(f_a)),
            )
            separation = abs(float(np.median(np.log(f_a) - np.log(f_b))))
            sidelobe_tolerance = SIDELOBE_RESOLUTIONS * resolution / float(np.median(f_a))
            much_weaker = float(np.median(other.snr_db)) - float(np.median(track.snr_db)) > SIDELOBE_SNR_GAP_DB
            if separation < tolerance or (much_weaker and separation < sidelobe_tolerance):
                duplicate = True
                break
        if not duplicate:
            kept.append(track)
    return kept


def _common_fate(a: RidgeTrack, b: RidgeTrack) -> bool:
    shared = _overlap(a, b)
    if shared is None:
        return False
    grid, f_a, f_b = shared
    ratio = np.log(f_a) - np.log(f_b)
    combined = np.interp(grid, a.times, a.sigma_hz) / f_a + np.interp(grid, b.times, b.sigma_hz) / f_b
    tolerance = max(COMMON_FATE_TOLERANCE, 1.5 * float(np.median(combined)))
    variation = 1.4826 * float(np.median(np.abs(ratio - np.median(ratio))))
    shift = float(np.ptp(np.log(f_a)))
    return variation < tolerance and shift > variation


def _group_tracks(tracks: list[RidgeTrack]) -> list[list[RidgeTrack]]:
    groups: list[list[RidgeTrack]] = []
    for track in sorted(tracks, key=lambda item: item.energy, reverse=True):
        home = next((group for group in groups if _common_fate(track, group[0])), None)
        if home is None:
            groups.append([track])
        else:
            home.append(track)
    return groups


def _assign_harmonics(medians: list[float]) -> tuple[int, ...]:
    lowest = min(medians)
    best_error = np.inf
    best: tuple[int, ...] = tuple(1 for _ in medians)
    for divisor in range(1, 13):
        fundamental = lowest / divisor
        numbers = [max(1, int(round(value / fundamental))) for value in medians]
        if max(numbers) > HARMONIC_LIMIT:
            continue
        error = max(abs(value / (n * fundamental) - 1.0) for value, n in zip(medians, numbers))
        if error + 1e-3 * divisor < best_error:
            best_error = error + 1e-3 * divisor
            best = tuple(numbers)
    return best


def _merge_group(group: list[RidgeTrack]) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    start = min(track.start for track in group)
    end = max(track.end for track in group)
    hop = min(float(track.times[1] - track.times[0]) for track in group)
    grid = np.arange(start, end + hop / 2, hop)
    anchor = max(group, key=lambda item: item.energy)
    values: list[FloatArray] = []
    sigmas: list[FloatArray] = []
    levels: list[FloatArray] = []
    for track in group:
        if track is anchor:
            offset = 0.0
        else:
            shared = _overlap(track, anchor)
            if shared is None:
                continue
            _, f_track, f_anchor = shared
            offset = float(np.median(np.log(f_track) - np.log(f_anchor)))
        inside = (grid >= track.start) & (grid <= track.end)
        values.append(np.where(inside, np.interp(grid, track.times, np.log(track.frequency_hz)) - offset, np.nan))
        sigmas.append(np.interp(grid, track.times, track.sigma_hz / track.frequency_hz))
        levels.append(np.where(inside, np.interp(grid, track.times, track.snr_db), 0.0))
    stack = np.vstack(values)
    sigma_stack = np.vstack(sigmas)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        centre = np.nanmedian(stack, axis=0)
    agrees = np.abs(stack - centre) < FRAME_AGREEMENT_LOG
    weights = np.where(agrees & np.isfinite(stack), 1.0 / sigma_stack**2, 0.0)
    denominator = weights.sum(axis=0)
    covered = denominator > 0
    numerator = np.sum(weights * np.nan_to_num(stack), axis=0)
    snr = np.max(np.vstack(levels), axis=0)
    return grid[covered], numerator[covered] / denominator[covered], np.sqrt(1.0 / denominator[covered]), snr[covered]


def _noise_inflation(residuals: FloatArray, sigma: FloatArray) -> float:
    if len(residuals) < 8:
        return 2.0
    standardized = residuals / sigma
    variance = float(np.mean(standardized**2))
    first = standardized[:-1] - standardized[:-1].mean()
    second = standardized[1:] - standardized[1:].mean()
    denominator = float(np.sqrt(np.sum(first**2) * np.sum(second**2)))
    correlation = float(np.sum(first * second) / denominator) if denominator > 0 else 0.0
    correlation = float(np.clip(correlation, 0.0, 0.9))
    factor = (1.0 + correlation) / (1.0 - correlation)
    return float(np.sqrt(max(1.0, variance) * factor))


def _trimmed_fit(
    times: FloatArray, values: FloatArray, sigmas: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, PassbyFit | None]:
    fit = fit_passby(times, values, sigmas)
    best = (times, values, sigmas, fit)
    original = len(times)
    for _ in range(TRIM_PASSES):
        if fit is None:
            break
        residuals = fit.residuals
        scale = max(1.4826 * float(np.median(np.abs(residuals - np.median(residuals)))), float(np.median(sigmas)))
        limit = max(TRIM_SCALES * scale, TRIM_SWING_FRACTION * float(np.ptp(values)))
        keep = np.abs(residuals) <= limit
        if keep.all() or keep.sum() < max(8, MIN_TRIM_FRACTION * original):
            break
        times, values, sigmas = times[keep], values[keep], sigmas[keep]
        fit = fit_passby(times, values, sigmas)
        best = (times, values, sigmas, fit)
    return best


def _build_curve(event_id: str, group: list[RidgeTrack]) -> DopplerCurve | None:
    grid, merged, sigma, snr = _merge_group(group)
    if len(grid) < 8:
        return None
    window = float(np.median([track.window_s for track in group]))
    smoothed = smooth_track(grid, merged, sigma)
    step = max(1, int(round(window / (grid[1] - grid[0]))))
    keep = np.zeros(len(grid), dtype=bool)
    keep[::step] = True
    keep &= smoothed.inlier
    if int(keep.sum()) < 8:
        return None
    times = grid[keep]
    values = smoothed.log_frequency[keep]
    sigmas = np.maximum(smoothed.sigma_log[keep], sigma[keep] / np.sqrt(step))
    times, values, sigmas, fit = _trimmed_fit(times, values, sigmas)
    curve_class = classify_curve(times, values, sigmas, fit)
    inflation = _noise_inflation(fit.residuals, sigmas) if fit else 1.0
    effective = sigmas * inflation
    quality = curve_quality(times, values, effective, fit, curve_class)
    medians = [float(np.median(track.frequency_hz)) for track in group]
    return DopplerCurve(
        event_id=event_id,
        times=times,
        log_frequency=values,
        sigma_log=sigmas,
        harmonic_numbers=_assign_harmonics(medians),
        component_frequencies_hz=tuple(medians),
        window_s=window,
        median_snr_db=float(np.median(snr)),
        curve_class=curve_class,
        quality=quality,
        fit=fit,
        noise_inflation=inflation,
        max_log_slope=float(np.max(np.abs(np.gradient(values, times)))),
        fit_residuals=fit.residuals if fit else np.zeros(0),
        raw_tracks=tuple(group),
    )


def _on_harmonics_of(curve: DopplerCurve, reference: DopplerCurve) -> NDArray[np.bool_]:
    lowest_ratio = min(reference.component_frequencies_hz) / reference.component_frequencies_hz[0]
    inside = (curve.times >= reference.times[0]) & (curve.times <= reference.times[-1])
    base = np.exp(np.interp(curve.times, reference.times, reference.log_frequency)) * lowest_ratio
    frequency = np.exp(curve.log_frequency)
    ratio = frequency / base
    nearest = np.clip(np.round(ratio), 1.0, MAX_CROSSING_HARMONIC)
    tolerance = np.maximum(CROSSING_LOG_TOLERANCE, 2.0 / (curve.window_s * frequency))
    return inside & (np.abs(ratio / nearest - 1.0) < tolerance)


def _flag_crossing_artifacts(curves: list[DopplerCurve]) -> list[DopplerCurve]:
    confirmed = [curve for curve in curves if curve.usable and len(curve.raw_tracks) >= 2]
    flagged: list[DopplerCurve] = []
    for curve in curves:
        if curve.usable and len(curve.raw_tracks) == 1 and confirmed:
            covered = np.zeros(len(curve.times), dtype=bool)
            for reference in confirmed:
                covered |= _on_harmonics_of(curve, reference)
            if float(covered.mean()) >= CROSSING_MIN_FRACTION:
                curve = replace(curve, curve_class=CurveClass.CROSSING_ARTIFACT, quality=0.0)
        flagged.append(curve)
    return flagged


def extract_doppler_curves(
    audio: FloatArray,
    sample_rate: int,
    progress: Callable[[float], None] | None = None,
) -> list[DopplerCurve]:
    data, rate = to_analysis_rate(audio, sample_rate)
    fmax = min(FMAX_HZ, rate / 2.0 * 0.95)
    tracks: list[RidgeTrack] = []
    for index, window in enumerate(WINDOWS_S):
        if len(data) < 2 * window * rate:
            continue
        spectrogram = whitened_spectrogram(data, rate, window, window * HOP_FRACTION, FMIN_HZ, fmax)
        tracks.extend(extract_ridges(spectrogram))
        if progress:
            progress((index + 1) / (len(WINDOWS_S) + 1))
    groups = _group_tracks(_deduplicate(tracks))
    curves: list[DopplerCurve] = []
    for group in groups:
        curve = _build_curve(f"E{len(curves) + 1:02d}", group)
        if curve is not None:
            curves.append(curve)
    curves = _flag_crossing_artifacts(curves)
    curves.sort(key=lambda item: item.start)
    return [_renamed(curve, f"E{i + 1:02d}") for i, curve in enumerate(curves)]


def _renamed(curve: DopplerCurve, event_id: str) -> DopplerCurve:
    return replace(curve, event_id=event_id)
