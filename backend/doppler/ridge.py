from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal

FloatArray = NDArray[np.float64]
LOG_BIN = 0.0015
MAX_RELATIVE_SLOPE_PER_S = 0.18
DETECTION_SNR_DB = 9.0
GAP_TOLERANCE_S = 1.2
MASK_HALF_WIDTH_LOG = 0.012
MAIN_LOBE_MASK_RESOLUTIONS = 3.0
MID_JUMP_PENALTY_DB = 4.0
FAR_JUMP_PENALTY_DB = 10.0
JITTER_WINDOW_FRAMES = 15
MAX_JITTER_FRACTION = 0.2
MAX_EMPTY_PATHS = 3


@dataclass(frozen=True)
class WhitenedSpectrogram:
    times: FloatArray
    frequencies: FloatArray
    power_db: FloatArray
    snr_db: FloatArray
    window_s: float

    @property
    def hop_s(self) -> float:
        return float(self.times[1] - self.times[0])

    @property
    def resolution_hz(self) -> float:
        return 1.0 / self.window_s


@dataclass(frozen=True)
class RidgeTrack:
    times: FloatArray
    frequency_hz: FloatArray
    snr_db: FloatArray
    sigma_hz: FloatArray
    window_s: float

    @property
    def start(self) -> float:
        return float(self.times[0])

    @property
    def end(self) -> float:
        return float(self.times[-1])

    @property
    def energy(self) -> float:
        return float(np.sum(np.maximum(self.snr_db, 0.0))) * float(self.times[1] - self.times[0])


def whitened_spectrogram(
    audio: FloatArray,
    sample_rate: int,
    window_s: float,
    hop_s: float,
    fmin: float,
    fmax: float,
) -> WhitenedSpectrogram:
    nperseg = max(64, int(round(window_s * sample_rate)))
    hop = max(1, int(round(hop_s * sample_rate)))
    nfft = int(2 ** np.ceil(np.log2(nperseg * 2)))
    freqs, times, stft = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - hop,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    band = (freqs >= fmin) & (freqs <= min(fmax, sample_rate / 2.0))
    freqs = freqs[band]
    power = np.abs(stft[band]) ** 2 + 1e-20
    power_db = 10.0 * np.log10(power).T
    width_bins = max(5, int(round(30.0 / (freqs[1] - freqs[0]))) | 1)
    return WhitenedSpectrogram(times, freqs, power_db, power_db - _local_floor(power_db, width_bins), window_s)


def _local_floor(power_db: FloatArray, width_bins: int) -> FloatArray:
    block = max(2, width_bins // 2)
    bins = power_db.shape[1]
    blocks = bins // block
    if blocks < 3:
        return ndimage.median_filter(power_db, size=(1, width_bins), mode="nearest")
    medians = np.median(power_db[:, : blocks * block].reshape(power_db.shape[0], blocks, block), axis=2)
    centres = (np.arange(blocks) + 0.5) * block
    position = np.clip((np.arange(bins) - centres[0]) / block, 0.0, blocks - 1.000001)
    lower = position.astype(int)
    fraction = position - lower
    return medians[:, lower] * (1.0 - fraction) + medians[:, lower + 1] * fraction


def _to_log_axis(spec: WhitenedSpectrogram) -> tuple[FloatArray, FloatArray]:
    log_axis = np.arange(np.log(spec.frequencies[0]), np.log(spec.frequencies[-1]), LOG_BIN)
    linear_index = (np.exp(log_axis) - spec.frequencies[0]) / (spec.frequencies[1] - spec.frequencies[0])
    resampled = ndimage.map_coordinates(
        spec.snr_db,
        np.meshgrid(np.arange(spec.snr_db.shape[0]), linear_index, indexing="ij"),
        order=1,
    )
    return log_axis, resampled


def _viterbi(score: FloatArray, reach: int) -> NDArray[np.int64]:
    frames, bins = score.shape
    near = max(1, reach // 4)
    mid = max(near, reach // 2)
    best = np.maximum(score, 0.0)
    cumulative = np.zeros_like(best)
    cumulative[0] = best[0]
    for t in range(1, frames):
        previous = cumulative[t - 1]
        carried = ndimage.maximum_filter1d(previous, size=2 * near + 1, mode="nearest")
        middle = ndimage.maximum_filter1d(previous, size=2 * mid + 1, mode="nearest") - MID_JUMP_PENALTY_DB
        far = ndimage.maximum_filter1d(previous, size=2 * reach + 1, mode="nearest") - FAR_JUMP_PENALTY_DB
        cumulative[t] = best[t] + np.maximum(np.maximum(carried, middle), far)
    path = np.zeros(frames, dtype=np.int64)
    path[-1] = int(np.argmax(cumulative[-1]))
    offsets = np.arange(-reach, reach + 1)
    magnitude = np.abs(offsets)
    penalty = np.where(magnitude <= near, 0.0, np.where(magnitude <= mid, MID_JUMP_PENALTY_DB, FAR_JUMP_PENALTY_DB))
    penalty = penalty + 0.02 * magnitude
    for t in range(frames - 2, -1, -1):
        centre = path[t + 1]
        lo = max(0, centre - reach)
        hi = min(bins, centre + reach + 1)
        local = cumulative[t, lo:hi] - penalty[lo - centre + reach : hi - centre + reach]
        path[t] = lo + int(np.argmax(local))
    return path


def _refine_frequency(spec: WhitenedSpectrogram, frame: int, frequency: float) -> tuple[float, float]:
    df = spec.frequencies[1] - spec.frequencies[0]
    index = int(np.clip(round((frequency - spec.frequencies[0]) / df), 1, len(spec.frequencies) - 2))
    span = max(1, int(round(0.5 * spec.resolution_hz / df)))
    lo = max(1, index - span)
    hi = min(len(spec.frequencies) - 2, index + span)
    peak = lo + int(np.argmax(spec.power_db[frame, lo : hi + 1]))
    left, centre, right = spec.power_db[frame, peak - 1 : peak + 2]
    curvature = left - 2 * centre + right
    shift = 0.0 if curvature >= 0 else float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return float(spec.frequencies[peak] + shift * df), float(spec.snr_db[frame, peak])


def _segments(active: NDArray[np.bool_], hop_s: float, min_frames: int) -> list[tuple[int, int]]:
    tolerance = int(round(GAP_TOLERANCE_S / hop_s))
    closed = ndimage.binary_closing(active, structure=np.ones(2 * tolerance + 1)) | active
    labels, count = ndimage.label(closed)
    spans: list[tuple[int, int]] = []
    for region in range(1, count + 1):
        frames = np.flatnonzero(labels == region)
        if frames[-1] - frames[0] + 1 >= min_frames:
            spans.append((int(frames[0]), int(frames[-1]) + 1))
    return spans


def _point_sigma(spec: WhitenedSpectrogram, snr_db: FloatArray, frequency: FloatArray) -> FloatArray:
    snr_linear = 10.0 ** (np.maximum(snr_db, 0.0) / 10.0)
    thermal = 0.6 * spec.resolution_hz / np.sqrt(snr_linear)
    slope = np.abs(np.gradient(frequency, spec.hop_s))
    smear = 0.08 * slope * spec.window_s
    return np.sqrt(thermal**2 + smear**2 + (0.03 * spec.resolution_hz) ** 2)


def extract_ridges(
    spec: WhitenedSpectrogram,
    max_ridges: int = 10,
    min_duration_s: float = 6.0,
) -> list[RidgeTrack]:
    log_axis, score = _to_log_axis(spec)
    reach = max(1, int(np.ceil(MAX_RELATIVE_SLOPE_PER_S * spec.hop_s / LOG_BIN)))
    mask_widths = np.maximum(
        MASK_HALF_WIDTH_LOG,
        MAIN_LOBE_MASK_RESOLUTIONS * spec.resolution_hz / np.exp(log_axis),
    )
    mask_bins = np.round(mask_widths / LOG_BIN).astype(int)
    min_frames = int(round(min_duration_s / spec.hop_s))
    tracks: list[RidgeTrack] = []
    frames = np.arange(score.shape[0])
    empty_paths = 0
    for _ in range(max_ridges * 2):
        if len(tracks) >= max_ridges:
            break
        path = _viterbi(score, reach)
        along_path = score[frames, path]
        jumps = np.abs(np.diff(path, prepend=path[0])).astype(float)
        jitter = ndimage.median_filter(jumps, size=JITTER_WINDOW_FRAMES, mode="nearest")
        smooth = jitter <= 1.0 + MAX_JITTER_FRACTION * reach
        active = smooth & (ndimage.median_filter(along_path, size=5, mode="nearest") >= DETECTION_SNR_DB)
        spans = _segments(active, spec.hop_s, min_frames)
        lo = np.maximum(path - mask_bins[path], 0)
        hi = np.minimum(path + mask_bins[path] + 1, score.shape[1])
        empty_paths = 0 if spans else empty_paths + 1
        if empty_paths >= MAX_EMPTY_PATHS:
            break
        for start, stop in spans:
            refined = [
                _refine_frequency(spec, frame, float(np.exp(log_axis[path[frame]])))
                for frame in range(start, stop)
            ]
            frequency = np.array([value[0] for value in refined])
            snr = np.array([value[1] for value in refined])
            tracks.append(
                RidgeTrack(
                    times=spec.times[start:stop].copy(),
                    frequency_hz=frequency,
                    snr_db=snr,
                    sigma_hz=_point_sigma(spec, snr, frequency),
                    window_s=spec.window_s,
                )
            )
        for frame in frames:
            score[frame, lo[frame] : hi[frame]] = -40.0
    return tracks
