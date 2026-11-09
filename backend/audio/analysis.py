from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal

FloatArray = NDArray[np.float64]
WAVEFORM_POINTS = 4000
SPECTROGRAM_VIEW_FRAMES = 1400
SPECTROGRAM_VIEW_BINS = 320
SPECTROGRAM_MAX_HZ = 4000.0
DB_FLOOR = -110.0


class Preprocessing(str, Enum):
    NONE = "none"
    WIND = "wind"
    BROADBAND = "broadband"


def compute_waveform(audio: FloatArray, sample_rate: int) -> dict[str, list[float] | float]:
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    size = max(1, len(mono) // WAVEFORM_POINTS)
    usable = len(mono) - len(mono) % size
    blocks = mono[:usable].reshape(-1, size)
    return {
        "minimum": np.round(blocks.min(axis=1), 5).astype(float).tolist(),
        "maximum": np.round(blocks.max(axis=1), 5).astype(float).tolist(),
        "seconds_per_point": size / sample_rate,
    }


@dataclass(frozen=True)
class SpectrogramView:
    times: list[float]
    frequencies: list[float]
    decibels: list[list[float]]


def compute_spectrogram(audio: FloatArray, sample_rate: int, png_path: Path | None = None) -> SpectrogramView:
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    nperseg = int(2 ** np.ceil(np.log2(max(0.25 * sample_rate, 256))))
    hop = max(nperseg // 4, max(1, len(mono) // (SPECTROGRAM_VIEW_FRAMES * 3)))
    freqs, times, values = signal.spectrogram(
        mono, fs=sample_rate, window="hann", nperseg=nperseg, noverlap=max(nperseg - hop, 0), mode="psd"
    )
    band = freqs <= min(SPECTROGRAM_MAX_HZ, sample_rate / 2)
    freqs = freqs[band]
    decibels = 10.0 * np.log10(values[band] + 1e-13)
    decibels = np.maximum(decibels, DB_FLOOR)
    if png_path is not None:
        _write_png(decibels, png_path)
    frame_step = max(1, decibels.shape[1] // SPECTROGRAM_VIEW_FRAMES)
    bin_step = max(1, decibels.shape[0] // SPECTROGRAM_VIEW_BINS)
    reduced = ndimage.maximum_filter(decibels, size=(bin_step, frame_step))[::bin_step, ::frame_step]
    return SpectrogramView(
        times=np.round(times[::frame_step], 3).astype(float).tolist(),
        frequencies=np.round(freqs[::bin_step], 2).astype(float).tolist(),
        decibels=np.round(reduced, 1).astype(float).tolist(),
    )


def _write_png(decibels: FloatArray, path: Path) -> None:
    low, high = np.percentile(decibels, [5.0, 99.8])
    scaled = np.clip((decibels - low) / max(high - low, 1e-6), 0.0, 1.0)
    image = cv2.applyColorMap((scaled[::-1] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def apply_preprocessing(audio: FloatArray, sample_rate: int, mode: Preprocessing) -> FloatArray:
    mono = (audio if audio.ndim == 1 else audio.mean(axis=1)).astype(np.float64)
    if mode is Preprocessing.NONE:
        return mono
    high_pass = signal.butter(4, 25.0, btype="highpass", fs=sample_rate, output="sos")
    filtered = signal.sosfiltfilt(high_pass, mono)
    if mode is Preprocessing.WIND:
        return filtered
    return _spectral_gate(filtered, sample_rate)


def _spectral_gate(audio: FloatArray, sample_rate: int) -> FloatArray:
    nperseg = 2048 if sample_rate >= 16000 else 1024
    freqs, times, stft = signal.stft(audio, fs=sample_rate, nperseg=nperseg)
    power = np.abs(stft) ** 2
    noise = np.percentile(power, 25.0, axis=1, keepdims=True)
    gain = np.clip(1.0 - noise / np.maximum(power, 1e-20), 0.08, 1.0)
    smoothed = ndimage.uniform_filter(gain, size=(3, 5))
    _, restored = signal.istft(stft * smoothed, fs=sample_rate, nperseg=nperseg)
    return restored[: len(audio)]
