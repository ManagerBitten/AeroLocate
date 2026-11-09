from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal

FloatArray = NDArray[np.float64]
BAND_LOW_HZ = 30.0
BAND_HIGH_HZ = 3000.0
ENVELOPE_RATE_HZ = 4.0
MIN_EVENT_S = 4.0
MERGE_GAP_S = 3.0
RISE_DB = 5.0
TONE_CHUNK_S = 3.0
MAX_TONE_CHUNKS = 12


class AircraftCategory(str, Enum):
    JET = "jet"
    PROPELLER = "propeller"
    TURBOPROP = "turboprop"
    HELICOPTER = "helicopter"
    UNKNOWN_AIRCRAFT = "unknown_aircraft"
    UNCLASSIFIED_TONAL = "unclassified_tonal"
    NOT_AIRCRAFT = "not_aircraft"


class ScoreKind(str, Enum):
    MODEL_PROBABILITY = "model_probability"
    HEURISTIC = "heuristic"


@dataclass(frozen=True)
class Classification:
    aircraft_score: float
    category: AircraftCategory
    score_kind: ScoreKind
    labels: dict[str, float]


class SoundClassifier(Protocol):
    name: str

    def classify(self, audio: FloatArray, sample_rate: int) -> Classification: ...


@dataclass(frozen=True)
class AircraftEvent:
    detection_id: str
    start_s: float
    end_s: float
    peak_s: float
    score: float
    score_kind: str
    classifier: str
    category: str
    dominant_frequencies_hz: tuple[float, ...]
    harmonic_spacing_hz: float | None
    spectral_energy_db: float
    snr_db: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _band_envelope(audio: FloatArray, sample_rate: int) -> tuple[FloatArray, FloatArray]:
    top = min(BAND_HIGH_HZ, 0.45 * sample_rate)
    sos = signal.butter(4, [BAND_LOW_HZ, top], btype="bandpass", fs=sample_rate, output="sos")
    filtered = signal.sosfiltfilt(sos, audio)
    hop = int(sample_rate / ENVELOPE_RATE_HZ)
    frames = len(filtered) // hop
    power = np.add.reduce(filtered[: frames * hop].reshape(frames, hop) ** 2, axis=1) / hop
    level = 10.0 * np.log10(power + 1e-14)
    return (np.arange(frames) + 0.5) / ENVELOPE_RATE_HZ, ndimage.uniform_filter1d(level, size=9, mode="nearest")


def energy_segments(audio: FloatArray, sample_rate: int) -> list[tuple[float, float, float, float]]:
    times, level = _band_envelope(audio, sample_rate)
    if len(times) < 4:
        return []
    floor = float(np.percentile(level, 15.0))
    active = level > floor + RISE_DB
    gap = int(MERGE_GAP_S * ENVELOPE_RATE_HZ)
    if gap > 0:
        active = ndimage.binary_closing(active, structure=np.ones(2 * gap + 1)) | active
    labels, count = ndimage.label(active)
    segments = []
    for region in range(1, count + 1):
        indices = np.flatnonzero(labels == region)
        start, end = float(times[indices[0]]), float(times[indices[-1]])
        if end - start < MIN_EVENT_S:
            continue
        peak_index = indices[int(np.argmax(level[indices]))]
        segments.append((start, end, float(times[peak_index]), float(level[peak_index] - floor)))
    if not segments and float(np.ptp(level)) < RISE_DB:
        segments.append((float(times[0]), float(times[-1]), float(times[len(times) // 2]), 0.0))
    return segments


def _chunk_tones(chunk: FloatArray, sample_rate: int) -> FloatArray:
    nperseg = min(len(chunk), int(2 ** np.ceil(np.log2(sample_rate))))
    freqs, power = signal.welch(chunk, fs=sample_rate, nperseg=nperseg)
    band = (freqs >= BAND_LOW_HZ) & (freqs <= min(BAND_HIGH_HZ, 0.45 * sample_rate))
    freqs, power = freqs[band], power[band] + 1e-20
    decibels = 10.0 * np.log10(power)
    floor = ndimage.median_filter(decibels, size=max(5, int(40.0 / (freqs[1] - freqs[0])) | 1), mode="nearest")
    peaks, _ = signal.find_peaks(decibels - floor, height=8.0, distance=max(1, int(4.0 / (freqs[1] - freqs[0]))))
    order = peaks[np.argsort((decibels - floor)[peaks])[::-1]][:8]
    return np.sort(freqs[order])


def spectral_features(audio: FloatArray, sample_rate: int) -> dict[str, Any]:
    chunk_size = int(TONE_CHUNK_S * sample_rate)
    step = max(chunk_size, len(audio) // MAX_TONE_CHUNKS)
    chunks = [audio[i : i + chunk_size] for i in range(0, max(len(audio) - chunk_size + 1, 1), step)]
    tones = max((_chunk_tones(chunk, sample_rate) for chunk in chunks), key=len)
    spacing = float(np.median(np.diff(tones))) if len(tones) >= 3 else None
    nperseg = min(len(audio), int(2 ** np.ceil(np.log2(sample_rate))))
    freqs, power = signal.welch(audio, fs=sample_rate, nperseg=nperseg)
    band = (freqs >= BAND_LOW_HZ) & (freqs <= min(BAND_HIGH_HZ, 0.45 * sample_rate))
    freqs, power = freqs[band], power[band] + 1e-20
    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
    centroid = float(np.sum(freqs * power) / np.sum(power))
    envelope = np.abs(signal.hilbert(audio[: min(len(audio), sample_rate * 30)]))
    envelope_rate = 200
    step = max(1, sample_rate // envelope_rate)
    reduced = envelope[:: step] - np.mean(envelope[:: step])
    if len(reduced) > 64:
        modulation_freqs, modulation = signal.welch(reduced, fs=sample_rate / step, nperseg=min(len(reduced), 512))
        slapping = (modulation_freqs >= 8.0) & (modulation_freqs <= 40.0)
        base = (modulation_freqs >= 45.0) & (modulation_freqs <= 90.0)
        modulation_ratio = float(modulation[slapping].max() / max(np.median(modulation[base]), 1e-20))
    else:
        modulation_ratio = 0.0
    return {
        "tones": tuple(round(float(value), 1) for value in tones),
        "spacing": spacing,
        "flatness": flatness,
        "centroid": centroid,
        "modulation_ratio": modulation_ratio,
        "energy_db": float(10.0 * np.log10(np.sum(power) * (freqs[1] - freqs[0]))),
    }


class SpectralHeuristicClassifier:
    name = "spectral_heuristic"

    def classify(self, audio: FloatArray, sample_rate: int) -> Classification:
        features = spectral_features(audio, sample_rate)
        tonal = len(features["tones"]) >= 3
        low_centred = features["centroid"] < 1200.0
        candidate = tonal or (features["flatness"] > 0.15 and low_centred)
        return Classification(
            aircraft_score=0.5 if candidate else 0.0,
            category=AircraftCategory.UNCLASSIFIED_TONAL if candidate else AircraftCategory.NOT_AIRCRAFT,
            score_kind=ScoreKind.HEURISTIC,
            labels={"tonal": float(tonal), "low_centred": float(low_centred)},
        )


def detect_events(
    audio: FloatArray,
    sample_rate: int,
    classifier: SoundClassifier,
    minimum_score: float = 0.25,
) -> list[AircraftEvent]:
    mono = (audio if audio.ndim == 1 else audio.mean(axis=1)).astype(np.float64)
    events: list[AircraftEvent] = []
    for start, end, peak, rise in energy_segments(mono, sample_rate):
        segment = mono[int(start * sample_rate) : int(end * sample_rate)]
        if len(segment) < sample_rate:
            continue
        result = classifier.classify(segment, sample_rate)
        if result.aircraft_score < minimum_score or result.category is AircraftCategory.NOT_AIRCRAFT:
            continue
        features = spectral_features(segment, sample_rate)
        events.append(
            AircraftEvent(
                detection_id=f"D{len(events) + 1:02d}",
                start_s=round(start, 2),
                end_s=round(end, 2),
                peak_s=round(peak, 2),
                score=round(result.aircraft_score, 3),
                score_kind=result.score_kind.value,
                classifier=classifier.name,
                category=result.category.value,
                dominant_frequencies_hz=features["tones"],
                harmonic_spacing_hz=features["spacing"],
                spectral_energy_db=round(features["energy_db"], 1),
                snr_db=round(rise, 1),
            )
        )
    return events


def overlapping_detection(events: Sequence[AircraftEvent], start: float, end: float) -> str | None:
    best: tuple[float, str] | None = None
    for event in events:
        overlap = min(end, event.end_s) - max(start, event.start_s)
        if overlap > 0 and (best is None or overlap > best[0]):
            best = (overlap, event.detection_id)
    return best[1] if best else None
