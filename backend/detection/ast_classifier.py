from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from backend.detection.events import AircraftCategory, Classification, ScoreKind

FloatArray = NDArray[np.float64]
MODEL_RATE_HZ = 16000
CLIP_SECONDS = 10
AIRCRAFT_LABELS = {
    "Aircraft": AircraftCategory.UNKNOWN_AIRCRAFT,
    "Aircraft engine": AircraftCategory.UNKNOWN_AIRCRAFT,
    "Fixed-wing aircraft, airplane": AircraftCategory.UNKNOWN_AIRCRAFT,
    "Jet engine": AircraftCategory.JET,
    "Propeller, airscrew": AircraftCategory.PROPELLER,
    "Helicopter": AircraftCategory.HELICOPTER,
}


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class AudioSpectrogramTransformerClassifier:
    name = "ast_audioset"

    def __init__(self, model_name: str, device: str = "auto") -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.torch = torch
        self.device = resolve_device(device)
        self.extractor: Any = AutoFeatureExtractor.from_pretrained(model_name)
        self.model: Any = AutoModelForAudioClassification.from_pretrained(model_name).to(self.device).eval()
        self.labels: dict[int, str] = self.model.config.id2label

    def _clips(self, audio: FloatArray, sample_rate: int) -> list[FloatArray]:
        divisor = int(np.gcd(sample_rate, MODEL_RATE_HZ))
        resampled = signal.resample_poly(audio, MODEL_RATE_HZ // divisor, sample_rate // divisor)
        size = CLIP_SECONDS * MODEL_RATE_HZ
        return [resampled[i : i + size] for i in range(0, max(len(resampled) - size // 2, 1), size)]

    def classify(self, audio: FloatArray, sample_rate: int) -> Classification:
        best: dict[str, float] = {}
        for clip in self._clips(audio, sample_rate):
            inputs = self.extractor(clip, sampling_rate=MODEL_RATE_HZ, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits[0]
            probabilities = self.torch.sigmoid(logits).detach().cpu().numpy()
            for index, label in self.labels.items():
                if label in AIRCRAFT_LABELS:
                    best[label] = max(best.get(label, 0.0), float(probabilities[index]))
        score = max(best.values(), default=0.0)
        specific = {
            label: value
            for label, value in best.items()
            if AIRCRAFT_LABELS[label] is not AircraftCategory.UNKNOWN_AIRCRAFT
        }
        if score < 0.05:
            category = AircraftCategory.NOT_AIRCRAFT
        elif specific and max(specific.values()) >= 0.03:
            category = AIRCRAFT_LABELS[max(specific, key=lambda label: specific[label])]
        else:
            category = AircraftCategory.UNKNOWN_AIRCRAFT
        return Classification(score, category, ScoreKind.MODEL_PROBABILITY, best)
