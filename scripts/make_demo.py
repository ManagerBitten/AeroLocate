from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.flight_data.providers.local_dataset import write_records_jsonl
from backend.flight_data.types import AircraftIdentity
from backend.validation.synthetic import (
    SyntheticAircraft,
    build_scene,
    observations_from_trajectory,
)

FLEET = (
    SyntheticAircraft("aaaaaa", 60.0, 350.0, 0.0, 75.0, 75.0, 450.0, 90.0),
    SyntheticAircraft("bbbbbb", 50.0, -600.0, 0.0, 200.0, 62.0, 380.0, 147.0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a clearly synthetic recording and matching flight dataset")
    parser.add_argument("--audio", type=Path, default=Path("data/demo/SYNTHETIC_recording.wav"))
    parser.add_argument("--flights", type=Path, default=Path("flight_datasets/SYNTHETIC_flights.jsonl"))
    arguments = parser.parse_args()
    scene = build_scene(FLEET, noise_rms=0.02, seed=5)
    arguments.audio.parent.mkdir(parents=True, exist_ok=True)
    arguments.flights.parent.mkdir(parents=True, exist_ok=True)
    sf.write(arguments.audio, scene.audio.astype(np.float32) * 0.2, scene.sample_rate)
    write_records_jsonl(
        arguments.flights,
        [
            observations_from_trajectory(AircraftIdentity(icao24=item.icao24, callsign=f"SYN{index}"), path, scene.frame)
            for index, (item, path) in enumerate(zip(FLEET, scene.trajectories))
        ],
    )
    print(
        json.dumps(
            {
                "audio": str(arguments.audio),
                "flights": str(arguments.flights),
                "recording_start_utc": scene.recording_start_utc,
                "true_latitude": scene.microphone_latitude,
                "true_longitude": scene.microphone_longitude,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
