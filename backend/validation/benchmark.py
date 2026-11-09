from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from backend.core.atmosphere import Atmosphere
from backend.core.geo import LocalFrame, haversine_m
from backend.doppler.engine import extract_doppler_curves
from backend.flight_data.reconstruction.trajectory import reconstruct
from backend.flight_data.types import AircraftIdentity, BoundingBox
from backend.geolocation.engine import LocalizationRequest, LocalizationStatus, localize
from backend.geolocation.search import TrackedFlight
from backend.validation.synthetic import (
    SyntheticAircraft,
    build_scene,
    observations_from_trajectory,
)

REGION_LAT_SOUTH = 0.09
REGION_LAT_NORTH = 0.05
REGION_LON_WEST = 0.14
REGION_LON_EAST = 0.11


@dataclass(frozen=True)
class Scenario:
    name: str
    aircraft: int
    time_half_window_s: float = 0.0
    impaired: bool = True
    high_altitude: bool = False
    control: str = ""
    extra_decoys: int = 2
    noise_rms: float = 0.04


SCENARIOS = (
    Scenario("ideal_two_aircraft", 2, impaired=False, extra_decoys=0, noise_rms=0.02),
    Scenario("one_aircraft", 1),
    Scenario("two_aircraft", 2),
    Scenario("three_aircraft", 3),
    Scenario("two_aircraft_time_unknown_60s", 2, time_half_window_s=60.0),
    Scenario("two_aircraft_noisy", 2, noise_rms=0.2),
    Scenario("high_altitude_cruise", 2, high_altitude=True),
    Scenario("control_wrong_flights", 1, control="wrong_flights"),
    Scenario("control_two_wrong_flights", 2, control="wrong_flights"),
    Scenario("control_no_aircraft", 0, control="no_aircraft"),
)


def random_aircraft(index: int, scenario: Scenario, rng: np.random.Generator, wander: float) -> SyntheticAircraft:
    if scenario.high_altitude:
        speed, altitude, cross, level, fundamental = (
            rng.uniform(210.0, 250.0),
            rng.uniform(9000.0, 11500.0),
            rng.uniform(1000.0, 6000.0) * rng.choice([-1.0, 1.0]),
            rng.uniform(4.0, 10.0),
            rng.uniform(70.0, 160.0),
        )
    else:
        speed, altitude, cross, level, fundamental = (
            rng.uniform(55.0, 110.0),
            rng.uniform(300.0, 1200.0),
            rng.uniform(200.0, 1500.0) * rng.choice([-1.0, 1.0]),
            rng.uniform(0.4, 1.2),
            rng.uniform(70.0, 190.0),
        )
    return SyntheticAircraft(
        icao24=f"a{index:05x}",
        closest_time_s=rng.uniform(40.0, 90.0),
        cross_track_m=float(cross),
        along_track_offset_m=0.0,
        heading_deg=rng.uniform(0.0, 360.0),
        speed_mps=float(speed),
        altitude_m=float(altitude),
        fundamental_hz=float(fundamental),
        level_at_1km=float(level),
        turn_rate_deg_s=float(rng.uniform(-0.6, 0.6)) if scenario.impaired else 0.0,
        climb_mps=float(rng.uniform(-3.0, 3.0)) if scenario.impaired else 0.0,
        frequency_wander=wander,
    )


def run_trial(scenario: Scenario, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    wander = float(rng.uniform(1e-4, 5e-4)) if scenario.impaired else 0.0
    aircraft = [random_aircraft(i, scenario, rng, wander) for i in range(scenario.aircraft)]
    truth_atmosphere = Atmosphere(
        surface_temperature_c=float(rng.uniform(0.0, 30.0)) if scenario.impaired else 15.0,
        wind_east_mps=float(rng.normal(0.0, 5.0)) if scenario.impaired else 0.0,
        wind_north_mps=float(rng.normal(0.0, 5.0)) if scenario.impaired else 0.0,
    )
    scene = build_scene(aircraft, noise_rms=scenario.noise_rms, seed=seed, atmosphere=truth_atmosphere)
    box = BoundingBox(
        scene.microphone_latitude - REGION_LAT_SOUTH,
        scene.microphone_longitude - REGION_LON_WEST,
        scene.microphone_latitude + REGION_LAT_NORTH,
        scene.microphone_longitude + REGION_LON_EAST,
    )
    frame = LocalFrame(*box.center, 0.0)
    report_interval = float(rng.uniform(5.0, 15.0)) if scenario.impaired else 5.0
    altitude_bias = float(rng.normal(0.0, 60.0)) if scenario.impaired else 0.0
    flights: list[TrackedFlight] = []
    if scenario.control != "wrong_flights":
        for item, trajectory in zip(aircraft, scene.trajectories):
            record = observations_from_trajectory(
                AircraftIdentity(icao24=item.icao24),
                trajectory,
                scene.frame,
                report_interval_s=report_interval,
                position_noise_m=15.0 if scenario.impaired else 0.0,
                velocity_noise_mps=1.0 if scenario.impaired else 0.0,
                altitude_noise_m=10.0 if scenario.impaired else 0.0,
                altitude_bias_m=altitude_bias,
                rng=rng,
            )
            built = reconstruct(record.observations, frame)
            if built is not None:
                flights.append(TrackedFlight(item.icao24, built))
    decoy_count = scenario.extra_decoys + (3 if scenario.control else 0)
    decoys = [
        replace(
            random_aircraft(100 + i, scenario, rng, 0.0),
            cross_track_m=float(rng.uniform(1500.0, 9000.0)) * float(rng.choice([-1.0, 1.0])),
            along_track_offset_m=float(rng.uniform(-6000.0, 6000.0)),
        )
        for i in range(decoy_count)
    ]
    if decoys:
        decoy_scene = build_scene(decoys, noise_rms=0.0, seed=seed + 1, duration_s=10.0)
        for item, trajectory in zip(decoys, decoy_scene.trajectories):
            record = observations_from_trajectory(
                AircraftIdentity(icao24=item.icao24), trajectory, scene.frame, report_interval_s=report_interval, rng=rng
            )
            built = reconstruct(record.observations, frame)
            if built is not None:
                flights.append(TrackedFlight(item.icao24, built))
    started = time.perf_counter()
    curves = extract_doppler_curves(scene.audio, scene.sample_rate)
    request = LocalizationRequest(
        box,
        scene.recording_start_utc - scenario.time_half_window_s,
        scene.recording_start_utc + scenario.time_half_window_s,
        Atmosphere(),
    )
    result = localize(curves, flights, request, compare=False)
    errors = [
        float(haversine_m(region.latitude, region.longitude, scene.microphone_latitude, scene.microphone_longitude))
        for region in result.regions
    ]
    in_credible = any(error <= max(region.semi_major_m, region.radius_m) for error, region in zip(errors, result.regions))
    truth_keys = {item.icao24 for item in aircraft}
    return {
        "scenario": scenario.name,
        "seed": seed,
        "status": result.status.value,
        "located_probability": result.located_probability,
        "usable_curves": sum(curve.usable for curve in curves),
        "aircraft": scenario.aircraft,
        "region_count": len(result.regions),
        "leading_error_m": errors[0] if errors else None,
        "best_region_error_m": min(errors) if errors else None,
        "uncertainty_radius_m": result.uncertainty_radius_m,
        "credible_area_km2": float(sum(region.area_km2 for region in result.regions)),
        "truth_in_credible_set": bool(in_credible) if errors else None,
        "matched_correctly": sum(match.flight_key in truth_keys for match in result.matches),
        "matched_wrongly": sum(match.flight_key not in truth_keys for match in result.matches),
        "seconds": time.perf_counter() - started,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scenario in sorted({trial["scenario"] for trial in trials}):
        rows = [trial for trial in trials if trial["scenario"] == scenario]
        located = [trial for trial in rows if trial["status"] == LocalizationStatus.LOCALIZED.value]
        answered = [trial for trial in rows if trial["region_count"]]
        statuses = {value.value: sum(trial["status"] == value.value for trial in rows) for value in LocalizationStatus}
        errors = [trial["leading_error_m"] for trial in located]
        coverage = [trial["truth_in_credible_set"] for trial in answered]
        summary[scenario] = {
            "trials": len(rows),
            "statuses": statuses,
            "localized_median_error_m": _percentile(errors, 50),
            "localized_p90_error_m": _percentile(errors, 90),
            "localized_median_stated_radius_m": _percentile([trial["uncertainty_radius_m"] for trial in located], 50),
            "credible_set_coverage": float(np.mean(coverage)) if coverage else None,
            "best_region_median_error_m": _percentile([trial["best_region_error_m"] for trial in answered], 50),
            "median_credible_area_km2": _percentile([trial["credible_area_km2"] for trial in answered], 50),
            "wrong_flight_matches": int(sum(trial["matched_wrongly"] for trial in rows)),
            "median_seconds": _percentile([trial["seconds"] for trial in rows], 50),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic validation of the AeroLocate inference chain")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--output", type=Path, default=Path("docs/validation_results.json"))
    arguments = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    journal = arguments.output.with_suffix(".jsonl")
    journal.parent.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []
    if journal.exists():
        trials = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    finished = {(trial["scenario"], trial["seed"]) for trial in trials}
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool, journal.open("a", encoding="utf-8") as handle:
        futures = {
            pool.submit(run_trial, scenario, 1000 * index + seed): (scenario.name, 1000 * index + seed)
            for index, scenario in enumerate(SCENARIOS)
            for seed in range(arguments.trials)
            if (scenario.name, 1000 * index + seed) not in finished
        }
        total = len(futures)
        for done, future in enumerate(as_completed(futures), start=1):
            name, seed = futures[future]
            try:
                trial = future.result()
            except Exception as error:
                failures.append({"scenario": name, "seed": seed, "error": f"{type(error).__name__}: {error}"})
            else:
                trials.append(trial)
                handle.write(json.dumps(trial) + "\n")
                handle.flush()
            print(f"{done}/{total} {name}#{seed}", flush=True)
    payload = {
        "note": "Synthetic recordings only. These figures say nothing about accuracy on real recordings.",
        "trials_per_scenario": arguments.trials,
        "summary": summarize(trials),
        "failures": failures,
        "trials": trials,
    }
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
