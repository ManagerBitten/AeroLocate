from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from backend.core.atmosphere import Atmosphere
from backend.core.geo import LocalFrame
from backend.flight_data.reconstruction.trajectory import HermiteTrajectory
from backend.flight_data.types import (
    AircraftIdentity,
    AltitudeKind,
    FlightRecord,
    PositionObservation,
)
from backend.propagation.simulator import solve_emission_time

FloatArray = NDArray[np.float64]
SYNTHETIC_PROVIDER = "synthetic"
WANDER_STEP_S = 0.25
WANDER_TIMESCALE_S = 4.0


@dataclass(frozen=True)
class SyntheticAircraft:
    icao24: str
    closest_time_s: float
    cross_track_m: float
    along_track_offset_m: float
    heading_deg: float
    speed_mps: float
    altitude_m: float
    fundamental_hz: float
    harmonics: tuple[float, ...] = (1.0, 0.7, 0.45, 0.3)
    level_at_1km: float = 0.5
    turn_rate_deg_s: float = 0.0
    climb_mps: float = 0.0
    throttle_ramp_per_s: float = 0.0
    frequency_wander: float = 0.0


@dataclass
class SyntheticScene:
    sample_rate: int
    duration_s: float
    recording_start_utc: float
    microphone_latitude: float
    microphone_longitude: float
    frame: LocalFrame
    atmosphere: Atmosphere
    aircraft: Sequence[SyntheticAircraft]
    audio: FloatArray = field(default_factory=lambda: np.zeros(0))
    trajectories: list[HermiteTrajectory] = field(default_factory=list)

    @property
    def microphone_enu(self) -> FloatArray:
        return np.array([0.0, 0.0, self.atmosphere.surface_elevation_m])


def aircraft_trajectory(
    aircraft: SyntheticAircraft,
    duration_s: float,
    recording_start_utc: float,
    surface_elevation_m: float,
    margin_s: float = 240.0,
    step_s: float = 1.0,
) -> HermiteTrajectory:
    heading = np.radians(aircraft.heading_deg)
    along = np.array([np.sin(heading), np.cos(heading)])
    cross = np.array([np.cos(heading), -np.sin(heading)])
    reference = cross * aircraft.cross_track_m + along * aircraft.along_track_offset_m
    relative = np.arange(-margin_s, duration_s + margin_s + step_s, step_s)
    offset = relative - aircraft.closest_time_s
    rate = np.radians(aircraft.turn_rate_deg_s)
    if abs(rate) > 1e-9:
        angle = heading + rate * offset
        north = aircraft.speed_mps / rate * np.sin(angle) - aircraft.speed_mps / rate * np.sin(heading)
        east = -aircraft.speed_mps / rate * np.cos(angle) + aircraft.speed_mps / rate * np.cos(heading)
        vx = aircraft.speed_mps * np.sin(angle)
        vy = aircraft.speed_mps * np.cos(angle)
    else:
        east = along[0] * aircraft.speed_mps * offset
        north = along[1] * aircraft.speed_mps * offset
        vx = np.full_like(offset, along[0] * aircraft.speed_mps)
        vy = np.full_like(offset, along[1] * aircraft.speed_mps)
    up = surface_elevation_m + aircraft.altitude_m + aircraft.climb_mps * offset
    positions = np.column_stack([reference[0] + east, reference[1] + north, up])
    velocities = np.column_stack([vx, vy, np.full_like(offset, aircraft.climb_mps)])
    return HermiteTrajectory(
        times=recording_start_utc + relative,
        positions=positions,
        velocities=velocities,
        velocity_measured=np.ones(len(relative), dtype=bool),
        altitude_sigma_m=np.full(len(relative), 25.0),
    )


def _pink_noise(samples: int, rng: np.random.Generator) -> FloatArray:
    spectrum = rng.normal(size=samples // 2 + 1) + 1j * rng.normal(size=samples // 2 + 1)
    spectrum /= np.sqrt(np.maximum(np.arange(len(spectrum)), 1.0))
    noise = np.fft.irfft(spectrum, n=samples)
    return noise / np.std(noise)


def _wander_phase(sigma: float, emission: FloatArray, rng: np.random.Generator) -> FloatArray:
    if sigma <= 0:
        return np.zeros_like(emission)
    grid = np.arange(emission[0], emission[-1] + WANDER_STEP_S, WANDER_STEP_S)
    decay = np.exp(-WANDER_STEP_S / WANDER_TIMESCALE_S)
    innovation = sigma * np.sqrt(1.0 - decay**2)
    state = np.zeros(len(grid))
    state[0] = rng.normal(0.0, sigma)
    for index in range(1, len(grid)):
        state[index] = decay * state[index - 1] + rng.normal(0.0, innovation)
    return np.interp(emission, grid, np.cumsum(state) * WANDER_STEP_S)


def render_aircraft(
    aircraft: SyntheticAircraft,
    trajectory: HermiteTrajectory,
    scene: SyntheticScene,
    rng: np.random.Generator,
) -> FloatArray:
    samples = int(scene.duration_s * scene.sample_rate)
    observer = scene.recording_start_utc + np.arange(samples) / scene.sample_rate
    coarse_times = np.arange(observer[0] - 0.1, observer[-1] + 0.2, 0.05)
    microphone = np.broadcast_to(scene.microphone_enu, coarse_times.shape + (3,))
    emission, _ = solve_emission_time(trajectory, coarse_times, microphone, scene.atmosphere)
    position, _ = trajectory.state(emission)
    distance = np.linalg.norm(position - scene.microphone_enu, axis=1)
    fine_emission = np.interp(observer, coarse_times, emission)
    fine_distance = np.interp(observer, coarse_times, distance)
    amplitude = aircraft.level_at_1km * 1000.0 / np.maximum(fine_distance, 50.0)
    relative = fine_emission - scene.recording_start_utc - aircraft.closest_time_s
    throttle_phase = 0.5 * aircraft.throttle_ramp_per_s * relative**2
    wander_phase = _wander_phase(aircraft.frequency_wander, fine_emission, rng)
    output = np.zeros(samples)
    for order, weight in enumerate(aircraft.harmonics, start=1):
        frequency = aircraft.fundamental_hz * order
        if frequency >= scene.sample_rate / 2:
            continue
        phase = 2 * np.pi * frequency * (fine_emission - fine_emission[0] + throttle_phase + wander_phase)
        output += weight * np.sin(phase + rng.uniform(0, 2 * np.pi))
    return amplitude * output


def build_scene(
    aircraft: Sequence[SyntheticAircraft],
    duration_s: float = 120.0,
    sample_rate: int = 8000,
    noise_rms: float = 0.05,
    seed: int = 0,
    recording_start_utc: float = 1_750_000_000.0,
    microphone_latitude: float = 51.47,
    microphone_longitude: float = -0.45,
    atmosphere: Atmosphere | None = None,
) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    atmosphere = atmosphere or Atmosphere()
    frame = LocalFrame(microphone_latitude, microphone_longitude, 0.0)
    scene = SyntheticScene(
        sample_rate=sample_rate,
        duration_s=duration_s,
        recording_start_utc=recording_start_utc,
        microphone_latitude=microphone_latitude,
        microphone_longitude=microphone_longitude,
        frame=frame,
        atmosphere=atmosphere,
        aircraft=aircraft,
    )
    samples = int(duration_s * sample_rate)
    mix = noise_rms * _pink_noise(samples, rng)
    for item in aircraft:
        trajectory = aircraft_trajectory(item, duration_s, recording_start_utc, atmosphere.surface_elevation_m)
        scene.trajectories.append(trajectory)
        mix = mix + render_aircraft(item, trajectory, scene, rng)
    scene.audio = mix
    return scene


def observations_from_trajectory(
    identity: AircraftIdentity,
    trajectory: HermiteTrajectory,
    frame: LocalFrame,
    report_interval_s: float = 5.0,
    position_noise_m: float = 0.0,
    velocity_noise_mps: float = 0.0,
    altitude_noise_m: float = 0.0,
    altitude_bias_m: float = 0.0,
    rng: np.random.Generator | None = None,
) -> FlightRecord:
    generator = rng or np.random.default_rng(0)
    times = np.arange(trajectory.t_start, trajectory.t_end, report_interval_s)
    position, velocity = trajectory.state(times)
    noisy = position.copy()
    noisy[:, :2] += generator.normal(0.0, position_noise_m, (len(times), 2)) if position_noise_m else 0.0
    noisy[:, 2] += altitude_bias_m + (generator.normal(0.0, altitude_noise_m, len(times)) if altitude_noise_m else 0.0)
    speed_noise = generator.normal(0.0, velocity_noise_mps, len(times)) if velocity_noise_mps else np.zeros(len(times))
    latitude, longitude, altitude = frame.from_enu(noisy)
    observations = []
    for i, timestamp in enumerate(times):
        ground_speed = float(np.hypot(velocity[i, 0], velocity[i, 1]) + speed_noise[i])
        observations.append(
            PositionObservation(
                timestamp=float(timestamp),
                latitude=float(latitude[i]),
                longitude=float(longitude[i]),
                altitude_m=float(altitude[i]),
                altitude_kind=AltitudeKind.GEOMETRIC,
                ground_speed_mps=ground_speed,
                track_deg=float(np.degrees(np.arctan2(velocity[i, 0], velocity[i, 1])) % 360.0),
                vertical_rate_mps=float(velocity[i, 2]),
                on_ground=False,
                provider=SYNTHETIC_PROVIDER,
                source_type="SIMULATED",
            )
        )
    return FlightRecord(identity=identity, observations=observations, providers=[SYNTHETIC_PROVIDER])
