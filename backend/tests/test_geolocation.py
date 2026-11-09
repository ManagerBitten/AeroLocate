from dataclasses import replace

import numpy as np
import pytest

from backend.core.atmosphere import Atmosphere
from backend.core.geo import LocalFrame, haversine_m
from backend.doppler.engine import extract_doppler_curves
from backend.flight_data.reconstruction.trajectory import reconstruct
from backend.flight_data.types import AircraftIdentity, BoundingBox
from backend.geolocation.engine import (
    MAX_LOCALIZING_SLANT_M,
    LocalizationRequest,
    LocalizationStatus,
    _surplus_constraints,
    localize,
    select_independent_curves,
)
from backend.geolocation.likelihood import prepare_event, profiled_chi2, speed_match_log_factor
from backend.geolocation.search import TrackedFlight
from backend.validation.synthetic import (
    SyntheticAircraft,
    build_scene,
    observations_from_trajectory,
)

ALPHA = SyntheticAircraft("aaaaaa", 60.0, 350.0, 0.0, 75.0, 75.0, 450.0, 90.0)
BRAVO = SyntheticAircraft("bbbbbb", 50.0, -600.0, 0.0, 200.0, 62.0, 380.0, 147.0)
DECOY = SyntheticAircraft("cccccc", 30.0, 9000.0, 3000.0, 300.0, 80.0, 500.0, 120.0)


@pytest.fixture(scope="module")
def two_aircraft_scene():
    scene = build_scene([ALPHA, BRAVO], noise_rms=0.02, seed=5)
    return scene, extract_doppler_curves(scene.audio, scene.sample_rate)


@pytest.fixture(scope="module")
def one_aircraft_scene():
    scene = build_scene([ALPHA], noise_rms=0.02, seed=5)
    return scene, extract_doppler_curves(scene.audio, scene.sample_rate)


def region_around(scene):
    return BoundingBox(
        scene.microphone_latitude - 0.09,
        scene.microphone_longitude - 0.14,
        scene.microphone_latitude + 0.05,
        scene.microphone_longitude + 0.11,
    )


def tracked(scene, box):
    frame = LocalFrame(*box.center, 0.0)
    flights = []
    for item, path in zip(scene.aircraft, scene.trajectories):
        record = observations_from_trajectory(AircraftIdentity(icao24=item.icao24), path, scene.frame)
        flights.append(TrackedFlight(item.icao24, reconstruct(record.observations, frame)))
    return flights


def request_for(scene, box, half_window=0.0):
    return LocalizationRequest(
        box,
        scene.recording_start_utc - half_window,
        scene.recording_start_utc + half_window,
        Atmosphere(),
    )


def error_m(scene, region):
    return float(haversine_m(region.latitude, region.longitude, scene.microphone_latitude, scene.microphone_longitude))


def test_chi2_profile_removes_source_frequency_offset():
    t = np.linspace(-30, 30, 50)[None, :]
    model = 0.05 * np.tanh(t / 5.0)
    weights = np.full_like(t, 1e6)
    residual = np.full_like(t, 0.4)
    assert profiled_chi2(residual, weights, t, model, np.array([0.01]))[0] == pytest.approx(0.0, abs=1e-6)


def test_one_aircraft_reports_mirror_ambiguity(one_aircraft_scene):
    scene, curves = one_aircraft_scene
    box = region_around(scene)
    result = localize(curves, tracked(scene, box), request_for(scene, box))
    assert result.status is LocalizationStatus.AMBIGUOUS
    assert result.latitude is None
    assert len(result.regions) == 2
    assert min(error_m(scene, region) for region in result.regions) < 60.0
    assert max(error_m(scene, region) for region in result.regions) == pytest.approx(700.0, abs=80.0)


def test_single_event_never_claims_a_location(one_aircraft_scene):
    scene, curves = one_aircraft_scene
    box = region_around(scene)
    alone = SyntheticAircraft("aaaaaa", 60.0, 350.0, 0.0, 75.0, 75.0, 450.0, 90.0, turn_rate_deg_s=0.5)
    turning = build_scene([alone], noise_rms=0.02, seed=5)
    extracted = extract_doppler_curves(turning.audio, turning.sample_rate)
    result = localize(extracted, tracked(turning, box), request_for(turning, box), compare=False)
    assert result.status is not LocalizationStatus.LOCALIZED
    assert result.latitude is None


def test_two_aircraft_resolve_the_ambiguity(two_aircraft_scene):
    scene, curves = two_aircraft_scene
    box = region_around(scene)
    result = localize(curves, tracked(scene, box), request_for(scene, box))
    assert result.status is LocalizationStatus.LOCALIZED
    assert error_m(scene, result.regions[0]) < 60.0
    assert result.uncertainty_radius_m > error_m(scene, result.regions[0])
    joint = result.comparison[-1]
    assert all(joint.credible_area_km2 < row.credible_area_km2 for row in result.comparison[:-1])
    assert {match.flight_key for match in result.matches} == {"aaaaaa", "bbbbbb"}
    assert result.coordinate_decimals <= 3


def test_unknown_clock_needs_three_aircraft(two_aircraft_scene):
    scene, curves = two_aircraft_scene
    box = region_around(scene)
    result = localize(curves, tracked(scene, box), request_for(scene, box, half_window=30.0), compare=False)
    assert result.status is not LocalizationStatus.LOCALIZED
    assert result.latitude is None


def test_track_speed_must_match_the_measured_speed(one_aircraft_scene):
    scene, curves = one_aircraft_scene
    box = region_around(scene)
    event = prepare_event(next(curve for curve in curves if curve.usable))
    trajectory = tracked(scene, box)[0].trajectory
    start = np.array([scene.recording_start_utc])
    matched = float(np.ravel(speed_match_log_factor(event, trajectory, start))[0])
    fast = replace(trajectory, velocities=trajectory.velocities * 1.25)
    mismatched = float(np.ravel(speed_match_log_factor(event, fast, start))[0])
    assert mismatched < matched - 8.0


def test_cruise_altitude_curves_cannot_confirm_a_location():
    assert _surplus_constraints(2, False) == 2
    assert _surplus_constraints(1, False) == 0
    assert _surplus_constraints(2, True) == 1
    assert _surplus_constraints(3, True) == 3
    assert MAX_LOCALIZING_SLANT_M <= 6000.0


def test_decoy_flight_is_not_matched(two_aircraft_scene):
    scene, curves = two_aircraft_scene
    box = region_around(scene)
    decoy = build_scene([DECOY], duration_s=120.0, noise_rms=0.0, seed=1)
    flights = tracked(scene, box) + tracked(decoy, box)
    result = localize(curves, flights, request_for(scene, box))
    assert result.status is LocalizationStatus.LOCALIZED
    assert "cccccc" not in {match.flight_key for match in result.matches}


def test_wrong_flights_do_not_produce_a_location(one_aircraft_scene):
    scene, curves = one_aircraft_scene
    box = region_around(scene)
    wrong = SyntheticAircraft("dddddd", 60.0, 2000.0, 0.0, 10.0, 230.0, 10500.0, 90.0)
    other = build_scene([wrong], noise_rms=0.0, seed=2)
    result = localize(curves, tracked(other, box), request_for(scene, box))
    assert result.status is LocalizationStatus.NOT_LOCALIZABLE
    assert result.regions == ()
    assert result.latitude is None


def test_no_flights_is_reported_as_not_localizable(one_aircraft_scene):
    scene, curves = one_aircraft_scene
    box = region_around(scene)
    result = localize(curves, [], request_for(scene, box))
    assert result.status is LocalizationStatus.NOT_LOCALIZABLE
    assert "No flight trajectories" in result.reasons[0]


def test_no_measurable_doppler_is_reported():
    audio = 0.1 * np.random.default_rng(1).normal(size=40 * 8000)
    curves = extract_doppler_curves(audio, 8000)
    box = BoundingBox(51.0, -1.0, 52.0, 0.0)
    result = localize(curves, [], LocalizationRequest(box, 0.0, 0.0))
    assert result.status is LocalizationStatus.NOT_LOCALIZABLE
    assert "Doppler" in result.reasons[0]


def test_repeated_geometry_is_counted_once(one_aircraft_scene):
    _, curves = one_aircraft_scene
    usable = [curve for curve in curves if curve.usable]
    chosen, dropped = select_independent_curves(usable + usable)
    assert len(chosen) == len(usable)
    assert dropped
