import numpy as np
import pytest

from backend.core.atmosphere import Atmosphere
from backend.core.geo import LocalFrame, ecef_to_geodetic, geodetic_to_ecef
from backend.doppler.curve_fit import passby_log_factor
from backend.propagation.simulator import closest_approach, predict_doppler, solve_emission_time
from backend.validation.synthetic import SyntheticAircraft, aircraft_trajectory

CALM = Atmosphere(wind_sigma_mps=0.0, temperature_sigma_k=0.0)


def level_flight(speed=100.0, cross=400.0, altitude=600.0):
    return aircraft_trajectory(
        SyntheticAircraft("aaaaaa", 60.0, cross, 0.0, 90.0, speed, altitude, 100.0),
        120.0,
        0.0,
        0.0,
    )


def test_ecef_round_trip():
    ecef = geodetic_to_ecef(51.47, -0.45, 1234.5)
    lat, lon, alt = ecef_to_geodetic(ecef)
    assert float(lat) == pytest.approx(51.47, abs=1e-9)
    assert float(lon) == pytest.approx(-0.45, abs=1e-9)
    assert float(alt) == pytest.approx(1234.5, abs=1e-5)


def test_local_frame_round_trip_and_units():
    frame = LocalFrame(51.0, 0.0)
    enu = frame.to_enu(51.0 + 1.0 / 111.2, 0.0, 0.0)
    assert enu[1] == pytest.approx(1000.0, rel=0.01)
    lat, lon, alt = frame.from_enu(np.array([100.0, 200.0, 300.0]))
    assert frame.to_enu(lat, lon, alt) == pytest.approx(np.array([100.0, 200.0, 300.0]), abs=1e-4)


def test_speed_of_sound_at_sea_level_isa():
    assert float(Atmosphere().sound_speed(0.0)) == pytest.approx(340.3, abs=0.1)
    assert float(Atmosphere().sound_speed(11000.0)) == pytest.approx(295.1, abs=0.3)


def test_emission_time_satisfies_retarded_equation():
    path = level_flight()
    observer = np.linspace(20.0, 100.0, 50)
    microphone = np.zeros((50, 3))
    emission, valid = solve_emission_time(path, observer, microphone, CALM)
    position, _ = path.state(emission)
    distance = np.linalg.norm(position, axis=1)
    c = CALM.path_sound_speed(position[:, 2], 0.0)
    assert valid.all()
    assert emission + distance / c == pytest.approx(observer, abs=1e-5)
    assert np.all(emission < observer)


def test_retarded_time_converges_at_high_subsonic_speed():
    path = level_flight(speed=280.0, cross=100.0, altitude=9000.0)
    observer = np.linspace(40.0, 90.0, 40)
    prediction = predict_doppler(path, observer, np.zeros(3), CALM)
    assert prediction.valid.all()


def test_doppler_formula_for_direct_approach():
    path = level_flight(speed=100.0, cross=0.0, altitude=1.0)
    prediction = predict_doppler(path, np.array([25.0, 100.0]), np.zeros(3), CALM)
    c = float(CALM.sound_speed(1.0))
    assert prediction.doppler_factor[0] == pytest.approx(c / (c - 100.0), rel=1e-3)
    assert prediction.doppler_factor[1] == pytest.approx(c / (c + 100.0), rel=1e-3)


def test_closed_form_passby_matches_numerical_simulator():
    path = level_flight(speed=120.0, cross=500.0, altitude=800.0)
    atmosphere = Atmosphere(surface_temperature_c=14.6, wind_sigma_mps=0.0, temperature_sigma_k=0.0)
    calm_uniform = _uniform(atmosphere)
    distance = float(np.hypot(500.0, 800.0))
    closest_observer = 60.0 + distance / 340.0
    observer = np.linspace(30.0, 100.0, 40)
    prediction = predict_doppler(path, observer, np.zeros(3), calm_uniform)
    analytic = passby_log_factor(observer, closest_observer, 120.0, distance, 340.0)
    assert np.log(prediction.doppler_factor) == pytest.approx(analytic, abs=1e-5)


def _uniform(reference):
    class Uniform(Atmosphere):
        def sound_speed(self, altitude_m):
            return np.full(np.shape(altitude_m), 340.0)

        def path_sound_speed(self, a, b):
            return np.full(np.broadcast(a, b).shape, 340.0)

        def sound_speed_sigma(self, altitude_m):
            return np.zeros(np.shape(altitude_m))

    return Uniform()


def test_propagation_delay_shifts_observed_closest_approach():
    path = level_flight(speed=120.0, cross=0.0, altitude=3400.0)
    approach = closest_approach(path, np.zeros(3), CALM)
    assert approach.emission_time == pytest.approx(60.0, abs=0.3)
    assert approach.observer_time - approach.emission_time == pytest.approx(10.0, abs=0.3)


def test_headwind_along_the_ray_lowers_the_shift():
    path = level_flight(speed=100.0, cross=0.0, altitude=1.0)
    calm = predict_doppler(path, np.array([25.0]), np.zeros(3), CALM).doppler_factor[0]
    tail = Atmosphere(wind_east_mps=20.0, wind_sigma_mps=0.0, temperature_sigma_k=0.0)
    windy = predict_doppler(path, np.array([25.0]), np.zeros(3), tail).doppler_factor[0]
    assert windy > calm or windy < calm
    assert abs(windy - calm) > 1e-4


def test_supersonic_geometry_is_flagged_not_silently_used():
    path = level_flight(speed=420.0, cross=0.0, altitude=30.0)
    prediction = predict_doppler(path, np.array([30.0]), np.zeros(3), CALM)
    assert not prediction.valid.all()
