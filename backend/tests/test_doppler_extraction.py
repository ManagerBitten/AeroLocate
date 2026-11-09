import numpy as np
import pytest

from backend.doppler.curve_fit import CurveClass, classify_curve, fit_passby, passby_log_factor
from backend.doppler.engine import _group_tracks, _trimmed_fit, extract_doppler_curves
from backend.doppler.kalman import smooth_track
from backend.doppler.ridge import RidgeTrack
from backend.propagation.simulator import predict_doppler
from backend.validation.synthetic import SyntheticAircraft, build_scene


def propeller(**overrides):
    values = dict(
        icao24="aaaaaa",
        closest_time_s=60.0,
        cross_track_m=350.0,
        along_track_offset_m=0.0,
        heading_deg=75.0,
        speed_mps=75.0,
        altitude_m=450.0,
        fundamental_hz=90.0,
    )
    values.update(overrides)
    return SyntheticAircraft(**values)


def test_fit_recovers_parameters_from_clean_curve():
    t = np.arange(0.0, 120.0, 0.7)
    truth = passby_log_factor(t, 60.0, 90.0, 700.0)
    rng = np.random.default_rng(1)
    sigma = np.full_like(t, 2e-4)
    fit = fit_passby(t, np.log(120.0) + truth + rng.normal(0, 2e-4, t.size), sigma)
    assert fit.closest_time == pytest.approx(60.0, abs=0.2)
    assert fit.speed_mps == pytest.approx(90.0, rel=0.03)
    assert fit.distance_m == pytest.approx(700.0, rel=0.05)
    assert classify_curve(t, np.log(120.0) + truth, sigma, fit) is CurveClass.DOPPLER_CONSISTENT


def test_burst_from_crossing_ridge_does_not_discard_a_good_curve():
    t = np.arange(0.0, 120.0, 1.4)
    clean = np.log(120.0) + passby_log_factor(t, 60.0, 80.0, 900.0)
    corrupted = clean.copy()
    corrupted[12:20] += 0.03
    sigma = np.full_like(t, 2e-4)
    kept_t, kept_v, kept_s, fit = _trimmed_fit(t, corrupted, sigma)
    assert len(kept_t) == len(t) - 8
    assert classify_curve(kept_t, kept_v, kept_s, fit) is CurveClass.DOPPLER_CONSISTENT
    assert fit.speed_mps == pytest.approx(80.0, rel=0.03)


def test_trimming_does_not_turn_a_rising_curve_into_doppler():
    t = np.arange(0.0, 120.0, 1.4)
    rising = np.log(120.0) + 0.002 * t + 0.01 * np.sin(t / 7.0)
    sigma = np.full_like(t, 2e-4)
    kept_t, kept_v, kept_s, fit = _trimmed_fit(t, rising, sigma)
    assert classify_curve(kept_t, kept_v, kept_s, fit) not in (CurveClass.DOPPLER_CONSISTENT, CurveClass.PARTIAL_DOPPLER)


def test_kalman_rejects_spikes():
    t = np.arange(0.0, 40.0, 0.2)
    clean = 0.002 * t
    noisy = clean + np.random.default_rng(0).normal(0, 1e-4, t.size)
    noisy[50] += 0.2
    noisy[120] -= 0.2
    result = smooth_track(t, noisy, np.full_like(t, 1e-4))
    assert not result.inlier[50]
    assert not result.inlier[120]
    assert np.max(np.abs(result.log_frequency - clean)) < 2e-3


def test_extracts_curve_matching_simulated_truth():
    scene = build_scene([propeller()], noise_rms=0.03, seed=3)
    curves = extract_doppler_curves(scene.audio, scene.sample_rate)
    usable = [curve for curve in curves if curve.usable]
    assert len(usable) == 1
    curve = usable[0]
    prediction = predict_doppler(
        scene.trajectories[0],
        scene.recording_start_utc + curve.times,
        scene.microphone_enu,
        scene.atmosphere,
    )
    model = np.log(prediction.doppler_factor)
    residual = curve.log_frequency - model
    residual -= np.median(residual)
    frequency_error_hz = 90.0 * np.median(np.abs(residual))
    assert frequency_error_hz < 0.15
    assert curve.fit.speed_mps == pytest.approx(75.0, rel=0.12)


def test_harmonics_merge_into_a_single_event():
    scene = build_scene([propeller()], noise_rms=0.02, seed=4)
    curves = [curve for curve in extract_doppler_curves(scene.audio, scene.sample_rate) if curve.usable]
    assert len(curves) == 1
    assert len(curves[0].harmonic_numbers) >= 2


def test_overlapping_aircraft_stay_separate():
    scene = build_scene(
        [
            propeller(),
            propeller(icao24="bbbbbb", closest_time_s=50.0, fundamental_hz=147.0, cross_track_m=-300.0, heading_deg=200.0, speed_mps=62.0, altitude_m=380.0),
        ],
        noise_rms=0.02,
        seed=5,
    )
    curves = [curve for curve in extract_doppler_curves(scene.audio, scene.sample_rate) if curve.usable]
    assert len(curves) == 2
    assert sorted(round(curve.fit.speed_mps, -1) for curve in curves) == [60.0, 80.0]


def test_hybrid_ridge_cannot_bridge_two_aircraft():
    t = np.arange(0.0, 60.0, 0.25)
    sigma = np.full_like(t, 0.05)
    snr = np.full_like(t, 30.0)
    first = 100.0 * np.exp(-0.002 * t)
    second = 150.0 * np.exp(0.0005 * np.abs(t - 30.0) - 0.01 * np.tanh((t - 30.0) / 6.0))
    hybrid = np.where(t < 34.0, first * 2.0, second * 1.0003)
    groups = _group_tracks(
        [
            RidgeTrack(t, first, snr * 2.0, sigma, 1.4),
            RidgeTrack(t, second, snr * 1.5, sigma, 1.4),
            RidgeTrack(t, hybrid, snr, sigma, 1.4),
        ]
    )
    anchors = [sorted(float(np.median(track.frequency_hz)) for track in group) for group in groups]
    assert not any(len([value for value in members if abs(value - 100.0) < 12.0 or abs(value - 150.0) < 12.0]) == 2 for members in anchors)


def test_throttle_ramp_is_not_called_doppler():
    scene = build_scene(
        [propeller(cross_track_m=0.0, altitude_m=30000.0, speed_mps=1.0, level_at_1km=30.0, throttle_ramp_per_s=0.0015)],
        noise_rms=0.02,
        seed=6,
    )
    curves = extract_doppler_curves(scene.audio, scene.sample_rate)
    assert curves
    assert all(curve.curve_class in (CurveClass.RISING_FREQUENCY, CurveClass.INCONSISTENT) for curve in curves)


def test_mains_hum_and_noise_are_rejected():
    rng = np.random.default_rng(7)
    t = np.arange(90 * 8000) / 8000.0
    audio = 0.2 * np.sin(2 * np.pi * 60.0 * t) + 0.02 * rng.normal(size=t.size)
    curves = extract_doppler_curves(audio, 8000)
    assert not any(curve.usable for curve in curves)
    assert any(curve.curve_class is CurveClass.MAINS_HUM for curve in curves)


def test_pure_noise_yields_no_usable_curve():
    audio = 0.1 * np.random.default_rng(8).normal(size=60 * 8000)
    assert not any(curve.usable for curve in extract_doppler_curves(audio, 8000))
