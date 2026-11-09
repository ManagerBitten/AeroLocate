import subprocess

import numpy as np
import pytest
import soundfile as sf

from backend.audio.analysis import Preprocessing, apply_preprocessing, compute_spectrogram, compute_waveform
from backend.audio.ingest import MediaRejected, extract_audio, load_audio, probe_media, sniff_signature
from backend.core.config import get_settings
from backend.detection.events import SpectralHeuristicClassifier, detect_events
from backend.validation.synthetic import SyntheticAircraft, build_scene

RATE = 22050


def tone(seconds=3.0, channels=2):
    t = np.arange(int(seconds * RATE)) / RATE
    data = 0.3 * np.sin(2 * np.pi * 220 * t)
    return np.column_stack([data] * channels)


def test_wav_round_trip_preserves_rate_and_channels(tmp_path):
    source = tmp_path / "a.wav"
    sf.write(source, tone(), RATE)
    settings = get_settings()
    info = probe_media(source, ".wav", settings)
    assert info.sample_rate == RATE and info.channels == 2 and not info.lossy
    audio, rate = load_audio(extract_audio(source, tmp_path / "out.wav", settings))
    assert rate == RATE and audio.shape[1] == 2


@pytest.mark.parametrize(
    "extension,arguments",
    [(".mp3", ["-c:a", "libmp3lame"]), (".m4a", ["-c:a", "aac"]), (".flac", ["-c:a", "flac"])],
)
def test_compressed_formats_are_extracted(tmp_path, extension, arguments):
    settings = get_settings()
    source = tmp_path / "a.wav"
    sf.write(source, tone(), RATE)
    target = tmp_path / f"b{extension}"
    subprocess.run([settings.ffmpeg_path, "-y", "-loglevel", "error", "-i", str(source), *arguments, str(target)], check=True)
    info = probe_media(target, extension, settings)
    assert info.duration_s == pytest.approx(3.0, abs=0.2)
    assert info.lossy == (extension != ".flac")
    audio, _ = load_audio(extract_audio(target, tmp_path / "out.wav", settings))
    assert len(audio) > RATE


@pytest.mark.parametrize("extension", [".mp4", ".mov", ".mkv", ".avi"])
def test_video_containers(tmp_path, extension):
    settings = get_settings()
    source = tmp_path / "a.wav"
    sf.write(source, tone(), RATE)
    target = tmp_path / f"v{extension}"
    command = [
        settings.ffmpeg_path, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3",
        "-i", str(source),
        "-c:v", "mpeg4", "-c:a", "aac" if extension != ".avi" else "libmp3lame", "-shortest",
        str(target),
    ]
    subprocess.run(command, check=True)
    info = probe_media(target, extension, settings)
    assert info.has_video
    assert len(load_audio(extract_audio(target, tmp_path / "out.wav", settings))[0]) > RATE


def test_embedded_metadata_is_kept_as_unverified_hints(tmp_path):
    settings = get_settings()
    target = tmp_path / "m.mp4"
    command = [
        settings.ffmpeg_path, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=2",
        "-c:a", "aac",
        "-metadata", "creation_time=2025-06-15T15:06:40Z",
        "-metadata", "location=+51.4700-000.4500/",
        str(target),
    ]
    subprocess.run(command, check=True)
    info = probe_media(target, ".mp4", settings)
    assert info.timestamp_candidates[0].trust == "unverified_metadata"
    assert info.embedded_location_hint == (51.47, -0.45)


def test_declared_type_must_match_contents(tmp_path):
    fake = tmp_path / "evil.wav"
    fake.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
    assert sniff_signature(fake) is None
    with pytest.raises(MediaRejected):
        probe_media(fake, ".wav", get_settings())


def test_truncated_media_is_rejected(tmp_path):
    fake = tmp_path / "bad.wav"
    fake.write_bytes(b"RIFF\x24\0\0\0WAVEjunk")
    with pytest.raises(MediaRejected):
        probe_media(fake, ".wav", get_settings())


def test_file_without_audio_is_rejected(tmp_path):
    settings = get_settings()
    target = tmp_path / "silent.mp4"
    command = [settings.ffmpeg_path, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", "-c:v", "mpeg4", str(target)]
    subprocess.run(command, check=True)
    with pytest.raises(MediaRejected):
        probe_media(target, ".mp4", settings)


def test_waveform_and_spectrogram_shapes(tmp_path):
    audio = tone(seconds=8.0)
    wave = compute_waveform(audio, RATE)
    assert len(wave["minimum"]) == len(wave["maximum"]) <= 4100
    view = compute_spectrogram(audio, RATE, tmp_path / "s.png")
    peak = np.array(view.decibels).mean(axis=1).argmax()
    assert view.frequencies[peak] == pytest.approx(220.0, abs=8.0)
    assert (tmp_path / "s.png").exists()


def test_preprocessing_leaves_input_untouched():
    audio = tone(seconds=3.0, channels=1)
    copy = audio.copy()
    processed = apply_preprocessing(audio, RATE, Preprocessing.BROADBAND)
    assert np.array_equal(audio, copy)
    assert len(processed) == len(audio)


def test_wind_filter_removes_rumble():
    t = np.arange(10 * RATE) / RATE
    rumble = np.sin(2 * np.pi * 6 * t)
    wanted = 0.1 * np.sin(2 * np.pi * 300 * t)
    cleaned = apply_preprocessing(rumble + wanted, RATE, Preprocessing.WIND)
    assert np.std(cleaned) == pytest.approx(np.std(wanted), rel=0.1)


def test_detects_aircraft_event_with_heuristic_label():
    aircraft = SyntheticAircraft("aaaaaa", 60.0, 350.0, 0.0, 75.0, 75.0, 450.0, 90.0)
    scene = build_scene([aircraft], noise_rms=0.02, seed=3)
    events = detect_events(scene.audio, scene.sample_rate, SpectralHeuristicClassifier())
    assert events
    assert events[0].score_kind == "heuristic"
    assert events[0].category == "unclassified_tonal"


def test_noise_only_recording_has_no_events():
    audio = 1e-4 * np.random.default_rng(1).normal(size=30 * 8000)
    assert detect_events(audio, 8000, SpectralHeuristicClassifier()) == []
