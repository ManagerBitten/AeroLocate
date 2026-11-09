import json
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.core.config import get_settings
from backend.flight_data.providers.local_dataset import write_records_jsonl
from backend.flight_data.types import AircraftIdentity
from backend.validation.synthetic import (
    SyntheticAircraft,
    build_scene,
    observations_from_trajectory,
)

ALPHA = SyntheticAircraft("aaaaaa", 60.0, 350.0, 0.0, 75.0, 75.0, 450.0, 90.0)
BRAVO = SyntheticAircraft("bbbbbb", 50.0, -600.0, 0.0, 200.0, 62.0, 380.0, 147.0)
PASSWORD = "a-long-enough-password"
SMALL_REQUEST = dict(
    region=dict(south=1, west=1, north=2, east=2),
    start_earliest_utc=0,
    start_latest_utc=0,
    authorization_basis="own_recording",
)


@pytest.fixture(scope="module")
def client():
    from backend.api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    built = build_scene([ALPHA, BRAVO], noise_rms=0.02, seed=5)
    path = tmp_path_factory.mktemp("media") / "recording.wav"
    sf.write(path, built.audio.astype(np.float32) * 0.2, built.sample_rate)
    settings = get_settings()
    settings.local_flight_dataset_dir.mkdir(parents=True, exist_ok=True)
    records = [
        observations_from_trajectory(AircraftIdentity(icao24=item.icao24, callsign=f"T{i}"), trajectory, built.frame)
        for i, (item, trajectory) in enumerate(zip(built.aircraft, built.trajectories))
    ]
    write_records_jsonl(settings.local_flight_dataset_dir / "synthetic.jsonl", records)
    return built, path


def login(client, email):
    assert client.post("/api/auth/register", json={"email": email, "password": PASSWORD}).status_code == 201
    token = client.post("/api/auth/token", data={"username": email, "password": PASSWORD}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def parameters(built, **extra):
    values = dict(
        region=dict(
            south=built.microphone_latitude - 0.09,
            west=built.microphone_longitude - 0.14,
            north=built.microphone_latitude + 0.05,
            east=built.microphone_longitude + 0.11,
        ),
        start_earliest_utc=built.recording_start_utc,
        start_latest_utc=built.recording_start_utc,
        authorization_basis="own_recording",
        providers=["local_dataset"],
    )
    values.update(extra)
    return json.dumps(values)


def upload(client, headers, path, body):
    with path.open("rb") as handle:
        return client.post("/api/analyses", headers=headers, files={"file": (path.name, handle)}, data={"parameters": body})


def wait_for(client, analysis_id, headers, timeout=240.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/analyses/{analysis_id}", headers=headers).json()
        if body["state"] in ("completed", "failed", "cancelled", "rejected"):
            return body
        time.sleep(0.25)
    raise AssertionError("analysis did not finish")


def test_endpoints_require_authentication(client):
    assert client.get("/api/analyses").status_code == 401
    assert client.get("/api/providers").status_code == 401


def test_weak_password_rejected(client):
    assert client.post("/api/auth/register", json={"email": "a@b.co", "password": "short"}).status_code == 422


def test_spoofed_media_rejected(client):
    headers = login(client, "spoof@example.com")
    response = client.post(
        "/api/analyses",
        headers=headers,
        files={"file": ("x.wav", b"MZ" + b"\0" * 2000, "audio/wav")},
        data={"parameters": json.dumps(SMALL_REQUEST)},
    )
    assert response.status_code == 415


def test_unsupported_extension_rejected(client):
    headers = login(client, "ext@example.com")
    response = client.post(
        "/api/analyses",
        headers=headers,
        files={"file": ("x.exe", b"MZ", "application/octet-stream")},
        data={"parameters": json.dumps(SMALL_REQUEST)},
    )
    assert response.status_code == 415


def test_authorization_basis_is_mandatory(client, scene):
    built, path = scene
    headers = login(client, "noauth@example.com")
    values = json.loads(parameters(built))
    values.pop("authorization_basis")
    assert upload(client, headers, path, json.dumps(values)).status_code == 422


def test_full_pipeline_upload_to_located_result(client, scene):
    built, path = scene
    headers = login(client, "owner@example.com")
    created = upload(client, headers, path, parameters(built))
    assert created.status_code == 202, created.text
    analysis_id = created.json()["id"]
    with client.websocket_connect(f"/api/analyses/{analysis_id}/progress") as socket:
        socket.send_json({"token": headers["Authorization"].split()[1]})
        assert "stage" in socket.receive_json()
    body = wait_for(client, analysis_id, headers)
    assert body["state"] == "completed", body["error"]
    assert body["localization_status"] == "localized"
    result = body["result"]
    assert result["latitude"] is not None
    assert {match["flight_key"] for match in result["matches"]} == {"aaaaaa", "bbbbbb"}
    assert all(curve["sigma_hz"] for curve in body["acoustic"]["curves"])
    assert len(result["candidates"]) == 2
    assert client.get(f"/api/analyses/{analysis_id}/waveform", headers=headers).status_code == 200
    spectrogram = client.get(f"/api/analyses/{analysis_id}/spectrogram", headers=headers).json()
    assert len(spectrogram["decibels"]) == len(spectrogram["frequencies"])
    assert client.get(f"/api/analyses/{analysis_id}/spectrogram.png", headers=headers).content[:4] == b"\x89PNG"
    assert client.get(f"/api/analyses/{analysis_id}/report.pdf", headers=headers).content[:4] == b"%PDF"
    assert len(client.get(f"/api/analyses/{analysis_id}/report.json", headers=headers).json()["sha256"]) == 64

    stranger = login(client, "stranger@example.com")
    assert client.get(f"/api/analyses/{analysis_id}", headers=stranger).status_code == 404
    assert client.get(f"/api/analyses/{analysis_id}/audio", headers=stranger).status_code == 404
    assert client.delete(f"/api/analyses/{analysis_id}", headers=stranger).status_code == 404

    assert client.delete(f"/api/analyses/{analysis_id}", headers=headers).status_code == 204
    assert client.get(f"/api/analyses/{analysis_id}", headers=headers).status_code == 404
    settings = get_settings()
    assert not (settings.upload_dir / analysis_id).exists()
    assert not (settings.artifact_dir / analysis_id).exists()


def test_recording_without_flights_is_not_localized(client, scene):
    built, path = scene
    headers = login(client, "noflights@example.com")
    created = upload(client, headers, path, parameters(built, region=dict(south=10.0, west=10.0, north=10.2, east=10.2)))
    body = wait_for(client, created.json()["id"], headers)
    assert body["localization_status"] == "not_localizable"
    assert body["result"]["latitude"] is None
    assert body["result"]["regions"] == []


def test_workload_estimate_explains_loose_constraints(client):
    headers = login(client, "workload@example.com")
    body = client.post(
        "/api/workload",
        headers=headers,
        json={"region": dict(south=30, west=-10, north=60, east=30), "start_earliest_utc": 0, "start_latest_utc": 86400},
    ).json()
    assert body["feasible"] is False
    assert body["advice"]


def test_audit_log_is_restricted(client):
    headers = login(client, "plain@example.com")
    assert client.get("/api/audit", headers=headers).status_code == 403


def test_cancellation_reaches_a_terminal_state(client, scene):
    built, path = scene
    headers = login(client, "cancel@example.com")
    analysis_id = upload(client, headers, path, parameters(built)).json()["id"]
    assert client.post(f"/api/analyses/{analysis_id}/cancel", headers=headers).status_code == 200
    assert wait_for(client, analysis_id, headers)["state"] == "cancelled"
