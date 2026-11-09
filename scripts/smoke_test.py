from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

import httpx
import websockets.sync.client

TERMINAL = ("completed", "failed", "cancelled", "rejected")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a recording through a running AeroLocate stack")
    parser.add_argument("--base", default="http://localhost:8080")
    parser.add_argument("--audio", type=Path, default=Path("data/demo/SYNTHETIC_recording.wav"))
    parser.add_argument("--start", type=float, default=1_750_000_000.0)
    parser.add_argument("--latitude", type=float, default=51.47)
    parser.add_argument("--longitude", type=float, default=-0.45)
    parser.add_argument("--timeout", type=float, default=600.0)
    arguments = parser.parse_args()
    client = httpx.Client(base_url=arguments.base, timeout=120.0)
    email = f"smoke-{secrets.token_hex(4)}@example.com"
    password = secrets.token_urlsafe(18)
    client.post("/api/auth/register", json={"email": email, "password": password}).raise_for_status()
    token = client.post("/api/auth/token", data={"username": email, "password": password}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    parameters = {
        "region": {
            "south": arguments.latitude - 0.09,
            "west": arguments.longitude - 0.14,
            "north": arguments.latitude + 0.05,
            "east": arguments.longitude + 0.11,
        },
        "start_earliest_utc": arguments.start,
        "start_latest_utc": arguments.start,
        "authorization_basis": "own_recording",
        "providers": ["local_dataset"],
    }
    with arguments.audio.open("rb") as handle:
        response = client.post(
            "/api/analyses",
            headers=headers,
            files={"file": (arguments.audio.name, handle)},
            data={"parameters": json.dumps(parameters)},
        )
    response.raise_for_status()
    analysis_id = response.json()["id"]
    socket_url = arguments.base.replace("http", "ws", 1) + f"/api/analyses/{analysis_id}/progress"
    stages: list[str] = []
    with websockets.sync.client.connect(socket_url) as socket:
        socket.send(json.dumps({"token": token}))
        deadline = time.time() + arguments.timeout
        while time.time() < deadline:
            message = json.loads(socket.recv(timeout=arguments.timeout))
            if message["stage"] not in stages:
                stages.append(message["stage"])
            if message["state"] in TERMINAL:
                break
    body = client.get(f"/api/analyses/{analysis_id}", headers=headers).json()
    result = body["result"] or {}
    print(
        json.dumps(
            {
                "state": body["state"],
                "error": body["error"],
                "stages": stages,
                "classifier": (body["acoustic"] or {}).get("classifier"),
                "classifier_note": (body["acoustic"] or {}).get("classifier_note"),
                "detections": len((body["acoustic"] or {}).get("detections", [])),
                "status": result.get("status"),
                "latitude": result.get("latitude"),
                "longitude": result.get("longitude"),
                "radius_m": result.get("uncertainty_radius_m"),
                "matched": sorted({match["flight_key"] for match in result.get("matches", [])}),
                "pdf_bytes": len(client.get(f"/api/analyses/{analysis_id}/report.pdf", headers=headers).content),
                "analysis_id": analysis_id,
            },
            indent=2,
        )
    )
    return 0 if body["state"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
