import gzip
import json

import httpx
import numpy as np
import pytest

from backend.core.geo import LocalFrame
from backend.flight_data.normalization.fusion import fuse_records
from backend.flight_data.normalization.units import parse_utc
from backend.flight_data.providers.adsbexchange import ADSBExchangeProvider, parse_trace
from backend.flight_data.providers.base import (
    HttpSession,
    ProviderAccessDenied,
    ProviderCapabilityError,
    ProviderRateLimited,
    ResponseCache,
)
from backend.flight_data.providers.flightradar24 import Flightradar24Provider
from backend.flight_data.providers.local_dataset import LocalFlightDatasetProvider, write_records_jsonl
from backend.flight_data.providers.opensky import OpenSkyProvider
from backend.flight_data.reconstruction.trajectory import reconstruct
from backend.flight_data.service import pad_box, retrieve_candidates
from backend.flight_data.types import (
    AircraftIdentity,
    AltitudeKind,
    BoundingBox,
    FlightRecord,
    PositionObservation,
    ProviderCapability,
    TimeWindow,
    ValueOrigin,
)

BOX = BoundingBox(51.0, -1.0, 52.0, 0.0)
T0 = 1_750_000_000.0


def session_for(handler):
    return HttpSession("test", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _: None)


def point(t, lat=51.5, lon=-0.5, alt=1000.0, speed=100.0, track=90.0, provider="p", source="ADSB", origin=ValueOrigin.REPORTED):
    return PositionObservation(t, lat, lon, alt, AltitudeKind.GEOMETRIC, speed, track, 0.0, False, provider, source, origin)


def token_client():
    return httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"access_token": "t", "expires_in": 1800})))


def test_parse_utc_requires_timezone():
    assert parse_utc("2025-06-15T15:06:40Z") == 1750000000.0
    with pytest.raises(ValueError):
        parse_utc("2025-06-15T15:06:40")


def test_flightradar24_converts_units_and_sends_documented_headers():
    seen = {}
    row = {
        "fr24_id": "39a84c3c",
        "hex": "4CA2BF",
        "callsign": "RYR1 ",
        "flight": "FR1",
        "lat": 51.5,
        "lon": -0.5,
        "track": 90,
        "alt": 10000,
        "gspeed": 200,
        "vspeed": 1000,
        "timestamp": "2025-06-15T15:06:40Z",
        "source": "ADSB",
        "type": "B738",
        "reg": "EI-XYZ",
    }

    def handler(request):
        seen[request.url.path] = request
        return httpx.Response(200, json={"data": [row]})

    provider = Flightradar24Provider("key", "https://fr24.test/api", session_for(handler), 60)
    records = provider.search_area(BOX, TimeWindow(T0, T0))
    request = seen["/api/historic/flight-positions/full"]
    assert request.headers["Accept-Version"] == "v1"
    assert request.headers["Authorization"] == "Bearer key"
    assert request.url.params["bounds"] == "52.000,51.000,-1.000,0.000"
    observation = records[0].observations[0]
    assert records[0].identity.icao24 == "4ca2bf"
    assert records[0].identity.callsign == "RYR1"
    assert observation.altitude_m == pytest.approx(3048.0)
    assert observation.ground_speed_mps == pytest.approx(102.89, abs=0.01)
    assert observation.vertical_rate_mps == pytest.approx(5.08)
    assert observation.altitude_kind is AltitudeKind.BAROMETRIC


def test_flightradar24_track_endpoint_marks_estimated_points():
    point_row = {"timestamp": "2025-06-15T15:06:40Z", "lat": 51, "lon": 0, "alt": 1000, "gspeed": 100, "vspeed": 0, "track": 10, "source": "ESTIMATED"}
    payload = [{"fr24_id": "abc", "tracks": [point_row]}]
    provider = Flightradar24Provider("k", "https://fr24.test/api", session_for(lambda r: httpx.Response(200, json=payload)))
    record = provider.get_track(AircraftIdentity(provider_flight_id="abc"), TimeWindow(T0, T0))
    assert record.observations[0].origin is ValueOrigin.INFERRED


def test_missing_key_is_reported_as_unavailable():
    provider = Flightradar24Provider("", "https://fr24.test/api", session_for(lambda r: httpx.Response(500)))
    status = provider.check_availability()
    assert not status.configured and not status.available
    assert not provider.is_configured()


def test_rate_limit_retries_then_surfaces():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "0"})

    with pytest.raises(ProviderRateLimited):
        session_for(handler).get_json("https://x.test/a")
    assert len(calls) == 5


def test_plan_denial_is_a_distinct_error():
    with pytest.raises(ProviderAccessDenied):
        session_for(lambda r: httpx.Response(402)).get_json("https://x.test/a")


def test_response_cache_expires(tmp_path):
    cache = ResponseCache(tmp_path, ttl_s=-1.0)
    cache.put("k", {"a": 1})
    assert cache.get("k") is None


def test_adsbexchange_has_no_historical_area_search_and_uses_hex_path():
    seen = []
    trace = {"icao": "a1b2c3", "timestamp": T0, "r": "N1", "t": "C172", "trace": [[0.0, 51.0, 0.0, 1000, 100, 90, 0, 0, None]]}

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=trace)

    provider = ADSBExchangeProvider("key", "https://adsbx.test/api/aircraft/v2", session_for(handler))
    assert not provider.supports(ProviderCapability.HISTORIC_AREA_SEARCH)
    with pytest.raises(ProviderCapabilityError):
        provider.search_area(BOX, TimeWindow(T0, T0))
    record = provider.get_track(AircraftIdentity(icao24="A1B2C3"), TimeWindow(T0, T0))
    assert seen[0].url.path.endswith("/traces-hist/2025/06/15/traces/c3/trace_full_a1b2c3.json")
    assert seen[0].headers["x-api-key"] == "key"
    assert record.identity.registration == "N1"


def test_trace_flags_and_fields():
    payload = {
        "icao": "abc123",
        "timestamp": 1000.0,
        "trace": [
            [0.0, 51.0, 0.0, 1000, 100.0, 90.0, 1, 0, None],
            [1.0, 51.0, 0.0, "ground", 0.0, 90.0, 0, None, None],
            [2.0, 51.0, 0.0, 1000, 100.0, 90.0, 8, 0, None],
            [3.0, 51.0, 0.0, 1000, 100.0, 90.0, 0, 0, {"flight": "ABC12  "}, "mlat", 1100],
        ],
    }
    record = parse_trace(payload)
    stale, ground, geometric, latest = record.observations
    assert stale.stale and stale.altitude_kind is AltitudeKind.BAROMETRIC
    assert ground.on_ground
    assert geometric.altitude_kind is AltitudeKind.GEOMETRIC
    assert latest.altitude_m == pytest.approx(335.28)
    assert latest.origin is ValueOrigin.MEASURED
    assert record.identity.callsign == "ABC12"


def test_opensky_enforces_documented_history_limits():
    provider = OpenSkyProvider("id", "secret", "https://os.test/api", "https://os.test/token", session_for(lambda r: httpx.Response(500)), token_client(), clock=lambda: T0)
    with pytest.raises(ProviderCapabilityError):
        provider.search_area(BOX, TimeWindow(T0 - 7200, T0 - 7100))
    assert provider.get_track(AircraftIdentity(icao24="abc123"), TimeWindow(T0 - 40 * 86400, T0)) is None


def test_opensky_parses_state_vector_order():
    row = ["abc123", "ABC1    ", "UK", T0, T0, -0.5, 51.5, 900.0, False, 80.0, 45.0, 2.0, None, 950.0, None, False, 2, 1]

    def handler(request):
        assert request.headers["Authorization"] == "Bearer t"
        return httpx.Response(200, json={"time": T0, "states": [row]})

    provider = OpenSkyProvider("id", "secret", "https://os.test/api", "https://os.test/token", session_for(handler), token_client(), clock=lambda: T0)
    records = provider.search_area(BOX, TimeWindow(T0 - 10, T0))
    observation = records[0].observations[0]
    assert (observation.latitude, observation.longitude) == (51.5, -0.5)
    assert observation.altitude_m == 950.0
    assert observation.source_type == "MLAT"
    assert records[0].identity.callsign == "ABC1"


def test_fusion_merges_by_hex_and_keeps_provenance():
    a = FlightRecord(AircraftIdentity(icao24="ABC123", callsign="AB1"), [point(0.0, provider="fr24"), point(10.0, provider="fr24")], ["fr24"])
    b = FlightRecord(AircraftIdentity(icao24="abc123", registration="N1"), [point(0.4, provider="adsbx"), point(20.0, provider="adsbx")], ["adsbx"])
    other = FlightRecord(AircraftIdentity(icao24="def456"), [point(0.0)], ["fr24"])
    fused = fuse_records([a, b, other])
    assert len(fused) == 2
    merged = next(item for item in fused if item.identity.icao24 == "abc123")
    assert merged.identity.registration == "N1"
    assert merged.providers == ["adsbx", "fr24"]
    assert len(merged.observations) == 3
    assert sum(merged.provenance_counts.values()) == 3


def test_two_adsb_providers_are_not_independent_confirmation():
    a = FlightRecord(AircraftIdentity(icao24="abc123"), [point(0.0, provider="fr24", source="ADSB")], ["fr24"])
    b = FlightRecord(AircraftIdentity(icao24="abc123"), [point(5.0, provider="adsbx", source="adsb_icao")], ["adsbx"])
    merged = fuse_records([a, b])[0]
    assert merged.independent_signal_sources == 1
    assert merged.provider_agreement_is_independent is False


def test_mlat_and_adsb_are_independent_signal_sources():
    a = FlightRecord(AircraftIdentity(icao24="abc123"), [point(0.0, source="ADSB")], ["fr24"])
    b = FlightRecord(AircraftIdentity(icao24="abc123"), [point(5.0, source="mlat")], ["adsbx"])
    assert fuse_records([a, b])[0].provider_agreement_is_independent is True


def test_altitude_conflict_is_recorded():
    a = FlightRecord(AircraftIdentity(icao24="abc123"), [point(0.0, alt=1000.0, provider="fr24")], ["fr24"])
    b = FlightRecord(AircraftIdentity(icao24="abc123"), [point(0.5, alt=1400.0, provider="adsbx")], ["adsbx"])
    assert "altitude differs" in fuse_records([a, b])[0].conflicts[0]


def test_callsign_only_records_need_trajectory_agreement():
    near = FlightRecord(AircraftIdentity(callsign="AB1"), [point(0.0), point(10.0)], ["a"])
    close = FlightRecord(AircraftIdentity(callsign="AB1"), [point(2.0, lon=-0.49)], ["b"])
    far = FlightRecord(AircraftIdentity(callsign="AB1"), [point(2.0, lon=3.0)], ["c"])
    assert len(fuse_records([near, close])) == 1
    assert len(fuse_records([near, far])) == 2


def test_reconstruction_honours_reported_velocity_and_bounds_interpolation():
    frame = LocalFrame(51.5, -0.5)
    observations = [point(float(t), lon=-0.5 + 0.0014 * t, track=90.0) for t in range(0, 120, 10)]
    trajectory = reconstruct(observations, frame)
    _, velocity = trajectory.state(np.array([35.0]))
    assert velocity[0, 0] == pytest.approx(100.0, rel=0.1)
    assert trajectory.velocity_measured.all()
    assert trajectory.interpolation_sigma_m(np.array([5.0]))[0] < trajectory.interpolation_sigma_m(np.array([1e6]))[0]


def test_reconstruction_suppresses_reported_noise_without_flattening_turns():
    frame = LocalFrame(51.5, -0.5)
    generator = np.random.default_rng(3)
    rate = np.radians(3.0)
    speed = 100.0
    times = np.arange(0.0, 240.0, 5.0)
    east = speed / rate * np.sin(rate * times) + generator.normal(0.0, 15.0, times.size)
    north = speed / rate * (1.0 - np.cos(rate * times)) + generator.normal(0.0, 15.0, times.size)
    latitude, longitude, _ = frame.from_enu(np.column_stack([east, north, np.full(times.size, 800.0)]))
    observations = [
        point(
            float(t),
            lat=float(a),
            lon=float(b),
            speed=speed + float(generator.normal(0.0, 1.0)),
            track=float(np.degrees(np.arctan2(np.cos(rate * t), np.sin(rate * t))) % 360.0),
        )
        for t, a, b in zip(times, latitude, longitude)
    ]
    trajectory = reconstruct(observations, frame)
    query = np.arange(30.0, 210.0, 1.0)
    _, velocity = trajectory.state(query)
    truth = np.column_stack([speed * np.cos(rate * query), speed * np.sin(rate * query)])
    assert np.sqrt(np.mean((velocity[:, :2] - truth) ** 2)) < 1.2
    assert trajectory.velocity_sigma_mps < 1.0


def test_reconstruction_rejects_position_spikes_and_estimated_points():
    frame = LocalFrame(51.5, -0.5)
    observations = [point(float(t), lon=-0.5 + 0.0014 * t) for t in range(0, 100, 10)]
    observations[4] = point(40.0, lon=5.0)
    observations.append(point(95.0, lon=8.0, origin=ValueOrigin.INFERRED))
    trajectory = reconstruct(observations, frame)
    assert 40.0 not in trajectory.times
    assert 95.0 not in trajectory.times


def test_reconstruction_marks_reporting_gaps():
    frame = LocalFrame(51.5, -0.5)
    observations = [point(0.0), point(5.0, lon=-0.493), point(120.0, lon=-0.33), point(125.0, lon=-0.323)]
    trajectory = reconstruct(observations, frame)
    assert len(trajectory.gaps) == 1
    assert trajectory.has_gap_within(50.0, 60.0)
    assert trajectory.completeness(0.0, 125.0) < 0.2
    assert trajectory.interpolation_sigma_m(np.array([62.0]))[0] > 500.0


def test_local_dataset_round_trip_and_filters(tmp_path):
    record = FlightRecord(AircraftIdentity(icao24="abc123", callsign="AB1"), [point(T0), point(T0 + 10.0)], ["x"])
    write_records_jsonl(tmp_path / "flights.jsonl", [record])
    provider = LocalFlightDatasetProvider(tmp_path)
    assert provider.search_area(BOX, TimeWindow(T0 - 1, T0 + 1))[0].identity.callsign == "AB1"
    assert provider.search_area(BOX, TimeWindow(0, 1)) == []
    assert provider.search_area(BoundingBox(10.0, 10.0, 11.0, 11.0), TimeWindow(T0, T0 + 1)) == []


def test_local_dataset_ignores_placeholder_files(tmp_path):
    (tmp_path / ".gitkeep").write_text("")
    provider = LocalFlightDatasetProvider(tmp_path)
    assert not provider.is_configured()
    assert provider.search_area(BOX, TimeWindow(T0, T0 + 1)) == []


def test_local_dataset_reads_gzipped_trace(tmp_path):
    trace = {"icao": "abc123", "timestamp": T0, "trace": [[0.0, 51.5, -0.5, 1000, 100, 90, 0, 0, None]]}
    with gzip.open(tmp_path / "trace_full_abc123.json", "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)
    assert len(LocalFlightDatasetProvider(tmp_path).search_area(BOX, TimeWindow(T0 - 1, T0 + 1))) == 1


def test_retrieval_reports_provider_failure_without_candidates():
    provider = Flightradar24Provider("k", "https://fr24.test/api", session_for(lambda r: httpx.Response(402)))
    outcome = retrieve_candidates([provider], BOX, TimeWindow(T0, T0 + 60), LocalFrame(51.5, -0.5))
    assert outcome.candidates == []
    assert "flightradar24" in outcome.provider_errors
    assert outcome.area_search_available is False


def test_pad_box_clamps_to_valid_range():
    padded = pad_box(BoundingBox(89.0, 179.0, 89.5, 179.9), 500000.0)
    assert padded.north == 90.0 and padded.east == 180.0
