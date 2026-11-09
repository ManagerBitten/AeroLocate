from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from backend.audio.analysis import (
    Preprocessing,
    apply_preprocessing,
    compute_spectrogram,
    compute_waveform,
)
from backend.audio.ingest import MediaRejected, extract_audio, load_audio, probe_media
from backend.core.atmosphere import Atmosphere
from backend.core.config import Settings
from backend.core.geo import LocalFrame
from backend.detection.events import (
    AircraftEvent,
    SoundClassifier,
    SpectralHeuristicClassifier,
    detect_events,
    overlapping_detection,
)
from backend.doppler.engine import DopplerCurve, extract_doppler_curves
from backend.flight_data.normalization.fusion import FusedFlight
from backend.flight_data.service import (
    CandidateFlight,
    RetrievalOutcome,
    build_providers,
    pad_box,
    retrieve_candidates,
)
from backend.flight_data.types import BoundingBox, TimeWindow
from backend.geolocation.engine import (
    LocalizationRequest,
    LocalizationResult,
    localize,
)
from backend.geolocation.search import TrackedFlight
from backend.models.database import geometry_value, session_scope
from backend.models.entities import (
    Analysis,
    DopplerObservation,
    FlightCandidateRow,
    ProviderObservation,
    RegionRow,
)

SEARCH_RADIUS_MARGIN_M = 30000.0
TRAJECTORY_EXPORT_LIMIT = 400


class JobCancelled(Exception):
    pass


class JobContext:
    def __init__(self, analysis_id: str) -> None:
        self.analysis_id = analysis_id

    def report(self, stage: str, progress: float) -> None:
        with session_scope() as session:
            analysis = session.get(Analysis, self.analysis_id)
            if analysis is None:
                raise JobCancelled("analysis was deleted")
            analysis.stage = stage
            analysis.progress = float(np.clip(progress, 0.0, 1.0))
            if analysis.cancel_requested:
                raise JobCancelled("cancelled by the user")


@lru_cache(maxsize=1)
def _load_ast(model_name: str, device: str) -> SoundClassifier:
    from backend.detection.ast_classifier import AudioSpectrogramTransformerClassifier

    return AudioSpectrogramTransformerClassifier(model_name, device)


def _build_classifier(settings: Settings) -> tuple[SoundClassifier, str | None]:
    if settings.classifier != "ast":
        return SpectralHeuristicClassifier(), None
    try:
        return _load_ast(settings.ast_model_name, settings.torch_device), None
    except (ImportError, OSError) as error:
        note = f"The aircraft sound model could not be loaded ({type(error).__name__}), so sounds are not classified."
        return SpectralHeuristicClassifier(), note


def serialize_curve(curve: DopplerCurve, detections: list[AircraftEvent]) -> dict[str, Any]:
    fit = curve.fit
    frequency = np.exp(curve.log_frequency)
    return {
        "event_id": curve.event_id,
        "detection_id": overlapping_detection(detections, curve.start, curve.end),
        "times_s": np.round(curve.times, 3).tolist(),
        "frequency_hz": np.round(frequency, 3).tolist(),
        "sigma_hz": np.round(frequency * curve.effective_sigma, 4).tolist(),
        "start_s": curve.start,
        "end_s": curve.end,
        "harmonic_numbers": list(curve.harmonic_numbers),
        "component_frequencies_hz": list(curve.component_frequencies_hz),
        "window_s": curve.window_s,
        "median_snr_db": round(curve.median_snr_db, 1),
        "curve_class": curve.curve_class.value,
        "quality": round(curve.quality, 3),
        "usable": curve.usable,
        "noise_inflation": round(curve.noise_inflation, 2),
        "closest_approach_s": None if fit is None else round(fit.closest_time, 2),
        "closest_approach_sigma_s": None if fit is None else round(fit.closest_time_sigma * curve.noise_inflation, 3),
        "implied_speed_mps": None if fit is None else round(fit.speed_mps, 1),
        "implied_speed_sigma_mps": None if fit is None else round(fit.speed_sigma * curve.noise_inflation, 1),
        "implied_slant_range_m": None if fit is None else round(fit.distance_m),
        "implied_slant_range_sigma_m": None if fit is None else round(fit.distance_sigma * curve.noise_inflation),
        "source_frequency_hz": None if fit is None else round(fit.source_frequency_hz, 2),
    }


def _candidate_summary(candidate: CandidateFlight, frame: LocalFrame, window: TimeWindow) -> dict[str, Any]:
    flight = candidate.fused
    trajectory = candidate.trajectory
    in_window = (trajectory.times >= window.start - 120.0) & (trajectory.times <= window.end + 120.0)
    indices = np.flatnonzero(in_window)
    if len(indices) > TRAJECTORY_EXPORT_LIMIT:
        indices = indices[:: len(indices) // TRAJECTORY_EXPORT_LIMIT + 1]
    latitude, longitude, altitude = frame.from_enu(trajectory.positions[indices])
    return {
        "key": candidate.key,
        "identity": asdict(flight.identity),
        "providers": flight.providers,
        "provenance_counts": flight.provenance_counts,
        "independent_signal_sources": flight.independent_signal_sources,
        "provider_agreement_is_independent": flight.provider_agreement_is_independent,
        "conflicts": flight.conflicts,
        "completeness": round(trajectory.completeness(window.start, window.end), 3),
        "gaps": [asdict(gap) for gap in trajectory.gaps if gap.start < window.end and gap.end > window.start],
        "observation_count": len(flight.observations),
        "path": {
            "times": trajectory.times[indices].tolist(),
            "latitude": np.round(latitude, 6).tolist(),
            "longitude": np.round(longitude, 6).tolist(),
            "altitude_m": np.round(altitude, 1).tolist(),
        },
    }


def serialize_result(result: LocalizationResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["status"] = result.status.value
    payload["regions"] = [asdict(region) for region in result.regions]
    payload["rejected_pairs"] = [asdict(item) for item in result.rejected_pairs[:200]]
    payload["refined"] = None if result.refined is None else asdict(result.refined)
    payload["comparison"] = [asdict(row) for row in result.comparison]
    return payload


def _time_bounds(request: dict[str, Any]) -> tuple[float, float]:
    earliest = float(request["start_earliest_utc"])
    latest = float(request["start_latest_utc"])
    return earliest, max(latest, earliest)


def _persist_flights(session: Any, analysis: Analysis, candidates: list[CandidateFlight], frame: LocalFrame) -> None:
    for candidate in candidates:
        flight: FusedFlight = candidate.fused
        latitude, longitude, altitude = frame.from_enu(candidate.trajectory.positions)
        step = max(1, len(latitude) // 2000)
        coordinates = ", ".join(
            f"{lon:.6f} {lat:.6f} {alt:.1f}"
            for lat, lon, alt in zip(latitude[::step], longitude[::step], altitude[::step])
        )
        row = FlightCandidateRow(
            analysis_id=analysis.id,
            flight_key=candidate.key,
            identity=asdict(flight.identity),
            providers=flight.providers,
            provenance={
                "counts": flight.provenance_counts,
                "conflicts": flight.conflicts,
                "independent_signal_sources": flight.independent_signal_sources,
            },
            track=geometry_value(session, f"LINESTRING Z ({coordinates})") if len(latitude[::step]) >= 2 else None,
        )
        for observation in flight.observations:
            row.observations.append(
                ProviderObservation(
                    timestamp=observation.timestamp,
                    latitude=observation.latitude,
                    longitude=observation.longitude,
                    altitude_m=observation.altitude_m,
                    altitude_kind=observation.altitude_kind.value,
                    ground_speed_mps=observation.ground_speed_mps,
                    track_deg=observation.track_deg,
                    vertical_rate_mps=observation.vertical_rate_mps,
                    provider=observation.provider,
                    source_type=observation.source_type,
                    origin=observation.origin.value,
                )
            )
        session.add(row)


def _persist_regions(session: Any, analysis: Analysis, result: LocalizationResult) -> None:
    for region in result.regions:
        rings = ", ".join(
            "((" + ", ".join(f"{lon:.6f} {lat:.6f}" for lon, lat in polygon) + "))" for polygon in region.polygons
        )
        session.add(
            RegionRow(
                analysis_id=analysis.id,
                rank=region.rank,
                probability=region.probability,
                area_km2=region.area_km2,
                shape=geometry_value(session, f"MULTIPOLYGON ({rings})") if rings else None,
            )
        )


def run_analysis(analysis_id: str, settings: Settings, source_path: Path) -> None:
    context = JobContext(analysis_id)
    with session_scope() as session:
        analysis = session.get(Analysis, analysis_id)
        if analysis is None:
            return
        analysis.state = "running"
        analysis.error = None
        analysis.doppler_points.clear()
        analysis.flights.clear()
        analysis.regions.clear()
        request = dict(analysis.request)
        extension = analysis.extension
    artifact_dir = settings.artifact_dir / analysis_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        _execute(context, settings, source_path, extension, request, artifact_dir)
    except JobCancelled as cancelled:
        _finish(analysis_id, "cancelled", str(cancelled))
    except MediaRejected as rejected:
        _finish(analysis_id, "rejected", str(rejected))
    except Exception as error:
        _finish(analysis_id, "failed", f"{type(error).__name__}: {error}")
        raise


def _finish(analysis_id: str, state: str, error: str | None) -> None:
    with session_scope() as session:
        analysis = session.get(Analysis, analysis_id)
        if analysis is not None:
            analysis.state = state
            analysis.stage = state
            analysis.error = error
            analysis.finished_at = datetime.now(timezone.utc)


def _execute(
    context: JobContext,
    settings: Settings,
    source_path: Path,
    extension: str,
    request: dict[str, Any],
    artifact_dir: Path,
) -> None:
    analysis_id = context.analysis_id
    context.report("probing media", 0.02)
    media = probe_media(source_path, extension, settings)
    with session_scope() as session:
        session.get(Analysis, analysis_id).media = media.to_dict()
    context.report("extracting audio", 0.05)
    original = extract_audio(source_path, artifact_dir / "original.wav", settings)
    audio, sample_rate = load_audio(original)
    context.report("rendering waveform and spectrogram", 0.1)
    waveform = compute_waveform(audio.astype(np.float64), sample_rate)
    spectrogram = compute_spectrogram(audio.astype(np.float64), sample_rate, artifact_dir / "spectrogram.png")
    (artifact_dir / "waveform.json").write_text(json.dumps(waveform), encoding="utf-8")
    (artifact_dir / "spectrogram.json").write_text(json.dumps(asdict(spectrogram)), encoding="utf-8")
    processed = apply_preprocessing(audio, sample_rate, Preprocessing(request.get("preprocessing", "none")))
    context.report("detecting aircraft sound", 0.18)
    classifier, classifier_note = _build_classifier(settings)
    detections = detect_events(processed, sample_rate, classifier)
    context.report("tracking Doppler curves", 0.25)
    curves = extract_doppler_curves(
        processed,
        sample_rate,
        progress=lambda fraction: context.report("tracking Doppler curves", 0.25 + 0.25 * fraction),
    )
    acoustic = {
        "duration_s": media.duration_s,
        "sample_rate": sample_rate,
        "classifier": classifier.name,
        "classifier_note": classifier_note,
        "detections": [event.to_dict() for event in detections],
        "curves": [serialize_curve(curve, detections) for curve in curves],
    }
    with session_scope() as session:
        analysis = session.get(Analysis, analysis_id)
        analysis.acoustic = acoustic
        for curve in curves:
            frequency = np.exp(curve.log_frequency)
            sigma = frequency * curve.effective_sigma
            session.add_all(
                DopplerObservation(
                    analysis_id=analysis_id,
                    event_id=curve.event_id,
                    time_s=float(t),
                    frequency_hz=float(f),
                    sigma_hz=float(s),
                )
                for t, f, s in zip(curve.times, frequency, sigma)
            )
    context.report("retrieving flight trajectories", 0.52)
    box = BoundingBox(**request["region"])
    earliest, latest = _time_bounds(request)
    window = TimeWindow(earliest - 60.0, latest + media.duration_s + 60.0)
    frame = LocalFrame(*box.center, 0.0)
    providers = build_providers(settings, request.get("providers") or None)
    outcome: RetrievalOutcome = retrieve_candidates(providers, pad_box(box, SEARCH_RADIUS_MARGIN_M), window, frame)
    with session_scope() as session:
        analysis = session.get(Analysis, analysis_id)
        _persist_flights(session, analysis, outcome.candidates, frame)
    context.report("inferring recording location", 0.6)
    atmosphere = Atmosphere(
        surface_temperature_c=float(request.get("temperature_c", settings.default_temperature_c)),
        surface_elevation_m=float(request.get("ground_elevation_m", settings.default_ground_elevation_m)),
        wind_east_mps=float(request.get("wind_east_mps", 0.0)),
        wind_north_mps=float(request.get("wind_north_mps", 0.0)),
        temperature_is_assumed="temperature_c" not in request,
        wind_is_assumed="wind_east_mps" not in request,
    )
    localization_request = LocalizationRequest(
        region=box,
        start_earliest=earliest,
        start_latest=latest,
        atmosphere=atmosphere,
        microphone_elevation_m=atmosphere.surface_elevation_m,
    )
    flights = [TrackedFlight(candidate.key, candidate.trajectory) for candidate in outcome.candidates]
    result = localize(
        curves,
        flights,
        localization_request,
        progress=lambda fraction: context.report("inferring recording location", 0.6 + 0.35 * fraction),
    )
    if outcome.provider_errors:
        result.diagnostics["provider_errors"] = outcome.provider_errors
    if not outcome.area_search_available:
        result.reasons.append(
            "No configured provider could search this area and time; historical access depends on API keys and plan."
        )
    matched_events = {match.event_id for match in result.matches}
    unrecognized = [
        curve.event_id
        for curve in curves
        if curve.event_id in matched_events and overlapping_detection(detections, curve.start, curve.end) is None
    ]
    if unrecognized and classifier.name != SpectralHeuristicClassifier.name:
        result.reasons.append(
            f"The sound model did not recognize aircraft during {', '.join(unrecognized)}. "
            "This match rests on the Doppler curve alone, so check the source of the sound by listening."
        )
    if media.lossy:
        result.diagnostics["lossy_audio"] = f"audio codec {media.codec} may distort weak tones"
    payload = serialize_result(result)
    payload["candidates"] = [_candidate_summary(candidate, frame, window) for candidate in outcome.candidates]
    payload["provider_quality"] = [asdict(report) for report in outcome.quality]
    payload["assumptions"] = {
        "temperature_c": atmosphere.surface_temperature_c,
        "temperature_is_assumed": atmosphere.temperature_is_assumed,
        "wind_east_mps": atmosphere.wind_east_mps,
        "wind_north_mps": atmosphere.wind_north_mps,
        "wind_is_assumed": atmosphere.wind_is_assumed,
        "ground_elevation_m": atmosphere.surface_elevation_m,
        "recording_clock_window_s": latest - earliest,
    }
    with session_scope() as session:
        analysis = session.get(Analysis, analysis_id)
        analysis.result = payload
        analysis.localization_status = result.status.value
        if result.latitude is not None and result.longitude is not None:
            analysis.estimate = geometry_value(session, f"POINT ({result.longitude} {result.latitude})")
        _persist_regions(session, analysis, result)
        analysis.state = "completed"
        analysis.stage = "completed"
        analysis.progress = 1.0
        analysis.finished_at = datetime.now(timezone.utc)


def delete_artifacts(analysis_id: str, settings: Settings) -> None:
    shutil.rmtree(settings.artifact_dir / analysis_id, ignore_errors=True)
    shutil.rmtree(settings.upload_dir / analysis_id, ignore_errors=True)
