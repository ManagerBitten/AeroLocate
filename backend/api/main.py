from __future__ import annotations

import asyncio
import hashlib
import json
import math
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, AsyncIterator, Awaitable, Callable

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from backend.api.report import render_pdf, report_payload
from backend.api.schemas import (
    AnalysisDetail,
    AnalysisRequest,
    AnalysisSummary,
    Credentials,
    TokenResponse,
    WorkloadEstimate,
    WorkloadRequest,
)
from backend.api.security import (
    RateLimiter,
    audit,
    create_token,
    current_user,
    hash_password,
    signing_key,
    user_from_token,
    verify_password,
)
from backend.audio.ingest import ALLOWED_EXTENSIONS, MediaRejected, probe_media
from backend.core.config import Settings, get_settings
from backend.flight_data.service import build_providers, pad_box
from backend.flight_data.types import BoundingBox, TimeWindow
from backend.models.database import geometry_value, get_session, init_database, session_scope
from backend.models.entities import Analysis, AuditLog, User
from backend.workers.dispatch import enqueue_analysis
from backend.workers.pipeline import delete_artifacts
from backend.workers.retention import purge_expired_analyses

CHUNK_BYTES = 1024 * 1024
ACTIVE_STATES = ("queued", "running")
TERMINAL_STATES = ("completed", "failed", "cancelled", "rejected")
METERS_PER_DEGREE = 111_320.0
SEARCH_CELL_M = 1600.0
SEARCH_TIME_STEP_S = 8.0
MAX_FEASIBLE_TILES = 4_000_000
MAX_FEASIBLE_PROVIDER_REQUESTS = 2000

SessionDep = Annotated[Session, Depends(get_session)]
UserDep = Annotated[User, Depends(current_user)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    signing_key(settings)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    init_database()
    await run_in_threadpool(purge_expired_analyses, settings)
    app.state.limiter = RateLimiter(
        settings.redis_url if settings.job_backend == "celery" else None, settings.rate_limit_per_minute
    )
    yield


app = FastAPI(title="AeroLocate", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in get_settings().cors_origins.split(",") if origin.strip()],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def rate_limit(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    limiter: RateLimiter | None = getattr(request.app.state, "limiter", None)
    if limiter is not None and request.url.path.startswith("/api") and request.url.path != "/api/health":
        key = request.client.host if request.client else "unknown"
        if not await run_in_threadpool(limiter.allow, key):
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429, headers={"Retry-After": "60"})
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def owned_analysis(analysis_id: str, session: Session, user: User) -> Analysis:
    analysis = session.get(Analysis, analysis_id)
    if analysis is None or analysis.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "analysis not found")
    return analysis


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/register", status_code=201)
def register(body: Credentials, request: Request, session: SessionDep, settings: SettingsDep) -> dict[str, str]:
    if not settings.allow_registration:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "registration is disabled")
    email = body.email.lower()
    if session.scalar(select(User).where(User.email == email)) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")
    user = User(email=email, password_hash=hash_password(body.password))
    session.add(user)
    session.flush()
    audit(session, request, user.id, "user.registered")
    session.commit()
    return {"id": user.id}


@app.post("/api/auth/token", response_model=TokenResponse)
def token(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
) -> TokenResponse:
    user = session.scalar(select(User).where(User.email == form.username.lower()))
    if user is None or not verify_password(form.password, user.password_hash):
        audit(session, request, None, "auth.failed", detail_email=form.username[:320])
        session.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "incorrect email or password")
    audit(session, request, user.id, "auth.succeeded")
    session.commit()
    return TokenResponse(access_token=create_token(user.id, settings))


@app.get("/api/providers")
def providers(_: UserDep, settings: SettingsDep) -> list[dict[str, object]]:
    statuses = []
    for provider in build_providers(settings):
        current = provider.check_availability()
        statuses.append(
            {
                "name": current.name,
                "configured": current.configured,
                "available": current.available,
                "capabilities": [capability.value for capability in current.capabilities],
                "detail": current.detail,
            }
        )
    return statuses


@app.post("/api/workload", response_model=WorkloadEstimate)
def workload(body: WorkloadRequest, _: UserDep, settings: SettingsDep) -> WorkloadEstimate:
    box = BoundingBox(body.region.south, body.region.west, body.region.north, body.region.east)
    height = (box.north - box.south) * METERS_PER_DEGREE
    width = (box.east - box.west) * METERS_PER_DEGREE * max(math.cos(math.radians(box.center[0])), 0.05)
    window_s = body.start_latest_utc - body.start_earliest_utc
    tiles = max(1, int(height * width / SEARCH_CELL_M**2)) * max(1, int(window_s / SEARCH_TIME_STEP_S))
    window = TimeWindow(body.start_earliest_utc, body.start_latest_utc + body.duration_s)
    requests = {
        provider.name: provider.estimate_requests(box, window)
        for provider in build_providers(settings, body.providers or None)
        if provider.is_configured()
    }
    advice: list[str] = []
    if not requests:
        advice.append("No flight data provider is configured, so only acoustic analysis can run.")
    if window_s > 3600:
        advice.append("A window beyond an hour cannot usually be resolved with one aircraft; narrow it if possible.")
    if height * width > 250_000 * 1e6:
        advice.append("The region is very large; every tracked flight inside it becomes a candidate.")
    feasible = tiles <= MAX_FEASIBLE_TILES and sum(requests.values()) <= MAX_FEASIBLE_PROVIDER_REQUESTS
    if not feasible:
        advice.append("These constraints are too loose for a reliable result; narrow the region or the time window.")
    return WorkloadEstimate(
        region_area_km2=height * width / 1e6,
        time_window_s=window_s,
        provider_requests=requests,
        initial_search_tiles=tiles,
        feasible=feasible,
        advice=advice,
    )


async def _store_upload(upload: UploadFile, destination: Path, limit: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while chunk := await upload.read(CHUNK_BYTES):
            size += len(chunk)
            if size > limit:
                handle.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file exceeds the upload limit")
            digest.update(chunk)
            handle.write(chunk)
    return digest.hexdigest(), size


@app.post("/api/analyses", response_model=AnalysisSummary, status_code=202)
async def create_analysis(
    request: Request,
    session: SessionDep,
    user: UserDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File()],
    parameters: Annotated[str, Form()],
) -> Analysis:
    try:
        body = AnalysisRequest.model_validate(json.loads(parameters))
    except (json.JSONDecodeError, ValidationError) as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error
    safe_name = Path(file.filename or "upload").name[:255]
    extension = Path(safe_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"unsupported file type: {extension}")
    running = session.scalar(
        select(func.count()).select_from(Analysis).where(Analysis.owner_id == user.id, Analysis.state.in_(ACTIVE_STATES))
    )
    if int(running or 0) >= settings.max_concurrent_jobs_per_user:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many analyses are already running")
    analysis = Analysis(
        owner_id=user.id,
        filename=safe_name,
        extension=extension,
        sha256="",
        size_bytes=0,
        authorization_basis=body.authorization_basis.value,
        authorization_note=body.authorization_note,
        request=body.model_dump(mode="json", exclude_none=True, exclude={"authorization_basis", "authorization_note"}),
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.retention_days),
    )
    session.add(analysis)
    session.flush()
    destination = settings.upload_dir / analysis.id / f"source{extension}"
    digest, size = await _store_upload(file, destination, settings.max_upload_bytes)
    try:
        await run_in_threadpool(probe_media, destination, extension, settings)
    except MediaRejected as rejected:
        delete_artifacts(analysis.id, settings)
        session.rollback()
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(rejected)) from rejected
    region = body.region
    analysis.sha256 = digest
    analysis.size_bytes = size
    analysis.search_region = geometry_value(
        session,
        f"POLYGON(({region.west} {region.south}, {region.east} {region.south}, {region.east} {region.north}, "
        f"{region.west} {region.north}, {region.west} {region.south}))",
    )
    audit(
        session,
        request,
        user.id,
        "analysis.created",
        analysis.id,
        sha256=digest,
        authorization_basis=body.authorization_basis.value,
    )
    session.commit()
    enqueue_analysis(analysis.id, destination, settings)
    return analysis


@app.get("/api/analyses", response_model=list[AnalysisSummary])
def list_analyses(session: SessionDep, user: UserDep) -> list[Analysis]:
    return list(
        session.scalars(select(Analysis).where(Analysis.owner_id == user.id).order_by(Analysis.created_at.desc()).limit(200))
    )


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisDetail)
def get_analysis(analysis_id: str, session: SessionDep, user: UserDep) -> Analysis:
    return owned_analysis(analysis_id, session, user)


@app.post("/api/analyses/{analysis_id}/cancel", response_model=AnalysisSummary)
def cancel_analysis(analysis_id: str, request: Request, session: SessionDep, user: UserDep) -> Analysis:
    analysis = owned_analysis(analysis_id, session, user)
    if analysis.state in ACTIVE_STATES:
        analysis.cancel_requested = True
        if analysis.state == "queued":
            analysis.state = "cancelled"
            analysis.stage = "cancelled"
        audit(session, request, user.id, "analysis.cancel_requested", analysis.id)
        session.commit()
    return analysis


@app.delete("/api/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str, request: Request, session: SessionDep, user: UserDep, settings: SettingsDep) -> Response:
    analysis = owned_analysis(analysis_id, session, user)
    if analysis.state in ACTIVE_STATES:
        raise HTTPException(status.HTTP_409_CONFLICT, "cancel the analysis before deleting it")
    delete_artifacts(analysis.id, settings)
    audit(session, request, user.id, "analysis.deleted", analysis.id, sha256=analysis.sha256)
    session.delete(analysis)
    session.commit()
    return Response(status_code=204)


def _artifact(analysis: Analysis, settings: Settings, name: str) -> Path:
    path = settings.artifact_dir / analysis.id / name
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "artifact not available yet")
    return path


@app.get("/api/analyses/{analysis_id}/waveform")
def waveform(analysis_id: str, session: SessionDep, user: UserDep, settings: SettingsDep) -> FileResponse:
    return FileResponse(_artifact(owned_analysis(analysis_id, session, user), settings, "waveform.json"), media_type="application/json")


@app.get("/api/analyses/{analysis_id}/spectrogram")
def spectrogram(analysis_id: str, session: SessionDep, user: UserDep, settings: SettingsDep) -> FileResponse:
    return FileResponse(_artifact(owned_analysis(analysis_id, session, user), settings, "spectrogram.json"), media_type="application/json")


@app.get("/api/analyses/{analysis_id}/spectrogram.png")
def spectrogram_png(analysis_id: str, session: SessionDep, user: UserDep, settings: SettingsDep) -> FileResponse:
    return FileResponse(_artifact(owned_analysis(analysis_id, session, user), settings, "spectrogram.png"), media_type="image/png")


@app.get("/api/analyses/{analysis_id}/audio")
def audio(analysis_id: str, session: SessionDep, user: UserDep, settings: SettingsDep) -> FileResponse:
    return FileResponse(_artifact(owned_analysis(analysis_id, session, user), settings, "original.wav"), media_type="audio/wav")


@app.get("/api/analyses/{analysis_id}/source")
def source_media(analysis_id: str, session: SessionDep, user: UserDep, settings: SettingsDep) -> FileResponse:
    analysis = owned_analysis(analysis_id, session, user)
    path = settings.upload_dir / analysis.id / f"source{analysis.extension}"
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "source media not available")
    return FileResponse(path, media_type="application/octet-stream", filename=analysis.filename)


@app.get("/api/analyses/{analysis_id}/report.json")
def report_json(analysis_id: str, request: Request, session: SessionDep, user: UserDep) -> JSONResponse:
    analysis = owned_analysis(analysis_id, session, user)
    audit(session, request, user.id, "analysis.exported", analysis.id, format="json")
    session.commit()
    return JSONResponse(report_payload(analysis))


@app.get("/api/analyses/{analysis_id}/report.pdf")
def report_pdf(analysis_id: str, request: Request, session: SessionDep, user: UserDep) -> Response:
    analysis = owned_analysis(analysis_id, session, user)
    audit(session, request, user.id, "analysis.exported", analysis.id, format="pdf")
    session.commit()
    return Response(render_pdf(analysis), media_type="application/pdf")


@app.get("/api/audit")
def audit_log(session: SessionDep, user: UserDep) -> list[dict[str, object]]:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "administrator access required")
    rows = session.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(500))
    return [
        {
            "created_at": row.created_at.isoformat(),
            "user_id": row.user_id,
            "action": row.action,
            "target": row.target,
            "detail": row.detail,
        }
        for row in rows
    ]


def _progress_snapshot(analysis_id: str, token_value: str, settings: Settings) -> dict[str, object] | None:
    with session_scope() as session:
        user = user_from_token(token_value, session, settings)
        analysis = session.get(Analysis, analysis_id)
        if user is None or analysis is None or analysis.owner_id != user.id:
            return None
        return {
            "state": analysis.state,
            "stage": analysis.stage,
            "progress": analysis.progress,
            "error": analysis.error,
            "localization_status": analysis.localization_status,
        }


@app.websocket("/api/analyses/{analysis_id}/progress")
async def progress_socket(websocket: WebSocket, analysis_id: str) -> None:
    await websocket.accept()
    settings = get_settings()
    try:
        credentials = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        token_value = str(credentials.get("token", ""))
        previous: dict[str, object] | None = None
        while True:
            snapshot = await run_in_threadpool(_progress_snapshot, analysis_id, token_value, settings)
            if snapshot is None:
                await websocket.close(code=4401)
                return
            if snapshot != previous:
                await websocket.send_json(snapshot)
                previous = snapshot
            if snapshot["state"] in TERMINAL_STATES:
                await websocket.close()
                return
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        return
