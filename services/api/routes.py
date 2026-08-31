"""FastAPI route handlers implementing the full endpoint surface (§8.2)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated, Any
import os
from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from packages.cs_core.models import (
    CameraRole,
    ModelStage,
    ReconciliationResolution,
    SessionStatus,
    UserRole,
)
from packages.cs_counting.event_handler import CountingEventHandler, estimate_simulated_area
from packages.cs_counting.events import GateCrossingRecorded, SessionAreaEstimateUpdated
from packages.cs_counting.gate import GateCrossingEvent
from packages.cs_storage.db import get_sync_session
from packages.cs_storage.errors import ActiveSessionConflictError
from packages.cs_storage.models_orm import (
    CameraORM,
    ConfigVersionORM,
    CountEventORM,
    DatasetVersionORM,
    DeploymentBundleORM,
    GateORM,
    JobORM,
    LineCalibrationORM,
    LineORM,
    ModelVersionORM,
    NodeORM,
    OutboxORM,
    ProductProfileORM,
    ReconciliationORM,
    SessionORM,
    SiteORM,
    TrainingRunORM,
)
from packages.cs_storage.repositories.calibration_repo import CalibrationRepository
from packages.cs_storage.repositories.camera_epoch_repo import CameraEpochRepository
from packages.cs_storage.repositories.config_repo import ConfigRepository
from packages.cs_storage.repositories.job_repo import JobRepository
from packages.cs_storage.repositories.ledger_repo import LedgerRepository
from packages.cs_storage.repositories.outbox_repo import OutboxRepository
from packages.cs_storage.repositories.reconciliation_repo import ReconciliationRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from packages.cs_storage.repositories.user_repo import UserRepository
from services.api.auth import CurrentUser, create_access_token, get_current_user, require_role

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas for requests & responses
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    role: UserRole


class CreateSiteRequest(BaseModel):
    name: str
    timezone: str = "Europe/Istanbul"
    locale: str = "tr_TR"


class CreateLineRequest(BaseModel):
    site_id: int
    name: str


class CreateCameraRequest(BaseModel):
    line_id: int
    node_id: int
    source_driver: str = "rtsp"
    source_config: dict[str, Any] = Field(default_factory=dict)
    role: CameraRole = CameraRole.COUNTING


class CreateGateRequest(BaseModel):
    line_id: int
    name: str
    order_index: int = 0


class CreateSessionRequest(BaseModel):
    line_id: int
    product_profile_id: int
    external_ref: str | None = None
    target_count: int | None = None
    vehicle_plate: str | None = None
    driver_name: str | None = None
    carrier_company: str | None = None


class ResolveReconciliationRequest(BaseModel):
    resolution: ReconciliationResolution
    resolved_count: int | None = None
    note: str | None = None


class CreateConfigRequest(BaseModel):
    payload: dict[str, Any]
    note: str | None = None


class ActivateBundleRequest(BaseModel):
    line_id: int
    model_version_id: int
    config_version_id: int
    calibration_id: int | None = None


class SubmitJobResponse(BaseModel):
    job_id: int
    status: str = "queued"
    kind: str


# ---------------------------------------------------------------------------
# Auth Endpoints
# ---------------------------------------------------------------------------

@router.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest, response: Response):
    """Authenticate user and issue cryptographically signed JWT session token / cookie (§8.1)."""
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        user = user_repo.authenticate(req.username, req.password)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

        token = create_access_token(data={"sub": str(user.id), "username": user.username, "role": user.role})
        response.set_cookie(key="session_token", value=token, httponly=True, max_age=86400)
        return LoginResponse(token=token, username=user.username, role=UserRole(user.role))


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


@router.post("/auth/change-password")
def change_password(
    req: ChangePasswordRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
):
    """Update password for authenticated user (§8.1)."""
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        auth_user = user_repo.authenticate(current_user.username, req.old_password)
        if not auth_user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password incorrect.")
        if len(req.new_password) < 6:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="New password must be at least 6 characters.")
        user_repo.update_password(current_user.username, req.new_password)
        return {"status": "success", "message": "Password changed successfully."}


class RegisterUserRequest(BaseModel):
    username: str
    password: str
    role: UserRole


@router.post("/auth/register", dependencies=[Depends(require_role(UserRole.ADMIN))])
def register_user(req: RegisterUserRequest):
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        user = user_repo.create_user(username=req.username, password=req.password, role=req.role.value)
        return {"id": user.id, "username": user.username, "role": user.role}


# ---------------------------------------------------------------------------
# Sites, Lines, Cameras
# ---------------------------------------------------------------------------

@router.get("/sites")
def list_sites(user: Annotated[CurrentUser, Depends(get_current_user)], limit: int = 100, offset: int = 0):
    with get_sync_session() as db:
        return db.query(SiteORM).order_by(SiteORM.id).offset(offset).limit(limit).all()


@router.post("/sites", dependencies=[Depends(require_role(UserRole.ADMIN))])
def create_site(req: CreateSiteRequest):
    with get_sync_session() as db:
        site = SiteORM(name=req.name, timezone=req.timezone, locale=req.locale)
        db.add(site)
        db.commit()
        db.refresh(site)
        return site


@router.get("/lines")
def list_lines(user: Annotated[CurrentUser, Depends(get_current_user)], limit: int = 100, offset: int = 0):
    with get_sync_session() as db:
        return db.query(LineORM).order_by(LineORM.id).offset(offset).limit(limit).all()


@router.post("/lines", dependencies=[Depends(require_role(UserRole.ADMIN))])
def create_line(req: CreateLineRequest):
    with get_sync_session() as db:
        line = LineORM(site_id=req.site_id, name=req.name, status="idle")
        db.add(line)
        db.commit()
        db.refresh(line)
        return line


class CreateNodeRequest(BaseModel):
    site_id: int
    hostname: str


@router.get("/nodes")
def list_nodes(user: Annotated[CurrentUser, Depends(get_current_user)], site_id: int | None = None, limit: int = 100, offset: int = 0):
    with get_sync_session() as db:
        q = db.query(NodeORM)
        if site_id is not None:
            q = q.filter(NodeORM.site_id == site_id)
        return q.order_by(NodeORM.id).offset(offset).limit(limit).all()


@router.post("/nodes", dependencies=[Depends(require_role(UserRole.ADMIN))])
def create_node(req: CreateNodeRequest):
    with get_sync_session() as db:
        node = NodeORM(site_id=req.site_id, hostname=req.hostname)
        db.add(node)
        db.commit()
        db.refresh(node)
        return node


@router.get("/cameras")
def list_cameras(user: Annotated[CurrentUser, Depends(get_current_user)], line_id: int | None = None, limit: int = 100, offset: int = 0):
    with get_sync_session() as db:
        q = db.query(CameraORM)
        if line_id is not None:
            q = q.filter(CameraORM.line_id == line_id)
        return q.order_by(CameraORM.id).offset(offset).limit(limit).all()


@router.post("/cameras", dependencies=[Depends(require_role(UserRole.ADMIN))])
def create_camera(req: CreateCameraRequest):
    with get_sync_session() as db:
        cam = CameraORM(
            line_id=req.line_id,
            node_id=req.node_id,
            source_driver=req.source_driver,
            source_config=req.source_config,
            role=req.role.value,
        )
        db.add(cam)
        db.commit()
        db.refresh(cam)
        line_id = cam.line_id

    # A live renderer resolves camera_id/gate_id once at construction and
    # caches them (see LiveStreamRenderer.reload_camera_context()) -- without
    # this, a camera registered for a line *after* its stream was already
    # opened would stay invisible to ledger writes until the process
    # restarted, same class of staleness reload_active_config()/
    # reload_perspective_calibration() already handle for their own state.
    from packages.cs_counting.stream_renderer import _renderers
    renderer = _renderers.get(line_id)
    if renderer is not None:
        renderer.reload_camera_context()

    return cam


@router.get("/gates")
def list_gates(user: Annotated[CurrentUser, Depends(get_current_user)], line_id: int | None = None, limit: int = 100, offset: int = 0):
    with get_sync_session() as db:
        q = db.query(GateORM)
        if line_id is not None:
            q = q.filter(GateORM.line_id == line_id)
        return q.order_by(GateORM.id).offset(offset).limit(limit).all()


@router.post("/gates", dependencies=[Depends(require_role(UserRole.ADMIN))])
def create_gate(req: CreateGateRequest):
    with get_sync_session() as db:
        gate = GateORM(line_id=req.line_id, name=req.name, order_index=req.order_index)
        db.add(gate)
        db.commit()
        db.refresh(gate)
        line_id = gate.line_id

    # Same staleness class as create_camera()'s reload_camera_context() call
    # above -- a gate registered for a line after its stream was already
    # opened would otherwise stay invisible (gate_id=None, ledger writes
    # skipped) until the process restarted.
    from packages.cs_counting.stream_renderer import _renderers
    renderer = _renderers.get(line_id)
    if renderer is not None:
        renderer.reload_camera_context()

    return gate


@router.post("/cameras/{cam_id}/test", dependencies=[Depends(require_role(UserRole.ADMIN))])
def test_camera_connection(cam_id: int):
    """Test connection reachability for camera source driver by actually opening it."""
    import cv2
    from packages.cs_core.camera_source import resolve_camera_source

    with get_sync_session() as db:
        cam = db.query(CameraORM).filter(CameraORM.id == cam_id).first()
        if not cam:
            raise HTTPException(status_code=404, detail="Camera not found")
        source_config = cam.source_config or {}

    source = resolve_camera_source(cam.source_driver, source_config)

    if source == "" or source is None:
        return {
            "status": "error",
            "connected": False,
            "camera_id": cam_id,
            "message": f"No usable source configured for driver '{cam.source_driver}' (source_config={source_config}).",
        }

    cap = cv2.VideoCapture(source)
    try:
        connected = cap.isOpened()
        message = (
            f"Camera feed connected successfully: {source}"
            if connected
            else f"Could not connect to camera feed: {source}"
        )
    finally:
        cap.release()

    return {"status": "ok" if connected else "error", "connected": connected, "camera_id": cam_id, "message": message}


class SetCameraFeedSourceRequest(BaseModel):
    source_config: dict[str, Any] = Field(default_factory=dict)
    source_driver: str | None = None


@router.post("/cameras/{cam_id}/source", dependencies=[Depends(require_role(UserRole.ADMIN))])
def set_camera_feed_source(cam_id: int, req: SetCameraFeedSourceRequest):
    """Persist a camera's connection info and (re)connect its live feed.

    Every camera gets a real monitoring feed via camera_feed.py regardless of
    role. A counting-role camera additionally has to be pushed into the real
    LiveStreamRenderer for its line -- that is the object the actual
    detection/tracking pipeline reads frames from, and it is keyed by
    line_id, not camera_id, so setting a counting camera's source here would
    silently do nothing to real detection without this extra step (found by
    testing this end-to-end: the counting engine kept running in demo mode
    after "adding" a counting camera through this exact route).
    """
    from packages.cs_core.camera_source import resolve_camera_source
    from packages.cs_counting.camera_feed import reconnect_camera_feed

    with get_sync_session() as db:
        cam = db.query(CameraORM).filter(CameraORM.id == cam_id).first()
        if not cam:
            raise HTTPException(status_code=404, detail="Camera not found")
        cam.source_config = req.source_config
        if req.source_driver:
            cam.source_driver = req.source_driver
        db.commit()
        source_driver, source_config, role, line_id = cam.source_driver, cam.source_config, cam.role, cam.line_id

    source = resolve_camera_source(source_driver, source_config)
    connected, message = reconnect_camera_feed(cam_id, source)

    if role == "counting":
        from packages.cs_counting.stream_renderer import _renderers, LiveStreamRenderer
        if line_id not in _renderers:
            _renderers[line_id] = LiveStreamRenderer(line_id=line_id)
        detect_ok, detect_msg = _renderers[line_id].set_camera_source(source)
        connected = connected and detect_ok
        message = detect_msg if not detect_ok else message

    return {"status": "ok" if connected else "error", "connected": connected, "camera_id": cam_id, "message": message}


@router.get("/cameras/{cam_id}/status")
def get_camera_feed_status(cam_id: int):
    """Cheap, real status for a non-counting camera's live feed.

    Reports the already-tracked in-memory CameraFeed state (connected,
    measured FPS, last error) without opening a new connection -- unlike
    POST /cameras/{id}/test, which does a fresh connect/disconnect probe.
    Meant for cheap polling (e.g. an alarm banner), not a substitute for
    the real test endpoint.
    """
    from packages.cs_counting.camera_feed import _camera_feeds
    feed = _camera_feeds.get(cam_id)
    if feed is None:
        return {"camera_id": cam_id, "known": False, "connected": False, "fps": None, "last_error": None}
    return {
        "camera_id": cam_id,
        "known": True,
        "connected": feed.connected,
        "fps": round(feed._fps_ema, 1) if feed._fps_ema is not None else None,
        "last_error": feed.last_error,
    }


@router.get("/cameras/{cam_id}/stream")
def stream_camera_feed_mjpeg(cam_id: int):
    """Real live MJPEG feed for a single non-counting camera (no detection overlay)."""
    from packages.cs_counting.camera_feed import get_camera_stream_generator
    return StreamingResponse(
        get_camera_stream_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


class CreateProductRequest(BaseModel):
    site_id: int
    name: str
    erp_material_code: str | None = None
    nominal_dims_mm: dict[str, Any] = Field(default_factory=dict)


@router.get("/products")
def list_products(user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    with get_sync_session() as db:
        return db.query(ProductProfileORM).all()


@router.post("/products", dependencies=[Depends(require_role(UserRole.ADMIN))])
def create_product(req: CreateProductRequest):
    with get_sync_session() as db:
        prof = ProductProfileORM(
            site_id=req.site_id,
            name=req.name,
            erp_material_code=req.erp_material_code,
            nominal_dims_mm=req.nominal_dims_mm,
        )
        db.add(prof)
        db.commit()
        db.refresh(prof)
        return prof


# ---------------------------------------------------------------------------
# Live SSE stream & Sessions
# ---------------------------------------------------------------------------

@router.get("/live/lines/{line_id}")
async def stream_live_counts(line_id: int, request: Request, user: Annotated[CurrentUser, Depends(get_current_user)]):
    """Server-Sent Events (SSE) stream for live counter, active session, and health (§9.6)."""
    # Lazy import (mirrors the other stream_renderer call sites in this module)
    # to avoid pulling in cv2 at API module import time. Used to surface the
    # renderer's real rolling FPS estimate (_fps_ema) on the live stream
    # rather than a hardcoded frontend number.
    from packages.cs_counting.stream_renderer import _renderers

    async def event_generator():
        while True:
            if await request.is_disconnected():
                logger.info(f"[SSE] Client disconnected from line {line_id} live stream; stopping.")
                break

            renderer = _renderers.get(line_id)
            fps_json = f"{renderer._fps_ema:.1f}" if (renderer is not None and renderer._fps_ema is not None) else "null"

            with get_sync_session() as db:
                session_repo = SessionRepository(db)
                ledger_repo = LedgerRepository(db)
                active_session = session_repo.get_active_session(line_id)

                if active_session:
                    net_count = ledger_repo.get_session_total_count(active_session.id)
                    data = (
                        f'{{"line_id": {line_id}, "session_id": {active_session.id}, '
                        f'"counted_total": {net_count}, "area_estimate": {active_session.area_estimate_total:.1f}, '
                        f'"status": "{active_session.status}", "discrepancy": {str(active_session.discrepancy_flag).lower()}, '
                        f'"fps": {fps_json}}}'
                    )
                else:
                    data = f'{{"line_id": {line_id}, "session_id": null, "counted_total": 0, "status": "idle", "fps": {fps_json}}}'

            yield f"data: {data}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/live/lines/{line_id}/stream")
def stream_camera_mjpeg(line_id: int):
    """Serve live MJPEG video stream with real-time OpenCV AI bounding boxes and amodal segmentations."""
    from packages.cs_counting.stream_renderer import get_stream_generator
    return StreamingResponse(
        get_stream_generator(line_id=line_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/lines/{line_id}/upload_video")
async def upload_line_video(line_id: int, file: UploadFile = File(...)):
    """Upload a real MP4/AVI factory video for live computer vision counting analysis."""
    from packages.cs_counting.stream_renderer import _renderers, LiveStreamRenderer
    os.makedirs("./data/videos", exist_ok=True)
    video_path = f"./data/videos/line_{line_id}_{file.filename}"
    content = await file.read()

    def _write_file() -> None:
        with open(video_path, "wb") as f:
            f.write(content)

    await asyncio.to_thread(_write_file)

    if line_id not in _renderers:
        _renderers[line_id] = LiveStreamRenderer(line_id=line_id)
    _renderers[line_id].set_video_source(video_path)

    return {"status": "ok", "message": f"Video uploaded and analysis started: {file.filename}", "video_path": video_path}


class SetCameraSourceRequest(BaseModel):
    source: str


@router.post("/lines/{line_id}/camera_source")
def set_line_camera_source(line_id: int, req: SetCameraSourceRequest, user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    """Switch active camera feed to RTSP stream, USB webcam (#0, #1), or synthetic conveyor."""
    from packages.cs_counting.stream_renderer import _renderers, LiveStreamRenderer
    if line_id not in _renderers:
        _renderers[line_id] = LiveStreamRenderer(line_id=line_id)

    success, msg = _renderers[line_id].set_camera_source(req.source)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "ok", "message": msg, "source": req.source}


@router.post("/sessions", response_model=dict[str, Any])
def open_session(req: CreateSessionRequest, user: Annotated[CurrentUser, Depends(get_current_user)]):
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        try:
            sess = session_repo.create_session(
                line_id=req.line_id,
                product_profile_id=req.product_profile_id,
                external_ref=req.external_ref,
                target_count=req.target_count,
                vehicle_plate=req.vehicle_plate,
                driver_name=req.driver_name,
                carrier_company=req.carrier_company,
            )
        except ActiveSessionConflictError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return {"id": sess.id, "status": sess.status, "line_id": sess.line_id, "external_ref": sess.external_ref}


@router.get("/sessions")
def list_sessions(line_id: int | None = None, user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        return session_repo.list_sessions(line_id=line_id)


@router.get("/sessions/{sess_id}")
def get_session(sess_id: int, user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        sess = session_repo.get_by_id(sess_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        return sess


@router.post("/sessions/{sess_id}/pause")
def pause_session(sess_id: int, user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        sess = session_repo.pause_session(sess_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"id": sess.id, "status": sess.status}


@router.post("/sessions/{sess_id}/resume")
def resume_session(sess_id: int, user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        sess = session_repo.resume_session(sess_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"id": sess.id, "status": sess.status}


class UpdateSessionRequest(BaseModel):
    product_profile_id: int | None = None
    target_count: int | None = None
    external_ref: str | None = None


@router.patch("/sessions/{sess_id}")
def update_session(sess_id: int, req: UpdateSessionRequest, user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        sess = session_repo.get_by_id(sess_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        if req.product_profile_id is not None:
            sess.product_profile_id = req.product_profile_id
        if req.target_count is not None:
            sess.target_count = req.target_count
        if req.external_ref is not None:
            sess.external_ref = req.external_ref
        db.commit()
        db.refresh(sess)
        return sess


class QuickLineSettingsRequest(BaseModel):
    belt_speed: float | None = None
    belt_direction: list[float] | None = None
    gate_x_pos: float | None = None
    pre_offset: float | None = None
    post_offset: float | None = None
    rtsp_url: str | None = None


@router.post("/lines/{line_id}/quick_settings")
def update_quick_line_settings(line_id: int, req: QuickLineSettingsRequest, user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    with get_sync_session() as db:
        line = db.query(LineORM).filter(LineORM.id == line_id).first()
        if not line:
            raise HTTPException(status_code=404, detail="Line not found")

        calib_repo = CalibrationRepository(db)
        if req.belt_speed is not None or req.belt_direction is not None:
            speed = req.belt_speed if req.belt_speed is not None else 6.5
            direction = req.belt_direction if req.belt_direction is not None else [1.0, 0.0]
            calib_repo.create_motion_calibration(line_id=line_id, belt_speed_px_per_frame=speed, belt_direction_vector=direction)

        if req.rtsp_url:
            cam = db.query(CameraORM).filter(CameraORM.line_id == line_id).first()
            if cam:
                cam.source_config = {"rtsp_url": req.rtsp_url}
                db.commit()

        from packages.cs_counting.stream_renderer import _renderers
        if line_id in _renderers:
            renderer = _renderers[line_id]
            if req.belt_speed is not None:
                renderer.belt_speed_px = float(req.belt_speed)
            if req.belt_direction is not None and len(req.belt_direction) > 0:
                renderer.belt_dir = 1 if req.belt_direction[0] >= 0 else -1
            if req.gate_x_pos is not None:
                renderer.gate_x = int(req.gate_x_pos)

        return {"status": "ok", "message": "Line and conveyor belt settings updated."}


@router.post("/sessions/{sess_id}/close")
def close_session(sess_id: int, user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        sess = session_repo.close_session(sess_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        return sess


@router.get("/sessions/{sess_id}/events")
def get_session_ledger_events(sess_id: int, user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    with get_sync_session() as db:
        ledger_repo = LedgerRepository(db)
        return ledger_repo.get_session_events(sess_id)


@router.get("/lines/{line_id}/defects")
def get_line_defect_events(line_id: int, user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    """Audit log of counted bags later excluded as defective (post-gate removal, §5.5)."""
    with get_sync_session() as db:
        ledger_repo = LedgerRepository(db)
        return ledger_repo.get_defect_events(line_id=line_id)


class DisputeDefectRequest(BaseModel):
    note: str | None = None


@router.post(
    "/events/{event_id}/dispute_defect",
    dependencies=[Depends(require_role(UserRole.OPERATOR, UserRole.ENGINEER))],
)
def dispute_defect_event(event_id: str, req: DisputeDefectRequest, user: Annotated[CurrentUser, Depends(get_current_user)]):
    """Overturn a defect flag found to be a false positive (§5.5).

    Does not erase the original AI detection -- appends a real, attributed
    annotation next to it. Idempotent: disputing an already-disputed event
    is a no-op, it does not overwrite who disputed it first.
    """
    with get_sync_session() as db:
        ledger_repo = LedgerRepository(db)
        event = ledger_repo.dispute_defect_event(event_id, disputed_by=user.username, note=req.note)
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found or not flagged as a defect")
        return event


@router.get("/sessions/{sess_id}/dispatch_report")
def get_session_dispatch_report(sess_id: int, user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    """Generate official Dispatch & Reconciliation Manifest with cryptographic seal (§5.7, §5.8)."""
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        ledger_repo = LedgerRepository(db)
        sess = session_repo.get_by_id(sess_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")

        events = ledger_repo.get_session_events(sess_id)
        prod = db.query(ProductProfileORM).filter(ProductProfileORM.id == sess.product_profile_id).first()
        prod_name = prod.name if prod else None
        erp_sku = prod.erp_material_code if (prod and prod.erp_material_code) else None

        line = db.query(LineORM).filter(LineORM.id == sess.line_id).first()
        site = db.query(SiteORM).filter(SiteORM.id == line.site_id).first() if line else None

        counted = sess.counted_total or 0
        target = sess.target_count
        diff = (counted - target) if target is not None else None
        # Real defect count from the ledger's own defect_reason field (§5.5 post-gate
        # exclusion), not a confidence-threshold heuristic.
        damaged_count = sum(1 for e in events if getattr(e, "defect_reason", None))
        simulated_count = sum(1 for e in events if getattr(e, "is_simulated", False))

        import hashlib
        import hmac
        from services.api.auth import SECRET_KEY
        data_str = f"SESS:{sess.id}|REF:{sess.external_ref}|COUNT:{counted}|PROD:{erp_sku}|TS:{datetime.now(timezone.utc).isoformat()}"
        crypto_seal = hmac.new(SECRET_KEY.encode(), data_str.encode(), hashlib.sha256).hexdigest().upper()

        return {
            "session_id": sess.id,
            # No fabricated fallback reference -- None when the session was
            # never given one, rather than a fake-but-plausible-looking
            # document number.
            "external_ref": sess.external_ref,
            # No real waybill/dispatch-note field is tracked on SessionORM yet;
            # omit rather than fabricate a document number.
            "waybill_no": None,
            "site_name": site.name if site else None,
            "line_name": line.name if line else None,
            "truck_plate": sess.vehicle_plate,
            "driver_name": sess.driver_name,
            "carrier_company": sess.carrier_company,
            "product_name": prod_name,
            "erp_sku": erp_sku,
            "target_count": target,
            "counted_total": counted,
            "difference": diff,
            "status": sess.status,
            "reconciliation_status": (
                None if diff is None
                else "EXACT RECONCILIATION" if diff == 0
                else "OVER DISPATCH" if diff > 0
                else "SHORT DISPATCH"
            ),
            "damaged_count": damaged_count,
            "simulated_count": simulated_count,
            "started_at": sess.opened_at.isoformat() if sess.opened_at else None,
            "closed_at": sess.closed_at.isoformat() if sess.closed_at else None,
            "crypto_seal": crypto_seal,
            "event_count": len(events),
        }


@router.post("/sessions/{sess_id}/submit")
def submit_session_to_erp(sess_id: int, user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    """Submit finalized count session to Transactional Outbox for ERP dispatch (§5.8, M7)."""
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        outbox_repo = OutboxRepository(db)
        sess = session_repo.get_by_id(sess_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")

        if sess.status == "reconcile_required":
            raise HTTPException(
                status_code=400,
                detail="Session requires human reconciliation before submission to ERP (§5.7).",
            )

        payload = {
            "session_id": sess.id,
            "line_id": sess.line_id,
            "product_profile_id": sess.product_profile_id,
            "counted_total": sess.counted_total,
            "area_estimate_total": sess.area_estimate_total,
        }
        outbox_entry = outbox_repo.create_entry(session_id=sess.id, payload=payload, external_ref=sess.external_ref)
        return {"status": "submitted_to_outbox", "outbox_id": outbox_entry.id}


# ---------------------------------------------------------------------------
# Reconciliations (§5.7)
# ---------------------------------------------------------------------------

@router.get("/reconciliations", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def list_reconciliations(limit: int = 100, offset: int = 0):
    with get_sync_session() as db:
        rec_repo = ReconciliationRepository(db)
        # ReconciliationRepository.list_open_reconciliations() has no limit/offset
        # of its own (out of scope here to change cs_storage repositories), so
        # pagination is applied to its already-ordered result.
        results = rec_repo.list_open_reconciliations()
        return results[offset : offset + limit]


@router.post("/reconciliations/{rec_id}/resolve", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def resolve_reconciliation(rec_id: int, req: ResolveReconciliationRequest, user: Annotated[CurrentUser, Depends(get_current_user)]):
    with get_sync_session() as db:
        rec_repo = ReconciliationRepository(db)
        resolved = rec_repo.resolve_reconciliation(
            reconciliation_id=rec_id,
            resolution=req.resolution.value,
            resolved_count=req.resolved_count,
            resolved_by=user.username,
            note=req.note,
        )
        if not resolved:
            raise HTTPException(status_code=404, detail="Reconciliation not found")
        return resolved


# ---------------------------------------------------------------------------
# Dataset, CVAT & Training Jobs (§5.8, §8.2)
# ---------------------------------------------------------------------------

@router.post("/datasets/extract", dependencies=[Depends(require_role(UserRole.ENGINEER))], status_code=202)
def start_extract_job(payload: dict[str, Any]):
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        job = job_repo.submit_job(kind="extract_frames", payload=payload)
        return SubmitJobResponse(job_id=job.id, kind=job.kind)


@router.post("/datasets/synthesize", dependencies=[Depends(require_role(UserRole.ENGINEER))], status_code=202)
def start_synthesize_job(payload: dict[str, Any]):
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        job = job_repo.submit_job(kind="synthesize", payload=payload)
        return SubmitJobResponse(job_id=job.id, kind=job.kind)


@router.post("/datasets/build", dependencies=[Depends(require_role(UserRole.ENGINEER))], status_code=202)
def start_build_dataset_job(payload: dict[str, Any]):
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        job = job_repo.submit_job(kind="build_dataset", payload=payload)
        return SubmitJobResponse(job_id=job.id, kind=job.kind)


@router.get("/datasets", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def list_datasets(limit: int = 100, offset: int = 0):
    with get_sync_session() as db:
        return db.query(DatasetVersionORM).order_by(DatasetVersionORM.id.desc()).offset(offset).limit(limit).all()


@router.post("/training/runs", dependencies=[Depends(require_role(UserRole.ENGINEER))], status_code=202)
def start_training_job(payload: dict[str, Any]):
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        job = job_repo.submit_job(kind="train", payload=payload, requires_gpu=True, priority=5)
        return SubmitJobResponse(job_id=job.id, kind=job.kind)


@router.get("/models", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def list_models(limit: int = 100, offset: int = 0):
    with get_sync_session() as db:
        return (
            db.query(ModelVersionORM)
            .order_by(ModelVersionORM.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )


@router.get("/models/{model_id}/download", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def download_model_onnx(model_id: int):
    """Stream the real exported ONNX artifact for a trained model version from disk (§8.2)."""
    with get_sync_session() as db:
        mv = db.query(ModelVersionORM).filter(ModelVersionORM.id == model_id).first()
        if not mv:
            raise HTTPException(status_code=404, detail="Model version not found")
        onnx_path = mv.onnx_path
        if not onnx_path or not os.path.isfile(onnx_path):
            raise HTTPException(status_code=404, detail=f"ONNX artifact not found on disk: {onnx_path}")
        filename = os.path.basename(onnx_path) or f"model_{model_id}.onnx"
        return FileResponse(onnx_path, media_type="application/octet-stream", filename=filename)


@router.post("/datasets/mine_hard_frames", dependencies=[Depends(require_role(UserRole.ENGINEER))], status_code=202)
def start_mine_hard_frames_job(payload: dict[str, Any]):
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        job = job_repo.submit_job(kind="mine_hard_frames", payload=payload)
        return SubmitJobResponse(job_id=job.id, kind=job.kind)


@router.post("/models/{model_id}/export", dependencies=[Depends(require_role(UserRole.ENGINEER))], status_code=202)
def start_export_job(model_id: int, payload: dict[str, Any]):
    """Retrain + export a new model version, recorded as lineage of `model_id`.

    There is no persisted PyTorch checkpoint per historical ModelVersion (only
    the final ONNX artifact + eval_scores are kept), so this is NOT a cheap
    re-serialization of the existing `model_id` -- it re-runs real training and
    registers a brand new ModelVersion, with `model_id` referenced as lineage
    via TrainingRunORM.base_model_version_id on the resulting job's result.
    """
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        job = job_repo.submit_job(kind="export_onnx", payload={**payload, "model_id": model_id})
        return SubmitJobResponse(job_id=job.id, kind=job.kind)


@router.post("/replay/runs", dependencies=[Depends(require_role(UserRole.ENGINEER))], status_code=202)
def start_replay_job(payload: dict[str, Any]):
    """Run the offline replay evaluation suite (accuracy/latency regression) as a background job."""
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        job = job_repo.submit_job(kind="replay", payload=payload)
        return SubmitJobResponse(job_id=job.id, kind=job.kind)


@router.post("/models/{model_id}/stage", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def update_model_stage(model_id: int, stage: ModelStage):
    with get_sync_session() as db:
        mv = db.query(ModelVersionORM).filter(ModelVersionORM.id == model_id).first()
        if not mv:
            raise HTTPException(status_code=404, detail="Model version not found")
        mv.stage = stage.value
        db.commit()
        return mv


@router.get("/models/shadow/comparison", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def get_shadow_model_comparison():
    """Retrieve live A/B comparison metrics between active and candidate shadow models (§6.4)."""
    with get_sync_session() as db:
        active_model = db.query(ModelVersionORM).filter(ModelVersionORM.stage == "active").order_by(ModelVersionORM.id.desc()).first()
        shadow_model = db.query(ModelVersionORM).filter(ModelVersionORM.stage == "shadow").order_by(ModelVersionORM.id.desc()).first()

        if not active_model or not shadow_model:
            return {
                "active_model_id": active_model.id if active_model else None,
                "shadow_model_id": shadow_model.id if shadow_model else None,
                "status": "no_shadow_pair" if not shadow_model else "no_active_model",
                "is_ready_for_promotion": False,
                "frames_compared": 0,
                "agreement_rate": 1.0,
                "mean_iou": 1.0,
            }

        from packages.cs_vision.shadow_evaluator import ShadowModelEvaluator
        evaluator = ShadowModelEvaluator(active_model_id=active_model.id, shadow_model_id=shadow_model.id)
        metrics = evaluator.get_comparison_summary()
        return metrics.__dict__



# ---------------------------------------------------------------------------
# Integrated CVAT & Labeling Studio (§6.3, §6.4)
# ---------------------------------------------------------------------------

class CreateCvatTaskRequest(BaseModel):
    name: str = "Factory_Conveyor_Real_Bags"
    source_dir: str = "./data/extracted_frames"
    project_id: int | None = None
    cvat_url: str | None = None
    auth_token: str | None = None


@router.get("/cvat/status")
def get_cvat_status(cvat_url: str | None = None):
    """Check connectivity to self-hosted CVAT instance (§6.3)."""
    import httpx
    target_url = (cvat_url or os.environ.get("CVAT_BASE_URL", "http://localhost:8088/api")).rstrip("/")
    ui_url = target_url.replace("/api", "")
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{target_url}/server/about")
            if 200 <= resp.status_code < 300:
                data = resp.json()
                return {
                    "status": "online",
                    "base_url": target_url,
                    "ui_url": ui_url,
                    "version": data.get("version", "2.x"),
                    "name": data.get("name", "CVAT"),
                }
            return {
                "status": "offline",
                "base_url": target_url,
                "ui_url": ui_url,
                "detail": f"CVAT returned HTTP {resp.status_code}",
            }
    except Exception as exc:
        return {
            "status": "offline",
            "base_url": target_url,
            "ui_url": ui_url,
            "detail": f"Could not connect to CVAT: {exc}",
        }


@router.post("/cvat/tasks", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def create_cvat_annotation_task(req: CreateCvatTaskRequest):
    """Create a standardized 2-class annotation task in CVAT and upload extracted frames (§6.3, §6.4)."""
    import json
    from pathlib import Path
    from packages.cs_data.cvat_client import CvatClient, CvatApiError
    target_url = (req.cvat_url or os.environ.get("CVAT_BASE_URL", "http://localhost:8088/api")).rstrip("/")
    client = CvatClient(base_url=target_url, auth_token=req.auth_token)

    try:
        task = client.create_task(name=req.name, project_id=req.project_id)
        task_id = task.get("id")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CVAT task creation failed: {exc}")

    frames_dir = Path(req.source_dir)
    uploaded_count = 0
    if frames_dir.exists() and frames_dir.is_dir():
        image_files = sorted(
            [f for f in frames_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )
        if image_files:
            try:
                client.upload_task_data(task_id=task_id, image_paths=image_files[:200])
                uploaded_count = min(len(image_files), 200)
            except Exception as exc:
                logger.warning(f"Could not upload all frames to CVAT: {exc}")

    return {
        "status": "created",
        "task_id": task_id,
        "name": req.name,
        "uploaded_frames": uploaded_count,
        "task_url": f"{target_url.replace('/api', '')}/tasks/{task_id}",
    }


@router.post("/cvat/sync_dataset", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def sync_cvat_dataset(payload: dict[str, Any] = None):
    """Verify and summarize data/real_bags directory for fine-tuning (§6.3)."""
    import json
    from pathlib import Path
    target_dir = Path("./data/real_bags")
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "images").mkdir(parents=True, exist_ok=True)

    ann_file = target_dir / "annotations.json"
    images_dir = target_dir / "images"
    has_annotations = ann_file.exists()
    image_count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpeg")))

    summary = {
        "status": "ready" if (has_annotations and image_count > 0) else "awaiting_export",
        "real_bags_path": str(target_dir.resolve()),
        "has_annotations": has_annotations,
        "image_count": image_count,
        "annotation_count": 0,
        "annotated_images": 0,
    }
    if has_annotations:
        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                coco = json.load(f)
            summary["annotation_count"] = len(coco.get("annotations", []))
            summary["annotated_images"] = len(coco.get("images", []))
        except Exception as exc:
            summary["error"] = f"Invalid annotations JSON: {exc}"
    return summary


# ---------------------------------------------------------------------------
# Configs, Calibrations & Deployment Bundles
# ---------------------------------------------------------------------------

@router.get("/configs/{line_id}", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def get_latest_config(line_id: int):
    with get_sync_session() as db:
        config_repo = ConfigRepository(db)
        cfg = config_repo.get_latest_config(line_id)
        if not cfg:
            raise HTTPException(status_code=404, detail="No config found for line")
        return {"id": cfg.id, "line_id": cfg.line_id, "payload": config_repo.get_effective_config_payload(cfg)}


@router.post("/configs/{line_id}", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def create_config_version(line_id: int, req: CreateConfigRequest, user: Annotated[CurrentUser, Depends(get_current_user)]):
    with get_sync_session() as db:
        config_repo = ConfigRepository(db)
        cfg = config_repo.create_config_version(line_id=line_id, payload=req.payload, note=req.note, created_by=user.username)
        return cfg


@router.post("/calibrations/{line_id}/motion", dependencies=[Depends(require_role(UserRole.ENGINEER))], status_code=202)
def start_motion_calibration_job(line_id: int, payload: dict[str, Any]):
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        job = job_repo.submit_job(kind="calibrate_motion", payload={**payload, "line_id": line_id})
        return SubmitJobResponse(job_id=job.id, kind=job.kind)


@router.post("/calibrations/{line_id}/scale", dependencies=[Depends(require_role(UserRole.ENGINEER))], status_code=202)
def start_scale_calibration_job(line_id: int, payload: dict[str, Any]):
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        job = job_repo.submit_job(kind="calibrate_scale", payload={**payload, "line_id": line_id})
        return SubmitJobResponse(job_id=job.id, kind=job.kind)


class SetPerspectiveCalibrationRequest(BaseModel):
    roi_src_points: list[list[float]]


@router.post("/calibrations/{line_id}/perspective", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def create_perspective_calibration(line_id: int, req: SetPerspectiveCalibrationRequest, user: Annotated[CurrentUser, Depends(get_current_user)]):
    """Real Stage 3 (perspective/ROI-warp) calibration: 4 operator-marked
    points -> a real cv2 homography, applied to this camera's real frames
    before detection (packages/cs_vision/calibration.py). Computed
    synchronously, unlike the motion/scale job-queue endpoints above: this
    is a deterministic, effectively-instant linear-algebra operation, not
    work that benefits from background processing.
    """
    from packages.cs_vision.calibration import compute_homography

    try:
        homography = compute_homography(req.roi_src_points)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with get_sync_session() as db:
        calib_repo = CalibrationRepository(db)
        calib = calib_repo.create_perspective_calibration(
            line_id=line_id,
            roi_src_points=req.roi_src_points,
            homography_matrix=homography,
            created_by=user.username,
        )
        calib_id = calib.id

    # Apply immediately to this line's live renderer, if one is already
    # running, instead of requiring a stream restart to pick up the change.
    from packages.cs_counting.stream_renderer import _renderers
    renderer = _renderers.get(line_id)
    if renderer is not None:
        renderer.reload_perspective_calibration()

    return {"calibration_id": calib_id, "stage": "perspective", "homography_matrix": homography}


class SetRoiPolygonRequest(BaseModel):
    roi_polygon: list[list[float]]


@router.post("/lines/{line_id}/roi", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def set_line_roi_polygon(line_id: int, req: SetRoiPolygonRequest, user: Annotated[CurrentUser, Depends(get_current_user)]):
    """Real counting-area ROI: [0,1]-normalized polygon points, merged into
    the line's active config and re-activated as a new bundle, then applied
    immediately to the live renderer if one is running (same shape as
    /calibrations/{line_id}/perspective). Previously the frontend's ROI tool
    (togglePerspectiveWarp's sibling, toggleRoiDraw) called nothing at all
    and just showed a fake "kaydedildi" success toast -- CountingEngine now
    has a real consumer for roi_polygon (packages/cs_counting/engine.py's
    configure()/process_frame()), so this endpoint is the missing link
    between the two.
    """
    if len(req.roi_polygon) < 3:
        raise HTTPException(status_code=400, detail="roi_polygon must contain at least 3 points.")

    with get_sync_session() as db:
        config_repo = ConfigRepository(db)
        bundle = config_repo.get_active_bundle(line_id)
        if bundle is None:
            raise HTTPException(
                status_code=400,
                detail="No active model deployment for this line. Please activate a model first.",
            )

        # Merge into the current effective payload rather than replacing it
        # outright, so an already-customized confidence_threshold/
        # merge_area_ratio/etc. on this line survives an ROI-only update.
        merged_payload = config_repo.get_effective_config_payload(bundle.config_version)
        merged_payload["roi_polygon"] = req.roi_polygon

        new_config = config_repo.create_config_version(
            line_id=line_id, payload=merged_payload, note="ROI update (UI)", created_by=user.username,
        )
        new_bundle = config_repo.create_and_activate_bundle(
            line_id=line_id,
            model_version_id=bundle.model_version_id,
            config_version_id=new_config.id,
            calibration_id=bundle.calibration_id,
            activated_by=user.username,
        )
        new_bundle_id, config_version_id = new_bundle.id, new_config.id

    from packages.cs_counting.stream_renderer import _renderers
    renderer = _renderers.get(line_id)
    if renderer is not None:
        renderer.reload_active_config()

    return {"config_version_id": config_version_id, "bundle_id": new_bundle_id, "roi_polygon": req.roi_polygon}


@router.post("/bundles/activate", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def activate_deployment_bundle(req: ActivateBundleRequest, user: Annotated[CurrentUser, Depends(get_current_user)]):
    with get_sync_session() as db:
        config_repo = ConfigRepository(db)
        bundle = config_repo.create_and_activate_bundle(
            line_id=req.line_id,
            model_version_id=req.model_version_id,
            config_version_id=req.config_version_id,
            calibration_id=req.calibration_id,
            activated_by=user.username,
        )
        bundle_id, line_id = bundle.id, bundle.line_id

    # Apply the newly-active config (confidence_threshold, merge_area_ratio,
    # roi_polygon, etc. -- see CountingEngine.configure()) to this line's
    # live renderer immediately, if one is running, instead of requiring a
    # stream restart. Previously activating a bundle never reached a
    # running engine at all -- see reload_active_config()'s docstring.
    from packages.cs_counting.stream_renderer import _renderers
    renderer = _renderers.get(line_id)
    if renderer is not None:
        renderer.reload_active_config()

    return bundle


# ---------------------------------------------------------------------------
# System Health, Jobs, Outbox
# ---------------------------------------------------------------------------

@router.get("/system/health")
def get_system_health(user: Annotated[CurrentUser, Depends(get_current_user)]):
    with get_sync_session() as db:
        nodes = db.query(NodeORM).all()
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc),
            "nodes": [{"id": n.id, "hostname": n.hostname, "status": n.status} for n in nodes],
        }


@router.get("/system/jobs")
def list_system_jobs(user: Annotated[CurrentUser, Depends(get_current_user)], limit: int = 50, offset: int = 0):
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        # JobRepository.list_jobs() only takes a limit (no offset); out of scope
        # here to add offset support to cs_storage repositories, so fetch enough
        # rows and slice the already created_at-desc ordered result.
        jobs = job_repo.list_jobs(limit=offset + limit)
        return jobs[offset : offset + limit]


@router.get("/system/outbox", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def list_outbox_entries():
    with get_sync_session() as db:
        return db.query(OutboxORM).order_by(OutboxORM.created_at.desc()).limit(50).all()


@router.post("/system/jobs/{job_id}/cancel", dependencies=[Depends(require_role(UserRole.ENGINEER))])
def cancel_job(job_id: int):
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        success = job_repo.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=400, detail="Cannot cancel job")
        return {"status": "cancelled", "job_id": job_id}


class SimulateBagRequest(BaseModel):
    direction: int = 1
    confidence: float = 0.99
    merge_flag: bool = False
    track_id: int | None = None
    defect_reason: str | None = None


@router.post("/sessions/{sess_id}/simulate_bag", dependencies=[Depends(require_role(UserRole.OPERATOR, UserRole.ENGINEER))])
def simulate_bag_crossing(sess_id: int, req: SimulateBagRequest = SimulateBagRequest(), user: Annotated[CurrentUser, Depends(get_current_user)] = None):
    """Simulate a physical conveyor bag passing the optical gate and update immutable ledger."""
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        ledger_repo = LedgerRepository(db)
        sess = session_repo.get_by_id(sess_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        if sess.status not in ["open", "counting"]:
            sess.status = "counting"
            db.commit()

        # Find camera, gate, bundle -- camera_id/gate_id/deployment_bundle_id
        # are all NOT NULL foreign keys on count_event, so a real crossing
        # genuinely cannot be recorded without all three actually existing
        # for this line. Previously fell back to a hardcoded `1` for
        # whichever was missing -- silently wrong on any line whose real
        # camera/gate/bundle didn't happen to have id 1 (verified directly:
        # a real ForeignKeyViolation, not a hypothetical), not a safe default.
        cam = db.query(CameraORM).filter(CameraORM.line_id == sess.line_id).first()
        gate = db.query(GateORM).filter(GateORM.line_id == sess.line_id).first()
        bundle = db.query(DeploymentBundleORM).filter(DeploymentBundleORM.line_id == sess.line_id, DeploymentBundleORM.deactivated_at == None).first()

        missing = [
            name for name, row in (("camera", cam), ("gate", gate), ("active model deployment", bundle))
            if row is None
        ]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"The following are not configured for this line; cannot run simulation: {', '.join(missing)}.",
            )

        cam_id = cam.id
        gate_id = gate.id
        bundle_id = bundle.id

        # Current sequence and unique track_id
        import time, random
        current_events_count = len(ledger_repo.get_session_events(sess_id))
        track_id = req.track_id if req.track_id is not None else (int(time.time() * 1000000 + random.randint(100, 999)) % 100000000)

        # Record in immutable ledger and update session totals -- routed
        # through CountingEventHandler (packages/cs_counting/event_handler.py),
        # the one shared implementation also used by LiveStreamRenderer's two
        # frame paths and InferenceWorker. area_estimate now comes from
        # estimate_simulated_area() (a flat multiply derived fresh from the
        # ledger-true counted_total), replacing this endpoint's previous
        # incremental +/-0.998 running delta, which had already diverged from
        # LiveStreamRenderer._process_simulated_frame's own (different) flat
        # multiply -- the two demo/manual paths now agree.
        event_handler = CountingEventHandler(db)
        crossing = GateCrossingEvent(
            track_id=track_id,
            crossing_seq=1,
            gate_id=gate_id,
            direction=req.direction,
            crossing_timestamp=datetime.now(timezone.utc),
            frame_index=(current_events_count + 1) * 300,
            monotonic_ns=time.perf_counter_ns(),
            confidence=req.confidence,
            merge_flag=req.merge_flag,
            centroid=(0.0, 0.0),
        )
        stream_epoch = CameraEpochRepository(db).get_current_epoch(cam_id) or 1
        ev, created = event_handler.handle_gate_crossing(GateCrossingRecorded(
            line_id=sess.line_id,
            camera_id=cam_id,
            session_id=sess.id,
            stream_epoch=stream_epoch,
            deployment_bundle_id=bundle_id,
            crossing=crossing,
            is_simulated=True,
            defect_reason=req.defect_reason,
            evidence_ref=f"/evidence/frames/sess_{sess.id}_trk_{track_id}.jpg",
        ))
        if created:
            net_count = event_handler.ledger_repo.get_session_total_count(sess.id)
            event_handler.handle_area_updated(SessionAreaEstimateUpdated(
                session_id=sess.id, area_estimate=estimate_simulated_area(net_count),
            ))
        db.refresh(sess)

        # Also reflect this manual "+1 Damaged" / "+2 Merged" trigger on the
        # live MJPEG demo feed (if a renderer for this line is running), so
        # the defect/multi badges in LiveStreamRenderer._process_simulated_frame
        # actually render instead of being permanently dead code.
        if req.defect_reason or req.merge_flag:
            from packages.cs_counting.stream_renderer import _renderers
            renderer = _renderers.get(sess.line_id)
            if renderer is not None:
                renderer.spawn_manual_bag(
                    defective=bool(req.defect_reason),
                    bag_count_estimate=2 if req.merge_flag else 1,
                    label=req.defect_reason,
                )

        return {
            "status": "ok",
            "session_id": sess.id,
            "counted_total": sess.counted_total,
            "area_estimate_total": round(sess.area_estimate_total, 1),
            "event_id": ev.event_id if ev else None,
            "direction": req.direction,
        }


