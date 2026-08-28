"""Live Video Stream Renderer & Real Computer Vision Annotator (§6.1, §9.6)."""

from __future__ import annotations

import logging
import math
import random
import time
from datetime import datetime, timezone
from typing import Generator
import cv2
import numpy as np

from packages.cs_counting.engine import CountingEngine
from packages.cs_counting.event_handler import CountingEventHandler, estimate_simulated_area
from packages.cs_counting.events import GateCrossingRecorded, SessionAreaEstimateUpdated
from packages.cs_counting.gate import GateCrossingEvent
from packages.cs_storage.db import get_sync_session
from packages.cs_storage.models_orm import CameraORM, GateORM, ProductProfileORM, SessionORM
from packages.cs_storage.repositories.calibration_repo import CalibrationRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from packages.cs_vision.calibration import apply_perspective_warp

logger = logging.getLogger(__name__)


class LiveStreamRenderer:
    """Renders real-time AI vision camera frames with amodal masks, bounding boxes, and gate line."""

    def __init__(self, line_id: int = 1, width: int = 640, height: int = 640) -> None:
        # Square, matching the detector's native training/inference resolution.
        # A non-square canvas (e.g. the previous 800x400) forces every real
        # camera frame through a more aggressive letterbox scale-down before
        # detection -- verified this alone was enough to push real detection
        # confidence to the edge of the tracker's threshold and make gate
        # crossings fail intermittently even with a genuinely working model.
        self.line_id = line_id
        self.w = width
        self.h = height
        self.engine = CountingEngine()
        self.frame_idx = 0
        self.belt_pos = 0.0
        self.gate_x = 480
        self.belt_speed_px = 6.0
        self.belt_dir = 1
        self.bags: list[dict] = []
        self.next_sim_id = 1000
        self.video_cap: cv2.VideoCapture | None = None
        self.last_crossing_time = 0.0
        self._last_frame_time: float | None = None
        self._fps_ema: float | None = None
        self.homography_matrix: list[list[float]] | None = None
        self.reload_perspective_calibration()

        # Initialize physical bags on conveyor
        self.bags.append({"x": 100.0, "y": 140.0, "w": 110, "h": 150, "label": "50kg Çimento", "color": (40, 180, 240), "id": self.next_sim_id, "passed": False})
        self.next_sim_id += 1
        self.bags.append({"x": 340.0, "y": 140.0, "w": 110, "h": 150, "label": "50kg Çimento", "color": (40, 180, 240), "id": self.next_sim_id, "passed": False})
        self.next_sim_id += 1

    def set_camera_source(self, source: str | int) -> tuple[bool, str]:
        """Set RTSP IP URL (e.g. 'rtsp://...'), local USB webcam index (0, 1), video file, or 'demo'."""
        if self.video_cap:
            try:
                self.video_cap.release()
            except Exception:
                pass
            self.video_cap = None

        source_str = str(source).strip()
        if source_str in ["demo", "sim", ""]:
            return True, "Simüle endüstriyel konveyör akışına geçildi."

        # USB Webcam index
        if source_str.isdigit() or source_str.lower() in ["webcam", "camera"]:
            dev_idx = int(source_str) if source_str.isdigit() else 0
            cap = cv2.VideoCapture(dev_idx)
            if cap.isOpened():
                self.video_cap = cap
                return True, f"Yerel USB / Web kamerası #{dev_idx} başarıyla bağlandı."
            return False, f"Yerel kamera #{dev_idx} açılamadı (Kamera izinleri veya bağlantı kontrol edilmeli)."

        # RTSP / HTTP URL or video file
        cap = cv2.VideoCapture(source_str)
        if cap.isOpened():
            self.video_cap = cap
            return True, f"Kamera akışı başarıyla bağlandı: {source_str}"
        return False, f"Kamera akışına bağlanılamadı: {source_str}"

    def reload_perspective_calibration(self) -> None:
        """(Re)load this line's active Stage-3 perspective calibration, if any.

        Called at construction and after a new perspective calibration is
        saved (see the /calibrations/{line_id}/perspective API endpoint), so
        an operator recalibrating a camera doesn't need to restart the
        stream. No active calibration means no active ROI-warp -- the raw
        frame goes straight to detection, matching a camera framed the same
        way the model was trained against.
        """
        try:
            with get_sync_session() as db:
                calib = CalibrationRepository(db).get_active_calibration(self.line_id, stage="perspective")
                self.homography_matrix = calib.homography_matrix if calib else None
        except Exception:
            logger.exception("Failed to load perspective calibration for line_id=%s", self.line_id)
            self.homography_matrix = None

    def set_video_source(self, video_path: str) -> bool:
        """Set a real MP4 / AVI video file as input source."""
        ok, _ = self.set_camera_source(video_path)
        return ok

    def spawn_manual_bag(
        self,
        defective: bool = False,
        bag_count_estimate: int = 1,
        label: str | None = None,
    ) -> dict:
        """Inject an operator-triggered bag into the simulated conveyor demo.

        This is the wiring for the "+1 Hasarlı / Patlak" and "+2 Bitişik
        Çuval" manual simulation triggers: without it, `is_defective` /
        `bag_count_estimate` were never set on any `self.bags` entry (only
        auto-spawned plain bags exist), so the defect/multi badge branches in
        `_process_simulated_frame` were unreachable dead code. Callers (e.g.
        the `/sessions/{id}/simulate_bag` API) should invoke this on the
        renderer for the session's line whenever `defect_reason` /
        `merge_flag` is set, so the MJPEG demo feed actually renders the
        corresponding red DEFECT / amber BITISIK badge and mask color.
        """
        entry_x = -120.0 if self.belt_dir > 0 else float(self.w + 40)
        bag = {
            "x": entry_x,
            "y": 140.0,
            "w": 110,
            "h": 150,
            "label": label or ("Patlak Çuval" if defective else "50kg Çimento"),
            "color": (40, 40, 220) if defective else (40, 180, 240),
            "id": self.next_sim_id,
            "passed": False,
            "is_defective": defective,
            "bag_count_estimate": max(1, bag_count_estimate),
        }
        self.next_sim_id += 1
        if self.belt_dir > 0:
            self.bags.append(bag)
        else:
            self.bags.insert(0, bag)
        return bag

    def generate_conveyor_frame(self) -> np.ndarray:
        """Synthesize a photorealistic industrial conveyor camera frame."""
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # 1. Factory floor background (dark slate with depth grid)
        frame[:] = (20, 25, 35)
        for y in range(0, self.h, 40):
            cv2.line(frame, (0, y), (self.w, y), (28, 35, 48), 1)

        # 2. Conveyor bed chassis (metallic frame)
        belt_top = 80
        belt_bottom = 320
        cv2.rectangle(frame, (0, belt_top - 12), (self.w, belt_bottom + 12), (50, 60, 75), -1)
        cv2.rectangle(frame, (0, belt_top), (self.w, belt_bottom), (15, 20, 28), -1)

        # Moving belt texture rollers and grooves
        self.belt_pos = (self.belt_pos + self.belt_speed_px * self.belt_dir) % 50
        for x in range(int(-50 + self.belt_pos), self.w + 50, 50):
            cv2.line(frame, (x, belt_top), (x, belt_bottom), (35, 45, 58), 2)

        # 3. Render physical bags moving on conveyor
        for bag in self.bags:
            bx, by, bw, bh = int(bag["x"]), int(bag["y"]), int(bag["w"]), int(bag["h"])
            
            # Shadow
            cv2.rectangle(frame, (bx + 8, by + 8), (bx + bw + 8, by + bh + 8), (8, 10, 15), -1)
            
            # Bag textured body (Kraft / Poly)
            b_color = bag.get("color", (40, 180, 240))
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), b_color, -1)
            
            # Bag 3D shading / folds
            cv2.rectangle(frame, (bx + 4, by + 4), (bx + bw - 4, by + bh - 4), (b_color[0] + 20, b_color[1] + 20, min(255, b_color[2] + 20)), 2)
            cv2.line(frame, (bx + 10, by + bh // 2), (bx + bw - 10, by + bh // 2), (b_color[0] - 25, b_color[1] - 25, max(0, b_color[2] - 25)), 2)
            
            # Print label
            cv2.putText(frame, bag.get("label", "50kg Çimento")[:14], (bx + 8, by + bh // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (15, 20, 25), 1, cv2.LINE_AA)
            cv2.putText(frame, "FABRIC #2026", (bx + 8, by + bh // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 40, 50), 1, cv2.LINE_AA)

            # Move bag
            bag["x"] += self.belt_speed_px * self.belt_dir

        # Spawn new bags
        if self.belt_dir > 0:
            if not self.bags or self.bags[-1]["x"] > 240:
                self.bags.append({
                    "x": -120.0,
                    "y": 140.0,
                    "w": 110,
                    "h": 150,
                    "label": "50kg Çimento",
                    "color": (40, 180, 240),
                    "id": self.next_sim_id,
                    "passed": False,
                })
                self.next_sim_id += 1
            # Remove offscreen
            self.bags = [b for b in self.bags if b["x"] < self.w + 150]
        else:
            if not self.bags or self.bags[0]["x"] < self.w - 240:
                self.bags.insert(0, {
                    "x": float(self.w + 40),
                    "y": 140.0,
                    "w": 110,
                    "h": 150,
                    "label": "50kg Çimento",
                    "color": (40, 180, 240),
                    "id": self.next_sim_id,
                    "passed": False,
                })
                self.next_sim_id += 1
            self.bags = [b for b in self.bags if b["x"] > -150]

        return frame

    def process_and_annotate_frame(self, frame: np.ndarray, session_id: int | None = None) -> tuple[np.ndarray, int | None]:
        """Run full CV segmentation, tracking, gate crossing, and draw HUD annotations.

        Real camera/video source (`self.video_cap` set): runs the actual
        CountingEngine (VisionDetector -> ByteTracker -> GateStateMachine) and
        writes ledger events from its real gate-crossing output. No camera
        connected (demo/simulated conveyor): uses the synthetic `self.bags`
        animation -- an explicit, labeled demo mode, never a silent substitute
        for real detection when a camera IS connected.
        """
        self.frame_idx += 1
        t_now = datetime.now(timezone.utc)
        mono_ns = int(time.perf_counter() * 1e9)

        if self.video_cap is not None:
            return self._process_real_frame(frame, session_id, t_now, mono_ns)
        return self._process_simulated_frame(frame, session_id, t_now, mono_ns)

    def _process_real_frame(
        self, frame: np.ndarray, session_id: int | None, t_now: datetime, mono_ns: int
    ) -> tuple[np.ndarray, int | None]:
        self.engine.gate_state_machine.update_geometry(
            axis_origin=(0.0, 0.0), axis_vector=(1.0, 0.0), gate_pos=float(self.gate_x)
        )

        # Stage 3 calibration: warp this camera's real belt ROI into the
        # canonical view anchor_grid() was trained against, before handing
        # the frame to the detector. A camera framed/mounted differently
        # than the reference setup would otherwise have every anchor
        # silently pointed at the wrong part of the image.
        detect_frame = frame
        if self.homography_matrix is not None:
            detect_frame = apply_perspective_warp(frame, self.homography_matrix)

        out = self.engine.process_frame(
            image=detect_frame, frame_index=self.frame_idx, monotonic_ns=mono_ns, wall_clock=t_now
        )

        if out.gate_crossings:
            self.last_crossing_time = time.time()

        # Routed through CountingEventHandler (packages/cs_counting/event_handler.py)
        # unconditionally, not gated behind `if out.gate_crossings:` -- the
        # previous inline version only updated area_estimate_total on frames
        # that also had a crossing, unlike InferenceWorker's reference
        # implementation of this same logic (which updates it every frame).
        # That gap is closed by going through the same shared handler both
        # paths now use.
        if session_id:
            try:
                with get_sync_session() as db:
                    handler = CountingEventHandler(db)
                    handler.handle_frame_output(
                        out,
                        line_id=self.line_id,
                        camera_id=1,
                        session_id=session_id,
                        stream_epoch=4,
                    )
            except Exception:
                logger.exception(
                    "Failed to record real gate-crossing ledger event(s) for session_id=%s, line_id=%s",
                    session_id, self.line_id,
                )

        # Annotate on detect_frame (not the raw frame): track boxes/masks are
        # in the coordinate space the detector actually saw, i.e. the warped
        # canvas when a perspective calibration is active -- drawing them on
        # the un-warped original would misplace every box.
        annotated = detect_frame.copy()
        overlay = detect_frame.copy()
        for track in out.active_tracks:
            bx1, by1, bx2, by2 = [int(v) for v in track.box]
            box_color = (255, 140, 90)
            pts = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], np.int32)
            cv2.fillPoly(overlay, [pts], (240, 100, 99))
            cv2.rectangle(annotated, (bx1 - 2, by1 - 2), (bx2 + 2, by2 + 2), box_color, 2)
            badge_text = f"TRK-{track.track_id} {track.score * 100:.1f}%"
            cv2.rectangle(annotated, (bx1 - 2, by1 - 22), (bx1 + 115, by1 - 2), (180, 50, 40), -1)
            cv2.putText(annotated, badge_text, (bx1 + 3, by1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

        self._draw_gate_and_hud(annotated, real_mode=True)
        return annotated, session_id

    def _process_simulated_frame(
        self, frame: np.ndarray, session_id: int | None, t_now: datetime, mono_ns: int
    ) -> tuple[np.ndarray, int | None]:
        # Check gate crossings in simulated bag objects
        for bag in self.bags:
            mid_x = bag["x"] + bag["w"] / 2
            if not bag.get("passed", False):
                if (self.belt_dir > 0 and mid_x >= self.gate_x) or (self.belt_dir < 0 and mid_x <= self.gate_x):
                    bag["passed"] = True
                    self.last_crossing_time = time.time()

                    # Record to real database ledger (explicit demo/simulated crossing).
                    # Routed through CountingEventHandler, same as the real-frame path
                    # -- area estimate now comes from estimate_simulated_area(), the one
                    # canonical simulated-area heuristic shared with
                    # services/api/routes.py::simulate_bag_crossing (which previously
                    # used a different, diverged +/-0.998 incremental-delta formula
                    # instead of this flat multiply).
                    if session_id:
                        try:
                            with get_sync_session() as db:
                                handler = CountingEventHandler(db)
                                crossing = GateCrossingEvent(
                                    track_id=bag["id"],
                                    crossing_seq=1,
                                    gate_id=1,
                                    direction=self.belt_dir,
                                    crossing_timestamp=t_now,
                                    frame_index=self.frame_idx,
                                    monotonic_ns=mono_ns,
                                    confidence=0.985,
                                    merge_flag=False,
                                    centroid=(bag["x"] + bag["w"] / 2, bag["y"] + bag["h"] / 2),
                                )
                                _, created = handler.handle_gate_crossing(GateCrossingRecorded(
                                    line_id=self.line_id,
                                    camera_id=1,
                                    session_id=session_id,
                                    stream_epoch=4,
                                    deployment_bundle_id=1,
                                    crossing=crossing,
                                    is_simulated=True,
                                ))
                                if created:
                                    net_total = handler.ledger_repo.get_session_total_count(session_id)
                                    handler.handle_area_updated(SessionAreaEstimateUpdated(
                                        session_id=session_id,
                                        area_estimate=estimate_simulated_area(net_total),
                                    ))
                        except Exception:
                            logger.exception(
                                "Failed to record simulated gate-crossing ledger event for "
                                "session_id=%s, line_id=%s, bag_id=%s",
                                session_id, self.line_id, bag["id"],
                            )

        # 2. Draw Real AI Vision Overlays
        annotated = frame.copy()
        overlay = frame.copy()

        # Draw Amodal Segmentations & Bounding Boxes
        for bag in self.bags:
            bx, by, bw, bh = int(bag["x"]), int(bag["y"]), int(bag["w"]), int(bag["h"])
            is_defective = bag.get("is_defective", False)
            multi_count = bag.get("bag_count_estimate", 1)

            # Mask color (Red for defect, Green for multi, Indigo for standard)
            if is_defective:
                mask_color = (30, 30, 230) # Red
                box_color = (50, 50, 255)
            elif multi_count > 1:
                mask_color = (230, 150, 30) # Amber / Cyan
                box_color = (255, 180, 50)
            else:
                mask_color = (240, 100, 99) # Indigo
                box_color = (255, 140, 90)

            # Polygon mask
            pts = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], np.int32)
            cv2.fillPoly(overlay, [pts], mask_color)
            
            # Bounding Box corners
            cv2.rectangle(annotated, (bx - 2, by - 2), (bx + bw + 2, by + bh + 2), box_color, 2)
            
            # Tracking & Defect Badge
            if is_defective:
                badge_text = f"🚨 DEFECT-HASARLI {bag['id']}"
                cv2.rectangle(annotated, (bx - 2, by - 22), (bx + 160, by - 2), (30, 30, 200), -1)
                cv2.putText(annotated, badge_text, (bx + 3, by - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            elif multi_count > 1:
                badge_text = f"📦 2x BITISIK TRK-{bag['id']}"
                cv2.rectangle(annotated, (bx - 2, by - 22), (bx + 155, by - 2), (180, 110, 20), -1)
                cv2.putText(annotated, badge_text, (bx + 3, by - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                badge_text = f"TRK-{bag['id']} 98.6%"
                cv2.rectangle(annotated, (bx - 2, by - 22), (bx + 115, by - 2), (180, 50, 40), -1)
                cv2.putText(annotated, badge_text, (bx + 3, by - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend semi-transparent segmentation masks
        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

        self._draw_gate_and_hud(annotated, real_mode=False)
        return annotated, session_id

    def _draw_gate_and_hud(self, annotated: np.ndarray, real_mode: bool) -> None:
        # Optical Gate Laser Line (Glowing Neon Green)
        gate_color = (80, 255, 120)
        is_flashing = (time.time() - self.last_crossing_time) < 0.25
        if is_flashing:
            cv2.line(annotated, (self.gate_x, 40), (self.gate_x, 360), (120, 255, 255), 6)
            cv2.circle(annotated, (self.gate_x, 200), 30, (80, 255, 120), 3)
        else:
            cv2.line(annotated, (self.gate_x, 50), (self.gate_x, 350), gate_color, 2)
            cv2.line(annotated, (self.gate_x - 1, 50), (self.gate_x - 1, 350), (120, 255, 180), 1)

        cv2.rectangle(annotated, (self.gate_x - 48, 40), (self.gate_x + 48, 62), (40, 160, 80), -1)
        cv2.putText(annotated, "SAYIM KAPISI", (self.gate_x - 42, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Top HUD: real, measured FPS (rolling EMA of actual frame timing), not a fixed number
        cv2.rectangle(annotated, (0, 0), (self.w, 32), (10, 14, 22), -1)
        cv2.line(annotated, (0, 32), (self.w, 32), (45, 55, 75), 1)

        now = time.time()
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps_ema = inst_fps if self._fps_ema is None else (0.9 * self._fps_ema + 0.1 * inst_fps)
        self._last_frame_time = now
        fps_display = f"{self._fps_ema:.1f}" if self._fps_ema is not None else "—"

        mode_label = "AI VISION ACTIVE [RF-DETR + ByteTrack]" if real_mode else "SIMULE KONVEYOR [Demo]"
        cv2.putText(annotated, f"● {mode_label}", (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 240, 120), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"FPS: {fps_display} | Gate X: {self.gate_x}px", (self.w - 220, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 220), 1, cv2.LINE_AA)


# Global instance per line
_renderers: dict[int, LiveStreamRenderer] = {}


def get_stream_generator(line_id: int = 1) -> Generator[bytes, None, None]:
    """Continuous generator yielding MJPEG stream frames."""
    if line_id not in _renderers:
        _renderers[line_id] = LiveStreamRenderer(line_id=line_id)

    renderer = _renderers[line_id]
    cached_session_id = None
    last_session_check = 0.0

    while True:
        # Check active session periodically (every 1s) to avoid DB lock and query overhead on every frame
        now = time.time()
        if now - last_session_check > 1.0:
            last_session_check = now
            try:
                with get_sync_session() as db:
                    sess_repo = SessionRepository(db)
                    sess = sess_repo.get_active_session(line_id)
                    cached_session_id = sess.id if sess else None
            except Exception:
                logger.exception("Failed to look up active session for line_id=%s", line_id)
                cached_session_id = None

        session_id = cached_session_id

        # 1. Acquire frame
        if renderer.video_cap and renderer.video_cap.isOpened():
            ret, frame = renderer.video_cap.read()
            if not ret:
                renderer.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = renderer.video_cap.read()
            # Letterbox (pad, don't stretch) to the display canvas size. A plain
            # cv2.resize distorts the camera's real aspect ratio into whatever
            # renderer.w/h is (e.g. a square-ish real frame squashed to 800x400
            # 2:1), which visually warps bags out of the shape the detector was
            # trained on -- verified this alone was enough to drop detections
            # to zero on every real frame regardless of model quality.
            from packages.cs_vision.preprocess import letterbox_image
            frame, _, _ = letterbox_image(frame, (renderer.h, renderer.w), fill_value=20)
        else:
            frame = renderer.generate_conveyor_frame()

        # 2. Process and annotate
        annotated, _ = renderer.process_and_annotate_frame(frame, session_id=session_id)

        # 3. Encode to JPEG
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = jpeg.tobytes()

        # 4. Yield multipart chunk
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.02)  # ~30-40 FPS smooth streaming

