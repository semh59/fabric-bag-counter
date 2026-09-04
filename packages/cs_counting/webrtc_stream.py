"""WebRTC Low-Latency Video Streaming Engine for Live Industrial HUD (§6.1, §9.6).

Provides real-time peer-to-peer WebRTC video streaming with sub-50ms latency using
aiortc and PyAV, delivering annotated bounding boxes and amodal segmentations directly
to modern web browsers without MJPEG buffering or HTTP overhead.
"""

from __future__ import annotations

import asyncio
import collections
import fractions
import logging
import time
from typing import Any

import av
import numpy as np
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)

logger = logging.getLogger(__name__)

# Default public STUN servers for WebRTC NAT traversal
DEFAULT_ICE_SERVERS = [
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
]


class AnnotatedVideoTrack(VideoStreamTrack):
    """Custom WebRTC VideoStreamTrack yielding real-time AI-annotated conveyor frames."""

    kind = "video"

    def __init__(self, line_id: int = 1, fps: int = 30) -> None:
        super().__init__()
        self.line_id = line_id
        self.fps = fps
        self.time_base = fractions.Fraction(1, fps)
        self._timestamp = 0
        self._start_time: float | None = None

    async def recv(self) -> av.VideoFrame:
        """Fetch next annotated frame from the counting engine renderer and package as av.VideoFrame."""
        pts = self._timestamp
        self._timestamp += 1

        # Maintain smooth target framerate pacing
        target_delay = 1.0 / self.fps
        if self._start_time is None:
            self._start_time = time.monotonic()
        else:
            expected_time = self._start_time + (pts * target_delay)
            now = time.monotonic()
            wait_seconds = expected_time - now
            if wait_seconds > 0.002:
                await asyncio.sleep(wait_seconds)

        from packages.cs_counting.stream_renderer import LiveStreamRenderer, _renderers

        if self.line_id not in _renderers:
            _renderers[self.line_id] = LiveStreamRenderer(line_id=self.line_id)
        renderer = _renderers[self.line_id]

        # Acquire frame (runs in worker thread if necessary, but get_next_annotated_frame is fast)
        bgr_frame = renderer.get_next_annotated_frame()

        # Wrap numpy BGR array directly into an av.VideoFrame (zero-copy when possible)
        video_frame = av.VideoFrame.from_ndarray(bgr_frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = self.time_base
        return video_frame


class WebRtcManager:
    """Manages WebRTC PeerConnections, SDP negotiations, and active video tracks per line."""

    def __init__(self, max_peers_per_line: int = 16) -> None:
        self.max_peers_per_line = max_peers_per_line
        self.active_pcs: set[RTCPeerConnection] = set()
        self.pcs_by_line: dict[int, set[RTCPeerConnection]] = collections.defaultdict(set)
        self._lock = asyncio.Lock()

    async def handle_offer(
        self,
        line_id: int,
        sdp: str,
        type_: str = "offer",
        ice_servers: list[RTCIceServer] | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Handle incoming browser SDP offer and generate answer with annotated video track."""
        if not sdp:
            raise ValueError("SDP offer content cannot be empty.")

        actual_type = kwargs.get("type", type_)


        async with self._lock:
            current_line_pcs = self.pcs_by_line[line_id]
            if len(current_line_pcs) >= self.max_peers_per_line:
                # Evict oldest peer connection to prevent resource exhaustion
                oldest = next(iter(current_line_pcs))
                await self._close_pc_internal(oldest, line_id)

            config = RTCConfiguration(iceServers=ice_servers or DEFAULT_ICE_SERVERS)
            pc = RTCPeerConnection(configuration=config)
            self.active_pcs.add(pc)
            self.pcs_by_line[line_id].add(pc)

        # Attach real-time video track
        video_track = AnnotatedVideoTrack(line_id=line_id)
        pc.addTrack(video_track)

        @pc.on("connectionstatechange")
        async def on_state_change() -> None:
            logger.info(f"[WebRTC Line {line_id}] Connection state -> {pc.connectionState}")
            if pc.connectionState in ("failed", "closed"):
                await self.close_pc(pc, line_id)

        # Process SDP offer and create answer
        offer = RTCSessionDescription(sdp=sdp, type=actual_type)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        assert pc.localDescription is not None
        logger.info(f"[WebRTC Line {line_id}] Negotiated WebRTC connection (active total: {len(self.active_pcs)})")
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def close_pc(self, pc: RTCPeerConnection, line_id: int | None = None) -> None:
        """Close a specific peer connection."""
        async with self._lock:
            await self._close_pc_internal(pc, line_id)

    async def _close_pc_internal(self, pc: RTCPeerConnection, line_id: int | None = None) -> None:
        self.active_pcs.discard(pc)
        if line_id is not None:
            self.pcs_by_line[line_id].discard(pc)
        else:
            for line_set in self.pcs_by_line.values():
                line_set.discard(pc)
        try:
            await pc.close()
        except Exception as e:
            logger.debug(f"[WebRTC] Error closing peer connection: {e}")

    async def close_all(self) -> None:
        """Tear down all active WebRTC connections."""
        async with self._lock:
            pcs = list(self.active_pcs)
            for pc in pcs:
                await self._close_pc_internal(pc)

    def get_stats(self, line_id: int | None = None) -> dict[str, Any]:
        """Return operational stats for active WebRTC streams."""
        if line_id is not None:
            line_pcs = self.pcs_by_line.get(line_id, set())
            return {
                "line_id": line_id,
                "active_peers": len(line_pcs),
                "peer_states": [pc.connectionState for pc in line_pcs],
            }
        return {
            "total_active_peers": len(self.active_pcs),
            "lines_streaming": {
                lid: len(pcs) for lid, pcs in self.pcs_by_line.items() if len(pcs) > 0
            },
        }


# Global WebRtcManager singleton
_webrtc_manager: WebRtcManager | None = None


def get_webrtc_manager() -> WebRtcManager:
    global _webrtc_manager
    if _webrtc_manager is None:
        _webrtc_manager = WebRtcManager()
    return _webrtc_manager
