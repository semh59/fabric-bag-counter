"""Tests for Real-Time WebRTC Low-Latency Video Streaming Pipeline (§6.1, §9.6)."""

from __future__ import annotations

import asyncio
import pytest
import av
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi.testclient import TestClient

from packages.cs_counting.webrtc_stream import AnnotatedVideoTrack, WebRtcManager, get_webrtc_manager
from services.api.main import app

client = TestClient(app)


def test_annotated_video_track_recv():
    async def _run():
        track = AnnotatedVideoTrack(line_id=1, fps=30)
        assert track.kind == "video"

        # Fetch first frame
        frame = await track.recv()
        assert isinstance(frame, av.VideoFrame)
        assert frame.width > 0
        assert frame.height > 0
        assert frame.pts == 0
        assert frame.time_base.numerator == 1
        assert frame.time_base.denominator == 30

        # Fetch second frame
        frame2 = await track.recv()
        assert frame2.pts == 1

    asyncio.run(_run())


def test_webrtc_manager_offer_answer():
    async def _run():
        manager = WebRtcManager()

        # Create client-side peer connection
        client_pc = RTCPeerConnection()
        client_pc.addTransceiver("video", direction="recvonly")
        offer = await client_pc.createOffer()
        await client_pc.setLocalDescription(offer)

        # Server processes offer and creates answer
        answer_dict = await manager.handle_offer(line_id=1, sdp=client_pc.localDescription.sdp, type="offer")
        assert "sdp" in answer_dict
        assert answer_dict["type"] == "answer"
        assert "m=video" in answer_dict["sdp"]

        # Client accepts answer
        answer_desc = RTCSessionDescription(sdp=answer_dict["sdp"], type=answer_dict["type"])
        await client_pc.setRemoteDescription(answer_desc)

        stats = manager.get_stats(line_id=1)
        assert stats["active_peers"] >= 1

        # Cleanup
        await client_pc.close()
        await manager.close_all()

    asyncio.run(_run())



def test_webrtc_api_endpoints():
    # 1. Stats endpoint
    res = client.get("/api/live/lines/1/webrtc/stats")
    assert res.status_code == 200
    stats = res.json()
    assert "active_peers" in stats

    # 2. Invalid offer error handling
    bad_res = client.post("/api/live/lines/1/webrtc/offer", json={"sdp": "", "type": "offer"})
    assert bad_res.status_code == 400
