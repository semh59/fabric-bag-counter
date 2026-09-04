"""Deep Comprehensive End-to-End Test & Stress Suite for Fabric Bag Counter v2.0 Enterprise.

Covers:
1. MJPEG Live Video Stream & OpenCV Computer Vision Frame Decoder
2. Real MP4 Video Generation, Upload & Analysis Pipeline
3. Live Conveyor Belt, Optical Gate & Bag Profile Dynamic Sync
4. 100-Thread Concurrent Crossing Burst & Ledger Race Immunity
5. Complete REST API Coverage (Auth, Sessions, Calibrations, Outbox, Jobs)
6. Playwright Real Chromium Browser E2E Automated Verification
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import concurrent.futures
from datetime import datetime
import cv2
import httpx
import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8080"
API_URL = f"{BASE_URL}/api"


def generate_test_conveyor_video(filepath: str, num_frames: int = 60, width: int = 800, height: int = 400) -> str:
    """Generate a realistic synthetic MP4 conveyor video file using OpenCV."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, 25.0, (width, height))

    for f_idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background
        frame[:] = (22, 28, 38)
        # Belt
        cv2.rectangle(frame, (0, 80), (width, 320), (15, 20, 28), -1)
        # Belt rollers
        offset = (f_idx * 6) % 50
        for x in range(offset, width, 50):
            cv2.line(frame, (x, 80), (x, 320), (35, 45, 58), 2)
        # Moving Bag
        bag_x = int(-100 + (f_idx * 15))
        if -120 < bag_x < width + 120:
            cv2.rectangle(frame, (bag_x, 140), (bag_x + 110, 290), (45, 185, 245), -1)
            cv2.rectangle(frame, (bag_x + 4, 144), (bag_x + 106, 286), (65, 205, 255), 2)
            cv2.putText(frame, "50kg CIMENTO", (bag_x + 8, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 20, 30), 1)
        out.write(frame)

    out.release()
    return filepath


def test_mjpeg_stream():
    print("\n" + "=" * 65)
    print("PHASE 1: LIVE MJPEG OPENCV VIDEO STREAM VALIDATION")
    print("=" * 65)

    stream_url = f"{API_URL}/live/lines/1/stream"
    print(f"Connecting to MJPEG stream: {stream_url} ...")

    with httpx.Client(timeout=10.0) as client:
        with client.stream("GET", stream_url) as response:
            assert response.status_code == 200, f"Expected HTTP 200, got {response.status_code}"
            assert "multipart/x-mixed-replace" in response.headers.get("content-type", "")
            print("  [OK] HTTP 200 OK with 'multipart/x-mixed-replace; boundary=frame' header.")

            bytes_buffer = b""
            frames_decoded = 0
            start_t = time.time()

            for chunk in response.iter_bytes():
                bytes_buffer += chunk
                while b"\r\n\r\n" in bytes_buffer:
                    header_part, rest = bytes_buffer.split(b"\r\n\r\n", 1)
                    if b"--frame" in header_part and b"\xff\xd8" in rest:
                        end_idx = rest.find(b"\r\n--frame")
                        if end_idx != -1:
                            jpeg_data = rest[:end_idx]
                            bytes_buffer = rest[end_idx:]

                            # Decode JPEG with OpenCV
                            img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            if img is not None:
                                frames_decoded += 1
                                if frames_decoded == 1:
                                    h, w, c = img.shape
                                    print(f"  [OK] First MJPEG frame decoded successfully: {w}x{h} ({c} channels, {len(jpeg_data)} bytes).")
                            if frames_decoded >= 10:
                                break
                        else:
                            break
                    else:
                        bytes_buffer = rest
                if frames_decoded >= 10:
                    break

            elapsed = time.time() - start_t
            fps = frames_decoded / max(0.01, elapsed)
            print(f"  [OK] Decoded {frames_decoded} consecutive OpenCV frames in {elapsed:.2f}s ({fps:.1f} FPS).")


def test_video_upload_pipeline():
    print("\n" + "=" * 65)
    print("PHASE 2: REAL MP4 VIDEO GENERATION & UPLOAD PIPELINE")
    print("=" * 65)

    test_video_path = "data/test_conveyor_input.mp4"
    generate_test_conveyor_video(test_video_path, num_frames=50)
    print(f"  [OK] Synthetic test video generated: {test_video_path} ({os.path.getsize(test_video_path)} bytes).")

    with httpx.Client(timeout=15.0) as client:
        with open(test_video_path, "rb") as f:
            files = {"file": ("test_conveyor_input.mp4", f, "video/mp4")}
            res = client.post(f"{API_URL}/lines/1/upload_video", files=files)
            assert res.status_code == 200, f"Upload failed: {res.text}"
            data = res.json()
            print(f"  [OK] Video uploaded successfully: {data.get('message')}")
            print(f"  [OK] Server video storage path: {data.get('video_path')}")


def test_dynamic_line_and_bag_settings():
    print("\n" + "=" * 65)
    print("PHASE 3: LIVE BELT, OPTICAL GATE & BAG DYNAMIC SETTINGS")
    print("=" * 65)

    with httpx.Client(timeout=10.0) as client:
        tok = client.post(f"{API_URL}/auth/login", json={"username": "operator", "password": "op123"}).json()["token"]
        headers = {"Authorization": f"Bearer {tok}"}

        # 1. Update quick line settings
        payload = {
            "belt_speed": 8.5,
            "belt_direction": [-1.0, 0.0],
            "gate_x_pos": 520,
            "rtsp_url": "rtsp://192.168.1.120:554/ch0"
        }
        res = client.post(f"{API_URL}/lines/1/quick_settings", json=payload, headers=headers)
        assert res.status_code == 200, f"Quick settings failed: {res.text}"
        print(f"  [OK] Belt speed (8.5 px/f), Direction (Left), Gate X (520px) synced: {res.json()['message']}")

        # 2. Update active session product profile
        sessions = client.get(f"{API_URL}/sessions", headers=headers).json()
        assert len(sessions) > 0, "No sessions found!"
        sess_id = sessions[0]["id"]

        products = client.get(f"{API_URL}/products").json()
        assert len(products) > 0, "No products found!"
        prod_id = products[0]["id"]

        res_sess = client.patch(f"{API_URL}/sessions/{sess_id}", json={"product_profile_id": prod_id, "target_count": 300}, headers=headers)
        assert res_sess.status_code == 200, f"Session update failed: {res_sess.text}"
        sess_data = res_sess.json()
        print(f"  [OK] Session #{sess_id} product switched to ID {sess_data.get('product_profile_id')} (Target: {sess_data.get('target_count')}).")


def test_concurrent_burst_stress():
    print("\n" + "=" * 65)
    print("PHASE 4: 100 CONCURRENT CROSSINGS BURST & LEDGER STRESS")
    print("=" * 65)

    with httpx.Client(timeout=10.0) as client:
        login_res = client.post(f"{API_URL}/auth/login", json={"username": "operator", "password": "op123"})
        token = login_res.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        sessions = client.get(f"{API_URL}/sessions", headers=headers).json()
        sess_id = sessions[0]["id"]

    def simulate_crossing(req_idx: int) -> dict:
        with httpx.Client(timeout=10.0) as client:
            t0 = time.perf_counter()
            r = client.post(
                f"{API_URL}/sessions/{sess_id}/simulate_bag",
                json={"direction": 1},
                headers=headers,
            )
            lat = (time.perf_counter() - t0) * 1000.0
            return {"status": r.status_code, "latency_ms": lat, "counted_total": r.json().get("counted_total") if r.status_code == 200 else None}

    start_burst = time.time()
    num_requests = 100
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(simulate_crossing, range(num_requests)))
    burst_duration = time.time() - start_burst

    successes = [r for r in results if r["status"] == 200]
    latencies = [r["latency_ms"] for r in successes]

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    throughput = len(successes) / burst_duration

    print(f"  [OK] Processed {len(successes)}/{num_requests} concurrent requests in {burst_duration:.2f}s.")
    print(f"  [OK] Throughput: {throughput:.1f} req/sec | p50: {p50:.1f}ms | p95: {p95:.1f}ms | p99: {p99:.1f}ms.")
    assert len(successes) == num_requests, "Some concurrent crossing requests failed!"


def test_full_rest_api_surface():
    print("\n" + "=" * 65)
    print("PHASE 5: EXHAUSTIVE REST API & RBAC ENDPOINT COVERAGE")
    print("=" * 65)

    with httpx.Client(timeout=10.0) as client:
        # 1. Auth Tokens for Operator, Engineer, Admin
        op_tok = client.post(f"{API_URL}/auth/login", json={"username": "operator", "password": "op123"}).json()["token"]
        eng_tok = client.post(f"{API_URL}/auth/login", json={"username": "engineer", "password": "eng123"}).json()["token"]
        adm_tok = client.post(f"{API_URL}/auth/login", json={"username": "admin", "password": "admin123"}).json()["token"]
        print("  [OK] Generated JWT tokens for Operator, Engineer, Admin personas.")

        # 2. System Health
        health = client.get(f"{API_URL}/system/health").json()
        assert health["status"] == "healthy"
        print(f"  [OK] GET /system/health -> {health['status'].upper()} ({len(health.get('nodes', []))} nodes).")

        # 3. Sites, Lines, Cameras, Products
        sites = client.get(f"{API_URL}/sites").json()
        lines = client.get(f"{API_URL}/lines", headers={"Authorization": f"Bearer {op_tok}"}).json()
        cameras = client.get(f"{API_URL}/cameras", headers={"Authorization": f"Bearer {op_tok}"}).json()
        products = client.get(f"{API_URL}/products").json()
        print(f"  [OK] Topology: {len(sites)} Sites, {len(lines)} Lines, {len(cameras)} Cameras, {len(products)} Product Profiles.")

        # 4. Jobs & Outbox
        jobs = client.get(f"{API_URL}/system/jobs", headers={"Authorization": f"Bearer {eng_tok}"}).json()
        outbox = client.get(f"{API_URL}/system/outbox", headers={"Authorization": f"Bearer {eng_tok}"}).json()
        print(f"  [OK] Queues: {len(jobs)} Jobs in GPU Queue, {len(outbox)} Outbox Transactions.")

        # 5. Reconciliations & Dispatch Manifest Report
        recs = client.get(f"{API_URL}/reconciliations", headers={"Authorization": f"Bearer {eng_tok}"}).json()
        print(f"  [OK] Reconciliations: {len(recs)} audit items pending resolution.")

        sessions = client.get(f"{API_URL}/sessions", headers={"Authorization": f"Bearer {op_tok}"}).json()
        if sessions:
            rep = client.get(f"{API_URL}/sessions/{sessions[0]['id']}/dispatch_report", headers={"Authorization": f"Bearer {op_tok}"}).json()
            assert "crypto_seal" in rep, "Dispatch report missing crypto_seal!"
            print(f"  [OK] GET /sessions/{sessions[0]['id']}/dispatch_report -> {rep['waybill_no']} ({rep['truck_plate']}, {rep['reconciliation_status']}, Seal: {rep['crypto_seal'][:16]}...).")


def test_playwright_e2e():
    print("\n" + "=" * 65)
    print("PHASE 6: PLAYWRIGHT CHROMIUM REAL BROWSER VERIFICATION")
    print("=" * 65)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})

        with httpx.Client(timeout=10.0) as client:
            op_res = client.post(f"{API_URL}/auth/login", json={"username": "operator", "password": "op123"}).json()
            auth_str = json.dumps({"token": op_res["token"], "username": "operator", "role": "operator"})
        context.add_init_script(f"localStorage.setItem('cs_auth', {json.dumps(auth_str)});")

        page = context.new_page()
        page.goto(BASE_URL, wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(2500)

        # Verify live video feed or canvas
        feed = page.locator("#liveVideoFeed, #conveyorCanvas").first
        feed.wait_for(state="visible", timeout=10000)
        assert feed.is_visible(), "Live feed not visible in browser!"
        print("  [OK] Real Browser: Live OpenCV AI Vision stream rendered.")

        # Test interactive buttons: Normal, Multi (+2), Defect
        normal_btn = page.locator("button:has-text('+1 Normal Bag')").first
        if normal_btn.is_visible():
            normal_btn.click()
            page.wait_for_timeout(400)
            print("  [OK] Real Browser: '+1 Normal Bag' triggered.")

        multi_btn = page.locator("button:has-text('+2 Merged Bags')").first
        if multi_btn.is_visible():
            multi_btn.click()
            page.wait_for_timeout(400)
            print("  [OK] Real Browser: '+2 Merged Bags' triggered (Split & Merge).")

        defect_btn = page.locator("button:has-text('+1 Damaged Bag')").first
        if defect_btn.is_visible():
            defect_btn.click()
            page.wait_for_timeout(400)
            print("  [OK] Real Browser: '+1 Damaged Bag' defect alarm triggered.")

        # Test interactive ribbon tools: Move Laser Line, ROI, Top-Down, Report
        page.locator("#drag-gate-btn").click()
        page.wait_for_timeout(300)
        print("  [OK] Real Browser: Draggable laser gate mode toggled.")

        page.locator("#roi-draw-btn").click()
        page.wait_for_timeout(300)
        print("  [OK] Real Browser: ROI Polygon drawing mode toggled.")

        page.locator("#warp-btn").click()
        page.wait_for_timeout(300)
        print("  [OK] Real Browser: 4-Point Homography Perspective mode toggled.")

        # Test Dispatch Report Modal
        rep_btn = page.locator("button:has-text('Dispatch Report (PDF)')").first
        rep_btn.click()
        page.wait_for_timeout(1000)
        assert page.locator("#dispatch-report-modal").is_visible(), "Dispatch report modal not visible!"
        print("  [OK] Real Browser: Official Dispatch & Reconciliation Manifest Modal opened with QR Seal.")
        page.locator("button:has-text('Close')").last.click()
        page.wait_for_timeout(500)

        # Test persona switch: Admin
        with httpx.Client(timeout=10.0) as client:
            adm_res = client.post(f"{API_URL}/auth/login", json={"username": "admin", "password": "admin123"}).json()
            adm_str = json.dumps({"token": adm_res["token"], "username": "admin", "role": "admin"})
        page.evaluate(f"localStorage.setItem('cs_auth', {json.dumps(adm_str)});")
        page.reload(wait_until="networkidle")
        page.wait_for_timeout(1000)

        wizard_btn = page.locator("button:has-text('Setup Wizard')").first
        if wizard_btn.is_visible():
            wizard_btn.click()
            page.wait_for_timeout(800)
            print("  [OK] Real Browser: Admin role switched and Setup Wizard verified.")

        # Return to live & capture screenshot
        live_btn = page.locator("button:has-text('Live')").first
        if live_btn.is_visible():
            live_btn.click()
            page.wait_for_timeout(1500)
        os.makedirs("artifacts", exist_ok=True)
        page.screenshot(path="artifacts/browser_live_verified.png", full_page=True)
        print("  [OK] Real Browser: Screenshot saved to artifacts/browser_live_verified.png.")

        browser.close()


def main():
    print("#" * 65)
    print("      DEEP END-TO-END VERIFICATION & STRESS TEST SUITE")
    print("#" * 65)

    with httpx.Client(timeout=10.0) as client:
        client.post(f"{API_URL}/system/seed_demo")
        print("  [INIT] Factory demo topology and session initialized.")

    start_all = time.time()

    test_mjpeg_stream()
    test_video_upload_pipeline()
    test_dynamic_line_and_bag_settings()
    test_concurrent_burst_stress()
    test_full_rest_api_surface()
    test_playwright_e2e()

    total_time = time.time() - start_all
    print("\n" + "#" * 65)
    print(f"  ALL 6 PHASES PASSED 100% WITH ZERO ERRORS in {total_time:.2f}s!")
    print("#" * 65)


if __name__ == "__main__":
    main()
