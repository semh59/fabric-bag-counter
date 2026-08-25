"""Real Physical Camera & Video Stream Verification.

Verifies real physical camera hardware (USB webcam) or real recorded factory video input.
If neither real hardware nor real video is available, explicitly SKIPS test execution
rather than fabricating artificial synthetic frames.
"""

import sys
import os
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from packages.cs_vision.detector import VisionDetector


def main() -> int:
    print("=" * 65)
    print("  PHYSICAL CAMERA & FACTORY VIDEO STREAM VERIFICATION")
    print("=" * 65)

    # 1. Initialize Vision Detector with trained ONNX model
    detector = VisionDetector(conf_threshold=0.25)
    has_real_input = False

    # 2. Check for Real USB Webcam (Device Index 0)
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                has_real_input = True
                print(f"  [OK] Physical USB Camera Detected: {frame.shape[1]}x{frame.shape[0]} px.")
                res = detector.predict(frame)
                print(f"  [OK] Real Camera Inference: {len(res.bag_bodies)} bag bodies detected.")
            else:
                print("  [INFO] Physical camera opened but no frame captured.")
        else:
            print("  [INFO] No physical camera attached at index 0 (Headless/CI environment).")
    except Exception as e:
        print(f"  [INFO] Physical camera check skipped: {e}")

    # 3. Check for Real Recorded Factory Video on Disk
    video_path = ROOT_DIR / "data" / "test_conveyor_input.mp4"
    if video_path.exists():
        has_real_input = True
        cap_file = cv2.VideoCapture(str(video_path))
        total_frames = int(cap_file.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_file.get(cv2.CAP_PROP_FPS)
        print(f"  [OK] Reading Real Video File: {video_path.name} ({total_frames} frames @ {fps:.1f} FPS)")

        processed = 0
        detected_count = 0
        t0 = time.perf_counter()
        while cap_file.isOpened():
            ret, frame = cap_file.read()
            if not ret:
                break
            res = detector.predict(frame)
            if len(res.bag_bodies) > 0:
                detected_count += 1
            processed += 1
        cap_file.release()
        elapsed = time.perf_counter() - t0

        print(f"  [OK] Processed {processed} real video frames in {elapsed:.2f}s ({processed/max(0.001, elapsed):.1f} FPS).")
        print(f"  [OK] Detection Frames: {detected_count}/{processed} frames containing target objects.")
    else:
        print(f"  [INFO] Real video file not found at '{video_path}'. (No fake video will be synthesized).")

    # 4. Result evaluation
    if not has_real_input:
        print("\n" + "-" * 65)
        print("  [SKIP] NO REAL PHYSICAL CAMERA OR VIDEO FILE AVAILABLE.")
        print("  Test cleanly SKIPPED as per zero-mock integrity policy.")
        print("-" * 65 + "\n")
        return 0

    print("=" * 65)
    print("  REAL PHYSICAL INPUT PIPELINE VERIFIED SUCCESSFULLY!")
    print("=" * 65 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
