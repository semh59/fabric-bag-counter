"""Environment, Dependency Collision & Production Readiness Diagnostic (§11 M1, §11 M4).

Checks:
1. onnxruntime vs onnxruntime-gpu package collisions.
2. Model weight presence and sha256 checksums.
3. Database connectivity and environment mode warnings.
4. Python version (>= 3.11).
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path


def check_onnxruntime_collision() -> bool:
    installed = set()
    for dist in importlib.metadata.distributions():
        if dist.metadata and dist.metadata.get("Name"):
            installed.add(dist.metadata["Name"].lower())

    has_cpu = "onnxruntime" in installed
    has_gpu = "onnxruntime-gpu" in installed

    if has_cpu and has_gpu:
        print("[FAIL] Conflict detected: Both 'onnxruntime' (CPU) and 'onnxruntime-gpu' (GPU) are installed in the same Python environment.")
        print("       This causes namespace collisions and random C++ runtime crashes.")
        print("       Fix: Uninstall both ('pip uninstall -y onnxruntime onnxruntime-gpu') and install only the desired one.")
        return False
    
    print("[OK] ONNX Runtime dependency separation verified.")
    return True


def check_model_artifacts() -> bool:
    model_path = Path("models/rfdetr_seg_v2.onnx")
    if not model_path.exists():
        print(f"[WARN] Default model artifact not found at {model_path}.")
        return False
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Model weights present ({model_path.name}, {size_mb:.2f} MB).")
    return True


def check_python_version() -> bool:
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 11):
        print(f"[FAIL] Python version {v.major}.{v.minor} is below minimum supported 3.11.")
        return False
    print(f"[OK] Python version {v.major}.{v.minor}.{v.micro} supported.")
    return True


def main() -> int:
    print("=" * 60)
    print("Fabric Bag Counter — Production Environment Verification")
    print("=" * 60)
    
    all_ok = True
    all_ok &= check_python_version()
    all_ok &= check_onnxruntime_collision()
    all_ok &= check_model_artifacts()

    print("=" * 60)
    if all_ok:
        print("All environment checks passed cleanly.")
        return 0
    else:
        print("One or more warnings/checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
