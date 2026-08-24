"""Hardcode scanner CI gate (§1, §12.3)."""

from __future__ import annotations

import os
import re
import sys

FORBIDDEN_PATTERNS = [
    (r"\bNUM_CAMERAS\s*=\s*\d+", "Hardcoded camera count forbidden (§1). Must come from camera table."),
    (r"\b(Hikvision|Dahua|Basler|FLIR|AxisCamera|Axis-Camera)\b", "Hardcoded camera brand forbidden in core (§1)."),
    (r"\bDEFAULT_BELT_SPEED\s*=\s*[0-9\.]+", "Fixed belt speed forbidden (§1). Must be calibrated via line_calibration."),
    (r"\bFIXED_PX_PER_MM\s*=\s*[0-9\.]+", "Fixed px-per-mm scale forbidden (§1). Must be calibrated via line_calibration."),
    (r"\bDEFAULT_MATERIAL_CODE\s*=\s*[\"'][^\"']+[\"']", "Fixed ERP material code forbidden (§1). Must come from product_profile."),
]

CORE_DIRS = ["packages/cs_core", "packages/cs_vision", "packages/cs_tracking", "packages/cs_counting", "packages/cs_storage"]


def scan_core_code() -> bool:
    print("[Hardcode Check] Scanning core packages for forbidden constants (§1)...")
    violations = []

    for cdir in CORE_DIRS:
        if not os.path.exists(cdir):
            continue
        for root, _, files in os.walk(cdir):
            for file in files:
                if file.endswith(".py"):
                    fpath = os.path.join(root, file)
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, start=1):
                            # Skip comments
                            stripped = line.strip()
                            if stripped.startswith("#"):
                                continue
                            for pattern, desc in FORBIDDEN_PATTERNS:
                                if re.search(pattern, line, re.IGNORECASE):
                                    violations.append(f"{fpath}:{line_num} - {desc}\n    Line: {stripped}")

    if violations:
        print("[FAIL] Generalization rule violations found:")
        for v in violations:
            print(f"  - {v}")
        return False

    print("[PASS] No forbidden hardcoded constants found across core codebase (§1).")
    return True


def main() -> None:
    success = scan_core_code()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
