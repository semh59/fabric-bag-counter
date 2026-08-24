"""License scanner CI gate (§2, §12.3)."""

from __future__ import annotations

import os
import sys
import importlib.metadata

ALLOWED_LICENSES = {
    "mit",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "apache 2.0",
    "apache-2.0",
    "isc",
    "postgresql",
    "python software foundation",
    "psf",
    "unlicense",
    "cc0-1.0",
    "zlib",
}

DISALLOWED_PATTERNS = [
    "agpl",
    "gpl",
    "mpl",
    "sspl",
    "busl",
    "pml",
    "elastic",
    "cc-by-nc",
    "proprietary",
]

BANNED_EXACT_PACKAGES = [
    "ultralytics",
    "boxmot",
    "yolo_tracking",
    "minio",
]


def check_licenses() -> bool:
    print("[License Check] Validating project license compliance...")
    
    # 1. Check mandatory provenance documents
    required_docs = ["THIRD_PARTY_NOTICES.md", "docs/provenance.md"]
    for doc in required_docs:
        if not os.path.exists(doc) or os.path.getsize(doc) < 50:
            print(f"[ERROR] Required license document '{doc}' is missing or empty!")
            return False

    # 2. Check installed distributions
    violations = []
    dists = list(importlib.metadata.distributions())
    for dist in dists:
        pkg_name = dist.metadata["Name"].lower()
        
        # Check banned packages
        if pkg_name in BANNED_EXACT_PACKAGES:
            violations.append(f"Forbidden package '{pkg_name}' is installed (§2.3)!")

        # Check license classifiers or license short string
        classifiers = [c.lower() for c in (dist.metadata.get_all("Classifier") or []) if "license ::" in c.lower()]
        license_header = (dist.metadata.get("License") or "").split("\n")[0].strip().lower()
        license_expr = (dist.metadata.get("License-Expression") or "").strip().lower()

        license_identifiers = " ".join(classifiers + [license_header, license_expr])

        # Check for disallowed patterns in license identifiers
        for bad in DISALLOWED_PATTERNS:
            if bad in license_identifiers:
                # LGPL exception check for dynamic AV/ffmpeg (§2.4)
                if "lgpl" in bad and pkg_name in ["av", "pyav"]:
                    continue
                # MPL-2.0 documented data-only root CA bundle exception (§2.1)
                if "mpl" in bad and pkg_name in ["certifi"]:
                    continue
                violations.append(f"Package '{pkg_name}' has prohibited license '{license_identifiers}' (pattern: {bad})")

    if violations:
        print("[FAIL] License violations found:")
        for v in violations:
            print(f"  - {v}")
        return False

    print("[PASS] All installed dependencies and documentation conform to license policy (§2).")
    return True


def main() -> None:
    success = check_licenses()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
