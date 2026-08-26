"""Build script for the Fabric Bag Counter web dashboard.

Compiles the actual Tailwind CSS utility classes used in index.html into a
local stylesheet (no CDN/runtime JIT dependency, so the dashboard works on an
offline factory network), then assembles the dist/ bundle served by FastAPI's
StaticFiles mount (see services/api/main.py).

Downloads the official standalone Tailwind CLI binary (no Node.js/npm
required) into web/.bin/ on first run and reuses it afterwards.
"""

from __future__ import annotations

import platform
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
BIN_DIR = WEB_DIR / ".bin"
TAILWIND_VERSION = "v4.3.3"


def _tailwind_asset_name() -> str:
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Windows":
        return "tailwindcss-windows-x64.exe"
    if system == "Darwin":
        return "tailwindcss-macos-arm64" if machine in ("arm64", "aarch64") else "tailwindcss-macos-x64"
    if system == "Linux":
        return "tailwindcss-linux-arm64" if machine in ("arm64", "aarch64") else "tailwindcss-linux-x64"
    raise RuntimeError(f"Unsupported platform for Tailwind standalone CLI: {system}")


def _ensure_tailwind_cli() -> Path:
    BIN_DIR.mkdir(exist_ok=True)
    asset_name = _tailwind_asset_name()
    dest = BIN_DIR / asset_name

    if not dest.exists():
        url = (
            "https://github.com/tailwindlabs/tailwindcss/releases/download/"
            f"{TAILWIND_VERSION}/{asset_name}"
        )
        print(f"[build] Downloading Tailwind CLI {TAILWIND_VERSION} ({asset_name})...")
        urllib.request.urlretrieve(url, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return dest


def build() -> None:
    tailwind_cli = _ensure_tailwind_cli()

    dist_dir = WEB_DIR / "dist"
    dist_dir.mkdir(exist_ok=True)

    dist_css = dist_dir / "tailwind.css"
    print("[build] Compiling Tailwind CSS from index.html usage...")
    subprocess.run(
        [
            str(tailwind_cli),
            "-i", str(WEB_DIR / "src" / "tailwind.input.css"),
            "-o", str(dist_css),
            "--minify",
            "--cwd", str(WEB_DIR),
        ],
        check=True,
    )

    # Single compiled stylesheet, copied (not recompiled) to the source
    # directory so index.html also renders correctly when opened directly.
    shutil.copy2(dist_css, WEB_DIR / "tailwind.css")

    shutil.copy2(WEB_DIR / "index.html", dist_dir / "index.html")

    print(f"[build] OK: {dist_css.relative_to(WEB_DIR.parent)} "
          f"({dist_css.stat().st_size} bytes) and dist/index.html are up to date.")


if __name__ == "__main__":
    try:
        build()
    except subprocess.CalledProcessError as exc:
        print(f"[build] Tailwind CLI failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(1)
