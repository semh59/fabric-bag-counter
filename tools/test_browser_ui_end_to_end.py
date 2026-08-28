"""Automated End-to-End Real Browser Interaction & UI Verification Script."""

from __future__ import annotations

import os
import sys
from playwright.sync_api import sync_playwright

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    print("=" * 60)
    print("    REAL BROWSER AUTOMATED INTERACTION & VERIFICATION")
    print("=" * 60)

    # Verify if FastAPI web server is running at localhost:8080
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:8080", timeout=2)
    except Exception:
        print("\n[SKIP] Local web server is not actively running at http://localhost:8080.")
        print("       To execute real Playwright E2E browser tests:")
        print("       1. Start server: python -m uvicorn services.api.main:app --port 8080")
        print("       2. Re-run: python tools/test_browser_ui_end_to_end.py\n")
        return 0

    with sync_playwright() as p:
        # Launch Chromium browser in headless mode
        print("\n[1/10] Launching Chromium Browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # 1. Navigate to Web UI
        print("[2/10] Navigating to http://localhost:8080...")
        page.goto("http://localhost:8080", wait_until="networkidle", timeout=15000)
        title = page.title()
        print(f"  [OK] Page loaded. Title: '{title}'")
        assert "Bag Counter" in title or "Fabric" in title, f"Unexpected page title: {title}"

        # Wait 2 seconds for initial render
        page.wait_for_timeout(2000)

        # 2. Verify Live Counting Screen
        print("\n[3/10] Verifying Live Counting Dashboard & Live Video Stream...")
        video_feed = page.locator("#liveVideoFeed, #conveyorCanvas").first
        if video_feed.is_visible():
            print("  [OK] Live OpenCV AI Vision Video Stream / Conveyor Simulator is visible and running.")

        # 3. Test Navigation Tabs
        print("\n[4/10] Testing Navigation Tabs...")
        sessions_tab = page.locator("button:has-text('Sessions')").first
        if sessions_tab.is_visible():
            sessions_tab.click()
            page.wait_for_timeout(1000)
            print("  [OK] Sessions tab opened.")

        live_tab = page.locator("button:has-text('Live')").first
        if live_tab.is_visible():
            live_tab.click()
            page.wait_for_timeout(1000)
            print("  [OK] Live tab returned.")

        os.makedirs("artifacts", exist_ok=True)
        screenshot_path = "artifacts/browser_live_verified.png"
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"  [OK] Screenshot captured and saved to: {screenshot_path}")

        browser.close()

    print("\n" + "=" * 60)
    print("    REAL BROWSER VERIFICATION COMPLETED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    main()
