# Fabric Bag Counter & Dispatch Reconciliation Web Dashboard (v2.0)

Industrial single-page application (SPA) dashboard for real-time fabric conveyor bag counting, defect detection, and ERP dispatch reconciliation.

## Architecture
- **Single Page Application (SPA)**: Pure HTML5, Tailwind CSS, Vanilla Modern ES6+ JavaScript.
- **Real-Time Video Stream**: Connects to FastAPI OpenCV MJPEG endpoint (`/api/v1/stream/video_feed`).
- **Live Counter & Telemetry**: Subscribes to real-time WebSocket (`/ws`) for instant batch count updates, line speed, and alert broadcasts.
- **REST Endpoints**:
  - `GET /api/v1/dispatch/orders` — Active dispatch orders and progress
  - `POST /api/v1/dispatch/orders` — Create new dispatch order
  - `POST /api/v1/counter/manual-adjust` — Manual +1 / -1 / tare adjustment
  - `POST /api/v1/auth/login` — Cryptographic JWT authentication
  - `GET /api/v1/audit/logs` — Real-time tamper-evident audit trail

## Files
- `index.html`: Main dashboard source
- `dist/index.html`, `dist/tailwind.css`: Production bundle served by FastAPI's static mount (`services/api/main.py`)
- `src/tailwind.input.css`: Tailwind entry point (scans `index.html` for utility classes actually used)
- `build.py`: Build script — compiles `tailwind.css` and refreshes `dist/`
- `package.json`: Project manifest

## Build

The dashboard uses Tailwind CSS compiled to a local stylesheet instead of the
`cdn.tailwindcss.com` runtime JIT script, so it works on an offline factory
network. Building requires only Python (no Node.js/npm):

```
python build.py
```

This downloads the official standalone Tailwind CLI binary into `web/.bin/`
(gitignored, cached after the first run) and regenerates `dist/tailwind.css`
and `dist/index.html` from the current source files.
