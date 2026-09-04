# Third-Party Software Notices and Licenses

This project (`fabric-bag-counter`) utilizes open-source components governed by permissive licenses (MIT, Apache 2.0, BSD, ISC, PSF). In compliance with commercial licensing guidelines, all AGPL-licensed libraries (e.g., Ultralytics YOLO, BoxMOT) are strictly prohibited and omitted.

---

## 1. Direct Dependencies and Licenses

| Package | License | Usage / Purpose |
|---|---|---|
| **FastAPI** | MIT | High-performance asynchronous REST & SSE API framework |
| **Starlette** | BSD-3-Clause | ASGI application toolkit and request routing |
| **Uvicorn** | BSD-3-Clause | Production ASGI web server |
| **Pydantic** | MIT | Data validation and settings management |
| **SQLAlchemy** | MIT | SQL toolkit and Object Relational Mapper |
| **Alembic** | MIT | Database schema migration framework |
| **NumPy** | BSD-3-Clause | Numerical arrays, matrix operations, and geometry calculations |
| **OpenCV (opencv-python-headless)** | Apache-2.0 | Computer vision image processing, CLAHE, perspective warp |
| **ONNX Runtime (onnxruntime)** | MIT | Cross-platform high-performance neural network inference |
| **PyTorch (torch)** | BSD-3-Clause | Deep learning model definition, training, and ONNX export |
| **PyAV (av)** | LGPL-2.1+ | Dynamic binding to FFmpeg shared libraries for RTSP/video decoding |
| **psycopg2-binary** | LGPL-3.0+ | PostgreSQL database adapter with dynamic C runtime binding |
| **asyncpg** | Apache-2.0 | High-performance asynchronous PostgreSQL client library |
| **PyJWT** | MIT | Cryptographic JSON Web Token encoding and verification |
| **bcrypt** | Apache-2.0 | Password hashing with cryptographic salt |
| **HTTPX** | BSD-3-Clause | Asynchronous and synchronous HTTP/OData client |
| **filterpy** | MIT | Kalman filtering for spatial state tracking |
| **asyncua** | LGPL-3.0+ | Industrial OPC-UA SCADA Server driver |
| **paho-mqtt** | EPL-2.0 / BSD-3-Clause | MQTT Industry 4.0 telemetry publisher |

---

## 2. LGPL Dynamic Linking Compliance

- **PyAV & FFmpeg**: PyAV dynamically links against unmodified LGPL-licensed FFmpeg shared libraries. No static linking is performed.
- **psycopg2**: Binary distribution connects dynamically to PostgreSQL `libpq` client libraries.
- **asyncua**: Pure-Python library used dynamically as an independent network SCADA driver.

---

## 3. ByteTrack Tracking Provenance

The multi-object tracking implementation in `packages/cs_tracking/` is a clean-room implementation of the ByteTrack algorithm (Zhang et al., ECCV 2022) with industrial conveyor extensions:
- Augmented with `BeltMotionModel` Kalman filter state transition matrix.
- Integrated with Directional Gate Hysteresis and Cryptographic Count Ledger.
- Free of AGPL or proprietary code.
