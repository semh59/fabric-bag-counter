# Fabric Bag Counter (v2.0 Enterprise)

High-precision industrial computer vision, conveyor bag counting, amodal instance segmentation, and ERP dispatch reconciliation platform.

---

## 🌟 Key Features

- **RF-DETR Seg AI Vision**: Deep instance segmentation network with multi-head attention and dynamic mask heads for industrial woven and kraft bags.
- **ByteTrack + BeltMotion**: Spatial multi-object tracking with conveyor motion vector compensation and directional gate hysteresis.
- **Two-Stage Gate State Machine**: Eliminates false positives from conveyor oscillations and bag jitter (PRE $\rightarrow$ GATE $\rightarrow$ POST).
- **Secondary Area Integrator**: Redundant cross-verification calculating continuous bag area integral $\int A(t)\,dt$ to flag underfilled or touching bags.
- **Cryptographic Audit Ledger**: Every count event is chained via SHA-256 Merkle tree hashing to generate tamper-evident dispatch manifests with QR seals.
- **SAP S/4HANA OData Integration**: Standardized Goods Movement (`API_MATERIAL_DOCUMENT_SRV`) and Outbound Delivery (`API_OUTBOUND_DELIVERY_SRV`) via Transactional Outbox.
- **Single Page Web Dashboard**: Real-time OpenCV AI Vision HUD, Live MJPEG stream, 60 FPS simulator, and 8 dedicated control-room operational views.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (optional for containerized deployment)

### 1. Local Development Setup

```bash
# Clone and enter directory
git clone https://github.com/semh59/fabric-bag-counter.git
cd fabric-bag-counter

# Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,vision]"

# Start API and Web Dashboard
uvicorn services.api.main:app --host 0.0.0.0 --port 8080 --reload
```

Open your browser at `http://localhost:8080`. Default credentials:
- **Admin**: `admin` / `admin123`
- **Operator**: `operator` / `op123`
- **Engineer**: `engineer` / `eng123`

---

## 🐳 Docker Deployment

### Server Stack (Database, API, Web UI, Background Workers)
```bash
docker compose up -d --build
```

### Edge Vision Worker (On Factory Machine with Cameras/GPU)
```bash
docker compose -f docker-compose.edge.yml up -d --build
```

---

## 🧪 Testing & Verification

Run the full automated test suite:
```bash
python -m pytest -q
```

Run official production acceptance test:
```bash
python tools/run_prod_test_execution.py
```

Run license policy compliance check:
```bash
python tools/license_check.py
```

---

## 📜 License

Distributed under the MIT License. See `THIRD_PARTY_NOTICES.md` for third-party component licenses.
