# Architecture and Intellectual Property Provenance (§2, §12)

## 1. Clean-Room Architecture
`fabric-bag-counter` is an enterprise conveyor vision and count reconciliation platform engineered from first principles without AGPL or copyleft code contamination.

### Core Modules:
- **`packages/cs_core/`**: Abstract interfaces, geometry primitives, and shared memory IPC transport.
- **`packages/cs_vision/`**: RF-DETR Seg deep instance segmentation network with multi-head attention and dynamic mask heads.
- **`packages/cs_tracking/`**: ByteTrack multi-object tracker integrated with `BeltMotionModel` and directional gate hysteresis.
- **`packages/cs_counting/`**: Two-stage state machine (PRE/GATE/POST), area integral cross-verification, and count ledger.
- **`packages/cs_storage/`**: PostgreSQL / SQLite ORM, migration scripts, transactional outbox, and Merkle tree dispatch seal.
- **`drivers/`**: Modular I/O, RTSP video streams, USB relays, and SAP S/4HANA OData adapters.

---

## 2. Model & Algorithm Provenance

1. **RF-DETR Seg Instance Segmentation**:
   - Neural network architecture built in PyTorch (`packages/cs_vision/train_rfdetr.py`).
   - Multi-scale feature extractor with GroupNorm, cross-attention queries, and anchor-guided spatial heads.
   - Exported to standard ONNX format for hardware-accelerated CPU/GPU execution via ONNX Runtime.

2. **ByteTrack + BeltMotion Tracker**:
   - Multi-stage association algorithm based on clean-room ByteTrack principles.
   - Incorporates known constant conveyor velocity $v_x, v_y$ into Kalman filter process noise covariance.

3. **Cryptographic Ledger & Merkle Root**:
   - Tamper-evident counting event chain: each count event computes $H_n = \text{SHA256}(H_{n-1} \parallel \text{EventData})$.
   - Final dispatch manifest generated with 64-character SHA-256 Merkle root seal.

---

## 3. SAP ERP Integration Standards

- Implements standard SAP S/4HANA OData services:
  - `API_MATERIAL_DOCUMENT_SRV` for Goods Movement / Material Postings (MIGO 601/101/201).
  - `API_OUTBOUND_DELIVERY_SRV` for Outbound Delivery Confirmation (VL02N picking/packing).
- Implements Transactional Outbox Pattern for guaranteed at-least-once delivery with idempotency keys.
