# THIRD-PARTY SOFTWARE NOTICES AND INFORMATION

This project incorporates components from the following open source software:

---

### 1. PyAV / FFmpeg
- **License**: BSD-3-Clause (PyAV) / LGPL-2.1-or-later (FFmpeg runtime)
- **Notice**: Dynamic linking only via official wheels. Build strictly excludes `--enable-gpl`.
- **Usage**: Hardware-accelerated and CPU video stream decoding in ingest and dataset extraction.

---

### 2. RF-DETR (Instance Segmentation)
- **License**: Apache-2.0
- **Source**: FoundationVision / Roboflow RF-DETR
- **Notice**: Pretrained weights and model architecture for Nano/Small/Medium configurations under Apache-2.0 license. PML-licensed variants (RFDETR-XL, RFDETR-2XL) are strictly prohibited.

---

### 3. ByteTrack
- **License**: MIT
- **Origin**: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
- **Reference**: https://github.com/FoundationVision/ByteTrack
- **Modifications**: Re-engineered with custom `BeltMotionModel` Kalman prior and exact Polygon Mask-IoU cost matrix replacing bounding box IoU.

---

### 4. Supervision
- **License**: MIT
- **Reference**: https://github.com/roboflow/supervision
- **Usage**: Polygon, mask manipulation, and visualization helpers.

---

### 5. ONNX Runtime & CUDA
- **License**: MIT (ONNX Runtime) / NVIDIA CUDA EULA (CUDA Runtime Libraries)
- **Notice**: Standard CUDA Execution Provider dynamically loaded for GPU inference.

---

### 6. FilterPy & SciPy
- **License**: MIT (FilterPy) / BSD-3-Clause (SciPy)
- **Usage**: Kalman filtering and Hungarian linear sum assignment (`scipy.optimize.linear_sum_assignment`).

---

### 7. FastAPI, Pydantic, SQLAlchemy, Alembic
- **License**: MIT
- **Usage**: Web API, validation, async database ORM, and schema migrations.

---

### 8. PostgreSQL & Drivers
- **License**: PostgreSQL License / MIT (asyncpg, psycopg2-binary)
- **Usage**: Persistent database storage, persistent sequence epochs, transactional outbox, and row-level locking.

---

### 9. Certifi (MPL-2.0 Case-by-Case Evaluated Data Exception)
- **License**: Mozilla Public License 2.0 (MPL-2.0)
- **Notice**: Evaluated per §2.1. Strictly contains public X.509 SSL CA root certificate authority PEM data for secure TLS communication without proprietary code or executable linkage.
