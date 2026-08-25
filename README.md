# Çuval Sayım Sistemi (Bag Counting System) v2

Yerel (On-Premise) Endüstriyel Konveyör Çuval Sayım, Örtüşme Ayrıştırma ve Sevkiyat Mutabakat Sistemi.

## Mimari ve Bileşenler
- **`packages/cs_core`**: Çekirdek veri sınıfları (`Frame`), domain modelleri, geometri yardımcıları ve 5 sürücü protokolü.
- **`packages/cs_storage`**: PostgreSQL / SQLAlchemy 2.0 modelleri, repository katmanı, kalıcı stream epoch ve Alembic şeması.
- **`packages/cs_vision`**: 
  - **Aktif Model (Primary)**: RF-DETR Seg ONNX Runtime modeli (`models/rfdetr_seg_v2.onnx`) ile nesne tespiti, örnek segmentasyonu ve baskı izi ayrıştırma.
  - **Geçici Fallback (Placeholder)**: Model dosyası eksik olduğunda veya kurtarma durumunda devreye giren OpenCV kontur tabanlı geçici yedekleme hattı.
- **`packages/cs_tracking`**: Özgün `BeltMotionModel`, Kalman filtresi, ByteTrack maske IoU ilişkilendirmesi, kalibrasyon tabanlı `merge_detector` (2, 3 ve 4+ çuval örtüşme desteği) ve `crossing_seq`.
- **`packages/cs_counting`**: `PRE -> GATE -> POST` durum makinesi, alan-integrali sayacı ve tutarsızlık tespiti.
- **`packages/cs_data`**: PyAV kare çıkarma, SSIM eleme, amodal sentetik veri üretimi, `hard_holdout` veri seti bölme, CVAT entegrasyonu, `mine_hard_frames` aktif öğrenme.
- **`packages/cs_eval`**: Replay harness, metrikler, skor tablosu üretici ve model karşılaştırıcı.
- **`services/`**: 
  - `Supervisor`, `Ingest` (kamera başına), `Inference` (tek GPU batching), `Jobrunner` (kira ve kalp atışı), `ERP Relay` (transactional outbox).
  - `API`: FastAPI tabanlı REST/SSE sunucu, kriptografik imzalı JWT kimlik doğrulama (`pyjwt`), `bcrypt` parola güvenliği ve RBAC (`admin`, `engineer`, `operator`).
- **`drivers/`**: RTSP, Dosya, CSV, SAP OData, Noop/USB Röle, Operatör/ERP kimlik sürücüleri.
- **`web/`**: `web/dist/index.html` üzerinde derlenmiş Single-Page Application (SPA) arayüzü; FastAPI tarafından doğrudan sunulur, Operatör ve Mühendis/Admin persona ayrımına ve Canvas ROI/Gate çizim editörüne sahiptir.
- **`models/`**: Eğitilmiş `rfdetr_seg_v2.onnx` ve `rf_detr_v2_1.onnx` ONNX model dosyaları.

## Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleme
pip install -e ".[dev,vision]"

# Model dosyasını eğitme ve dışa aktarma (gerektiğinde)
python packages/cs_vision/train_rfdetr.py

# Testleri çalıştırma
pytest -v

# Lisans ve kod kurallarını denetleme
python tools/license_check.py
python tools/hardcode_check.py

# Fiziksel kamera / gerçek video doğrulama testi
python tools/test_real_physical_camera.py

# Docker ile başlatma
docker compose up --build
```
