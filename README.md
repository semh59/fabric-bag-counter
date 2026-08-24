# Çuval Sayım Sistemi (Bag Counting System) v2

Yerel (On-Premise) Konveyör Çuval Sayım ve Sevkiyat Mutabakat Sistemi.

## Mimari ve Bileşenler
- **`packages/cs_core`**: Çekirdek veri sınıfları (`Frame`), domain modelleri, geometri yardımcıları ve 5 sürücü protokolü.
- **`packages/cs_storage`**: PostgreSQL / SQLAlchemy 2.0 modelleri, repository katmanı, kalıcı stream epoch ve Alembic şeması.
- **`packages/cs_vision`**: RF-DETR Seg ONNX sarmalayıcı, ön ve son işleme, maske ayrıştırma.
- **`packages/cs_tracking`**: Özgün `BeltMotionModel`, Kalman filtresi, ByteTrack maske IoU ilişkilendirmesi, `merge_detector` ve `crossing_seq`.
- **`packages/cs_counting`**: `PRE -> GATE -> POST` durum makinesi, alan-integrali sayacı ve tutarsızlık tespiti.
- **`packages/cs_data`**: PyAV kare çıkarma, SSIM eleme, amodal sentetik veri üretimi, `hard_holdout` veri seti bölme, CVAT entegrasyonu, `mine_hard_frames` aktif öğrenme.
- **`packages/cs_eval`**: Replay harness, metrikler, skor tablosu üretici ve model karşılaştırıcı.
- **`services/`**: Supervisor, Ingest (kamera başına), Inference (düğüm başına tek GPU), API (FastAPI), Jobrunner (kira ve kalp atışı), ERP Relay (transactional outbox).
- **`drivers/`**: RTSP, Dosya, CSV, SAP OData, Noop/USB Röle, Operatör/ERP kimlik sürücüleri.
- **`web/`**: Operatör ve Mühendis/Admin persona ayrımına sahip modern web arayüzü ve Canvas ROI/Gate çizim editörü.

## Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleme
pip install -e ".[dev,vision]"

# Testleri çalıştırma
pytest -v

# Lisans ve kod kurallarını denetleme
python tools/license_check.py
python tools/hardcode_check.py

# Docker ile başlatma
docker compose up --build
```
