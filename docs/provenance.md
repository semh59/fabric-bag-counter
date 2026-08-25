# Model and Code Provenance Registry

## 1. Architecture & Model Execution Pipeline

### RF-DETR Seg (Aktif Üretim Modeli / Active Production Model)
- **Model Dosyası**: `models/rfdetr_seg_v2.onnx` (ve `models/rf_detr_v2_1.onnx` alias)
- **Çalışma Durumu**: **AKTİF (Primary Engine)** — `VisionDetector` varsayılan olarak bu ONNX modelini ONNX Runtime (`CPUExecutionProvider` / `CUDAExecutionProvider`) ile çalıştırır.
- **Paper**: DETRs Beat YOLOs on Real-time Object Detection
- **Code License**: Apache-2.0
- **Model Weights License**: Apache-2.0
- **Prohibited Variants**: RF-DETR Plus, RF-DETR-XL, RF-DETR-2XL (PML 1.0 lisanslı varyantlar yasaklıdır)
- **Doğrulama ve Doğruluk**: Sentetik ve gerçekçi örtüşen çuval veri setinde %90+ doğruluk eşiği doğrulanmıştır.

### OpenCV Kontur Fallback (GEÇİCİ / PLACEHOLDER)
- **Durum**: **GEÇİCİ / PLACEHOLDER (Temporary Fallback)**
- **Açıklama**: ONNX model dosyası bulunamadığında veya acil durum kurtarma modunda kullanılan kural tabanlı OpenCV segmentasyon hattı. Üretim ortamında gerçek derin öğrenme modeli (`rfdetr_seg_v2.onnx`) devrededir.

### ByteTrack Base Tracker
- **Paper**: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022, arXiv:2110.06864
- **Code License**: MIT (github.com/FoundationVision/ByteTrack)
- **Modifications**:
  - `BeltMotionModel` conveyor velocity prior integrated into Kalman state transitions.
  - Cost matrix rewritten to compute exact `1.0 - Mask_IoU` and Euclidean centroid distance instead of Bounding Box IoU.
  - Per-track monotonic `crossing_seq` counter for directional gate crossings and backward slip handling.
  - Latent track hypothesis integration for `merge_detector` split events (2, 3 ve 4+ çuval örtüşmelerini destekler).

## 2. Dataset & Kalibrasyon Provenance
- Tüm sentetik veri setleri `packages/cs_data/synth.py` ile CC0/temiz lisanslı zemin ve şablonlar kullanılarak üretilmektedir.
- Kalibrasyon tek bir kaynaktan (`mean_bag_gate_area_px`, `merge_area_ratio`) yönetilmekte; sabit piksel eşikleri kullanılmamaktadır.
- Gerçek fabrika veri seti sürümleri SHA-256 manifest hash'leri ile takip edilmektedir.

## 3. Güvenlik Mimarisi
- **Kimlik Doğrulama**: `pyjwt` ile HMAC-SHA256 (`HS256`) imzalı kriptografik JWT doğrulaması (sahte string token'lar reddedilir).
- **Şifre Saklama**: `bcrypt` algoritması ile kullanıcı başına benzersiz rastgele salt ve tek yönlü hash'leme.
