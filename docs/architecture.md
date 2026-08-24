# Sistem Mimarisi ve Veri Akışı

## 1. Process ve Servis Mimarisi

```
                                      N × Kamera (RTSP)
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Edge Node                                                                        │
│                                                                                  │
│   Supervisor  ───►  Ingest Worker (Kamera 1)  ──┐                                │
│                     Ingest Worker (Kamera 2)  ──┼──►  Shared Memory Ring Buffer  │
│                     ...                         │              │                 │
│                     Ingest Worker (Kamera N)  ──┘              ▼                 │
│                                                          Inference Worker        │
│                                                          (Tek GPU Batching)      │
│                                                                │                 │
└────────────────────────────────────────────────────────────────┼─────────────────┘
                                                                 │
                                                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Core Node                                                                        │
│                                                                                  │
│   PostgreSQL 16 (Sayım Defteri / Ledger, Oturum, Kalibrasyon, Job Kuyruğu, vb.)  │
│          ▲                              ▲                            ▲           │
│          │                              │                            │           │
│      Jobrunner                      FastAPI                      ERP Relay       │
│   (Model, Sentetik,              (Canlı SSE,                  (Transactional     │
│   Kalibrasyon, Madencilik)      Arayüz, Kontrol)                  Outbox)        │
│                                         ▲                            │           │
│                                         │                            ▼           │
│                                   React Arayüzü                  Müşteri ERP     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Doğruluk ve Güvenilirlik İlkeleri
1. **Tek Doğruluk Kaynağı**: `count_event` tablosudur. `session.counted_total` bu tablodan türetilir.
2. **Kalıcı Stream Epoch**: `camera_epoch` tablosunda saklanan monoton artan sayaç. Yeniden başlatmalarda çakışmayı önler.
3. **`crossing_seq` Sayacı**: Track başına her gate geçişinde artırılır; geri kayıp tekrar geçen çuvalın çift veya eksik sayılmasını engeller.
4. **Çift Kalibrasyon**: Aşama 1 `motion` (Lucas-Kanade optik akış, modelsiz) ve Aşama 2 `scale` (model sonrası seyrek video üzerinde alan medyanı ile `px_per_mm`).
5. **Kare Düşürme Güvenliği**: Ardışık düşen kare sayısı `max_consecutive_drops` sınırını aşarsa oturum otomatik olarak `degraded` durumuna geçer ve mutabakat gerektirir.
6. **Alan İntegrali Kontrolü**: Takipten bağımsız ikinci tahmin `(toplam_maske_alanı / ortalama_alan)`. Ayrışma eşiği aşılırsa `discrepancy_flag` açılır.
7. **İdempotent Outbox**: ERP'ye gönderim outbox modeli ve adaptör durum sorgulamasıyla sağlanır; cevapsız CSV aktarımları insan onaylı mutabakata yönlendirilir.
