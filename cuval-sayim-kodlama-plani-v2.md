# Çuval Sayım Sistemi — Kodlama Planı (v2)

> **Kapsam:** Yalnızca yazılım. Elektrik, otomasyon, aydınlatma, iş güvenliği, KVKK ve
> saha montajı ayrı bir dokümanın konusudur.
>
> **Hedef okuyucu:** Bu projeyi hiç bilmeyen bir geliştirici veya kodlama asistanı.
> Belirsiz bırakılmış hiçbir karar yoktur. Bir şey tanımlı değilse, tanımlanana kadar
> kodlanmaz — varsayım yapılmaz.
>
> **v1'den farkı:** Denetim sonucu bulunan 9 kritik hata, 7 belirsizlik ve 3 eksik
> düzeltildi. Değişenler §16'da listelenmiştir.

---

## 0. Amaç

Konveyör bandı üzerinde geçen çuvalları kamerayla sayan, sayımı bir sevkiyat oturumuna
bağlayan ve sonucu ERP'ye aktaran **yerel (on-premise) bir yazılım ürünü**.

**Bu bir tek-tesis projesi değildir.** Mevcut tesis "ilk müşteri"dir. Her tasarım kararı
şu soruyla sınanır: *ikinci müşteride bu ne olur?*

Ürünün asıl başarı ölçütü doğruluk değil:

> **Yeni bir tesiste kabul edilebilir doğruluğa kaç günde ulaşılıyor?**

---

## 1. Genellik ilkeleri (ihlal edilemez)

Aşağıdakiler çekirdek kodda sabit değer olarak geçemez:

| Yasak sabit | Kaynağı |
|---|---|
| Kamera sayısı | `camera` tablosu |
| Kamera markası / modeli | Yalnızca opsiyonel keşif eklentisinde |
| Çözünürlük, FPS | Akıştan okunur |
| Bant hızı, bant yönü | Optik akışla ölçülür → `line_calibration` |
| Piksel–mm ölçeği | Model destekli kalibrasyon → `line_calibration` |
| ROI / gate koordinatları | `config_version` |
| Çuval boyutu / ağırlığı | `product_profile` |
| Eşikler (güven, IoU, alan, gate) | `config_version` |
| ERP tipi, hareket tipi, malzeme kodu | `erp_adapter` config + `product_profile` |

CI'da `tools/hardcode_check.py` bu kuralı denetler (§12.3).

---

## 2. Lisans politikası

**Kapsam: tümü.** Üretim, eğitim, etiketleme, geliştirme bağımlılıkları. İstisna yoktur.

### 2.1 İzin verilen

`MIT`, `BSD-2-Clause`, `BSD-3-Clause`, `Apache-2.0`, `ISC`, `PostgreSQL`, `Python-2.0`,
`Unlicense`, `CC0-1.0`, `Zlib`

> **MPL-2.0 çıkarıldı (v2 düzeltmesi).** Dosya bazlı zayıf kopyaleft içerir. "Sıfır
> kopyaleft riski" duruşuyla tam uyuşmadığı için izin listesinden alındı. Zorunlu bir
> ihtiyaç doğarsa vaka bazında ve yazılı gerekçeyle değerlendirilir.

### 2.2 Yasaklı

`AGPL-*`, `GPL-*`, `LGPL-*` (tek istisna §2.4), `SSPL`, `BUSL`, `PML-*`, `CC-BY-NC-*`,
`Elastic-*`, ve "custom" / "proprietary" işaretli her şey.

### 2.3 Adı geçen yasaklar (dolaylı bağımlılık olarak bile)

- `ultralytics` (AGPL-3.0)
- `boxmot` / `yolo_tracking` (AGPL-3.0)
- `rfdetr[plus]`, `RFDETRXLarge`, `RFDETR2XLarge` (PML 1.0)
- `abewley/sort` (GPL-3.0)
- `minio` (AGPL-3.0)
- YOLOv5 / v8 / v11 ağırlıkları ve bunlardan türetilmiş checkpoint'ler

**Kural:** Yasaklı repoların kaynak kodu incelenmez. "Nasıl yapmışlar" diye bile
açılmaz. Bakmadığını kanıtlamak, bakıp benzemediğini kanıtlamaktan kolaydır.

### 2.4 LGPL istisnası — yalnızca FFmpeg / PyAV

Video çözme için pratik alternatifi yoktur. Koşullar:

- Yalnızca dinamik linkleme (PyAV standart wheel)
- FFmpeg'in **LGPL build**'i; `--enable-gpl` içeren build kullanılmaz
- `THIRD_PARTY_NOTICES.md` içinde belgelenir

### 2.5 NVIDIA katmanı — dürüst çerçeve (v2 düzeltmesi)

> v1'de TensorRT ve DeepStream "NVIDIA özel EULA" gerekçesiyle yasaklanmıştı. **Bu
> gerekçe tutarsızdı:** ONNX Runtime'ın CUDA sağlayıcısı zaten CUDA runtime'a bağlıdır
> ve `nvidia/cuda` temel imajı da NVIDIA konteyner lisansına tabidir. Özel NVIDIA
> lisansından tamamen kaçmak mümkün değildir.

Doğru konumlandırma:

- **Asıl endişe kopyaleft riskidir** ve NVIDIA bileşenlerinde bu risk yoktur
- CUDA runtime ve sürücü **kabul edilmiştir** (kaçınılmaz, ücretsiz)
- TensorRT ve DeepStream **yasak değil, gereksiz**: ONNX Runtime yeterliyse eklenmez
  (az bağımlılık ilkesi). DeepStream ayrıca segmentasyon için özel maske parser'ı
  yazmayı zorunlu kılar — bu tek başına yeterli bir dışlama gerekçesidir.
- M5 benchmark'ı yetersizlik gösterirse TensorRT eklenebilir; karar teknik, hukuki değil

### 2.6 Ağırlık lisansları

Kod lisansı ile önceden eğitilmiş ağırlık lisansı farklı olabilir. Her model için ikisi
ayrı doğrulanır ve `docs/provenance.md`'ye yazılır.

### 2.7 Köken kaydı

Her modülün başında:

```python
# Kaynak: Zhang et al., "ByteTrack" (ECCV 2022), arXiv:2110.06864
# Referans kod: github.com/FoundationVision/ByteTrack (MIT)
# Özgün kısım: BeltMotionModel, maske-IoU maliyet matrisi
```

`THIRD_PARTY_NOTICES.md` ve `docs/provenance.md` repo kökünde; CI boş olmadıklarını
doğrular.

### 2.8 Sabitlenen sürümler

Lisans değişikliği riskine karşı, aşağıdakilerin sürümü `pyproject.toml` ve
`docker-compose.yml`'de **tam olarak** sabitlenir (aralık değil):

`rfdetr`, `supervision`, `cvat` imajı, `onnxruntime-gpu`, `av`

Sürüm yükseltmesi öncesi lisans yeniden doğrulanır.

---

## 3. Teknoloji yığını

| Katman | Seçim | Lisans |
|---|---|---|
| Dil (backend/ML) | Python 3.12 | PSF |
| Segmentasyon | RF-DETR Seg (Nano / Small / Medium) | Apache-2.0 |
| Eğitim | PyTorch | BSD-3 |
| Çıkarım | ONNX Runtime (CUDA EP) | MIT |
| Takip | ByteTrack (vendored) + özgün `BeltMotionModel` | MIT |
| Atama | `scipy.optimize.linear_sum_assignment` | BSD-3 |
| Kalman | `filterpy` | MIT |
| CV | OpenCV, supervision | Apache-2.0, MIT |
| Video decode | PyAV (FFmpeg LGPL, dinamik) | BSD-3 |
| Augmentation | Albumentations | MIT |
| Etiketleme | CVAT self-hosted | MIT |
| Metrikler | pycocotools, TrackEval | BSD, MIT |
| API | FastAPI, Pydantic v2, SQLAlchemy 2, Alembic | MIT |
| Veritabanı | PostgreSQL 16 | PostgreSQL |
| Arayüz | TypeScript, React 18, Vite, TailwindCSS, TanStack Query | MIT |
| Dağıtım | Docker + Compose | Apache-2.0 |
| Test | pytest, pytest-asyncio | MIT |

### 3.1 Bilinçli olarak dışarıda

| Dışarıda | Gerekçe |
|---|---|
| TensorRT, DeepStream | §2.5 — gereksiz bağımlılık, özel parser yükü |
| Kubernetes | Tek düğüm; over-engineering |
| Celery / RabbitMQ / Redis | PostgreSQL job kuyruğu yeterli; az bileşen = az arıza |
| MLflow / DVC | Arayüzden yönetim şartı kendi sürümleme tablolarımızı gerektiriyor |
| Mikroservisler | Genellik arayüz soyutlamasıyla gelir, servis bölmekle değil |
| MinIO | AGPL-3.0; dosyalar yerel diskte içerik-adresli düzende |

---

## 4. Mimari

### 4.1 Repo yapısı

```
cuval-sayim/
├── packages/
│   ├── cs_core/
│   │   ├── models.py            # domain modelleri, enum'lar
│   │   ├── frame.py             # Frame veri sınıfı (§4.5)
│   │   ├── interfaces/
│   │   │   ├── video_source.py
│   │   │   ├── erp_adapter.py
│   │   │   ├── io_controller.py
│   │   │   ├── session_identity.py
│   │   │   └── frame_transport.py
│   │   └── geometry.py          # ROI, gate, maske yardımcıları
│   ├── cs_vision/               # ONNX sarmalayıcı, ön/son işleme
│   ├── cs_tracking/             # vendored ByteTrack + BeltMotionModel + merge_detector
│   ├── cs_counting/             # gate durum makinesi + area_counter
│   ├── cs_data/                 # extract, ssim, synth, split, cvat, mining
│   ├── cs_eval/                 # replay harness, metrikler, skor tablosu
│   └── cs_storage/              # SQLAlchemy modelleri, repository'ler, migration
├── services/
│   ├── supervisor/
│   ├── ingest/
│   ├── inference/
│   ├── api/
│   ├── jobrunner/
│   └── erp_relay/
├── web/
├── drivers/
│   ├── video_rtsp/  video_file/
│   ├── erp_csv/     erp_sap_odata/
│   ├── io_noop/     io_usb_relay/
│   └── identity_operator/  identity_erp/
├── tools/
│   ├── license_check.py
│   ├── hardcode_check.py
│   ├── replay.py
│   └── benchmark.py
├── docs/  (provenance.md, architecture.md, annotation_guide.md)
├── THIRD_PARTY_NOTICES.md
├── docker-compose.yml
└── pyproject.toml
```

**Kural:** `services/` içindeki hiçbir modül başka bir `services/` modülünü import
edemez. Paylaşım yalnızca `packages/` üzerinden.

### 4.2 Process topolojisi

```
N × kamera (RTSP)
      ↓
supervisor ──başlatır/durdurur──▶ ingest × N (kamera başına 1 process)
                                        ↓  paylaşımlı bellek halkası
                                  inference (düğüm başına 1, GPU, batch)
                                        ↓
                                  PostgreSQL
                                  ↙     ↓     ↘
                          jobrunner    api    erp_relay
                                        ↑
                                  React arayüzü
```

**Gerekçeler:**

- `ingest` kamera başına ayrı → bir kamera çökünce diğerleri devam eder
- `inference` düğüm başına **tek** → GPU'yu bölmek batch verimliliğini yok eder,
  VRAM'de modelin N kopyasını tutar, CUDA context geçişine sokar.
  **Kamera başına çıkarım konteyneri performansı düşürür, artırmaz.**
- Ham kareler ağdan geçmez (1080p ≈ 6 MB/kare)

### 4.3 Çok düğüm hazırlığı

- `camera.node_id` alanı bugünden mevcut (hepsi aynı düğüm)
- `FrameTransport` arayüzü: `SharedMemoryTransport` (varsayılan),
  `ZeroMqTransport` (ileride)
- Konteyner sınırı = **düğüm sınırı**:
  - `edge-node` imajı: supervisor + ingest'ler + inference
  - `core` imajı: api + jobrunner + erp_relay
- Düğümler arası ağdan geçen tek şey `count_event` kayıtlarıdır

### 4.4 Sürücü arayüzleri

```python
class VideoSource(Protocol):
    def open(self, config: dict, epoch: int) -> None: ...
    def read(self) -> Frame | None: ...
    def close(self) -> None: ...
    @property
    def is_connected(self) -> bool: ...

class ErpAdapter(Protocol):
    def submit_session(self, payload: SessionPayload) -> ErpResult: ...
    def query_status(self, external_ref: str) -> ErpStatus: ...
    @property
    def supports_status_query(self) -> bool: ...   # v2: CSV adaptörü False döner

class IoController(Protocol):
    def set_signal(self, name: str, value: bool) -> None: ...
    def read_signal(self, name: str) -> bool: ...

class SessionIdentity(Protocol):
    def acquire(self, line_id: int) -> SessionRef | None: ...

class FrameTransport(Protocol):
    def publish(self, frame: Frame) -> PublishResult: ...
    def consume(self, timeout_ms: int) -> list[Frame]: ...
```

Uygulamalar `drivers/` altında, Python `entry_points` ile keşfedilir.

**İlk sürümde yazılacaklar:** RTSP + Dosya, CSV + SAP OData, No-op + USB röle,
Operatör + ERP kimliği. Modbus/OPC UA/ONVIF/barkod arayüzde tanımlı, kodu ilk ihtiyaçta.
**Kullanılmayan sürücü yazılmaz.**

### 4.5 Kare taşıma katmanı (v2 — yeni bölüm)

> Sistemin en riskli parçasıdır ve v1'de tasarlanmamıştı.

**`Frame` veri sınıfı:**

```python
@dataclass(frozen=True)
class Frame:
    camera_id: int
    stream_epoch: int
    frame_index: int         # epoch içinde 0'dan artan
    monotonic_ns: int        # ingest'in monotonik saati (yakalama anı)
    wall_clock: datetime     # yalnızca görüntüleme için
    shm_name: str
    shape: tuple[int, int, int]
    dtype: str
```

Görüntü verisi `Frame` içinde taşınmaz; paylaşımlı bellek bloğunun adı taşınır.

**Halka arabellek (ring buffer):**

- Kamera başına ayrı halka, `ring_slots` slot (varsayılan 8, `config_version`'da)
- Her slot bir `multiprocessing.shared_memory` bloğu, ön-tahsisli
- Metadata küçük bir `multiprocessing.Queue` üzerinden geçer
- Slot yaşam döngüsü: `ingest` yazar → yayınlar → `inference` tüketir → serbest bırakır

**Kare düşürme politikası (sayım doğruluğunu doğrudan etkiler):**

- `inference` yavaş kalırsa `ingest` **bloke olmaz** — en eski slotu düşürür
- Her düşen kare `dropped_frame` sayacına işlenir ve loglanır
- **Kritik kural:** Bir oturum boyunca ardışık düşen kare sayısı
  `max_consecutive_drops` eşiğini aşarsa (varsayılan 3), oturum `degraded` olur.
  Sebep: art arda düşen kareler bir çuvalın gate'i hiç görülmeden geçmesine yol açabilir.
- Düşen kare oranı Sistem ekranında canlı gösterilir

**Batch penceresi:**

- `inference` en fazla `batch_wait_ms` (varsayılan 30 ms) bekler, sonra eldeki
  kareleri batch'ler — tam kamera sayısını beklemez
- Batch içindeki kareler farklı `monotonic_ns` taşıyabilir; her kare kendi
  zaman damgasıyla işlenir

**Zaman semantiği (v2 — netleştirildi):**

- Tüm sayım mantığı **`frame_index` + `monotonic_ns`** üzerine kurulur
- `wall_clock` yalnızca arayüz gösterimi ve ERP payload'ı içindir
- `count_event.crossing_timestamp` = gate geçişinin gerçekleştiği karenin `wall_clock`'u
- `count_event.frame_index` ayrıca saklanır → adli incelemede tam kare bulunabilir

---

## 5. Veri modeli

Tüm ayar ve sürüm kayıtları **değişmezdir**. Güncelleme yok, yeni sürüm satırı açılır.

### 5.1 Tesis hiyerarşisi

```sql
site            (id, name, timezone, locale, created_at)
node            (id, site_id, hostname, gpu_info, status, last_heartbeat)
line            (id, site_id, name, status)
camera          (id, line_id, node_id, source_driver, source_config JSONB,
                 role, enabled, created_at)
gate            (id, line_id, name, order_index)
product_profile (id, site_id, name, nominal_weight_g, nominal_dims_mm JSONB,
                 template_images JSONB, erp_material_code)
```

`camera.role`: `counting` | `vehicle_watchdog` | `auxiliary`

### 5.2 Stream epoch — kalıcı sayaç (v2 düzeltmesi)

> **v1 hatası:** `epoch` process belleğinde tutuluyordu. `ingest` yeniden başladığında
> sıfırlanıyor, aynı oturumdaki eski satırlarla `UNIQUE` çakışması yaratıyor ve
> **sayım olayı sessizce düşüyordu.**

Çözüm — kamera başına kalıcı sequence:

```sql
camera_epoch (camera_id INT PRIMARY KEY REFERENCES camera(id),
              current_epoch BIGINT NOT NULL DEFAULT 0);
```

```sql
-- ingest her açılışta / her reconnect'te:
UPDATE camera_epoch SET current_epoch = current_epoch + 1
WHERE camera_id = :cid RETURNING current_epoch;
```

Epoch değeri hiçbir koşulda geri gitmez. `ingest` başlayamazsa epoch tüketilmiş olur —
zararsızdır, boşluk sorun değildir.

### 5.3 Kalibrasyon — iki aşamalı (v2 düzeltmesi)

> **v1 hatası (bootstrap kısır döngüsü):** `px_per_mm`, ortalama çuval alanı ve
> `merge_detector` eşikleri çuval maskelerinden ölçülüyordu — yani çalışan bir model
> gerektiriyordu. Ama kalibrasyon kurulum sihirbazının 5. adımında, model ise 9.
> adımdaydı. Sıra tutarsızdı.

```sql
line_calibration (
  id, line_id, stage, created_at, created_by,
  -- Aşama 1: hareket (model gerektirmez)
  belt_speed_px_per_frame REAL,
  belt_direction_vector JSONB,
  -- Aşama 2: ölçek (model gerektirir)
  px_per_mm REAL,
  mean_bag_gate_area_px REAL,
  bag_area_stddev_px REAL,
  source_video_ref TEXT,
  source_model_version_id BIGINT,
  is_active BOOLEAN
);
```

`stage`: `motion` | `scale`

**Aşama 1 — hareket kalibrasyonu (modelsiz):**
ROI içinde seyrek optik akış (Lucas-Kanade). Bandın piksel/kare hızı ve yön vektörü
ölçülür. Sihirbazın erken adımında, model olmadan çalışır.

**Aşama 2 — ölçek kalibrasyonu (model sonrası):**
İlk model eğitildikten sonra, **seyrek akışlı (örtüşmesiz) bir video** üzerinde
otomatik çalışan bir job:
1. Yalıtılmış (komşusuyla örtüşmeyen) maskeleri filtreler
2. Gate bölgesindeki alanlarının medyanını ve standart sapmasını hesaplar
3. `product_profile.nominal_dims_mm` ile karşılaştırıp `px_per_mm` türetir
4. Sonucu `line_calibration` (stage=`scale`) olarak yazar

**Bağımlılık kuralı:** `merge_detector` ve `area_counter`, aktif bir `scale`
kalibrasyonu yoksa **çalışmaz ve devre dışı kalır** — hatalı eşikle çalışmaktansa
kapalı kalmaları tercih edilir. Bu durum arayüzde uyarı olarak gösterilir.

### 5.4 Sürümleme çekirdeği

```sql
dataset_version   (id, site_id, name, manifest_hash, frame_count, synthetic_count,
                   split_spec JSONB, annotation_guide_version, created_at)
training_run      (id, dataset_version_id, base_model_version_id, run_kind,
                   hyperparams JSONB, status, log_ref, metrics JSONB,
                   started_at, finished_at)
model_version     (id, training_run_id, onnx_hash, onnx_path, eval_scores JSONB,
                   stage, created_at)
config_version    (id, line_id, payload JSONB, payload_schema_version INT,
                   note, created_by, created_at)
deployment_bundle (id, line_id, model_version_id, config_version_id,
                   calibration_id, git_commit, activated_at, deactivated_at,
                   activated_by)
```

`model_version.stage`: `draft` | `shadow` | `active` | `retired`
`training_run.run_kind`: `base` | `site_adaptation` (§7.6)

**`payload_schema_version` (v2 eklendi):** Yeni bir eşik eklendiğinde eski
`config_version` satırları o alandan yoksun kalır. Okuma katmanı, şema sürümüne göre
eksik alanları belgelenmiş varsayılanlarla doldurur. Varsayılanlar
`cs_core/config_defaults.py` içinde sürüm sürüm tutulur.

`config_version.payload` içeriği: ROI poligonu, gate geometrisi, PRE/POST sınırları,
güven eşiği, maske IoU eşiği, merge tespit eşikleri, alan-integrali parametreleri,
`ring_slots`, `batch_wait_ms`, `max_consecutive_drops`, sarı ışık eşiği.

### 5.5 Sayım defteri (Count Event Ledger)

```sql
count_event (
  event_id             UUID PRIMARY KEY,
  session_id           BIGINT NOT NULL,
  line_id              INT NOT NULL,
  camera_id            INT NOT NULL,
  stream_epoch         BIGINT NOT NULL,
  track_id             INT NOT NULL,
  crossing_seq         INT NOT NULL,      -- v2: track başına artan geçiş sırası
  gate_id              INT NOT NULL,
  crossing_timestamp   TIMESTAMPTZ NOT NULL,
  frame_index          BIGINT NOT NULL,   -- v2: adli inceleme için
  direction            SMALLINT NOT NULL, -- +1 ileri, -1 geri
  confidence           REAL,
  merge_flag           BOOLEAN DEFAULT FALSE,
  deployment_bundle_id BIGINT NOT NULL,
  evidence_ref         TEXT,
  created_at           TIMESTAMPTZ DEFAULT now(),
  UNIQUE (session_id, camera_id, stream_epoch, track_id, gate_id, crossing_seq)
);
```

> **v2 düzeltmesi — `crossing_seq`:** v1'deki kısıt `(session, camera, epoch, track, gate)`
> idi. Bir çuval ileri geçip geri kayıp tekrar ileri geçtiğinde — aynı `track_id` ile —
> ikinci geçiş kısıt tarafından reddediliyor ve **sistem eksik sayıyordu.** Bu, tam olarak
> önlenmeye çalışılan hata türüydü. `crossing_seq`, tracker tarafından track başına
> tutulan ve her gate geçişinde artan bir sayaçtır.

**`stream_epoch` neden zorunlu:** `track_id` global fiziksel kimlik değildir; akış
yeniden bağlandığında sıfırlanır. Gerçek kimlik `camera_id + stream_epoch + track_id`
üçlüsüdür.

**Toplam sayım (v2 netleştirildi):** `session.counted_total` inference tarafından
artırılmaz. Oturum kapanışında ledger'dan tek sorguyla türetilir:

```sql
SELECT COALESCE(SUM(direction), 0) FROM count_event WHERE session_id = :sid;
```

Ledger tek doğruluk kaynağıdır; hiçbir sayaç ondan bağımsız tutulmaz.

### 5.6 Oturum

```sql
session (id, line_id, product_profile_id, external_ref, target_count,
         status, opened_at, closed_at, locked_at,
         counted_total, area_estimate_total, discrepancy_flag,
         reconciliation_id)
```

`status`: `open` | `counting` | `paused` | `degraded` | `closed` |
`reconcile_required` | `reconciled`

**İlke:** Bir kamera N saniye bağlantıyı kaybettiyse ve bant çalışmaya devam ettiyse,
sistem "kaldığım yerden devam" demez. O sürede ne geçtiği bilinmiyor.
**Gözlem kaybı = sayım bütünlüğü kaybı.**

### 5.7 Mutabakat akışı (v2 — yeni)

> **v1 eksiği:** `discrepancy_flag` ve `degraded` durumlarında "uyarı verilir" deniyor,
> sonrası tanımsızdı. İlk planda eleştirilen hatanın aynısı tekrarlanmıştı.

```sql
reconciliation (
  id, session_id, trigger_reason, opened_at,
  assigned_role, evidence_refs JSONB,
  resolution,              -- accept_system | manual_override | void_session
  resolved_count INT,
  resolved_by, resolved_at, note
);
```

`trigger_reason`: `degraded_session` | `count_area_mismatch` | `erp_conflict` |
`operator_request`

**Akış:**

1. Tetikleyici oluşur → `session.status = reconcile_required`, `reconciliation` kaydı açılır
2. Oturum **ERP'ye gönderilmez** (outbox kaydı oluşmaz)
3. Arayüzde `engineer` rolüne bildirim düşer
4. İnceleme: ledger dökümü, düşen kare aralıkları, `merge_flag` oranı, alan tahmini,
   varsa video kanıtı
5. Karar: sistem sayımını kabul / elle sayıyla değiştir / oturumu iptal et
6. Kayıt kapanır → `session.status = reconciled`, gerekiyorsa outbox oluşturulur

**Kural:** `reconcile_required` durumundaki bir oturum asla otomatik gönderilmez ve
zaman aşımıyla kendiliğinden kapanmaz. İnsan kararı zorunludur.

### 5.8 Job kuyruğu ve outbox

```sql
job (id, kind, payload JSONB, status, priority, requires_gpu,
     attempts, max_attempts, lease_until, heartbeat_at,
     last_error, created_at, started_at, finished_at)

outbox (id, session_id, payload JSONB, status, attempts,
        next_attempt_at, external_ref, last_error, created_at)
```

`job.kind`: `extract_frames` | `synthesize` | `build_dataset` | `train` |
`export_onnx` | `evaluate` | `replay` | `calibrate_motion` | `calibrate_scale` |
`mine_hard_frames`

**Kira ve kalp atışı (v2 eklendi):** `jobrunner` eğitim ortasında çökerse job sonsuza
dek `running` kalırdı. Çözüm: alınırken `lease_until = now() + lease_duration`, çalışırken
periyodik `heartbeat_at` güncellemesi. Süresi dolmuş job'lar bir süpürücü tarafından
`queued`'a döndürülür ve `attempts` artırılır; `max_attempts` aşılırsa `failed`.

Her iki tüketici de `SELECT ... FOR UPDATE SKIP LOCKED` kullanır.

---

## 6. Çekirdek algoritma

### 6.1 Problemin gerçek şekli

Saha gerçeği: çuvallar bant üzerinde **tek katman halinde yatık** durur, banttan taşma
olmaz, ancak **kenarlarından birbirinin üzerine sık sık biner** (kiremitleme).

| Hata modu | Olasılık | Sonuç |
|---|---|---|
| **Yanlış birleşme** — iki çuval tek maske | **Yüksek, baskın** | **Eksik sayım** |
| Yanlış bölünme — bir çuval iki maske | Orta | Fazla sayım |
| Kimlik değişimi | Orta | Fazla/eksik |
| Tam örtüşme sonrası kayıp | Düşük, sıfır değil | Fazla sayım |

**Tasarım yanlış birleşmeye karşı optimize edilir.** Eksik sayım sevkiyatta en maliyetli
hatadır ve bu sahanın fiziği tam olarak onu üretir.

### 6.2 Neden RF-DETR Seg

- Query tabanlı, **NMS kullanmaz** → birbirine değen nesnelerde bastırma hatası oluşmaz
- Instance segmentation → maske konturu bbox'tan çok daha ayırt edici
- Apache-2.0, ağırlıklar dahil

Nano ile başla; `evaluate` sonucuna göre Small/Medium'a çık.
**Nano–Large serbest, XL/2XL yasak (PML 1.0).**

### 6.3 Anotasyon kuralı — amodal (v2 — kritik karar)

> **v1 eksiği:** Örtüşen çuvalın nasıl etiketleneceği tanımsızdı. Bir örtüşme
> projesinde bu, her şeyi belirleyen karardır. Tanımsız bırakılırsa beş etiketçi beş
> farklı şey yapar.

**Karar: amodal etiketleme.** Çuvalın **tahmini tam gövdesi** işaretlenir — üstüne
binen çuval tarafından kapatılan kısım dahil.

**Gerekçe:** Alan tabanlı sinyallerin tamamı (merge_detector'ın alan sinyali,
`area_counter`, ölçek kalibrasyonu) tutarlı bir "tek çuval alanı" tanımına dayanır.
Yalnızca görünen kısım etiketlenirse bu alan örtüşme oranıyla değişir ve tüm eşikler
anlamsızlaşır.

**Kurallar (`docs/annotation_guide.md`):**

- Kapatılan kenar, çuvalın görünen kenarlarından makul şekilde tahmin edilerek uzatılır
- Görünürlüğü %25'in altına düşen çuval `heavily_occluded` niteliğiyle işaretlenir
- Tamamen görünmeyen çuval etiketlenmez
- Her çuvala `visible_ratio` niteliği girilir (0.0–1.0, kabaca)
- Kılavuz sürümlenir; `dataset_version.annotation_guide_version` bunu kaydeder
- **Sentetik veri de amodal maske üretir** — kural her iki kaynakta aynıdır

**Etiketleme tutarlılığı:** İlk 100 karede iki etiketçi bağımsız çalışır, IoU uyumu
ölçülür. %0.85 altındaysa kılavuz netleştirilir ve tekrarlanır.

### 6.4 Sınıflar (v2 netleştirildi)

> **v1 hatası:** `merge_detector`'ın 4. sinyali "aynı maskede iki logo" diyordu ama
> logo sınıfı veri setinde, etiketlemede veya eğitimde hiçbir yerde tanımlı değildi.

**Karar: iki sınıf, baştan sona takip edilir.**

| Sınıf | Tip | Kullanım |
|---|---|---|
| `bag_body` | Instance segmentation maskesi | Birincil sayım |
| `print_mark` | Bounding box | `merge_detector` sinyali 4 (ikincil) |

`print_mark`, ürün üzerindeki sabit baskı/logo şablonudur. `product_profile`'a bağlıdır;
tanımlı değilse sinyal 4 devre dışı kalır ve `merge_detector` üç sinyalle çalışır.

**Kritik denge:** Çuval ters/yan geçtiğinde baskı görünmez. Bu yüzden `print_mark`
**asla birincil sayım kaynağı değildir** ve etiketleme setinde baskının göründüğü ve
görünmediği örnekler dengeli olarak yer alır — model gövde şeklinden ayrım yapmayı
baskıdan bağımsız öğrenmelidir.

### 6.5 Sentetik veri — kaynaşmaya karşı ana silah

`cs_data/synth.py`

**Girdi:** `product_profile.template_images`, hattın boş bant görüntüleri,
`line_calibration` (ölçek, yön).

**Üretim:**

1. Boş bant arka planı seç
2. 1–4 çuval yerleştir, bant yönü boyunca
3. **Kiremitleme örtüşmesini bilinçli üret** — komşu örtüşme oranı, gerçek videodan
   ölçülen dağılıma göre örneklenir (§14, M1 girdisi). Ölçüm yoksa varsayılan olarak
   %0–45 düzgün dağılım kullanılır ve bu durum `dataset_version` notuna yazılır
4. Z-sırasını rastgele belirle
5. Perspektif, ölçek, dönme, ışık, gölge, motion blur, JPEG bozulması
6. **Amodal maskeleri** COCO formatında yaz (§6.3)
7. `print_mark` kutularını, çuval yönüne göre görünür/görünmez olarak yaz

**Eğitim akışı:** sentetik ön-eğitim → gerçek veriyle fine-tune.

### 6.6 Takip

**Temel içgörü: bütün çuvallar birbirinin aynıdır.** Görsel ReID (StrongSORT,
DeepOCSORT) burada değer üretmez. Kimlik sürekliliği hareket modelinden gelmelidir.

**`BeltMotionModel` (özgün):**

- ROI içinde seyrek optik akış (Lucas-Kanade) ile bant hızı sürekli tahmin edilir
- Düşük geçiren filtre ile `line_calibration`'a yazılır
- Her track'in Kalman filtresi bu hız vektörünü **prior** olarak kullanır
- Sonuç: bir çuval birkaç kare kaybolsa bile nerede yeniden görüneceği fizikle
  neredeyse deterministiktir

**Eşleştirme:**

- ByteTrack'in iki aşamalı ilişkilendirmesi (düşük skorlu tespitler de kullanılır)
- Maliyet: `1 - maskIoU` ve merkez mesafesinin ağırlıklı toplamı
  (**bbox IoU kullanılmaz** — örtüşmede yanıltıcıdır)
- Atama: `linear_sum_assignment`
- Her track `crossing_seq` sayacı taşır (§5.5)

### 6.7 Kaynaşma tespiti (`merge_detector.py`)

Bir maskenin iki çuval içerip içermediğini bağımsız sinyallerle sorgular:

1. **Alan:** maske alanı > `mean_bag_gate_area_px × merge_area_ratio` (varsayılan 1.5)
2. **Şekil:** konveks kabuk açığı, en-boy oranı sapması
3. **Zamansal:** önceki karelerde iki ayrı track bu bölgeye giriyor muydu
4. **Baskı:** aynı maskede iki `print_mark` (yalnızca profil tanımlıysa)

**Karar:** İki veya daha fazla sinyal aynı yönü gösterirse "iki nesne hipotezi" açılır.

**Maske zorla kesilmez.** Tracker iki hipotezi destekliyorsa iki nesne **latent track**
olarak taşınır. Gerçekten ayırmak gerekirse watershed tohumları olarak alan ağırlık
merkezleri kullanılır — düz geometrik çizgiyle kesilmez.

Ayrıştırılan olaylar `merge_flag = true` ile işaretlenir; oranı izlenmesi gereken bir
sağlık göstergesidir.

**Ön koşul:** Aktif `scale` kalibrasyonu yoksa modül devre dışıdır (§5.3).

### 6.8 Sayım kapısı

Üç bölge: `PRE → GATE → POST`

```
YENİ TRACK → KARARLI TRACK → PRE → GATE GEÇİŞİ (doğru yönde)
           → POST DOĞRULAMASI → LEDGER → SAYILDI
```

- Anchor: **maske ağırlık merkezi, bant eksenine izdüşürülmüş**
  (bbox alt-orta noktası kullanılmaz — deforme/dönmüş çuvalda kararsızdır)
- Ters yön geçişi `direction = -1` olarak yazılır, `crossing_seq` yine artar
- İlke: *"bir maske gördüm = 1 çuval"* değil,
  *"bir fiziksel nesnenin gate'i geçtiğini doğruladım = 1 çuval"*

### 6.9 Alan-integrali sayacı — bağımsız ikinci tahmin

```
tahmin = (gate hattından geçen toplam maske alanı) / mean_bag_gate_area_px
```

Takipten tamamen bağımsızdır. Kaynaşma olsa bile alan korunur — yanlış birleşmeye karşı
doğal kontroldür.

Ledger sayacı ile bu tahmin `discrepancy_threshold`'u aşacak şekilde ayrışırsa
`session.discrepancy_flag` set edilir ve §5.7 mutabakat akışı tetiklenir.

**Ön koşul:** Aktif `scale` kalibrasyonu (§5.3).

### 6.10 Aktif öğrenme — zor kare madenciliği (v2 — geri eklendi)

> **v1 eksiği:** Konuşmada üzerinde anlaşılan ama dokümana girmeyen mekanizma.
> Etiketleme yükünü düşüren ana araçtır.

`cs_data/mining.py`, `mine_hard_frames` job'ı olarak çalışır. Toplanan kareler:

- Düşük güven skorlu tahminler
- **İki farklı model sürümünün çeliştiği** kareler
- **Ledger sayacı ile alan tahmininin uyuşmadığı** zaman aralıkları
- `merge_flag` açılan olaylar
- Track fragmentation / ID switch şüphesi olan anlar
- Yeni kamera veya belirgin ışık değişimi örnekleri
- **İnsanın sonradan yüksek güvenli bir tahmini yanlış bulduğu örnekler**

> Seçim kriteri **yalnızca düşük güven olamaz** — sinir ağlarının en tehlikeli
> hataları yüksek güvenle yapılanlardır.

Toplanan kareler CVAT'a görev olarak düşer, hızlı insan onayı/düzeltmesiyle veri setine
eklenir, yeni bir `dataset_version` açılır.

---

## 7. Model stratejisi (v2 — genişletildi)

> **v1 eksiği:** Ürünleşmenin çekirdeği olan temel model / tesis adaptasyonu ayrımı
> yalnızca sihirbaz adımı olarak geçiyordu, eğitim stratejisi olarak tanımlı değildi.
> Her tesiste sıfırdan eğitim yapılırsa ürün ölçeklenmez.

### 7.1 Temel model (`run_kind = base`)

Tüm tesislerden biriken veriyle eğitilen genel "torba/çuval" modeli. Her kurulumdan
sonra havuz büyür ve bir sonraki kurulum kolaylaşır. Ürünün bileşik getirisi buradadır.

- Girdi: tüm sitelerin `dataset_version`'larının birleşimi + sentetik havuz
- Merkezi olarak, kurulumdan bağımsız eğitilir
- `model_version` olarak saklanır ve yeni tesislere başlangıç noktası olur

### 7.2 Tesis adaptasyonu (`run_kind = site_adaptation`)

Yeni sahada birkaç yüz gerçek kare + o ürüne özel sentetik veri ile fine-tune.
`training_run.base_model_version_id` temel modeli işaret eder.

**Amaç sıfırdan eğitim değil, uyarlamadır.**

### 7.3 Veri seti bölünmesi ve hard holdout (v2 — genişletildi)

**Sızıntı önleme:** Bölme kareler çıkarıldıktan sonra rastgele yapılmaz. Doğru sıra:
**önce** video/oturum/gün/kamera bazında böl, **sonra** her bölümün kendi karelerini
çıkar. Aynı oturumdan hiçbir kare iki sete giremez.

**Hard holdout (v1'de eksikti):** Sabit yüzde bölünmesi genelleme gücünü göstermez.
`split_spec` şunu tanımlar:

- `train`: seçilmiş kameraların seçilmiş günleri
- `val`: ayrı oturumlar
- `hard_holdout`: **hiç görülmemiş bir kameranın hiç görülmemiş bir vardiyası**,
  tercihen yoğun kiremitleme içeren

`hard_holdout` eğitim ve hiperparametre seçiminde **hiç kullanılmaz**. Yalnızca kabul
kararında bir kez bakılır. Skorları `model_version.eval_scores` içinde ayrı raporlanır.

### 7.4 Değerlendirme metrikleri

Kare bazlı mAP yeterli değildir. Ölçülecekler:

- Oturum başına mutlak hata (± kaç çuval)
- Tam doğru oturum oranı
- 1000 çuvalda fazla (FP) / eksik (FN)
- **Kaynaşma kaynaklı eksik sayım oranı** (ana metrik)
- ID switch, track fragmentation
- Tam örtüşme sonrası kimlik koruma oranı
- Sistematik sapma (sürekli fazla mı eksik mi)
- Ledger sayacı ile alan tahmini arasındaki ortalama sapma
- Düşen kare oranı

Hedef doğruluk tek bir yüzde olarak tanımlanmaz. 1000 çuvalda %95 = 50 çuval hata
demektir ve sevkiyatta kabul edilemez.

---

## 8. Roller ve API (v2 — yeni bölüm)

> **v1 eksiği:** "Kimlik doğrulama" yazıyordu ama rol modeli ve endpoint listesi yoktu.
> Arayüzü kodlayacak biri için bu bloke edicidir.

### 8.1 Roller

| Rol | Yetki |
|---|---|
| `operator` | Canlı görüntüleme, oturum açma/kapatma, hedef girme, gönderim onayı |
| `engineer` | + veri, eğitim, model aktifleştirme, ayar, kalibrasyon, mutabakat |
| `admin` | + kullanıcı yönetimi, tesis/hat/kamera CRUD, sürücü konfigürasyonu |

Kimlik doğrulama: oturum çerezi + sunucu tarafı oturum tablosu. Harici IdP ilk sürümde
yoktur.

### 8.2 Endpoint yüzeyi (özet)

```
POST   /auth/login                        herkes
GET    /sites, /lines, /cameras           operator+
POST   /sites, /lines, /cameras           admin
POST   /cameras/{id}/test                 admin      bağlantı testi
GET    /cameras/{id}/preview              engineer+  canlı önizleme (MJPEG)

GET    /live/lines/{id}                   operator+  SSE sayaç akışı
POST   /sessions                          operator+  oturum aç
POST   /sessions/{id}/close               operator+
GET    /sessions/{id}/events              operator+  ledger dökümü
POST   /sessions/{id}/submit              operator+  onayla ve gönder

GET    /reconciliations                   engineer+
POST   /reconciliations/{id}/resolve      engineer+

POST   /datasets/extract                  engineer+  job
POST   /datasets/synthesize               engineer+  job
POST   /datasets/build                    engineer+  job
GET    /datasets                          engineer+
POST   /cvat/tasks                        engineer+
POST   /cvat/tasks/{id}/pull              engineer+

POST   /training/runs                     engineer+  job
GET    /training/runs/{id}/log            engineer+  WebSocket
POST   /models/{id}/export                engineer+  job
POST   /models/{id}/evaluate              engineer+  job
POST   /models/{id}/stage                 engineer+  shadow/active/retired

GET    /configs/{line_id}                 engineer+
POST   /configs/{line_id}                 engineer+  yeni sürüm
POST   /calibrations/{line_id}/motion     engineer+  job
POST   /calibrations/{line_id}/scale      engineer+  job
POST   /bundles/activate                  engineer+

GET    /system/health, /system/jobs        operator+
GET    /system/outbox                      engineer+
POST   /system/jobs/{id}/cancel            engineer+
```

Tüm uzun işlemler job döndürür (`202` + `job_id`), senkron beklemez.

---

## 9. Arayüz

**İlke: işlevi yüksek, kullanımı basit.** Kullanıcı vardiya sorumlusudur, veri bilimci
değildir.

### 9.1 Tasarım kuralları

- Her ekranda tek birincil eylem
- ML jargonu görünmez: "epoch" değil "eğitim turu", "mAP" değil "model puanı"
- Gelişmiş ayarlar katlanmış `Gelişmiş` başlığı altında
- Uzun işlemler job'dır: başlat, kapat, sonra dön
- Tehlikeli eylemler (model aktifleştirme, oturum gönderme, mutabakat kapatma) onay ister
- Türkçe birincil dil; `site.locale` alanı ve i18n altyapısı baştan kurulur

### 9.2 Persona ayrımı (v2 eklendi)

Giriş yapan rolün gördüğü menü farklıdır:

- **`operator` görünümü:** yalnızca Canlı, Oturumlar, Sistem (salt okunur).
  Üç menü öğesi, sade.
- **`engineer` / `admin` görünümü:** tüm ekranlar.

Aynı arayüz, farklı yüzey. Operatörün eğitim ekranını hiç görmemesi bilinçli bir
sadeleştirmedir.

### 9.3 Ekranlar

| Ekran | Rol | Birincil eylem | İçerik |
|---|---|---|---|
| Canlı | operator+ | — | Hat sayaçları, maske overlay, oturum durumu, sağlık |
| Oturumlar | operator+ | Gönder | Liste, detay, ledger, gönderim öncesi onay |
| Mutabakat | engineer+ | Karara bağla | Açık kayıtlar, kanıt, çözüm |
| Kurulum | admin | Sihirbazı sürdür | §9.4 |
| Veri | engineer+ | Veri topla | Kayıt, kare çıkarma, sentetik, dataset sürümleri |
| Etiketleme | engineer+ | CVAT'ta aç | Görev oluştur, ilerleme, sonucu çek |
| Eğitim | engineer+ | Eğitimi başlat | Basit form + canlı log, run listesi |
| Model | engineer+ | Aktifleştir | Sürümler, puanlar, gölge mod, geri alma |
| Ayar | engineer+ | Kaydet | ROI/gate çizimi, eşikler, config geçmişi |
| Sistem | operator+ | — | Kameralar, job kuyruğu, GPU, outbox, düşen kare oranı |

### 9.4 Kurulum sihirbazı

Teknik olmayan bir kurulumcunun baştan sona tamamlayabileceği akış:

1. Tesis oluştur
2. Hat ekle
3. Kamera ekle → sürücü seç, bağlantıyı test et, canlı önizleme
4. ROI ve sayım kapısını **fare ile çiz** (canlı görüntü üzerinde canvas)
5. **Hareket kalibrasyonu** → bant yönü ve hızı otomatik ölçülür (modelsiz)
6. Ürün profili tanımla → şablon görselleri, ağırlık, boyut, baskı örnekleri
7. Veri topla → belirtilen süre kayıt alınır
8. Etiketleme görevi oluştur → CVAT'a gider
9. Sentetik üret + adapte et → temel modelden fine-tune (tek düğme, arkada job)
10. **Ölçek kalibrasyonu** → seyrek akışlı video üzerinde otomatik (model sonrası)
11. Doğrula → replay skor tablosu, `hard_holdout` sonuçları
12. Gölge mod → sistem sayar, karar vermez
13. Devreye al → `deployment_bundle` oluşur

**Bu sihirbaz varsa ürün vardır.** Yoksa her kurulum geliştiricinin sahada olmasını
gerektirir.

> Adım 5 ve 10'un ayrı olması §5.3'teki bootstrap zorunluluğundandır — ölçek
> kalibrasyonu çalışan bir model gerektirir.

### 9.5 Etiketleme konusunda bilinçli sapma

Maske çizim arayüzü sıfırdan yazılmaz. CVAT MIT lisanslıdır ve olgundur. Kendi
arayüzümüzden görev oluşturulur, CVAT'a yönlendirilir, sonuç REST API ile çekilir.

"Tek arayüz" ilkesinden tek sapma budur. Kendi maske editörünü yazmak aylık bir iştir
ve hiçbir şey kazandırmaz.

### 9.6 Canlı veri

- Sayaçlar: SSE
- Eğitim logları: WebSocket
- Diğer her şey: TanStack Query polling

---

## 10. GPU paylaşım politikası (v2 — düzeltildi)

> **v1 hatası:** "Aktif oturum varken ağır job başlatma" kuralı, 7/24 çalışan bir
> tesiste eğitimin **hiç başlamaması** anlamına geliyordu.

Üç modlu politika, `config_version`'da hat bazında ayarlanır:

| Mod | Davranış |
|---|---|
| `strict` | Aktif oturum varken `requires_gpu` job'ı başlamaz (varsayılan) |
| `window` | Tanımlı bakım penceresinde çalışır (örn. 02:00–05:00, hafta içi) |
| `always` | Her zaman çalışır; çıkarım önceliklidir |

**Ek güvenlik:** Mod ne olursa olsun, eğitim job'ı çalışırken `inference`'ın p95
gecikmesi eşiği aşarsa job **otomatik duraklatılır** ve arayüzde bildirilir. Sayım
doğruluğu her zaman eğitimden önceliklidir.

`window` modu için `maintenance_window` alanı `line` tablosunda tutulur. Kesintisiz
çalışan tesislerde `always` + otomatik duraklatma kombinasyonu önerilir.

---

## 11. Kilometre taşları

Her kilometre taşının bir kabul kapısı vardır. Kapı geçilmeden sonrakine tam kapasiteyle
geçilmez. Efor tahminleri kabadır ve tek geliştirici varsayar.

### M0 — İskelet · ~1 hafta

- Monorepo, `pyproject.toml`, Docker Compose (postgres + api + web)
- Alembic ile §5'teki şemanın tamamı (`camera_epoch`, `reconciliation`, job kirası dahil)
- `cs_core` domain modelleri, `Frame` sınıfı, **beş sürücü arayüzü** (uygulamalar boş)
- FastAPI kabuğu + React kabuğu + rol tabanlı kimlik doğrulama (§8)
- **`tools/license_check.py`** — Python (`pip-licenses`) **ve npm (`license-checker`)**
  taraması; Docker temel imajları elle listeli. İhlalde build fail.
- **`tools/hardcode_check.py`** — §1'deki yasak sabitleri tarar
- `THIRD_PARTY_NOTICES.md`, `docs/provenance.md`, `docs/annotation_guide.md` (v1)

**Kapı:** Boş sistem ayağa kalkıyor, CI iki kapıyı da geçiyor, arayüzden
tesis/hat/kamera kaydı oluşturulabiliyor, üç rol farklı menü görüyor.

### M1 — Veri boru hattı · ~2–3 hafta

- `extract_frames.py` (PyAV) + SSIM tabanlı benzer kare eleme
- **`synth.py`** — amodal maskeli, kiremitleme örtüşmeli sentetik üreteç (§6.5)
- `split_dataset.py` — video/oturum bazlı bölme + **`hard_holdout`** (§7.3)
- CVAT entegrasyonu: görev push, sonuç pull, COCO dönüşümü, iki sınıf (§6.4)
- `dataset_version` kaydı, manifest hash'i, anotasyon kılavuzu sürümü
- Etiketleme tutarlılık ölçümü (iki etiketçi IoU uyumu)
- Arayüz: Veri + Etiketleme ekranları

**Kapı:** Arayüzden video yüklenip 500 karelik amodal etiketli set + sentetik set
üretilebiliyor; split bağımsızlığı ve `hard_holdout` ayrılığı otomatik doğrulanıyor;
etiketçi uyumu ≥ 0.85.

### M2 — Eğitim, arayüzden · ~2 hafta

- RF-DETR Seg eğitim job'ı, `run_kind` ayrımı (`base` / `site_adaptation`)
- Canlı log akışı (WebSocket), job kirası ve kalp atışı
- ONNX export + **denklik testi**: aynı 50 girdi için PyTorch ve ONNX çıktıları
  maske IoU ≥ 0.99 ve skor farkı ≤ 1e-3
- `model_version` kaydı, aşama yönetimi
- **`calibrate_scale` job'ı** (§5.3 aşama 2)
- GPU paylaşım politikası (§10)
- Arayüz: Eğitim + Model ekranları

**Kapı:** Arayüzden tıklayarak eğitilen bir model ONNX olarak registry'de kayıtlı;
ölçek kalibrasyonu otomatik üretilmiş; denklik testi geçiyor.

### M3 — Sayım çekirdeği (offline) · ~3 hafta

- `BeltMotionModel` + hareket kalibrasyonu
- ByteTrack (vendored) + maske-IoU maliyet matrisi + `crossing_seq`
- **`merge_detector.py`** (§6.7)
- `gate.py` PRE/GATE/POST durum makinesi
- `area_counter.py`
- Tamamen offline: girdi video dosyası, çıktı sayı ve olay listesi

**Kapı:** Tek video dosyasında gold-standard sayımla karşılaştırılabilir sonuç;
kaynaşma tespiti loglardan doğrulanabiliyor; kalibrasyon yokken modüller güvenli
şekilde devre dışı kalıyor.

### M4 — Replay harness · ~1–2 hafta

- Senaryo kütüphanesi: yoğun kiremitleme, seyrek akış, dur-kalk, geri kayma,
  ışık değişimi, yeniden başlatma, kare düşme, tam oturum
- `tools/replay.py <bundle> <senaryo_seti>` → tek komut, tek skor tablosu (§7.4)
- Model karşılaştırma ekranı (iki sürüm yan yana)
- **`mine_hard_frames`** job'ı (§6.10)
- Her commit bu tabloyu üretir; nightly CI'da koşar

**Kapı:** Bir model değişikliğinin iyileştirme mi gerileme mi olduğu tek tabloyla
cevaplanabiliyor; zor kare madenciliği CVAT'a görev üretebiliyor.

> Projenin en değerli tek yazılım varlığıdır. Atlanırsa her model değişikliği kumar olur.

### M5 — Canlı pipeline · ~3 hafta

- `supervisor` — `camera` tablosundan işçi türetimi, çalışma zamanında ekleme/çıkarma
- `ingest` × N + **`SharedMemoryTransport`** (§4.5), kalıcı epoch (§5.2)
- `inference` — batch penceresi, kare düşürme politikası, düşen kare sayaçları
- Ledger yazımı, `crossing_seq`, idempotency kısıtı
- Sağlık durum makinesi: `ÇALIŞIYOR → DEGRADED → MANUEL MOD → MUTABAKAT GEREKLİ`
- Kademeli benchmark: 1 → 3 → N kamera; decode FPS, çıkarım FPS, p50/p95/p99 gecikme,
  düşen kare, VRAM, ID switch, sayım hatası
- Arayüz: Canlı + Sistem ekranları

**Kapı:** N kamera canlı çalışıyor; kamera koparıldığında sistem sessizce devam etmiyor,
`degraded`'a geçiyor; `ingest` yeniden başlatıldığında epoch artıyor ve ledger çakışması
olmuyor; ardışık kare düşmesi `degraded` tetikliyor.

### M6 — Oturum, ayar, kurulum sihirbazı · ~3 hafta

- Oturum yaşam döngüsü, kilitleme, `counted_total` türetimi (§5.5)
- **Mutabakat akışı** (§5.7) + Mutabakat ekranı
- Hedef girişi, sarı/kırmızı eşikler
- ROI/gate çizim arayüzü (canvas)
- `config_version` sürümleme, `payload_schema_version` geriye dönük okuma
- **Kurulum sihirbazı** (§9.4)
- `deployment_bundle` üretimi, aktifleştirme akışı, her `count_event`'e yazılması

**Kapı:** Sıfırdan bir hat yalnızca arayüz kullanılarak kurulabiliyor, terminal
gerekmedi; bir mutabakat kaydı açılıp kapatılabiliyor.

### M7 — ERP · ~2 hafta

- Transactional Outbox: oturum kilidi ve gönderim kaydı **aynı transaction'da**
- `erp_relay` — exponential backoff'lu tüketici
- `ErpAdapter`: CSV ve SAP OData uygulamaları
- **İdempotency:** timeout'ta hemen yeniden deneme yapılmaz. `supports_status_query`
  true ise önce ERP'den kaydın durumu okunur, gerçekleşmişse mutabakata alınır.
  False ise (örn. CSV) olay `reconcile_required` durumuna alınır ve insan kararına
  bırakılır — kör yeniden deneme yapılmaz.
  Hedef: "exactly-once delivery" değil, **"at-least-once + iş seviyesinde mutabakat"**
- Gönderim öncesi onay ekranı (gönderim sonrası düzeltme ERP'de ters kayıt gerektirir)

**Kapı:** "Ağ zaman aşımı + yeniden deneme" senaryosunda sıfır çift kayıt;
`supports_status_query = false` adaptörde olay mutabakata düşüyor.

---

## 12. Kodlama standartları

### 12.1 Genel

- Tip ipuçları zorunlu; `mypy --strict` `packages/` üzerinde
- `ruff` (lint + format)
- Her public fonksiyonda docstring
- Sihirli sayı yok — tüm eşikler `config_version`'dan
- Log: yapılandırılmış JSON; her satırda `line_id`, `session_id`, `bundle_id`,
  varsa `camera_id`, `stream_epoch`, `frame_index`

### 12.2 Test

- `cs_counting` ve `cs_tracking` için **%80 üstü kapsam**
- Sentetik videolarla entegrasyon testleri (bilinen doğru sayı)
- Ledger idempotency testi: aynı olayı iki kez yazmayı dene, tek satır kalmalı
- Geri kayma testi: ileri → geri → ileri senaryosunda net sayım 1 olmalı
- Epoch testi: `ingest` yeniden başlat, ledger çakışması olmamalı

### 12.3 CI kapıları

1. `ruff` + `mypy`
2. `pytest`
3. **`license_check.py`** — `pip-licenses` + `license-checker` (npm) + Docker imaj
   listesi; izin verilmeyen tek bir lisans build'i düşürür
4. **`hardcode_check.py`** — çekirdek kodda kamera sayısı, çözünürlük, marka adı,
   bant hızı, sabit eşik arar
5. Nightly: `replay.py` regresyon skor tablosu; belirlenen eşiğin altına düşerse uyarı

---

## 13. Bağımlılık akışı

```
M0 ─▶ M1 ─▶ M2 ─▶ M3 ─▶ M4 ─▶ M5 ─▶ M6 ─▶ M7
             │      │
             │      └─ M3, M2'nin ölçek kalibrasyonuna bağlıdır (§5.3)
             └─ M2'nin site_adaptation'ı M1'in dataset'ine bağlıdır
```

**M3 ve M4, M5'ten önce gelir:** Sayım mantığını canlı RTSP karmaşasına bulaşmadan,
tekrarlanabilir dosyalar üzerinde doğru yapmak çok daha ucuzdur.

---

## 14. Sahadan beklenen girdiler

Kod bunları bekler, varsayım yapmaz. Kod yazmayı engellemezler ama ilgili kilometre
taşından önce gereklidirler.

| Girdi | Kullanıldığı yer | Gerekli olduğu an |
|---|---|---|
| **Örtüşme oranı dağılımı** (gerçek videodan ölçülecek) | `synth.py` parametreleri | M1 |
| Ürün şablon görselleri + baskı örnekleri | `product_profile` | M1 |
| Boş bant görüntüleri | Sentetik arka plan | M1 |
| Seyrek akışlı (örtüşmesiz) referans video | Ölçek kalibrasyonu | M2 |
| Gold-standard elle sayım | Kabul kriteri | M3 |
| Kamera topolojisi (hat–kamera eşlemesi) | `camera.line_id` | M5 |
| ERP sevkiyat süreci, hareket tipi, malzeme kodu | `ErpAdapter` | M7 |

**Örtüşme dağılımı notu:** Yanlış dağılımla üretilen sentetik veri, üretilmemesinden
kötüdür — modeli yanlış yöne çeker. Ölçüm bir saatlik iştir, atlanmamalıdır.

**Gold-standard notu:** Tek kişinin elle saydığı rakam "gerçek doğru" kabul edilmez.
İki bağımsız sayım; fark varsa kare-kare video incelemesiyle uzlaşma. Aksi halde model
doğru sayarken insan hatası "model hatası" gibi görünür.

---

## 15. Ölçülmeden kesinleşmeyecekler

Bu plandaki hiçbir performans rakamı gerçek donanımda ölçülmeden kesin kabul edilmez:

- Model boyutu (Nano / Small / Medium) → M2 sonrası eval
- Tek düğümün kaç kamera taşıdığı → M5 kademeli benchmark
- ONNX Runtime yeterliliği → M5; yetmezse TensorRT teknik gerekçeyle eklenir (§2.5)
- `ring_slots`, `batch_wait_ms`, `max_consecutive_drops` → M5 yük testi
- `merge_area_ratio` ve diğer merge eşikleri → M3/M4 replay ayarı
- Doğruluk hedefi → M4 replay skorları + `hard_holdout` + gold-standard

---

## 16. v1'den v2'ye değişenler

**Kritik hatalar (kod yazılsaydı bozulacaktı):**

1. `stream_epoch` kalıcı hale getirildi — `camera_epoch` tablosu (§5.2). v1'de process
   belleğindeydi, yeniden başlatmada ledger çakışması ve sessiz olay kaybı üretiyordu.
2. `crossing_seq` eklendi (§5.5). v1 kısıtı geri kayan çuvalın ikinci geçişini
   reddediyor, **eksik sayıma** yol açıyordu.
3. Bootstrap kısır döngüsü çözüldü — kalibrasyon `motion` / `scale` olarak ikiye
   ayrıldı (§5.3), sihirbaz adımları buna göre yeniden sıralandı (§9.4).
4. Kare taşıma katmanı tasarlandı (§4.5) — `Frame` şeması, halka arabellek, kare
   düşürme politikası, düşme kaynaklı `degraded` kuralı.
5. Zaman semantiği netleşti (§4.5) — `frame_index` + `monotonic_ns` esas; ledger'a
   `frame_index` eklendi.
6. `print_mark` sınıfı baştan sona tanımlandı (§6.4). v1'de `merge_detector` var olmayan
   bir çıktıya bağlıydı.
7. Anotasyon kuralı belirlendi: **amodal** (§6.3), gerekçesi ve kılavuzuyla.
8. NVIDIA lisans gerekçesi düzeltildi (§2.5) — CUDA kaçınılmaz olduğu için v1'in
   tutarsız duruşu terk edildi.
9. Lisans taraması npm ve Docker imajlarını kapsayacak şekilde genişletildi (§12.3).

**Belirsizlikler:**

10. `counted_total` ledger'dan türetiliyor (§5.5)
11. Mutabakat akışı ve tablosu tanımlandı (§5.7)
12. Job kirası ve kalp atışı eklendi (§5.8)
13. Roller ve endpoint yüzeyi yazıldı (§8)
14. `payload_schema_version` eklendi (§5.4)
15. GPU politikası üç moda ayrıldı, 7/24 tesis sorunu çözüldü (§10)
16. `supports_status_query` eklendi, desteklemeyen adaptörde davranış tanımlandı (§4.4, M7)

**Eksikler:**

17. Aktif öğrenme / zor kare madenciliği geri eklendi (§6.10)
18. Temel model / tesis adaptasyonu ayrımı strateji olarak yazıldı (§7.1–7.2)
19. `hard_holdout` tanımlandı (§7.3)

**Küçükler:** efor tahminleri, persona ayrımı (§9.2), MPL-2.0 çıkarıldı (§2.1), sürüm
sabitleme (§2.8), ONNX denklik toleransı (M2), `site.locale` (§5.1).
