# Çuval Etiketleme Kılavuzu (Amodal Annotation Guide) — Sürüm 2.0

## 1. Temel İlke: Amodal Etiketleme
Konveyör bandında çuvallar kenarlarından birbirinin üzerine bindiğinde (kiremitleme / shingling), **tahmini tam çuval gövdesi** etiketlenir.
Üstüne binen komşu çuvalın kapattığı alan kesilmez; çuvalın görünen kenarları ve simetrisi takip edilerek tam kontur çizilir.

### Gerekçe
- `merge_detector` ve `area_counter` gibi hacim ve alan korunumuna dayalı modüller, tutarlı bir "tek çuval alanı" (nominal area) referansına ihtiyaç duyar.
- Yalnızca görünen (modal) alan etiketlenirse, örtüşme oranı değiştikçe maske alanı küçülür ve alan integrali tutarsızlaşır.

## 2. Sınıf Tanımları

### Sınıf 1: `bag_body` (Instance Segmentation Poligonu)
- Çuvalın dış hatlarını belirleyen çokgen maske.
- Kapatılan kenarlar makul şekilde uzatılarak tam gövde tamamlanır.
- **Nitelikler:**
  - `visible_ratio`: `0.0` ile `1.0` arasında yaklaşık görünür alan oranı.
  - `heavily_occluded`: Görünürlük %25'in altına düşmüşse `true`.
  - `completely_occluded`: Tamamen görünmeyen çuval **etiketlenmez**.

### Sınıf 2: `print_mark` (Bounding Box)
- Çuval üzerindeki sabit logo, marka, ürün metni veya barkod alanı.
- Yalnızca kameraya bakan yüzeyde net görünüyorsa işaretlenir.
- Çuval ters veya yan geçtiğinde baskı görünmüyorsa **etiketlenmez**.
- `print_mark` ikincil bir doğrulama sinyalidir (özellikle `merge_detector` sinyal 4 için), asla birincil sayım nesnesi değildir.

## 3. Tutarlılık ve Kalite Kontrolü
1. İlk 100 karede en az iki etiketçi bağımsız olarak çalışır.
2. Maske IoU uyumu (`inter_annotator_iou`) hesaplanır.
3. Ortalama uyum **≥ 0.85** olmalıdır. Düşükse kılavuz üzerinde mutabakat sağlanır ve tekrar edilir.
4. Her veri seti sürümü `dataset_version.annotation_guide_version = "2.0"` kaydını taşır.

## 4. Etiketlenen Veriyi Eğitime Bağlama

Gerçek kamera görüntüsü şu adımlarla modele girer (sentetik ön-eğitim → gerçek veriyle fine-tune, §6.5):

1. **Video → kare:** `packages/cs_data/extract_frames.py::extract_video_frames(video_path, output_dir="data/real_bags/images")` ile videodan kareler çıkarılır.
2. **CVAT görevi oluşturma:** Görevi CVAT arayüzünden elle kurmak yerine (etiket adı/tipini yanlış girme riski taşır) `CvatClient.create_task(name, project_id=None)` çağrılır — bu, §2'deki `bag_body`/`print_mark` etiket şemasını (`CvatClient.get_labels_spec()`) doğrudan CVAT'ın REST API'sine gönderir, böylece görev her zaman bu kılavuzla birebir eşleşen şemayla açılır.
3. **Kare yükleme:** `create_task()` yalnızca boş görev kabuğunu oluşturur -- kareleri göreve koymaz. `CvatClient.upload_task_data(task_id, image_paths, image_quality=70)` adım 1'de çıkarılan kareleri gerçek `multipart/form-data` isteğiyle CVAT'a yükler.
4. **Etiketleme:** CVAT arayüzünde açılan görev, bu kılavuzdaki `bag_body` (poligon) / `print_mark` (dikdörtgen) kurallarına göre elle etiketlenir. İlk 100 kare için §3'teki iki-etiketçi/IoU-uyum kontrolü uygulanır (`CvatClient.calculate_inter_annotator_agreement`).
5. **Dışa aktarma:** Tamamlanan görev ham COCO formatında dışa aktarılır (`packages/cs_data/cvat_client.py::CvatClient.parse_coco_annotations` bu formatı okur).
6. **Yerleşim:** Dışa aktarılan `annotations.json` dosyası `data/real_bags/annotations.json` olarak, görüntüler `data/real_bags/images/<file_name>` olarak yerleştirilir (COCO `images[].file_name` alanıyla eşleşmeli).
7. **Eğitim:** `packages/cs_vision/train_rfdetr.py::train_and_export_model()` bu klasörü otomatik bulur, gerçek kareleri sentetik sahnelerle karıştırıp (varsayılan 3x oversample) eğitir. Klasör/`annotations.json` yoksa sessizce sadece sentetik veriyle eğitime devam eder — hata vermez. `bag_body` kutusunun merkezine düşen bir `print_mark` etiketi varsa o kutunun sınıflandırma hedefi otomatik `1` (baskı işareti var) olarak türetilir; aksi halde `0`.

### CVAT bağlantı ayarları
`CvatClient(base_url=..., auth_token=...)` -- `base_url` öntanımlı olarak `http://localhost:8080/api`'dir (yerel/tek-makine CVAT kurulumu varsayımı); ayrı bir CVAT sunucusu kullanılıyorsa gerçek adres ve CVAT hesap tokenı (CVAT arayüzü → hesap ayarları → API token) ile geçilir. `create_task`/`upload_task_data` her ikisi de `Authorization: Token <auth_token>` başlığıyla kimlik doğrular; token verilmezse istek kimliksiz gönderilir (CVAT sunucu ayarına göre reddedilebilir).
