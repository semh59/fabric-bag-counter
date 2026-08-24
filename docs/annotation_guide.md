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
