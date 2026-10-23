# Plaka quadrilateral Detection teknik uygulama roadmap'i

Durum: **uygulama devam ediyor; canlı sistem değişmedi**. Public mühendislik iskeleti ve özel,
review-only annotation pilotu hazırlanmıştır. Training, model seçimi, deployment ve doğruluk iddiası
henüz yapılmamıştır.

Referans: `docs/plans/plate-quadrilateral-detection-roadmap.md`. Bu dosya referans roadmap'in yerine geçmez; mevcut repository akışına göre uygulanabilir iş paketlerini, doğrulama kapılarını ve geriye dönük uyumluluk stratejisini ayrıntılandırır.

## 1. Amaç, kapsam ve varsayımlar

Amaç, plakanın eksenlere paralel bir kutusunu değil, perspektif etkisini temsil eden görünür dört dış köşesini doğrudan project-owned Detection modeliyle tahmin etmek; geçerli dörtgeni orijinal görüntü koordinatlarına taşımak; plakayı homografiyle doğrultmak ve mevcut Recognition modelini değiştirmeden bu görüntüyle beslemektir.

Hedef akış:

```text
EXIF yönü düzeltilmiş görüntü
  -> letterbox
  -> project-owned Detection v2
       -> objectness + xyxy kutu + 4 sıralı köşe
  -> kutu tabanlı NMS; aynı adayın köşelerini koru
  -> inverse-letterbox ile orijinal görüntü koordinatları
  -> quadrilateral kalite/geometri kapısı
       -> geçerli: perspektif rectification -> mevcut Recognition
       -> geçersiz: açık UNRESOLVED/manual-review sonucu
  -> sürümlü sonuç sözleşmesi ve görselleştirme
```

Varsayımlar:

- İlk sürüm tek sınıflı plaka Detection görevini ve mevcut `YoloXMultiScaleDetector` omurga/FPN-PAN yapısını korur.
- Recognition ağırlıkları, decoder ve aspect-preserving giriş sözleşmesi bu çalışma boyunca dondurulur. Böylece OCR farkının crop değişiminden geldiği ölçülebilir.
- Mevcut yetkili `detection-v3` görüntüleri ana domain verisidir: 7.524 train ve 837
  validation örneği vardır. Canonical 8.361 label'ın tamamı beş alanlı YOLO bbox formatındadır;
  gerçek dış plaka köşesi yoktur. Bu nedenle re-annotation yolu seçilmiştir.
- Dört köşe sırası, EXIF yönü düzeltilmiş orijinal görüntüde `top-left, top-right, bottom-right, bottom-left` olacaktır.
- Eforlar mühendislik için kişi-gün aralığıdır; annotation süresi, annotation tedariki, model training süresi, GPU bekleme süresi, güvenlik/hukuk incelemesi ve rollout gözlem süresi hariçtir.
- Annotation SLA'i, hedef CPU/GPU, latency/memory bütçesi, ikinci annotator kapasitesi ve sayısal
  model kabul eşikleri halen `TBD`'dir.

Kapsam dışı:

- Recognition mimarisini, karakter kümesini veya decoder'ı değiştirmek.
- Harici pretrained Detection/segmentation/OCR ağırlığı eklemek.
- Yetkisiz görüntü toplamak veya özel veriyi public repository'ye taşımak.
- Dört köşenin yeterli olduğu kanıtlanmadan pixel-level segmentation'a geçmek.
- Development-validation sonucunu bağımsız saha doğruluğu gibi sunmak.

## 2. Mevcut sistem ve tespit edilen sorunlar

### 2.1 Mevcut Detection ve training akışı

- `src/anpr_engine/detection/model.py`, stride 8/16/32 özelliklerini kullanan tek sınıflı, anchor-free, üç seviyeli YOLOX-benzeri bir model içerir. Her hücre yalnızca bir objectness logiti ve dört kutu regresyon değeri üretir.
- Çok ölçekli loss, dinamik pozitif atama, focal-style objectness ve GIoU kutu loss'u kullanır. Pozitif atama kutu geometrisine bağlıdır.
- `src/anpr_engine/detection/data.py`, manifestteki her örnek için tam olarak bir class-0 YOLO kutusu bekler. Boş/no-plate label, birden fazla plaka veya köşe label'ı mevcut veri sözleşmesinde yoktur.
- Augmentation yalnız parlaklık, kontrast ve yatay flip uygular. Yatay flip kutuyu günceller; köşe kimliği veya geometrik transform altyapısı bulunmaz.
- Dataset/config/checkpoint kimliği SHA-256 ile doğrulanır; training run manifesti, seed, environment ve checkpoint kimliği kaydedilir. Bu tekrar üretilebilirlik mekanizması korunmalıdır.
- Mevcut checkpoint `format_version: 1.0` ve `architecture: yolox_multiscale_v1` kimliğine bağlıdır. Loader yalnız mevcut iki mimariyi tanır.

### 2.2 Mevcut inference ve entegrasyon akışı

- Bağımsız inference yolu kutuları decode eder, objectness eşiği ve kutu NMS uygular, inverse-letterbox ile `xyxy` koordinatını orijinal görüntüye taşır ve eksenlere paralel crop üretir.
- `AnprV1Pipeline`, 384 x 384 Detection girişi, `0.15` confidence ve `0.30` NMS eşiğiyle adayları sıralar; en yüksek skorlu kutuyu seçer.
- Seçilen kutu sıfır padding ile kırpılır. Ardından `BLUE_BAND_GLYPH_QUAD_V1`, glyph ve mavi bant tabanlı deterministik bir görüntü işleme heuristic'i dener. Bu aşama gerçek model köşe tahmini değildir ve kalite kapısı geçilmezse kutu crop'una geri döner.
- Upload yolu PIL ile EXIF orientation düzeltir; bağımsız `detect_images` yolu OpenCV ile doğrudan decode eder. Koordinat sözleşmesi açısından bu iki yolun davranışı açıkça birleştirilmelidir.
- Public ve private şemalar kutu ve mevcut crop policy'sine bağlıdır. UI, review/export ve fixture'lar da kutu alanlarını tüketir.

### 2.3 Temel sorunlar

1. **Geometri kaybı:** `xyxy` kutu perspektif plakayı, çevre gövdeyi veya plate holder'ı kapsayabilir; gerçek plaka düzlemini ifade etmez.
2. **Dolaylı refinement:** Heuristic, Detection çıktısından değil crop içi karakter/mavi bant sinyallerinden yararlanır; bulanıklık, glare, düşük çözünürlük, farklı plaka görünümü ve kısmi görünürlükte fallback üretir.
3. **Training-label uyumsuzluğu:** Mevcut YOLO kutuları dört köşe için ground truth değildir. Kutunun dört köşesini quad label olarak kullanmak mevcut hatayı modele öğretir ve yasaklanmalıdır.
4. **Negatif ve çoklu örnek sınırı:** Loader her görüntüde tek plaka varsayar. No-plate false-positive davranışı ve çoklu plaka training'i mevcut sözleşmeyle ölçülemez.
5. **Geometrik augmentation eksikliği:** Perspective/affine/rotation/crop dönüşümleri ve köşe yeniden sıralama altyapısı yoktur.
6. **Metrik boşluğu:** Mevcut ölçümler kutu precision/recall/mAP/IoU ile sınırlıdır; polygon IoU, corner error, valid-quad, rectification ve uçtan uca Recognition etkisi ölçülmez.
7. **Çalışma yolu sapması:** Standalone inference config varsayılanları ile donmuş ANPR runtime eşikleri aynı değildir; EXIF davranışı da farklıdır. Tek bir sürümlü inference policy tanımlanmalıdır.
8. **Sözleşme bağımlılığı:** API, web UI, review store, export, test fixture'ları ve model bundle kimliği mevcut `bbox_xyxy` ve `BLUE_BAND_GLYPH_QUAD_V1` anlamına bağlıdır. Alan anlamını sessizce değiştirmek geriye dönük uyumsuzluk yaratır.

## 3. Hedef model mimarisi

### 3.1 Önerilen MVP: `yolox_multiscale_quad_v2`

İlk aday, mevcut backbone, FPN/PAN ve üç decoupled scale head'i korur. Her grid adayı için çıkışlar:

| Çıkış | Kanal | Açıklama |
| --- | ---: | --- |
| Objectness | 1 | Mevcut tek sınıflı aday skoru |
| Box regression | 4 | Mevcut center/width/height encoding ve türetilmiş `xyxy` |
| Corner regression | 8 | Sıralı dört köşe için `(x, y)` çiftleri |

MVP toplamı seviye başına 13 kanaldır. Köşe branch'i box branch'inden ayrılmalı veya en azından son predictor katmanı ayrı olmalıdır; böylece gradient ve loss bileşenleri bağımsız izlenebilir. Mevcut objectness/box ağırlıkları isim ve shape uyduğu ölçüde donmuş v1 checkpoint'inden yüklenir; yeni corner predictor rastgele ve deterministik seed ile initialize edilir. Bu işlem yeni bir checkpoint üretir, eski checkpoint'i değiştirmez.

Önerilen corner encoding:

- Her köşe, eşleşen target box merkezine ve genişlik/yüksekliğine göre normalize offset olarak encode edilir.
- Training target'ı ground-truth box ve gerçek corner label'dan; inference decode'u aynı adayın tahmin edilen box'ından üretilir.
- Bu seçim, köşeleri ilgili adayla doğal olarak bağlı tutar ve farklı feature stride'larında aynı anlamı korur.
- Offset aktivasyonu, clamp sınırı ve sayısal stabilizasyon aralığı deney öncesi belirtilecektir: `TBD`.

Alternatif encoding'ler ancak kontrollü ablation ile denenir: grid-cell-relative koordinatlar veya ROI tabanlı ikinci kademe corner refiner. İlk model başarısız olmadan ikinci model ailesine geçilmez.

### 3.2 Loss ve seçim ilkesi

Toplam loss kavramsal olarak:

```text
L_total = L_objectness + lambda_box * L_GIoU + lambda_corner * L_corner
          + opsiyonel lambda_geometry * L_geometry
```

- Pozitif atama ilk deneyde mevcut kutu tabanlı dinamik atamayı korur; böylece box baseline ile karşılaştırma kontrollü olur.
- `L_corner`, yalnız geçerli/annotated köşelerde uygulanır. İlk aday Smooth L1 veya Wing/Huber ailesinden seçilecektir; seçim ve hyperparameter `TBD` ve pilot dağılımı görülmeden sabitlenmeyecektir.
- Kısmi/occluded köşeler için per-corner visibility/uncertainty maskesi annotation sözleşmesine eklenir. Dört güvenilir köşe yoksa örnek box/objectness training'ine katkı verebilir fakat rectification-supervision metriğinin paydasında ayrı raporlanır.
- Optional convexity/order/edge consistency loss yalnız direct regression temelinin hataları analiz edildikten sonra ablation olarak değerlendirilir. Postprocessing kapısının yerini almaz.
- Model seçimi salt en düşük loss ile yapılmaz. Predeclared validation bileşik skoru; box parity, corner accuracy, valid-quad oranı ve frozen-Recognition sonucunu içerir. Formül ve ağırlıklar Phase 0 sonunda `TBD` alanlarından çıkarılıp training başlamadan kilitlenir.

### 3.3 Kontrollü alternatif

Direct head, küçük veya kuvvetli perspektifli plakalarda gerekli köşe hassasiyetini sağlayamazsa tek alternatif deney, Detection feature'larından çalışan hafif project-owned ROI corner refiner'dır. Kabulü için direct head'e göre anlamlı kalite artışı ve hedef host bütçesine uyum birlikte gerekir. Harici model veya ağırlık kullanılmaz.

## 4. Quadrilateral ve corner detection sözleşmesi

### 4.1 Canonical geometri

- Kaynak koordinat sistemi: EXIF yönü düzeltilmiş orijinal görüntü, origin sol üst, `x` sağa ve `y` aşağı.
- Köşe sırası: TL -> TR -> BR -> BL, saat yönünde.
- Annotation noktası: plate holder ve araç gövdesi hariç, plaka yüzeyinin görünür dış sınırı.
- Saklama: loss için normalize koordinatlar; audit/debug için orijinal pixel koordinatına kayıpsız dönüştürülebilir temsil.
- Derived box: `min/max` corner koordinatlarından üretilir; köşe ground truth'unun yerine geçmez.

### 4.2 Geometri kalite kapısı

Bir quad ancak aşağıdakilerin tümünü geçerse warp edilir:

- Bütün değerler finite ve doğru shape/order'dadır.
- Dört nokta in-bounds veya yalnız önceden tanımlanmış küçük tolerans içindedir. Büyük taşmalar sessizce clamp edilmez.
- Polygon self-intersection içermez, convex'tir ve signed area yönü canonical sırayla uyumludur.
- Alan, her edge uzunluğu, karşı edge oranları, iç açılar ve perspective distortion belirlenen aralıktadır: eşikler `TBD`.
- Quad'dan türetilen box ile aday box arasındaki tutarlılık kabul aralığındadır: eşik `TBD`.
- Homography matrix finite, invertible ve condition-number sınırını geçmez: sınır `TBD`.

Kapı başarısızsa v2 sonucu `UNRESOLVED_GEOMETRY`/manual review olarak işaretlenir. Legacy crop'a sessiz fallback yapılmaz. Shadow/rollback sırasında legacy çalıştırılacaksa sonuç açıkça `LEGACY_BOX_HEURISTIC` policy kimliğiyle ayrı tutulur.

### 4.3 Rectification

- Kaynak noktalar inverse-letterbox sonrası, orientation-corrected orijinal görüntü üzerindedir.
- Target genişliği üst/alt edge; target yüksekliği sol/sağ edge uzunluklarından, belgelenmiş `max` veya robust aggregation kuralıyla hesaplanır. Nihai kural `TBD`.
- Minimum/maximum boyut, toplam pixel bütçesi, interpolation ve border mode config ile sürümlenir; sayısal değerler performans ölçümünden sonra `TBD` olarak kapatılır.
- Warp sonucu RGB kanal sırasıyla mevcut `_aspect_fit`/Recognition preprocessing sözleşmesine girer.
- Kaynak quad, derived box, warp boyutu, kalite-kapısı nedeni ve method/version provenance olarak tutulur; yüklenen görüntü veya crop kalıcı log/metric'e yazılmaz.

## 5. Dataset, annotation ve augmentation gereksinimleri

### 5.0 Uygulama kararı ve dış veri politikası

- Türk `detection-v3`, bütün fine-tune ve resmi development-validation karşılaştırmalarının ana
  veri kaynağıdır. Mevcut split üyeliği değişmez.
- Bbox köşeleri hiçbir koşulda quad ground truth yapılmaz.
- İlk annotation pilotu, sabit `20260926` seed'i ve 256 örnek üst sınırıyla; split, A/B/C/D
  cohort, bbox alan quantile'ı ve en-boy oranı quantile'ı üzerinde deterministik round-robin
  örneklenir. Bu sayı eğitim yeterliliği iddiası değil, bounded fizibilite/review kapsamıdır.
- Pixel-only `BLUE_BAND_GLYPH_QUAD_V1` yalnız review önerisi üretir. Bütün öneriler
  `REVIEW_REQUIRED` kalır; öneri bulunmayan kayıtlar sahte köşeyle doldurulmaz.
- CCPD, dört köşe içeren yardımcı bir geometri pretraining kaynağı olarak yalnız ilk Türk pilot
  baseline'ı yetersiz kalırsa değerlendirilecektir. Domain farkı nedeniyle resmi Türk validation
  yerine geçmez. Akademik/ticari-olmayan UFPR/LRLPR kaynakları mevcut ürün akışına alınmaz.
- Pilot kuyruğu ve insan kararları private evidence alanında kalır. Public repository'ye görüntü,
  plaka metni, özel mutlak yol veya record-level özel örnek eklenmez.

### 5.1 Dataset envanteri ve split bütünlüğü

- Mevcut manifestlerin image count, çözünürlük, plate pixel size, perspektif, ışık, blur, glare, occlusion, edge-of-frame ve no-plate dağılımı kimlik ifşa etmeden çıkarılır.
- Aynı fiziksel plaka, capture session, video burst ve near-duplicate görüntüler split'ler arasında sızmayacak grup anahtarıyla ayrılır.
- Mevcut train/validation üyeliği mümkün olduğunca korunur. Annotation-only revision yeni bağımsız örnek gibi sayılmaz.
- Test/holdout varsa training/model-selection için açılmaz. Yoksa bağımsız kabul iddiası yapılmaz ve `TBD: independent holdout` açık kalır.
- No-plate negatif set ve gerçek çoklu-plaka gereksinimi envanterde karara bağlanır. Mevcut loader bunları desteklemediği için ihtiyaç varsa schema v2 kapsamına açıkça eklenir.

### 5.2 Annotation şeması

Önerilen özel manifest/label kaydı en az şunları içerir:

- `sample_id`, split/group kimliği ve image/annotation content hash'i.
- Her plate instance için class, dört ordered normalized corner, derived box.
- `corner_visibility` veya eşdeğer per-corner durum: `VISIBLE`, `OCCLUDED`, `CLIPPED`, `AMBIGUOUS`.
- Instance-level `geometry_usable_for_rectification` ve gerekçe kodu.
- Annotation schema version, revision ve audit durumu.

Format (JSON/JSONL/başka) repository standardına göre Phase 1'de seçilecektir: `TBD`. Eski YOLO text label'ları read-only uyumluluk yolu üzerinden box-only örnek olarak yüklenebilir; bunlar corner loss'a katkı vermez.

### 5.3 Annotation kalite kontrolü

- Ayrıntılı guideline; plate holder, vida/çerçeve, gölge, yansıma, kısmi görünürlük ve belirsiz sınır örneklerini kapsar.
- Otomatik validator: finite/in-bounds, sıra, convexity, self-intersection, minimum area/edge ve derived-box tutarlılığı.
- Pilot örneklerin bir bölümü bağımsız ikinci annotator tarafından işaretlenir. Örnekleme oranı ve kabul edilebilir inter-annotator corner error `TBD`.
- Uyuşmazlıklar adjudication ile çözülür; belirsiz noktalar sahte kesinlikle yerleştirilmez.
- Dataset content hash, annotation revision ve split hash training receipt'e girer.

### 5.4 Synchronized augmentation

Görüntü, box, corner ve visibility aynı transform matrisiyle dönüştürülür:

- Mevcut brightness/contrast korunur; blur, noise, compression, exposure/glare için gerçekçi aralıklar pilotla belirlenir.
- Horizontal flip sonrası köşe koordinatları dönüşür ve kimlikler `TL<->TR`, `BL<->BR` olacak şekilde yeniden sıralanır.
- Küçük rotation, scale, translation, affine ve perspective transform aralıkları `TBD` olarak veri dağılımından türetilir.
- Crop/translate sonrası görünmeyen köşeler maskelenir; geçersiz veya okunamaz target üreten transform reddedilir.
- Vertical flip ve gerçek kullanımda olmayan aşırı perspective varsayılan olarak kapalıdır.
- Transform parametreleri ve RNG seed, hata tekrar üretimi için izlenebilir olmalıdır.

## 6. Training değişiklikleri

1. `DetectorConfig` için yeni architecture/config union eklenir; corner loss ağırlıkları, encoding sürümü, augmentation ve geometry gate parametreleri strict validation alır.
2. Dataset çıktı tipi box-only tensor yerine sürümlü structured target'a taşınır. Legacy tiny/v1 training testleri korunur.
3. Model output için adlandırılmış/typed bir v2 decode contract oluşturulur; `(objectness, boxes, corners)` eşleşmesi flatten ve multi-scale concat sırasında kaybolmaz.
4. Warm-start loader yalnız shape/name uyumlu v1 parametrelerini alır, eksik/fazla key listesini receipt'e yazar; corner branch init'i doğrulanır. V1 checkpoint loader davranışı değişmez.
5. İlk training aşamasında opsiyonel kısa corner-head warm-up, ardından tüm ağın kontrollü fine-tune'u değerlendirilir. Freeze süresi, optimizer, LR, scheduler, batch, epoch ve loss ağırlıkları `TBD`.
6. Loglar total/objectness/box/corner loss, pozitif sayısı, gradient normları, invalid decoded quad oranı ve validation metriklerini ayrı tutar.
7. Early stopping ve resmi candidate selection, Phase 0'da predeclare edilen metriktedir. Seed tekrarı sayısı `TBD`; tek şanslı run release kanıtı sayılmaz.
8. Yeni checkpoint `format_version` ve `architecture` kimliğiyle yazılır. Dataset/config/code state, environment, warm-start source SHA-256 ve çıktı SHA-256 receipt'te bulunur.

## 7. Inference ve postprocessing değişiklikleri

- Decode, her aday için box, objectness ve dört corner üretir. Confidence filtering ve box NMS sonrası seçilen index ile corner tensor aynı index'te tutulur.
- NMS ilk sürümde box IoU kullanır; polygon NMS yalnız çoklu/çakışan aday hataları kanıtlanırsa ablation'dır.
- Standalone inference ile `AnprV1Pipeline` ortak decode/policy fonksiyonlarını kullanmalıdır. Threshold, input size, max detections ve EXIF davranışı sürümlü config'ten gelmeli; hard-coded sapma kalmamalıdır.
- Inverse-letterbox hem box hem corner için tek test edilmiş transform üzerinden yapılır. Pixel ve normalized koordinatlar açıkça isimlendirilir.
- Seçim kuralı başlangıçta mevcut davranışla uyumlu `objectness desc, deterministic tie-break` olarak kalır. Quad validity candidate seçiminden önce mi sonra mı uygulanacağı Phase 4 karşılaştırmasıyla kilitlenir; tercih, en yüksek skorlu **geçerli** quad'ı seçmek fakat aday reddini raporlamaktır.
- Geometri gate ve warp saf/deterministik fonksiyonlara ayrılır. Hata kodları; invalid order, out-of-bounds, degenerate edge/area, excessive distortion ve homography failure'ı ayırır.
- `BLUE_BAND_GLYPH_QUAD_V1` yeni strict v2 yolunda çağrılmaz. V1 runtime ve rollback image'ı onu kullanmaya devam edebilir.
- Debug/annotated artifact'larda box yanında polygon edge ve köşe marker'ları gösterilir; hassas artifact saklama politikası mevcut no-retention sınırına uyar.

## 8. API, UI ve geriye dönük uyumluluk

### 8.1 Sürümleme stratejisi

- Mevcut v1 endpoint/result şemalarının alan anlamı değiştirilmez.
- Yeni result contract sürümü `TBD` adla eklenir ve en az şunları taşır:
  - dört ordered normalized corner ve coordinate-space/version bilgisi,
  - derived legacy box,
  - `boundary_status`, `rectification_status`, method/version ve reason code,
  - Detection/corner confidence semantiği,
  - warp output dimensions ve güvenli provenance.
- Private inference schema, public BFF mapping, generated contract, fixtures, export ve review modelleri birlikte sürümlenir.
- Eski tüketiciler derived `box_xyxy_normalized` alabilir; bunun quad'dan türetildiği dokümante edilir. V1'e yeni alan sokulacaksa yalnız additive ve schema-valid bir değişiklik yapılır; varsayılan tercih ayrı v2 sözleşmesidir.

### 8.2 UI/review değişiklikleri

- Orijinal görüntü üzerinde dört edge ve TL/TR/BR/BL marker'ları; ayrı olarak rectified crop gösterilir.
- UI; `NO_PLATE`, `VALID_QUAD`, `UNRESOLVED_GEOMETRY`, `RECOGNITION_FAILED` durumlarını karıştırmaz.
- Manual review, kutu doğruluğundan ayrı boundary/rectification değerlendirmesi alır. Owner-adjusted box akışı korunur; corner correction eklenmesi bu release için `TBD` ve ayrıca scope kararıdır.
- EN/TR metinleri, accessibility label'ları ve JSON export yeni durumları aynı semantikle taşır.
- Model hash readiness, bir aktif inference, bounded/no queue, timeout/cancellation, upload validation ve no-retention politikaları korunur.

### 8.3 Checkpoint ve runtime uyumluluğu

- `yolox_multiscale_v1` loader ve private frozen bundle değişmeden kalır.
- V2 ayrı architecture ID, checkpoint formatı, registry/model-card kaydı, SHA-256 ve bundle version alır.
- Runtime feature flag/model bundle seçimi process başlangıcında yapılır; request bazında gizli model karışımı yapılmaz.
- Eski ve yeni model aynı artifact adına yazılmaz; eski image + weight bundle hızlı rollback için saklanır.

## 9. Test stratejisi, metrikler ve kabul kriterleri

### 9.1 Test katmanları

**Unit testleri**

- Corner encode/decode round-trip; farklı stride ve aspect ratio.
- Letterbox/inverse-letterbox box ve corner round-trip; odd padding ve non-square görüntü.
- Horizontal flip reorder; affine/perspective transform ile bilinen nokta eşleşmesi.
- Polygon order, convexity, self-intersection, area/edge, out-of-bounds ve homography hata yolları.
- V1/v2 checkpoint yükleme, partial warm-start ve yanlış identity'de fail-closed davranışı.
- NMS sonrası box-corner index eşleşmesi ve deterministic tie-break.

**Dataset/annotation testleri**

- Schema, content hash, split leakage, duplicate/group kontrolü.
- Box-only legacy örnek ile corner-labeled örneğin loss masking davranışı.
- Zero/multiple instance desteği scope'a alınırsa collate/loss/metric testleri.

**Integration/contract testleri**

- Detection -> valid quad -> warp -> frozen Recognition tam yolu.
- Invalid quad'ın Recognition'a girmemesi ve açık unresolved sonucu.
- Public/private schema, generated client, export, review store ve EN/TR UI fixture'ları.
- EXIF orientation, JPEG/PNG, edge-of-frame, no-plate, multiple candidate, timeout ve cancellation.
- Log/trace/metric içinde görüntü, crop, plate text veya özel path bulunmaması.

**Regression/performance testleri**

- V1 checkpoint ve v1 fixture'larının sonuç/sözleşme regresyonu.
- Aynı approved validation girdilerinde legacy ile v2 paired comparison.
- Hedef hostta cold/warm latency, stage latency, CPU, peak RSS/device memory, startup ve artifact overhead.

### 9.2 Ölçülecek metrikler

| Alan | Metrikler |
| --- | --- |
| Box Detection | precision, recall, F1, mAP@0.50, mAP@0.50:0.95, matched IoU, miss/duplicate/background FP; mevcut hesapla parity |
| Corner | mean/median/p90 normalized corner error, per-corner error, PCK@`TBD`, corner visibility slice'ları |
| Polygon | polygon IoU mean/median/p10, IoU eşik geçiş oranları `TBD`, valid-quad rate, invalid reason dağılımı |
| Rectification | homography success, mirrored/flipped rate, plate-boundary coverage, warp artifact/manual quality, abstention rate |
| Recognition | frozen model RAW exact, CER, character accuracy, length accuracy; valid-quad ve tüm-popülasyon sonuçları ayrı |
| Operasyon | p50/p95/max per-stage ve end-to-end latency, cold start, CPU, peak RSS/device memory, timeout/cancel rate |
| Güvenlik/uyumluluk | schema pass, hash mismatch fail-closed, no-retention/log leakage, v1 regression sonuçları |

Corner error normalizasyonu, plate diagonal veya `sqrt(area)` seçeneklerinden annotation pilotundan önce biri seçilerek sabitlenir: `TBD`. Başarısız Detection, invalid quad ve abstention örnekleri paydadan çıkarılmaz; conditional ve end-to-end metrikler birlikte raporlanır. Perspektif, plate pixel size, blur/glare, occlusion, edge-of-frame ve plate-format slice'ları örnek sayısı mahremiyet ve istatistik açısından yeterliyse ayrıca raporlanır.

### 9.3 Kabul kriterleri

Sayısal eşikler Phase 0 baseline ölçümünden sonra ve ilk v2 training sonucuna bakılmadan önce doldurulacaktır:

| Kapı | Kabul kriteri |
| --- | --- |
| Annotation | Otomatik validator %100 pass; audited subset oranı `TBD`; inter-annotator NCE `TBD`; split leakage = 0 |
| Geometri | Deterministik unit/property testleri pass; valid-quad rate >= `TBD`; mirrored/flipped rectification = 0 on curated geometry fixtures |
| Box parity | Recall düşüşü <= `TBD` yüzde puan; FP/mAP regresyon sınırları `TBD` |
| Corner/polygon | NCE p90 <= `TBD`; polygon IoU hedefi >= `TBD`; perspective/small-plate slice tabanları `TBD` |
| Uçtan uca | Frozen-Recognition RAW exact/CER'de predeclared sınırın üzerinde regresyon yok; boundary veya Recognition metriğinde paired iyileşme `TBD` |
| Operasyon | p95 latency, peak memory, CPU ve startup hedef host bütçesi içinde: `TBD` |
| Uyumluluk | V1 loader/route/fixtures pass; v2 schema ve UI pass; hash mismatch/no-retention/fail-closed kontrolleri pass |

Kabul kararı `accept`, `revise` veya `reject` olarak yazılı verilir. Güzel overlay tek başına kabul değildir. Paired bootstrap confidence interval veya eşdeğer istatistiksel yöntem, örnek sayısı yeterliyse kullanılır; yöntem ve confidence düzeyi `TBD`.

## 10. Performans yaklaşımı

- 8-channel corner predictor'ın parameter/FLOP/memory artışı ölçülür; teorik tahmin gerçek hedef-host profiliyle doğrulanır.
- Warp yalnız seçilmiş ve geçerli adayda çalışır; her aday için tam-resolution warp yapılmaz.
- Tensor->CPU kopyaları, OpenCV conversion ve artifact yazımı stage bazında profillenir.
- Input size artırımı varsayılan çözüm değildir. Küçük plaka hatası görülürse 384 baseline'a karşı kalite/maliyet ablation'ı yapılır; candidate seçimi kaynak bütçesini de geçmelidir.
- Mixed precision, TorchScript/compile veya quantization bu çalışmanın prerequisite'i değildir. Ancak hedef bütçe aşılırsa doğruluk regresyon testiyle ayrı optimizasyon fazına alınabilir.
- Bir aktif inference ve mevcut worker sınırı korunur; corner model için ayrı sürekli worker açılmaz.

## 11. Riskler, bağımlılıklar ve rollback

| Risk/bağımlılık | Etki | Azaltım / karar |
| --- | --- | --- |
| Gerçek corner label yok veya kalitesi düşük | Model yanlış hedefi öğrenir; proje durabilir | Annotation pilotu ve bağımsız audit; box corner'larını GT yapmama; Phase 1 go/no-go |
| Veri azlığı / perspective çeşitliliği yetersiz | Overfit ve slice başarısızlığı | Split-group audit, gerçekçi augmentation, seed tekrarları; yeni veri ihtiyacını `TBD` olarak yükseltme |
| Uncertain/occluded köşeler | Gürültülü loss ve invalid warp | Per-corner mask, unusable-for-rectification durumu, slice raporu |
| Box performansı düşer | Plate miss/FP artar | Warm-start, multi-task ağırlık ablation'ı, box parity gate |
| Köşe sırası/letterbox hatası | Mirror, yanlış homografi | Tek canonical transform, round-trip/property testleri, visual golden fixtures |
| Invalid quad sessiz fallback | Yanlış başarı görünümü | Strict v2 unresolved durumu; policy/version provenance |
| OCR iyileşmez veya kötüleşir | Kullanıcı hedefi karşılanmaz | Recognition'ı dondurup paired E2E ölçüm; model reddi/rollback |
| CPU/memory/latency artışı | Shared host kararsızlığı | Tek-pass MVP, stage profiling, bütçe gate'i; ROI alternatifini koşullu tutma |
| API/UI tüketicisi kırılır | Release regresyonu | Ayrı schema sürümü, derived bbox, contract fixtures ve v1 paralel destek |
| Checkpoint yanlış eşleşir | Yanlış model/sonuç | Yeni architecture/format ID, SHA-256, fail-closed startup |
| Özel veri veya artifact sızıntısı | Privacy/release engeli | Mevcut no-retention sınırı, log scan, public-tree scan, private receipt |
| Annotation/training compute erişimi | Takvim belirsizliği | Kapasite ve donanımı Phase 0'da `TBD`'den çıkarma; mühendislik eforundan ayrı planlama |

Rollback planı:

1. Mevcut `yolox_multiscale_v1` checkpoint, Recognition checkpoint, `BLUE_BAND_GLYPH_QUAD_V1` policy ve release image immutable tutulur.
2. V2 ayrı bundle/registry kimliğiyle shadow modda çalışır; live routing değişmez.
3. Switch ancak kabul kapıları ve ayrı kullanıcı onayı sonrası yapılır. Config/bundle switch atomik ve startup hash doğrulamalı olmalıdır.
4. Invalid geometry artışı, Detection/OCR regresyonu, timeout, memory pressure, schema/UI uyumsuzluğu veya hash sorunu rollback tetikler. Sayısal trigger'lar Phase 0'da `TBD` olarak kapatılır.
5. Rollback, önceki image ve bundle'a geri dönerek eski crop policy'sini geri yükler; v2 artifact silme veya v1 checkpoint üzerine yazma gerekmez.
6. Rollback sonrası ilgili candidate devre dışı kalır, olay ve ölçümler hassas veri içermeden kaydedilir; tekrar rollout yeni onay ister.

## 12. Fazlara ayrılmış uygulama planı

### Faz 0 — Baseline, envanter ve karar sözleşmesi

**Amaç:** Mevcut davranışı dondurmak, bilinmeyenleri görünür kılmak ve model sonucuna bakmadan kabul kurallarını belirlemek.

**Yapılacak işler:**

- Dataset/annotation/permission, split ve duplicate envanteri.
- V1 model/config/crop policy/Recognition hash'lerini ve runtime eşiklerini kaydetme.
- Aynı approved validation setinde box, heuristic refinement, frozen-Recognition ve kaynak baseline'ı.
- EXIF ve standalone/runtime policy farkı için tek canonical karar.
- Metric tanımı, slice'lar, sayısal kabul/regresyon ve rollback eşiklerini `TBD`'den çıkarma.
- Teknik ADR: direct corner head, schema versioning ve strict fallback politikası.

**Teslimatlar:** Baseline raporu, veri envanteri, metric/acceptance manifesti, architecture/coordinate ADR, risk register güncellemesi.

**Doğrulama yöntemi:** Hash ve run receipt kontrolü; baseline'ın aynı seed/config ile yeniden üretimi; acceptance metninin training öncesi review/sign-off'u.

**Riskler:** Yetkili label/veri erişimi yok; baseline hostu hedef deployment'ı temsil etmiyor; mevcut set no-plate/perspective slice içermiyor.

**Tahmini efor:** 3-5 kişi-gün. Veri erişimi ve benchmark çalışma süresi hariç.

**Çıkış kapısı:** True-corner annotation için hukuki/operasyonel yetki ve yeterli pilot aday; predeclared metrikler. Yoksa proje durur.

### Faz 1 — Annotation schema, guideline ve pilot

**Amaç:** Denetlenebilir, geometrik olarak geçerli ve belirsizliği temsil eden ground truth üretmek.

**Yapılacak işler:**

- Sürümlü label/manifest schema ve validator tasarımı.
- Corner order, visible boundary, holder/occlusion/glare/clipping guideline'ı.
- Pilot annotation, second-pass audit ve adjudication.
- Group-aware split leakage ve near-duplicate kontrolü.
- Pilot dağılıma göre annotation kabul eşiklerini kapatma.

**Teslimatlar:** Annotation specification, validator/test fixtures, audited pilot, dataset revision/hash receipt, go/no-go raporu.

**Doğrulama yöntemi:** Otomatik validator %100; bağımsız audit ve NCE/disagreement raporu; split leakage = 0.

**Riskler:** Köşelerin görünür olmaması, annotator tutarsızlığı, public/private sınır ihlali, annotation hacminin yetersiz kalması.

**Tahmini efor:** 4-7 kişi-gün mühendislik + `TBD` annotation eforu.

**Çıkış kapısı:** Pilot kalite eşiğini geçer; aksi halde guideline/re-annotation veya stop kararı.

### Faz 2 — Veri hattı ve geometrik augmentation

**Amaç:** Box ve corner hedeflerini aynı dönüşümlerle deterministik ve doğru üretmek.

**Yapılacak işler:**

- Structured target ve v2 dataset loader/collate.
- Box-only legacy label için açık masked compatibility.
- Letterbox/corner forward-inverse transform.
- Flip reorder, affine/perspective ve photometric augmentation.
- Invalid transform rejection ve visibility mask propagation.
- Dataset/schema/content hash'i training contract'a bağlama.

**Teslimatlar:** V2 data pipeline, config alanları, unit/property tests, synthetic geometry golden fixtures.

**Doğrulama yöntemi:** Round-trip tolerance `TBD`; golden fixtures; yüzlerce seeded random transform üzerinde invariant/property testleri; v1 testlerinin geçmesi.

**Riskler:** Coordinate-space karışıklığı, rounding/padding hatası, augmentation'ın gerçek dışı target üretmesi.

**Tahmini efor:** 5-8 kişi-gün.

**Çıkış kapısı:** Tüm geometry/data testleri geçer ve visual pilot overlay audit'i onaylanır.

### Faz 3 — Model, loss ve reproducible training

**Amaç:** V1 box yeteneğini koruyan, aynı aday için sıralı dört köşe üreten ilk v2 modeli eğitmek.

**Yapılacak işler:**

- 13-kanallı decoupled head ve typed decode contract.
- Masked corner loss; loss/gradient telemetry.
- V1 warm-start ve yeni checkpoint format/architecture ID.
- Staged training, hyperparameter ve sınırlı ablation matrisi.
- Seed tekrarları, early stopping ve predeclared candidate selection.
- Registry/model-card taslağı ve training receipt.

**Teslimatlar:** Reproducible training pipeline, candidate checkpoint(ler), run manifest/loglar, ablation ve model-card taslağı.

**Doğrulama yöntemi:** CPU architecture smoke, backward/finite-gradient testleri, overfit-small-batch testi, aynı config/seed receipt kontrolü, validation metric üretimi.

**Riskler:** Multi-task gradient çatışması, small-plate hassasiyeti, overfit, training kararsızlığı, compute kısıtı.

**Tahmini efor:** 7-12 kişi-gün + `TBD` training/GPU süresi.

**Çıkış kapısı:** Box parity ve minimum corner/polygon development gate'lerini geçen en az bir reproducible candidate; aksi halde revise/reject.

### Faz 4 — Inference, geometri gate ve rectification

**Amaç:** Tahmin edilen köşeleri güvenli biçimde orijinal koordinata taşıyıp yalnız geçerli quad'ı Recognition girdisine çevirmek.

**Yapılacak işler:**

- Box/corner bağlı decode, filtering, NMS ve deterministic selection.
- Common orientation/letterbox inverse policy.
- Quad validator ve reason code'lar.
- Homography/warp boyutlandırma ve pixel-budget sınırı.
- Strict unresolved davranışı; legacy heuristic'in v2 yolundan ayrılması.
- Polygon/crop debug artifact ve provenance.

**Teslimatlar:** V2 inference/postprocessing modülleri, geometry fixtures, benchmark hooks, traceable crop provenance.

**Doğrulama yöntemi:** Synthetic known-homography round-trip; mirror/flip testleri; invalid quad matrisi; end-to-end frozen-Recognition smoke; v1 loader/runtime regresyonu.

**Riskler:** Corner order bug'ı, out-of-bounds clamp'ın hatayı gizlemesi, warp artifact, NMS sonrası candidate mismatch.

**Tahmini efor:** 5-8 kişi-gün.

**Çıkış kapısı:** Deterministik geometri suite'i geçer; invalid quad hiçbir yolda v2 Recognition'a girmez.

### Faz 5 — Contract, servis, UI ve review entegrasyonu

**Amaç:** Yeni geometriyi tüketicilere sürümlü ve geriye uyumlu biçimde sunmak.

**Yapılacak işler:**

- Private/public v2 schema, generated contracts ve fixtures.
- Inference service/BFF mapping ve status/error semantics.
- Polygon overlay, rectified crop, EN/TR ve accessibility.
- Review/export alanları ve derived legacy box.
- Model bundle/readiness SHA-256 ve feature flag seçimi.
- Privacy/no-retention, timeout, cancellation ve concurrency kontrolleri.

**Teslimatlar:** Sürümlemeli sözleşmeler, servis/UI/review entegrasyonu, migration note ve rollback runbook güncellemesi.

**Doğrulama yöntemi:** Schema/contract tests, browser tests, API fixtures, privacy scan, cancellation/timeout ve hash-mismatch fail-closed testleri.

**Riskler:** V1 tüketici kırılması, durumların UI'da karışması, artifact retention, contract drift.

**Tahmini efor:** 5-8 kişi-gün.

**Çıkış kapısı:** V1 ve v2 test matrisleri birlikte geçer; no-retention ve compatibility review onaylanır.

### Faz 6 — Paired değerlendirme, performans ve release kararı

**Amaç:** V2'nin geometriyi ve uçtan uca sonucu iyileştirdiğini, box ve operasyonel davranışı bozmadığını kanıtlamak.

**Yapılacak işler:**

- Aynı approved girdilerde v1/v2 paired run.
- Box, corner, polygon, rectification, Recognition ve slice metrikleri.
- Abstention/failure'ları tam paydada raporlama.
- Hedef host cold/warm p50/p95 latency, CPU, RSS/device memory, startup.
- Error taxonomy ve manual qualitative audit.
- Acceptance manifestine göre accept/revise/reject kararı.

**Teslimatlar:** Karşılaştırma raporu, performance raporu, failure gallery'nin privacy-safe özeti, release decision record.

**Doğrulama yöntemi:** Önceden kilitli eşiklere otomatik gate; paired istatistik; receipt/hash kontrolü; bağımsız review.

**Riskler:** Validation'a aşırı uyum, küçük slice'lar, hedef host erişimi yok, OCR artarken Detection recall düşmesi.

**Tahmini efor:** 6-10 kişi-gün + `TBD` benchmark/training çalışma süresi.

**Çıkış kapısı:** Bütün zorunlu eşikler geçer. Herhangi bir kritik gate geçmezse rollout yoktur.

### Faz 7 — Shadow rollout, onaylı switch ve rollback provası

**Amaç:** Canlı routing'i riske atmadan candidate'ı doğrulamak ve geri dönüşü kanıtlamak.

**Yapılacak işler:**

- Ayrı immutable v2 bundle/image ve registry kaydı.
- Opt-in bounded shadow; canlı yanıtı değiştirmeden karşılaştırma.
- Privacy-safe telemetry ve rollback trigger gözlemi.
- Rollback provası: v1 image/bundle/readiness/contract.
- Ayrı kullanıcı onayı sonrası kontrollü switch ve smoke.

**Teslimatlar:** Shadow raporu, signed model identity, rollout checklist, rollback prova kaydı ve final change set.

**Doğrulama yöntemi:** Shadow acceptance, startup hash, resource limit, proxy/HTTPS/schema/privacy smoke ve rollback time `TBD` hedefi.

**Riskler:** Shadow trafik/veri izni, resource contention, config drift, rollback artifact'ının erişilemez olması.

**Tahmini efor:** 3-5 kişi-gün + `TBD` gözlem süresi.

**Çıkış kapısı:** Açık kullanıcı release onayı ve başarılı rollback provası. Onay yoksa v1 live kalır.

## 13. Özet uygulama tablosu

| Faz | Amaç | Yapılacaklar | Teslimat | Başarı ölçütü | Efor |
| --- | --- | --- | --- | --- | --- |
| 0 — Baseline ve karar sözleşmesi | Mevcut durumu dondurmak ve eşikleri önceden belirlemek | Veri/envanter, V1 ölçüm, hash/threshold kaydı, ADR ve acceptance manifesti | Baseline + envanter + ADR + predeclared gate'ler | Kaynak/model identity doğrulandı; corner yokluğu kanıtlandı; özel sayısal eşikler `TBD` | Mühendislik tamam; özel threshold kapısı açık |
| 1 — Annotation pilotu | Güvenilir dört-köşe ground truth üretmek | Deterministik seçim, review-only öneri, yerel düzeltme UI, ikinci audit, split/duplicate kontrolü | 256 kayıtlı private pilot, append-only review log, validator, audited annotation release | Tüm pilot kararları tamam; validator %100; leakage 0; audit/NCE eşiği `TBD` | Araçlar tamam; insan review/ikinci audit `TBD` |
| 2 — Veri ve augmentation | Tüm geometrileri senkron ve deterministik dönüştürmek | Structured target, inverse transform, flip reorder, affine/perspective, masking | V2 data pipeline + geometry tests | Round-trip/property/golden testleri ve V1 regresyonu geçer | 5-8 kişi-gün |
| 3 — Model ve training | Box parity koruyarak corner head eğitmek | 13-kanal head, corner loss, warm-start, training/ablation, checkpoint v2 | Candidate checkpoint + receipt + model-card taslağı | Box parity ve corner/polygon gate'leri `TBD`; reproducible run | 7-12 kişi-gün + compute `TBD` |
| 4 — Inference ve rectification | Geçerli quad'dan güvenli Recognition crop'u üretmek | Decode/NMS association, quad gate, inverse mapping, homography, strict unresolved | V2 inference/postprocess + provenance | Geometry suite geçer; invalid quad Recognition'a girmez; mirror/flip 0 | 5-8 kişi-gün |
| 5 — Contract ve ürün entegrasyonu | V2 çıktıyı geriye uyumlu sunmak | API/schema, service/BFF, UI, review/export, hash/privacy | V2 sözleşmeler + entegrasyon + migration/rollback notu | V1+V2 contract/UI/privacy/fail-closed testleri geçer | 5-8 kişi-gün |
| 6 — Değerlendirme ve karar | Kalite ve kaynak bütçesini kanıtlamak | Paired V1/V2 metrik, slice, OCR, latency/memory, audit | Comparison/performance raporu + karar kaydı | Tüm predeclared kalite/operasyon eşikleri geçer | 6-10 kişi-gün + çalışma süresi `TBD` |
| 7 — Shadow ve rollout | Riski sınırlı release ve doğrulanmış rollback | Immutable bundle, shadow, trigger izleme, rollback prova, onaylı switch | Shadow raporu + rollout/rollback kaydı | Shadow pass; rollback prova pass; açık kullanıcı onayı | 3-5 kişi-gün + gözlem `TBD` |

Toplam tahmini mühendislik eforu: **38-63 kişi-gün**. Annotation, training/benchmark compute, erişim beklemeleri ve shadow gözlem süresi bu toplama dahil değildir ve Phase 0-1 sonunda ayrıca planlanmalıdır.

## 14. Git ve değişiklik yönetimi

- Uygulama kullanıcı onayıyla başlamıştır; source, schema, test ve private review-only pilot çıktıları
  çalışma ağacında tutulur. Referans roadmap'in üzerine yazılmaz.
- Kullanıcı onaylasa bile bütün fazlar tamamlanıp ilgili testler geçmeden commit oluşturulmaz.
- Final değişiklik seti kullanıcıya diff, test sonuçları, model/data identity ve bilinen limitlerle sunulur.
- Yalnız açık izinle tek commit oluşturulur. Push için commit izninden ayrı, açık onay alınır.
- Branch, pull request, deployment ve live routing değişikliği ayrıca açık yetki gerektirir.
