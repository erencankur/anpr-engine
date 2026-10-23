# Öğrenilmiş dört köşe: CPU aday seçimi ve yerel entegrasyon

Tarih: 2026-10-15. Seçim: HyperLPR YOLOv5 multitask `y5fu_640x_sim.onnx`.
Bu seçim yerel geliştirme adayıdır; üretim doğruluğu veya tüm plaka türleri
garantisi değildir. Model Çin plakalarına yönelik olduğundan TR domain farkı vardır.

## Artifact ve lisans kaydı

| Alan | Değer |
| --- | --- |
| Model kimliği | `hyperlpr-y5fu-640-corners-v1` |
| Dosya büyüklüğü | 3.930.791 byte (yaklaşık 3,75 MiB) |
| SHA256 | `0306de937471b87f56eb3f5620815e7e7058f8ab7428e0734fc64627cc4d716c` |
| Dağıtım revision'ı | `lets-mica/mica-ai`, `c2cef59250213f159d31dfe433491806f9eef4a7` |
| Kaynak | Sabit revision'dan HTTPS ile indirilen ONNX; private model dizininde |
| Lisans beyanı | Mirror model README'si weights'i HyperLPR3 v20230229, Apache-2.0 olarak belirtir; Apache metni artifact yanında saklanır |

Kaynaklar: [weights/model README](https://github.com/lets-mica/mica-ai/blob/c2cef59250213f159d31dfe433491806f9eef4a7/model-tools/plate/README.md),
[640 ONNX artifact](https://github.com/lets-mica/mica-ai/blob/c2cef59250213f159d31dfe433491806f9eef4a7/model-tools/plate/models/y5fu_640x_sim.onnx),
[resmi upstream preprocessing/decode](https://github.com/szad670401/HyperLPR/blob/9307450f7b7915be18f23a539ec05b41fe6629f4/Prj-Python/hyperlpr3/inference/multitask_detect.py).
Model dağıtıcısının lisans beyanı kayıtlıdır; bu bağımsız hukuk incelemesi değildir.
Upstream model sunucusunda TLS doğrulaması başarısız oldu; doğrulama kapatılmadı.
HTTPS, revision-pinned mirror kullanıldı. Kullanıcı fotoğrafları dış servise gönderilmedi.

## Uygulanan akış

- `hyperlpr_corners.py`: OpenCV DNN, CPU hedefi ve tek thread; giriş
  `[1,3,640,640]`, RGB/255 siyah letterbox; çıkış `[1,25200,15]`.
- Native sekiz koordinat TL/TR/BR/BL olarak korunur. Köşe başına confidence
  üretilmediği için böyle bir değer uydurulmaz. Objectness × layout score
  detection skoru olarak kullanılır; bu da doğruluk garantisi değildir.
- Pozitif alan, bounds, finite, convexity ve minimum crop boyutu denetlenir.
  Crossing/out-of-bounds köşeler kırpılıp veya yeniden sıralanıp geçerli sayılmaz.
  Raporlanan box aynı model köşelerinin zarfıdır. NMS ve en fazla 5 aday vardır.
- Model köşeleri doğrudan homography'ye gider. Kontur, mavi bant, glyph fitting
  ve legacy detector bu yolda kullanılmaz; eski detector ağı yüklenmez.
  Ana OCR değişmedi. Model dosyaları public repo/build context'e eklenmedi.
- İlk geçişte plaka yoksa aynı ağla sabit -30/+30 yön kurtarma denenir;
  en fazla **3 detector geçişi** vardır. Ek ağ veya worker yoktur. İlk uygun
  yön seçilir; OCR çıktısı geometri/yön seçmez. OCR yalnızca bir kez çağrılır.
  Fallback warp öncesi görüntü en uzun kenarı 1280 px'e indirilir; köşeler
  inverse affine + ölçek dönüşümüyle özgün fotoğrafa döner.
- Bu politika önceki tek-geçiş hedefinden kontrollü sapmadır: normal girdide
  1 geçiş, tespit kaçırıldığında en fazla 3. Sınırsız açı araması yoktur.
- Model kimliği/hash, crop yöntemi ve BFF sağlayıcı ayarı birlikte güncellendi.
  Yan yana dörtgen/warp/OCR preview aynı model geometrisinden gelir.

## Aday karşılaştırması ve kısa testler

| Kontrol | 320 girişli aday | Seçilen 640 girişli aday |
| --- | --- | --- |
| İndirilen ONNX boyutu | 2.343.161 byte | 3.930.791 byte |
| 10 mevcut geliştirme upload'u | 7 doğru kabul, 1 yanlış kabul, 2 doğru aday | 10 doğru kabul, 0 yanlış, 0 aday/çekimser |
| Dörtgen bulunan upload | 10/10 | 10/10 |
| Sıcak HTTP upload medyanı | 257,86 ms | 320,00 ms |
| Örneklenen Python + OCR RSS | 1268,86 MiB | 1266,72 MiB |
| Karar | Varsayılan için reddedildi | Yerel aday olarak seçildi |

Profiller aynı Mac/tek thread ve mevcut 10 fotoğraf/3 plaka ile yapıldı.
Web/tarayıcı RSS'ye dahil değildir; ölçüm container peak veya Linux bütçesi
kanıtı değildir. 640 idle sonrası 564,86 MiB; worker kapanışı/restart ve
plakasız kontrol geçti. Kaynak uygunluğu düşük RAM'li sunucuda hâlâ **TBD**.

İlk tek-geçiş ±25° testinde küçük plaka tespitleri kaçtı. Yön kurtarma sonrası
farklı açılardan oluşan 12 kontrol yapıldı: 2 kaynak fotoğraf × identity,
-20°, +20°, -35°, +35°, keystone. Maksimum görüntü kenarı 1600 px.
12/12 dörtgen, 9 doğru kabul + 3 doğru kontrol adayı, 0 yanlış okuma görüldü.
Bu sentetik regresyondur, bağımsız gerçek dünya holdout'u değildir.
Fiziksel köşe ground truth'u olmadığı için corner error/IoU **TBD**.
Üç geçişli sentetik boş görüntü kontrolünde detection 219,09 ms sürdü;
tek soğuk örnek ölçümüdür, gecikme üst sınırı garantisi değildir.

Gerçek WebKit'te normal fotoğraf + 35° döndürülmüş JPEG upload'u geçti:
model ID ve `HYPERLPR_FOUR_CORNERS_V1`, SVG/API köşe eşleşmesi, iki preview
ve OCR görünümü doğrulandı; console hata/uyarı yok. Eski screenshot dosyaları
ve önceki kullanıcı PNG'si bu tur diskte yoktu; o girdiler için yeni modele
özgü başarı iddiası yapılmaz ve kullanıcıdan yeni örnek istenmedi.

Standart başlatıcıyla 39084/39085 üzerinde yeniden açılan site de gerçek HTTP
upload ile kontrol edildi: 640 model kimliği/hash, dört köşe ve iki preview
doğrulandı; `recognition.prediction` beklenen metni verdi. Public API'nin
`unresolved` durumu insan incelemesi zorunluluğudur; doğru okuma otomatik
doğrulanmış sonuç veya başarı garantisi olarak sunulmaz. Geçici test portları
39086/39087 ve test tarayıcısı kapatıldı; normal yerel site açık bırakıldı.

## Faz durumu

| Faz | Teslimat / doğrulama | Durum / kalan |
| --- | --- | --- |
| M0 | Geometri yokken OCR durur; legacy rollback ayrı | Tamamlandı |
| M1 | İki gerçek checkpoint, sabit revision/hash, weights lisans beyanı ve model seçimi | Yerel seçim tamamlandı; bağımsız lisans/domain incelemesi yayın öncesi |
| M2 | CPU adapter, gerçek corner çıkışı, tek aktif detector, homography, schema/BFF/launcher | Yerel entegrasyon tamamlandı |
| M3 | 10 gerçek geliştirme fotoğrafı + 12 bounded perturbation + geometri birim testleri | Yerel regresyon tamamlandı; bağımsız ground truth/gece/örtülme kapsamı bekliyor |
| M4 | Kısa HTTP RSS/gecikme/idle/restart + gerçek browser | Mac kontrolü tamamlandı; sunucu/Linux ölçümü bekliyor |
| M5 | Yerel başlatıcı 640 config'e bağlandı; eski ayar ayrı legacy başlatıcıda | Yerel seçim tamamlandı; canlıya yayın, commit ve push yapılmadı |

Son genel kontrol: 414 Python testi geçti / 3 opt-in atlandı; 6 web akış
testi, build/typecheck/lint, mypy ve public validator başarılı. Gerçek model
HTTP profili ve browser kontrolleri ayrıca çalıştırıldı. Eğitim/CUDA, yeni
Python ortamı veya dependency kurulumu yok; toplam model indirmesi yaklaşık 6 MB.

## Kullanım ve rollback

Yerel private `Start Local Website.command` yeni 640 config'i seçer.
Eski açık süreçlerin yeni ağı yüklemek için yeniden başlatılması gerekir.
`Start Local Website Legacy.command` önceki native OCR + eski detector ayarını
korur; aynı portta iki başlatıcı çalıştırılmaz. Yerel IP kota istisnası korunur.

Genel yapılandırma şablonu: `configs/public/web-ocr-learned-corners-template.json`.
Web'de `ANPR_DETECTION_PROVIDER=hyperlpr640` ve
`ANPR_RECOGNITION_PROVIDER=paddleocr`; inference'ta yeni OCR JSON'u kullanılır.
ONNX dosyası belirtilen yola read-only provision edilir; runtime indirme yapmaz.
Linux Docker/Compose için bu ek model mount'u ve ortam ayarı ayrıca gerekir.
`corner_model` kaldırılıp web detector provider `project` yapılırsa eski yola
dönülür. İki servis birlikte geri alınır; yeni model sessizce legacy ağı açmaz.
Çift satır layout desteklenmediği için reddedilir; aşırı açı/örtülme/gece
başarısız olabilir. Eksik/uygunsuz geometri OCR'a verilmez.
