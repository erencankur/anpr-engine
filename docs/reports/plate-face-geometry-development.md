# Plaka yüzeyi dörtgeni: düşük kaynaklı yerel geliştirme

Tarih: 2026-10-15. Kapsam: mevcut detector + native Paddle OCR + web akışı.
İki sorun fotoğrafının orijinal JPEG/PNG dosyaları sağlanmadı; bu iki örneğin
düzeldiğine dair kanıt yoktur. Ekran görüntüleri model girdisi olarak kullanılmadı.

## Uygulanan akış

1. Detector'ın `xyxy` kutusunun çevresini yatayda her yöne kutu genişliğinin
   %25'i, düşeyde her yöne yüksekliğin %60'ı kadar aç; görüntü sınırlarına kes.
   Böylece sıfır-paylı kutunun kestiği köşe, çevrede görünüyorsa kurtarılabilir.
2. İşleme bölgesinin en uzun kenarını en fazla 960 piksele indir.
   Beyaz/mavi renk maskesinde en büyük 32 konturu değerlendir.
3. Dört köşe, dışbükeylik, kutuyla örtüşme, plaka benzeri oran, karakter
   bileşenleri ve görüntü desteği kontrollerini uygula. ROI sınırında kesilen
   veya birbirine yakın puanlı belirsiz adayları reddet. OCR metni/güveni
   köşe seçiminde kullanılmaz; OCR sonucuna göre tekrar deneme yoktur.
4. Kabul edilen yüzeyi `PLATE_FACE_CONTOUR_V1` ile perspektif düzelt.
   Çıktı en fazla 1280×384 pikseldir. Ardından mevcut karakter/mavi bant
   çerçevelemesini uygula; uygunsuzsa düzeltilmiş tam yüzeyi kullan.
5. Özgün görüntü koordinatlarındaki aynı dörtgeni opsiyonel `plate_geometry`
   alanıyla API'den arayüze taşı. SVG ve kullanıcı tıklamasıyla oluşturulan
   işaretli indirme aynı köşeleri kullanır. Köşeler yoksa kesikli yaklaşık
   detector kutusu ve açık uyarı göster; eski OCR kırpımına dön.
6. Güven eşiğini geçen OCR metnini TR biçim kontrolünden bağımsız göster.
   Örneğin `82TR37` görünür kalabilir fakat il kodu denetimi nedeniyle
   `invalid` olarak işaretlenir. Düşük güvende metin yine gösterilmez.

Bu, öğrenilmiş corner/keypoint modeli değildir. Renk/karakter koşulları
sağlanmayan gece, yansıma, beyaz gövdeyle birleşmiş yüzey, iki satırlı plaka,
örtülme ve çok sert perspektifte başarısız olabilir. Tahmin edilen konturun
fiziksel plaka köşeleriyle örtüştüğü küçük gerçek kümede ölçülmedi: **TBD**.

## Fazlar ve doğrulama

| Faz | Amaç | Yapılan işler | Teslimat | Başarı ölçütü / doğrulama | Risk | Tahmini aktif efor | Durum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | Kutu dışındaki köşeleri kurtarmak | Geniş bağlam, kontur filtreleri, bounded homography | `plate_face_geometry.py` | Sentetik eğik plaka: kutu dışındaki sol alt köşe; köşe hatası <5 px. Büyük ROI: <15 px | Heuristik gerçek sınır yerine başka yüzey seçebilir | 0.5–1 gün | Yerel test tamamlandı |
| G2 | Tek geometriyi bütün akışa taşımak | OCR bağlantısı, gerçek kırpım provenance'ı, opsiyonel şemalar, BFF kontrolü, SVG/Canvas | Python/TS sözleşmeleri ve web güncellemesi | Aynı API köşeleri SVG/Canvas'a gider; ters/bozuk dörtgen reddedilir; eski yanıtlar kabul edilir | EXIF/ölçek ve sözleşme uyumsuzluğu | 0.5 gün | Yerel test tamamlandı |
| G3 | OCR ve biçim kontrolünü ayırmak | `display_text` ve TR-validated `tr_text` ayrımı | Runtime/API/testler | Yüksek güvenli `82TR37` görünür ve invalid; düşük güven gizli | Geçersiz biçim, doğru plaka sanılabilir; uyarı korunur | 0.25 gün | Yerel test tamamlandı |
| G4 | Regresyon ve kaynak kontrolü | Tek thread kısa profil, 10 geliştirme upload'u, plakasız kontrol, idle/restart, WebKit | Bu rapor; private aggregate profil | 10/10 exact match, 0 yanlış, worker kapanışı; genel testler | Küçük geliştirme kümesi temsilî değil | 0.5 gün | Yerel kontrol tamamlandı |
| G5 | Sorun örneklerini doğrulamak | Orijinal iki fotoğrafta kırpım, köşe ve metin kontrolü | Yeni regression fixture/aggregate sonuç | Gerçek sınır köşe hatası ve iki örnekte exact match; tolerans TBD | Dosyalar henüz yok | Dosyalar hazırsa 1–2 saat | Bekliyor |
| G6 | Genel perspektif başarısı | Temsilî eğik/gece/küçük/plakasız veri; gerekirse öğrenilmiş köşe modeli | Ayrı kabul ölçümü | Corner error, accepted-error, abstention ve OCR exact match hedefleri TBD | Annotation/model/lisans/veri bağımlılıkları | Kapsam/veri seçilince TBD | Ertelenen annotation ve model çalışması |

## Son yerel bulgular

- 10 geliştirme fotoğrafı / 3 farklı plaka: 10 doğru metin, 0 yanlış, 0 çekimser.
  Bunlar eğitimden bağımsız holdout veya üretim doğruluğu kanıtı değildir.
- Dörtgen: 4 estimated, 6 unresolved. Altı başarısız köşe sonucu saklanmaz;
  eski OCR kırpımı ve açık kesikli-kutu uyarısıyla devam edilir.
- İlk tam-yüzey kırpım denemesinde 9 doğru / 1 çekimser görüldü. Yüzey
  düzeltmesi sonrasına mevcut piksel temelli çerçeveleme eklenince 10/10'a
  döndü. 0.99 güven eşiği düşürülmedi; iki OCR çalıştırılmadı.
- Native CPU profilinde sıcak upload medyanı 269.54 ms; yeniden açılış
  1062.19 ms. Ana Python + worker örneklenen toplam RSS 1380.52 MiB;
  idle sonrası 601.16 MiB. Web/tarayıcı dahil değil; Linux/container peak değil.
  Önceki ölçüm 1328.66 MiB idi; bu koşu daha yüksek. Düşük RAM'li sunucuda
  bütçeye uygunluk ölçülmeden garanti edilemez.
- Gerçek WebKit: bir unresolved fotoğrafta kesikli kutu, bir estimated
  fotoğrafta API köşeleriyle birebir SVG, doğru OCR metni ve Canvas çizim
  koordinatları doğrulandı. Console hata/uyarı yok. Bu kontrol, ilk sorun
  ekran görüntüsündeki fotoğrafın doğrulanması anlamına gelmez.
- Ek model yüklemesi, GPU/CUDA, training, dataset export veya yeni bağımlılık yok.
- Genel Python suite: 396 geçti, 2 private opt-in testi varsayılan olarak
  atlandı; gerçek model HTTP profili ayrıca çalıştırıldı. Dört web akış testi,
  production build, TypeScript, lint, mypy (174 dosya), public validator
  (376 dosya / 70 bağlantı) ve diff whitespace kontrolü başarılı.
- Test tarayıcısı ve bu tur açılan iki yerel servis kapatıldı.

## Geriye uyumluluk ve rollback

- `plate_geometry` v1 sözleşmesine opsiyonel eklenir; eski responses geometry
  olmadan çalışır. Strict haricî istemciler yeni alanı kabul etmek için kendi
  sözleşmelerini güncellemelidir; ek alan herkeste otomatik uyum garantilemez.
- OCR modeli/weights değişmedi. Şema, mevcut legacy ve onaylı Paddle
  kimlik/hash çiftlerini ayrı ayrı doğrular; kimlik ile başka model hash'i
  karıştırılamaz. Güven ve biçim denetimi doğruluk kanıtı değildir.
- Geometri regresyonunda OCR JSON'una `"estimate_plate_geometry": false`
  ekleyip servisleri yeniden başlat. Yeni kontur/homography devre dışı kalır;
  eski kırpım kullanılır, köşeler unresolved gösterilir. Bu tercih varsayılan
  olarak `true`dur. Daha geniş sorun halinde önceki web/inference sürümlerini
  birlikte geri yükle; mevcut weights/bundle korunur.
- Kod değişiklikleri commit/push edilmedi; kullanıcı onayı olmadan yayın yok.

Yerel çalıştırma: [Mac website rehberi](../operations/macos-local-website.md).
Yayın ve sunucu ölçümleri: [düşük kaynak roadmap'i](../plans/low-resource-website-completion.md).

## W10 — yan yana görüntüler ve köşe arama iyileştirmesi

2026-10-15 devam turu. Yukarıdaki W9 ölçümleri tarihsel sonuçlardır;
bu bölüm son uygulamayı ve ölçümü açıklar.

| Faz | Amaç / yapılan işler | Teslimat | Doğrulama | Durum / kalan | Risk / tahmini aktif efor |
| --- | --- | --- | --- | --- | --- |
| C1 | Desteklenen dış yüzeyi koru; yoksa daha sıkı mavi maskesinde iç sınırları, sonra Canny kenarlarını ara. Aynı yüzeyin yakın izlerini birleştir; mavi bandı içermeyen adayları reddet | 960 px ROI içinde en fazla 16 dış + 16 iç + 16 kenar konturu | Koyu mavi gölge, beyaz gövde, eğik plaka ve kutu dışındaki köşe testleri | Yerel uygulama tamamlandı | Hâlâ görüntü işleme tahmini, öğrenilmiş detector değil; 0.5–1 gün |
| C2 | Tam düzleştirilmiş yüzeyi ve OCR'ın gerçekten kullandığı çerçevelenmiş kırpımı ayrı üret | Opsiyonel `plate_crops`: `rectified` ve `recognition_input` | BGR/RGB, gerçek JPEG boyutu, piksel/byte sınırı, pipeline bağlantısı | Tamamlandı | Preview downscale/compression vardır; OCR preview'den beslenmez; 0.25 gün |
| C3 | Özgün dörtgen ve plaka önizlemelerini yan yana göster; mobilde alt alta yerleştir | TR/EN UI, blob URL yaşam döngüsü, mevcut CSP korunması | Gerçek WebKit: iki preview açılır; SVG API köşeleriyle aynı; 390 px taşma yok; clear URL'leri revoke eder | Tamamlandı | Eksik preview ile eksik geometri ayrı mesajlar; 0.5 gün |
| C4 | Doğruluk/kaynak regresyonu, sözleşmeler ve kapanış | Kısa CPU profili ve test sonuçları | Son koşuda 10/10 OCR, 8 estimated/2 unresolved; plakasız kontrol; idle/restart | Yerel kontrol tamamlandı | 10 fotoğraf/3 plaka bağımsız holdout değildir; 0.5 gün |
| C5 | Bildirilen sert perspektifli fotoğrafı tekrar üret ve gerçek köşelerle karşılaştır | Orijinal JPEG/PNG ile regression kanıtı | Köşe hatası ve exact match; tolerans TBD | Bekliyor: attachment'ta belirtilen screenshot dosyası bu tur diskte bulunamadı; orijinal fotoğraf yok | Bu vaka için düzelme iddiası yok; dosya hazırsa 1–2 saat |
| C6 | Daha geniş perspektif/gece/küçük plaka kapsamı | Temsilî veri ve bağımsız köşe doğruluğu ölçümü | Corner error, yanlış kabul ve çekimserlik hedefleri TBD | Bekliyor; köşe annotation/eğitimi hâlâ ertelendi | Bu yöntem bütün koşullarda fiziksel köşeleri garanti etmez; veri/kapsama göre TBD |

- Sağ panelde önce dört köşeden homography ile üretilen tam plaka, sonra
  OCR'a verilen kırpım görünür. İkincisi mevcut karakter/mavi bant çerçeveleme
  aşaması nedeniyle birinciden farklı olabilir; ikisi aynı görselmiş gibi sunulmaz.
- İki JPEG preview'nın her biri en fazla 640×192 px / 12.000 byte;
  base64 toplamı en fazla 32.000 karakter. 64.000-byte BFF yanıt sınırı
  artırılmadı. Byte, gerçek JPEG header boyutu ve metadata alanları doğrulanır.
  Preview'de EXIF yoktur; diske, loglara, quota DB'ye veya eğitime yazılmaz.
- BFF/UI daha büyük model girdisini tekrar işlemeye çalışmaz. Preview kalite
  veya boyut düşürmesi OCR sonucunu değiştirmez. Blob URL'leri yeni işleme,
  yeni fotoğraf, clear ve unmount sırasında bırakılır; `img-src data:` eklenmedi.
- Köşe bulunamadığında düzeltilmiş görüntü uydurulmaz: kesikli yaklaşık
  kutu ve açıklama görünür. Köşe bulunduğu halde preview yoksa farklı mesaj vardır.
- İlk iç/kenar arama denemeleri 9 doğru / 1 çekimser üretti. Desteklenen dış
  yüzeyin iç/kenar izleriyle değiştirilmesini önleyen seçim sırası sonrası
  son koşu 10 doğru / 0 yanlış / 0 çekimser oldu; 0.99 eşik değişmedi.
- Son gerçek HTTP profili: 8 estimated / 2 unresolved (W9: 4 / 6).
  Bu sayı corner ground truth doğruluğu değildir. Sıcak upload medyanı
  279.52 ms; örneklenen Python + worker toplam RSS 1381.05 MiB, idle
  602.22 MiB. Web/tarayıcı hariç; Linux peak veya az RAM'li sunucu garantisi değil.
- Tek CPU thread, sıfır kuyruk ve lazy/idle OCR worker korunur. Ağır eğitim,
  GPU/CUDA, toplu export veya yeni model/bağımlılık indirmesi yapılmadı.
- Genel Python suite 400 geçti / 2 private opt-in atlandı. Gerçek model
  HTTP profili ayrıca çalıştırıldı. Web dört akış testi, build/typecheck/lint,
  Python lint/mypy ve public sözleşme kontrolü başarılı. Test süreçleri kapatıldı.
- Rollback: mevcut `estimate_plate_geometry=false` ayarı yeni kontur ve
  preview yolunu devre dışı bırakır. Önceki web/inference çıktıları birlikte
  geri alınabilir. Commit, push veya yayın yapılmadı.

## W11 — orijinal sorun girdisinde doğrulanan düzeltme

Kullanıcı orijinal PNG'yi sağladı. Önceki “orijinal bekleniyor” durumu bu
fotoğraf için kapandı; diğer örnekler ve bağımsız köşe ground truth'u bundan
ayrı kalır. Görsel public kaynaklara, fixture'lara veya eğitim kümesine eklenmedi.

| Adım | Kanıt / teslimat | Durum |
| --- | --- | --- |
| Hatanın gerçek girdide tekrarı | Önce dörtgen yok; bandın bir harfi OCR'a karışıyor; metin kabul edilmiyor | Tamamlandı |
| Kök neden | Gölgede sabit 135 parlaklık filtresi yüzeyi kesiyor; konkav dış iz/sadeleştirme koyu köşeyi atıyor; sabit aydınlık yüzey oranı geçerli adayları reddediyor | Ölçülerek belirlendi |
| Geometri düzeltmesi | Fallback nötr yüzey + doygun mavi ayrımı, destekli convex hull, kenar çizgisi fitting ve kesişim; Otsu yerel kontrast desteği | Tamamlandı; eşikler bu geliştirme kümesinde, bağımsız kalibrasyon TBD |
| Gerçek OCR çıktısı | Düzeltilmiş yüzey doğru metni okuyor; skor yaklaşık 0.98066, mevcut 0.99 kabul eşiğinin altında | Doğru aday; otomatik kabul edilmiş sonuç değil |
| API / UI | `candidate_prediction` opsiyonel; kabul edilen `prediction` ayrı/null; durum ve uyarı unresolved kalır; aday görünür; otomatik tahmin onayı kapalı | Gerçek WebKit ile doğrulandı |
| Regresyon | 10 geliştirme görselinde 10 dörtgen tahmini; 9 kabul edilen doğru metin + 1 doğru aday; yanlış kabul yok | Yerel kontrol; köşe doğruluğu oranı veya holdout benchmark'ı değil |
| Orijinal görselin testi | `test_supplied_perspective_upload_rectifies_and_returns_ocr_candidate`; gerçek HTTP-ASGI, detector ve native OCR | Private opt-in olarak ayrıca geçti |
| Geniş koşullar | Gece, örtülme, parlamalar, iki satır ve başka plaka türleri; gerçek köşe toleransı/metric | Bekliyor; destek/başarı garantisi verilmez |

Runtime hiçbir görselin köşe koordinatlarını veya beklenen metnini içermez.
Testin dosya/etiketi environment'tan sağlanır. Çıktıdan sonuca göre geometri
seçme, birden fazla OCR ile güven yükseltme veya 0.99 eşiğini düşürme yoktur.
`candidate_prediction`, yalnızca karakterleri normalize edilmiş model okumasıdır;
biçim kontrolü veya aday gösterimi doğruluk kanıtı değildir. Aday JSON indirmesinde
`model_prediction` olarak yer alabilir; review outcome kullanıcı kararına kadar
`UNRESOLVED` kalır. Bu gösterim kabul edilmiş metin sayısını artırmış sayılmaz:
önceki 10 kabul yerine son koşuda 9 kabul / 1 inceleme adayı vardır.

Geometri hâlâ detector kutusunun ardından çalışan görüntü işleme tahminidir;
eğitilmiş corner modeli değildir. Fiziksel köşelere nicel hata ölçümü **TBD**.
Konveks zarf, kenar destek ve azami köşe kayması sınırlarıyla kısıtlanır;
belirsiz/uygunsuz adayda dörtgen uydurulmaz. 960 px ROI, 48 kontur, 1280×384
warp, iki küçük preview, tek CPU thread ve idle çıkışı korunur.

Son otomatik kontroller: 401 Python testi geçti / 3 private opt-in atlandı;
orijinal görsel testi ayrıca 1 geçti. Dört web akış testi, build/typecheck/lint,
Python lint/mypy ve public validator başarılı. Gerçek WebKit'te API dörtgeni
SVG ile aynı, iki preview yüklü, doğru aday görünür, kabul düğmesi kapalı;
konsol hata/uyarı yok. Desktop ve mobile görünümler gözle kontrol edildi.
Yeni weight/bağımlılık indirilmedi; GPU/eğitim çalıştırılmadı; commit/push yok.
