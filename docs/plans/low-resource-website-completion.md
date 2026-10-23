# Düşük kaynaklı web sitesi: uygulama ve kalan yayın işleri

Tarih: 2026-10-14. Amaç: tek araç fotoğrafından plaka kutusu ve plaka metni.
Sunucu kapasitesi kullanıcı tarafından yalnızca “az” olarak belirtildi; sayısal kapasite TBD.
Bu kayıt canlıya yayın yapıldığı veya hatasızlık garantisi verildiği anlamına gelmez.

## Tamamlanan yerel uygulama

| Faz | Amaç / yapılan işler | Teslimat | Doğrulama | Durum |
| --- | --- | --- | --- | --- |
| W1 | Mevcut detector arkasında yalnızca seçilen OCR'ı çalıştırmak | Opsiyonel PaddleOCR yolu; eski OCR ağı bu yolda yüklenmez | Eski recognizer kurulursa hata veren birim testi | Tamamlandı |
| W2 | Boşta RAM ve işlemci kullanımını azaltmak | İlk plaka isteğinde worker açılışı; 60 saniye boşta kapanış; varsayılan tek CPU thread | Idle, yeniden açılış, stale timer, shutdown testleri | Tamamlandı |
| W3 | Web/API uyumluluğu | Yapılandırılmış sağlayıcıya göre model kimliği ve SHA doğrulaması; düşük güvende okunamadı | Her iki sağlayıcı için web akış testi; API sözleşmesi | Tamamlandı |
| W4 | İşlevsel kontrol | İki özel geliştirme görseliyle gerçek HTTP-ASGI endpoint smoke testi | Gerçek detector + PaddleOCR; diske görüntü export edilmez | Yerel kontrol tamamlandı; bağımsız doğruluk ölçümü değil |
| W5 | Kaynak sınırları | Tek CPU thread; tek işlem; kısa gerçek HTTP profili; idle çıkışı ve yeniden açılış | 10 geliştirme upload'u + plakasız kontrol; 60 saniye idle; ayrıca gerçek restart | Yerel ölçüm tamamlandı; Linux/sunucu ölçümü bekliyor |
| W6 | Yayın ve geri dönüş | Linux hash kilidi, ayrı OCR venv Dockerfile'ı, Compose override ve runbook | Build context testi; Compose birleşimi; Linux image ve staging upload gerekli | Paketleme dosyaları hazır; Linux build ve canlı geçiş tamamlanmadı |
| W7 | Aynı OCR modelini daha hafif çalıştırmak | Doğrudan Paddle Inference adapter'ı; 17 dağıtımlı native kilit | 221 kırpımda metin/güven eşleşmesi; 10 gerçek upload; idle/restart | Yerel kontroller tamamlandı; native yayın adayı hazır, Linux testi bekliyor |
| W8 | Mac'te yerel website akışını tamamlamak | Tek komutlu başlatıcı; standalone web; local HTTP cookie düzeltmesi; venv yolu korunması; 15 saniye idle | Gerçek WebKit'te 2 farklı plaka + plakasız fotoğraf + temizle; regression testleri | Tamamlandı; Linux yerel akışın önkoşulu değil |
| W9 | Perspektif geometrisini OCR ve arayüz arasında eşleştirmek | Genişletilmiş bağlamda hafif kontur tahmini; aynı dörtgenle homography/SVG/indirme; çözümlenmeyen köşe uyarısı; OCR ve biçim kontrolünü ayırma | Sentetik kesilmiş köşe + 10 gerçek geliştirme upload'u + WebKit + sözleşme/rollback testleri | Yerel uygulama tamamlandı; gerçek köşe modeli değil, orijinal sorun fotoğrafları bekliyor |
| W10 | Orijinal dörtgeni ve düzleştirilmiş plakayı yan yana göstermek | Tam yüzey + gerçek OCR kırpımı için bounded JPEG önizlemeleri; mobil yerleşim; iç kontur/kenar fallback ve gölge kontrolü | 10/10 OCR; 8/10 dörtgen tahmini; gerçek WebKit'te iki önizleme, SVG eşleşmesi ve blob temizliği | Yerel uygulama tamamlandı; sorun fotoğrafı ve bağımsız köşe doğrulaması bekliyor |
| W11 | Bildirilen eğik fotoğrafı gerçekten düzeltmek | Orijinal kullanıcı görselinde hata üretimi; gölge/konkav kontur/kenar desteği düzeltmesi; aday ile kabul edilmiş OCR ayrımı | Orijinal girdide gerçek API + WebKit geçti; dörtgen, iki preview ve doğru kontrol gerektiren aday görünür; 10 geliştirme görselinde 9 kabul + 1 doğru aday | Bu örnek doğrulandı; bağımsız köşe doğruluğu ve geniş koşul kapsamı bekliyor |

W9 ayrıntıları ve kalan işler: [perspektif düzeltme geliştirme raporu](../reports/plate-face-geometry-development.md).
Köşe annotation/eğitimi ertelenmiş durumdadır; W9 bunların tamamlandığı anlamına gelmez.

## Çalıştırma seçimi

2026-10-15 yeni karar: kontur tahmini genel perspektif çözümü sayılmaz.
[Öğrenilmiş köşe detector geçiş planı](learned-corner-detector-cpu-migration.md)
önceliklidir. Perspektif modunda dörtgen bulunamazsa bbox OCR fallback'i kapatıldı;
model geçişi tamamlanmadı. Yeni fotoğraf sağlamak kullanıcının önkoşulu değildir.

- Inference: `ANPR_WEB_OCR_CONFIG` değerini okunabilir, mutlak JSON dosyası yoluna ayarla.
  Şablon: `configs/public/web-ocr-runtime-template.json`. Şablondaki yollar kurulmuş runtime'a göre değiştirilmeli.
- Web: `ANPR_RECOGNITION_PROVIDER=paddleocr`. İki servisin ayarı birlikte değiştirilir.
- JSON model dizini hash'i başlangıçta kontrol edilir. Worker ilk gerçek OCR isteğinde açılır;
  `/readyz` tek başına Paddle bağımlılıklarının çalıştığını kanıtlamaz.
- Worker Python ortamı ana PyTorch ortamından ayrıdır. Mevcut `runtimes/paddleocr-worker/requirements.txt`
  sürümleri korunur. macOS sanal ortamı Linux'a kopyalanmaz.
- Worker ve ayrı Linux Python ortamını ekleyen Dockerfile hazırdır. Build ve çalıştırma
  adımları [paketleme tarifinde](../operations/paddleocr-low-resource-release.md) bulunur.
  Mevcut canlı image'lar bu ortamı içermez; yeni iki image birlikte build edilmelidir.
- Yeni yayın şablonu `backend=paddle_inference` seçer. Eski config'ler `backend`
  içermiyorsa genel PaddleOCR yolunu seçmeye devam eder. Slim image'da bu genel
  yol kurulmaz; rollback eski image'lara veya project OCR seçimine yapılır.
- Tek servis worker'ı ve tek replica kullanılır. OCR thread sayısı 1; ana detector CPU thread sayısı 1.
  CPU thread sınırı RAM sınırı değildir; container CPU/RAM sınırları ayrıca uygulanır.
- 0.99 eşiği başlangıç tercihidir; %99 doğruluk garantisi değildir. Daha çok sonucu reddedebilir.
  Okunamayan sonucu kesin plaka olarak göstermemek, her görselde sonuç döndürmekten önceliklidir.

## Yayın öncesinde kalan zorunlu işler

| İş | Neden kaldı / yapılacak | Kabul ölçütü | Tahmini aktif efor |
| --- | --- | --- | --- |
| Sunucu bütçesi | Toplam ve kullanılabilir RAM, CPU ve mimari bilgisi alınmalı | Diğer servisleri sıkıştırmadan ölçüme uygun bütçe; değerler TBD | 15–30 dakika |
| Linux build ve smoke | Dockerfile, wheel-only hash kilidi, minimal context ve model mount hazır; yerel Docker daemon kapalı | Linux CPU import ve gerçek upload başarılı; image kimlikleri ve üretilen lisans envanteri kayıtlı | Builder hazırsa 0.5 gün, indirme/build hariç |
| Sunucuda kaynak kontrolü | Yerel 60 saniye idle ve restart geçti; Linux container toplam bellek/CPU bilinmiyor | OOM yok; idle sonrası worker yok; inference+web bütçe içinde; gecikme hedefi TBD | 30–60 dakika |
| Temsilî görsel kontrolü | Mevcut 10 fotoğraf/3 plaka geçti; bağımsız gündüz/gece/eğik/küçük/plakasız görseller ayrıca gerekli | Exact match, kabul edilenlerde hata ve reddetme oranı ayrı raporlanır; hedef TBD | Veri hazırsa 1–2 saat |
| Yayın onayı | Mevcut image'lar korunarak iki servis birlikte değiştirilir | Tarayıcı upload testi başarılı; geri dönüş denenmiş | 30–60 dakika |

Önceki yerel ölçümde yalnızca Paddle worker yaklaşık 866 MiB RSS kullanmıştı.
Bu turdaki gerçek HTTP ölçümünde ana süreç + worker birlikte yaklaşık 1491.67 MiB
(1.46 GiB) örneklenen toplam RSS kullandı. Idle worker çıkışı sonrası ana süreç
492.56 MiB kaldı; web belleği dahil değildir. Ayrıntılar:
[kısa CPU profil raporu](../reports/web-ocr-low-resource-profile.md).
Sonraki native çalışma yolunda toplam örneklenen RSS 1328.66 MiB (yaklaşık 1.30 GiB)
oldu; tasarruf yaklaşık %10.9, web belleği dahil değil. Ayrıntılar:
[native runtime karşılaştırması](../reports/paddle-native-runtime-comparison.md).
Bu, tüm servis toplamı veya Linux tepe bellek garantisi değildir. Az RAM'li sunucuda uygunluğu
ölçmeden “rahat çalışır” denemez. Bütçe yetmezse canlı geçiş yapılmaz; küçük runtime/model
denemesi veya dış servis ancak maliyet/veri paylaşımı kararı sonrası ayrıca değerlendirilir.

## Rollback

1. Yayın öncesi mevcut iki image kimliğini ve ayarlarını kaydet; eski model bundle'ını koru.
2. Yeni sürümde OOM, timeout veya belirgin doğruluk kaybında yeni trafik kabulünü durdur.
3. Web ve inference'ı birlikte önceki image/ayarlarına geri al.
4. Yeni kodla legacy moda dönülecekse inference'ta `ANPR_WEB_OCR_CONFIG` kaldırılır,
   web'de `ANPR_RECOGNITION_PROVIDER=project` seçilir; servisler yeniden başlatılır.
5. Readiness ve gerçek upload testi yap. Otomatik iki-OCR fallback yoktur; RAM'i ikiye katlamaz.

## Bu turdaki doğrulama sonuçları

- Python genel suite: 377 geçti, 2 özel-model testi varsayılan olarak atlandı.
- Yeni özel-model smoke testi ayrıca etkinleştirilerek çalıştırıldı: 1 geçti (iki görsel).
  Test worker'ın gerçekten çalıştığını, boş/kararsız sonuçta çekimser kalmayı,
  döndürülen metnin ilgili geliştirme hedefiyle eşleşmesini ve kapanışta worker çıkışını denetler.
  İki görselin mutlaka ikisinde de metin döndürülmesini şart koşmaz; doğruluk benchmark'ı değildir.
- Web: production build, TypeScript ve lint başarılı; 3 web akış testi geçti.
- Mypy: 118 kaynak dosyası başarılı; yeni Python dosyaları lint kontrolünden geçti.
- Public validator: 357 dosya, sözleşmeler ve doküman bağlantıları başarılı.

## Devam turundaki doğrulama

- Paddle backend'e `cpu_threads=1` doğrudan verildi; kütüphanenin varsayılanı 10'dur.
- Kaynak sözleşmesi en fazla iki model taşıyan süreç (detector + OCR worker), idle
  durumda bir süreç olarak düzeltildi. Tek eşzamanlı inference ve sıfır kuyruk korunur.
- Linux amd64 bağımlılık kilidi derleme/weight indirmeden çözüldü; Compose override
  ana dosyayla birleştirilerek Docker CLI tarafından doğrulandı.
- Minimal build context ve model mount testleri eklendi. Linux image build/çalıştırma
  testi yerel Docker daemon erişilemediği için tamamlanmadı.
- Gerçek HTTP profilinde 10/10 geliştirme sonucu ve plakasız kontrol geçti;
  60 saniye worker çıkışı ve ayrı gerçek yeniden açılış kontrolü geçti.
- Son kontrol: 380 Python testi geçti, 2 özel-model testi varsayılan olarak atlandı;
  gerçek model HTTP profilleri ayrıca çalıştırıldı. Mypy 119 dosyada başarılı;
  public validator 364 dosya ve 63 doküman bağlantısını doğruladı. Diff whitespace
  kontrolü geçti. Büyük Docker build, CUDA ve dataset export çalıştırılmadı.

## Daha hafif runtime turu

- Native yayın adayı aynı PaddleOCR ağırlıklarını doğrudan Paddle Inference ile
  çalıştırır. 221 kırpımda ham metin uyuşmazlığı 0, en büyük güven farkı 0.0 bulundu.
- Aynı 10 geliştirme fotoğrafı ve plakasız kontrol geçti. Örneklenen toplam RSS
  1491.67 MiB'den 1328.66 MiB'ye indi. Warm upload medyanı bu ayrı koşuda
  257.55 ms'den 289.63 ms'ye çıktı; hız üstünlüğü iddia edilmiyor.
- Linux native CPU bağımlılık kilidi 17 dağıtım içerir; önceki genel OCR kilidi 70'tir.
  Yeni aday Dockerfile ve config native yolu açıkça seçer; eski config varsayılanı korunur.
- Tam CI tip kontrolünde test tanımlarındaki sorunlar düzeltildi; önceki eklemelerdeki
  45 dosyanın biçimlendirmesi mekanik olarak düzenlendi. Çalışma algoritmaları bu
  biçimlendirme değişiklikleriyle değiştirilmedi.
- Son kontroller: 384 Python testi geçti, 2 özel-model testi normal suite'te atlandı;
  gerçek HTTP/native parity kontrolleri ayrıca yapıldı. CI mypy kapsamı 146 dosyada,
  src+scripts kapsamı 120 dosyada başarılı. Lint ve 171 dosyalık format kontrolü
  geçti. İki generated contract doğrulandı. Public validator 370 dosya/66 yerel
  bağlantıyı doğruladı. Compose birleşimi ve diff whitespace kontrolü geçti.
- Linux build, sunucuda ölçüm, canlı yayın, commit ve push henüz yapılmadı.

## Mac yerel website turu

Gerçek browser UI → Next BFF → Python detector → native Paddle OCR → UI akışı
tamamlandı. Yerel HTTP session cookie'nin Secure flag'i localhost moduyla
sınırlandırılarak düzeltildi; venv Python symlink'inin korunması için başlatıcı
düzeltildi. İki farklı plaka, plakasız kontrol ve oturumu temizleme geçti.
Üretim HTTPS cookie güvenliği korunur. Kullanım ve kalan işler:
[Mac yerel website rehberi](../operations/macos-local-website.md).
Son kontrol: 386 Python testi ve 4 web testi başarılı; Python/TypeScript tip,
lint, format, build, public boundary ve shutdown-port kontrolleri geçti. Test
sonunda yerel web/inference/OCR süreçleri ve bu tur açılan tarayıcı kapatıldı.

## Kapsam dışı işler ve yetki sınırı

CUDA, yeni model eğitimi, toplu PNG export, kalan corner annotation ve uzun soak testi bu
web hedefi için zorunlu değildir. Bunlar çalıştırılmadı. Yeni bir YOLO ağına geçiş yapılmadı;
mevcut detector korunmuştur. Bağımsız doğruluk kanıtı olmadan “hatasız” iddiası yapılmaz.
Canlı deploy, commit, push veya PR yapılmadı. Tüm testler ve kullanıcı incelemesi sonrası
yalnızca açık izinle tek commit; push için ayrıca izin gerekir.
