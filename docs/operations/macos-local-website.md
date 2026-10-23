# Mac üzerinde düşük yükle yerel website

Yerel website akışı 2026-10-14 tarihinde Mac üzerinde gerçek WebKit tarayıcısı,
production Next.js çıktısı ve gerçek detector + native Paddle OCR worker ile doğrulandı.
Linux veya Docker yerel geliştirme için gerekli değildir.

2026-10-15 güncellemesi: hazır private başlatıcı artık köşeleri doğrudan tahmin
eden HyperLPR 640 ONNX detector + mevcut native Paddle OCR kullanır. Köşeler
konturdan üretilmez. Model seçimi, kısa testler, kaynak tüketimi ve kalan yayın
kriterleri [öğrenilmiş köşe raporunda](../reports/learned-corner-cpu-selection.md).
Aşağıdaki önceki test sayıları ve yüzey raporu tarihsel kontrol kaydıdır;
güncel sonuç 414 Python / 6 web testi ve gerçek model/tarayıcı kontrolüdür.
Geri dönüş için aynı private dizindeki `Start Local Website Legacy.command`
korundu; iki başlatıcı aynı anda çalıştırılmaz. Standart başlatıcıyla yerel
site yeniden açıldı. İşin bitince Terminal'de Ctrl+C ile kapatabilirsin.

## Açma ve kapatma

Private çalışma alanında `Start Local Website.command` başlatıcısı hazırlandı.
Finder'da çift tıklanabilir veya Terminal'de dosyanın tam yolu çalıştırılabilir.
Başlatıcı adresi ekrana yazar: `http://127.0.0.1:39084`.
Terminal'de Ctrl+C ile website, inference servisi ve varsa OCR worker kapanır.
Bu test turunun sonunda servisler ve test tarayıcısı durduruldu.

Genel başlatıcı, private dosya yollarını public kaynaklara gömmeden çalışır:

```sh
.venv/bin/python scripts/launch_local_website.py \
  --model-bundle /absolute/private/model-bundle \
  --ocr-config /absolute/private/mac-ocr-config.json
```

OCR JSON'undaki Python, worker ve model dizini Mac'te var olmalı. Linux `/opt`
yollarını içeren public şablon doğrudan Mac'te kullanılmaz. Private hazır config
mevcut Paddle venv'i ve modeli gösterir. Venv Python symlink'i korunur; gerçek
base Python yoluna çözülmez. Başlatıcı CPU thread'i 1 ve backend'i
`paddle_inference` olarak seçer; yerel idle varsayılanı 15 saniyedir.

Web kaynakları değişirse, servisleri kapattıktan sonra bir kez build yap:

```sh
cd apps/web
NODE_OPTIONS=--max-old-space-size=768 NEXT_TELEMETRY_DISABLED=1 npm run build
```

Next config build worker sayısını 1'e sınırlar. Başlatıcı dev server yerine
standalone production çıktısını açar ve küçük statik asset dizinini bu çıktıya
kopyalar. Her açılışta build veya paket yüklemesi yapmaz.

## Normal kullanım

1. İstersen Türkçe düğmesine bas.
2. Bildirim kutusunu işaretle ve tek JPEG/PNG araç fotoğrafı seç.
3. “Görüntüyü işle”ye bas. Plaka sınırını ve metnini kontrol et. Köşeler
   tahmin edildiyse dörtgen, bulunamadıysa açık uyarıyla kesikli yaklaşık kutu görünür.
   Dörtgen varsa sağda “Perspektifi düzeltilmiş plaka” ve “OCR’a verilen kırpım”
   önizlemeleri açılır. Dar ekranda bunlar özgün fotoğrafın altında görünür.
   OCR kabul eşiğini geçmediyse metin “OCR adayı / kontrol gerekli” olarak
   gösterilebilir. Bu otomatik kabul değildir; uyarı ve unresolved durumu korunur.
   “Tahmini onayla” kapalı kalır; görselden kontrol ettiğin metni düzeltme
   alanıyla kendin kaydedebilirsin.
4. Yeni fotoğraf seçebilir veya “Geçerli oturumu temizle”ye basabilirsin.
5. İşin bitince Terminal'de Ctrl+C'ye bas.

Yerel quota yalnızca bu çalıştırmanın geçici dizinindedir; her açılışta sıfırlanır.
Yerel başlatıcı, kendi `127.0.0.1` istemcisini günlük kotadan muaf tutar;
Mac üzerinde beş deneme sınırı yoktur. Servis yalnızca loopback'e bağlıdır.
Paylaşılan sunucudaki kalıcı quota ve diğer IP'lerin sınırı değişmedi; canlıda
istisna yalnızca `ANPR_RATE_LIMIT_EXEMPT_IPS` içindeki tam, güvenilir istemci
IP adreslerine uygulanır. Forwarding header'ı göndererek muafiyet alınamaz.
Secret süreç environment'ında paylaşılır; OCR config dosyasına veya public repo'ya
yazılmaz. Geçici OCR config/quota dosyaları kapanışta temizlenir.

## Doğrulama ve kaynaklar

2026-10-15 perspektif geliştirmesi tamamlandı; ayrıntılar ve sınırlamalar
[plaka yüzeyi raporunda](../reports/plate-face-geometry-development.md).
Bu yeni sürüm için web build yenilendi. Önceden açık servisleri Ctrl+C ile
kapatıp aynı başlatıcıyı yeniden aç; eski süreçler yeni Python kodunu kendiliğinden yüklemez.

| Kontrol | Sonuç |
| --- | --- |
| Türkçe UI'da iki farklı plaka | HTTP 200; beklenen metin; doğru metin UI'da görünür; tespit kutusu var |
| Sentetik plakasız fotoğraf | HTTP 200; plaka metni yok; tespit kutusu yok |
| Oturumu temizleme | Önceki sonuç kaldırıldı |
| Gerçek yerel HTTP session cookie | HttpOnly + SameSite strict; HTTPS-only flag yalnızca yerel modda kaldırılır |
| Public/HTTPS cookie | Secure flag korunur; regresyon testinde doğrulandı |
| Tarayıcı konsolu | Son başarılı koşuda hata/uyarı yok |
| OCR idle kapanışı | 15 saniye sonrasında worker kalmadı |
| Başlatıcı kapanışı | Kendi süreçleri kapatıldı; başka süreçlere dokunulmadı |

Son otomatik kontroller: 386 Python testi geçti, 2 private-model testi normal
suite'te atlandı; gerçek tarayıcı/model kontrolü ayrıca yapıldı. Dört web akış
testi, web build/lint/TypeScript, Python lint/format/mypy ve public validator geçti.
Kapanıştan sonra 39084 ve 39085 portlarının kapalı olduğu ayrıca doğrulandı.

Node heap üst sınırı 512 MiB'dir; Node toplam RSS sınırı değildir. Inference
tek CPU thread, tek worker ve sıfır kuyruk kullanır. Model ilk plaka isteğinde
yüklenir; yerel toplam RAM hard-limit'i uygulanmadığı için model yükü devam eder.
Önceki kısa native profilde ana Python + OCR worker yaklaşık 1.30 GiB örneklenen
RSS kullandı; web ve tarayıcı buna dahil değildi. Bu tur yeni peak benchmark'ı
yapılmadı. Uzun soak, CUDA, training veya toplu görüntü export yapılmadı.

## Kalan işler

| İş | Öncelik / ne zaman | Neden |
| --- | --- | --- |
| Kendi gerçek fotoğraflarınla kısa kontrol | İstediğinde; 3–5 fotoğrafı sırayla | Mevcut dev set küçük; gece/eğik/küçük plakaları temsil etmiyor |
| Sunucudaki kullanılabilir RAM ve CPU | Yayın öncesi | Mac ölçümü sunucunun kapasitesini kanıtlamaz |
| Linux image build ve upload testi | Linux hosting'e yayın zamanı | Yerel Mac akışının önkoşulu değildir |
| Canlı yayın ve rollback kontrolü | Paket/sunucu kontrolü sonrası | Yeni iki servis birlikte yayınlanmalı |
| Commit / push | Kullanıcının açık izniyle | Tek commit için izin; push öncesi ayrıca izin |
