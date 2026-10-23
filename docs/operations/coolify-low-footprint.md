# Paylaşılan Coolify sunucusunda düşük RAM/disk kullanımı

Tarih: 2026-10-21. Kullanıcının sağladığı çıktıda Coolify ve çok sayıda uygulama
çalışıyor. Bu nedenle ayrı standalone Compose projesiyle ikinci ANPR kopyası
açılmaz; var olan ANPR uygulaması kesin olarak tanımlandıktan sonra onun web
ve inference servisleri birlikte güncellenir. Burada sunucu değişikliği yapılmadı.

23 Ekim güncellemesi: [PyTorch'suz lean runtime](lean-corners-runtime.md) artık
yerel aday olarak hazır; Linux build/image boyutu ve Coolify doğrulaması
hâlâ bekliyor. Aşağıdaki başlangıç ölçümleri ve eski yol notları tarihseldir.

## Hazırlanan varsayılanlar

| Kaynak | Seçim | Bedeli / sınırı |
| --- | --- | --- |
| OCR RAM | İlk istekte lazy yükleme; son OCR'dan 15 saniye sonra worker kapatma | Sonraki aralıklı istekte yeniden yükleme gecikmesi; ana Python/detector RAM'i sıfırlanmaz |
| Inference | Tek thread, tek aktif istek, kuyruk yok | Yoğun eşzamanlı istekler bekletilmeden reddedilir |
| Detector | Tek HyperLPR 640 ağı; normal 1 / miss halinde en fazla 3 geçiş | 320 adayı yanlış kabul nedeniyle seçilmedi; bellek uğruna kalite düşürülmez |
| OCR bağımlılıkları | Native Paddle yolu, tek worker; ağır PaddleOCR/PaddleX framework yolu kullanılmaz | Ana Python ortamında mevcut PyTorch bağımlılığı hâlâ var |
| Log diski | Learned overlay'de servis başına json-file 5m × 3 | İki servis için yaklaşık 30 MB log hedefi; metadata, son kayıt taşması ve harici Coolify saklaması ayrıca |
| Model dosyaları | Image dışında tek read-only kopya | Private bundle legacy doğrulama bağımlılığı nedeniyle hâlâ gerekli; silinmez |
| Kod aktarımı | Küçük allowlisted kaynak paketi | Dataset, PNG export, fotoğraf, venv, node_modules ve weight dahil değil |
| Çalıştırma | Start sırasında build/pull yok | İlk image hazırlığı ayrı builder'da, mevcut image'lar hedefe aktarılmalı |

15 saniye seçimi yalnızca learned-corner yayın JSON'una uygulanır. Eski
legacy config ve kullanıcıya ait private config'ler değiştirilmez. Güncel image
ve ayarlar Coolify'a uygulanıp container yeniden oluşturulmadan canlıda etkili
olmaz. Mevcut çalışan logları küçülttüğü veya silindiği iddia edilmez.

Log rotation referansı: [Docker json-file](https://docs.docker.com/engine/logging/drivers/json-file/).
Global Docker daemon ayarı değiştirilmez. Coolify'ın kendi log saklama/driver
ayarlarıyla uyum deployment öncesi kontrol edilir.

## Önce yalnızca ANPR'yi ölç

Container adları değişebilir. Mevcut listedeki web/inference çiftinin gerçekten
ANPR olduğundan image/Compose proje bilgisiyle emin ol. Sonra sadece o iki
container için `docker stats --no-stream` kullan. Tam `docker inspect` veya
environment dökümü paylaşma; secret içerebilir. Seçili image'ların boyutları
`docker image inspect` ile okunabilir. Image boyutları shared layer nedeniyle
toplanarak gerçek ilave disk kullanımı hesaplanamaz.

Şu anki container'ın boşta RSS'i ve image boyutunu bilmeden "şu kadar MB'a
indirdik" denmez. Yeni sürümde bir upload, 15 saniye idle ve tekrar upload ile
RSS/gecikme/worker çıkışı ölçülür. Mac'te eski profilde yaklaşık 1,24 GiB
Python+OCR ve idle sonrası yaklaşık 565 MiB vardı; web/tarayıcı dahil değildir,
Coolify bütçe kanıtı değildir. Bu tur yeni RSS profili çalıştırılmadı.

## Disk ve RAM için yapılmayacaklar

- `docker system prune`, genel volume temizliği, swap kapatma/cache boşaltma yok.
- Başka uygulamaya ait image/container/volume veya model/dataset dosyası silinmez.
- İnference 2 GiB / web 512 MiB tavanını test etmeden düşürmek optimizasyon
  sayılmaz; OOM'a yol açabilir. Bu tavan ayrılmış/sürekli kullanılan RAM değildir.
- Canlı paylaşılan host'ta ağır image build yapılmaz; rollback için önceki tek
  doğrulanmış ANPR sürümü korunur. Eski release/build cache temizliği ancak
  hedefler tek tek incelendikten ve kullanıcı açıkça izin verdikten sonra yapılır.

## Kalan gerçek optimizasyon

| İş | Durum |
| --- | --- |
| Ana runtime'ın PyTorch/legacy bundle bağımlılığını kaldıran ayrı lean yol | Yerel aday tamamlandı; gerçek-model smoke geçti, Linux build/ölçüm bekliyor |
| Linux image/rootfs ve bağımlılık boyutu | Hedef image'lar oluşturulmadan ölçülemez |
| Coolify mevcut servise uygulama | Kimlik/config eşleştirmesi ve yayın onayı bekliyor |
| Sunucuda idle/aktif kaynak profili | Mevcut iki container seçilip ölçülmeli; ardından gerçek bütçe belirlenir |

Bu ilk aşama "mümkün olan minimum" iddiası değil, kaliteyi değiştirmeden
gereksiz idle ve log büyümesini azaltan kontrollü ayarlardır.

Yerel doğrulama: **444 Python testi geçti, 3 private opt-in atlandı**, 12,07
saniye; thread sınırları 1. Mypy/Ruff, public validator, diff kontrolü ve üç
Compose dosyasının `config --quiet` birleşimi geçti. Docker daemon/image build,
canlı değişiklik, disk temizliği ve yeni RSS ölçümü yapılmadı. Eski paketler
korunarak düşük-tüketim ayarlı kaynak paketi ayrıca hazırlandı.
