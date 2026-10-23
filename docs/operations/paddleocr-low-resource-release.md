# PaddleOCR düşük kaynaklı web sürümü

Bu dosya yeni sürümün Linux amd64 paketleme tarifidir. Mevcut canlı Compose image
kimlikleri korunmuştur. Yeni image henüz Linux'ta build edilip çalıştırılmamıştır:
yerel Docker daemon erişilebilir değil. Yerel makinedeki yaklaşık 14 GiB boş alan
üzerinde büyük image/weight arşivleri üretilmedi.

2026-10-17: öğrenilmiş dört köşe için ek opt-in config, read-only ONNX mount'u
ve eşleşen web provider [köşe yayın tarifinde](learned-corners-release.md).
Bu hazırlık mevcut canlı image seçimini veya legacy varsayılanını değiştirmez.

## Hazır dosyalar

- `deploy/docker/inference.paddleocr.amd64.Dockerfile`: güncel kaynaklı CPU inference
  image'ına ayrı `/opt/paddle-venv` ortamını ekler. Worker ve config içerir; weight içermez.
- `deploy/docker/requirements.paddle-native-amd64.txt`: Python 3.12 / Linux x86_64 için
  wheel-only çözülmüş, transitif sürümleri ve SHA256 hash'leri sabitlenmiş bağımlılıklar.
  PaddlePaddle 3.3.1 ve headless OpenCV kullanır; 17 dağıtım içerir, GPU paketi yoktur.
  Genel yol için eski `requirements.paddleocr-amd64.txt` kilidi (70 dağıtım) korunur.
- `runtimes/paddleocr-worker/native_backend.py`: mevcut modelin doğrudan CPU inference
  adapter'ı; ağ ve ağırlıklar değişmez. 221 kırpımda genel worker ile aynı metin/güven
  çıktılarını verdi. Yayın config'i `backend=paddle_inference` seçer.
- `deploy/docker/compose.paddleocr-candidate.yaml`: web ve inference image'larını
  birlikte seçer; Paddle model dizinini read-only mount eder. Mevcut CPU/RAM/PID
  limitleri, özel ağ ve tek worker sınırı ana Compose dosyasından gelir.
- `scripts/profile_web_ocr.py`: gerçek loopback HTTP upload ile en fazla 10
  geliştirme görselini işler; bellek, gecikme, kabul/ret ve idle kapanışını kaydeder.
  Görseller yalnızca bellekte JPEG'e dönüştürülür. Rapor plaka metni/görsel/yol içermez.

Linux kilidi Python 3.12 kullanır; mevcut macOS testi Python 3.11'dir. Platforma
bağlı bağımlılık çözümü farklı olabilir (native kilidinde NumPy 2.2.6). Kilidin
çözülmesi image'ın çalıştığını kanıtlamaz; Linux gerçek-model testi gereklidir.

## Linux image oluşturma

Build'i yeterli alanı olan ayrı bir builder üzerinde, sırayla yap. Kaynak kökünde:

```sh
ANPR_PADDLE_CONTEXT=$(mktemp -d)
ANPR_WEB_CONTEXT=$(mktemp -d)
python scripts/prepare_phase2d_context.py inference-paddleocr-amd64 "$ANPR_PADDLE_CONTEXT"
python scripts/prepare_phase2d_context.py web "$ANPR_WEB_CONTEXT"

docker build --platform linux/amd64 \
  -f "$ANPR_PADDLE_CONTEXT/deploy/docker/inference.amd64.Dockerfile" \
  -t anpr-engine-inference:web-cpu-base "$ANPR_PADDLE_CONTEXT"
docker build --platform linux/amd64 \
  --build-arg ANPR_INFERENCE_BASE_IMAGE=anpr-engine-inference:web-cpu-base \
  -f "$ANPR_PADDLE_CONTEXT/deploy/docker/inference.paddleocr.amd64.Dockerfile" \
  -t anpr-engine-inference:paddleocr-candidate "$ANPR_PADDLE_CONTEXT"
docker build --platform linux/amd64 \
  -f "$ANPR_WEB_CONTEXT/deploy/docker/web.Dockerfile" \
  -t anpr-engine-web:paddleocr-candidate "$ANPR_WEB_CONTEXT"
docker image inspect --format '{{.Id}}' anpr-engine-inference:paddleocr-candidate
docker image inspect --format '{{.Id}}' anpr-engine-web:paddleocr-candidate
```

Base image güncel koddan yeniden oluşturulmalıdır: eski canlı inference image'ı
OCR seçimini içermediği için kullanılamaz. Dockerfile ayrı OCR venv'inin import
kontrolünü build sırasında yapar. `/licenses/paddleocr` kurulu dağıtımlara ait
lisans ve native artifact envanterini taşır. Build çıktıları ve bu envanter
incelenmeden yayınlanmış/uyumlu olarak işaretlenmez.

## Çalıştırma ayarları

Yeni iki image'ın doğrulanmış `sha256:` kimliklerini `ANPR_PADDLE_INFERENCE_IMAGE`
ve `ANPR_PADDLE_WEB_IMAGE` değişkenlerine koy. Paddle inference dizinini sunucuda
ayrı tut ve `ANPR_PADDLE_MODEL_SOURCE` olarak ver. Dizin üç inference dosyasını
ve şablonda belirtilen dizin hash'ini karşılamalıdır. Eski model bundle mount'u
da gereklidir; legacy OCR ağı Paddle modunda RAM'e yüklenmez.

Native slim image'da PaddleOCR/PaddleX kurulmaz. `backend=paddleocr` seçimi ancak
tam eski OCR ortamı veya tam OCR image'ıyla çalışır. `backend` alanı olmayan eski
JSON config'ler geriye uyumluluk için genel PaddleOCR yolunu seçer; yeni şablonda
bu alan açıkça `paddle_inference` olarak verilmiştir. Adapter yalnızca mevcut
Latin CTC/48-piksel modelini destekler.

Sunucu mimarisi, kullanılabilir RAM ve mevcut iki image'ın rollback kimlikleri
kaydedilir. Coolify'da aşağıdaki iki Compose dosyasının birleşmiş çıktısı kullanılır:

```sh
docker compose -f deploy/docker/compose.public-candidate.yaml \
  -f deploy/docker/compose.paddleocr-candidate.yaml config --quiet
```

Mevcut credential ve sürüm değişkenleri korunur. Kontrol çıktısını credential
değerleriyle birlikte kaydetme. Ana Compose limitleri inference 2048 MiB / 0.75 CPU,
web 512 MiB / 0.25 CPU'dur; bunlar mevcut üst sınırlardır, ölçülmüş ihtiyaç değildir.
Toplam kullanılabilir bütçe yetmiyorsa bu sürüm başlatılmaz. RAM sınırını daha küçük
seçmek modelin ihtiyacını azaltmaz; işlem sırasında OOM oluşabilir.

## Kısa ölçüm ve kabul

Özel geliştirme manifesti, inventory ve dataset root'u ile:

```sh
python scripts/profile_web_ocr.py \
  --model-bundle "$ANPR_PROFILE_BUNDLE" --ocr-config "$ANPR_PROFILE_OCR_CONFIG" \
  --manifest "$ANPR_PROFILE_MANIFEST" --inventory "$ANPR_PROFILE_INVENTORY" \
  --dataset-root "$ANPR_PROFILE_DATASET_ROOT" --limit 10 --output "$ANPR_PROFILE_REPORT"
```

Config yolları ilgili ortamda çalışır Python, worker ve model dizinini göstermeli.
Script en fazla 60 saniye idle bekler, sonra test servisini ve worker'ı kapatır.
Gecikme ölçümüne upload/decode/detection/OCR dahildir. Rapor yalnızca geliştirme
regresyonu içindir; Linux'ta bile bağımsız üretim doğruluk ölçümü değildir.

RSS ölçümü ana Python süreci ve OCR worker toplamını 250 ms aralıklarla örnekler;
shared sayfaları iki kez sayabilir, anlık tepeyi kaçırabilir ve web belleğini içermez.
Linux container denemesinde ayrıca inference+web toplamını `docker stats` ile,
OOM durumunu container inspect ile ve container peak belleğini cgroup ile ölç.

Yerel native ölçüm yaklaşık 1328.66 MiB toplam örneklenen RSS gösterdi; web
dahil değildir. Önceki yola göre azalma yaklaşık %10.9'dur; az RAM'li sunucuya
uygunluğu hâlâ sunucu ölçümüyle belirlenir. Ayrıntılar:
[aynı modelin runtime karşılaştırması](../reports/paddle-native-runtime-comparison.md).

| Kontrol | Kabul |
| --- | --- |
| Gerçek upload | Detector bulunuyor; OCR modeli kimliği doğru; beklenen metin veya açık çekimser sonuç |
| Plakasız görsel | Plaka metni yok; sentetik beyaz kontrol ayrıca geçer |
| Kaynak | OOM yok; inference+web bütçeye sığar; CPU ve gecikme hedefi sunucu ölçümüyle TBD |
| Idle | 60 saniye sonunda OCR worker çıkmış; ana servis hazır |
| Yeniden istek | Worker tekrar açılır; model kimliği ve sonuç doğrulanır |
| Rollback | Önceki iki image/ayar birlikte geri alınabilir; upload tekrar geçer |

Yayın aşamasında yeni web ve inference birlikte değiştirilir. Hata olursa iki
image ve OCR ortam ayarları önceki değerlere döndürülür; eski bundle korunur.
Kodda legacy moda dönüş için inference'tan `ANPR_WEB_OCR_CONFIG` kaldırılır ve
web `ANPR_RECOGNITION_PROVIDER=project` olarak yeniden başlatılır. Diğer projelere
ve Docker genel kaynaklarına dokunulmaz.

Canlıya geçiş onayı bu tarifin build ve Linux testleri tamamlandıktan sonra alınır.
Kullanıcının önceki talebi uyarınca commit için açık izin; push için ayrıca izin gerekir.
