# Paylaşılan sunucuda sınırlı Linux build ve inference RAM testi

Tarih: 2026-10-23. Bunlar kullanıcının sunucuda çalıştıracağı kontrollü test
adımlarıdır; gerçek Linux build henüz burada çalıştırılmadı. Mevcut Coolify
uygulamasını yeniden dağıtmaz, port açmaz, commit/push veya cache temizliği yapmaz.
Build sıfır etkili değildir: disk/network I/O ve Docker daemon image yükleme
maliyeti builder limitleri dışında da olabilir. Trafiğin az olduğu saati seç.
En güvenli seçenek ayrı builder'dır; aşağıdaki yöntem aynı host'ta sınır koyar.

İlk sunucu denemesi son import adımında `detection.quadrilateral` modülü
pakette eksik olduğu için başarısız oldu. Allowlist düzeltildi. Export edilmiş
context'te, tam geliştirme checkout'una fallback yasaklanarak gerçek import
testi eklendi; eksik dosya senaryosu da negatif kontrolde tekrar üretiliyor.
Bu hata RAM/OOM değildir. Yeni v2 kaynak paketi ve yeni source/context dizinleri
kullanılır; aynı builder cache'i korunur. Eski kaynaklar ve canlı site değişmez.

## A. Mac: güncel test kaynak paketini gönder

Test paketi `anpr-corners-source-20261023-linux-test-v1.tar.gz` adındadır.
Kaynak paketi küçük; photo/model/secret içermez. SCP ile sunucuya gönder,
ardından `ssh root@SERVER_IP` ile bağlan. Şifreyi komuta yazma. Yeni SSH
fingerprint'ini sağlayıcı kaydıyla doğrula; eski site dizinine açma.

## B. Sunucu: ön kontrol ve yeni çalışma alanı

```sh
free -m
ANPR_DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}')
df -h "$ANPR_DOCKER_ROOT"
docker buildx version
```

Operasyonel emniyet payı olarak en az 6 GiB available RAM ve 15 GiB boş Docker
disk alanı yoksa dur. Bunlar ölçülmüş minimum gereksinim değil, paylaşılan
sunucuyu koruyan başlangıç eşikleridir. Buildx yoksa otomatik kurulum yapma.

```sh
mkdir /root/anpr-lean-lab-20261023
cd /root/anpr-lean-lab-20261023
mkdir source models test
tar -xzf /root/anpr-lean-test-source-20261023.tar.gz -C source
python3 source/scripts/prepare_phase2d_context.py inference-paddleocr-amd64 /root/anpr-lean-lab-20261023/context
```

Her komut başarılı olmalı; dizin zaten varsa veya tar/context hatası varsa dur,
var olan dosyanın üzerine açma. Context'te yalnız allowlisted kod/config vardır.

## C. Yalnız bu test için ayrı, sınırlı builder

```sh
cd /root/anpr-lean-lab-20261023
docker buildx create --name anprlean20261023test \
  --driver docker-container \
  --driver-opt memory=3g --driver-opt memory-swap=3g \
  --driver-opt cpu-period=100000 --driver-opt cpu-quota=100000 \
  --driver-opt restart-policy=no \
  --buildkitd-config source/deploy/docker/buildkit.low-resource.toml \
  --bootstrap unix:///var/run/docker.sock
```

`--use` yoktur; Coolify/default builder seçimi değiştirilmez. Builder adı zaten
varsa değiştirmeye çalışma. Max parallelism 1, bu builder'da otomatik GC kapalı.
Yalnız bu builder'ın kendi cache/metadata alanı oluşur; disk üst sınırı değildir.

Gerçek container limitlerini build öncesinde kontrol et:

```sh
docker inspect --format 'MEM={{.HostConfig.Memory}} MEMSWAP={{.HostConfig.MemorySwap}} PERIOD={{.HostConfig.CpuPeriod}} QUOTA={{.HostConfig.CpuQuota}}' buildx_buildkit_anprlean20261023test0
```

Beklenen MEM/MEMSWAP 3221225472, PERIOD/QUOTA 100000. Farklıysa build başlatma.

## D. Sadece lean inference image'ını build et

Önce bu yeni image etiketi mevcut mu bak; mevcutsa başka sürümü ezme:

```sh
docker image ls anpr-engine-inference:lean-lab-20261023
```

Satır varsa dur. Yoksa:

```sh
docker buildx build --builder anprlean20261023test \
  --platform linux/amd64 --load --progress=plain \
  -f context/deploy/docker/inference.lean-corners.amd64.Dockerfile \
  -t anpr-engine-inference:lean-lab-20261023 context
```

Web/Next image'ı bu tur build edilmez. Build bir CPU ile yavaş olabilir; süre
TBD. Image/disk/cache alanı büyüyebilir. RAM yetersizliği veya sunucu trafiğinde
belirgin yavaşlama varsa Ctrl+C, sonra yalnız bu builder için stop kullan.

Build başarılı veya başarısız olduktan sonra builder RAM'ini bırak:

```sh
docker buildx stop anprlean20261023test
docker image inspect --format 'ID={{.Id}} SIZE_BYTES={{.Size}}' anpr-engine-inference:lean-lab-20261023
```

Stop cache'i veya image'ı silmez. Genel prune/rm/down/restart kullanma.

## E. Mac: yalnız iki model ve bir izinli test fotoğrafını gönder

Sunucudaki yeni `models`/`test` dizinlerine native Paddle `inference/` dizinini,
640 ONNX dosyasını ve tek JPEG fotoğrafı SCP ile aktar. Eski `.pt` bundle,
dataset, venv veya node_modules gönderme. Paket/model dosyaları site tarafından
servis edilmez. Test fotoğrafı izinli olmalı; çıktı plaka metnini yazmaz.

Hedef düzen:

```text
/root/anpr-lean-lab-20261023/
  models/inference/inference.json
  models/inference/inference.pdiparams
  models/inference/inference.yml
  models/y5fu_640x_sim.onnx
  test/plate.jpg
```

Sunucuda yalnız bu yeni kopyaların container UID'si tarafından okunmasını sağla:

```sh
cd /root/anpr-lean-lab-20261023
find models/inference -type d -exec chmod 755 {} +
find models/inference -type f -exec chmod 644 {} +
chmod 644 models/y5fu_640x_sim.onnx test/plate.jpg
```

Başka model veya uygulama dizininde chmod çalıştırma. Runtime hash kontrolü
eksik/bozuk/fazladan model dosyası varsa testi durdurur; dosya silerek atlatma.

## F. Ağsız, portsuz, tek container test

```sh
docker run --name anprleanram20261023 \
  --cpus=0.75 --memory=2g --memory-swap=2g --pids-limit=96 \
  --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=128m,uid=10001,gid=10001 \
  --log-driver=json-file --log-opt max-size=5m --log-opt max-file=3 \
  --mount type=bind,src=/root/anpr-lean-lab-20261023/models/inference,dst=/models/paddleocr/inference,readonly \
  --mount type=bind,src=/root/anpr-lean-lab-20261023/models/y5fu_640x_sim.onnx,dst=/models/corners/y5fu_640x_sim.onnx,readonly \
  --mount type=bind,src=/root/anpr-lean-lab-20261023/test/plate.jpg,dst=/test/plate.jpg,readonly \
  anpr-engine-inference:lean-lab-20261023 \
  python /app/runtimes/lean-smoke/smoke.py --image /test/plate.jpg
```

Image zaten yerelde olmalı; test öncesi inspect başarılı olmalı. İsim varsa
üzerine yazma. Docker komutuna `-p`/`--privileged`/`--restart` ekleme.
Test kısa bir model isteği + warm istek, 16 saniyelik idle kontrolü, restart ve
plakasız kontrol yapıp çıkar. HTTP sunucusu açmaz; aynı ASGI request akışı
bellekte çağrılır. 2 GiB bir limit, sürekli kullanım veya başarı garantisi değildir.

Çıktıda `cold/warm/after_idle/after_close` ve memory current/peak MiB alanları
vardır. Cgroup değeri ana süreç + OCR worker + charged file cache'i içerir;
process RSS ile aynı ölçüm değildir. Alan null ise ölçüm yoktur, sıfır tüketim
değildir. Reading present/consistent doğruluğu kanıtlamaz; accuracy_benchmark false.

## G. Son kontrol ve paylaşılacaklar

```sh
docker inspect --format 'EXIT={{.State.ExitCode}} OOM={{.State.OOMKilled}}' anprleanram20261023
free -m
```

Başarıda EXIT=0, OOM=false ve checks alanları olumlu olmalı. `torch_loaded`
ve `legacy_bundle_used` false olmalıdır. OOM/137/failed olursa otomatik
yeniden deneme veya limit artırma yapma; sonuçları incele. Container durmuş
olarak kalır; inceleme bitmeden silinmez. Image/model/build cache kalır,
çalışan test servisi/port kalmaz. Canlı ANPR veya diğer uygulamalara stop/rm yok.

Yalnız image boyutu/kimliği, smoke JSON'u, EXIT/OOM ve gerekirse hassas bilgi
içermeyen build hata satırlarını paylaş. Tam inspect/env/config dökümü paylaşma.

Resmi referanslar: [builder limitleri](https://docs.docker.com/build/builders/drivers/docker-container/),
[BuildKit max parallelism](https://docs.docker.com/build/buildkit/configure/).
