# v9: mobil başlık + OCR, kontrollü Linux testi ve yayın

Bu rehber mevcut Coolify servisini test aşamasında değiştirmez. Yereldeki v9
kaynak paketi küçük ve modelsizdir. Git commit/push ile canlı yayın aynı işlem
değildir: aşağıdaki **Linux kapısı geçilmeden** Coolify yeniden başlatılmaz.
Push öncesinde ayrı kullanıcı onayı korunur. Projenin hatasız OCR veya bağımsız
holdout kabulü tamamlanmış değildir.

Mobil başlık tek satırdır. 420 px ve altında yalnız logo bırakılır, iletişim
bağlantısı korunur. Dil TR/EN, tema güneş/ay bölmeli seçicidir. İlk açılış TR ve
koyu; kullanıcının tema tercihi çerezle korunur. Referans:
[VibeSofters](https://vibesofters.com/tr). Referansta tema seçici görünmediğinden
tema için aynı bölmeli kontrol dili kullanılmıştır; birebir mevcut kontrol
kopyalandığı iddia edilmez.

Yerel yayın öncesi kontrol: 484 Python testi (4 özel opt-in varsayılan olarak
atlandı), 6 web testi, web üretim derlemesi/lint, 156 Python dosyası için mypy,
Ruff ve public sınır/sözleşme taramaları başarılı. Gerçek tarayıcıda 320, 375,
390, 430, 500, 768 ve 1280 px genişliklerinde iletişim bağlantısı logo ile aynı
satırda kaldı; yatay taşma görülmedi. TR/EN ve açık/koyu geçişleri, seçili durum
ve tema çerezi kontrol edildi. Bunlar henüz yeni sürümün Linux ölçümü değildir.

## 1. Hazırlık ve kapsam

- Public kaynak: kod, şablonlar, testler, lisanslar ve dokümanlar.
- Public dışında: model ağırlıkları, özel config/credential, fotoğraflar,
  dataset, venv, build çıktıları ve özel ölçüm kayıtları.
- Yeni etiketler: `anpr-engine-inference:ocr-v9-20261023`,
  `anpr-engine-web:mobile-v9-20261023`. Önceki etiketlerin üzerine yazılmaz.
- Build: mevcut özel builder `anprlean20261023test`, 1 CPU / 3 GiB RAM,
  ek swap yok, tek build adımı. İki imaj sırayla derlenir.
- Test: 0,75 CPU / 2 GiB; ağ yok, host portu yok, salt okunur modeller.
- Önceki modeller `/root/anpr-lean-lab-20261023/models/`, izinli smoke
  fotoğrafı `/root/anpr-lean-lab-20261023/test/plate.jpg` konumunda varsayılır.
  Eksikse dur; başka uygulamadan dosya taşıma/silme yapılmaz.

## 2. Sunucuda ön kontrol

Mac'ten v9 paketini SCP ile sunucuda `/root/anpr-release-v9-20261023.tar.gz`
konumuna gönder; SSH şifresini komuta veya repoya yazma.

```sh
free -m
ANPR_DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}')
df -h "$ANPR_DOCKER_ROOT"
docker buildx inspect anprlean20261023test
```

Başlangıçta en az 6 GiB available RAM ve 15 GiB boş Docker diski yoksa dur.
Bunlar emniyet payıdır, ölçülmüş minimum gereksinim değildir. Paylaşılan sunucuda
işlem sıfır etkili değildir; trafik düşükken yap. Global prune/down/restart yok.

## 3. Yeni ve ayrı kaynak/context

```sh
mkdir /root/anpr-release-v9-20261023
cd /root/anpr-release-v9-20261023
mkdir source
tar -xzf /root/anpr-release-v9-20261023.tar.gz -C source
python3 source/scripts/prepare_phase2d_context.py inference-paddleocr-amd64 /root/anpr-release-v9-20261023/context-inference
python3 source/scripts/prepare_phase2d_context.py web /root/anpr-release-v9-20261023/context-web
```

Her komutun başarılı olduğunu kontrol et. Dizin/context zaten varsa dur;
eski kaynakların üzerine açma. Context üreticisi yalnız allowlist dosyalarını
kopyalar, repo veya model dizinini Docker'a topluca göndermez.

## 4. Sınırlı build

```sh
docker buildx inspect anprlean20261023test --bootstrap
docker inspect --format 'MEM={{.HostConfig.Memory}} MEMSWAP={{.HostConfig.MemorySwap}} PERIOD={{.HostConfig.CpuPeriod}} QUOTA={{.HostConfig.CpuQuota}}' buildx_buildkit_anprlean20261023test0
docker image ls anpr-engine-inference:ocr-v9-20261023
docker image ls anpr-engine-web:mobile-v9-20261023
```

Limitler MEM/MEMSWAP=3221225472 ve PERIOD/QUOTA=100000 olmalı. Farklıysa dur.
Yeni etiketler zaten varsa üzerine yazma. İki build'i **sırayla** çalıştır:

```sh
cd /root/anpr-release-v9-20261023
docker buildx build --builder anprlean20261023test --platform linux/amd64 --load --progress=plain -f context-inference/deploy/docker/inference.lean-corners.amd64.Dockerfile -t anpr-engine-inference:ocr-v9-20261023 context-inference
docker buildx build --builder anprlean20261023test --platform linux/amd64 --load --progress=plain -f context-web/deploy/docker/web.Dockerfile -t anpr-engine-web:mobile-v9-20261023 context-web
docker buildx stop anprlean20261023test
docker image inspect --format 'ID={{.Id}} SIZE_BYTES={{.Size}}' anpr-engine-inference:ocr-v9-20261023 anpr-engine-web:mobile-v9-20261023
```

İlk build başarısızsa ikinciyi başlatma. Hata/host yavaşlamasında Ctrl+C ve yalnız
bu builder için `docker buildx stop anprlean20261023test` kullan. BuildKit limitleri
Docker daemon image import'u, disk veya network için tam üst sınır değildir.

## 5. İzole gerçek model/RAM smoke

Önce dosyaların var olduğunu doğrula:

```sh
test -r /root/anpr-lean-lab-20261023/models/inference/inference.yml && echo OCR_OK
test -r /root/anpr-lean-lab-20261023/models/y5fu_640x_sim.onnx && echo CORNERS_OK
test -r /root/anpr-lean-lab-20261023/test/plate.jpg && echo PHOTO_OK
docker ps -a --filter name=anprv9ram20261023 --format '{{.Names}} {{.Status}}'
```

Üç OK çıkmalı, test container adı kullanılmıyor olmalı. Eksik dosyada izinleri
topluca değiştirme. `--mount` eksik kaynakta yeni boş dizin oluşturmaz; durur.

```sh
docker run --name anprv9ram20261023 --pull=never \
  --cpus=0.75 --memory=2g --memory-swap=2g --pids-limit=96 \
  --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=128m,uid=10001,gid=10001 \
  --log-driver=json-file --log-opt max-size=5m --log-opt max-file=3 \
  --mount type=bind,src=/root/anpr-lean-lab-20261023/models/inference,dst=/models/paddleocr/inference,readonly \
  --mount type=bind,src=/root/anpr-lean-lab-20261023/models/y5fu_640x_sim.onnx,dst=/models/corners/y5fu_640x_sim.onnx,readonly \
  --mount type=bind,src=/root/anpr-lean-lab-20261023/test/plate.jpg,dst=/test/plate.jpg,readonly \
  anpr-engine-inference:ocr-v9-20261023 \
  python /app/runtimes/lean-smoke/smoke.py --image /test/plate.jpg
```

Bu test birkaç seri istek + 16 saniye idle bekleme yapıp çıkar; kalıcı servis,
host portu veya yeni ağ açılmaz. Aynı ASGI yükleme yolu test edilir. Plaka metni
loglanmaz; okumanın var/tutarlı olması tek başına doğru okunduğunu kanıtlamaz.

```sh
docker inspect --format 'EXIT={{.State.ExitCode}} OOM={{.State.OOMKilled}}' anprv9ram20261023
free -m
```

Yayın kapısı: EXIT=0, OOM=false; tüm olumlu smoke kontrolleri true,
`torch_loaded=false`, `legacy_bundle_used=false`; cgroup current/peak alanları
dolu olmalı. Zaman aşımı/OOM/worker kapanmaması/host yavaşlaması varsa yayınlama,
limiti otomatik artırma. Gecikmeyi önceki aynı host ölçümüyle kıyasla; yeni bir
keyfi SLA uydurulmaz. JSON ve imaj kimliklerini paylaş; env/credential dökümü yok.

## 6. Ancak başarılı test sonrasında Coolify geçişi

Mevcut ANPR uygulamasında yalnız iki imaj etiketini yeni v9 etiketleriyle değiştir.
Registry pull/build tetiklemeden mevcut yerel imajlarla yeniden oluşturma yolunu
kullan; sürümün bu davranışı belirsizse dur. Domain/TLS, proxy labels, özel ağlar,
credential değişkenleri, quota volume, salt okunur model mountları, runtime
profili ve mevcut CPU/RAM sınırları korunur. Yeni ikinci Compose projesi başlatılmaz.

Her iki servis healthy olduktan sonra gerçek web upload'unda dörtgen, düzeltilmiş
plaka, metin/aday ayrımı ve mobil başlık kontrol edilir. Yetkili birkaç farklı
görüntüyle sırayla dene; yük/stres testi yapma. Başarı kanıtı gelmeden canlıya
geçildiği veya doğruluk kabulünün tamamlandığı söylenmez.

## 7. Geri dönüş

Bu hazırlığın başlangıç referansı web `anpr-engine-web:violet-v6-20261023`,
inference `anpr-engine-inference:lean-lab-20261023` çiftidir. Geçiş öncesi canlı
etiket/ID'nin hâlâ bu olduğunu güvenli inspect ile doğrula; farklıysa gerçek
mevcut çifti kaydet ve onu geri dönüş hedefi yap. Eski imajlar silinmez.

Sorunda aynı Coolify servisine önceki imaj çiftini geri seç ve yerel imajlarla
yeniden oluştur. Model/config/network/quota değişmediği için veri göçü yoktur.
Global Docker temizliği, eski model silme, repo force-push veya git reset yoktur.
