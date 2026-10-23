# Canlıya alma: ilk kurulum ve tek komutla başlatma

23 Ekim: [lean runtime](lean-corners-runtime.md) PyTorch/legacy bundle olmadan
aynı köşe/OCR modellerini çalıştırır. Opt-in config'i
`configs/public/live-lean-corners-template.json`; Linux/Coolify testi bekliyor.
Eski config varsayılanı değişmedi. Coolify'da ikinci uygulama açma.

Bu tarif **Linux x86_64/amd64 VPS + Docker Compose v2 + HTTPS reverse proxy**
içindir. Mac yerel website akışı değişmez. Paylaşımlı PHP hosting veya Docker
erişimi olmayan hosting'de bu script çalışmaz. Coolify yönetimli mevcut servisi
bu komutla ayrıca başlatma; [overlay tarifini](learned-corners-release.md)
Coolify yapılandırmasına uygula. DNS/TLS ve sunucu kaynakları script tarafından
oluşturulmaz veya varsayılmaz. Linux image'ın gerçek-model testi henüz yapılmadı.

Coolify kullanan paylaşılan sunucu için [düşük RAM/disk rehberi](coolify-low-footprint.md)
uygulanır: ikinci standalone proje açma. Learned config'in OCR idle süresi
15 saniyedir; learned overlay iki servisin Docker loglarını sınırlar. Bunlar
yerelde hazırlandı; mevcut canlı container'lara otomatik uygulanmadı.

## Önce sunucu bilgilerini tek komutla öğren

Sunucuda kaynakları açtıktan sonra (config/şifre gerektirmez):

```sh
python3 scripts/deploy_learned_corners.py doctor
```

Bu hafif komut OS/mimari, mantıksal CPU, Linux host RAM'i, kaynak dizininin boş
disk alanı ve Compose CLI sürümünü JSON olarak verir. Şifre, IP, hostname veya
environment dökümü yoktur; çıktıyı paylaşabilirsin. Docker daemon'a bağlanmaz,
model yüklemez, build/pull veya servis başlatmaz. Bilinmeyen RAM `null` olur.
Hosting/cgroup kotası host RAM'den farklı olabilir; doctor raporu uygunluk
onayı değildir. Docker storage boş alanı, Linux build ve gerçek upload ayrıca
doğrulanmalıdır. 2560 MiB karşılaştırması mevcut iki container tavanıdır;
ölçülmüş minimum RAM gereksinimi değildir.

## 1. Güncel kaynakları paketle (Mac; hafif işlem)

Commit/push yapılmadığı için yalnızca GitHub'dan clone etmek yeni kodu getirmez.
Proje kökünde:

```sh
python3 scripts/deploy_learned_corners.py package --output "$HOME/anpr-corners-source.tar.gz"
```

Bu yalnızca allowlist'teki kaynakları arşivler; özel modeller, fotoğraflar,
secret, `.git`, ortamlar ve node_modules dahil değildir. Önceki arşivin üzerine
yazmaz; tekrar çalıştırırken farklı bir çıktı adı seç. İçerik sıkıştırılır,
dataset/PNG export'u yapılmaz. Yeni arşivi `scp`/SFTP ile sunucuya gönder.

## 2. Sunucuda yeni, boş bir dizine aç

Örnekte `anpr-release-1` yeni ve boş olmalı; mevcut site dizinine açma:

```sh
mkdir "$HOME/anpr-release-1"
tar -xzf "$HOME/anpr-corners-source.tar.gz" -C "$HOME/anpr-release-1"
cd "$HOME/anpr-release-1"
docker compose version
uname -m
```

Son komut `x86_64` olmalı. Docker kurulu ve kullanıcın daemon'a erişebiliyor
olmalı. Script Docker kurmaz veya root yetkisi istemez.

## 3. Model dosyalarını bir kez özel olarak aktar

SFTP/scp ile aşağıdaki mevcut dosya/dizinleri sunucuda ayrı tut. Mac'te ortak
kök proje dışındaki private çalışma dizinidir; public repoda mutlak yol tutulmaz.
Bunları GitHub'a koyma. Dataset veya Python ortamlarını aktarmana gerek yok.

| Mac'teki kaynak (ortak köke göre) | Önerilen sunucu yolu |
| --- | --- |
| `models/active/` içindeki `detection.pt`, `recognition.pt`, `bundle-manifest.json` | `/srv/anpr/models/bundle/` |
| `models/candidates/paddleocr-latin-v5/smoke-v1/inference/` içeriği | `/srv/anpr/models/paddleocr/inference/` |
| `models/candidates/hyperlpr-corners-v1/y5fu_640x_sim.onnx` ve yanındaki lisans kaydı | `/srv/anpr/models/corners/` |

`models/active` symlink ise bağlantıyı değil gerçek dosya içeriğini aktar.
Model dosyaları container UID `10001` tarafından okunabilmeli; yazma yetkisi
gerekmez. Root olmayan kullanıcı `/srv` oluşturamıyorsa kendi özel dizinini
kullanıp config'e o gerçek mutlak yolu yaz. Var olan başka dosyaları silme.

20 Ekim güncellemesi: tüm model hash'lerini ağları RAM'e yüklemeden ve Docker
gerektirmeden kontrol edebilirsin (Mac veya Linux):

```sh
python3 scripts/deploy_learned_corners.py verify-models --config "$HOME/anpr-live.json"
```

Bu komut yalnızca `model_bundle`, `paddle_model`, `corner_model` alanlarını
gerektirir. Üç alanı içeren ayrı, izinleri 600 olan bir JSON da kullanılabilir.
Model byte hash'leri ve bundle manifest kimlikleri kontrol edilir; dosya adı
veya yalnızca varlığı yeterli sayılmaz. Symlink yerine gerçek dosya/dizin kullan.

## 4. Özel config'i bir kez doldur

```sh
cp configs/public/live-corners-template.json "$HOME/anpr-live.json"
chmod 600 "$HOME/anpr-live.json"
python3 -c 'import secrets; print(secrets.token_hex(32))'
nano "$HOME/anpr-live.json"
```

Son üretilen değeri `internal_credential` olarak kullan; paylaşma. Dosyadaki:

| Alan | Ne yazacaksın? |
| --- | --- |
| `public_origin` | Gerçek domain, ör. `https://plaka.senin-domainin.com`; sonda yol yok |
| `model_bundle`, `paddle_model`, `corner_model` | 3. adımdaki gerçek sunucu yolları |
| `application_version` ve iki image etiketi | Bu sürüme özel adlar; önceki sürüm etiketlerini yeniden kullanma |
| `project_name` | Varsayılan `anpr-corners`; başka uygulamanın Compose adını kullanma |
| `web_port` | Reverse proxy'nin bağlanacağı boş yerel port; varsayılan 39084 |
| `release_checks_confirmed` | Başlangıçta false. Kaynak bütçesi/lisans ve kontrollü ilk başlatmayı değerlendirdikten sonra true |

Bu onay otomatik doğruluk sertifikası değildir. Sonraki gerçek upload testi
başarılı olmadan kullanıcı trafiğini yeni servise yönlendirme.

## 5. İlk image build'i (yalnızca açık komutla; ağır olabilir)

```sh
python3 scripts/deploy_learned_corners.py build --config "$HOME/anpr-live.json"
```

Üç image sırayla hazırlanır; GPU/eğitim yok ama paket indirmesi, disk ve RAM
tüketimi vardır. **Az RAM'li canlı sunucuda build yapma.** Aynı config'in model
ve secret içermeyen build ayarlarıyla ayrı Linux amd64 builder kullanabilir,
hazırlanan image'ları `docker save`/`docker load` ile sunucuya aktarabilirsin.
Bu arşivler büyük olabilir; alanı önce kontrol et. Script kendi başına build,
registry push veya model indirmesi yapmaz. Başarısız/yarım build sonrasında
yeni image etiketleri seç; otomatik prune veya dosya silme yoktur.

Builder config'inde yalnızca `inference_image` ve `web_image` alanları yeterlidir;
canlı servis şifresi veya modeller builder'a aktarılmaz. İzinleri yine 600 olmalı.

## 6. Ön kontrol ve çalıştırma

```sh
python3 scripts/deploy_learned_corners.py check --config "$HOME/anpr-live.json"
python3 scripts/deploy_learned_corners.py start --config "$HOME/anpr-live.json"
```

`check` servis açmaz. Linux/amd64, Docker, yerel image kimlikleri, tüm model
hash'leri/bundle manifesti ve Compose ayarını kontrol eder. Runtime hash kontrolü
de ayrıca korunur. `start` yerel image ID'lerini sabitler;
build/pull yapmadan başlar ve en fazla 240 saniye health bekler. Health başarılı
olması gerçek OCR upload'unun doğru çalıştığı anlamına gelmez.

Container'lar yalnızca bu `project_name` altında yönetilir. Web yalnızca
`127.0.0.1:39084` üzerinde yayınlanır; private inference portu dışarı açılmaz.
Container'lar daemon/host restart'ında kendiliğinden başlamaz (`restart: no`);
tekrar aynı `start` komutunu çalıştır. Model RAM'ini azaltmak için container
memory limit'ini gelişigüzel düşürme. Mevcut tavan inference 2 GiB + web 512 MiB;
OS, proxy ve diğer uygulamalar için ayrıca yer gerekir. Hedef ihtiyaç TBD'dir.

## 7. HTTPS domain'i bağla ve tek gerçek upload ile kontrol et

DNS sunucuyu göstermeli; mevcut TLS reverse proxy web'i
`http://127.0.0.1:39084` adresine yönlendirmeli. Nginx kullanıyorsan **mevcut,
sertifikalı HTTPS server bloğundaki** konum ayarının örneği:

```nginx
location / {
    client_max_body_size 10m;
    proxy_pass http://127.0.0.1:39084;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-For "";
    proxy_read_timeout 130s;
}
```

Bu tam TLS/domain kurulum dosyası değildir. Proxy istemcinin `X-Real-IP`
değerini güvenilir kaynakla değiştirmeli. CDN/proxy zinciri varsa gerçek IP
güven zinciri ayrıca kurulmalı; aksi halde quota ortak proxy IP'sini sayabilir.
Mevcut canlı site varsa trafiği değiştirmeden önce rollback ayarını sakla;
küçük sunucuda iki model servisini paralel çalıştırmak RAM'i artırır.

Domain'i tarayıcıda aç; bildirim kutusunu işaretle ve bir araç fotoğrafı yükle.
Orijinalde dörtgeni, yanında düzleştirilmiş plakayı ve OCR metnini kontrol et.
Ardından bir plakasız görüntü dene: metin uydurmamalı. Cookie/Origin/HTTPS
nedeniyle direkt düz HTTP portundan test etmek üretim akışına eşdeğer değildir.

## Bundan sonra günlük kullanım

```sh
cd "$HOME/anpr-release-1"
python3 scripts/deploy_learned_corners.py start --config "$HOME/anpr-live.json"
```

| İhtiyaç | Script komutu (`--config "$HOME/anpr-live.json"` ekle) |
| --- | --- |
| Yalnızca ön kontrol | `python3 scripts/deploy_learned_corners.py check` |
| Çalıştır / yeniden aç | `python3 scripts/deploy_learned_corners.py start` |
| Durum | `python3 scripts/deploy_learned_corners.py status` |
| Durdur, verileri koru | `python3 scripts/deploy_learned_corners.py stop` |

Hata halinde script secret/çözülmüş Compose config'i basmaz. Sunucuda yalnızca
seçili projenin durumunu/loglarını incele; hassas logları paylaşma. Otomatik
rollback, DNS değişikliği, dosya/volume silme, commit ve push yoktur. Önceki
yayın geri dönüşü [rollback tarifindedir](learned-corners-release.md).

Referans: [Compose up / wait](https://docs.docker.com/reference/cli/docker/compose/up/).

## Script doğrulama kaydı (2026-10-17)

429 Python testi geçti, 3 private opt-in atlandı. Helper için 6 test;
özel config izinleri, HTTPS ayarı, secret'sız/weight'siz kaynak paketi,
check/start/stop komut sınırları ve yayın onayı olmadan start engeli denetlendi.
Mypy, Ruff ve public validator geçti. Docker build/start, sunucu sağlık ve
gerçek Linux upload testi bu tur çalıştırılmadı; scriptin sunucuda uçtan uca
doğrulandığı iddia edilmez. Commit veya push yapılmadı.
