# PyTorch'suz CPU corner + OCR runtime

Güncel yayın adımları: [v9 kontrollü Linux testi ve yayın](release-v9-controlled-rollout.md).
Aşağıdaki ilk ölçümler tarihsel geliştirme kaydıdır. Güncel kodda detector en
fazla dört, OCR en fazla üç seri geçiş yapar; normal başarı yolu tek geçiştir.

Tarih: 2026-10-23. Yeni `lean_corners` opt-in yolu hazırlanıp Mac'te gerçek
modelle doğrulandı. Canlı Coolify servisine uygulanmadı. Eski `legacy` profil,
eski Dockerfile ve model dosyaları korunur; kendiliğinden fallback yoktur.

## Değişiklik ve sınır

| Alan | Lean profil |
| --- | --- |
| Detector | Aynı hash'li HyperLPR 640 ONNX; OpenCV DNN CPU |
| OCR | Aynı hash'li native Paddle modeli; tek lazy subprocess |
| Ana Python import'ları | Torch/Paddle/PaddleOCR/PaddleX import edilmez; Paddle yalnızca worker içinde |
| Eski bundle | `detection.pt`, `recognition.pt`, bundle manifest ve eski ağ config'i gerekmez |
| Resim akışı | JPEG/PNG, native dört köşe, homography, iki preview, aynı normalize/biçim kontrolü |
| Geometri hatası | OCR'a bbox fallback verilmez; crop_refinement_failed/çekimserlik korunur |
| Kaynak sınırı | Tek thread/istek, kuyruk yok; detector en fazla 3 geçiş, OCR en fazla 1 çağrı |
| Image tarifi | Tek venv; 24 hash-pinned dağıtım; torch/paddleocr/paddlex yok |
| JSON / web kimliği | Aynı detector/OCR kimlikleri ve API; model ağırlığı/decode eşiği değişmedi |

`src/anpr_engine/integration/lean_corners.py` ayrı dar yoldur. Eski eğitim ve
model kodları silinmez. Detection/integration package export'ları lazy yapıldı;
HTTP katmanı decoder/kimlikler için eski Torch pipeline'ını import etmez.
Ortama paket kurmadan import blocker ile bu sınır test edildi.

## Kısa Mac doğrulaması

Torch/Paddle import'u ana süreçte engellenmiş ayrı interpreter kullanıldı.
Eski bundle yolu verilmeden gerçek ASGI HTTP akışı çalıştı: iki mevcut fotoğraf
ve bir +35 derece varyant, doğru metin veya doğru kontrol adayı; aynı native
köşe yöntemi, iki preview. Fotoğraflar bellekte en uzun kenarı 1600 px olacak
şekilde sınırlandı. Bir plakasız kontrol geçti, OCR worker idle sonrası kapandı.
İlk kontrolün sınırsız boyutlu döndürülmüş girdisi upload limitine takıldı;
boyut sınırlı tekrar geçti. Upload limitleri gevşetilmedi.

| Örneklenen RSS (Mac) | MiB |
| --- | ---: |
| Ana süreç, pipeline init öncesi | 70,91 |
| Ana süreç, model init sonrası | 87,41 |
| Üç upload sonrası ana süreç + OCR worker | 1088,52 |
| OCR idle kapanışı sonrası ana süreç | 309,78 |

Bu tek kısa koşudur; RSS örnekleri tepe/container ölçümü değildir ve shared
sayfalar iki kez sayılabilir. Web/tarayıcı dahil değildir. Private smoke
config'i 1 saniye idle kullanır; yayın JSON'u 15 saniyedir. Önceki yaklaşık
565 MiB idle ve 1,24 GiB örneklenmiş maksimumla birebir A/B karşılaştırması
değildir. Eski canlı sürümün boşta 163 MiB toplamından daha düşük olduğu
iddia edilmez. Yeni Linux image boyutu ve hedef RAM bütçesi **TBD**.

## Mac'te opt-in kullanım

Mevcut yerel web build'iyle; OCR JSON'u 640 modelini göstermeli:

```sh
.venv/bin/python scripts/launch_local_website.py \
  --runtime-profile lean_corners --ocr-config /absolute/private/ocr-640.json
```

`--model-bundle` gerekmez. Legacy başlangıç varsayılanı korunur; eski komutlar
aynı davranır. İki başlatıcı aynı portta birlikte çalıştırılmaz. Kod değişikliği
önceden açık Python sürecini kendiliğinden yenilemez.

## Linux paketi / script

`configs/public/live-lean-corners-template.json` kopyalanıp özel config olarak
doldurulur (`chmod 600`). `runtime_profile=lean_corners` açık seçilir;
`model_bundle` alanı kullanılmaz. Diğer domain/credential/image/provider ve
kaynak bütçesi kontrolleri [yayın rehberindeki](live-corners-quickstart.md) gibi.

```sh
python3 scripts/deploy_learned_corners.py verify-models --config /private/live-lean.json
python3 scripts/deploy_learned_corners.py build --config /private/live-lean.json
python3 scripts/deploy_learned_corners.py check --config /private/live-lean.json
```

Build yalnızca ayrı, yeterli kaynağı olan Linux amd64 builder'da açıkça yapılır;
Mac'te veya paylaşılan canlı host'ta burada çalıştırılmadı. Builder config'i
`runtime_profile=lean_corners` ve iki yeni image etiketi içerir; secret/model yok.
Script bu seçimde base+Paddle+web üçlüsü yerine lean inference + web olmak üzere
iki image hazırlar. Yeni Dockerfile eski Torch image'ını parent olarak kullanmaz.

`deploy/docker/compose.lean-corners-candidate.yaml` **tam tanımdır**; eski base
Compose üzerine merge edilmez, böylece gereksiz `/models` bundle mount'u kalmaz.
Yalnız ONNX dosyası ve OCR model dizini read-only bağlanır. Limitler/log/özel ağ
ve quota korunur. Eski kaynak paketi korunur; yeni tarihli paket kullanılır.

Sunucu **Coolify** olduğundan scriptle ikinci proje başlatılmaz. Mevcut ANPR
uygulamasındaki image/config/provider/mount'lar eşlenir; Coolify domain, ağ,
quota volume ve proxy etiketleri korunur. Yeni image hash'leri ve mount'ları
doğrulanmadan eski bundle mount'u kaldırılmaz. Eski özel modeller silinmez.

## Kalan kabul ve rollback

1. Linux build/import + gerçek upload ve idle/restart testi; image/RAM ölçümü.
2. Coolify mevcut uygulamasına kontrollü geçiş ve kaynak sınırlarının ölçümü.
3. Bağımsız holdout/köşe etiketleri, gece/örtülme ve lisans kapsam incelemesi.

Rollback'te önceki web+inference image'ları, `runtime_profile=legacy` ve önceki
bundle/config/mount'lar birlikte geri seçilir. Global Docker temizliği, dosya
silme, commit/push veya canlı yayın bu çalışma kapsamında yapılmadı.

Son genel doğrulama: 453 Python testi geçti, 4 private opt-in atlandı. Gerçek
lean HTTP smoke ayrıca seçilerek geçti (4,93 saniye). Önceki corner yolu ile
geometri/preview/recognition/model kimliği eşitliği birim testinde doğrulandı.
Mypy (182 dosya), Ruff, public validator ve Compose config kontrolü başarılı.
Bağımlılık kilidi metadata çözümüyle üretildi; wheel kurulumu veya image build
yapılmadı. Eski yerel/public varsayılan legacy profil olarak korundu.
