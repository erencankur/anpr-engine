# Öğrenilmiş dört köşe: opt-in yayın ve geri dönüş

Adım adım ilk kurulum ve tek komut script'i: [kolay yayın rehberi](live-corners-quickstart.md).

Tarih: 2026-10-17. Bu hazırlık canlıya uygulanmadı. Mac'te Docker build,
eğitim, dependency kurulumu veya dataset export'u yapılmaz. Yerel başlatıcı
değişmez. Mevcut kanıt ve sınırlar [model raporundadır](../reports/learned-corner-cpu-selection.md).

## Paket ve seçim

Önce [Paddle yayın tarifindeki](paddleocr-low-resource-release.md) güncel kaynaklı
inference base, Paddle inference ve web image'ları ayrı builder'da hazırlanır.
Allowlist yalnızca kod ve iki JSON config'i ekler; weight içermez. Image içinde
`/app/configs/web-ocr-learned-corners.json` yeni config'idir; eski config hâlâ
varsayılandır. Köşe modeli ayrı overlay ile seçilir.

| Ayar | Beklenen değer |
| --- | --- |
| Detector | `hyperlpr-y5fu-640-corners-v1` |
| ONNX SHA256 | `0306de937471b87f56eb3f5620815e7e7058f8ab7428e0734fc64627cc4d716c` |
| ONNX mount | `/models/corners/y5fu_640x_sim.onnx`, read-only dosya |
| Web provider | `ANPR_DETECTION_PROVIDER=hyperlpr640` |
| OCR | Mevcut native Paddle; `ANPR_RECOGNITION_PROVIDER=paddleocr` |
| CPU / çağrı | 1 thread; normal 1, miss halinde en fazla 3 detector geçişi; OCR en fazla 1 kez |

Sunucuya ayrı provision edilmiş ONNX'in hash'i ve dağıtıcı lisans kaydı
doğrulanır. `ANPR_CORNER_MODEL_SOURCE` mevcut, okunabilir **dosyanın** mutlak
yoludur; klasör değil. Runtime indirme yapmaz. Eksik model mount'u sessizce
klasör oluşturmaz; yanlış hash başlangıçta reddedilir. Web provider ve model
kimliği birlikte seçilmelidir. Görseller, weights ve private bundle Git'e eklenmez.

Önceki iki image kimliği rollback için saklanır. Yeni doğrulanmış image kimlikleri
`ANPR_PADDLE_INFERENCE_IMAGE` ve `ANPR_PADDLE_WEB_IMAGE` ile verilir. Eski private
bundle ve Paddle model mount'ları hâlâ gereklidir. Compose sırası:

```sh
docker compose -f deploy/docker/compose.public-candidate.yaml \
  -f deploy/docker/compose.paddleocr-candidate.yaml \
  -f deploy/docker/compose.learned-corners-candidate.yaml config --quiet
```

Bu yalnızca ayarı doğrular; servis başlatmaz. Birleşik config'i secret
değerleriyle raporlara kaydetme. `ANPR_CORNER_MODEL_SOURCE` verilmezse seçim
başarısız olmalıdır. Yeni overlay port açmaz, limitleri büyütmez. Ana dosyanın
inference 2048 MiB / 0.75 CPU ve web 512 MiB / 0.25 CPU sınırları korunur;
bunlar ihtiyaç ölçümü veya küçük sunucuya uygunluk garantisi değildir.

## Yayın öncesi kabul

| Kontrol | Kabul / mevcut durum |
| --- | --- |
| Linux amd64 import ve gerçek weight yükleme | Bekliyor; yalnızca Mac doğrulandı |
| Gerçek HTTP upload | Model ID/hash, dörtgen, iki preview ve OCR birlikte doğrulanmalı |
| Plakasız / çözülemeyen köşe | Metin uydurulmamalı; geometri yoksa OCR çağrılmamalı |
| Web sınırları | Origin/session/quota/tek aktif istek kuralları korunmalı |
| RSS / gecikme / idle / restart | Hedef sunucuda ölçülmeli; bütçe ve eşikler TBD |
| Doğruluk / köşe hassasiyeti | Bağımsız veri ve köşe ground truth'u bekliyor; sentetik testler yerine geçmez |
| Lisans | Dağıtıcı Apache-2.0 beyanı kayıtlı; yayın sahibinin kapsam incelemesi bekliyor |

Bu kontroller tamamlanmadan üretime hazır/garantili olarak sunulmaz. Kullanıcının
fotoğrafları yeniden yüklemesi veya yerelde eğitim yapması önkoşul değildir.

## Rollback

Yeni servisler durdurulur; devam eden istek bitmeden sağlayıcı değiştirilmez.
Önceki web ve inference image kimlikleri birlikte seçilir. Eski Paddle detector
akışına dönüşte learned-corners overlay kaldırılır ve web detector provider
`project` olur (environment'ta kalmış değeri de temizle). Önceki tam legacy
yayına dönüşte Paddle overlay de kaldırılır. Veriler, quota, bundle ve modeller
silinmez. Readiness + kısa upload kontrolünden sonra trafik geri verilir.
Yerelde karşılığı ayrı legacy başlatıcıdır; aynı portta eşzamanlı çalıştırılmaz.
