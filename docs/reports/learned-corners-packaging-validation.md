# Düşük yükle köşe detector / OCR devam çalışması

Tarih: 2026-10-17. Model ve OCR ağı değiştirilmedi; yeni eğitim, indirme,
dependency kurulumu, Docker build veya canlı yayın yapılmadı. Commit/push yok.

## Tamamlananlar

| İş | Teslimat / sonuç |
| --- | --- |
| Image config eksikliği | Learned 640 JSON allowlist'e ve Paddle Dockerfile'a eklendi; eski config varsayılanı korunur |
| Opt-in yayın seçimi | `compose.learned-corners-candidate.yaml`: read-only ONNX dosyası, zorunlu host yolu, 640 web provider/config eşleşmesi |
| Geometri regresyonu | Warp köşe rolleri/RGB kanalları, crossing/out-of-bounds/NaN/degenerate/ince/uygunsuz aspect ve çıktı boyutu sınırı testleri |
| Paketleme regresyonu | Config ve adapter build context'te; weight/dataset yok; mount/provider eşleşmesi doğrulandı |
| Operasyon | [Yayın ve rollback tarifi](../operations/learned-corners-release.md); roadmap'te güncel fazlar ile eski araştırma ayrıldı |

## Bu tur çalıştırılan kontroller

- Python suite: **423 geçti, 3 private opt-in atlandı**, 12,93 saniye.
  `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
- Yeni/değişen geometri ve paketleme kapsamı: 22 test geçti.
- Değişen 3 Python dosyasında mypy, Ruff lint/format başarılı.
- Public validator: 61 JSON/JSONL, 25 YAML, 13 schema, 2 OpenAPI,
  81 yerel doküman bağlantısı; 384 dosyada yasak private içerik yok.
- Üç Compose dosyası gerçek `docker compose ... config --quiet` ile birleşti.
  Yalnızca doğrulama amaçlı sahte image/yol/credential değerleri kullanıldı;
  daemon başlatılmadı, image veya servis çalıştırılmadı.
- `git diff --check` başarılı. UI değişmediği için web yeniden build edilmedi.

Bu tur yeni gerçek-model doğruluk veya RSS profili çalıştırılmadı. Önceki
[Mac model raporundaki](learned-corner-cpu-selection.md) 10 geliştirme / 12
sentetik kontrol ve yaklaşık 1,24 GiB Python+OCR RSS ölçümü tarihsel sonuçtur;
bugünkü testler yeni model doğruluğu veya sunucu bütçesi kanıtı değildir.

## Kalan ve nedenleri

| Kalan | Gereken kanıt / karar |
| --- | --- |
| Bağımsız gerçek fotoğraflarda doğruluk | Yeni holdout; mevcut örnekleri tekrar ölçmek yerine bağımsız değerlendirme |
| Fiziksel köşe hatası / polygon IoU | Güvenilir köşe ground truth'u; metrik ve eşik TBD |
| Gece, yansıma, örtülme kapsamı | İzinli ve etiketli bağımsız örnekler; sentetik testler eşdeğer değil |
| Linux amd64 image ve kaynak bütçesi | Hedefe uygun builder/sunucu ve gerçek model smoke/RSS/gecikme ölçümü |
| Yayın lisans kapsamı | Dağıtıcı beyanının yayın sahibi tarafından değerlendirilmesi |
| Canlı geçiş | Testler sonrası ayrı kullanıcı onayı; mevcut canlıya dokunulmadı |

Bu eksikler kullanıcıya yeni fotoğraf yükletmek veya Mac'te CUDA/eğitim
çalıştırmak için gerekçe değildir; şimdilik açık kabul kriterleri olarak kalır.
Legacy GitHub reposuna, diğer çalışma klasörlerine veya yerel origin'e
bu devam çalışmasında müdahale edilmedi.
