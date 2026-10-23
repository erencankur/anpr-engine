# Aynı OCR modelinin daha hafif çalıştırılması — 2026-10-14

Karar: yeni Linux yayın adayı, mevcut fine-tuned PaddleOCR modelini doğrudan
Paddle Inference ile çalıştıracak şekilde hazırlanmıştır. Ağ ve ağırlıklar
değişmedi; preprocessing ve CTC çözümleme sınırlandırılmış bir CPU adapter'ındadır.
Model dizini SHA256 başlangıçta doğrulanır. Eski genel PaddleOCR worker yolu da
kendi tam bağımlılık ortamıyla kullanılabilir.

## Çıktı eşleşmesi

221 doğrulanmış geliştirme kırpımı iki worker ile sırayla işlendi. Aynı anda iki
OCR modeli yüklenmedi. Sonuç metinleri yalnızca bellekte karşılaştırıldı; rapora
plaka metni veya özel görsel yolu yazılmadı.

| Kontrol | Sonuç |
| --- | --- |
| Aynı model/artifact | Dizinin SHA256 kimliği aynı |
| Kırpım sayısı | 221 |
| Ham metin uyuşmazlığı | 0 |
| En büyük güven değeri farkı | 0.0 |
| Güven farkı kabul toleransı | 0.00001 |
| CPU thread | Her iki worker'da 1 |

Bu kontrol çalışma yollarının aynı çıktıyı verdiğini gösterir; veri setine karşı
yeni bir bağımsız doğruluk ölçümü değildir.

## Gerçek HTTP upload profili

macOS arm64/Python 3.11 üzerinde aynı 10 fotoğraf (3 farklı plaka), JPEG upload
endpoint'i ve aynı detector kullanıldı. Aşağıdaki değerler ayrı kısa koşulardan
gelir; istatistiksel performans üstünlüğü veya Linux kapasite garantisi değildir.

| Ölçüm | Genel PaddleOCR yolu | Doğrudan Paddle Inference |
| --- | --- | --- |
| Doğru / yanlış / çekimser | 10 / 0 / 0 | 10 / 0 / 0 |
| Örneklenen ana süreç + worker en yüksek RSS | 1491.67 MiB | 1328.66 MiB |
| İlk upload | 3102.94 ms | 2547.87 ms |
| Sonraki upload medyanı | 257.55 ms | 289.63 ms |
| OCR çıkışı sonrası ana süreç RSS | 492.56 MiB | 490.91 MiB |
| Sentetik plakasız kontrol | Geçti | Geçti |
| Gerçek worker yeniden açılışı | Ayrı kontrol geçti | Geçti |

Örneklenen toplam RSS yaklaşık 163 MiB (%10.9) azaldı. Yeni yol yaklaşık
1.30 GiB kullanmaya devam ediyor; web belleği dahil değildir. Warm medyan
bu koşuda daha yüksektir. Daha hızlı olduğu iddiası yapılmaz.

Önceki varsayılan 60 saniye idle kontrolü korunur. Native profil kısa tutulmak
için 1 saniye idle ile yapıldı; yayın şablonunda idle 60 saniyedir. Worker çıkışı,
yeniden upload'da açılışı, doğru çıktı ve servis kapanışı gerçek süreçle doğrulandı.
RSS 250 ms aralıklarla örneklendi; shared sayfalar iki kez sayılabilir ve anlık
tepe kaçabilir. Container peak ölçümü yapılmadı.

## Paketleme ve sınırlamalar

- Genel Linux OCR kilidi 70 dağıtım; native CPU kilidi 17 dağıtım içerir.
  Native adayda PaddleOCR/PaddleX import edilmez; headless OpenCV ve Paddle kullanılır.
- Adapter yalnızca `latin_PP-OCRv5_mobile_rec`, BGR, `3x48x320` başlangıç shape'i
  ve CTC modelini kabul eder. Yeni model ailesi sessizce bu yola geçirilmez.
- Linux kilidi Python 3.12/NumPy 2.2.6, yerel karşılaştırma Python 3.11/mevcut
  OCR ortamıyla yapıldı. Linux gerçek-model testi hâlâ gereklidir.
- İnference ve web servislerinin seçimi, API model kimliği ve weight hash'i korunur.
  Slim image'da genel `backend=paddleocr` seçilemez; tam OCR ortamı gerekir.
  Rollback mevcut iki image'a veya mevcut ana ortamın project OCR yoluna yapılır.

Hazır ağın CPU çalıştırılması için [Paddle Inference CPU API](https://www.paddlepaddle.org.cn/inference/guides/x86_cpu_infer/paddle_x86_cpu.html)
incelendi. Resize ve CTC davranışı kurulu PaddleX 3.7.2 kaynaklarıyla karşılaştırıldı;
[resmi PaddleOCR inference kaynağı](https://github.com/PaddlePaddle/PaddleOCR/blob/main/tools/infer/predict_rec.py)
da referans alındı. Linux paketleme ve rollback adımları
[yayın tarifinde](../operations/paddleocr-low-resource-release.md) bulunur.
