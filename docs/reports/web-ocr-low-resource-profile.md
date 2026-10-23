# Web OCR kısa CPU profili — 2026-10-14

Ölçüm ortamı macOS arm64, Python 3.11; ana detector ve PaddleOCR thread sınırı 1.
Gerçek loopback HTTP upload endpoint'i, mevcut YOLOX detector ve fine-tuned
PaddleOCR modeli kullanıldı. Görseller diske export edilmedi. Raporlarda plaka
metinleri, tahminler, görseller veya özel dosya yolları tutulmadı.

## Varsayılan 60 saniye idle ile 10 görsel

| Ölçüm | Sonuç |
| --- | --- |
| Geliştirme görseli / farklı plaka sayısı | 10 / 3 |
| Doğru okuma / yanlış okuma / çekimser | 10 / 0 / 0 |
| Detector kaçırma | 0 |
| Güven eşiği | 0.99 |
| İlk upload gecikmesi | 3102.94 ms |
| Sonraki 9 upload medyanı | 257.55 ms |
| İstek öncesi ana süreç RSS | 296.23 MiB |
| Ana süreç + OCR worker örneklenen en yüksek toplam RSS | 1491.67 MiB (yaklaşık 1.46 GiB) |
| OCR kapandıktan sonra ana süreç RSS | 492.56 MiB |
| İlk isteğe kadar OCR worker yüklenmemesi | Geçti |
| 60 saniye boşta worker çıkışı | Geçti |
| Sentetik plakasız beyaz görsel | Geçti |
| Test servisi kapanışı | Geçti |

Bellek 250 ms aralıklarla ana süreç ve OCR worker RSS toplamıyla örneklendi.
Shared sayfalar iki kez sayılabilir, anlık tepe kaçabilir. Web süreci bu ölçüme
dahil değildir. Bu bir Linux container peak ölçümü değildir.

## Gerçek worker yeniden açılışı

Ayrı kısa kontrol için idle süresi 1 saniyeye indirildi; aynı model ve tek CPU
thread kullanıldı. Bir görsel işlendi, worker çıkışı beklendi ve yeni upload yapıldı.

| Kontrol | Sonuç |
| --- | --- |
| Idle sonrası worker çıkışı | Geçti |
| Yeni istekte worker yeniden açılması | Geçti |
| Yeniden açılış sonrası OCR güven değerinin gelmesi | Geçti |
| Yeniden açılış sonrası doğru plaka | Geçti |
| Yeniden açılış upload gecikmesi | 1644.83 ms |
| Servis kapanışı | Geçti |

İlk kontrol gerçek varsayılan 60 saniyelik davranışı; ikinci kontrol gerçek
subprocess yeniden açılışını doğrular. Bu ikinci süre üretim config'ine uygulanmadı.

## Karar

Bu sürüm geliştirme setinde işe yarıyor ve idle worker belleğini bırakıyor.
Ancak yaklaşık 1.46 GiB işlem sırası RSS toplamı, az RAM'li sunucu açısından
önemli bir maliyettir. Ayrıca ana süreç inference cache'leri nedeniyle ilk
başlangıç belleğine tamamen dönmüyor. Web ve diğer servisler için ayrıca alan gerekir.

10 fotoğraf aynı 3 plakayı içerir; bağımsız doğruluk, gece/eğik/küçük plaka
kapsamı veya genel %100 doğruluk iddiası yapılamaz. Sonuçlar geliştirme regresyonudur.
Linux amd64/Python 3.12 image'ı, container bütçesi ve gerçek sunucu performansı
henüz doğrulanmamıştır. Yayın hazırlığı için
[Linux paketleme tarifi](../operations/paddleocr-low-resource-release.md) kullanılabilir.
