# Sunucu kapasitesi için hafif envanter komutu

Tarih: 2026-10-21. `deploy_learned_corners.py doctor` eklendi. Amaç canlı
sunucu bilgisi eksikliğini kullanıcıya elle CPU/RAM araştırması yaptırmadan
azaltmak; otomatik üretim kabulü veya kaynak optimizasyonu iddiası değildir.

| Tamamlanan | Kanıt |
| --- | --- |
| Config'siz doctor | OS/mimari, CPU sayısı, Linux MemTotal/MemAvailable, kaynak filesystem boş alanı |
| Sınırlı Docker kontrolü | Yalnızca Compose CLI sürümü; 10 saniye timeout; daemon'a istek yok |
| Paylaşılabilir çıktı | Şifre, hostname, IP, environment ve model yolları rapora alınmaz |
| Bilinmeyen veriler | RAM okunamazsa null; her durumda production_readiness NOT_VERIFIED |
| Kaynak uyarısı | 2560 MiB mevcut container tavanıyla karşılaştırılır; minimum ihtiyaç veya hosting kotası sayılmaz |
| Yerel doğrulama | Mac'te doctor çalıştı; Linux amd64 olmadığı belirtildi; model/servis açılmadı |
| Birim testleri | Helper'ın 20 testi; düşük/yeterli/bilinmeyen RAM, sınırlı komut ve beklenmeyen çıktı filtreleme |
| Paket | Bağlantılı yayın/rollback/model raporları kaynak arşivine eklendi; eski arşivler korunur |

Linux davranışı küçük test girdileriyle denetlendi; gerçek Linux sunucuda test
yapılmadı. Host RAM cgroup/hosting kotası değildir; kaynak diski Docker storage
alanıyla aynı olmayabilir. Boş alanın yeterli görünmesi build onayı değildir.
Yerel Mac envanteri sunucunun özellikleri gibi kullanılmaz.

## Kalan

Sunucuda `python3 scripts/deploy_learned_corners.py doctor` çıktısı, hosting
türü ve gerçek platform kotaları gerekli. Sonrasında Linux build/gerçek upload,
RSS/gecikme/idle/restart, domain/HTTPS/proxy ve yayın lisans kabulü doğrulanır.
Bağımsız köşe ground truth'u/holdout ve gece/örtülme kapsamı hâlâ eksiktir.
Canlıya geçiş, commit/push, ağır build ve model eğitimi bu tur yapılmadı.

Kullanım: [kolay yayın rehberi](../operations/live-corners-quickstart.md).

Son genel suite: **443 geçti, 3 private opt-in atlandı**, 14,36 saniye;
OMP/OpenBLAS/MKL thread sınırları 1. Mypy, Ruff, public validator ve diff
whitespace kontrolü geçti. Yeni OCR doğruluk veya RAM profili çalıştırılmadı.
