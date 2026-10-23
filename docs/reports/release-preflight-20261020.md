# Hafif yayın ön kontrolü ve gerçek-model tekrar testi

Tarih: 2026-10-20. Commit/push, canlı yayın, Docker build, eğitim veya paket
kurulumu yapılmadı. Diğer çalışma klasörleri/legacy repo değiştirilmedi.

| Tamamlanan | Doğrulama |
| --- | --- |
| Servis öncesi tüm model bütünlüğü | 640 ONNX, bundle manifest + detection/recognition byte hash'leri ve Paddle dizin hash'i; dosyalar stream ile okunur, ağ yüklenmez |
| Docker'sız kontrol | `verify-models --config ...` Mac ve Linux'ta; yalnızca üç model yolu gerekir |
| Builder ayrımı | Build config'i yalnızca iki image etiketi gerektirir; canlı secret veya modeller builder'a gerekmez |
| Bozuk artifact koruması | ONNX/bundle/OCR değişimi, ek OCR dosyası, uygunsuz manifest ve symlink testleri |
| Kaynak paketinin yeterliliği | Paket ayrı dizine açılarak iki allowlisted Docker context'in paketten yeniden hazırlanması; build/daemon yok |
| Gerçek dört-köşe + OCR | Mevcut iki izinli geliştirme fotoğrafıyla CPU HTTP/ASGI smoke; doğru okuma/adayı, native köşe yöntemi, iki preview, legacy detector'ın yüklenmemesi |
| Kapanış | Smoke sonunda OCR worker kapandı; açık web servisi başlatılmadı |

Gerçek-model smoke **1 test / 2 fotoğraf, 4,81 saniye** sürdü. İki sonuç
beklenen metinle eşleşti; bu yeni bağımsız veri değildir, genel doğruluk
iddiası değildir. Tek thread sınırları kullanıldı; RAM profili tekrarlanmadı.

Genel Python suite: **438 geçti, 3 opt-in atlandı**, 10,88 saniye.
Private learned-model smoke ayrıca açıkça seçilerek çalıştırıldı. Helper'ın
15 testi, değişen üç Python dosyasında mypy/Ruff ve public validator kontrolü
yapıldı. UI değişmedi; web build/browser tekrar çalıştırılmadı.

Güncellenen kaynak arşivi private release dizininde farklı tarihli dosyaya
hazırlandı; önceki 17 Ekim arşivi silinmedi/üzerine yazılmadı. Model ve secret
pakete eklenmedi. Güncel işlem adımları [kolay yayın rehberinde](../operations/live-corners-quickstart.md).

| Kalan | Gereken |
| --- | --- |
| Sunucuya aktarım / Linux amd64 build ve gerçek upload | Uygun sunucu/builder; burada erişim ve hedef bilgisi yok |
| Domain / HTTPS / proxy ve kaynak bütçesi | Hosting bilgisi, hedef CPU/RAM, DNS/TLS ayarı ve hedef ölçümü |
| Bağımsız doğruluk / fiziksel köşe hatası | Ayrı holdout + ground truth; gece/örtülme kapsamı; metrik/eşik TBD |
| Yayın lisans kabulü ve trafik geçişi | Yayın sahibinin kararı ve ayrı canlı geçiş onayı |

Bu eksikler model geliştirmesi tamamlandı veya üretim sorunsuz denilerek
kapatılamaz. Kullanıcıdan yeniden fotoğraf istenmedi; CUDA/eğitim önkoşulu yok.
