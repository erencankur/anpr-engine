# Perspektif sorunu: öğrenilmiş dört-köşe detector'a geçiş

Tarih: 2026-10-15. Karar: yeni tekil fotoğraf istemek veya kontur eşiklerini
fotoğraf bazında değiştirmek genel çözüm değildir. Kullanıcı ek örnek vermek
zorunda değildir; mevcut izinli görseller ve önceden tanımlanmış perspektif
dönüşümleriyle regresyon kapsamı genişletilir.

**Son uygulama:** gerçek HyperLPR YOLOv5 640 ONNX ağı CPU'ya entegre edildi;
640 yerel aday seçildi, 320 doğruluk kaybı nedeniyle varsayılan yapılmadı.
M1–M4 yerel kontrolleri ve yerel başlatıcı geçişi tamamlandı. Eski aşağıdaki
“model henüz yok” kayıtları ilk araştırmanın tarihsel durumudur. Güncel kanıt,
sınırlamalar ve kalan yayın işleri: [model seçim raporu](../reports/learned-corner-cpu-selection.md).

## Teşhis ve değişen kabul ölçütü

İlk teşhiste aktif model yalnızca `xyxy` üretiyordu. `PLATE_FACE_CONTOUR_V1` bir öğrenilmiş
corner modeli değildir. Önceki birkaç örneğin düzelmesi model geçişinin
tamamlandığını kanıtlamaz. Yeni örnekte yazılı plakalık kutuya dahil olmuş,
geometri çözülememiş ve eski bbox kırpımından hatalı OCR adayı görünmüştür.

Perspektif modu açıkken güvenilir dörtgen yoksa OCR artık çağrılmaz.
Bu durumda `crop_refinement_failed`, null prediction ve not_run biçim kontrolü
döner. Dörtgen/preview uydurulmaz. Bu güvenli davranış detection sorununu
çözmüş sayılmaz. Legacy yolda `estimate_plate_geometry=false` açık bbox rollback'tir;
bu modun bbox kırpımı perspektif desteği olarak sunulmamalıdır.

## Güncel uygulama durumu (2026-10-17)

| Faz | Tamamlanan | Kalan |
| --- | --- | --- |
| M0 | Geometri yokken OCR durur; rollback ayrı | Yeni yerel iş yok |
| M1 | HyperLPR 640 seçimi, revision/hash ve dağıtıcı lisans kaydı | Yayın öncesi lisans kapsam incelemesi |
| M2 | CPU adapter, gerçek köşeler, homography, schema/BFF/preview | Yeni yerel iş yok |
| M3 | 10 geliştirme fotoğrafı, 12 sentetik kontrol; bozuk geometri/warp testleri | Bağımsız gerçek veri ve köşe ground truth'u; gece/örtülme kapsamı |
| M4 | Mac RAM/gecikme/idle/restart ve browser kontrolü | Sunucu ölçümü; bütçe TBD |
| M5 | Yerel başlatıcı; opt-in image config + read-only ONNX/provider overlay | Linux image build/smoke ve canlı yayın onayı |

[Yayın/rollback tarifi](../operations/learned-corners-release.md) hazırdır;
paketleme hazırlığı Linux image'ın test edildiği anlamına gelmez. Tek thread,
normal 1 / miss halinde en fazla 3 model geçişi uygulanır. Learned model'den
geri dönüş için `corner_model` kaldırılıp web provider `project` yapılır;
yalnızca `estimate_plate_geometry=false` yeni ağı devreden çıkarmaz.

Son düşük-yük kontrolü: **423 Python testi geçti, 3 private opt-in atlandı**.
Paketleme, Compose ve geometri kanıtları [17 Ekim raporunda](../reports/learned-corners-packaging-validation.md).

20 Ekim devam kontrolü: **438 genel Python testi geçti**; native corner + OCR
ile iki fotoğraflık private smoke ayrıca geçti. Tam model hash ön kontrolü,
Docker'sız doğrulama ve ayrı builder config'i hazır.
[Güncel kanıt ve kalanlar](../reports/release-preflight-20261020.md).

21 Ekim: sunucu kapasitesi bilgisi için model/daemon açmayan `doctor` komutu
eklendi. [Kontrol kapsamı ve kalan dış bağımlılıklar](../reports/host-doctor-20261021.md).

23 Ekim: PyTorch ve eski bundle olmadan çalışan opt-in `lean_corners` yolu,
tek venv Docker tarifi ve model bütünlük/başlatıcı seçimi hazır. Mac'te gerçek
HTTP smoke geçti; Linux image boyutu ve kaynak uygunluğu henüz ölçülmedi.
[Lean kanıt ve kullanım](../operations/lean-corners-runtime.md).

## İlk uygulama planı (tarihsel durum ve efor tahminleri)

| Faz | Amaç / işler | Teslimat | Doğrulama / kabul | Risk | Aktif efor | Durum |
| --- | --- | --- | --- | --- | --- | --- |
| M0 | Yanlış geometriyle OCR'ı kes; legacy rollback'i koru | Runtime guard + worker çağrılmadığını denetleyen test | Geometri yokken recognition null, OCR süresi 0; rollback ayrı çalışır | Köşe bulunamayan düz fotoğraflarda da çekimserlik artabilir | 0.25 gün | Uygulandı; 402 genel Python testi geçti |
| M1 | CPU üzerinde gerçek dört nokta üreten hazır ağı, checkpoint kaynağını ve lisans kapsamını doğrula | Sabit revision/URL/hash, model kartı ve kullanım kararı | Bbox-only ağ veya rastgele başlatılmış corner head kabul edilmez | Kod lisansı weights kapsamını otomatik kanıtlamaz; veri alanı farkı | 0.5–1 gün | Araştırma başladı; kabul edilmiş checkpoint yok |
| M2 | Geçerli checkpoint'i mevcut provider sınırına bağla; bbox + dört gerçek keypoint'i aynı geçişte al | Opt-in CPU adapter; crop homography ve model kimliği provenance'ı | Letterbox ters dönüşümü, sıra, convexity, finite/bounds ve crop testleri | Şema/hash eşleştirmeleri ve runtime uyumluluğu | 1–2 gün | Bekliyor: M1 |
| M3 | Sadece mevcut izinli veriyle perspektif/rotasyon/ölçek/ışık regresyonu yap | Önceden belirlenmiş perturbation matrisi; aggregate sonuçlar | Corner error, polygon IoU, kabul edilen OCR hata ve çekimserlik ayrı; eşikler TBD | Sentetik veri gerçek dünya holdout'u değildir | 0.5–1 gün | Bekliyor: M2 |
| M4 | Tek CPU thread, tek model geçişi, düşük input boyutu ve web akışı ölç | RAM/gecikme raporu, aynı dörtgen + warp preview | Mevcut küçük gerçek kümede kabul kaybı/yanlış okuma yok; bütçe TBD | Çok küçük model küçük plakaları kaçırabilir | 0.5 gün | Bekliyor: M3 |
| M5 | Web ve inference'ı birlikte değiştir; önceki sürümü sakla | Kullanıcıya sunulan doğrulanmış aday ve rollback | Gerçek upload + worker çıkışı; açık yayın onayı | Dağıtım farklılıkları | 0.5 gün | Bekliyor; commit/push/yayın yetkisi yok |

Eforlar tahmindir; model/veri/lisans bulunabilirliği ve indirme süresi dahil değildir.
CUDA ve yerel eğitim bu planın önkoşulu değildir. Yeni model mevcut detector'ın
yerine aday olur; iki detector sürekli birlikte çalıştırılmaz. Ana OCR korunur.

## İlk araştırma bulgusu (tarihsel; seçim öncesi)

HyperLPR'nin resmi `MultiTaskDetectorDNN` kaynak kodu OpenCV DNN ile ONNX
çıkarımı ve bbox'a ek sekiz köşe koordinatı içerir. Bu teknik olarak adaydır;
proje Çin plakalarına odaklanır ve Türk plakalarında uygunluğu kanıtlanmış değildir.
Resmi kod deposu Apache-2.0 lisans metni taşır. Checkpoint'in dağıtım/kullanım
kapsamı, erişilebilirliği, hash'i ve TR başarısı ayrıca doğrulanmadan aktive edilmez.
İncelenen Git tree'de ONNX binary bulunmadı; settings'teki model konumunu
çalışan, kabul edilmiş bir artifact gibi varsaymak doğru değildir.

- İncelenen revision: `9307450f7b7915be18f23a539ec05b41fe6629f4`.
- [Resmi detector kaynak kodu](https://github.com/szad670401/HyperLPR/blob/9307450f7b7915be18f23a539ec05b41fe6629f4/Prj-Python/hyperlpr3/inference/multitask_detect.py)
- [Resmi model ayarları](https://github.com/szad670401/HyperLPR/blob/9307450f7b7915be18f23a539ec05b41fe6629f4/Prj-Python/hyperlpr3/config/settings.py)
- [Resmi proje/lisans](https://github.com/szad670401/HyperLPR)

Bu ilk araştırma anında model indirilmemişti. Sonraki yerel model seçimi ve
entegrasyon yukarıda kayıtlıdır; genel perspektif doğruluğu garantisi verilmez.
