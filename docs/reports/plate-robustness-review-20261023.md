# Detection ve OCR dayanıklılık incelemesi — 23 Ekim 2026

## Kapsam ve sınırlar

Beş kullanıcı ekran görüntüsündeki iki tespit kaçırması ve bir OCR karakter
hatası incelendi. Ekran görüntüleri orijinal yüklemeler değildir. Tanı sırasında
yalnız görüntü bölümleri bellekte ayrıldı; bu koordinatlar veya beklenen plaka
metinleri uygulama koduna eklenmedi. Bu çalışma bağımsız bir doğruluk ölçümü
veya tüm gelecek girdiler için hatasızlık garantisi değildir.

## Uygulanan değişiklikler

| Alan | Değişiklik | Kaynak sınırı / güvenlik |
| --- | --- | --- |
| Detection | İlk geçiş başarısızsa aynı modelde 1/3 bağlam ölçeğinde ikinci geçiş | Sabit giriş tensörü, kaynak görüntüyü büyüten kopya yok; normal başarı tek geçiş |
| Recovery | Bağlam geçişi de başarısızsa mevcut ±30° kurtarma geçişleri | Toplam en fazla dört seri detector geçişi; eşik ve dörtgen geçerlilik kontrolleri korunur |
| OCR | Belirsiz veya 32 pikselden kısa kırpımda gri ve beyaz kenarlıklı görünümler | Tek worker, aynı ağırlıklar, toplam en fazla üç seri okuma |
| OCR kararı | Tam metin üzerinde en az iki görünümün anlaşması; destekleyen skorların en düşüğü | Karakter/il kodu değiştirme, skor yükseltme veya format kuralıyla metin uydurma yok |
| İzlenebilirlik | Görünümlere farklı kırpım kimliği/ön işleme imzası; seçilen OCR girdisinin önizlemesi | Gösterilen girdi ile kullanılan görüntü tutarlı tutulur |
| Arayüz | Görüntü, düzeltilmiş plaka, iki sonuç kartı ve tek kısa uyarı | Teknik ayrıntılar ve doğrulama/indirme araçları kapalı açılır bölümlerde |

Yüksek güvenli ve yeterince büyük OCR girdisi tek okumada kalır. Çoklu görünüm
aynı modelden gelir: anlaşma bağımsız kanıt değildir ve ortak model hatasını
ortadan kaldırmaz. Güven eşiği halen geçici bir işletim noktasıdır; kalibrasyon
TBD. Üç görünüm çelişirse ilk okuma yalnız kontrol gerektiren aday olarak kalır.
Biçim doğrulaması OCR metninden ayrıdır; geçersiz il kodu metni gizlemez.

## Tanı sonuçları

Her ekran görüntüsü bölümü için özgün ekran pikseli, %75 yeniden boyutlandırma
ve JPEG kalite 80 varyantı incelendi. Aynı görüntünün varyantları bağımsız örnek
sayılmaz. Dört köşenin etiketlenmiş gerçek değerleri bulunmadığından corner
hatası veya polygon IoU raporlanamaz.

| Tanı | Sonuç | Yorum |
| --- | --- | --- |
| Önceki detector, beş ekran bölümü | 3/5 tespit | İki kaçırma bu temsilde tekrar üretildi |
| Güncel detector, beş bölüm × üç varyant | 15/15 tespit | Küçük, türetilmiş bir tanı setidir; genelleme oranı değildir |
| Güncel Paddle OCR, aynı 15 varyant | 12/15 beklenen metin | Birinci örneğin iki varyantı ve ikinci örneğin küçültülmüş varyantı halen sorunlu |
| CCT-XS-v2 global OCR adayı, aynı 15 varyant | 11/15 beklenen metin | Daha iyi olmadığı için üretime entegre edilmedi |
| Siyah ve beyaz boş görüntü | 0/2 tespit | Yalnız basit negatif kontrol; gerçek negatif test seti değildir |

Kullanıcının ikinci örneğindeki 0→3 hatası aynı biçimde tekrar üretilemedi:
ekran görüntüsünden ayrılan temel girdide mevcut OCR doğru metni okudu.
Dolayısıyla özgün yüklemedeki bu hatanın kesin çözüldüğü iddia edilmez.
Birinci örnekte kırpım ve OCR halen zayıftır. Tek görüntüyü düzelten büyük
dikey pay gibi denemeler, genelleme kanıtı olmadığından uygulamaya alınmadı.

## Doğrulama ve kalan yayın kapıları

Doğrulama sonucu: 471 Python testi geçti, 4 opt-in test normal koşuda atlandı.
Gerçek lean HTTP/model smoke testi ayrıca opt-in ile çalıştırıldı ve geçti.
Ruff ve dört değişen kaynak dosyası için mypy başarılı. Web TypeScript/lint,
tek worker üretim derlemesi ve 6 web sınır testi başarılı.
Public adayın şema, sözleşme, doküman bağlantısı ve hassas dosya taramaları geçti.

Yerel gerçek tarayıcı yüklemesinde özgün görüntü/dörtgen ve düzeltilmiş plaka,
iki sonuç kartı ve iki varsayılan kapalı ayrıntı bölümü doğrulandı. Teknik
ayrıntılar açıldığında gerçek OCR girdisi gösteriliyor. 320 ve 390 piksel mobil
genişliklerinde yatay taşma bulunmadı; iletişim bağlantısı görünür kaldı.
Bu UI kontrolü canlı sunucu veya Linux kaynak ölçümü değildir.

| Faz | Durum | Kalan / kabul ölçütü |
| --- | --- | --- |
| Genel ölçek/bağlam kurtarma | Uygulandı | Daha geniş, bağımsız pozitif ve negatif görüntü setinde recall ve yanlış pozitif ölçümü TBD |
| Muhafazakâr OCR çoklu görünüm | Uygulandı | Türkçe plaka holdout üzerinde tam metin doğruluğu, CER, yanlış kabul ve kalibrasyon TBD |
| Sonuç alanını sadeleştirme | Uygulandı ve yerelde doğrulandı | Canlıda yeni web imajıyla son kontrol |
| Kaynak tüketimi | Mac üzerinde seri CPU akışı | Yeni kod için Linux cgroup peak RAM, soğuk/sıcak gecikme, kötü durum zaman aşımı ve idle-release tekrar ölçülmeli |
| Canlıya geçiş | Yapılmadı | Hem inference hem web için yeni etiketli imaj; önce tek istekli smoke, sonra kontrollü geçiş |

Önceki Linux RAM/gecikme sayıları yeni çoklu geçiş kodu için ölçüm değildir.
Yeni model veya ağır üretim bağımlılığı eklenmedi; yine de daha çok seri geçiş
CPU süresini artırabilir. Her worker isteğinin zaman aşımı ayrı olduğundan
üç OCR denemesi ile toplam inference deadline uyumu Linux üzerinde sınanmalıdır.

Canlıdaki v6 web ve mevcut inference imajları bu çalışma sırasında değiştirilmedi.
Geri dönüşte önceki web/inference imaj çifti ve aynı salt okunur model mountları
korunmalıdır. Kullanıcı onayı olmadan commit veya push yapılmaz.

## Model kaynakları

- [HyperLPR resmi detector uygulaması](https://github.com/szad670401/HyperLPR/blob/9307450f7b7915be18f23a539ec05b41fe6629f4/Prj-Python/hyperlpr3/inference/multitask_detect.py): letterbox ve çıktı düzeni incelemesi.
- [HyperLPR resmi proje kapsamı](https://github.com/szad670401/HyperLPR): Çin odaklı modelin ülke/domain kayması riski devam eder.
- [fast-plate-ocr resmi model listesi](https://github.com/ankandrew/fast-plate-ocr/blob/master/docs/inference/model_zoo.md): CCT-XS-v2 global adayı yalnız özel değerlendirme alanında sınandı; üretim modeli değiştirilmedi.

Reddedilen OCR adayının ONNX SHA-256 değeri:
`8031afb5fdc6b4d80462c9d542f1284ebd2cfddf5dbacd62609848d7e2855f44`.
Aday dosyaları ve değerlendirme ortamı özel çalışma alanındadır; public
pakete model ağırlığı veya kullanıcı ekran görüntüsü eklenmez.
