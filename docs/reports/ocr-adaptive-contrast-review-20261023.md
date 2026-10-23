# OCR iyileştirmesi: uyarlanabilir kontrast ve yinelenen görünüm denetimi

Tarih: 23 Ekim 2026. Önceki çalışma:
[Detection/OCR dayanıklılık incelemesi](plate-robustness-review-20261023.md).

## Uygulanan değişiklik

Yüksek güvenli, yeterince büyük kırpımların tek okuma yolu korunur. Küçük veya
belirsiz kırpımlarda daha önce kullanılan gri doğrulama görünümü, yalnız düşük
kontrast varsa kontrollü parlaklık genişletmesiyle değiştirilir. Sonraki görünüm
yine beyaz kenarlıklı kırpımdır. Model, ağırlıklar, üretim bağımlılıkları ve en
fazla üç seri OCR çağrısı sınırı değişmez.

Kontrast kararı, gri görüntünün %2 ve %98 parlaklık yüzdelikleri arasındaki
farka dayanır. Fark 16–96 aralığındaysa (üst sınır hariç) bu aralık 0–255'e
genişletilir. Daha düz girdilerde gürültüyü büyütmemek için genişletme yapılmaz;
normal/yüksek kontrastta önceki gri görünüm kullanılır. Bu sınırlar geçici bir
ön işleme politikasıdır, bağımsız holdout üzerinde kalibre edilmiş değildir.

Bu işlem boyutu, dörtgeni veya karakterleri kesmez; yapay ayrıntı üretmez,
karakter/il kodu değiştirmez. OCR'a verilen seçili görünüm, mevcut teknik
önizlemeye aynı yardımcı işlevle aktarılır.

Ayrıca:

- Aynı piksel girdisi ikinci kez çalıştırılmaz ve bağımsız bir oy sayılmaz.
  Özellikle zaten gri görüntülerde temel görüntü ve gri görünüm aynı olabilir.
- Okumalar anlaşmıyorsa ilk okumanın metni **ve kendi güven puanı** korunur;
  başka bir adayın puanı ilk metne aktarılmaz. Sonuç kabul edilmez.
- Anlaşma halinde destekleyen skorların en düşüğü kullanılır. Skor yükseltme
  veya biçim kuralıyla harf/rakam düzeltme yapılmaz.
- Birden çok okuma kullanılan kararlar `consensus-v2` sürümüyle izlenir.

## Küçük, eşleştirilmiş geliştirme kontrolü

Beş kullanıcı ekran görüntüsünden ayrılan bölümler ve daha önce kullanılan iki
özel kontrol fotoğrafı kullanıldı. Önce detector ve homografi ile kırpım
üretildi; **kırpım üzerinde** normal, karartılmış, düşük kontrast ve hafif bulanık
varyantlar oluşturuldu. Böylece yedi kaynaktan 28 ilişkili tanı girdisi elde
edildi. Bu test, farklı koşullarda kaynak fotoğrafın detection başarısını ölçmez.

Her görünüm aynı model/tek worker ile seri olarak okundu. Önceki sabit gri +
beyaz kenar politikası ve yeni uyarlanabilir politika aynı okumalar üzerinden
karşılaştırıldı. Yeni politika ayrıca gerçek `WebOcrRuntime` çağrısıyla
karşılaştırılarak seçilen metin ve çağrı sayısı doğrulandı.

| Ölçüm | Önceki politika | Yeni politika |
| --- | --- | --- |
| Beklenen metni üreten girdi (kabul veya kontrol adayı) | 22/28 | 24/28 |
| Yanlış kabul edilen sonuç | 0 | 0 |
| Toplam mantıksal OCR çağrısı | 64 | 64 |
| Önceden doğruyken yanlışlaşan girdi | — | 0 |

Net artış iki girdidir. Dört kalan hata aynı
zayıf kaynak örneğin varyantlarıdır; bu örnekte kırpım/karakter ayrıntısı sorunu
halen vardır. Yedi kaynak, bağımsız bir holdout değildir; tabloda genel doğruluk
oranı, istatistiksel anlamlılık veya gelecekte hatasızlık iddiası yoktur.
Görünümler aynı OCR modelinden geldiği için anlaşma da doğruluk kanıtı değildir.

Özel geliştirme raporu: `ocr-adaptive-contrast-regression-20261023-v1.json`.
SHA-256: `c22df7b2c9c6f0feedc2ebd389767cbdb39654095c272608a722a688979d233b`.
Görseller, metin bazlı özel sonuçlar ve değerlendirme betiği public pakete eklenmez.

## Yazılım ve kaynak doğrulaması

| Kontrol | Sonuç |
| --- | --- |
| Python regresyonları | 484 geçti, 4 özel opt-in test varsayılan koşuda atlandı |
| Gerçek lean HTTP/model smoke | Ayrıca opt-in ile geçti; üç pozitif okuma ve boş görüntü kontrolü |
| Worker yaşam döngüsü | Smoke sonunda idle süresi sonrası worker bırakıldı |
| Statik kontroller | Değişen kaynak ve test dosyasında Ruff; kaynak dosyasında mypy başarılı |
| Üretim bağımlılığı / model indirme / eğitim | Bu iterasyonda yok |

Tüm native kontroller tek CPU iş parçacığıyla ve seri yürütüldü. Karşılaştırmada
yeni politikanın önceden ısıtılmış OCR çağrıları ortalama yaklaşık 126 ms, en
uzun yaklaşık 203 ms sürdü. Bu rakamlar kaynak görüntü detection süresini, soğuk
başlangıcı, HTTP veya sunucu kısıtlarını kapsamaz.

Mac gerçek HTTP smoke sırasında istek sonrasında parent + worker toplam RSS
yaklaşık 1.063 GiB olarak örneklendi; worker bırakıldıktan sonra parent yaklaşık
310 MiB idi. Bunlar cgroup peak veya üretim RAM sınırı değildir. macOS ölçümleri
önceki Linux ölçümleriyle doğrudan karşılaştırılamaz. Test süreçleri kapanır;
bu değer kalıcı arka plan eğitim yükü değildir.

## Kalan işler ve geri dönüş

| İş | Durum / kabul kapısı |
| --- | --- |
| Düşük kontrast doğrulama görünümü | Uygulandı, sınırlı geliştirme kontrolünde iyileşme gösterdi |
| Tekrar eden piksel oyları / aday güven puanı | Düzeltildi ve regresyon testleri eklendi |
| Kalan zayıf kırpım ve OCR hataları | Çözülmedi; daha geniş, bağımsız pozitif/negatif set ve karakter hata ölçümü gerekli |
| Güven/ön işleme kalibrasyonu | TBD; 28 türetilmiş örnekle eşik optimize edilmez |
| Linux kaynak kontrolü | Yeni kodla peak RAM, cold/warm ve en kötü durum gecikmesi ölçülmeli |
| Canlı yayın | Bu çalışma sırasında yapılmadı |

Detection ve web tasarımı bu iterasyonda değiştirilmedi. Yerel inceleme sunucusu
yeni OCR kodunu yüklemek için yeniden başlatılabilir; canlı yapılandırma değişmez.
Yeni kaynak paketi eski paketin üzerine yazılmaz. Bir sonraki kontrollü yayında
önceki imaj çifti ve model mountları korunur; sorun halinde önceki imaj çiftine
dönülür. Commit veya push yapılmaz.
