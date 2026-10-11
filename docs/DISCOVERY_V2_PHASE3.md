# Discovery V2 — Phase 3: Opportunity Gap Analysis & Challenger Research

## Kısa sonuç

**90 seans analiz edildi; test edilen iki challenger için daha iyi kârlılık kanıtı çıkmadı.**
Prebreak daha fazla aday buluyor ama değerlendirmede maliyet sonrası20g forward
ortalaması−%2,30; Relaunch−%7,67 ve yalnız17 olay. Champion raw discovery aynı
ölçümde−%2,62. Bu sayılar production T1/T2/stop getirisi veya trade expectancy değildir.
En güçlü doğrulanmış kazanım, gap/attribution ve karşılaştırmanın ölçülebilir hale
gelmesi; yeni kazandıran strateji bulunmuş olduğu iddia edilmiyor.

## Kapsam ve kanıt standardı

Production VCE/RVOL, eşikler, T1/T2, stop/trailing, scheduler, risk planner, paper
trading ve deployment değiştirilmedi. Yeni kod yalnız `swing_trader/discovery/research/`
ve `swing_trader/discovery/operations/` altında. Yeni dependency ve OHLCV API isteği yok.
9 Ekim'deki19 champion PROSPECTIVE_SHADOW kaydı değiştirilmedi.
Başlangıçta çalışma ağacında bulunan production değişiklikleri bu turda yapılmadı
ve geri alınmadı.

Bu rapor **HISTORICAL_RESEARCH** analizidir. 10 Ekim2026 NASDAQ+NYSE master'ının
geçmişe uygulanması survivorship/listing-selection bias içerir. Delisted evreni yok;
bugünkü NASDAQ/NYSE/security-type sınıflandırması geçmiş listing kanıtı değildir.
Hedef small/mid-cap olsa da geçmiş cap bilinmediği için analiz evreni gerçekten
small/mid-cap doğrulanmış bir evren olarak sunulamaz.

Mevcut stratejinin parametreleri ve şirket listesi bu tarihlerin sonrasında seçilmiş
olabilir. Kronolojik değerlendirme dilimi **bağımsız forward/OOS validation değildir**.
Bugünkü market cap/float/earnings geçmiş kararlarına taşınmadı. Fiyat/likidite uygunluğu
bir uygulanabilirlik proxy'sidir; tam trade eligibility veya manuel fill garantisi değil.

## 1. Veri envanteri ve sabit araştırma protokolü

Kaynak: Phase2 `full/preparation.json` içindeki immutable nesne kimlikleri;
Phase2 `full_final/inputs.json` içinden config ve master. Current facts kullanılmadı.
`research/__main__.py::inventory`, sonuçlara bakmadan
`output/discovery_v2/phase3/inventory.json` dosyasını oluşturdu.

- 5.245 aday;5.221 immutable OHLCV nesnesi.24 adayda nesne yok (12 mapping,12 veri hatası).
- SPY:4 Eylül2025–9 Ekim2026,277 bar. VIX:9 Eylül–9 Ekim2026,23 bar.
- Sabit130-bar araştırma warm-up +30-forward koşuluyla en çok118 ortak takvim
  karar seansı mümkün. Veri yeterliliği ticker/seans bazında ayrıca kontrol edilir.
- Son90 olgunlaşmış karar seansı: **21 Nisan–27 Ağustos2026**. Envanterde en az130
  mevcut barlı hisse sayısı günlük en az4.814; bu, henüz kesintisizlik/fiyat/DV kontrolü
  uygulanmamış envanter sayısıdır.
- Araştırma: ilk30 seans; purge: sonraki30; değerlendirme: son30. Araştırmanın son
  kararının30g sonucu değerlendirme başlamadan biter. Purge dönemi sadece betimleyici
  tabloda tutulur, araştırma/değerlendirme karşılaştırmasına karışmaz.
  Kesin tarihler: araştırma **21 Nisan–2 Haziran**, purge **3 Haziran–16 Temmuz**,
  değerlendirme **17 Temmuz–27 Ağustos2026**.
- Ana sonuç: next-open→20.seans close,50bps roundtrip.5/10/30 seans ve20/100bps
  duyarlılık olarak birlikte raporlanır; en iyi horizon sonradan seçilmedi.
- Ana fırsat etiketi: next-open→20.seans close **≥%10**. MFE≥%10/20/30 ve diğer close
  eşikleri ayrı; MFE gerçekleşmiş getiri sayılmaz.
- Aynı ticker'da ilk kronolojik olay alınır, sonraki30 seans aynı olay penceresidir.
  Tek uzun rallinin her günü bağımsız başarı sayılmaz. Olay başlangıcındaki trigger
  yakalama ve sonraki30 günde herhangi bir trigger görülmesi ayrı ölçümlerdir.

Protokol `research/protocol.py::PROTOCOL` içinde, envantere kopyalanarak sonuçlardan
önce sabitlendi. İki hipotez, tek parametre seti; grid/random search veya optimizasyon yok.
130 bar, bu araştırmanın ortak ve muhafazakâr warm-up koşuludur; production'ın
RVOL52/VCE55 minimumuyla aynı değildir.52–129 barlı IPO'ların bu ana cohort dışında
kalması "production kesin tarayamaz" anlamına gelmez. Eksik geçmiş kategorisi bu
araştırma tanımına aittir. Karşılaştırılan bütün yollar aynı130-bar koşulunu kullanır.

### Adjustment ve veri sınırları

`research/features.py::decision_frame/features`: provider'ın split-adjusted OHLC'si,
karar tarihinden sonraki cache'te kayıtlı split çarpanlarıyla o tarihin fiyat birimine
geri ölçeklenir; hacim ters ölçeklenir. AdjClose/Close katsayısı karar tarihindeki
katsayıya normalize edilir, gelecekteki ortak temettü faktörü seçimden çıkarılır.
Bu işlemin split/temettü değişmezliği ve gelecekteki fiyat değişiminin geçmiş
seçimi etkilememesi test edildi. Fakat sonradan düzeltilmiş vendor verisi gerçek
o-gün bilinen veri haline dönüşmez; eksik corporate-action kaydı bağımsız doğrulanamaz.

Forward ölçüm aynı cache revizyonunun split/dividend-adjusted oranlarıdır; broker
nakit getirisi değildir. İlk planlı sonraki seansın Open'ı kullanılır; eksik giriş barı
atlanmaz. Horizon içinde eksik/zero-volume bar varsa sonuç geçerli fill proxy'si
sayılmaz, missing/unreliable olarak sayılır. Bu eksikler sinyal sayısından silinmez.
20/50/100bps maliyet entry/exit fiyatlarına yarı yarıya olumsuz uygulanır; gerçek
spread/impact ve gap fill verisi yoktur. Düşük likiditenin uygulanabilirlik riski için
[SEC'nin small-cap likidite çalışması](https://www.sec.gov/files/marketstructure/research/small_cap_liquidity.pdf)
bağlam sağlar; bizim sayıların kaynağı yerel cache'tir.

## 2. Champion predicate'leri ve atıf yöntemi

Kaynak `small_cap/signals.py::check_vce_breakout/check_rvol_thrust`:

| Yol | Zorunlu koşullar |
|---|---|
| VCE | Dün ATR14% / önceki baseline <0,8; kapanış önceki20 High üstünde; önceki close'a göre yeşil; MA50 üstünde; bugün dahil mean50 hacme göre RVOL≥1,5 |
| RVOL | 2,5≤RVOL<4; yeşil; günlük değişim<%8; MA20 üstünde |

`research/features.py::features` bütün predicate'leri short-circuit olmadan ayrı
hesaplar. Bu bir production motor değişikliği değildir. Her trigger-positive
ticker/seans ve ticker başına deterministik bir negatif seans, orijinal iki production
fonksiyonuyla karşılaştırılır; uyuşmazlık olursa çalışma hata vererek durur.

Ek koşullar `research/panel.py::selected_details` üzerinden Phase2 `shadow.evaluate`
ve mevcut production swing/RSI/Weinstein/scoring fonksiyonlarıyla çalıştırılır.
Bütün trigger'lar, ortak evrendeki bütün challenger gözlemleri ve ana güçlü-fırsat
kümesinin bağımsız başlangıçları için bu detaylar saklanır. Bütün negatif ticker/günler
temel panelde kalır; yalnız kazananlardan strateji seçilmez.

Tarihsel float olmadığı için Quality numeric sonucu production bilinmeyen-float
varsayılanıyla **diagnostic** skordur. VIX'in tamamı analiz tarihleri sonrasında:
tam rejim UNKNOWN; dolayısıyla kalite kapısı da UNKNOWN. SPY-only bağlam ayrı
gösterilir, VIX=0 uydurularak BULL ilan edilmez.

Bir predicate'i kaldırma analizinde, aynı yolun **diğer tüm predicate'lerini** geçen
ve champion'ın o gün seçmediği ek adaylar alınır. Kaybedenler ve maliyetler dahil edilir.
Bu tablolar tanısal ablation'dır;10 deneme arasından en iyisi production'a önerilmez.

## 3. Önceki deneyler: yeniden keşif yapılmadı

`scripts/discover_signal_families.py::p_momentum_continuation`:20g getiri>%15,
close>MA20>MA50, yeşil, bugün<%8. Yeni relaunch bu kural değildir.
`p_tight_consolidation_break`: geçmiş10g bant<%8,10g tepe kırılımı, yeşil,MA50;
önceden denenmiş genel sıkı-konsolidasyon kuralı da yeni diye sunulmadı.

Mevcut artefaktlar salt-okunur yeniden sayıldı; bunlar bu turda yeniden çalıştırılmış
veya bağımsız doğrulanmış getiri ölçümleri değildir:

| Eski kaynak / seri | Örneklem | Kayıtlı eski `r` ortalaması |
|---|---|---:|
| `third_pathway.json`, Momentum devamı Q≥80 | n110,9 Eylül2024–29 Nisan2026 | −%1,50 |
| Aynı serinin1 Haziran2025 öncesi | n28 | −%3,50 |
| Aynı dosya, sıkı konsolidasyon Q≥80 | n45,29 Ağustos2024–26 Mart2026 | +%2,46 |
| Sıkı konsolidasyon train | n12 | −%0,05 |
| `tight_consolidation_live.json`, V1 hacim≥1,5 | n63,10 Eylül2024–28 Nisan2026 | −%0,39 |
| V2 hacim≥2 | n33,17 Ekim2024–27 Nisan2026 | −%1,41 |
| V3 hacim+olay sınırları | n54,10 Eylül2024–28 Nisan2026 | +%0,46 |
| V4 bant<%6+hacim | n31,10 Eylül2024–28 Nisan2026 | −%0,43 |

Bu `r` değerleri eski engine/exit harness çıktısıdır, Phase3 forward-close getirisi
değildir. Eski yorumdaki başka n/EV sayılarını güncel gerçekler gibi kopyalamadık.
Kaynak `measure_third_pathway.py::run_pattern/stats`, `measure_tight_consolidation.py`.
Current shares/historical cap, eski evren, örtüşen işlemler ve eski holdout sorunları
nedeniyle bu dosyalar yeni hipotez için bağımsız doğrulama oluşturmaz.

## 4. Önceden sabitlenen iki challenger

| Hipotez | Tek sabit tanım | Ekonomik gerekçe / risk |
|---|---|---|
| PREBREAK_ACCUMULATION | MA20>MA50; close/prior20High0,97–1,00;5g range/20g range≤0,5;10g up/down dolar hacim≥1,2;median5/median20 hacim≤0,8 | Kırılımdan önce daralan arzı arar. Teyitsiz bekleme/başarısız kırılım riski; eski yalnız hacim ignition değildir. |
| IMPULSE_BASE_RELAUNCH | t−10→t−5 getiri%10–40; önceki5g bant≤%8;MA20>MA50;close önceki5 High üstünde;RVOL20≥1,2;bugün<%8 | Önceki ivmenin sindirilmesi sonrası talebin dönmesini arar. Eski genel continuation/tight10 tekrarından farklı, ama otomatik edge sayılmaz. |

Üç yol aynı130-bar yeterlilik, fiyat$7–200,20g ortalama dolar hacim≥$5M,
aynı tarihler, next-open, maliyet ve horizon'larla karşılaştırılır.
Karşılaştırılan **raw discovery yöntemidir**, doğrulanmış bütün production trade
gates'i geçen nihai portföy veya exit stratejisi değildir. Challenger'ın kalite
skoru yeni stratejiye göre yeniden kalibre edilmedi; production eşiği değiştirilmedi.

Belirsizlik: ticker-kümeli bootstrap1.000 çekiliş/sabit seed7301; aynı tarihli
şoklar için date-kümeli duyarlılık; farkta ortak cluster ağırlıkları. Değerlendirme
dilimi30 gün olduğundan30g outcome'lar için yalnız bir takvim bloğu vardır.
Bu aralıklar çok-rejim zaman serisi doğrulaması veya bağımsız30 günlük örneklemler
gibi yorumlanamaz. Fark anlamlı görünse bile bağımsız forward doğrulama gereklidir.

## 5. Çalıştırılmış araştırma sonuçları

Birincil sonuç dosyaları: `output/discovery_v2/phase3/run90/summary_final.json`
ve `supplement_final.json`. Önceki kısmi `summary.json` geçerli sonuç değildir.

**5.245 ×90 =472.050 ticker/seans satırı** işlendi;19 dakika23,109sn panel süresi.
9.512 detaylı gate/score satırı, **8.715 gerçek-veri direct predicate parity kontrolü,
0 uyuşmazlık**. Günlük130-bar kesintisiz coverage4.813–4.944; toplam438.829 ticker/seans.
Fiyat/DV ve geçmişi birlikte uygun günlük evren2.102–2.239; toplam**196.045** ticker/seans.
Güncel evrende olup o gün fiyatı olmayan61–226 aday tutuldu, sessizce kaybolmadı.
Bu son sayı, kısa/geçmişi eksik bütün adayların toplamı değildir.

### 5.1 Güçlü hareket etiketleri ve yakalama sınırı

Aşağıdaki sayı **fiyat/likidite uygun,20.seans close getirisi≥%10** olan ilk kronolojik
olay başlangıçlarıdır: **3.537**. Aynı ticker'da30-seans kümelenmesi uygulanmıştır.
Bu ekonomik olarak tamamen farklı3.537 ralli veya3.537 tam uygun trade demek değildir.

| Olay başlangıcındaki durum | Sayı |
|---|---:|
| Champion trigger yok; bilinen downstream kapılardan da en az biri FAIL | 3.067 |
| Champion trigger yok; swing/RSI/stage bilinen kontroller açık | 451 |
| Champion trigger var; bilinen downstream kapılar açık | 12 |
| Champion trigger var; bilinen downstream ret var | 7 |
| Toplam | 3.537 |

**19 başlangıçta trigger** görülmesi (%0,54), executable opportunity recall değildir.
Başlangıç, gelecekteki20g getirisine göre seçilen bir takvim ankerasıdır; rally'nin
gerçek başlangıcı veya üretimdeki entry günü değildir. Pencere başında daha önce
oluşmuş sinyaller gözlenmiyor olabilir. Olaydan sonra gözlenen30-gün alanında herhangi
bir trigger797 kümede görülür; bu sayı geç sinyalleri de içerir ve son ankeralarda
90-gün tarama sınırı nedeniyle sağdan sansürlüdür. **797/3.537 de executable recall
diye sunulamaz.** "Sistem fırsatların%99,46'sını kaçırıyor" sonucu çıkarılmadı.

İstenen altı kategori açısından:

1. Bütün gerekli trade kapılarını geçtiği doğrulanmış fırsat: **kanıtlanmış0;
   gerçek uygun sayı UNKNOWN** (tarihsel cap/earnings/float ve tam rejim yok).
2. Bu3.537'lik altkümede fiyat/DV ret0, tanım gereği. Ayrı fiyat/DV-dışı taramada
   trigger'ı olan güçlü295 olay bulundu; bunlar uygun evrenin denominator'una katılmaz.
3. Trigger + bilinen downstream ret:7.
4. Trigger yok + bütün diğer trade koşulları geçti: doğrulanamaz.451 olay yalnız
   bilinen OHLCV-based downstream kontrolleri geçer; tam eligible değildir.
5. Öncelikli sınıflamada kalan3.530 olayda metadata UNKNOWN vardır; ancak bunun
   tek ret nedeni olduğu söylenmez. Yukarıdaki matrix bilinen retleri de korur.
6. Ayrı fiyat/DV/geçmiş-dışı güçlü etiket havuzunda4.497 kümelenmiş olay vardır;
   225'i130-bar araştırma geçmişini sağlamaz. Bu ayrı havuzlar kendi içinde
   kümelendiğinden birbirine/toplam3.537'ye eklenerek bağımsız fırsat sayılmaz.

| Horizon | Close≥%10 olay / başlangıç trigger | Close≥%20 | Close≥%30 | MFE≥%10 | MFE≥%20 | MFE≥%30 |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 2.823 /24 | 985 | 354 | 3.858 | 1.481 | 617 |
| 10 | 3.368 /19 | 1.395 | 629 | 4.694 | 2.107 | 993 |
| 20 | 3.537 /19 | 1.758 | 931 | 5.266 | 2.771 | 1.485 |
| 30 | 3.472 /36 | 1.867 | 1.048 | 5.503 | 3.162 | 1.812 |

Her horizon/eşik kendi içinde kümelenir; sütunlar toplanmaz. MFE sayıları
high'a satılabildiğimiz anlamına gelmez. Bütün ayrıntılar outcome-group JSON'larında.

### 5.2 Trigger failure attribution ve kaldırma deneylerinin kaybedenleri

Soldaki ret sayıları aynı3.537 güçlü başlangıç üzerinde, nonexclusive'dir.
Sağdaki deneyler17 Temmuz–27 Ağustos değerlendirme döneminde o predicate kaldırılınca
gelen **bütün ek adaylar** üzerinde; diğer yol/koşullar korunur, mevcut champion
aynı gün seçmişse ek aday sayılmaz. Getiriler20g forward close,50bps maliyetlidir.

| Predicate | Güçlü başlangıçlarda FAIL | Ek raw / bağımsız olay | Eklerin ortalama / medyan net getirisi |
|---|---:|---:|---:|
| VCE squeeze | 3.150 | 846 /592 | −%1,11 /−%1,36 |
| VCE önceki20 High | 3.393 | 78 /67 | −%1,60 /−%0,77 |
| VCE yeşil | 2.486 | 0 /0 | Hesaplanamaz; diğer koşullarla redundant |
| VCE MA50 | 1.761 | 6 /5 | −%11,05 /−%7,21 |
| VCE minimum hacim | 2.998 | 513 /331 | −%2,60 /−%5,03 |
| RVOL alt sınır | 3.377 | 20.301 /2.240 | +%0,93 /+%0,31 |
| RVOL üst sınır | 56 | 26 /23 | +%4,29 /−%0,22 |
| RVOL yeşil | 2.486 | 102 /94 | −%2,11 /−%0,59 |
| RVOL günlük değişim tavanı | 67 | 147 /139 | −%2,69 /−%2,00 |
| RVOL MA20 | 2.164 | 72 /66 | +%2,38 /−%1,71 |

Bu deneyler gate kaldırma önerisi değildir. Örneğin RVOL üst sınırı kaldırmanın
ortalaması+%4,29 görünse de23 olay, medyan negatif,95% ticker-CI[−%0,82,+%10,21];
en iyi3 olay çıkarılınca+%0,09. MA20 kaldırma CI[−%1,57,+%6,74], en iyi3 hariç−%0,25.
RVOL alt sınırını kaldırma araştırma döneminde−%0,87, değerlendirmede+%0,93;
20.301 ek raw sinyale yol açıyor. Dönem farkı, yoğunluk ve çoklu deneme göz ardı edilerek
"daha çok kazandırır" denemez. Bütün10 ablation önceden tanısal amaçla çalıştırıldı;
en iyi görünen sonuçtan üçüncü challenger türetilmedi.

Güçlü3.537 ankerada ayrıca swing2.666, RSI401, Weinstein1.365 kez FAIL.
Quality gerçek ret sayısı UNKNOWN; float/regime yokken varsayılan skorla ret uydurulmadı.

Değerlendirme dönemindeki bütün champion raw tetiklerden991'i130-bar geçmişli;
684'ü fiyat/DV'den elenir,307'si ortak araştırma evrenine girer.307 üzerinde
RSI65, stage67, swing26 ret vardır (örtüşür). Bilinen downstream kontrollerden
163 raw sinyal geçer; bunlar tam eligible değil, metadata/quality UNKNOWN'dur.
Bu altkümenin143 kümelenmiş olayında20g net ortalama−%1,29. Stage-ret grubunun
net ortalaması−%6,20, swing-ret grubunun−%3,82; RSI-ret grubunun+%0,46 olması
RSI'ı kaldırmanın kanıtı değildir. Bunlar izole nedensel deney/production exit değildir.

### 5.3 Güçlü hareketlerden önce görülen özellikler

Median değerler;90 seansın kümelenmiş ankeraları. Negatif kontrol grubu gelecekte
20g close getirisi≤0 olan olaylardır. Bu **sonuç-koşullu betimleme**, predictive model
veya tek başına entry kuralı değildir. Takvim/volatilite etkileri tam eşlenmiş değildir.

| Feature | Güçlü close≥%10 | Tüm uygun başlangıçlar | Negatif başlangıçlar |
|---|---:|---:|---:|
| Squeeze ratio | 1,006 | 0,942 | 0,954 |
| RVOL50 | 0,912 | 0,862 | 0,915 |
| Önceki5g getiri | −%2,55 | +%0,69 | +%2,81 |
| Önceki20g getiri | +%0,08 | +%4,20 | +%5,94 |
| Önceki20 High mesafesi | −%10,00 | −%5,39 | −%3,48 |
| Up/down dolar-hacim10 oranı | 0,893 | 1,225 | 1,575 |

Güçlü ankeraların median ATR14'ü%4,53,20g dolar hacmi$48,22M.
Bu örneklemde tipik güçlü başlangıç, zaten zirve kıran/hacmi2,5x olan bir gün değil;
öncesinde düşüş yaşamış ve20g tepeden uzaktadır. Bu, VCE/RVOL'un neden o ankeralarda
tetiklenmediğini açıklar. **"Düşeni al" kuralını doğrulamaz**; negatif örnekler,
koşullu örnek seçimi ve piyasa dönemi birlikte test edilmeden yeni trigger yapılmadı.

### 5.4 Fair comparison: bütün kazananlar ve kaybedenler

Raw sinyal / bağımsız olay; net değerler next-open→20g close ve50bps.
Pozitif oran brüt close-return>0'dır; maliyet sonrası win rate diye adlandırılmaz.
Bütün yöntemlerde gerçek full trade eligibility UNKNOWN; fiyat/DV/geçmiş ortak PASS.

| Dönem / yöntem | Raw / olay | Net ortalama | Net medyan | Pozitif oran | En iyi3 olay hariç net |
|---|---:|---:|---:|---:|---:|
| Araştırma Champion | 472 /372 | +%1,31 | −%0,50 | %49,87 | +%0,34 |
| Araştırma Prebreak | 491 /270 | +%1,25 | +%0,69 | %57,04 | +%0,29 |
| Araştırma Relaunch | 18 /18 | +%1,08 | −%2,61 | %44,44 | −%3,01 |
| Değerlendirme Champion | 307 /250 | **−%2,62** | −%1,34 | %45,60 | −%3,49 |
| Değerlendirme Prebreak | 1.016 /468 | **−%2,30** | −%3,13 | %36,54 | −%2,58 |
| Değerlendirme Relaunch | 19 /17 | **−%7,67** | −%6,29 | %23,53 | −%9,67 |

Araştırma champion372 olaydan371'inde20g sonuç geçerli; diğer satırlarda tabloda
verilen olayların hepsinde sonuç geçerli. Eksik1 sonuç sinyal sayısından silinmedi.
30-seans diliminde her ticker en çok1 kez bağımsız sayıldığı için best3 ticker
çıkarma, bu tabloda best3 olay çıkarmayla aynı sonucu veriyor.

Prebreak değerlendirmede champion'ın o dönem hiç seçmediği**419 farklı ticker**
buluyor. Aynı-gün champion dışı1.013 raw /467 olayın net ortalaması−%2,25.
Relaunch15 yeni ticker; aynı-gün ek18 raw /16 olayda net ortalama−%7,07.
Bu "ek" Finviz legacy üyeliğine değil **champion trigger'a** göredir.

| Değerlendirme yöntemi |20g net ortalama95% ticker-CI| Champion farkının95% CI | Aynı-gün uygun evrenine karşı net fark |
|---|---|---|---:|
| Champion | [−%4,88,−%0,40] | — | −1,58 puan |
| Prebreak | [−%3,12,−%1,41] | [−2,18,+2,68] puan | −0,50 puan |
| Relaunch | [−%11,14,−%4,17] | [−9,44,−0,81] puan | −4,12 puan |

Prebreak'in champion'a göre küçük ortalama farkı belirsizlik aralığında0'ı içerir;
aynı-gün benchmark'ını da geçmez. Relaunch17 olayla önceden sabitlenen30-cluster
yeterlilik eşiğinin altındadır. Olumsuz sonuç gözlenmiştir, uzun dönem edge hakkında
güvenilir kesinlik yoktur. Date-kümeli duyarlılık da JSON'da, tek30-gün takvim
bloğunun sınırlamasını ortadan kaldırmaz.

Baseline'ın ticker başına ilk günlerden seçilmiş bağımsız örnek ortalaması
değerlendirmede+%3,29'dur; yöntemler farklı günlerde yoğunlaştığından bu tek sayıyla
"champion5,91 puan geride" gibi kontrolsüz bir iddia yapılmaz. Yukarıdaki **aynı-gün**
karşılaştırması her sinyal tarihindeki bütün uygun evren ortalamasını kullanır.

### 5.5 Horizon, maliyet, adverse path ve rejim

Değerlendirme dönemi net ortalama(%),50bps:

| Yöntem | 5g | 10g | 20g | 30g |
|---|---:|---:|---:|---:|
| Champion | −1,31 | −1,91 | −2,62 | −4,55 |
| Prebreak | −1,10 | −0,90 | −2,30 | −3,78 |
| Relaunch | −2,87 | −1,33 | −7,67 | −10,49 |

20g maliyet duyarlılığı20/50/100bps sırasıyla: Champion **−2,33/−2,62/−3,10**;
Prebreak **−2,01/−2,30/−2,79**; Relaunch **−7,39/−7,67/−8,13**.
Brüt20g ortalamalar da negatiftir:−%2,13 /−%1,81 /−%7,21. Sadece maliyet eklemek
sonucun yönünü değiştirmiş değildir.

| Değerlendirme yolu | Median MFE20 | Median MAE20 | MAE20 alt%10 | Median DV M$ / ATR14% | Mutlak next-open gap>%5 |
|---|---:|---:|---:|---:|---:|
| Champion | +%7,19 | −%7,42 | −%24,31 | 42,08 /4,42 | 28/250 |
| Prebreak | +%4,14 | −%6,75 | −%15,10 | 54,03 /2,97 | 3/468 |
| Relaunch | +%3,50 | −%9,06 | −%19,23 | 58,90 /4,07 | 1/17 |

Prebreak daha düşük volatilite/gap profili gösteriyor ama pozitif net getiri
göstermiyor. MFE pozitifken close getirisi negatif olabilir; MFE'yi exit kârı
diye okumak özellikle bu tabloda hatalı olur. Champion'ın4 olayında geçmiş zero-volume
uyarısı var; diğer ikisinde0. Gerçek spread/impact/fill ölçülmedi.

Tam rejim90 seans boyunca UNKNOWN (VIX yok). Ayrı SPY-only bağlamında değerlendirme:
Champion BULL171 olay/−%3,21, CAUTION79/−%1,33; Prebreak BULL408/−%2,63,
CAUTION60/−%0,01; Relaunch BULL15/−%8,72, CAUTION2/+%0,20.
Bu küçük gruplar yeni regime-switch kuralını doğrulamaz.

Outlier kontrolü zorunlu: örneğin **4 Haziran SDOT** ankerasında
20g close+%331,39, MFE+%837,22, MAE−%57,38 görülüyor. Cache'teki uç fiyat yolu,
doğrulanmış small/mid-cap trade veya uygulanabilir kâr değildir; cap/fill/corporate
action bağımsız doğrulaması yok. Büyük kazananları listeleyerek strateji seçilmedi;
outlier çıkarma ve bütün negatif sonuçlar yukarıdaki karşılaştırmada korunur.

## 6. Operasyonel devam — opt-in, scheduler yok

Bir sonraki tamamlanmış EOD için aşağıdaki komutlar mevcut Phase2 katmanını kullanır.
Her çıktı dizini/dosya adı yeni olmalı; `--previous` son preparation/book olmalı.
ET kapanış+30dk guard uygulanır. Bunlar bu turda veri indirmek için çalıştırılmadı:

```powershell
python -m swing_trader.discovery.phase2 prepare --plan output/discovery_v2/phase2/full_plan.json --cache output/discovery_v2/phase2/cache --run output/discovery_v2/next_eod_data --previous output/discovery_v2/phase2/full/preparation.json --workers 2 --min-interval 0.5
python -m swing_trader.discovery.phase2 scan --snapshot output/discovery_v2/final_snapshots/20261010T201236851614Z --preparation output/discovery_v2/next_eod_data/preparation.json --cache output/discovery_v2/phase2/cache --settings output/discovery_v2/phase2/effective_patch.json --output output/discovery_v2/next_eod_scan
python -m swing_trader.discovery.phase2 track --scan output/discovery_v2/next_eod_scan/scan.json --previous output/discovery_v2/phase2/opportunities_current.json --cohort PROSPECTIVE_SHADOW --output output/discovery_v2/next_champion_book.json
python -m swing_trader.discovery.operations --scan output/discovery_v2/next_eod_scan --cache output/discovery_v2/phase2/cache --previous output/discovery_v2/phase3/challengers_prospective_oct09.json --output output/discovery_v2/next_challenger_book.json
```

Güncel fundamentals enrichment istenirse Phase2 `facts` ayrı opt-in çalıştırılıp
facts-aware yeni scan oluşturulur. Eski/current facts yeni tarihe taşınmaz.
`operations.observe` challenger kayıtlarını champion kitabına append etmeyi reddeder;
entry açılışından sonra prospective backdating yapmaz; her hipotez için ayrı kimlik
ve30-seans örtüşme kümesi tutar. Gelecekte `phase2 forward` her kitabı ayrı ölçebilir.

### Aynı-seans legacy membership

`api/routers/scanner.py` mevcut `smallcap/history/{run_id}` GET sonucu ve
`smallcap_signal_runs.stats_json`: `bar_session`, `stocks_scanned`, `scanned_members`,
`universe_ranked_total`, fallback ve cut kayıtları kullanılabilir. Scan çalıştıran
POST çağrısı yapılmamalı. DB storage constructor'ı CREATE/MIGRATION yapabildiği için
salt-okunur export bunun üzerinden alınmamalı; mevcut GET veya read-only SQL export
kullanılmalı, kullanıcı/token/secret rapora eklenmemeli.

Sanitize edilmiş run listesinden bu tur eklenen daha muhafazakâr offline adapter:

```powershell
python -m swing_trader.discovery.operations.legacy --input SANITIZED_RUNS.json --run-id RUN_ID --session YYYY-MM-DD --settings output/discovery_v2/phase2/effective_patch.json --output output/discovery_v2/same_session_legacy.json
```

Sonraki `scan` komutuna `--legacy output/discovery_v2/same_session_legacy.json`
eklenir. Gün/seans, membership completeness ve fallback koşulları sağlanmazsa
UNKNOWN kalır. Bu metadata gerçek query-başına tam Finviz yanıtını kanıtlamaz;
static fallback Finviz sonucu değildir. Legacy membership yoksa challenger-added
yalnız **champion trigger'a göre ek** demektir, Finviz broad-only demek değildir.
Bu adapter ek olarak `min(ranked_total,cap)` ile seçilen sayının eşitliğini,
static-merge minimumunu ve her üyenin bar tarihini doğrular. Phase1 adapter'ın
yalnız rank telemetry varlığından yaptığı çıkarım, küçük/merged listelerde yeterli
olmayabilir. Mevcut run131 bu ek testlerden geçti:8 Ekim57 üye,9 Ekim listesi yok.
Kaynak `operations/legacy.py::adapt`; production/Phase1 adapter değiştirilmedi.

### Bu tur oluşturulan gerçek ileriye dönük kayıtlar

`operations.observe` 11 Ekim2026 **09:45:07 UTC** başlangıç timestamp'iyle,
9 Ekim tamamlanmış EOD barlarını kullanarak,12 Ekim referans açılışı gelmeden çalıştı.
Champion'ın önceki19 kaydı korunurken ayrı `challengers_prospective_oct09.json`
kitabına PREBREAK6 / RELAUNCH0 gözlem kaydedildi. Forward komutu da çalıştırıldı:
**24 PENDING,0 MATURE**. Bu kayıtlar tarihsel sonuç özeti görülmeden oluşturuldu.

| Ticker | Ref.close $ | 20g dolar hacim M$ | Uyarı |
|---|---:|---:|---|
| ANDG | 56,82 | 27,30 | cap/earnings/quality UNKNOWN |
| HWKN | 130,00 | 21,98 | cap/earnings/quality UNKNOWN |
| JOYY | 81,05 | 27,97 | cap/earnings/quality UNKNOWN |
| PRTH | 7,81 | 18,14 | cap/earnings/quality UNKNOWN |
| RCBC | 54,98 | 5,55 | cap/earnings/quality ve data_quality UNKNOWN |
| TK | 15,27 | 8,59 | cap/earnings/quality UNKNOWN |

Hepsinde9 Ekim champion trigger FALSE, legacy membership UNKNOWN, context
Weinstein stage2. Bu ekonomik koşullar/fill veya kazanç doğrulaması değildir.
Araştırma feature/observer ATR alanı **ATR14%**; Phase2'nin diğer ATR tanımlarıyla
karıştırılmamalı. Hiçbiri canlı işlem tavsiyesi değildir.

Değiştirilmeyen champion kitabının başlangıç ve son SHA256 değeri:
`2c1736a5b8f4789e59879f9a141f58a7920f2fcab167f156cd43be9c60587a5a`.

## 7. Testler, tekrar üretilebilirlik ve nihai sınıflandırma

**502 test geçti**: önceki481 + Phase3 için21 yeni test. Son çalıştırma24,14sn,
3 mevcut protobuf/joblib deprecation uyarısı; pytest async fixture ayarı uyarısı
da mevcut. Başarısız test yok. `test_backtest.py` gibi import sırasında ağ/backtest
çalıştıran eski araştırma scriptleri unit-test kapsamı diye çalıştırılmadı.

```powershell
python -m pytest tests swing_trader/discovery swing_trader/tests api/tests -q -p no:cacheprovider --basetemp output/discovery_v2/phase3/tests-handoff --junitxml output/discovery_v2/phase3/test_results_handoff.xml
```

Yeni testler: production predicate parity; gelecekteki fiyatların geçmiş seçimi
değiştirmemesi; future split ve dividend factor normalizasyonu; next-open ve maliyet;
eksik barın atlanmaması;30-seans kümeleri; bootstrap determinismi ve aynı örnekte
farkCI=0; purge pencereleri; aynı girdilerle90-seans mini-panelin iki çalıştırmada
aynı veri/detay üretmesi; ayrı challenger kitabı/dedup; geçmişe prospective kayıt
reddi; manifest bütünlüğü; static merge ve mixed-session legacy reddi.

Ölçüm komutları:

```powershell
python -m swing_trader.discovery.research inventory --preparation output/discovery_v2/phase2/full/preparation.json --cache output/discovery_v2/phase2/cache --output NEW_INVENTORY.json
python -m swing_trader.discovery.research run --preparation output/discovery_v2/phase2/full/preparation.json --cache output/discovery_v2/phase2/cache --inputs output/discovery_v2/phase2/full_final/inputs.json --inventory output/discovery_v2/phase3/inventory.json --output NEW_RUN_DIRECTORY
python -m swing_trader.discovery.research summarize --preparation output/discovery_v2/phase2/full/preparation.json --cache output/discovery_v2/phase2/cache --output NEW_RUN_DIRECTORY
python -m swing_trader.discovery.operations.audit supplement --run NEW_RUN_DIRECTORY
python -m swing_trader.discovery.operations.audit seal --run NEW_RUN_DIRECTORY
```

Her ticker için immutable panel+detay checkpoint'i, kaynak/config/preparation kimliği
ve panel hash'i tutulur. Kaynak/plan değişmişse resume reddedilir; tamamlanmış çıktı
üzerine yazılmaz. Source bundle ve manifest tekrar üretim için saklanır. Tam90-gün
araştırmayı iki kez çalıştırdığımız iddia edilmez: gerçek veride predicate parity,
panel checksum'ları ve deterministik mini-panel tekrarı ayrı doğrulamalardır.

Eklenen dosyalar:

| Dosya | İşlev |
|---|---|
| `research/protocol.py` | Ön kayıtlı iki hipotez, split/horizon/maliyet sözleşmesi |
| `research/__main__.py` | Offline inventory/run/summarize CLI |
| `research/features.py` | Geçmişe kesilmiş selection özellikleri, ayrı forward outcome'lar |
| `research/panel.py` |90-seans checkpoint'li panel, orijinal predicate kontrolleri |
| `research/analysis.py` | Olay kümeleri, gap attribution, costs, bootstrap, ablation |
| `research/test_research.py` | Deterministik araştırma testleri |
| `operations/__main__.py` | Ayrı pre-entry prospective challenger observer |
| `operations/legacy.py` | Muhafazakâr aynı-seans recorded legacy adapter |
| `operations/audit.py` | Ek tanı, aynı-tarih benchmark, source/artifact seal |
| `operations/test_operations.py` | Observer/legacy/mini-panel testleri |
| İki `__init__.py` | İzole paketler; production import zincirine eklenmedi |

### Özetleme düzeltmesinin audit izi

İlk özet yazımı sonlu olmayan bir ATR-squeeze oranında `allow_nan=False` kontrolüne
takıldı; kısmi `summary.json` kullanılmadı/silinmedi. Strateji seçimi, panel ve
predicate kuralları değiştirilmeden, dağılım fonksiyonuna sonlu değer kontrolü ve
`nonfinite_or_missing` sayısı eklendi; regresyon testi geçti. Nihai artefaktlar
`summary_final.json` / `supplement_final.json`. Örneğin tüm-uygun başlangıçların
squeeze dağılımında1/7.060 değer sonlu değildir; güçlü grupta0/3.537.

Paneli üreten orijinal kod `run90/source/`, düzeltmenin olduğu özetleme kodu
`run90/postprocess_source/` altında. Manifest10.528 dosyayı hash'ler ve kısmi
summary'yi açıkça dışlar. Bu düzeltme performans sonucu görülüp bir parametrenin
iyileştirilmesi değildir. Pandas object→bool dönüşümü için analiz çalışmasında
FutureWarning görüldü; kullanılan runtime'da hesaplama tamamlandı.
Son salt-okunur kontrol: **10.528/10.528 manifest checksum eşleşti**;
champion kitabı başlangıç hash'iyle aynı. `summary_final.json` SHA256:
`c56b5d0f900a3cba03571f36a32f5d8c0938e59489c19602d17b15b9352e73e8`.

## 8. Sekiz sorunun cevabı ve Phase 4 önceliği

1. **Kaç uygulanabilir yükseliş yakalandı?** Tam uygulanabilirlik sayısı UNKNOWN.
   Fiyat/DV/geçmiş uygun3.537 güçlü ankeranın19'unda trigger,12'sinde bilinen
   downstream kontroller açık. Bu,19 uygulanabilir trade veya executable recall değildir.
2. **En büyük kayıp nerede?** Seçilen başlangıç ankeralarında3.518 trigger-yok;
   squeeze/tepe/hacim şartları büyük yapısal ayrımı oluşturuyor. Fakat bütün ralli
   boyunca kaçırma oranı bu sayıdan çıkarılamaz. Nihai uygunluğu kanıtlama önündeki
   veri engeli tarihsel cap/float/earnings ve VIX'tir. Finansal kaybın nedensel
   dağılımı ölçülmedi; bir gate'i kaldırmak çoğunlukla negatif ek adaylar getiriyor.
3. **Yükseliş öncesi özellikler?** Tipik güçlü ankerada5g getiri−%2,55,20g tepeden
   uzaklık−%10, RVOL0,91 ve squeeze1,01. Bunlar VCE/RVOL giriş tanımı değildir;
   sonuç-koşullu betimlemedir, yeni dönüş stratejisi kanıtı değildir.
4. **Farklı setup aileleri?** Kırılım öncesi arz daralması ve impulse-base-relaunch
   ayrı ve ekonomik olarak anlaşılabilir iki hipotez olarak sınandı. Mevcut sabit
   tanımları için edge kanıtı yok; üçüncü bir kalıp sonuçlara bakarak icat edilmedi.
5. **Daha çok aday?** Prebreak evet: değerlendirmede1.016 raw/468 olay,
   champion307/250;419 yeni ticker. Relaunch hayır:19 raw/17 olay.
6. **Ekler daha iyi net forward sonuç veriyor mu?** Kanıtlanmadı. Prebreak ekleri
   −%2,25, Relaunch ekleri−%7,07; Prebreak-champion farkCI0'ı içeriyor.
7. **Retrospektif/forward ayrımı?** Bütün getiri tabloları retrospektif ve survivor-
   biased. Forward tarafta mevcut19 champion ve ayrı6 challenger kaydı var;
   challenger24 horizon PENDING. Henüz ileriye dönük doğrulanmış getiri yok.
8. **Phase 4 sırası?** Aşağıdaki ölçüm/gözetim sırası; production değişikliği değil.

| Öncelik / öneri | Sınıf | Sonraki test / başarı ve başarısızlık ölçütü |
|---|---|---|
| Ayrı snapshot, aynı maliyet/next-open, bütün kaybedenleri içeren araştırma altyapısını koru | **VERIFIED** |502 test,8.715 parity kontrolü,0 uyuşmazlık; aynı-girdi ve artifact integrity testleri başarısızsa sonuç yayınlama |
| Önce iki tarafı sansürlenmeyen executable-episode benchmark kur; EOD ankerası ile gerçekçi gecikmiş manuel entry'yi ayır | **PROMISING BUT UNPROVEN** | Önceden tanımlı episode/entry zamanları ve önceki sinyaller birlikte izlenmeli; sağ/sol sansür varsa UNKNOWN, recall diye sunulmamalı. Orta zorluk, yerel ölçüm çalışması |
| Sonra aynı-seans legacy kanıtı ve kısa-listedeki cap/earnings/float gözlem timestamp'leriyle frozen prospective kitapları izle | **VERIFIED** (kayıt mekanizması), **INSUFFICIENT DATA** (getiri iddiası) | Bilinen metadata coverage artmalı, bağımsız ve olgunlaşmış cohort oluşmalı; üyelik/entry verisi yoksa broad-only/fill üstünlüğü UNKNOWN. Düşük-orta teknik zorluk; takvim beklemek gerekir |
| Mevcut sabit Prebreak varyantını production'a taşı | **REJECTED** | Değerlendirme net negatif ve champion farkı belirsiz. Daha fazla sinyal gerekçe değildir; bu tanım için eşik taraması yapılmayacak |
| Mevcut Relaunch varyantını production'a taşı | **REJECTED**; aile hakkında genelleme **INSUFFICIENT DATA** |17 olay ön kayıtlı30-cluster minimumunun altında, gözlenen sonuçlar olumsuz. Küçük örnekten uzun dönem kesinlik üretilmez |
| Eski genel momentum continuation/tight-breakout'u yeni keşif gibi geri ekle | **REJECTED** | Eski olumsuz/kararsız sonuçlar ve yeni bağımsız kanıt yokluğu |
| RVOL üst sınırı/MA20/alt sınırı kaldırarak kazancı artır | **INSUFFICIENT DATA** | Çoklu ablation, medyan/outlier/dönem sorunları var. Önceden belirlenmiş ayrı ileri test olmadan trading kapısı değiştirilmemeli |

İleri araştırmada yükseltme kararı yalnız ortalama pozitif diye verilmemeli:
aynı-tarih comparator'a net katkı, kötü sonuçlar/tail risk, gap uygulanabilirliği,
outlier hariç dayanıklılık ve yeterli bağımsız zaman blokları birlikte aranmalı.
Bu turdan production entegrasyonu, yeni risk/exit kuralı veya canlı işlem tavsiyesi
çıkmadı. Çalıştırmalar tamamlandı; otomatik job eklenmeden duruldu.
