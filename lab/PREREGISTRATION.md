# Lab v3 — Ön Kayıt (Pre-Registration)

**Yazıldığı an: 2026-11-21, hiçbir Lab v3 sonucu görülmeden.** Veri indirme sürerken
yazıldı. Bu dosyanın SHA256'sı sonuç raporunda yazılır. Aşağıdaki kurallar, parametreler
ve başarı ölçütleri sonuç görüldükten sonra **değiştirilmez**. Değişiklik gerekirse yeni
bir hipotez numarasıyla eklenir ve "sonuç görüldükten sonra" diye etiketlenir.

## 0. Ortak ölçüm çerçevesi (bütün stratejiler)

- **Veri:** yfinance günlük OHLCV, auto_adjust=True, 2020-06-01 → 2026-11-20.
  Değerlendirme dönemi **2021-01-04 → 2026-11-20** (önceki barlar warmup).
- **Evren (havuz):** 2026-10-10 Phase-1 snapshot'ındaki NASDAQ/NYSE CANDIDATE hisseleri.
  Her gün t için: Close 7–250$, 63-seans ortalama hacim > 300K, 20-seans ortalama
  dolar hacim > 5M$, market cap (t tarihindeki hisse sayısı × Close) 300M–10B$.
  **Bilinen yanlılık:** 2026-10-10'dan önce delist olanlar yok (survivorship). Long
  stratejilerde mutlak getiri iyimser çıkar. Bu nedenle birincil ölçüt havuza göre
  **fark** getiridir (havuz da aynı yanlılığı taşır).
- **Giriş:** sinyal barı t'nin kapanışından sonra karar, **t+1 Open**. Look-ahead yok:
  t+1 Open dışında t+1 bilgisini hiçbir karar kullanmaz.
- **Maliyet:** birincil **50 bps gidiş-dönüş** (spread + kayma, small/mid-cap için
  muhafazakâr). Duyarlılık: 20 ve 100 bps.
- **Aynı bar belirsizliği:** stop önce (muhafazakâr).
- **Benchmark (havuz farkı):** her işlem için aynı giriş günündeki tüm havuz üyelerinin
  aynı giriş (t+1 Open) → aynı çıkış tarihi Close getirisinin eşit ağırlıklı ortalaması.
  Fark = işlem net getirisi − havuz brüt ortalaması − aynı maliyet.
- **Bağımsızlık:** bir hissede açık pozisyon varken aynı hisseye yeni giriş yok.

## 1. Terfi ölçütleri (hepsi birlikte sağlanmalı)

| # | Ölçüt |
|---|---|
| P1 | ≥150 işlem ve işlemler ≥30 farklı takvim ayına yayılmış |
| P2 | 50 bps net **havuz farkı** ortalaması > 0 ve tarih-kümeli bootstrap %95 CI alt sınırı > 0 |
| P3 | 2021–2026 takvim yıllarının en az 4/6'sında net havuz farkı ortalaması > 0 |
| P4 | En iyi %5 işlem çıkarılınca 50 bps net ortalama getiri > 0 |
| P5 | 100 bps maliyette net ortalama getiri ≥ 0 |
| P6 | Portföy simülasyonunda (aşağıda) maksimum drawdown < %35 |

Hiçbir strateji geçemezse sonuç "**edge bulunamadı**" olarak raporlanır; parametre
taraması yapılmaz.

## 2. Hipotezler (parametreler sabit)

### H0 — Champion (canlı ürün, birebir replika)
Canlı auto-scan akışı production fonksiyonlarıyla: Finviz emülasyonu (Q6/Q6b/Q7/Q7b) →
`scan_stock` → rejim eşiği (auto_scan min_quality 60, top_n 15, rejim tabanları) →
`rank_score` sıralaması → kapasite (BEAR 0, CAUTION 1) → duplicate/cooldown →
t+1 Open onayı (gap +5/−7, 2.5×ATR14 stop, tip tavanı) → `execution.evaluate` çıkışı.
Ayarlar: `output/discovery_v2/phase2/effective_patch.json` (canlı yamanın 2026-10-11 kopyası).

### H1 — Gap-and-Hold (haber/bilanço reaksiyonu devamı)
Ekonomik gerekçe: kurumsal bilgi şoklarına yavaş tepki (PEAD / news momentum), small
cap'te daha güçlü olduğu literatürde belgeli.
- Sinyal t: Open_t / Close_{t-1} − 1 ≥ **%5**, Volume_t ≥ **3×** önceki 50 seans ortalama,
  Close_t ≥ Open_t, (Close_t − Low_t)/(High_t − Low_t) ≥ **0.5**.
- Giriş t+1 Open; t+1 Open < Low_t ise giriş yok.
- Stop: **Low_t** (gap günü dibi). Hedef yok. Zaman çıkışı: girişten sonra **20.** seans Close.

### H2 — Trend İçinde Aşırı Satım Dönüşü (kısa vadeli mean reversion)
Ekonomik gerekçe: likidite sağlama primi / kısa vadeli reversal; uzun trend filtresiyle.
- Sinyal t: Close_t > SMA200_t, RSI(2)_t < **5** (Wilder), Close_t > SMA50_t × 0.95.
- Giriş t+1 Open. Çıkış: Close > SMA5 olan ilk günün Close'u, en geç **10.** seans Close.
- Felaket stop: giriş × **0.85** (stop-first, gap'te Open).

### H3 — 52-Hafta Zirvesi Yakınlığı + Göreli Güç (haftalık)
Ekonomik gerekçe: George & Hwang (2004) 52-hafta zirvesi çapalama etkisi.
- Her haftanın son seansı t: Close_t ≥ **0.95 ×** 252-seans en yüksek High,
  Close_t > SMA50_t, 126-seans getirisi havuzun o günkü **en üst %20'si** içinde.
  Bu adaylardan 126-seans getirisi en yüksek **10** hisse.
- Giriş t+1 Open. Stop: giriş − **2.5×ATR14**. Zaman çıkışı **20.** seans Close.

## 3. Portföy simülasyonu (P6 ve toplam getiri için)

10.000$ başlangıç, işlem başına sermayenin **%10'u** (H3'te en çok 10 eşzamanlı), nakit
yetmezse işlem atlanır, sıra: aynı gün sinyaller strateji kendi sıralamasıyla
(H0 rank_score, H1 hacim katı, H2 RSI2 küçükten, H3 126g getiri). 50 bps maliyet.
H0 ayrıca canlı boyutlandırmayla (%1.5 risk, tip tavanı) raporlanır.

## 4. Raporlama zorunlulukları

Bütün kaybedenler dahil; outlier çıkarılmış ve çıkarılmamış; yıl, rejim (SPY 200g üstü/
altı) ve maliyet kırılımları; evren yaklaşımının kayıtlı canlı taramalara göre
doğruluğu (precision/recall); H0 replikasının canlı sinyallerle birebirlik oranı.

---

# Ek A — H4 ve H5 (bilgi tabanlı swing sinyalleri)

**Yazıldığı an: 2026-11-22, H0–H3 sonuçları görüldükten SONRA, H4/H5 verisi indirilmeden
ÖNCE.** H0–H3'ün başarısız olması bu iki hipotezin seçilme gerekçesidir (fiyat/hacim
kalıpları yerine fiyat-dışı bilgi). Bölüm 0 (ortak çerçeve), 1 (terfi ölçütleri P1–P6),
3 (portföy) ve 4 (raporlama) aynen geçerlidir. Parametreler sonradan değiştirilmez.

### H4 — İçeriden Alım Kümesi (SEC Form 4)
Ekonomik gerekçe: yöneticilerin kendi parasıyla açık piyasa alımı özel bilgi taşır;
etki small cap'te en güçlüdür (Lakonishok & Lee 2001; Cohen, Malloy & Pomorski 2012).
- Veri: SEC DERA "Insider Transactions Data Sets" (Form 3/4/5), 2020Q3 → yayımlanmış son çeyrek.
- Alım: Form 4, non-derivative, TRANS_CODE = `P`, edinme (`A`), adet > 0, fiyat > 0.
  Sahip rolü Director veya Officer içermeli (yalnız %10 sahibi sayılmaz).
- Küme olayı: dosyalama tarihi F'de, ihraççı sembolü için (F−30 gün, F] aralığında
  dosyalanmış uygun alımlarda **≥2 farklı** sahip ve toplam alım değeri **≥100.000$**.
  Aynı sembolde önceki olaydan sonra **60 takvim günü** içinde yeni olay yok.
- Bilgi zamanı: Form 4 en geç F günü 22:00 ET'de kamuya açıktır. Sinyal barı t = F'ye
  eşit veya önceki son seans (havuz uygunluğu ve ATR burada ölçülür). Giriş: F'den
  **sonraki ilk seansın Open'ı**.
- Stop: giriş − **2.5×ATR14** (t itibarıyla). Hedef yok. Zaman çıkışı **20.** seans Close.
- Aynı gün sıralama: toplam alım değeri büyükten küçüğe.

### H5 — Bilanço Sürprizi + Pozitif Tepki (PEAD)
Ekonomik gerekçe: yatırımcıların kazanç bilgisine yavaş tepkisi (Bernard & Thomas 1989);
sürpriz ve piyasa tepkisi birlikte en güçlü (Chan, Jegadeesh & Lakonishok 1996).
- Veri: yfinance `get_earnings_dates(limit=40)` — açıklama zaman damgası, EPS tahmini,
  açıklanan EPS, Surprise(%). Havuz (Bölüm 0) hisseleri.
- Tepki seansı R: açıklama saati ET < 09:30 ise aynı gün, aksi halde sonraki ilk seans.
- Sinyal: Surprise(%) ≥ **+10**, Reported EPS > EPS Estimate, |EPS Estimate| ≥ **0.02**,
  Close_R > Close_{R−1}, R'de hisse havuzda.
- Giriş: R'den sonraki seansın Open'ı. Stop **2.5×ATR14** (R itibarıyla), hedef yok,
  zaman çıkışı **20.** seans Close. Aynı gün sıralama: Surprise(%) büyükten küçüğe.
- Bilinen risk: yfinance tahmin/EPS değerleri sonradan revize edilmiş olabilir.

---

# Ek B — H0F: float bileşeni çalışırken canlı ürün (düzeltme kararı)

**Yazıldığı an: 2026-11-22, H0F sonucu görülmeden.** Amaç terfi değil, production float
hatasının hangi yönde düzeltileceğine karar vermek.

- **H0F:** H0 ile birebir aynı; tek fark `scan_stock`'a verilen `floatShares`. H0'da canlı
  paritesi için 0 (Finviz Overview'da Float yok). H0F'de yaklaşık zaman-noktası float:
  `hisse_sayısı_t × (floatShares_bugün / sharesOutstanding_bugün)`, hisse sayısını aşamaz.
  Eşikler, ayarlar, seçim, onay ve çıkış aynı.
- **Karar kuralı:** H0F'nin 50 bps net işlem başı ortalaması H0'dan **büyük** VE %10/işlem
  portföy toplam getirisi H0'dan **büyük** ise production'a gerçek float verisi bağlanır.
  Aksi halde float skora girmez; kod bunu açıkça yapar (sabit "bilinmiyor" puanı yerine
  bileşen bilinçli olarak devre dışı, seçim bugünküyle aynı kalır).
- **Bilinen yanlılık:** float oranı bugünden geriye uygulanıyor (zaman-noktası değil).

---

# Ek C — H4 ve H5 gölge modu (ileri test) ve terfi kapısı

**Yazıldığı an: 2026-11-22, hiçbir canlı gölge sonucu yokken.** H4 ve H5, Ek A'daki kurallarla
(parametreler aynen) production'da **gölge modda** çalışır: gerçek para yok, ana ürünün
paper işlemlerine ve kapasitesine dokunmaz, ayrı defterde (shadow_*) tutulur.

- **Kurallar:** Ek A ile birebir (H4: ≥2 farklı Director/Officer, (F−30, F] içinde ≥100K$ açık
  piyasa alımı, 60 gün tekrar yok; H5: Surprise ≥ +10, Reported > Estimate, |Estimate| ≥ 0.02,
  Close_R > Close_{R−1}). Havuz Bölüm 0 (fiyat 7–250, avgvol63 > 300K, dv20 > 5M$, cap
  300M–10B$, NASDAQ/NYSE hisse senedi). Giriş sinyal barından sonraki ilk seansın Open'ı,
  stop giriş − 2.5×ATR14 (sinyal barı itibarıyla), 20. seans kapanışında zaman çıkışı,
  stop önce, gap'te Open'dan dolum, hisse başına tek pozisyon.
- **Karar zamanı:** her sinyal `decided_at` ile kaydedilir. Karar girişin açılışından sonra
  verildiyse `late_decision` işaretlenir (geç cron). Bu işlemler raporda ayrıca sayılır.
- **H4 ısınma:** gözlem başlangıcından sonraki ilk 30 gün içinde oluşan kümeler (pencere
  gözlem öncesine taşıyor) `warmup` işaretlenir ve terfi değerlendirmesine girmez.
- **Maliyet:** 50 bps gidiş-dönüş (birincil). Kıyas: aynı giriş Open → çıkış Close IWM getirisi.
- **Terfi kapısı (her strateji ayrı, otomatik terfi YOK, karar insanda):**
  - F1: canlı başlangıçtan ≥ 6 ay geçti VE kapanmış (warmup/late hariç) işlem sayısı
    H5 ≥ 150, H4 ≥ 50.
  - F2: 50 bps net ortalama getiri > 0 VE giriş-günü kümeli bootstrap %95 CI alt sınırı > 0.
  - F3: %10/işlem, nakit sınırlı portföy toplam getirisi > aynı dönemde IWM al-tut.
  - F4: portföy maksimum drawdown < %35.
  - Durum: F1 sağlanmadan **TOPLANIYOR**. F1 sağlanınca F2–F4 hepsi geçerse **TERFİ ADAYI**;
    net ortalama ≤ 0 ise **KALDI** (gölge kapatılır); aksi halde **BELİRSİZ** (12. aya kadar
    devam, 12. ayda hâlâ TERFİ ADAYI değilse KALDI).

---

# Ek D — Bilanço kapısı düzeltmesi (D1) ve büyük hisse deneyi H4L / H5L (D2)

**Yazıldığı an: 2026-11-22, D1 ve D2'nin hiçbir sonucu görülmeden.** Bilinen tek şey veri
kapsamı: lab OHLCV'sinde bugün ≥10B$ olan 841 hisse, `earnings_full`'da bunların 796'sı var.

## D1 — Bilanço kapısı: fail-open düzeltmesinin yönü

Bulgu (Ek B raporu): `filters.check_earnings` yfinance boş/hatalı döndüğünde "geç" diyor;
canlı SAIC bilanço gününde açıldı. Kural (sinyal günü veya sonraki 0..N gün içinde bilanço
varsa ret) kodda veri yokken uygulanmıyor.

- **H0** (bugünkü): lab bilanço dosyası yok/boş → sinyal geçer (canlıyla aynı fail-open).
- **H0EC** (düzeltilmiş): lab bilanço verisi yok → sinyal reddedilir (fail-closed).
- **H0E0** (çeldirici kontrolü): bilanço kapısı tamamen kapalı. Kapının kendisi para
  kazandırıyor mu sorusunu cevaplar; kapı zararlıysa onu "düzeltmek" zararı büyütür.
- Ölçüt (Ek B ile aynı): 50 bps net işlem başı ortalama ve %10/işlem portföy toplam getirisi.
- **Karar kuralı:**
  - H0E0, H0'dan **hem** net/işlem **hem** portföy toplamında büyükse → kapı zararlı kabul
    edilir; fail-closed production'a **uygulanmaz**, kapının kaldırılması kullanıcıya ayrı
    karar olarak sunulur.
  - Aksi halde → production düzeltmesi: ikinci bağımsız kaynak (tarama başına tek istekle
    bilanço takvimi) eklenir; iki kaynak da bilinmiyorsa sinyal reddedilir (fail-closed) ve
    sebebi görünür şekilde kaydedilir; backtest aynı kuralı uygular (veri yok → ret).
  - H0EC raporlanır; az sayıda sinyali etkilediği için karar ölçütü değildir.
- **Bilinen yanlılık:** lab'daki "veri yok" Kasım 2026 indirmesine aittir, canlıdaki o
  günkü hata/boşluk değildir.

## D2 — H4L / H5L: aynı bilgi sinyalleri, büyük ve likit hisselerde

Ekonomik gerekçe: Ek A'da H4/H5 havuzu brüt yendi (+0.19 / +0.27 puan) ama 50 bps
small/mid-cap maliyetini aşamadı. Büyük hisselerde işlem maliyeti çok daha düşüktür. Karşı
argüman (önceden yazılır): içeriden alım ve PEAD etkisi literatürde büyük hisselerde
**daha zayıftır**; düşük maliyet, zayıf edge'i telafi etmeyebilir.

- **Kurallar:** H4 ve H5, Ek A ile birebir; tek fark havuz ve maliyet.
- **Havuz L:** Bölüm 0'daki aynı hisse kümesi; her gün t için Close ≥ 7$ (üst fiyat sınırı
  yok), 63-seans ortalama hacim > 300K, 20-seans ortalama dolar hacim > 5M$, market cap
  (t tarihindeki hisse sayısı × Close) **≥ 10B$** (üst sınır yok).
- **Veri:** H4 aynı SEC DERA veri seti. H5 aynı `earnings_full` (eksik büyük hisseler aynı
  betikle bir kez indirilir; yine eksik kalanlar dışlanır ve sayısı raporlanır).
- **Maliyet:** birincil **10 bps gidiş-dönüş** (≥10B$ hisselerde kote spread tipik 1–5 bps;
  açılış emri kayması dahil muhafazakâr). Duyarlılık: 20 ve 50 bps raporlanır.
- **Benchmark:** havuz L farkı (Bölüm 0 tanımı, havuz L ile); endeks **SPY**.
- **Terfi ölçütleri:** P1–P6, şu farklarla: P2 ve P4 10 bps net, P5 **20 bps**'te net
  ortalama ≥ 0 (birincil maliyetin 2 katı, Bölüm 1'deki 50→100 oranı gibi). Portföy (P6)
  10 bps maliyetle. Ek olarak (kapı değil, raporlanır): portföy toplamı vs SPY al-tut.
- **Sonuç:** Geçen strateji Ek C düzeniyle gölge moda eklenir (maliyet 10 bps, kıyas SPY);
  geçemezse "edge bulunamadı" raporlanır, parametre taraması yapılmaz.
- **Bilinen yanlılıklar:** survivorship (2023'te batan SIVB/FRC/SBNY ve satın alınan büyük
  hisseler yok — long için iyimser); `facts` olmayan büyük hisselerde hisse sayısı bugünkü
  değerle sabit (geri alım yapanlarda geçmiş cap biraz düşük ölçülür, yalnız 10B sınırında
  etkili); yfinance EPS tahminleri sonradan revize edilmiş olabilir.

### D1 notu — sonuç KISMEN görüldükten sonra (2026-11-22, etiketli sapma)

D1 varyantları ilk kez koşulunca (yalnız log özetindeki brüt ortalamalar görüldü: H0 345 işlem
+0.04%, H0EC 243 işlem +0.31%, H0E0 356 işlem +0.04%; net/portföy değerlendirmesi yapılmadı)
H0 sinyallerinin **168/539'unda** lab bilanço verisinin boş olduğu görüldü. Kök neden: `output/
lab/earnings` dosyalarının 130'u indirme sırasında Yahoo kısıtlamasıyla boş yazılmış ("No
earnings dates found"); aynı hisselerin hepsi `earnings_full`'da aynı fonksiyondan
(`get_earnings_dates(limit=40)`) dolu. Bu bir veri hatasıdır, kural değişikliği değildir.

- **Onarım:** `lab.common.earnings_dates` boş/yoksa `earnings_full` zaman damgalarına düşer.
  Onarım öncesi iki kaynağın ortak hisselerde aynı tarihleri verdiği doğrulanır.
- **Birincil D1 değerlendirmesi onarılmış veriyle** yapılır; onarımsız sonuçlar da raporlanır.
- **Karar kuralı aynen geçerli.** H0 taban çizgisi (Ek B/Lab v3'teki −0.46%) onarılmış veriyle
  yeniden raporlanır; fark olursa açıkça yazılır.

---

# Ek F — H6 "Sade VCE": canlı ürünün sadeleştirilmiş rakibi (gölge ileri test)

**Yazıldığı an: 2026-11-22, H6'nın hiçbir sonucu görülmeden.** Gerekçe Ek E tanısıdır (sonuç
görüldükten SONRA yapılmış bir tanı): canlı ürünün edge'i evren + tetikte; sonraki kapılar,
kalite skoru, rejim eşiği ve çıkış mekanizması ölçülebilir değer katmıyor; RVOL yolu sıfır.
Bu yüzden H6 geçmiş veriyle KANITLANAMAZ (aynı veriden türetildi); tek geçerli test görülmemiş
gelecektir.

**Tasarım ilkesi (işarete göre seçim değil):** canlı ürünün en küçük çalışan çekirdeği —
evren + sert filtreler + bilanço kapısı + VCE tetiği + sabit tutuş. Tetikten sonraki HER katman
çıkarılır (doğru yönü gösterenler dahil: swing onayı, gap filtresi), RVOL yolu çıkarılır.

- **Sinyal (karar t+1 açılışından önce):** t = son tamamlanmış seans. Evren = canlı ürünün
  evreni (`SmallCapEngine.get_small_cap_universe`, canlı ayarlar). Barlar = canlı tarama ile aynı
  (`DataFetcher.fetch_multiple_stocks_batch(period="6mo")`). Son barı t olan her hisse için:
  canlı sert filtreler (fiyat / cap / dolar hacim) VE canlı bilanço kapısı (iki kaynak,
  bilinmiyorsa ret) VE canlı VCE tetiği (`check_all_triggers` → `vce_breakout.passed`).
- **Yok:** RVOL thrust, swing onayı, RSI kapısı, Weinstein, kalite skoru/eşiği, rejim eşiği,
  top N, kapasite, gap filtresi, stop, hedef, trail.
- **Giriş:** t'den sonraki ilk seansın Open'ı. **Çıkış:** giriş günü 1. seans sayılarak
  **8. seansın Close'u** (canlı ortanca tutuş). Eksik barda bekle. Hisse başına tek pozisyon.
- **Aynı gün sıralama (portföy):** 20-seans ortalama dolar hacim büyükten küçüğe.
- **Maliyet:** 50 bps gidiş-dönüş (birincil, H0 ile kıyaslanabilir). 20 bps de raporlanır.
- **Lab sağlık kontrolü (karar DEĞİL, etiketli in-sample):** H6 aynı replika verisinde koşulur.
  Brüt havuz farkı ≤ 0 ise gölgeye alınmaz (zaten geçmişte bile değer yoksa ileri test anlamsız).
- **Gölge terfi kapısı (Ek C düzeni):** F1 ≥ 6 ay VE ≥ 100 kapanmış (geç karar hariç) işlem;
  F2 50 bps net ortalama > 0 VE CI alt sınırı > 0; F3 %10/işlem portföy > IWM al-tut; F4 max
  DD < %35. Ayrıca aynı dönemde canlı ürünün kendi paper işlemleri (aynı 50 bps maliyetle)
  yan yana raporlanır. Otomatik terfi yok.
- **Parametreler** H4/H5'ten ayrı bir hash ile kilitlenir; H4/H5 kuralları ve hash'i değişmez.

---

# Ek G — Durgun gün statik liste birleşimi (H0S)

**Yazıldığı an: 2026-11-22, H0S'nin hiçbir sonucu görülmeden.** Canlı evren, Finviz 30'dan az
hisse döndürdüğünde (`min_finviz_tickers_skip_static_merge=30`) 234 hisselik statik listeyle
birleştiriliyor (tavan 300). 2021–2026'da bu günlerin oranı %14 (201 gün, Finviz emülasyonu).
Lab replikası (H0) bu birleşimi içermiyordu.

- **H0S:** H0 ile birebir aynı; tek fark durgun günlerde evren = Finviz üyeleri + statik liste
  (canlı sırayla, tekilleştirilmiş, 300 tavan). Statik liste bugünkü koddaki liste.
- **Ölçüt:** 50 bps net işlem başı ortalama ve %10/işlem portföy toplamı (onarılmış bilanço verisi).
- **Karar kuralı:** H0S, H0'dan **hem** net/işlem **hem** portföy toplamında büyükse birleşim kalır.
  Aksi halde birleşim kaldırılır: durgun günde yalnız Finviz'in döndürdüğü hisseler taranır
  (statik liste yalnız kullanıcı Finviz'i kapatırsa kullanılır).
- **Ayrıca raporlanır:** statik liste hisselerinden gelen sinyal sayısı ve 8 seans havuz farkı.
- **Bilinen yanlılık:** statik listenin 32 hissesi lab verisinde yok (delist/ad değişikliği) →
  dışarıda; liste geçmişte farklı olabilirdi (bugünkü liste geriye uygulanıyor).

---

# Ek H — Az işlenen kamu verisi: SEC başvuruları ve FINRA açığa satış hacmi (H7, H8, H10)

**Yazıldığı an: 2026-11-22, hiçbir veri indirilmeden ve sonuç görülmeden.** Gerekçe: grafik
sinyallerinin avantajı sıfır (Ek E). Yasal bilgi avantajı ancak herkese açık ama az işlenen
veriden gelebilir. Kaynaklar: SEC EDGAR çeyreklik form indeksi (`full-index/YYYY/QTRn/form.gz`),
SEC `company_tickers.json` (CIK→ticker, bugünkü eşleme — survivorship ile tutarlı), FINRA Reg SHO
günlük konsolide açığa satış dosyaları (`CNMSshvolYYYYMMDD.txt`). Ortak çerçeve Bölüm 0 + Ek D1
onarılmış bilanço verisi.

## H7 — Aktivist giriş (ilk Schedule 13D)
Ekonomik gerekçe: %5+ pay alıp şirkete müdahale niyetini açıklayan yatırımcı (Brav, Jiang, Partnoy
& Thomas 2008); etki küçük hisselerde güçlü.
- Olay: form tipi `SC 13D` veya `SCHEDULE 13D` (değişiklikler `/A` hariç), dosyalama günü F.
  Hedef şirket = indeks satırlarından CIK'i lab evrenindeki bir ticker'a eşlenen; iki taraf da
  eşlenirse başvuru başlığındaki SUBJECT COMPANY.
- Sinyal barı t = F'ye eşit veya önceki son seans; t'de havuz üyesi (Bölüm 0). Giriş F'den sonraki
  ilk seansın Open'ı. Stop giriş − 2.5×ATR14 (t), 20. seans kapanışında zaman çıkışı (H4/H5 ile
  aynı, ayar yok). Aynı hissede önceki olaydan sonra 60 gün yeni olay yok. Sıralama: 20 seans dolar
  hacim (büyükten küçüğe). Terfi: Bölüm 1 P1–P6 (50 bps); geçerse Ek C düzeninde gölgeye alınır.

## H8 — Seyreltme filtresi (hisse ihracı / satış kaydı)
Ekonomik gerekçe: küçük şirketler yükselişlerde yeni hisse satar (ATM, halka arz sonrası satış);
arz baskısı yükselişi keser. Grafik sinyalleri bunu görmez.
- Form kümesi D: `S-1, S-3, S-3ASR, F-1, F-3, F-3ASR, 424B1, 424B2, 424B3, 424B4, 424B5, 424B7`
  (ilk başvurular; `/A` hariç), ihraççının CIK'iyle. Sinyal günü t için [t−30 takvim günü, t]
  aralığında D'den başvuru varsa sinyal elenir.
- **H0D** = H0 − elenen sinyaller (bilanço kapısı gibi seçimden önce). **H6D** = H6 adaylarına aynı.

## H10 — Yoğun açığa satış filtresi
Ekonomik gerekçe: açığa satanlar ortalamada bilgili yatırımcılardır; yoğun açığa satılan hisseler
sonraki günlerde zayıf kalır (Boehmer, Jones & Zhang 2008).
- Oran_t = ShortVolume / TotalVolume (FINRA CNMS, t günü kapanıştan sonra yayımlanır → t kararında
  kullanılabilir). SVR5 = son 5 seansın (t dahil) ortalaması. Sinyal, t günü havuz üyeleri
  (FINRA verisi olanlar) arasında SVR5'in **en üst %20'sindeyse** elenir.
- **H0Q** = H0 − elenen sinyaller; **H6Q** = H6 adaylarına aynı.
- Bilinen sınırlılık: FINRA TotalVolume konsolide hacim değildir; piyasa yapıcı açığa satışı oranı
  şişirir (gürültülü ölçü).

## Filtre karar kuralı (H8 ve H10 için ayrı ayrı)
Bir filtre production'a (canlı tarama kapısı olarak) yalnız ikisi birlikte sağlanırsa girer:
- (a) H0'ya uygulanınca 50 bps net/işlem VE %10/işlem portföy toplamı ikisi de H0'dan iyi;
- (b) filtrenin H0 ∪ H6 aday sinyallerinden (tekilleştirilmiş) elediklerinin 8 seans havuz farkı
  ortalaması < 0 VE gün-kümeli %95 CI üst sınırı < 0 (elenenler istatistiksel olarak kötü).
H6'nın kuralları Ek F'de dondurulmuştur; filtre H6 gölge koşusuna EKLENMEZ (H6D/H6Q yalnız rapor).
Üç hipotez birlikte test ediliyor; tek bir sınırda geçiş, production'da canlı izlemeyle doğrulanır.

---

# Ek J — H6R "Profesyonel VCE": H6 + standart risk kuralları

**Yazıldığı an: 2026-11-22, H6R'nin hiçbir sonucu görülmeden.** H6 (Ek F) işlem başına canlı üründen
iyi, portföyde yıkıcıydı (max DD −76%): piyasa freni ve stop yoktu. Eklenen iki kural veriden
türetilmedi; her profesyonel kırılım stratejisinin standart risk katmanı ve canlı üründe zaten var.

- **Sinyal:** H6 ile birebir (canlı evren + sert filtreler + bilanço kapısı + canlı VCE).
- **Piyasa freni (canlıyla aynı, `auto_entry_capacity`):** sinyal günü rejim BEAR → yeni giriş yok;
  CAUTION → aynı anda en çok `caution_max_concurrent` (1) açık pozisyon; BULL → sınır yok. Rejim =
  canlı `regime_from_spy_close` (1 yıllık SPY + VIX), replika rejim serisi.
- **Çıkış:** giriş − 2.5×ATR14 (sinyal günü) felaket stop'u (stop önce, gap'te Open) VEYA giriş günü
  1. seans sayılarak 8. seans kapanışı. Hedef/trail yok.
- **Sıralama / tek pozisyon:** H6 ile aynı (20 seans dolar hacim; hisse başına tek pozisyon).
- **Değerlendirme:** Bölüm 1 P1–P6, 50 bps; portföy %10/işlem nakit sınırlı (kaldıraç yok).
  20 bps ayrıca raporlanır. Kıyas: H0 (−0.24%/işlem, −11.8%) ve H6.
- **Karar:** (1) P1–P6 hepsi geçerse → gölgeye girer VE canlı paper ürünün H6R kurallarına
  geçirilmesi önerilir (eski kurallar kıyas olarak izlenir). (2) Geçmez ama H0'dan hem net/işlem hem
  portföyde iyiyse → yalnız gölge ileri test. (3) Aksi → reddedilir.
- Etiket: VCE ve 8 seans tutuş Ek E tanısından geliyor → in-sample sonuç iyimser kabul edilir;
  nihai karar ileri veride.

### Ek J2 — H6S: piyasa freni küçük hisse endeksinden (SONUÇ GÖRÜLDÜKTEN SONRA, etiketli)

H6R'nin max düşüşü (−51%) 2021-01 → 2023-05 aralığında: aynı dönemde SPY +13.6%, IWM −15.2%;
410 işlemin 402'si SPY tabanlı rejimin "BULL" dediği günlerde açıldı. Kural a priori doğru (küçük
hisse alıyorsan kendi piyasana bak) ama fikir bu sonuç görüldükten sonra geldi → in-sample sonuç
iyimser sayılır, karar ileri veride.

- **H6S:** H6R ile birebir; tek fark rejim = canlı `regime_from_spy_close` fonksiyonu **IWM**
  kapanışlarına (1 yıl) + aynı VIX ile uygulanır.
- **Karar:** Ek J kuralıyla aynı. Ayrıca aynı değişiklik (IWM rejimi) canlı replikaya (H0) uygulanıp
  raporlanır; canlı ürüne geçiş ancak ileri veride doğrulanırsa.

### Ek J3 — Kullanıcı kararı: canlı paper ürün H6S'ye geçirildi (2026-11-22)

Ek J2'deki "canlı ürüne geçiş ancak ileri veride doğrulanırsa" kuralının **bilinçli istisnası**,
kullanıcının açık kararıyla (paper trading; gerçek para yok). Ölçüm bütünlüğü korunur:
otomatik girişin ana stratejisi H6S (aynı kurallar, aynı kod: `live_vce_hits`, IWM rejimi,
2.5×ATR stop, 8 seans, pozisyon başına hesabın %10'u, nakit sınırı); eski kurallar ayrı sanal
hesapla kontrol grubu olarak açmaya devam eder (`paper_trades.strategy`). Gölge H6S kaydı ve F1–F4
kapısı değişmez; nihai değerlendirme ileri veride.

---

# Ek L — H6S risk kuralları: sektör sınırı (L1) ve kayıp serisi freni (L2)

**Yazıldığı an: 2026-11-22, hiçbir sonuç görülmeden.** Taban: H6S lab işlemleri (Ek J2) ve %10/işlem,
nakit sınırlı portföy (50 bps). Her kural AYRI ölçülür; tek parametre seti, tarama yok.

- **L1 — Sektör sınırı:** aynı sektörde aynı anda en çok **2** açık/bekleyen H6S pozisyonu. Karar anında
  (sinyal günü) sayılır; sınır doluysa sinyal atlanır. Sektör: lab'da `info_frame().sector`, canlıda
  Finviz sektörü. Sektörü bilinmeyen hisse sınırlanmaz.
- **L2 — Kayıp serisi freni:** giriş gününden ÖNCE kapanmış son **3** H6S işleminin hepsi 50 bps net
  zararlıysa yeni pozisyon **yarım boyut** (hesabın %5'i); ilk kârlı kapanıştan sonra tam boyuta dönülür.
- **Karar kuralı (her kural için):** portföy toplam getirisi ≥ taban VE max düşüş ≤ taban (ikisi birden;
  biri kötüleşirse eklenmez). İkisi de geçerse birleşimi raporlanır; birleşim aynı kurala tabidir.
- Geçen kural canlı H6S hesabına eklenir; gölge H6S (Ek J2, donmuş) değişmez ve kuralın kontrolü olur.

# Ek M — Kâğıttan Midas'a: uygulanabilir çıkış, boyutlama, gece boşluğu, boş nakit

**Yazıldığı an: 2026-11-22, hiçbir sonuç görülmeden.** Taban: canlı ürün = H6S + L2 freni, Ek L portföyü
(%10 bileşik, nakit sınırlı, 50 bps). Tek parametre seti, tarama yok.

**Gün içi sıralama düzeltmesi (gerçeklik şartı, sonuca bakılmadan):** Ek L portföyü giriş günü boyutunu o
günün KAPANIŞ değerlemesiyle hesaplıyor ve o günün kapanışında çıkan işlemin nakdini/sonucunu aynı günün
AÇILIŞINDAKİ girişe (ve frene) kullandırıyordu — küçük ileri bakış. Ek M portföyü günü sırayla işler:
açılışta gerçekleşen çıkışlar (açılıştan dolan stop, M1 açılış çıkışı) → açılış girişleri (boyut önceki
kapanış değerlemesiyle) → gün içi/kapanış çıkışları → kapanış değerlemesi. Taban bu düzeltmeyle yeniden
hesaplanır; eski değer de yanında raporlanır. Tüm M kararları düzeltilmiş tabana göredir.

- **M1 — Süre çıkışı 9. seans açılışında** (8. seans kapanışı yerine). Stop çıkışları aynı; stop 9. seans
  açılışına kadar geçerli (açılış stopun altındaysa zaten açılıştan çıkılır). Gerekçe: NYSE kapanışı
  Türkiye saatiyle 24:00 (yazın 23:00), açılış 17:30 (yazın 16:30); alış zaten açılışta → tüm emirler tek
  saatte. **Karar:** portföy toplamı ≥ taban − 2 puan VE max düşüş taban − 1 puandan kötü değil VE işlem
  başı net ≥ taban − 0.05 puan → benimse. Tolerans önceden ilan edildi: gece yarısı emri kaçırmanın
  beklenen maliyeti bu farktan büyük.
- **M2 — Eşit riskli boyut:** pozisyon tutarı = özsermaye × r ÷ (2.5 × ATR14 ÷ sinyal kapanışı), en çok
  %20. r, aday işlemlerin ortalama tutarı %10 olacak şekilde YALNIZ ATR'den çözülür (getiri kullanılmaz).
  Fren r'yi yarıya indirir. **Karar:** toplam ≥ taban VE max düşüş ≤ taban VE iki yarının her birinde
  (2021–2023 / 2024–son, her biri 10.000$'dan) toplam ≥ tabanın o yarıdaki toplamı.
- **M3 — Gece boşluğu (tanımlayıcı):** sinyal günü kapanışında girmek (aynı çıkış) ertesi açılışa göre
  işlem başı ne kazandırırdı? Kurmaya değer eşiği (kapanıştan önce tahmini tarama + Türkiye saatiyle
  ~23:45 emir gerektirir): işlem başı iyileşme ≥ +0.30 puan VE iki yarıda da pozitif. Değilse kapanır.
- **M4 — Boş nakit (tanımlayıcı):** taban portföyün ortalama boş nakit oranı ve bu nakdin 3 aylık hazine
  bonosu getirisiyle (^IRX, günlük tahakkuk) ekleyeceği toplam. Karar kuralı yok; bilgi olarak raporlanır.
- **Gerçeklik şartı (canlı, sonuca bakılmadan):** canlı H6S hesabı lab gibi bileşik olur: hesap =
  başlangıç + kapanmış H6S işlemlerinin maliyet sonrası kâr/zararı; pozisyon boyutu ve nakit sınırı
  bundan hesaplanır (şu an sabit başlangıç tutarı kullanılıyor — gerçek Midas hesabı böyle davranmaz).

# Ek N — 2024–26 açığı: teşhis (N0) ve çekirdek IWM (N1/N2)

**Yazıldığı an: 2026-11-22. N0 teşhisi GÖRÜLDÜ (etiketli); N1/N2 portföy sonuçları görülmeden yazıldı.**
Taban: canlı ürün H6S + L2 freni, Ek M portföyü (doğru gün içi sıra, %10 bileşik, nakit sınırlı),
birincil maliyet 50 bps gidiş-dönüş + Midas 1.5$/emir, 10.000$. Duyarlılık: 20 bps + 1.5$ (canlı ayar).

**N0 — teşhis (görüldü, 2024-01-04 → 2026-11-17, 50 bps + 1.5$):** portföy −8.5% vs IWM +43.0%.
Maliyetsiz +22.7% → maliyet/ücret ~31 puan. Ortalama maruziyet %33.6 (nakit %66). BULL rejimindeki 392
işlem brüt −0.08% vs aynı pencerede IWM 0.00 (seçim edge'i yok); işlem başı brüt +0.71% < maliyet
~0.80%. IWM, CAUTION/BEAR günlerinde (%40.5) bileşik +27% yaptı; IWM tabanlı rejim IWM'i
zamanlayamıyor (2021–23'te de BEAR günleri +11.2%, BULL günleri −8.6%). Bu yüzden sinyal kuralları
2024–26'ya göre DEĞİŞTİRİLMEZ (eğri uydurma); düzeltme adayı yalnız yapısal: boş nakdin maruziyeti.

- **N1 — Çekirdek IWM, hep açık.** Boş nakit IWM'de durur. Her gün AÇILIŞTA (açılış çıkışlarından sonra,
  girişlerden önce): o günün girişleri için nakit yetmiyorsa IWM satılır (gereken + 1 dilim, dilim =
  önceki kapanış özsermayesinin %10'u; IWM yetmiyorsa taban gibi giriş atlanır). Girişlerden sonra nakit
  3 dilimi aşıyorsa fazlası IWM'e (nakit 1 dilime iner). Kapanış çıkışlarının nakdi ertesi açılışta
  değerlendirilir. Her IWM emri 1.5$ + 1 bps/kenar. Boyut/fren özsermayeyi (nakit + IWM + pozisyonlar)
  kullanır. Sinyaller, çıkışlar, fren taban ile aynı.
- **N2 — Çekirdek yalnız BEAR dışında.** N1 ile aynı; t kapanışında rejim BEAR ise t+1 açılışında tüm
  IWM satılır ve BEAR bitene kadar alınmaz (N0'a göre muhtemelen N1'den kötü — bilerek kayıtta).
- **N3 — tanımlayıcı:** IWM al-tut (tek emir) ve aynı dönemde N1/N2'nin IWM'e göre farkı; uydunun
  (hisse işlemlerinin) çekirdeğe katkısı = N1 − (yalnız çekirdek, işlemsiz).

**Karar (birincil maliyette; N1 ve N2 için ayrı; ikisi de geçerse toplamı yüksek olan):**
(a) tüm dönem toplamı ≥ taban + 5 puan; (b) iki yarının her biri (2021–23 / 2024–son, her biri
10.000$'dan) ≥ tabanın o yarısı − 3 puan; (c) tüm dönem max düşüş > −35% (§1 P6). Duyarlılıkta (20 bps)
(a) da sağlanmalı. Geçmezse canlı ürün değişmez; açık nedeniyle birlikte raporlanır.

**Ek M/N notu (2026-11-22, kullanıcı ilkesi):** faiz kullanılmaz. M4'ün faiz kısmı ve ürün içi faiz
hatırlatması kaldırıldı; boş nakit yalnız swing işlemleriyle değerlendirilecek. (Yukarıdaki geçmiş
ön kayıt metni denetim izi için değiştirilmedi.)

# Ek O — Swing tarafı: kazananı koştur (O1), büyük pozisyon (O2), CAUTION sınırı (O3)

**Yazıldığı an: 2026-11-22, O1–O3'ün hiçbir sonucu görülmeden.** Kullanıcı ilkesi: kazanç yalnız swing
işlemlerinden (faiz yok, endeks çekirdeği yok). Ek N teşhisi: hesap ~%66–74 nakitte, maliyet işlem başı
brüt kazancı yiyor, kazanç kuyrukta (en iyi %5 çıkınca ortalama negatif). İkinci sinyal ailesi (H1–H5)
ölçülmez: hepsi 50 bps sonrası negatif (Lab v3 §2, Ek A) — boşluğu zararla doldurmak olur.

Taban: canlı ürün H6S + L2 freni, Ek M portföyü (doğru gün içi sıra), 10.000$, birincil maliyet 50 bps +
Midas 1.5$/emir; duyarlılık 20 bps + 1.5$. Her varyant TEK parametre setiyle, tarama yok.

- **O1 — Kazananı koştur.** 8. seans kapanışı giriş açılışının ÜSTÜNDEYSE çıkılmaz; 9. seanstan itibaren
  iz süren stop: stop = max(önceki stop, giriş sonrası en yüksek kapanış − 2.5 × sinyal günü ATR14),
  her seans önceki kapanışa göre güncellenir; gün içi Low ≤ stop → min(Open, stop) dolum. En çok 40
  seans (40. kapanışta çık). 8. kapanış giriş açılışının altında/eşitse taban gibi çıkılır. Gerekçe: edge
  kuyrukta; 8 seans kazananları kesiyor; para kazananda daha uzun çalışır, ek işlem/maliyet yok.
- **O2 — Pozisyon %15** (fren yarısı %7.5). Gerekçe: nakit çoğu zaman boşta; sabit 1.5$ ücret 1.500$'da
  gidiş-dönüş %0.20 (1.000$'da %0.30).
- **O3 — CAUTION sınırı kaldırılır** (CAUTION'da da BULL gibi giriş; BEAR yasağı aynen kalır). Gerekçe:
  CAUTION günlerin %19–27'si; IWM rejimi IWM'i zamanlayamıyor (Ek N0). Etiket: CAUTION'da alınan
  işlemlerin (sınır 1) sonucu Ek N0'da görüldü (2021–23 n=17 −3.44; 2024–26 n=21 +15.43).

**Karar (her varyant tabana karşı, ayrı ayrı):** (a) tüm dönem toplamı ≥ taban + 5 puan; (b) iki yarının
her biri (2021–23 / 2024–son, her biri 10.000$'dan) ≥ tabanın o yarısı − 2 puan; (c) max düşüş ≥ taban
− 3 puan VE > −35%; (d) duyarlılıkta (20 bps) (a) de sağlanır. Birden fazla geçerse geçenlerin
birleşimi aynı kurallarla denenir ve en iyi tek varyantın toplamından düşük değilse benimsenir; değilse
en yüksek toplamlı tek varyant. Hiçbiri geçmezse canlı değişmez.

**Ek O kullanıcı kararı (2026-11-22, SONUÇ GÖRÜLDÜKTEN SONRA, etiketli sapma):** kullanıcı O2'yi ön kayıt
şartı (c, max düşüş) geçmemesine rağmen benimsedi: canlı H6S pozisyon büyüklüğü %10 → %15 (fren %7.5).
Gölge H6S kilidi (PARAMS_H6S) değişmez; bu yalnız canlı paper hesabın boyut ayarıdır.

# Ek P — H9 "Büyük Koşu Avcısı": göstergeleri büyük koşulara göre öğren, görülmemiş yıllarda sına

**Yazıldığı an: 2026-11-22, hiçbir özellik/model sonucu görülmeden.** Önceki işler fikirleri tek tek
sınadı (VCE, RVOL, gap, dip avı, 52H zirve, Form 4, PEAD); bu ek ters yönden gider: hedef = büyük koşu,
tüm göstergeler birlikte, model eğitim yarısında öğrenir, karar yalnız test yarısında.
Bilinen (betimleyici, tüm dönem): evrende 20 seansta +%50 olasılığı %1.29, −%30 %3.08; VCE sinyalinde
+%50 %5.48 (özellik/model bazında hiçbir şey görülmedi).

- **Evren:** §0 havuzu (`pool_mask`). Survivorship yanlılığı var → birincil ölçüt havuza göre fark.
- **Bölme:** eğitim = sinyal günü 2021-01-04 → 2023-11-30; test = 2024-01-02 → son−20 seans (etiket
  örtüşmesine karşı ~1 ay ambargo). Test yarısı model/eşik/özellik seçiminde KULLANILMAZ.
- **Etiket:** koşu = max(Close[t+1..t+20]) / Open[t+1] − 1 ≥ +%50 (giriş fiyatına göre, işlem yapılabilir).
- **Özellikler (t kapanışında bilinen, sabit liste, 35):** getiri 1/5/20/60/120 seans; Close/SMA20/50/200;
  52H zirve/dip uzaklığı, 52H zirveden beri gün; RSI14; ATR14/Close, ATR5/ATR50, BB genişliği /
  120 seans ortalaması; gün aralığı, kapanış konumu, bugünkü gap, son 20 seansın en büyük günlük getirisi;
  RVOL (V/ort63), 5 günlük RVOL, ort20/ort63 hacim, log dolar hacim, yükselen gün hacim payı (20);
  log market cap, log fiyat, 252 seanslık hisse sayısı değişimi; IWM'e göre 20/60 seans göreli güç, IWM
  20 seans getirisi, VIX, IWM rejimi (BULL/CAUTION/BEAR); sonraki/önceki bilançoya gün (60'ta kesik;
  bilanço takvimi önceden ilan edilir); havuzun SMA50 üstü oranı.
- **Model:** sklearn HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
  max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
  random_state=7). Ayar taraması YOK.
- **Seçim (test):** skor ≥ eğitim yarısı skorlarının 99. yüzdeliği (eşik eğitimden sabit); aynı gün
  skora göre sıralı. H6S çerçevesi: IWM rejimi (BEAR giriş yok, CAUTION'da en çok 1), hisse başına tek
  pozisyon, t+1 açılışta giriş, 2.5×ATR14 stop, **20. seans kapanışında çıkış** (etiket penceresi).
- **Portföy:** Ek O portföyü (doğru sıra), %15 pozisyon (canlı), L2 freni, 10.000$, 50 bps + 1.5$;
  duyarlılık 20 bps + 1.5$.

**Terfi kararı (yalnız test yarısı, hepsi birlikte):** (P1) ≥100 işlem; (P2) işlem başı net (50 bps) > 0
VE havuza göre fark ortalaması > 0 ve gün kümeli bootstrap %95 CI alt sınırı > 0; (P3) portföy toplamı >
IWM al-tut VE > H6S (%15) aynı dönemde; (P4) max düşüş > −35%; (P5) 2024/2025/2026 yıllarının en az
ikisinde havuza göre fark > 0. Geçerse H9 dondurulmuş hash'le gölge ileri teste girer ve kullanıcıya canlı
paper için sunulur; geçmezse kapanır. Betimleyici: test yarısında seçilenlerde koşu/çöküş oranı vs taban.

## Ek P sonucu (H9) ve P2 — H9b: hedef doğrudan işlem parası

**H9 sonucu (görüldü):** test AUC 0.873; seçilenlerde koşu %12.0 (taban %1.5) AMA çöküş %22.1 (taban
%3.0); 540 işlem, net −3.58%/işlem, havuz farkı −2.90 (CI −6.06…+0.30); portföy −79.7%. P2–P5 ❌ → H9
kapandı. Tanı: model "koşu olasılığını" öğrendi = oynaklık (ATR% önemin büyük kısmı); oynaklık iki yönlü.

**H9b — yazıldığı an: 2026-11-22, H9 sonucundan SONRA (etiketli: test yarısına İKİNCİ bakış, tek
deneme, başka varyant yok).** Değişen tek şey hedef: her hisse-günü için GERÇEK işlemin getirisi —
t+1 açılış girişi, 2.5×ATR14 stop (gün içi Low ≤ stop → min(Open, stop)), en çok 20 seans, 20. kapanışta
çıkış; yüzde, [−60, +150]'de kırpılır. Model: sklearn HistGradientBoostingRegressor (aynı
hiperparametreler, loss=squared_error). Seçim: tahmin ≥ eğitim yarısı tahminlerinin 99. yüzdeliği.
Evren, bölme, özellikler, rejim çerçevesi, portföy ve P1–P5 terfi kuralları H9 ile birebir aynı.

# Ek Q — H6SR "Kazananı koştur" gölge ileri testi (Ek O1 kuralı, canlıya dokunmadan)

**Yazıldığı an: 2026-11-22.** Ek O1 geçmiş testte işlem başı neti +0.79 → +1.40'a çıkardı ama portföyde
yol bağımlı çıktı (2021–23 −35 puan, 2024–26 +37 puan) → canlıya alınmadı. Kullanıcı kararıyla karar
geleceğin verisine bırakılır: gölge defterde H6S ile yan yana, AYNI sinyaller ve AYNI fren.

- **Giriş/fren (H6S ile birebir):** H6 isabetleri, dolar hacim sırası, IWM rejimi (BEAR giriş yok,
  CAUTION'da en çok 1 açık H6SR), hisse başına tek pozisyon, t+1 açılış, 2.5×ATR14 stop.
- **Çıkış (Ek O1 birebir, `lab/swing_o.walk_runner`):** 8. seans kapanışı giriş açılışının üstündeyse
  çıkılmaz; iz süren stop = max(önceki stop, girişten beri en yüksek kapanış − 2.5×ATR14), gün içi
  Low ≤ stop → min(Open, stop); en çok 40 seans. Değilse 8. kapanışta çıkış.
- **Kilit:** ayrı hash (`PARAMS_H6SR`); H4/H5/H6/H6S kilitleri değişmez. Canlı paper ürün değişmez.
- **Kapı (Ek C düzeni, otomatik terfi yok):** F1 ≥ 6 ay VE ≥ 100 uygun kapanmış işlem; F2 50 bps net
  ortalama > 0 VE CI alt sınırı > 0; F3 %10 portföy > IWM al-tut; F4 max düşüş > −35%. Ek olarak aynı
  dönemde H6S ile kıyas raporlanır; H6SR, H6S'nin yerine ancak F1–F4 geçer VE portföy toplamı H6S'den
  yüksekse kullanıcıya önerilir.

# Ek R — H6S'nin dokunulmamış yıllarda sınavı (2016–2020, kurallar donuk)

**Yazıldığı an: 2026-11-24, 2016–2020 dönemine ait hiçbir fiyat, sinyal veya sonuç görülmeden; veri henüz
indirilmedi.** Yalnız veri erişimi yoklandı (8 hisse: Yahoo hisse sayısı serisi 2015 sonunda başlıyor ve
bölünmeye göre DÜZELTİLMEMİŞ; bilanço takvimi 2015 öncesine iniyor; 2014 barları mevcut — getiri/sinyal
bakılmadı).

**Gerekçe.** H6S'nin kanıtı 2021–2026'ya ait ve iyimser: IWM freni fikri sonuç görüldükten sonra geldi,
2024–26 yarısı iki kez kullanıldı, kazancın tamamı tek bir düşüşten (2022) kaçmaktan geliyor. Projedeki
bütün ayar ve ölçümler 2021-01 → 2026-11 (Lab v3) ya da 2024-06 → 2026-05 (scripts/) verisini kullandı;
**2016–2020 hiçbir kararda kullanılmadı.** Donuk kuralları orada çalıştırmak, ileri testi beklemeden
yapılabilen tek "görülmemiş veri" sınavıdır. Dönem üç ayrı küçük hisse düşüşü içerir (2016 başı, 2018 son
çeyrek, 2020 Şubat–Mart) ve güçlü yükseliş yılları (2016–17, 2019, 2020 sonu).

**Donuk olanlar (hiçbiri değiştirilemez).** Kurallar ve kod: commit 45ebb3c; `PARAMS_H6S` kilidi
47dfd9d0…; ayarlar canlı yamanın 2026-10-11 kopyası (in-sample ile aynı). Sinyal yolu in-sample ile aynı
betikler: `lab/h0_signals.py --universe emulated` (Finviz emülasyonu + canlı motor) → `lab/h6_signals.py`
(canlı VCE tetiği + bilanço kapısı) → IWM rejimli `lab/swing_o.simulate` (BEAR giriş yok, CAUTION'da en çok
1; t+1 açılış; 2.5×ATR14 stop; 8. seans kapanışı; dolar hacim sırası; hisse başına tek pozisyon). Portföy:
`lab/swing_o.portfolio` (doğru gün içi sıra, L2 kayıp serisi freni, 10.000$).

**Veri.** yfinance günlük OHLCV (auto_adjust), aynı evren anlık görüntüsü (2026-10-10 adayları),
2014-10-01 → 2021-03-05. **Sinyal günleri 2016-01-04 → 2020-12-31** (5 takvim yılı). Hisse sayısı:
`get_shares_full(start=2015-01-01)`, in-sample ile aynı kurallar (ilk gözlem geriye taşınır; serisi olmayan
hisse güncel hisse sayısına düşer). Ülke/sektör/float: güncel. Bilanço takvimi: `get_earnings_dates(limit=100)`.

- **Bölünme düzeltmesi (veri görülmeden bulunan emülasyon kusuru, bu sınavda düzeltilir).** Yahoo fiyatları
  bugüne kadarki bölünmelere göre düzeltilmiş, hisse sayıları ise o günkü ham sayı. İkisinin çarpımı,
  sonradan bölünen hisselerde piyasa değerini bölünme oranı kadar küçültür (ör. Tesla 2016: gerçek ~30 milyar
  $, hesaplanan ~2 milyar $ → evrene sızar); ters bölünenlerde 7 $ altı kâğıtları 7 $ üstü gösterir. Kural:
  F(t) = t'den SONRAKİ bölünme oranlarının çarpımı (Yahoo "Stock Splits", bugüne kadar). O günkü gerçek
  fiyat = düzeltilmiş fiyat × F(t); gerçek ortalama hacim = düzeltilmiş ÷ F(t); piyasa değeri = gerçek fiyat ×
  o günkü hisse sayısı. Evren üyeliği, havuz, sert filtreler ve motorun gördüğü 6 aylık pencere bu gerçek
  değerlerle hesaplanır; getiriler düzeltilmiş fiyatla (değişmez). In-sample kökünde bölünme dosyası yok →
  o sonuçlar aynen kalır (aynı kusurun 2021–26'daki etkisi ayrıca, etiketli olarak ölçülebilir).
- **Bilanço veri kalitesi kuralı (canlıdaki "bilinmiyorsa ret" ilkesinin geçmiş veri karşılığı).** Sinyal
  günü d için takvim "biliniyor" sayılır ancak d'den önceki son kayıt ile d ve sonrasındaki ilk kayıt
  mevcutsa VE aralarındaki süre ≤ 155 günse (arada eksik çeyrek yok). Değilse sinyal reddedilir. Reddedilen
  sayı ve kuralsız sonuç betimleyici olarak raporlanır.
- **Bilinen yanlılık (in-sample'dakinden BÜYÜK).** Evren 2026'da hâlâ listede olan hisseler; 2016–2020'de
  listeden çıkan / satın alınan şirketler yok. Mutlak getiri iyimser çıkar. Bu yüzden: olumlu sonuç
  "destekleyici ama iyimser", olumsuz sonuç "güçlü olumsuz kanıt" sayılır. Yanlılık göstergesi olarak
  havuzun işlem başı getirisi IWM'inkiyle yan yana raporlanır.

**Ölçütler (2016–2020; birincil maliyet 50 bps gidiş-dönüş, portföyde ayrıca Midas 1.5$/emir).**

| # | Ölçüt |
|---|---|
| R0 | Yeterlilik: ≥150 H6S işlemi ve ≥30 farklı ay. Sağlanmazsa hüküm verilmez ("yetersiz veri"). |
| R1 | İşlem başı net ortalama (50 bps) > 0 VE havuza göre fark ortalaması > 0 |
| R2 | Havuz farkının gün kümeli bootstrap %95 CI alt sınırı > 0 (`lab/evaluate.boot_ci`) |
| R3 | Canlı ayar portföyü (%15 pozisyon, L2 freni, 50 bps + 1.5$, 10.000$) toplam getirisi > IWM al-tut. Pencere: 2015-12-31 kapanışı → son H6S çıkış günü kapanışı; portföy ilk işleme kadar nakitte. |
| R4 | Aynı portföyün max düşüşü > −35% |
| R5 | Frenin değeri: aynı portföy ayarında IWM frenli H6S toplamı ≥ frensiz varyantın toplamı (BEAR'de giriş serbest, CAUTION sınırsız; stop ve tutuş aynı) VE max düşüşü daha sığ |

**Hüküm (önceden sabit).**
- **GEÇTİ:** R1, R2, R3, R4 birlikte.
- **KISMEN:** R1 sağlanır VE canlı ayar portföyünün toplamı > 0, ama R2 / R3 / R4'ten en az biri sağlanmaz.
- **KALDI:** R1 sağlanmaz veya canlı ayar portföyünün toplamı ≤ 0.

R5 ayrıca geçti/kaldı olarak raporlanır (fren 2022 dışındaki düşüşlerde de işe yarıyor mu).

**Betimleyici okumalar (hükmü değiştirmez).** %10 pozisyon; 20 bps; 25.000$ hesap; yıllık tablo (portföy ve
IWM); üç düşüş penceresi (2015-12-31→2016-02-11, 2018-08-31→2018-12-24, 2020-02-19→2020-03-18) ve rejim
dağılımı; §1 P1–P6 (P3 için havuz farkı pozitif yıl sayısı x/5 olarak verilir); H6SR çıkışı (Ek O1) aynı
sinyallerde; bölünme düzeltmesinin, hisse sayısı yedeğinin ve bilanço kuralının etkilediği sinyal sayıları.

**Sonuca göre ne yapılacak (önceden sabit).** Hüküm ne olursa olsun canlı ürünün kuralları ve gölge kilitleri
DEĞİŞMEZ; nihai hakem gölge ileri testtir. GEÇTİ / KISMEN → H6S'ye güven artar, başka işlem yok. KALDI →
kullanıcıya açıkça "2021–26 kazancı döneme özgü olabilir" denir. **Bu dönem bir kez kullanılır:** sonuç
görüldükten sonra 2016–2020 verisinde kural, parametre veya varyant denemesi yapılmaz (yapılırsa bu dönem
de in-sample olur); yeni fikirler yalnız ileri testte sınanır.

# Ek R sonucu ve etiketli ek kontroller (2026-11-24, SONUÇ GÖRÜLDÜKTEN SONRA yazıldı)

Yukarıdaki Ek R metni değiştirilmedi (bölüm SHA256, LF: 31fe5e5edea8b1135e97607a8ab7453b4a91e844c0cc56da42f2e7ff188632f4;
`lab/oos_r.prereg_sha256`). **Hüküm: KALDI** — 971 işlem, net −0.26%, havuz farkı −0.00 [−0.55, +0.58], canlı ayar
portföyü −56.0% (düşüş −63.4) vs IWM +97.5%; R1–R5 ✘. Ayrıntı: LAB_V3_REPORT.md Ek R.

Sonuç görüldükten sonra yapılan, hükmü DEĞİŞTİRMEYEN iki tanı (strateji varyantı değil, veri işleme kontrolü):
- **Düzeltmesiz 2016–2020** (in-sample ile aynı kusurlu evren emülasyonu): yine KALDI (−36.9%). Kötü sonuç
  bölünme düzeltmesinden gelmiyor.
- **Düzeltilmiş 2021–26** (ön kayıtta "ayrıca, etiketli olarak ölçülebilir" denen in-sample'a yeniden bakış):
  +66.2% → −30.2%. Eski kârın +5.055 $'ı (toplam +6.624 $) canlı evrende imkânsız tek işlemdi (AMC, gerçek
  fiyat 3.06 $). Bu bulgu H6S'nin 2021–26 kanıtını geçersiz kılar; Ek J/L/M/N/O rakamları kusurlu evrenledir.

Bundan sonraki kural (yeni ön kayıtlar için): hipotezler düzeltilmiş evrende iki dönemde de sınanır
(`output/lab_oos` 2016–2020, `output/lab_is_split` 2021–26). Ek R'nin "yeni fikirler yalnız ileri testte" cümlesi
H6S varyantları için geçerlidir; H6S'den bağımsız, önceden kilitlenmiş hipotezlerin (ör. H4, H5) 2016–2020'de
sınanması ayrı bir ön kayıtla yapılabilir.

# Ek S — H4 ve H5'in iki dönemde sınavı (düzeltilmiş evren; 2016–2020 ilk kez, 2021–26 yeniden)

**Yazıldığı an: 2026-11-24, Ek R sonucundan SONRA; H4/H5'in 2016–2020 verisi (SEC Form 4, bilanço sürprizleri)
indirilmeden ve bu iki sinyalin düzeltilmiş evrendeki hiçbir sonucu görülmeden.** Görülenler: (i) H4/H5'in
kusurlu evrendeki 2021–26 sonuçları (Ek A: H4 net +0.23%, havuz farkı +0.19 [−0.53, +0.94]; H5 net −0.26%,
havuz farkı +0.27 [−0.02, +0.57]); (ii) 2016–2020'de yalnız H6S'nin ve havuz/IWM'in getirileri (Ek R).

**Gerekçe.** Kullanıcı kararı (2026-11-24): H6S'nin kanıtı çöktükten sonra gölgedeki iki bilgi tabanlı sinyal
aynı yöntemle sınansın. İkisi de fiyat kalıbı değil kamuya açık bilgi kullanıyor; kusurlu evrende havuzu
brüt yendiler ama maliyeti aşamadılar.

**Donuk olanlar.** Kurallar Ek A ile birebir (kilit `PARAMS_SHA256` 178c293e…); kod `lab/info_signals.py`
(commit b77d7e5; tek ekleme değerlendirme dönemi sonu `EVAL_END`). Havuz §0, bölünme düzeltmeli (o günün
gerçek fiyatı 7–250 $, gerçek ortalama hacim > 300K, dolar hacim > 5M $, piyasa değeri 300M–10B $). Giriş
sinyal barından sonraki ilk açılış; stop giriş − 2.5×ATR14; 20. seans kapanışı; hisse başına tek pozisyon;
aynı gün sıralama H4'te alım değeri, H5'te sürpriz yüzdesi. Maliyet 50 bps gidiş-dönüş (20 bps ayrıca).
Portföy §3 (`lab/evaluate.portfolio`: 10.000$, %10/işlem, nakit sınırlı, doğru gün içi sıra).

**Veri.**
- **2016–2020** (`output/lab_oos`; fiyat, bölünme ve hisse sayısı Ek R'de indirildi): H4 için SEC DERA
  "Insider Transactions Data Sets" 2015Q3–2020Q4; H5 için havuza en az bir gün girmiş 1.710 hissenin
  `get_earnings_dates(limit=100)` kayıtları (zaman damgası, tahmin, açıklanan, sürpriz). Sinyal günleri
  2016-01-04 → 2020-12-31.
- **2021–26 düzeltilmiş** (`output/lab_is_split`): Ek A'nın aynı verisi (SEC 2020Q3–2026Q1; `earnings_full`,
  limit=40), evren ve havuz bölünme düzeltmeli. Sinyal günleri 2021-01-04 → 2026-11-20 (H4 verisi 2026Q1'de biter).
- **Bilinen sınırlar:** survivorship (evren 2026'da listede olanlar; mutlak getiri iyimser, havuz farkı kısmen
  nötr); Form 4'teki sembol o günün sembolüdür (sembol değiştiren şirketler düşer, yeniden kullanılan semboller
  gürültü ekler); Yahoo tahmin/EPS değerleri sonradan revize edilmiş olabilir; eski yıllarda açıklama saat
  damgası güvenilmez olabilir → her dönem için havuz içi H5 olaylarının saat dağılımı raporlanır; olayların
  ≥%30'u [05:00–09:30) ∪ [16:00–20:00] ET dışındaysa o dönemin H5 sonucu "zamanlama belirsiz" diye etiketlenir
  (hüküm yine verilir).

**Ölçütler (her sinyal için, her dönemde ayrı).**

| # | Ölçüt |
|---|---|
| S1 | Yeterlilik: H5 ≥ 150, H4 ≥ 50 işlem ve ≥ 30 farklı ay |
| S2 | İşlem başı net ortalama (50 bps) > 0 |
| S3 | Havuza göre fark > 0 VE gün kümeli bootstrap %95 CI alt sınırı > 0 (`lab/evaluate.boot_ci`) |
| S4 | §3 portföyünün toplamı > IWM al-tut (pencere: dönem başından önceki son kapanış → son çıkış günü; portföy ilk işleme kadar nakitte) |
| S5 | Portföyün max düşüşü > −35% |

**Hüküm (her sinyal için, önceden sabit).**
- **GEÇTİ:** 2016–2020'de S1–S5 hepsi VE düzeltilmiş 2021–26'da S2 ile S4 sağlanır ve havuz farkı > 0.
- **KISMEN (gerçek ama küçük):** havuz farkı iki dönemde de > 0 VE iki dönemin işlemleri birlikte alındığında
  gün kümeli %95 CI alt sınırı > 0; ama GEÇTİ şartları sağlanmaz.
- **KALDI:** diğer her durum. S1 2016–2020'de sağlanmazsa hüküm "yetersiz veri".

**Sonuca göre (önceden sabit).** GEÇTİ → sinyal kullanıcıya canlı paper adayı olarak sunulur (karar kullanıcıda);
gölge kaydı ve kilidi aynen sürer. KISMEN → gölgede kalır; maliyeti aşacak yeni bir tasarım ancak yeni ön kayıt
ve ileri testle. KALDI → aday sayılmaz; gölge kaydı kendi kapısına (Ek C) kadar sürer. Hiçbir durumda parametre
değiştirilmez ve bu veride H4/H5 varyantı denenmez. Betimleyici: yıllık tablo, 20 bps, §1 P1–P6, çıkış nedenleri.

# Ek S sonucu (2026-11-24, SONUÇ GÖRÜLDÜKTEN SONRA yazıldı)

Yukarıdaki Ek S metni değiştirilmedi (bölüm SHA256, LF: 14b3007e1ce759fd056fbe6510fdc02c52fc1520bc7dee8834f0c6eddbe1aeb2;
`lab/oos_s.prereg_sha256`). **Hüküm: H4 KALDI, H5 KALDI.** H4: havuz farkı 2016–2020 +0.12 [−0.73, +0.98],
2021–26 +0.30 [−0.40, +1.02], birlikte +0.22 [−0.32, +0.75]; portföy +15.8% / −11.8% vs IWM +102% / +48%.
H5: havuz farkı 2016–2020 −0.75 [−1.11, −0.38], 2021–26 +0.23 [−0.06, +0.53], birlikte −0.12 [−0.36, +0.11];
portföy −17.9% / −26.9% vs IWM +104% / +40%. Ön kayıt gereği ikisi de aday sayılmaz; gölge kayıtları Ek C
kapısına kadar sürer. Ayrıntı: LAB_V3_REPORT.md Ek S.

# Ek T — Farklı türden iki fikir, büyük ve likit hisselerde: MOM (lider rotasyonu) ve DIP (panik dipleri)

**Yazıldığı an: 2026-11-24, Ek R ve Ek S sonuçlarından SONRA; bu iki fikrin büyük hisse havuzundaki hiçbir
sonucu görülmeden ve sürekli veri kökü (2014–2026) indirilmeden.** Görülenler: küçük/orta havuzda H2 (RSI2
dönüşü) ve H3 (52 hafta zirvesi + göreli güç) 2021–26 sonuçları (kusurlu evren; ikisi de maliyet sonrası
negatif), büyük havuzda yalnız H4L/H5L (Ek D2). Genel piyasa bilgisi (veriden değil): büyük hisse momentumu
2016–2020 ve 2023–2025'te iyi, 2021–22'de kötüydü; RSI2 dönüşü 2010 sonrası zayıfladı.

**Gerekçe (kullanıcı kararı, 2026-11-24).** Küçük/orta hisselerde günlük al-sat ailesinden ~15 fikir maliyeti
aşamadı. İki yapısal fark denenir: (1) maliyetin çok düşük olduğu büyük ve likit hisseler; (2) MOM'da az işlem
ve uzun tutuş, DIP'te kırılımın tersi (düşüşü almak). İkisi de literatürün standart kurallarıyla, tek
parametre setiyle; tarama yok.

**Evren ve veri.** Havuz L = Ek D2 havuzu, bölünme düzeltmeli: o günün gerçek fiyatı ≥ 7 $, gerçek ortalama
hacim (63 seans) > 300K, 20 seans ortalama dolar hacim > 5M $, piyasa değeri ≥ 10 milyar $ (tavan yok, ülke
filtresi yok). Veri: tek bir sürekli kök (`output/lab_liq`), yfinance günlük OHLCV + bölünme olayları,
2014-10-01 → 2026-11-20; hisse listesi = mevcut iki kökte en az bir gün havuz L'nin piyasa değeri dışındaki
şartlarını sağlamış hisseler; hisse sayısı `get_shares_full(start=2015-01-01)`. İki dönem ayrı ayrı, her biri
10.000$'dan: **A = 2016-01-04 → 2020-12-31, B = 2021-01-04 → 2026-11-20.** Bilinen yanlılık: survivorship
(evren 2026'da listede olanlar) → mutlak getiri iyimser; bu yüzden havuz farkı da ölçüttür.

**MOM — lider rotasyonu (aylık, 12-1 momentum).**
- Karar günü t: her ayın son seansı (dönem başından önceki son seans dahil; dönemin son seansı hariç). İşlem
  t'den sonraki ilk seansın açılışında.
- Uygun hisse: t'de havuz L'de, Close_t > SMA200_t, skoru hesaplanabiliyor. Skor = Close_{t−21} / Close_{t−252} − 1.
- Sat: elde tutulan hisse t'de uygun değilse ya da uygunlar içinde skor sırası 20'den büyükse.
- Al: yalnız sistemin piyasa rejimi t'de BULL ise (canlı `regime_from_spy_close`, SPY 1 yıl + VIX; CAUTION ve
  BEAR'de yeni alım yok, eldekiler kendi satış kuralına kadar tutulur). Boş yerler elde olmayan en yüksek
  skorlu uygun hisselerle doldurulur; en çok **10** pozisyon; her alım özsermayenin **%10**'u.
- Stop ve hedef yok. Dönem sonunda açık pozisyonlar son seansın kapanışıyla kapatılır.

**DIP — panik dipleri (Ek §2 H2 kuralı birebir, havuz L'de).**
- Sinyal t: Close_t > SMA200_t, RSI(2)_t < 5 (Wilder), Close_t > SMA50_t × 0.95, t'de havuz L'de.
- Giriş t+1 açılış. Çıkış: Close > SMA5 olan ilk günün kapanışı, en geç 10. seans kapanışı; felaket stop'u
  giriş × 0.85 (stop önce, gap'te açılıştan). Hisse başına tek pozisyon; aynı gün sıralama RSI(2) küçükten büyüğe.
- Her işlem özsermayenin %10'u, nakit yetmezse atlanır (en çok 10 eşzamanlı).

**Maliyet ve portföy.** Birincil: **10 bps gidiş-dönüş** (Ek D2'nin büyük hisse maliyeti) **+ Midas 1.5$/emir**,
10.000$; `lab/swing_o.portfolio` (doğru gün içi sıra, kayıp serisi freni yok). Duyarlılık: 20 bps; 25.000$ ve
50.000$ hesap.

**Ölçütler (her fikir için; her dönemde ayrı).**

| # | Ölçüt |
|---|---|
| T0 | Yeterlilik: her dönemde ≥ 100 işlem |
| T1 | Para: portföy toplam getirisi > SPY al-tut (dönem başından önceki son kapanış → dönemin son seansı), **iki dönemde de** |
| T2 | Risk: portföy max düşüşü > −35%, iki dönemde de |
| T3 | Seçim: havuz L'ye göre fark (aynı giriş ve çıkış zamanlaması, eşit ağırlıklı havuz L üyeleri) iki dönemde de > 0 VE iki dönem birlikte gün kümeli bootstrap %95 CI alt sınırı > 0 |

**Hüküm (önceden sabit).**
- **GEÇTİ:** T0–T3 hepsi.
- **KISMEN:** T0 ve T1 sağlanır (iki dönemde de SPY'ı geçer) ama T2 veya T3 sağlanmaz.
- **KALDI:** T1 herhangi bir dönemde sağlanmaz. T0 sağlanmazsa "yetersiz veri".

**Sonuca göre (önceden sabit).** GEÇTİ → kullanıcıya ürünün ana stratejisi olarak önerilir (mimari değişikliği;
karar kullanıcıda) ve donmuş kurallarla gölge ileri teste alınır. KISMEN → yalnız gölge ileri test. KALDI →
kapanır; bu veride varyantı denenmez. Betimleyici (hükmü değiştirmez): yıllık tablo, IWM kıyası, ortalama
nakit oranı, yıllık emir sayısı, MOM'un rejim kuralsız hali, iki fikrin yarı yarıya iki alt hesapla birleşimi.

# Ek T sonucu (2026-11-24, SONUÇ GÖRÜLDÜKTEN SONRA yazıldı)

Yukarıdaki Ek T metni değiştirilmedi (bölüm SHA256, LF: 281227721bb3d79be60c9bd335ae9784698d25a8081caa9f73860c80fc1a59a3;
`lab/liq_t.prereg_sha256`). **Hüküm: MOM KALDI, DIP KALDI.**
- MOM: A (2016–2020) portföy +300.9%, düşüş −32.4%, SPY +102.0% (T1 ✔ T2 ✔); B (2021–26) +22.0%, düşüş −71.0%,
  SPY +99.2% (T1 ✘ T2 ✘). Havuz farkı +14.0 / +2.7, birlikte +7.8 [+0.3, +16.8] (T3 ✔). T1 dönem B'de
  sağlanmadığı için KALDI.
- DIP: havuz farkı +0.03 / +0.02 (birlikte +0.02 [−0.05, +0.10]); portföy −59.7% / −100% (sabit emir ücreti);
  T1–T3 ✘.
Ön kayıt gereği ikisi de kapandı; bu veride varyant denenmez. Geçmiş veri üzerindeki strateji arayışı kapatıldı.
Ayrıntı: LAB_V3_REPORT.md Ek T.

# Ek U — MOM "lider rotasyonu" gölge ileri testi (kullanıcı kararı; Ek T'deki "KALDI → kapanır" sonucunun etiketli istisnası)

**Yazıldığı an: 2026-11-25, Ek T sonucu GÖRÜLDÜKTEN SONRA; ileri testin hiçbir verisi yokken.** MOM geçmiş sınavda
KALDI (2016–2020 +301% / düşüş −32%; 2021–26 +22% / düşüş −71%, SPY +99%) ve Ek T'ye göre kapanması gerekiyordu.
Kullanıcı kararıyla (2026-11-25) yalnız **gölge ileri teste** alınır: gerçek para yok, canlı paper ürün değişmez,
geçmiş sınavın hükmü değişmez. Gerekçe: projede havuzu istatistiksel olarak yenen tek sinyal buydu; değeri ancak
görülmemiş gelecekte ölçülebilir. MOM gerçek para adayı DEĞİLDİR.

**Kurallar (Ek T MOM ile birebir, donuk; ayrı kilit `PARAMS_MOM`).**
- Havuz L: o günün fiyatı ≥ 7 $, 63 seans ortalama hacim > 300K, 20 seans ortalama dolar hacim > 5M $, piyasa
  değeri ≥ 10 milyar $.
- Karar günü t: her ayın son seansı. İşlem t'den sonraki ilk seansın açılışında.
- Uygun hisse: t'de havuz L'de, Close_t > SMA200_t, skor hesaplanabiliyor. Skor = Close_{t−21} / Close_{t−252} − 1.
- Sat: elde tutulan hisse uygun değilse ya da uygunlar içinde skor sırası 20'den büyükse.
- Al: yalnız rejim BULL ise (canlı `regime_from_spy_close`, SPY 1 yıl + VIX); en yüksek skorlu, elde olmayan
  uygun hisselerle en çok 10 pozisyona tamamlanır. Stop ve hedef yok.
- Değerlendirme maliyeti: 10 bps gidiş-dönüş; portföyde ayrıca 1.5$/emir, 10.000$, pozisyon başına %10.

**Canlı uygulamanın lab'dan farkları (kuralı değiştirmez).** Piyasa değeri canlı kaynaktan gelir (Finviz taraması:
piyasa değeri > 10 milyar $, fiyat > 7 $, ortalama hacim > 300K; lab'da hisse sayısı × fiyattı); fiyat ve hacim
canlı fetcher'dan (yfinance). Alım kararında ertesi açılış henüz bilinmediği için açılış barı gelmeyen alım
"atlandı" olur ve yeri bir sonraki ay sonuna kadar boş kalır (lab sıradaki hisseyi alırdı). Evren ya da ay sonu
barları alınamazsa karar ertelenir ve yedek tetikleyiciler yeniden dener; kaçırılan ay sonu sonradan işlenir ve
alım/satım "geç karar" işaretlenir (değerlendirmeye girmez).

**Kapı (Ek C düzeni; otomatik terfi yok).** F1: ≥ 24 ay VE ≥ 40 uygun kapanmış işlem. F2: işlem başı net ortalama
(10 bps) > 0 VE giriş günü kümeli bootstrap %95 CI alt sınırı > 0. F3: portföy (%10/pozisyon, 10 bps + 1.5$/emir)
toplamı > aynı dönemde SPY al-tut. F4: max düşüş > −35%. Durum: F1'e kadar TOPLANIYOR; sonra F2–F4 hepsi →
TERFİ ADAYI; net ortalama ≤ 0 → KALDI; 36. ayda hâlâ aday değilse KALDI; aksi halde BELİRSİZ. "TERFİ ADAYI" yalnız
tartışmayı açar; geçmiş sınavdaki −71% düşüş o tartışmada kullanıcıya birlikte sunulur.

**Başlangıç.** İlk karar günü gölge koşusunun MOM'u ilk çalıştırdığı günden sonraki ilk ay sonu seansıdır
(2026-11-30 → 2026-12-01 açılışı). H4/H5/H6/H6S/H6SR kilitleri ve kapıları değişmez.

# Ek V — PRO: büyüyen lider hisse + kırılım + dar stop + kazananı süresiz tutma + değişken boyut

**Yazıldığı an: 2026-11-26, Ek R, S ve T sonuçlarından SONRA; bu kural setinin hiçbir sonucu görülmeden ve
bilanço (satış/kâr) verisi indirilmeden.** Görülenler: Ek R–T'nin tamamı (özellikle havuz L'de MOM işlemleri:
kazancın birkaç dev işlemde toplandığı, 2021–22'de −71% düşüş; H6S ve H6SR sonuçları). Genel piyasa bilgisi
(veriden değil): büyüme liderleri 2016–2020 ve 2023–2025'te çok iyi, 2021–22'de çok kötüydü; bu bilgi kuralların
seçimini (rejim şartı, stop) etkilemiş olabilir.

**Gerekçe (kullanıcı kararı, 2026-11-26; Ek T'deki "geçmiş veri arayışı kapatıldı" sonucunun etiketli istisnası).**
Kullanıcı kazanan swing trader'ların nasıl kazandığını sordu. Yayımlanmış yöntemlerinin (O'Neil, Minervini;
Kullamägi) üründen farkı beş başlıkta toplandı: lider/büyüyen hisse seçimi, dar stop, kazananı süresiz tutma,
değişken boyut, düşük maliyet. Bunlar TEK paket, TEK parametre setiyle sınanır; tarama yok. Parametreler bu
yöntemlerin yayımlanmış standart değerleridir, bu veride ayarlanmadı. Bu, projenin ~18. hipotezidir (çoklu test):
GEÇTİ sonucu bile kanıt sayılmaz, yalnız ileri test hakkı verir. Gün içi (kırılım anında) giriş günlük veriyle
sınanamaz; giriş ertesi açılıştandır.

**Evren ve veri.** Kök `output/lab_liq` (Ek T ile aynı fiyat, bölünme ve hisse sayısı verisi; yeniden indirilmez).
Havuz G (o günün gerçek değerleriyle, bölünme düzeltmeli): fiyat ≥ 10 $, 63 seans ortalama hacim > 300K, 20 seans
ortalama dolar hacim > 10M $, piyasa değeri ≥ 1 milyar $ (tavan yok, ülke filtresi yok; kıyas ETF'leri hariç).
İki dönem ayrı ayrı, her biri 10.000$'dan: **A = 2016-01-04 → 2020-12-31, B = 2021-01-04 → 2026-11-20.**

**Büyüme verisi.** SEC'in toplu XBRL dosyası (`companyfacts.zip`), hisse → CIK eşlemesi SEC `company_tickers.json`
(güncel). Çeyreklik satış ve net kâr şöyle kurulur:
- Satış etiketi (aynı dönem için birden çok etiket varsa en büyük değer): `Revenues`,
  `RevenueFromContractWithCustomerExcludingAssessedTax`, `RevenueFromContractWithCustomerIncludingAssessedTax`,
  `SalesRevenueNet`, `SalesRevenueGoodsNet`, `RevenuesNetOfInterestExpense`. Net kâr: `NetIncomeLoss`, yoksa
  `ProfitLoss`. Birim USD.
- Her (başlangıç, bitiş) dönemi için İLK dosyalanan değer ve o dosyanın tarihi kullanılır (sonradan düzeltilmiş
  değer kullanılmaz). Çeyrek = süresi 80–100 gün olan dönem. Yıl sonu çeyreği ayrıca raporlanmadıysa:
  yıllık (350–380 gün) − aynı başlangıçlı dokuz aylık (260–285 gün); bilinme tarihi ikisinin geç olanı.
- Değer, dosyalama tarihinden SONRAKİ ilk seanstan itibaren bilinir sayılır (dosya tarihi < sinyal günü t).

**Büyüme şartı (t günü).** Q = t'den önce bilinen, hem satışı hem net kârı olan en yeni çeyrek; Q'nun dönem sonu
t'den en çok 200 gün önce. Q−4 = dönem sonu Q'dan 350–380 gün önce olan çeyrek, t'den önce biliniyor.
Şart: Satış(Q) / Satış(Q−4) − 1 ≥ **%25**, Satış(Q−4) > 0 VE NetKâr(Q) > NetKâr(Q−4). Veri eksikse hisse UYGUN
DEĞİL (bilinmiyorsa ret; çeyreklik XBRL dosyalamayan yabancı şirketler bu yüzden dışarıda kalır).

**Lider şartı (t günü; Minervini "trend şablonu", yayımlandığı gibi).** Close > SMA150 ve Close > SMA200;
SMA150 > SMA200; SMA200_t > SMA200_{t−21}; SMA50 > SMA150; Close > SMA50; Close ≥ 1.30 × 252 seansın en düşük
Low'u; Close ≥ 0.75 × 252 seansın en yüksek High'ı; göreli güç: Close_t / Close_{t−252} − 1 değerinin t günü
havuz G üyeleri içindeki yüzdelik sırası ≥ **80**.

**Tetik (t günü).** Ürünün VCE kırılımı, aynen (`signals.check_vce_breakout` sert kapıları): ATR%(14)_{t−1} <
0.8 × ortalama ATR%(t−20 … t−6); Close_t > önceki 20 seansın en yüksek High'ı; Close_t > Close_{t−1};
Close_t > SMA50; Hacim_t ≥ 1.5 × 50 seans ortalama hacim (t dahil). Vektörel uygulama üretim fonksiyonuyla parite
testine tabidir.

**Piyasa şartı.** Yeni alım yalnız sistemin piyasa rejimi t'de BULL ise (canlı `regime_from_spy_close`, SPY 1 yıl +
VIX; Ek T ile aynı). Eldekiler kendi çıkış kuralına kadar tutulur.

**Giriş ve çıkış.** Sinyal = t'de havuz G ∧ büyüme ∧ lider ∧ tetik ∧ piyasa. Giriş t+1 açılışı (açılış barı yoksa
işlem yok). Hisse başına tek açık pozisyon.
- Stop = max(Low_t, 0.92 × giriş). Giriş ≤ Low_t ise işlem yok. Giriş günü dahil her gün: Low ≤ stop → min(Open, stop).
- İz süren çıkış: Close < SMA50 olan ilk günün ERTESİ açılışında satış. Süre sınırı ve kâr hedefi yok.
- Dönem sonunda açık pozisyon son seansın kapanışıyla kapatılır.

**Boyut ve portföy.** 10.000$. İşlem başına risk = özsermayenin %1'i (önceki gün kapanış özsermayesi):
pozisyon = min(%25 × özsermaye, %1 × özsermaye ÷ stop uzaklığı), stop uzaklığı = (giriş − stop) / giriş. Son 2
kapanan işlem de (maliyet sonrası) zarardaysa risk %0.5'e iner, ilk kârlı kapanışta %1'e döner. Nakit yetmezse
işlem atlanır; borç yok. Aynı gün birden çok sinyalde öncelik 12 aylık getirisi yüksek olanındır.
**Maliyet (birincil): 20 bps gidiş-dönüş + Midas 1.5$/emir.** Duyarlılık: 10 ve 50 bps; 25.000$ hesap.

**Ölçütler (her dönemde ayrı).**

| # | Ölçüt |
|---|---|
| V0 | Yeterlilik: her dönemde ≥ 60 sinyal düzeyi işlem |
| V1 | Para: portföy toplam getirisi > SPY al-tut (dönem başından önceki son kapanış → dönemin son seansı), **iki dönemde de** |
| V2 | Risk: portföy max düşüşü > −35%, iki dönemde de |
| V3 | Seçim: havuz G'ye göre fark (aynı giriş açılışı ve aynı çıkış zamanlaması, t günü havuz G üyelerinin eşit ağırlıklı getirisi) iki dönemde de > 0 VE iki dönem birlikte giriş günü kümeli bootstrap %95 CI alt sınırı > 0 |

**Hüküm (önceden sabit).** GEÇTİ: V0–V3 hepsi. KISMEN: V0 ve V1 sağlanır ama V2 veya V3 sağlanmaz. KALDI: V1
herhangi bir dönemde sağlanmaz. V0 sağlanmazsa "yetersiz veri" (şartlar gevşetilmez).

**Sonuca göre (önceden sabit).** GEÇTİ ya da KISMEN → yalnız donmuş kurallarla gölge ileri test; ürünün ana
stratejisi olması ayrı bir kullanıcı kararıdır ve ileri test sonucunu bekler. KALDI ya da yetersiz veri → kapanır;
bu veride varyantı denenmez. Betimleyici (hükmü değiştirmez; hangi parçanın ne kattığını göstermek için):
büyüme şartsız, lider şartsız, piyasa şartsız, iz süren çıkış yerine 8. seans kapanışı, değişken boyut yerine sabit
%10; yıllık tablo; en iyi 5 işlemin payı; büyüme verisi kapsama oranı; IWM kıyası.

**Bilinen yanlılıklar.** Survivorship (evren 2026'da listede olanlar; hisse → CIK eşlemesi güncel) → mutlak getiri
iyimser. Dosyalama tarihi basın açıklamasından geç olduğundan büyüme bilgisi gerçekte olduğundan geç kullanılır
(temkinli yön). Günlük barda stop, stop fiyatından (boşlukta açılıştan) gerçekleşmiş sayılır.

# Ek V sonucu (2026-11-26, SONUÇ GÖRÜLDÜKTEN SONRA yazıldı)

Yukarıdaki Ek V metni değiştirilmedi (bölüm SHA256, LF: c482e3da361748a2272bb3e37c47b1f1e45e7e2312ba851386ae6871a17c7a2a;
`lab/pro_v.prereg_sha256`). **Hüküm: KALDI.**
- A (2016–2020): 104 işlem, portföy +6.4%, düşüş −14.7%, SPY +102.0% (V1 ✘, V2 ✔). B (2021–26): 161 işlem, portföy
  +52.2%, düşüş −26.2%, SPY +99.2% (V1 ✘, V2 ✔). Havuz G farkı +0.83 / +2.32, birlikte +1.74 [−0.69, +4.62] (V3 ✘).
- Hesap iki dönemde de artıda ve düşüş sınırın içinde kaldı, ama endeksin çok gerisinde: paranın ~%75'i nakitte
  bekledi; kazanma oranı %25 / %16; en iyi 3 işlem çıkarılınca ortalama işlem iki dönemde de zararda.
Ön kayıt gereği kapandı; bu veride varyantı denenmez. Betimleyici kıyaslar (büyüme şartsız, lider şartsız, …)
hüküm değildir ve sonradan seçilemez. Ayrıntı: LAB_V3_REPORT.md Ek V.

# Ek W — GRW "büyüme + kırılım" gölge ileri testi (kullanıcı kararı; Ek V'deki "KALDI → kapanır" sonucunun etiketli istisnası)

**Yazıldığı an: 2026-11-26, Ek V sonucu GÖRÜLDÜKTEN SONRA; ileri testin hiçbir verisi yokken.** Ek V'nin ön
kayıtlı paketi (PRO) KALDI. Betimleyici kıyaslarından biri — paketten lider şartı çıkarılmış hali — iki dönemde
+36.4% / +64.0% (düşüş −21.4 / −22.8; SPY +102.0 / +99.2) verdi ve havuz farkı iki dönemde ayrı ayrı sıfırın
üstündeydi (+2.84 [+0.31, +6.07] / +2.38 [+0.50, +4.47]). Bu, beş kıyastan sonuç görüldükten sonra seçilmiştir
(çoklu karşılaştırma) ve endeksi iki dönemde de geçememiştir: **kanıt değildir, gerçek para adayı DEĞİLDİR.**
Kullanıcı kararıyla (2026-11-26) yalnız **gölge ileri teste** alınır; canlı paper ürün ve geçmiş sınavın hükmü
değişmez. Gerekçe: fiyat dışı bir bilginin (satış ve kâr büyümesi) kırılım sinyaline değer kattığı projedeki ilk
gözlem; değeri ancak görülmemiş gelecekte ölçülebilir.

**Kurallar (Ek V'nin "lider şartsız" kıyası ile birebir, donuk; ayrı kilit `PARAMS_GRW`).**
- Havuz G: o günün fiyatı ≥ 10 $, 63 seans ortalama hacim > 300K, 20 seans ortalama dolar hacim > 10M $, piyasa
  değeri ≥ 1 milyar $.
- Tetik (t): ürünün VCE kırılımı (`signals.check_vce_breakout` sert kapıları).
- Büyüme (t): Ek V "Büyüme verisi" ve "Büyüme şartı" aynen — SEC XBRL, dönemi ilk dosyalayan belgedeki değer,
  dosya tarihi < t; son çeyrek satışı yıllık ≥ %25 artmış VE net kâr yıl önceki çeyrekten yüksek; çeyrek en çok
  200 gün eski; veri yoksa ret.
- Piyasa: yalnız rejim BULL ise alım (canlı `regime_from_spy_close`, SPY 1 yıl + VIX).
- Giriş t+1 açılışı; hissede açık GRW pozisyonu varken yeni sinyal alınmaz. Stop = max(Low_t, 0.92 × giriş);
  giriş ≤ Low_t ise işlem yok. Çıkış: Low ≤ stop → min(Open, stop); Close < SMA50 → ertesi açılış. Süre sınırı yok.
- Değerlendirme maliyeti: 20 bps gidiş-dönüş; portföyde ayrıca 1.5$/emir, 10.000$; boyut = %1 risk ÷ stop
  uzaklığı, en çok %25, son 2 kapanan işlem zarardaysa risk %0.5.

**Canlı uygulamanın lab'dan farkları (kuralı değiştirmez).** Aday listesi Finviz taramasından gelir (piyasa değeri
> 300 mn $, fiyat > 10 $, ortalama hacim > 300K, 20 günlük yeni zirve, fiyat SMA50 üstünde, gün artıda — VCE'nin
zorunlu kıldığı şartların üst kümesi); havuz ve tetik şartları canlı barlardan yeniden ölçülür, piyasa değeri
Finviz'den (lab'da hisse sayısı × fiyattı). Büyüme verisi SEC'in etiket bazlı uç noktasından (`companyconcept`)
alınır; lab'daki toplu dosyayla aynı kayıtlar. Finviz, SPY barı ya da SEC'e ulaşılamazsa karar ertelenir ve yedek
tetikleyiciler yeniden dener; girişin açılışından sonra verilen kararlar "geç" işaretlenir (değerlendirmeye
girmez). Elenen kırılımlar nedenleriyle kaydedilir (NOT_BULL, NO_GROWTH_DATA, GROWTH_FAIL, TICKER_BUSY).
Portföy değerlemesi gölge raporunun düzenindedir (aynı günün çıkışları girişlerden önce nakde döner).

**Kapı (Ek C düzeni; otomatik terfi yok).** F1: ≥ 24 ay VE ≥ 60 uygun kapanmış işlem. F2: işlem başı net ortalama
(20 bps) > 0 VE giriş günü kümeli bootstrap %95 CI alt sınırı > 0. F3: portföy (yukarıdaki boyut, 20 bps +
1.5$/emir) toplamı > aynı dönemde SPY al-tut. F4: max düşüş > −35%. Durum: F1'e kadar TOPLANIYOR; sonra F2–F4
hepsi → TERFİ ADAYI; net ortalama ≤ 0 → KALDI; 36. ayda hâlâ aday değilse KALDI; aksi halde BELİRSİZ. "TERFİ ADAYI"
yalnız tartışmayı açar; geçmiş sınavda endeksin gerisinde kaldığı o tartışmada kullanıcıya birlikte sunulur.

**Başlangıç.** Gölge koşusunun GRW'yi ilk çalıştırdığı seans. H4/H5/H6/H6S/H6SR/MOM kilitleri ve kapıları değişmez.

# Ek X — TRD: büyüyen şirket + yükselen eğilim, haftalık karar (kırılım kalıbı yok)

**Yazıldığı an: 2026-11-28, Ek R–W sonuçlarından SONRA; bu kural setinin hiçbir sonucu görülmeden.** Görülenler:
Ek R–W'nin tamamı. Özellikle Ek V'nin betimleyici kıyasları: büyüme şartı çıkarılınca havuz farkı sıfıra iniyor
(+13.0% / −58.9%); lider şartı değer katmıyor; "büyüme + kırılım" hesabı iki dönemde de büyütüyor (+36.4% / +64.0%)
ama SPY'ın (+102.0% / +99.2%) gerisinde ve paranın ~%75'i boşta. Ek T: büyük hisselerde aylık 12-1 momentum
rotasyonu 2016–2020 +301%, 2021–26 +22% ve düşüş −71%. 2026-11-28'de 1.834 küçük/orta hissede son 10 seansın VCE
sayımı (toplam 5 isabet) ve "3 ayda %30 düşmüş hisse" haftalık kontrolü (betimleyici; bu kuralın girdisi değil).
Genel piyasa bilgisi (veriden değil): büyüme hisseleri 2016–2020 ve 2023–2025'te çok iyi, 2021–22'de çok kötüydü.

**Gerekçe (kullanıcı kararı, 2026-11-28; Ek T ve Ek V'deki "kapanır" sonuçlarının etiketli istisnası).** Kullanıcı
ürünün tek grafik kalıbına (VCE) bağımlılığını sorun olarak gördü ve "radikal karar" istedi. Önerilen ve kabul
edilen ilke: önce şirket, sonra grafik — yalnız büyüyen şirketler alınır; kırılım/sıkışma kalıbı şart değildir,
yükselen eğilim yeter. Bu, projenin ~19. hipotezidir ve **kirlenmiş bir sınavdır**: iki dönem de daha önce
kullanıldı ve büyüme şartının bu veride işe yaradığı görüldü. GEÇTİ sonucu bile kanıt değildir; geçerli kanıt
yalnız ileri testten gelir. TEK kural, TEK parametre seti; tarama yok. Parametreler Ek V'den aynen (büyüme
tanımı, havuz, %8 stop, %1 risk, maliyet) ya da yayımlanmış standart değerlerdir (50 ve 200 günlük ortalama,
haftalık karar); bu veride ayarlanmadı.

**Evren ve veri.** Ek V ile aynı: kök `output/lab_liq` (yeniden indirilmez), havuz G (fiyat ≥ 10 $, 63 seans
ortalama hacim > 300K, 20 seans ortalama dolar hacim > 10M $, piyasa değeri ≥ 1 milyar $; kıyas ETF'leri hariç),
büyüme verisi `sec_facts/quarters.parquet` (Ek V'de kurulan tablo). İki dönem ayrı ayrı, her biri 10.000$'dan:
**A = 2016-01-04 → 2020-12-31, B = 2021-01-04 → 2026-11-20.**

**Uygunluk (t günü).** Havuz G ∧ büyüme şartı (Ek V ile aynı fonksiyon: son bilinen çeyrekte satış yıllık ≥ %25,
net kâr artmış, veri ≤ 200 gün; bilinmiyorsa ret) ∧ eğilim: **Close_t > SMA50_t VE SMA50_t > SMA200_t.**

**Karar günü.** Her haftanın son seansı (bir sonraki seans başka bir ISO haftasındaysa). Emirler bir sonraki
seansın açılışında gerçekleşir. Karar günleri arasında yalnız stop çalışır.

**Satış.**
- Stop = 0.92 × giriş fiyatı (sabit %8). Giriş günü dahil her gün: Low ≤ stop → min(Open, stop).
- Karar günü t'de eldeki hissede Close_t < SMA50_t → bir sonraki açılışta satış (açılış barı yoksa bar gelene
  kadar beklenir). Süre sınırı ve kâr hedefi yok; büyüme şartının sonradan bozulması tek başına satış sebebi değil.
- Dönem sonunda açık pozisyon son seansın kapanışıyla kapatılır.

**Alım.** Karar günü t'de piyasa rejimi BULL ise (canlı `regime_from_spy_close`, SPY 1 yıl + VIX; Ek V ile aynı):
adaylar = t'de uygun olup elde olmayan hisseler; sıra = 12 aylık getiri (Close_t / Close_{t−252} − 1; hesaplanamıyorsa
0) büyükten küçüğe. Bir sonraki açılışta sırayla alınır; açılış barı yoksa o aday atlanır. Rejim BULL değilse yeni
alım yok; eldekiler kendi satış kuralına kadar tutulur. Satılan hisse sonraki karar günlerinde uygunsa yeniden
alınabilir (bekleme süresi yok). Hisse başına tek açık pozisyon.

**Boyut ve portföy.** 10.000$. Ek V formülü aynen: pozisyon = min(%25 × özsermaye, risk × özsermaye ÷ 0.08),
risk = %1 (→ pozisyon özsermayenin %12.5'i, en çok 8 hisse); özsermaye = önceki gün kapanış değeri, o günün
açılış satışları işlendikten sonra. Son 2 kapanan işlem de (maliyet sonrası) zarardaysa risk %0.5 (→ %6.25),
ilk kârlı kapanışta %1'e döner. Nakit yetmezse aday atlanır; borç yok.
**Maliyet (birincil): 20 bps gidiş-dönüş + Midas 1.5$/emir.** Duyarlılık: 10 ve 50 bps; 25.000$ hesap.

**Ölçütler (her dönemde ayrı).**

| # | Ölçüt |
|---|---|
| X0 | Yeterlilik: her dönemde ≥ 60 portföy işlemi |
| X1 | Para: portföy toplam getirisi > SPY al-tut (dönem başından önceki son kapanış → dönemin son seansı), **iki dönemde de** |
| X2 | Risk: portföy max düşüşü > −35%, iki dönemde de |
| X3 | Seçim: portföy işlemlerinin havuz G'ye göre farkı (aynı giriş açılışı ve aynı çıkış zamanlaması, karar günü havuz G üyelerinin eşit ağırlıklı getirisi) iki dönemde de > 0 VE iki dönem birlikte giriş günü kümeli bootstrap %95 CI alt sınırı > 0 |

**Hüküm (önceden sabit).** GEÇTİ: X0–X3 hepsi. KISMEN: X0 ve X1 sağlanır ama X2 veya X3 sağlanmaz. KALDI: X1
herhangi bir dönemde sağlanmaz. X0 sağlanmazsa "yetersiz veri" (şartlar gevşetilmez).

**Sonuca göre (önceden sabit; kullanıcı kararı 2026-11-28).** Kullanıcı canlı SANAL (paper) ürünün ana kuralının
H6S'den alınmasına karar verdi (H6S iki dönemde de kaldı, Ek R). X0, X1 ve X2 üçü de sağlanırsa (GEÇTİ, ya da X2'si
sağlanan KISMEN) canlı sanal ana kural TRD olur ve TRD gölge ileri teste eklenir. Aksi halde (KALDI, yetersiz veri
ya da X2'si sağlanmayan KISMEN) TRD kapanır, bu veride varyantı denenmez ve canlı sanal ana kural "büyüme + kırılım"
(GRW, Ek W'nin donmuş kuralı) olur. Her iki durumda da hiçbir kural gerçek para adayı sayılmaz; bu ancak ileri test
kapısından sonra ayrı bir karardır. Betimleyici (hükmü ve canlı seçimi DEĞİŞTİRMEZ): büyüme şartsız (havuz G ∧
eğilim), piyasa şartsız, sıra 12 aylık getiri yerine satış büyümesi; yıllık tablo; en iyi 5 işlemin payı; boşta
nakit oranı; yıllık emir sayısı; IWM kıyası.

**Bilinen yanlılıklar.** Kirlenme (yukarıda). Survivorship (evren 2026'da listede olanlar; hisse → CIK eşlemesi
güncel) → mutlak getiri iyimser. Dosyalama tarihi basın açıklamasından geç olduğundan büyüme bilgisi gerçekte
olduğundan geç kullanılır (temkinli yön). Günlük barda stop, stop fiyatından (boşlukta açılıştan) gerçekleşmiş
sayılır. Haftalık karar + ertesi açılış, elle (Midas) uygulanabilir bir tempodur; gün içi giriş sınanmaz.

# Ek X sonucu (2026-11-28, SONUÇ GÖRÜLDÜKTEN SONRA yazıldı)

**Hüküm: KALDI** (X1 iki dönemde de sağlanmadı; X2 dönem B'de sağlanmadı). Ön kayıt bölüm SHA256
`22856e87eeb941b4f88de64f0a34475017b5a76a4f73b7b0590a31cc29e94b7d`; koşu `python lab/trend_x.py run`,
çıktı `output/lab_liq/x_result.json`.

| | A 2016–2020 | B 2021–2026 |
|---|---|---|
| Portföy | **+44.3%** | **−68.2%** |
| Max düşüş | −25.7% | **−76.8%** |
| SPY al-tut (IWM) | +102.0% (+86.4%) | +99.2% (+40.1%) |
| İşlem / kazanma | 314 / %33.4 | 578 / %18.2 |
| Ort. kazanç / kayıp | +20.9% / −6.8% | +29.9% / −7.7% |
| Stop ile çıkan | 143 / 314 | 412 / 578 |
| Havuz farkı [CI] | +1.18 [−0.68, +3.18] | −0.99 [−2.97, +1.21] |
| Boşta nakit | %31 | %37 |

İki dönem birlikte n = 892, havuz farkı −0.23 [−1.70, +1.42]. X0 ✔, X1 ✘, X2 ✘, X3 ✘. Duyarlılık (10 bps / 50 bps /
25.000$): A +52.7 / +49.8 / +48.1; B −67.4 / −71.0 / −61.4. Betimleyici (hükmü değiştirmez): büyüme şartsız
−20.6% / −81.8%; piyasa şartsız +48.7% (düşüş −37.4) / −62.6%; satış büyümesi sırası +44.7% / −53.7%.

**Okuma.** Nakit sorunu çözüldü (boşta %75 → %31–37) ama para gelmedi: sıra 12 aylık getiriye göre olduğundan
kural en oynak hisseleri seçti ve sabit %8 stop dönem B'de işlemlerin %71'inde çalıştı (işlemlerin %43'ü ilk 5
seansta kapandı). Kırılım şartını kaldırmak "büyüme"nin seçim avantajını da götürdü (havuz farkı ≈ 0). Büyüme şartı
yine en az kötü parça (büyüme şartsız sürüm iki dönemde de daha kötü).

**Sonuç (önceden sabit kural uygulandı).** TRD kapanır; bu veride varyantı denenmez (stop genişletme, farklı sıra
ya da farklı ortalama denemek = sonradan seçme). Kullanıcı kararı (2026-11-28) gereği canlı SANAL ürünün ana kuralı
H6S'den "büyüme + kırılım"a (GRW, Ek W'nin donmuş kuralı) geçer. GRW gerçek para adayı değildir; kanıt yalnız
gölge ileri testten gelebilir.

# Ek Y — RDG: GRW sinyallerine "okuma notu" ileri testi (geçmiş veriyle sınanmaz)

**Yazıldığı an:** 2026-11-28, kullanıcı kararıyla ("okuma notlarını başlat"), hiçbir okuma notu üretilmeden ve hiçbir
sonuç görülmeden.

**Neden.** 10 işlemlik elle kontrol tablosunda (GRW lab işlemlerinden rastgele seçim) kuralın neredeyse satışı olmayan
şirketleri de "büyüyen" saydığı görüldü (ör. LWLG: çeyrek satışı 0.16 M$, bir yıl önce 0.02 M$ → "+%595"). Soru:
şirketi okuyan bir yorum (ne satıyor, büyüme gerçek mi, haberler, sektör), GRW'nin işlemlerini para açısından
ayırabilir mi?

**Neden geçmişte sınanmaz.** Dil modeli geçmişte ne olduğunu (hissenin sonradan ne yaptığını) biliyor olabilir;
geriye dönük her okuma testi sonucu görmüş sayılır. Bu yüzden yalnız ileri test: not, girişin açılışından önce
yazılır ve sonuç ondan sonra oluşur. Modelin bilgi kesim tarihi sinyal gününden öncedir.

**Kapsam.** Gölge defterde GRW için PENDING yazılan her kayıt (yalnız SPY BULL iken oluşur). GRW kuralı, girişleri,
çıkışları ve boyutu DEĞİŞMEZ; okuma yalnız kayda yazılır, canlı sanal ürünün alımlarını etkilemez.

**Girdi (sabit).** Şirket adı, sektör, sanayi, iş tanımı (ilk 1.200 karakter), piyasa değeri, çalışan sayısı
(yfinance); SEC büyüme verisi (son çeyrek sonu, satış ve bir yıl önceki satış, satış büyümesi %, net kâr/zarar ve
bir yıl önceki); sinyal günü kapanışı, kırılım günü dibi, 12 aylık getiri; yayım zamanı okuma anından önce olan en
çok 10 haber başlığı. Model internete erişmez; yalnız bu kartı ve kendi bilgisini kullanır.

**Model ve talimat.** Canlı sunucunun dil modeli istemcisi (sağlayıcı ve model adı her kayda yazılır), sıcaklık 0.
Talimat metni `swing_trader/shadow/reading.py` içinde sabittir; SHA256'sı (`READ_PROMPT_SHA256`) her kayda yazılır.
Talimat ya da girdi listesi değişirse bu ek kapanır ve yeni ek yazılır.

**Çıktı.** JSON: `verdict` ∈ {AL, ATLA}, `reasons` (en çok 3 kısa madde), `red_flags` (en çok 3). Model yoksa,
yanıt ayrıştırılamazsa ya da hata olursa okuma "yok" sayılır (nedeni kayda yazılır) ve değerlendirmeye girmez.

**Zaman.** Okuma, girişin açılışından (09:30 ET) önce yazılmalıdır; sonra yazılan okuma `late` işaretlenir ve
değerlendirmeye girmez. Okunamayan kayıt sonraki koşularda yeniden denenir. Okuma eksikliği koşu sağlığını
"eksik" yapmaz (yedek tetikleyiciler bunun için döngüye girmez).

**Değerlendirme.** İlk kez, zamanında okunmuş ve kapanmış GRW işlemi ≥ 40 VE her grupta (AL, ATLA) ≥ 10 olunca;
en geç 2028-06-30'da, o güne kadar ne varsa (yeterlilik sağlanmadıysa yalnız betimleyici).
- **Y1:** AL grubunun işlem başı ortalama net getirisi > ATLA grubunun.
- **Y2:** ATLA grubunun işlem başı ortalama net getirisi < 0 (okumanın elediği işlemler gerçekten para kaybettirdi).
- **Hüküm:** Y1 ∧ Y2 → GEÇTİ: okumanın canlı sanal ürüne süzgeç olarak eklenmesi kullanıcıya önerilir (kanıt değil
  aday; örneklem küçük). Aksi → KALDI: okuma canlıya eklenmez; talimat ya da model değiştirilip yeniden denenmez
  (sonradan seçme), yeni bir okuma fikri ancak yeni ekle ve farklı türden olursa denenir.
- ATLA grubu 60 kapanmış işlemde de 10'a ulaşmazsa: "okuma ayırt etmiyor" → KALDI.

**Bilinen sınırlar.** Örneklem küçük (lab'da yılda ~45 işlem; piyasa BULL değilken hiç yok). Haber başlıkları
yfinance'ın döndürdüğüyle sınırlı. Model sağlayıcısı sunucu ayarına bağlıdır; sağlayıcı ya da model değişirse kayıtlar
ayrı tutulur ve hüküm yalnız ilk (en çok kayıtlı) model için verilir.
