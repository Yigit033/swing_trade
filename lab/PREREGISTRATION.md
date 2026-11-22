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
