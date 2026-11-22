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
