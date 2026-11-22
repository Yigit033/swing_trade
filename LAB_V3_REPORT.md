# Lab v3 — Canlı Ürünün Birebir Replikası + Ön Kayıtlı Alternatifler

**22 Kasım 2026.** Production kodu ve production DB değiştirilmedi. Kod: [lab/](lab/).
Ön kayıt: [lab/PREREGISTRATION.md](lab/PREREGISTRATION.md), SHA256
`1cd64f8aac6a3dc47058d0b3a53c339e318e18dc1a54769c45c380c44a5370b9`. Bu dosya sonuçlardan
önce yazıldı; parametreler sonradan değiştirilmedi.

## 1. Kısa sonuç

**Bugünkü canlı ürün, 2021–2026 boyunca maliyet sonrası para kaybettirirdi. Aynı
günlerde rastgele hisse almaktan da iyi değil.** Ön kayıtlı üç alternatif strateji
de kriterleri geçemedi. Bu veriyle günlük-bar, long-only, fiyat/hacim tabanlı
small/mid-cap swing stratejisinde kanıtlanabilir bir edge bulunamadı.

| Strateji | İşlem | Net ort. (50 bps) | Medyan | Win | PF | Havuza göre fark (95% CI) | Portföy (%10/işlem) | Max DD | Terfi |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| **H0 canlı ürün** | 345 | **−0.46%** | −0.54% | 46% | 0.87 | −0.18 [−1.16, +0.74] | **−18.7%** | −28.6% | ❌ |
| H1 gap-and-hold | 2,337 | −1.68% | −5.49% | 32% | 0.78 | −1.36 [−2.27, −0.50] | −95.7% | −96.3% | ❌ |
| H2 trend içi RSI2 dönüşü | 14,145 | −0.21% | +0.59% | 59% | 0.89 | +0.08 [−0.06, +0.21] | −60.5% | −63.5% | ❌ |
| H3 52-hafta zirvesi + RS | 1,702 | −0.52% | −4.79% | 42% | 0.92 | −0.29 [−1.18, +0.65] | −69.6% | −71.2% | ❌ |

Aynı dönemde al-tut getirisi: **IWM +42%, IJR +57%, SPY +102%**.

> **Düzeltme (Ek D1, 22 Kasım 2026):** lab'ın bilanço tarihleri H0 sinyallerinin %31'inde
> (168/539) indirme hatası yüzünden boştu; bilanço kapısı bu sinyallerde çalışmamıştı. Kapı
> düzgün çalışınca **H0: 331 işlem, net −0.24%/işlem, havuz farkı +0.00 [−0.97, +0.97],
> portföy −11.8%, max DD −22.3%.** Sonuç değişmiyor: ürün maliyet sonrası kaybettiriyor ve
> endeksin çok gerisinde.

H0 canlı pozisyon boyutuyla (10.000$ hesap, işlem başı ortalama 1.396$) 6 yılda
**brüt +1.297$, 50 bps maliyetle −1.111$**. Paper trading maliyet kesmediği için
ekranda hafif kârlı görünebilir; gerçek işlem maliyeti bu sonucu tersine çevirir.

**Sonuçlar iyimser tarafta.** Evren bugün listede olan hisselerden oluşuyor; 2021–2026
arasında delist olan hisseler yok (survivorship). Gerçek sonuç daha kötü olur. Havuz
farkı bu yanlılığın bir kısmını nötrler.

## 2. Replikanın canlıya sadakati (ölçüldü)

Replika, production fonksiyonlarını doğrudan çağırıyor: `regime_from_spy_close`,
`SmallCapEngine.scan_stock`, `effective_scan_thresholds`, `auto_entry_capacity`,
tracker onay matematiği (birim testle canlı tracker'a eşitlendi) ve
`execution.evaluate`. Ayarlar, canlı Supabase yamasının 11 Ekim kopyası.

22 kayıtlı canlı tarama günüyle karşılaştırma ([lab/parity.py](lab/parity.py)):

| Katman | Sonuç |
|---|---|
| Finviz evren emülasyonu | recall %88.8, precision %90.3 |
| Motor kararı (aynı hisse, aynı gün) | **%98.3** aynı (1.296/1.319) |
| Kalite skoru | medyan fark **0.0**, %95.5'i ±2 puan içinde |
| Son aylardaki gerçek canlı işlemler | KYMR, ASST, SAIC, SEI replikada da aynı günlerde seçiliyor |

Bilinen sapmalar: geçmiş Finviz cap verisi yok (yfinance hisse sayısı × fiyat ile
yaklaşık), bilanço tarihleri yfinance (sinyal hisselerinin 415/418'inde mevcut),
delist hisseler yok, fill'ler günlük OHLC varsayımı.

## 3. Para nereye gidiyor (H0, tanı amaçlı; strateji değişikliği gerekçesi DEĞİL)

| Kırılım | n | Net ort. | Toplam net katkı |
|---|---:|---:|---:|
| STOPPED | 75 | −11.74% | −881 puan |
| TIMEOUT | 224 | +2.30% | +515 puan |
| TRAILED | 41 | +1.10% | +45 puan |
| TARGET | 5 | +32.2% | +161 puan |
| RVOL thrust pathway | 222 | −0.72% | −160 puan |
| VCE pathway | 123 | 0.00% | 0 puan |

- T1'e yalnızca işlemlerin **%7.8'i** ulaşıyor; ortalama R **+0.05**. Stop'a değen
  işlemler ortalama −%11.7 kaybediyor (2.5×ATR, %14–16 tavan). Kazananlar hedefe
  değil, çoğunlukla süre dolunca küçük kârla çıkıyor.
- 6 yılın 3'ünde net negatif (2021 −1.87%, 2022 −3.17%, 2025 −1.46%).
- Bu kırılımlar sonuç görüldükten sonra yapıldı. Bunlara bakıp eşik veya stop
  ayarlamak yeniden curve-fit olur. Ancak yeni ön kayıtla, ayrı veride test edilebilir.

## 4. Bu çalışmada bulunan production hataları

| # | Hata | Etki |
|---|---|---|
| 1 | Finviz `Overview` görünümünde **Float sütunu yok**. `get_ticker_metadata` her hisse için `floatShares=0` döndürüyor. | Kalite skorunun %25 ağırlıklı float bileşeni canlıda **sabit 5 puan**; float hiç ayırt etmiyor. Kayıtlı taramalarla doğrulandı: skor farkları tam float bant adımlarıydı. |
| 2 | `market_calendar` 2025-01-09'u (ulusal yas günü, NYSE kapalı) işlem günü sayıyor. | O gün açık olan pozisyonlar `REVIEW_REQUIRED: MISSING_SESSION` ile donar. Geçmiş, ama takvim tablosu eksik. |
| 3 | `smallcap_backtest.py` canlıyı ölçmüyor: RSI kapısını VCE'ye de uyguluyor, `rank_score` yerine `quality_score` ile sıralıyor, canlıda olmayan tip-özel eşikler kullanıyor, `scan_stock`'a rejim geçirmiyor, sahte cap/float sabitleri kullanıyor. | Bu motorla alınan geçmiş kararlar canlı ürünü yansıtmıyordu. |

Hiçbiri bu çalışmada düzeltilmedi (production'a dokunulmadı).

## 5. Tekrar üretim

```text
python lab/fetch_ohlcv.py                      # 5.237 hisse, 2020-06 → 2026-11-20
python lab/fetch_facts.py <tickers>            # hisse sayısı/ülke/float; --earnings
python lab/h0_signals.py --universe recorded   # parite girdisi
python lab/parity.py
python lab/h0_signals.py --universe emulated   # 154.878 tarama, 539 sinyal
python lab/h0_sim.py                           # seçim + onay + production çıkışı
python lab/challengers.py
python lab/evaluate.py                         # output/lab/report.json
python -m pytest tests/test_lab_v3.py          # 10 test (onay paritesi dahil)
```

---

## Ek A — Bilgi tabanlı swing sinyalleri (H4, H5)

Ön kayıt: `lab/PREREGISTRATION.md` Ek A (SHA256 `4c055d62…`). Veri indirilmeden önce yazıldı.
Kod: [lab/info_signals.py](lab/info_signals.py).

| Strateji | İşlem | Net ort. | Havuza göre fark (95% CI) | Pozitif yıl | Portföy | Max DD | Terfi |
|---|---:|---:|---|---:|---:|---:|---|
| H4 içeriden alım kümesi (SEC Form 4, 2021–2026Q1) | 935 | +0.23% | +0.19 [−0.53, +0.94] | 3/6 | −7.7% | −41.9% | ❌ |
| H5 bilanço sürprizi + pozitif tepki | 6,678 | −0.26% | **+0.27 [−0.02, +0.57]** | **5/6** | −51.1% | −62.8% | ❌ |

- **İkisi de havuzu brüt olarak yeniyor** (H0–H3'te yalnız H2 marjinal yeniyordu), ama fark
  (0.2–0.3 puan) 50 bps işlem maliyetinden küçük. Sonuç maliyet sonrası negatif.
- H5 en tutarlı sinyal: 6 yılın 5'inde havuzu geçti, CI alt sınırı −0.02 (sınırda).
- Portföyde nakit kısıtı yüzünden alınabilen işlemlerin ortalaması daha kötü
  (H4 −0.36%, H5 −0.73%). Avantaj kırılgan; hangi işlemin alındığına bağlı.
- Sonuç görüldükten sonra yapılan duyarlılık analizi (kanıt değil): H4 + ürünün mevcut BEAR
  giriş yasağı → net +0.75%, fark +0.42 [−0.41, +1.21], portföy yine −18%.
- Veri notu: SEC setinde birkaç hatalı fiyat girişi var (ör. 7 trilyon dolarlık "alım"). Kural
  gereği temizlenmedi; temizlenince sonuç değişmiyor (+0.24%).

---

## Ek B — Üç production hatasının düzeltilmesi (2026-11-22)

### B1. Float (ön kayıt Ek B ile karar verildi)

Düzeltmenin yönü sonuç görülmeden yazılan kuralla ölçüldü: aynı canlı replika, tek fark
`scan_stock`'a gerçek (yaklaşık zaman-noktası) float verilmesi.

| | İşlem | Net/işlem (50 bps) | Havuza göre fark | Portföy | Max DD |
|---|---:|---:|---|---:|---:|
| Bugünkü (float bilinmiyor) | 345 | −0.46% | −0.18 [−1.16, +0.74] | −18.7% | −28.6% |
| H0F: float çalışıyor | 305 | **−0.97%** | −0.72 [−1.95, +0.50] | **−29.3%** | **−42.7%** |

Gerçek float ürünü kötüleştiriyor; kurala göre float skora **bağlanmadı**. Kod artık bunu
açıkça yapıyor: `scoring.calculate_quality_score` float girdisini yok sayıp sabit nötr ham
puanı kullanıyor (seçim bugünküyle birebir aynı, kalibre eşikler korunuyor). Yan etki olarak
iki tutarsızlık kapandı: `/lookup` yfinance float'ıyla farklı skor üretiyordu, backtest modu
45M sahte float uyduruyordu (canlıdan +6.25 puan).

### B2. Takvim

`market_calendar` tek seferlik NYSE kapanışlarını (2012 Sandy ×2, 2018-12-05, 2025-01-09)
içeriyor. Ayrıca NYSE kuralı açık hale getirildi: 1 Ocak Cumartesi ise önceki Cuma açık.
Testler yeni kodda geçiyor, eski kodda kırılıyor (negatif kontrol).

### B3. Backtest motoru canlıyı ölçüyor

`smallcap_backtest.py` yeniden yazıldı; her adım canlının kendi fonksiyonunu çağırıyor:
onay `tracker.confirm_entry_levels`/`gap_percent` (tracker'dan çıkarıldı, canlı da bunu
kullanıyor), çıkış `execution.evaluate`, seçim `select_scan_signals` (canlı tarayıcı da bunu
kullanıyor), kapasite `auto_entry_capacity`, rejim 1 yıllık SPY geçmişiyle. Canlıda karşılığı
olmayan kurallar silindi: VCE'ye uygulanan RSI kapısı, kalite sıralaması, tipe özgü tabanlar,
EMA trend filtresi, Tip C açılış kontrolü, %7 stop tabanı, swing-low stop, drawdown
duraklatması, 2 kayıptan sonra kalıcı yasak, 45M sahte float. 13 ölü ayar modelden, kayıtlı
yamanın budama listesinden ve Ayarlar sayfasından kaldırıldı. Backtest sayfası varsayılan
olarak "canlı ürünle birebir" çalışıyor (manuel kısıtlar isteğe bağlı).

Doğrulama: aynı 16 hisse, 2026-06 → 11: KYMR, ASST, SEI işlemleri lab replikasıyla aynı
giriş, stop, çıkış günü ve getiri (fark yalnız 5 bps slippage). İlk denemede ASST bir gün
kayıktı; kök neden backtest'in rejimi 7 aylık SPY geçmişiyle (canlı: 1 yıl) hesaplamasıydı.
Düzeltildi ve regresyon testi eklendi.

### Yeni bulgu: bilanço kapısı veri hatasında açık kalıyor (Ek D1'de DÜZELTİLDİ)

Canlı SAIC işlemi (sinyal 2026-08-31) bilanço gününde açıldı: SAIC 31 Ağustos 07:00 ET'de
açıkladı, kural 0–3 gün içindeki bilançoyu reddetmeli. `filters.check_earnings`, yfinance
boş/hatalı döndüğünde "geç" diyor (fail-open). Lab'ın SAIC bilanço dosyası da boştu, bu yüzden
replika canlıyla aynı davranışı gösterdi. Bugünkü veriyle backtest SAIC'i doğru şekilde reddediyor.
Düzeltme (yeniden deneme ya da fail-closed) ayrı bir karar.

## Ek C — H4 / H5 gölge modu (ileri test)

`swing_trader/shadow/` (kurallar, veri, defter, koşucu, rapor), `api/routers/shadow.py`,
`/shadow` sayfası, `.github/workflows/shadow-run.yml` (açılış öncesi 07:15/08:15 ET cron).
Ana ürünün işlemlerinden ayrı `shadow_*` tabloları; gerçek para yok; terfi otomatik değil.
Kurallar ve terfi kapısı (F1–F4) `lab/PREREGISTRATION.md` Ek C'de, sonuç görülmeden yazıldı.
`PARAMS_SHA256` değişirse koşucu yeni sinyal üretmeyi durdurur.

Lab paritesi (production kural fonksiyonları, lab verisi üzerinde):

| Kontrol | Sonuç |
|---|---|
| H5 işlemleri (400 rastgele) | giriş/çıkış günü, fiyatı, nedeni birebir |
| H4 işlemleri (250 rastgele) | birebir |
| H5 olayları (300 hisse) | aynı küme |
| H4 olayları (tümü, 6.013) | aynı küme |

Not: SEC verisinde bazı semboller parantezli ("(CALX)"). Lab bunları evren dışı saydı;
ön kayda sadakat için production da aynısını yapıyor.

---

## Ek D — Bilanço kapısı düzeltmesi (D1) ve büyük hisse deneyi (D2)

Ön kayıt: [lab/PREREGISTRATION.md](lab/PREREGISTRATION.md) Ek D, sonuçlardan önce yazıldı
(SHA256 `fe4614b7d1c68993e97063eeae7f3a3a20499339c1a12ecf34379f17237e3b59`). Sonradan eklenen
**D1 notu** etiketli bir sapmadır (aşağıda); notla birlikte dosyanın SHA256'sı
`3d5979d80dbafc21968a3517ed6b23761fb489debbffd315a70554872fa35c9d`.

### D1 — Bilanço kapısı

**Veri hatası (sonuç kısmen görüldükten sonra bulundu, ön kayda etiketli not olarak yazıldı).**
`output/lab/earnings` dosyalarının 130'u indirme sırasında Yahoo kısıtlamasıyla boş yazılmıştı;
H0 sinyallerinin 168/539'unda kapı veri yokluğundan geçiyordu. Aynı fonksiyonla çekilmiş
`earnings_full` dosyaları ortak 1.281 hissede birebir aynı tarihleri veriyor (0 uyuşmazlık);
lab artık boş dosyada ona düşüyor. Canlı SAIC'in "replikada da aynı" çıkması bu iki ayrı veri
hatasının çakışmasıydı.

| Onarılmış veri | İşlem | Net/işlem (50 bps) | Havuz farkı (95% CI) | Portföy | Max DD |
|---|---:|---:|---|---:|---:|
| **H0** kapı çalışıyor (veri yoksa geç) | 331 | **−0.24%** | +0.00 [−0.97, +0.97] | **−11.8%** | −22.3% |
| H0EC veri yoksa ret (düzeltme) | 328 | −0.31% | −0.04 [−1.02, +0.91] | −13.6% | −23.9% |
| H0E0 kapı kapalı (çeldirici) | 356 | −0.46% | −0.17 [−1.17, +0.80] | −19.1% | −29.2% |
| *Onarımsız H0 (Lab v3 ilk rapor)* | 345 | −0.46% | −0.18 [−1.16, +0.74] | −18.7% | −28.6% |

**Karar (ön kayıtlı kural):** kapı kapalıyken sonuç her iki ölçütte de kötü → kapı para
kazandırıyor (+0.22 puan/işlem, +7 puan portföy) → fail-closed düzeltmesi uygulandı. H0EC
yalnız 5 sinyali etkiliyor (3 işlem), karar ölçütü değil.

**Float kararı yeniden kontrol edildi:** onarılmış veriyle H0F 294 işlem, net −0.73%, portföy
−23.4% (H0: −0.24%, −11.8%) → gerçek float yine kötü, Ek B kararı geçerli.

**Production düzeltmesi** (`filters.check_earnings`, `data/earnings_calendar.py`):

| Durum | Eski | Yeni |
|---|---|---|
| Hissenin tarihleri yakında bilanço gösteriyor | ret | ret |
| Hissenin tarihleri boş/hata, takvim bilanço gösteriyor (SAIC) | **geç** | ret |
| İkisi de bilinmiyor | **geç** | ret (sebep: "Earnings date unverified") |
| Biri biliniyor ve yakında bilanço yok | geç | geç |
| Beklenmedik hata | **geç** | ret |

Takvim tarama başına bir kez çekiliyor (canlı deneme: 335 şirket, 1.9 sn); hisse başına süre
değişmedi. Ürünün backtest'i bilanço verisi olmayan hissenin sinyalini eliyor ve sayıyor
(`skip_earnings_unknown`). Açık liste veren çağıranlar (discovery, lab, testler) değişmedi.
Testler: `swing_trader/tests/test_earnings_gate.py` (21); eski kodda SAIC durumu "geç" veriyor.

### D2 — Büyük hisseler (≥10B$, 10 bps maliyet, SPY kıyası)

Havuz L: günde ortalama 665 hisse (1.218 farklı). Eksik 30 bilanço dosyası (hepsi 250$ üstü
hisseler) bir kez indirildi; 2 yeni halka arz hâlâ boş, dışlandı.

| 2021–2026 | H4L içeriden alım | H5L bilanço sürprizi |
|---|---:|---:|
| İşlem / ay | 340 / 63 | 3.474 / 71 |
| Brüt ortalama | +0.65% | +0.25% |
| Net 10 / 20 / 50 bps | +0.55 / +0.45 / +0.15% | +0.15 / +0.05 / −0.25% |
| Aynı giriş-çıkışta SPY | +0.56% | +0.61% |
| Havuz L farkı (95% CI) | +0.32 [−0.58, +1.28] | −0.05 [−0.36, +0.27] |
| En iyi %5 çıkarılınca (10 bps) | −0.68% | −1.18% |
| Pozitif fark yılı | 3/6 | 3/6 |
| Portföy %10/işlem (10 bps) | +22.4% (DD −18.7%) | −23.5% (DD −37.9%) |
| SPY al-tut, aynı dönem | +103.7% | +95.3% |
| Kriter | P2, P3, P4 ❌ | P2, P3, P4, P6 ❌ |

**Sonuç: edge bulunamadı.** Ön kayıttaki karşı argüman doğru çıktı: düşük maliyet işe yaramıyor
çünkü edge büyük hisselerde kayboluyor. H4L'nin pozitif getirisi aynı günlerdeki SPY getirisine
eşit (piyasa betası) ve birkaç büyük kazançtan geliyor. Ön kayıt gereği parametre taraması yok,
gölge moda eklenmedi.

Tekrar üretim: `python lab/h0_sim.py --earnings {live,closed,off}` → `python lab/d1_eval.py`;
`python lab/info_signals.py H4 H5 --pool large` → `python lab/d2_eval.py`.

---

## Ek E — Pipeline tanısı: hangi katman para kazandırıyor? (22 Kasım 2026)

**Tanı.** Sonuçlar görüldükten sonra yapıldı; buradaki bulgulara bakıp ayar değiştirmek eğri
uydurma olur (yeni ön kayıt + görülmemiş veri gerekir). Kod: [lab/pipeline_audit.py](lab/pipeline_audit.py).

Ölçü: sinyal günü d → d+1 açılışta al, d+8 kapanışta sat (canlı ortanca tutuş 8 seans); aynı
günkü havuz ortalamasına göre **fark** (puan, maliyet hariç). Bir katman işe yarıyorsa elediği
hisseler geçirdiklerinden kötü gitmeli.

| Katman | Geçen n | Geçen fark | Elenen n | Elenen fark | Yorum |
|---|---:|---:|---:|---:|---|
| Canlı evren (Finviz emülasyonu) | 154.431 | **+0.14** [+0.05, +0.23] | — | — | küçük ama gerçek |
| Sert filtreler (fiyat/cap/dolar hacim) | 146.131 | +0.15 | 8.300 | −0.08 | doğru yönde |
| Tetik (VCE / RVOL) | 7.796 | **+0.46** | 138.335 | +0.13 | +0.33 değer |
| Swing onayı | 6.974 | +0.52 | 822 | −0.08 | doğru yönde |
| RSI kapısı | 5.132 | +0.49 | 1.842 | +0.60 | etkisiz |
| Weinstein aşama kapısı | 3.582 | +0.32 | 1.550 | **+0.89** | **ters yönde** (iyileri eliyor) |
| Kalite eşiği | 539 | +0.49 | 3.043 | +0.29 | zayıf |
| Bilanço kapısı | 478 | +0.08 | 61 | +3.71 [−0.4, +8.1] | belirsiz* |
| Rejim eşiği + top N | 375 | −0.14 | 103 | +0.88 | ters yönde |
| Kapasite / aynı hisse / cooldown | 351 | −0.10 | 24 | −0.71 | gürültü |
| Gap filtresi | 328 | +0.15 | 23 | −3.69 | doğru yönde |

Ters/etkisiz bulguların hiçbiri istatistiksel olarak anlamlı değil; yön bilgisi olarak okunmalı.
*Bilanço kapısı: tam sistemde (D1) +0.22 puan/işlem kazandırdı, ama elediği 61 sinyal tek başına
iyi gitti; değeri belirsiz, kural kararı (ön kayıtlı) geçerli.

**Huninin sonunda açılan işlemler havuzu 8 seansta ancak +0.15 puan geçiyor; 50 bps maliyet
bunun 3 katı.** Edge'in neredeyse tamamı evren seçimi ve tetikte; sonraki katmanlar eklemiyor.

**Çıkış motoru** (aynı 331 giriş, brüt ortalama): gerçekleşen **+0.26%**; stop/hedef olmadan
8 seans tutmak +0.48%, tip süresi kadar tutmak +0.49%. Stop'lanan 69 işlem −10.96% (sabit
tutuşta −9.45%), trail'lenen 39 işlem +1.91% (sabit tutuşta +5.22%); 5 hedef işlemi +32.7%
(sabit +16.4%). Çıkış mekanizması işlem başına ~0.2 puan kaybettiriyor.

**Kalite skoru sıralama yapmıyor** (8 seans fark): <60 +0.76, 60–70 +0.26, 70–78 +0.21,
78–82 −0.46, 82–86 +2.24, 86+ +0.14 — tekdüze değil. (Eski "Q80+ çok daha iyi" bulgusu kusurlu
backtest'le ölçülmüştü.)

**Tetik yolu:** VCE +1.62 [−1.3, +4.3] (211 sinyal), RVOL −0.24 [−1.4, +1.0] (328 sinyal). VCE
yönü iyi ama kanıtsız; RVOL değer katmıyor. **Yıllar:** −0.6, +2.7, +2.8, −0.8, −2.0, +2.5 —
kararsız.

**Ürün ölçüm sorunu:** paper trading işlem maliyeti düşmüyor (yalnız stop gap kayması); ekrandaki
kâr/zarar brüt, gerçek paradan işlem başına ~0.5 puan iyimser.

---

## Ek F — H6 "Sade VCE" ve bu turun production düzeltmeleri (22 Kasım 2026)

### F1. Canlı evren bozuktu: ürün Finviz yerine bayat statik listeyi tarıyordu

| Dönem | Production'daki finvizfinance | Sonuç |
|---|---|---|
| 11 Eki deploy'u → bugün | 1.5.0 | Finviz sitesini değiştirince (Kasım başı; kütüphane 10 Kasım'da uyum sağladı) sorgular **hata vermeden boş** döndü → evren sessizce 234 hisselik statik liste |
| 22 Kas deploy'ları | 1.7.1 (sürüm sabit değildi) | Kütüphane yeni siteyi okuyor ama bizim Temmuz yamamız tablo okuyucusunu ezip **her sorguyu** düşürdü → yine statik liste |

Statik liste ölçülmüş canlı evren değil; delist olmuş hisseler (GATO, JNPR, SLCA…) bile içinde.
Kasım'daki canlı paper işlemleri büyük olasılıkla bu listeden geldi. Kesin başlangıç tarihi Fly
loglarında ("Finviz returned only 0 tickers") görülebilir; burada doğrulanmadı.

**Düzeltme:** yama sürüme duyarlı (yeni kütüphanede kütüphanenin kendi okuyucusu; ticker'lar doğrulanıyor),
`finvizfinance==1.7.1` sabitlendi, Finviz'den hiç hisse gelmezse tarama **durur**
(`universe_unavailable`, arayüzde kırmızı uyarı; otomatik tarama tekrar dener). Gerçek veriyle
doğrulandı: 20 Kasım evreni Finviz'den hatasız geldi.

**Açık konu (değiştirilmedi):** Finviz 30'dan az hisse döndürdüğünde (20 Kasım: 18) evren statik
listeyle birleştiriliyor (`min_finviz_tickers_skip_static_merge`). Durgun günlerde taranan evrenin
çoğu ölçülmemiş statik liste oluyor. Lab replikası bu birleşimi içermiyor; ölçülmeden değiştirilmedi.

### F2. Ekrandaki kâr/zarar maliyetsizdi

Paper fiyatları maliyet içermiyor. Performans ve ana sayfa artık **net** (maliyet sonrası) sonucu
ana rakam olarak gösteriyor; brüt alt bilgi. Maliyet tek ayardan: `slippage_bps_per_side`
(backtest de aynısını düşer). Varsayılan 5 → **10 bps/kenar** (önceki proxy ölçümü gidiş-dönüş
~0.19 puan; 5 ölçümsüzdü). Canlıdaki kayıtlı ayar yaması 5'i sabitliyorsa Ayarlar'dan değişir.

### F3. H6 — sade VCE rakibi (ön kayıt Ek F, sonuçtan önce)

Canlı ürünün en küçük çekirdeği: canlı evren + sert filtreler + bilanço kapısı + VCE tetiği, giriş
ertesi açılış, 8. seans kapanışında çıkış. RVOL, RSI, Weinstein, skor, rejim, top N, gap, stop yok.

**Lab sağlık kontrolü (etiketli in-sample — fikir bu veriden çıktı, karar DEĞİL):**

| 2021–2026 | H0 canlı ürün | H6 sade VCE |
|---|---:|---:|
| İşlem (ay) | 331 | 2.066 (70) |
| Net/işlem 50 bps / 20 bps | −0.24% / — | **+0.10% / +0.40%** |
| Havuz farkı (95% CI) | +0.00 [−0.97, +0.97] | +0.46 [−0.26, +1.23] |
| Pozitif fark yılı | — | 4/6 |
| Portföy %10/işlem (50 bps) | −11.8% | **−34.1%** |
| Max DD | −22.3% | **−75.6%** |

İşlem başına H6 daha iyi, portföy olarak çok daha kötü: stop ve rejim freni olmadan aynı anda çok
sayıda korelasyonlu işleme giriyor; 2022'de birlikte kaybediyor (yıl −3.18%/işlem). Ön kayıttaki
sağlık kuralı (brüt fark > 0) sağlandı → gölgeye alındı. Beklenti: F4 (DD < %35) büyük engel.

**Gölge:** `swing_trader/shadow` H6 adımı canlı ürünün kendi fonksiyonlarını çağırıyor; H4/H5
kilidi (`178c293e…`) değişmedi, H6 ayrı kilitli (`86a8947e…`). Production çıkış kuralı lab'daki
300 H6 işlemini birebir üretiyor. Gölge sayfası aynı dönemde canlı ürünün paper işlemlerini aynı
50 bps maliyetle yan yana gösteriyor. Kapı: ≥6 ay, ≥100 işlem, net > 0 (CI), portföy > IWM, DD < %35.

---

## Ek G — Durgun gün statik liste birleşimi (22 Kasım 2026)

Canlı evren, Finviz 30'dan az hisse döndürdüğünde statik listeyle birleşiyor. 2021–2026'da bu
günlerin oranı **%14** (201 gün). Ön kayıt Ek G (sonuçtan önce) karar kuralıyla ölçüldü.

| | Taranan | Sinyal | Sinyal oranı |
|---|---:|---:|---:|
| Durgun günlerde Finviz hisseleri | 3.616 | 19 | %0.53 |
| Durgun günlerde eklenen statik hisseler | 40.181 | **3** | %0.008 |

Statik hisselerin %30'u sert filtrede (büyük cap / yüksek fiyat / düşük hacim), %70'i tetikte
eleniyor: önceden süzülmemiş isimler neredeyse hiç kırılım vermiyor.

| | İşlem | Net/işlem (50 bps) | Portföy | Max DD |
|---|---:|---:|---:|---:|
| H0 (birleşimsiz) | 331 | −0.240% | −11.8% | −22.3% |
| H0S (canlı tasarım) | 333 | −0.236% | −11.7% | −22.6% |

**Karar (ön kayıtlı kural): birleşim kalıyor** — iki ölçüt de çok az yukarıda; pratik etkisi 6 yılda
2 işlem. Davranışı değiştirmeyen temizlik: statik listedeki **28 ölü hisse** (üç bağımsız yfinance
denemesinde veri yok: satın alınan/birleşen/iflas eden) çıkarıldı; sinyal üretemiyorlardı, her
taramada boşa istek ve hata üretiyorlardı. Liste 234 → 206.

---

## Ek H — Az işlenen kamu verisi: SEC başvuruları + FINRA açığa satış (22 Kasım 2026)

"Başkasının görmediği bilgi" yasal olarak ancak herkese açık ama az işlenen veridir. Ön kayıt Ek H
(veri indirilmeden önce). Veri: SEC EDGAR çeyreklik indeks 2020Q3–2026Q4 (~964 bin ilgili başvuru),
SEC CIK→ticker, FINRA günlük konsolide açığa satış (1.501 gün, 0 hata). 13D hedef şirket eşlemesi
başvuru başlıklarıyla doğrulandı (12/12).

| Hipotez | Ölçüm | Sonuç | Karar |
|---|---|---|---|
| **H7** ilk 13D (aktivist giriş) | 449 işlem, 71 ay | brüt −0.89%, net −1.39%/işlem, havuz farkı −0.95 [−2.78, +0.94], kazanma %40, portföy −47.9% (DD −51.8%) | ❌ P2–P6 |
| **H8** seyreltme filtresi (son 30 gün S-1/S-3/F-x/424B) | H0'da 539 sinyalin 70'i elendi | H0: net −0.24 → **+0.26%**, portföy −11.8 → **+2.9%**, DD −22.3 → −15.9 (a ✔) · elenenlerin 8 seans farkı +0.52 [−1.97, +3.74] vs kalanlar +0.64 (b ✘) · H6: net +0.10 → +0.11, portföy −34 → −44 | ❌ (b) |
| **H10** yoğun açığa satış filtresi (SVR5 üst %20) | H0'da 102 sinyal elendi | H0: net −0.24 → −0.47%, portföy −12.5% · elenenler +0.63 vs kalanlar +0.63 | ❌ (a, b) |

**Yorum.** Küçük hisselerde 13D sonrası fiyat düşüyor: hareketin büyüğü başvurudan önce yaşanıyor ve
küçük şirketlerde 13D'lerin çoğu aktivist değil finansman/kurtarma yatırımcısı. Açığa satış oranı
(FINRA TotalVolume konsolide hacim değil) bilgi taşımıyor. **H8 tek umut verici iz:** canlı üründe
belirgin iyileşme, ama elenen sinyaller tek başına kötü değil ve H6'da etkisi yok — iyileşme
büyük olasılıkla stop/seçim etkileşimi ve az sayıda işlemden (≈38) geliyor. Ön kayıtlı ikinci şart
bunu ayıklamak için konmuştu; production'a alınmadı. Geçerli test: ileri veri (yeni ön kayıt).

Tekrar üretim: `python lab/fetch_altdata.py` → `python lab/info_edgar.py build` →
`python lab/h0_sim.py --exclude output/lab/altdata/ex_{dilution,shortvol}.parquet` →
`python lab/info_edgar.py evaluate`.

---

## Ek I — Kıdemli swing trader denetimi (22 Kasım 2026)

Canlı ürün, profesyonel bir swing trader'ın kontrol listesine karşı kodda tek tek doğrulandı.

| Kural | Durum |
|---|---|
| Piyasa yönü: ayı piyasasında yeni alım yok, temkinli piyasada sınırlı | ✔ BEAR 0, CAUTION 1 |
| Her işlemde sabit risk (%1.5) + ATR stop (2.5×, tip tavanı) | ✔ |
| Bilanço öncesi alım yok | ✔ (Ek D1'de düzeltildi) |
| Likidite (≥5M$/gün) ve hacimli kırılım (RVOL ≥1.5) | ✔ |
| Gap ile kovalamama (+5 / −7) | ✔ |
| Kısmi kâr (T1'de 1/3) + iz süren stop + zaman stopu | ✔ |
| Doğru evren / dürüst (net) sonuç | ✔ (Ek F'de düzeltildi) |
| **Nakit sınırı (kaldıraç yok)** | ✘ → **düzeltildi** |
| **Hisse ihracı / seyreltme kontrolü** | ✘ → **işaret + ileri ölçüm eklendi** |
| Sektör yoğunlaşması sınırı | ✘ ölçülmedi (sıradaki) |
| Düşüş / kayıp serisi freni | ✘ ölçülmedi |

**Nakit sınırı.** BULL rejiminde eşzamanlı pozisyon sınırı yoktu ve nakit kontrol edilmiyordu. Lab
replikasında paper portföy zirvede 19 pozisyon / 28.401$ taşıyordu (10.000$ hesap, günlerin %3.4'ünde
hesap aşılıyor) — nakit hesapta (Midas) açılamaz işlemler. Gerçeklik şartı olarak uygulandı (sonuca
göre karar verilmedi): otomatik giriş ve backtest, açık + bekleyen + yeni tutar > hesap ise girmez.

| 10.000$ hesap, 2021–2026 | Önce | Nakit sınırıyla |
|---|---:|---:|
| Zirve bağlanan sermaye | 28.401$ | 9.930$ |
| Canlı lotla net | −470$ | −387$ |
| %10/işlem portföy / Max DD | −11.8% / −22.3% | −8.4% / −19.9% |

**Seyreltme kontrolü.** Tarama sonuçlarında son 30 günde S-1/S-3/F-x/424B başvurusu olan hisseye
"⚠ Hisse ihracı başvurusu" işareti (kapı değil — Ek H'de sağlamlık şartını geçemedi). Gölge raporu
canlı ürünün kapanan işlemlerini bu işarete göre ayırıyor: fark ileri veride sürerse filtre kapı olur.
Gerçek veri denemesi: ASTS (424B5, 17–18 Kas), IONQ (424B7, 13 Kas) yakalandı.

**Yan bulgu (düzeltildi).** SEC, olmayan dosyaya (federal tatilde günlük indeks, ör. 11 Kas Veterans
Day — NYSE açık) 404 yerine gzip'li S3 `AccessDenied` 403 döndürüyor. Gölge H4 günleri sırayla
işlediği için ilk SEC tatilinde kilitlenirdi; artık bu yanıt "dosya yok" sayılıyor, gerçek engeller
(hız sınırı) hâlâ hata olarak yükseliyor.

---

## Ek J — H6S "Profesyonel VCE": en güçlü aday (22 Kasım 2026)

Ön kayıt Ek J (H6R, sonuçtan önce) ve Ek J2 (H6S, H6R sonucu görüldükten SONRA, etiketli).
H6'nın tek eksiği profesyonel risk katmanıydı: piyasa freni ve stop.

| 2021–2026 (in-sample) | H0 canlı | H6 | H6R (SPY freni) | **H6S (IWM freni)** |
|---|---:|---:|---:|---:|
| İşlem | 331 | 2.066 | 1.522 | 1.145 |
| Net/işlem 50 bps | −0.24% | +0.10% | +0.39% | **+0.64%** |
| Havuz farkı (95% CI) | +0.00 | +0.46 [−0.26, +1.23] | +0.78 [−0.02, +1.65] | **+0.96 [+0.02, +1.97]** |
| %10/işlem portföy 50 bps | −11.8% | −34.1% | +13.6% | **+64.5%** (CAGR 8.9%) |
| … 20 bps | — | −7.4% | +48.9% | **+101.9%** (CAGR 12.7%) |
| Max DD | −22.3% | −75.6% | −51.2% | **−31.5%** |
| Kriter (P1–P6) | — | — | P2 P4 P5 P6 ✘ | **P3 P4 ✘** (yıl 3/6; en iyi %5 çıkarılınca negatif) |

Aynı dönem IWM al-tut +42%, SPY +102%.

**H6R → H6S neden?** H6R'nin −51% düşüşü 2021-01 → 2023-05'te: SPY +13.6% iken IWM −15.2%; işlemlerin
402/410'u SPY tabanlı rejimin "BULL" dediği günlerde açıldı. Küçük hisse alan sistemin freni küçük
hisse endeksine bakmalı. H6S'de 2022'de yalnız 4 işlem açıldı.

**Karar (ön kayıtlı):** tüm kriterleri geçmedi ama H0'dan açıkça iyi → **gölgede ileri test**
(`H6S`, kilit `47dfd9d0…`; H4/H5/H6 kilitleri değişmedi). Production çıkış kuralı lab'daki 300
H6S işlemini birebir üretiyor. Aynı IWM freni canlı replikaya (seçim/kapasite katmanında, yaklaşık)
uygulandığında da iyileşme var: net −0.26 → −0.18%, portföy −8.4 → −5.3%, DD −19.9 → −16.5%.
Canlı ürüne geçiş ön kayıt gereği ileri veride doğrulanırsa.

**Dürüst sınırlar:** VCE + 8 seans Ek E tanısından, IWM freni H6R sonucundan geldi → in-sample sayılar
iyimser. Kazanç büyük kazananlara bağımlı (P4). Survivorship. Karar F1–F4 ile ileri veride (~6 ay,
≥100 işlem).

---

## Ek K — Canlı paper ürün H6S'ye geçti (22 Kasım 2026, kullanıcı kararı)

| Katman | Eski kurallar (artık kontrol grubu) | **H6S (ana strateji)** |
|---|---|---|
| Sinyal | VCE + RVOL, RSI/Weinstein/swing kapıları, kalite eşiği, rejim tabanı, top N | Canlı evren + sert filtreler + bilanço kapısı + **yalnız VCE** |
| Piyasa freni | SPY rejimi: BEAR 0, CAUTION 1 | **IWM (küçük hisse) rejimi**: BEAR 0, CAUTION 1 |
| Boyut | %1.5 risk, tip tavanı | **Hesabın %10'u**, nakit sınırı (kaldıraç yok) |
| Giriş | Ertesi açılış, gap filtresi +5/−7, cooldown 5 gün | Ertesi açılış, **gap filtresi ve cooldown yok** (ölçülen kural) |
| Çıkış | 2.5×ATR stop, T1'de 1/3, T2, iz süren stop, tip süresi | **2.5×ATR felaket stop'u** veya **8. seans kapanışı** |

Uygulama: `paper_trades.strategy` ('H6S' / 'LEGACY'; eski satırlar boş = LEGACY), her stratejinin kendi
sanal hesabı (tekrar, cooldown, nakit, kapasite strateji bazında), `execution.evaluate` H6S'de hedef
ve trail atlar, onay H6S'de gap filtresi uygulamaz. Ayarlar → Otomatik tarama: "Ana strateji: H6S"
(kapatınca tek adımda eski kurallara dönüş), "Eski kurallar kontrol grubu". Performans sayfası
H6S / Eski kurallar / Tümü seçicisiyle ayrı gösterir; işlemlerde "H6S" / "Eski" rozeti.

---

## Ek L — H6S risk kuralları: sektör sınırı ve kayıp serisi freni (22 Kasım 2026)

Ön kayıt Ek L (sonuçtan önce; tek parametre seti). Karar: toplam getiri düşmeyecek VE düşüş artmayacak.

| H6S lab replikası, %10/işlem, 50 bps | İşlem (portföyün aldığı) | 6 yıl portföy | Max DD | Karar |
|---|---:|---:|---:|---|
| Taban | 1.145 (680) | +64.5% | −31.5% | — |
| **L1 sektör ≤ 2** | 681 (596) | **−6.8%** | −49.2% | ❌ eklenmedi |
| **L2 kayıp serisi freni** (3 zarar → yarım boyut) | 1.145 (715, 194 yarım) | **+67.1%** | **−29.9%** | ✅ **eklendi** |
| L1 + L2 | 681 (618) | +19.4% | −36.9% | ❌ |

**L1 neden kötü:** sınır 569 sinyali kesti ve işlem başı net +0.64 → +0.14%'e düştü. Bir sektör kırılım
verdiğinde hisseleri birlikte yükseliyor (sektör momentumu); sınır tam da en kazançlı kümeyi kesiyor.
**L2 neden iyi:** seri zararlar genelde piyasa ritminin bozulduğu dönemlere denk geliyor; yarım boyut
zararı küçültüyor, boşa çıkan nakit sonraki kazançlı işlemlere gidiyor (alınan işlem 680 → 715).

Canlı: `auto_scan.h6s_loss_streak_brake` (açık), kural `paper_trading/risk_rules.py`; geçmiş okunamazsa
temkinli taraf (yarım boyut). Midas Emirleri sayfası fren aktifken uyarı gösteriyor. Gölge H6S (donmuş)
frensiz devam ediyor → frenin ileri veride kontrolü.

## Ek M — Kâğıttan Midas'a: çıkış saati, boyutlama, gece boşluğu, boş nakit, gerçek ücret (22 Kasım 2026)

Ön kayıt Ek M (SHA256 `f359c1e4…`, 12:40Z, sonuçtan önce). Taban = canlı ürün (H6S + L2 freni).
Betik: `lab/execution_m.py` → `output/lab/m_execution.json`.

**Önce bir düzeltme (bulunan hata, örtbas yok):** lab portföyü bir günün kapanışında çıkan işlemin nakdini
aynı günün AÇILIŞINDAKİ girişe kullandırıyordu ve boyutu giriş günü kapanışıyla hesaplıyordu. Gün içi
gerçek sırayla taban **+67.1% → +54.7%** (max DD −29.4%). `lab/evaluate.portfolio` düzeltildi; önceki
eklerdeki portföy sayıları bu kadar iyimserdir (stratejiler arası kıyas aynı yanlılıkla yapıldığı için
sıralamalar değişmez). Ek L freni düzeltilmiş sırayla yeniden doğrulandı: frensiz +50.0% → frenli +54.7%,
DD eşit (−29.4%) → **geçiyor**.

| Ölçüm (6 yıl, $10K, %10/işlem, 50 bps) | İşlem başı net | Portföy | Max DD | Karar |
|---|---:|---:|---:|---|
| Taban (H6S + fren, düzeltilmiş sıra) | +0.64% | **+54.7%** | −29.4% | — |
| M1 çıkış 9. seans açılışında | +0.75% | +59.9% | **−35.1%** | ❌ DD 5.7 puan kötü (tolerans 1) |
| M2 eşit riskli boyut (r=%0.88, ort. tutar %8.7) | +0.64% | +44.1% | −22.9% | ❌ toplam düştü |
| M3 sinyal kapanışında giriş (gece boşluğu) | +0.16 puan | — | — | ❌ eşik +0.30 (kurmaya değmez) |
| M4 boş nakit | — | — | — | **kapatıldı — kullanıcı ilkesi: faiz yok** |

**Asıl bulgu — boş nakit:** hesap zamanın ortalama **%74'ünde nakitte**. Sinyaller kümelenerek geliyor: o
günlerde nakit yetmiyor (1.145 sinyalin ~480'i atlandı), diğer günler para boş duruyor. Faiz kullanıcı
ilkesiyle kapalı; boş nakit ancak swing işlemleriyle değerlendirilecek (açık iş).

**IWM kıyası (dürüst tablo):** aynı dönemde IWM fiyatı +46.6% (DD −31.9%). H6S (+54.7%) toplamda IWM'i
ancak az farkla geçiyor ve **farkın tamamı 2021–2023'ten** geliyor: 2021–23 H6S +50.8% / IWM +6.2%;
2024–26 H6S **+2.6%** / IWM +38.1%. Yani H6S'nin kazancı büyük ölçüde rejim freninin 2022 düşüşünden
kaçmasından geliyor; yükselen piyasada (2024–26) endeksin çok gerisinde kaldı. İleri test bunu çözecek.

**Gerçek Midas ücreti (sonradan, gerçeklik kontrolü):** Midas ABD işlemlerinde emir başına sabit **1.5$**
alıyor (getmidas.com/ucretler). 10.000$ hesapta 1.000$'lık pozisyonda gidiş-dönüş %0.30, fren yarım
boyutunda %0.60. Fren bu ücretle yeniden ölçüldü:

| Hesap, spread/kayma | Frensiz | Frenli | Fren kuralı (toplam ≥ VE DD ≤) |
|---|---|---|---|
| 10K$, 20 bps (canlı ayar) + 1.5$ | +60.1% / −28.9% | +63.2% / −28.1% | ✅ |
| 10K$, 50 bps + 1.5$ (kötümser) | +30.6% / −30.9% | +35.7% / −31.8% | DD 0.9 puan kötü |
| 25K$, 20 bps + 1.5$ | +73.0% / −28.4% | +76.7% / −26.4% | ✅ |
| 25K$, 50 bps + 1.5$ | +42.2% / −29.7% | +46.5% / −30.6% | DD 0.9 puan kötü |

Fren her senaryoda toplam getiriyi +3–5 puan artırıyor; DD etkisi ±1 puan içinde (gürültü) → korunuyor.
Hesap büyüdükçe sabit ücretin payı küçülüyor (25K$'da aynı strateji +11–14 puan daha fazla).

**Canlıya alınanlar (gerçeklik şartları, sonuca bakılmadan):**
- **Bileşik hesap:** H6S hesabı = başlangıç + kapanmış H6S işlemlerinin maliyet+ücret sonrası sonucu;
  pozisyon boyutu ve nakit sınırı bundan (`paper_trading/costs.account_equity`). Önceden sabit 10.000$'dan
  boyutluyordu — gerçek hesap böyle davranmaz.
- **Midas ücreti:** yeni ayar `commission_usd_per_side` (1.5$); net P/L (Performans), H6S hesabı ve
  backtest her emirde düşüyor (kısmi satış = 3. emir).
- **Emirler sayfası:** bileşik hesap, emir ücreti, boş nakit oranı.

**Kapananlar (tekrar önerme):** M1 (çıkış 9. seans açılışı — satış kapanışta kalıyor, Türkiye saatiyle
~24:00/yazın 23:00), M2 eşit riskli boyut, M3 kapanışta giriş.

## Ek N — 2024–26'da neden IWM'in gerisinde kaldık? (teşhis + çekirdek IWM denemesi)

Ön kayıt: `lab/PREREGISTRATION.md` Ek N (SHA256 `a7815903…`), betik `lab/core_n.py`, çıktı
`output/lab/n_core.json`. Taban = canlı ürün (H6S + fren), Ek M portföyü, 10.000$, Midas 1.5$/emir.

**N0 — teşhis (2024-01 → 2026-11, 50 bps + 1.5$):** portföy −8.5% / IWM +43.0%.

| Neden | Ölçüm |
|---|---|
| Para çoğunlukla nakitte | Ortalama maruziyet %34 → yükselişin çoğu kaçıyor (maliyetsiz bile +22.7%) |
| Maliyet + ücret | 413 işlem; işlem başı brüt +0.71% < maliyet ~0.80% → ~31 puan erime |
| Seçim edge'i yok | BULL rejimindeki 392 işlem brüt −0.08%, aynı pencerede IWM 0.00 |
| Rejim freni IWM'i zamanlamıyor | CAUTION/BEAR günlerinde (%40) IWM bileşik +27% (2021–23'te de BEAR günleri +11%) |
| Kuyruk bağımlılığı | En iyi %5 işlem çıkarılınca brüt ort. −1.48% (2021–23'te −1.11%) |

Sinyal kuralları 2024–26'ya göre değiştirilmedi (eğri uydurma olurdu). Yapısal tek aday: boş nakdin
maruziyeti.

| | Taban | **N1 boş nakit IWM'de** | N2 IWM, BEAR'de nakit | IWM al-tut |
|---|---:|---:|---:|---:|
| 50 bps — 6 yıl | +35.7% | **+68.9%** | +49.4% | +49.1% |
| 2021–23 | +42.6% | +35.7% | +28.8% | +4.5% |
| 2024–26 | −8.5% | +20.4% | +12.8% | +42.8% |
| Max düşüş | −31.8% | **−36.6%** | −32.9% | −31.9% |
| 20 bps (canlı) — 6 yıl | +63.2% | +102.6% | +79.5% | +49.1% |
| 2021–23 / 2024–26 | +52.8 / +3.5 | +46.0 / +33.7 | +38.6 / +25.2 | +4.5 / +42.8 |
| Max düşüş | −28.1% | −33.6% | −29.9% | −31.9% |

N1: ~190 IWM emri / 6 yıl (ayda ~2–3), ort. %55 IWM + %26 hisse. "Çekirdeksiz" mod tabanı birebir verdi.

**Karar (ön kayıtlı):** N1 ❌ — (a) +33 puan ✅, (b) 2021–23 yarısı −6.9 puan ❌ (tolerans 3), (c) max düşüş
−36.6% ❌ (sınır −35%). N2 ❌ — (b) −13.8 puan. **Canlı ürün değişmedi.**

**Dürüst sonuç:** 2024–26 açığının kök nedeni hisse işlemlerinin o dönemde maliyet sonrası endeksi
yenememesi + paranın üçte ikisinin nakitte durması. IWM çekirdeği açığın ~üçte ikisini kapatıyor ama
bedeli kötü yıllarda daha derin düşüş (2022'de IWM'le birlikte düşüyor) — ön kayıtlı şartlar bunu kabul
etmedi. Kullanıcı yönü: kazanç swing işlemlerinden gelecek (faiz yok, endeks çekirdeği değil) → boş nakit
ve maliyet erimesi swing tarafında çözülmeli. Hisse seçiminin edge'i ileri testte (Mayıs 2027) anlaşılır.

## Ek O — Swing tarafı: kazananı koştur, büyük pozisyon, CAUTION sınırı

Ön kayıt: `lab/PREREGISTRATION.md` Ek O (SHA256 `00778d9e…`), betik `lab/swing_o.py`, çıktı
`output/lab/o_swing.json`. Taban = canlı H6S + fren, Ek M portföyü (parite: 35.7% = 35.7%), 10.000$.
İkinci sinyal ailesi ölçülmedi: H1–H5'in hepsi 50 bps sonrası negatif.

| 50 bps + 1.5$ | 6 yıl | 2021–23 | 2024–26 | Max düşüş | İşlem | Net/işlem | Nakit | Karar |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Taban | +35.7% | +42.6% | −8.5% | −31.8% | 663 | +0.79% | %74 | — |
| O1 kazananı koştur (iz süren stop, ≤40 seans) | +43.2% | **+7.2%** | +28.4% | −29.1% | 500 | +1.40% | %65 | ❌ (b) 2021–23 −35 puan, (d) 20 bps'de +1.4 |
| O2 pozisyon %15 | **+66.2%** | +69.0% | −5.1% | **−37.9%** | 508 | +0.98% | %70 | ❌ (c) düşüş −35'i aşıyor |
| O3 CAUTION sınırı yok | −6.6% | +11.1% | −17.5% | **−52.5%** | 881 | +0.20% | %67 | ❌ hepsi |

20 bps + 1.5$ (canlı maliyet): taban +63.2% (DD −28.1), O1 +64.6% (−27.7), O2 +106.4% (−32.8), O3 +22.5% (−47.4).

**Okuma:**
- **O3:** CAUTION'da tek pozisyon sınırı ürünün en değerli kurallarından biri — kaldırınca toplam −42
  puan, düşüş −52%. Rejim freni işlemler için çalışıyor (IWM'i zamanlamasa da).
- **O1:** işlem başı net neredeyse ikiye katlanıyor (+0.79 → +1.40) ama para kazananlarda uzun kalınca
  kümelenen sinyallerin farklı bir alt kümesi alınıyor; sonuç dönemlere göre ters dönüyor (2021–23
  çöküyor, 2024–26 düzeliyor). Sağlam değil.
- **O2:** aynı edge'e daha büyük bahis — iki yarıda da daha çok para, düşüş de orantılı derin. Edge
  değil risk ayarı; ön kayıtlı düşüş sınırını geçemedi. Ayar zaten kullanıcıda (`h6s_position_pct`).

**Sonuç:** canlı ürün değişmedi. Swing kuralları bu sinyalle sınırına yakın; 2024–26 açığının çözümü
kural ayarı değil, sinyalin kendisi — o da ancak ileri testte (Mayıs 2027) ölçülür.

**Ek O kullanıcı kararı (2026-11-22):** O2 benimsendi — canlı H6S pozisyonu %15 (fren %7.5). Ön kayıtlı
düşüş şartını geçmediği bilinerek alındı (etiketli sapma). Beklenen (geçmiş test, 50 bps + 1.5$): +66%
/ max düşüş −38%; 20 bps'de +106% / −33%.
