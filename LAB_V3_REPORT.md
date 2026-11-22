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

### Yeni bulgu (düzeltilmedi): bilanço kapısı veri hatasında açık kalıyor

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
