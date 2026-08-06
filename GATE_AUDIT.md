# Gate Denetimi

> **Ölçüm:** 2026-08-04 · `scripts/measure_gate_value.py` + `measure_remaining_gates.py`
> 78 Q80+ sinyal / 21 ay / 995 ticker · gerçek motor (`scan_stock`) + gerçek exit
> (2.5×ATR, T1 %33, chandelier trail) + slippage
> **Taban:** EV **+3.11%** · WR %56 · PF 2.00

Her kapı tek tek devre dışı bırakıldı, sonuç tabanla karşılaştırıldı
(**bırak-birini-çıkar**, skor bileşenlerine uygulanan yöntemin aynısı).

## Silme ilkesi

```
Ölçüldü + etkisi yok        → SİL   (zararlı olması gerekmez)
Ölçüldü + işe yarıyor       → KAL
Ölçüldü + zararlı           → SİL
Ölçüm bozuk / kanıt yok     → KAL   → sonra DOĞRU ölç
Hiçbir kod okumuyor         → SİL   (ölçüme bile gerek yok)
```

---

## 🟢 KALAN KAPILAR — 6

### Ölçüldü, koruyucu değeri kanıtlı (3)

| Kapı | ΔEV | TRAIN | OOS | Engellediği sinyaller |
|---|---|---|---|---|
| **Weinstein Stage 3/4** | **−0.98** | −1.46 | −0.54 | 18 @ **−2.10%** |
| **Swing onayı** (MA20 üstü + 5g momentum) | **−0.50** | −1.00 | −0.10 | 4 @ **−7.16%** |
| **RSI > 70** (VCE ve Type S muaf) | **−0.44** | −0.75 | −0.27 | 23 @ +1.16% |

Üçü de **TRAIN ve OOS'ta aynı yönde.**

- **Weinstein en güçlü:** engellediği 18 sinyalin EV'si −2.10%. Dağıtım (Stage 3)
  ve düşüş (Stage 4) fazına girmek doğrudan para kaybı.
- **Swing onayı en keskin:** yalnız 4 sinyal engelliyor ama EV'leri −7.16%.
- **RSI nüanslı:** engellediği sinyaller pozitif (+1.16%) ama kaliteyi
  seyreltiyor → toplam EV −0.44 düşüyor. `measure_rsi_gate.py`: eşiği 75/80/100'e
  çekmek her seferinde negatif-EV sinyal ekliyor.

### Çekirdek (ölçüldü, kapı değil filtre/eşik) (3)

| Katman | Kanıt |
|---|---|
| Evren filtresi (fiyat / mcap / **$5M dolar-hacim**) | Kesişim testi: likit +3.31% / illikit −2.14% |
| Tetik (VCE + RVOL thrust) | VCE R10 +5.2% t=2.6 OOS ✓ · RVOL EV +0.96% OOS +2.10% |
| Kalite eşiği (Q78/80/82) | Q80 +2.41% vs Q73 +1.18% |

*(Ayrıca `insufficient_data` ve `scan_error` — teknik zorunluluk, kaldırılamaz.)*

---

## 🔴 SİLİNENLER

### Gate'ler — ölçüldü, ΔEV tam 0.00, 21 ayda hiç ateşlenmedi (5)

| Kapı | Neden hiç ateşlenmiyordu |
|---|---|
| **Geç giriş** (5g>%30 & RSI>65) | VCE muafiyeti + Weinstein + swing onayı bu vakaları zaten eliyordu |
| **Dağıtım günü** (hacim≥2× & değişim≤−%5) | VCE **ve** RVOL **ikisi de** yeşil kapanış istiyor → "hacimli düşüş günü" sinyal olarak hiç oluşamıyor |
| **R:R** (rejime göre 1.0-2.0) | 2.5×ATR stop + T1 %10 + T2 tavanları matematiksel olarak neredeyse hep R:R(T2) > 2.0 üretiyor |
| **Zayıf trend** (markdown & strength<10) | VCE/RVOL + swing onayından geçen sinyal tanımı gereği yükseliş trendinde; "markdown" fazı hiç birlikte oluşmuyor |
| **OBV dağıtım** | 21 ayda **1 kez** ateşlendi — engellediği sinyal **+%14.79 kazandı.** Koruyucu değeri sıfır ölçüldü (n=1 olduğu için "zararlı" kanıtlanamaz, ama "faydalı" kesinlikle çürütüldü) |

⚠️ **R:R notu:** İlk ölçümüm **geçersizdi** — kapıyı `min_rr_at_entry=0` ile
kapatmaya çalıştım ama gate rejime göre koda gömülü değerleri kullanıyordu, ayar
yalnız *bilinmeyen rejim* dalını etkiliyordu. Değerler ayara taşındıktan sonra
gate gerçekten kapatıldı ve ΔEV 0.00 doğrulandı.
**Exit parametreleri (stop çarpanı / T2 tavanları) materyal olarak değişirse bu
kapı yeniden ölçülmeli** — o zaman bağlayabilir.

### Hiçbir kodun okumadığı ayarlar (grep ile doğrulandı)

```
scan_gates.parabolic_five_day_return_gt
scan_gates.extreme_five_day_return_gt
scan_gates.extreme_rsi_gt
```

UI'da ayarlanabiliyorlardı ama **hiçbir etkileri yoktu.**

### Aynı turda silinen diğerleri

| Ne | Neden |
|---|---|
| Composite sıralama: metot + 4 ağırlık + kovalama cezası + erken-birikim bonusu (~130 satır) | Hiçbiri ölçülmemişti; tavan 15/15 taramada bağlamadı. Yerine ölçülmüş tek satır: dolar-hacim |
| Finviz Q1/Q2/Q3 (73 satır) | Recall ölçümü: %0.5-2 katkı |
| `_parse_percent` | Composite gidince kullanılmaz kaldı |
| ~52 satır mezar taşı yorumu + bayat docstring | Silinen özellikleri anlatıyorlardı |

**Toplam silinen:** 5 gate · 21 ayar · 15 UI alanı · ~350 satır kod.

---

## Neden bu kadar kapı hiç ateşlenmiyordu?

Tek bir yapısal sebep: **VCE ve RVOL thrust ikisi de "yeşil kapanış + trend
üstü" istiyor.** Bu iki şart, aşağı yönlü/dağıtım/geç-giriş senaryolarını
tetik aşamasında zaten eliyor. Sonraki kapılar aynı şeyi tekrar kontrol
ediyordu — biri yeterliydi.

Kapılar zamanla eklendi ama **hangi tetiğin ne garanti ettiği hiç ölçülmedi.**
Bu denetim o boşluğu kapattı.

---

## 3. tur — SKOR DEĞİŞTİRİCİLERİ (2026-08-05)

Kapılar bittikten sonra geriye skorun kendi içi kalmıştı: **14 bonus + 21 ceza.**
Hiçbiri hiç ölçülmemişti. Harness: `scripts/measure_score_modifiers.py`
(95 sinyal / 21 ay, bırak-birini-çıkar + Q80 seçim testi + TRAIN/OOS).

### Bulgu 1 — 14 bonusun ayırt etme gücü SIFIR

| Ölçüm | Sonuç |
|---|---|
| Bonus tavanına (30) dayanan sinyal oranı | **%100** |
| Tavan üstü aşılıp atılan puan | ortalama **+29.7**, maks +48 (ham toplam ~60) |
| Tek bonusu sıfırlamanın ΔEV'si | 14/14'ünde **tam 0.00** |

Sebep: bonuslar toplanıp `min(bonus, 30)` ile kırpılıyordu. Ham toplam her zaman
tavanın iki katıydı, dolayısıyla **hangi bonus açık olursa olsun skora giden net
sayı aynıydı: +30.** 40 satır kod, 14 ayarlanabilir parametre ve 14 UI slider'ı
tek bir sabitin işini yapıyordu — dahası kullanıcıya "bunu kısarsam skor değişir"
izlenimi veriyordu, halbuki değişmiyordu.

→ Kod tek satıra indi: `bonus = st.bonus_cap`. **Davranış birebir aynı.**

Alternatif (tavanı kaldırıp bonusların gerçekten ayırt etmesini sağlamak)
kasıtlı olarak yapılmadı: ham toplam 40-78 bandına çıkar, bu ÖLÇÜLMEMİŞ bir
davranış değişikliğidir ve Q78/80/82 eşiklerinin tümünün yeniden kalibrasyonunu
gerektirir. Sonraki turun konusu.

### Bulgu 2 — 21 cezanın 16'sı işe yaramıyordu

**Hiç ateşlemeyen (10):** `pen_5d_gt_30`, `pen_b_rsi_gt_75`, `pen_b_rsi_gt_80`,
`pen_below_ma50`, `pen_not_swing_ready`, `pen_obv_distribution`, `pen_parabolic`,
`pen_spread_risk`, `pen_today_gt_15`, `pen_weak_trend_phase`
— kapı denetimindeki aynı yapısal sebep: girdilerini VCE/RVOL tetiği zaten
garanti ediyor (yeşil kapanış + MA50/MA20 üstü + swing_ready hard gate).

**Ateşleyip ΔEV 0.00 verenler (6) — ceza YÖNÜ TERSTİ:**

| Ceza | Ateşleme | Cezalananların EV'si |
|---|---|---|
| `pen_5d_gt_40` | 3 | **+20.36%** |
| `pen_ext_day_gt_25` | 1 | **+21.12%** |
| `pen_ext_day_gt_20` | 1 | **+15.16%** |
| `pen_5d_gt_25` | 5 | **+13.24%** |
| `pen_ma20_falling` | 2 | **+11.24%** |
| `pen_b_rsi_gt_85` | 1 | +8.75% |

Yani "çok koşmuş, cezalandır" sezgisi bu sistemde **yanlış**: cezalanan
sinyaller ortalamanın iki-üç katı kazandı. Sadece cezalar Q80 seçimini
değiştirecek kadar büyük olmadığı için zarar görünmüyordu — ama gerekçesi
olmayan bir kural her an büyütülebilir, o yüzden silindi.

**Kalanlar (5) — çıkarınca EV DÜŞÜYOR, yani koruyorlar:**

| Ceza | Ateşleme | ΔEV (çıkarınca) |
|---|---|---|
| `pen_c_rsi_gt_60` | 12 | −0.20 |
| `pen_c_rsi_gt_65` | 17 | −0.16 |
| `pen_today_gt_10` | 3 | −0.10 |
| `pen_a_rsi_gt_65` | 4 | −0.09 |
| `pen_a_rsi_gt_70` | 28 | −0.07 |

RSI merdiveni tutuyor; aşırı-uzama merdivenleri tutmuyor. Tip B'de RSI cezası
yok — Type B'nin tanımı zaten "yüksek RSI ile koşuyor".

### `pen_spread_risk` — 9 gün önce eklenmişti, ölüydü

S1 (2026-07-27) düşük dolar-hacim **VE** ATR>%8 birleşimini cezalıyordu.
21 ayda bu iki koşul **hiç birlikte gerçekleşmedi** (dvol<7M: 3 sinyal — üstelik
hepsi kazandı; ATR>%8: 5 sinyal). Kapı fiilen ölüydü.

Silmeden önce **saf ATR>%8 cezası** da ölçüldü (koruma sessizce kaybolmasın):

| ATR% bandı | n | TÜM EV | TRAIN | OOS |
|---|---|---|---|---|
| 0-4% | 306 | +0.50% | −0.45% | +0.95% |
| 4-6% | 142 | **+3.72%** | +5.59% | +2.32% |
| 6-8% | 16 | −2.61% | **+9.09%** | **−9.63%** |
| 8%+ | 5 | −6.90% | −8.25% | −4.86% |

6-8% bandı dönemler arası **işaret değiştiriyor** (n=16). Q80'de ATR tavanı tek
bir işlemi eliyor → tek gözleme eğri uydurmak olurdu. **Tavan eklenmedi, ceza
silindi, bulgu kayda geçti.**

### Parite doğrulaması (asıl güvence)

Aynı 95 barı eski ve yeni kodla skorlayıp karşılaştırdık:

```
SKOR FARKI (yeni - eski):  değişmeyen 84/95 · artan 11 (ort +12.4) · azalan 0
Q80 SEÇİMİ:  ESKİ n=78 EV +3.11% WR 56%  ->  YENİ n=78 EV +3.11% WR 56%
             Q80'e yeni giren 0 · Q80'den düşen 0
SIRALAMA  :  8 sinyalli tek kalabalık gün (2024-09-20) — top_3/4/10 AYNI
```

11 sinyalin skoru yükseldi (silinen ceza kadar) ama **hiçbiri kararı
değiştirmedi**: ne eşiği geçen küme, ne sıralama, ne EV. Ölçümün "ΔEV 0.00"
iddiası doğru sebepten doğruydu.

**Bu turda silinen:** 14 bonus · 16 ceza · 33 ayar · 31 UI slider'ı · scoring.py
571 → 434 satır.

---

## 4. tur — TAVAN DIŞI EKLEMELER + ÖLÜ ÇIKTI TEMİZLİĞİ (2026-08-05)

### `+8` premium-VCE ve `+5` tight-coil — tavana tabi DEĞİL

3. turda skorun içindeki 35 değiştirici ölçüldü. Ama iki ekleme
`calculate_quality_score` **dışında**, engine.py'de skora doğrudan biniyor —
yani bonus tavanına takılmıyor, gerçekten eşiği kaydırıyor. Hiç ölçülmemişti.
Harness: `scripts/measure_post_cap_bonuses.py` (123 sinyal / 21 ay).

**İlk sonuç yanıltıcıydı** — kayda geçsin: bayrakları `signal_lab.json`'daki
`vce_premium` / `vce_tight_coil` alanlarından okudum, "ikisi de hiç ateşlemiyor"
çıktı. Yanlıştı: engine bu bayrakları `boosters`'a yazıyor ama **sinyal
sözlüğüne hiç koymuyordu**, dolayısıyla okunan değer hep False'tu. Bayrak
yokluğunu "özellik çalışmıyor" diye okumak yanlış atıf. Doğru kaynak
`trigger_details.vce_metrics`. (Bayraklar artık sinyal sözlüğüne de yazılıyor.)

Doğru ölçümle:

| Kurgu | n | EV | WR | TRAIN | OOS |
|---|---|---|---|---|---|
| MEVCUT (+8, +5) | 105 | +4.19% | 60% | +4.86% | +3.60% |
| premium YOK | 71 | +5.77% | 69% | +8.09% | +3.75% |
| coil YOK | 100 | +4.21% | 59% | +4.90% | +3.64% |

İlk bakışta "+8'i sil, EV +1.58 artıyor" gibi duruyor. **Ama bu çeldirici:**
eklemeyi kaldırmak barajı da yükseltiyor ve *herhangi* bir baraj yükseltmesi
EV'yi artırır. Doğru soru: aynı sinyal sayısına **düz eşik yükseltmesiyle**
inince ne oluyor?

| n=71'de | EV |
|---|---|
| premium YOK (Q80'de) | +5.77% |
| **düz eşik yükseltmesi (+8 dahil skorla)** | **+6.33%** |

Bonuslu skor, bonussuz skordan **daha iyi sıralıyor**. Yani +8 gerçek bilgi
taşıyor — mis-rank etmiyor. **İkisi de KALIYOR.**

### Ama asıl bulgu: doğru bilgi yanlış yerde kullanılıyor

+8, 34 sinyali Q80'in üstüne **taşıyor** ve o 34'ün EV'si **+0.89% / WR %41**
(taban +4.19% / %60). Yani ekleme bir *sıralama* özelliği olarak iyi, bir
*baraj* özelliği olarak kötü. Eşik eğrisi (bonuslar dahil):

| Eşik | n | sinyal/ay | EV | WR | TRAIN | OOS |
|---|---|---|---|---|---|---|
| Q78 | 123 | 5.9 | +3.87% | 60% | +4.63% | +3.22% |
| Q80 | 105 | 5.0 | +4.19% | 60% | +4.86% | +3.60% |
| **Q82** | 87 | 4.1 | +5.30% | 66% | +6.14% | **+4.59%** |
| Q84 | 69 | 3.3 | +6.52% | 72% | +8.97% | +4.40% |
| Q86 | 52 | 2.5 | +7.54% | 73% | +10.92% | +4.41% |
| Q88 | 36 | 1.7 | +8.34% | 75% | +10.76% | +5.31% |

TRAIN monoton tırmanıyor (aşırı uyum imzası) ama **OOS Q82'de tepe yapıp
düzleşiyor**. Toplam getiri (EV × sinyal/ay, OOS) ise neredeyse sabit:
Q80 → 18.0, Q82 → 18.8, Q84 → 14.5, Q86 → 11.0.

**Karar: eşik DEĞİŞTİRİLMEDİ.** Q80 ile Q82 arasında toplam getiri farkı ölçüm
gürültüsü içinde (18.0 vs 18.8) ve bağlayıcı kısıt frekans (canlıda ayda
0.6-4 sinyal görülüyor). Barajı yükseltmek işlem başı EV'yi güzelleştirir ama
toplam kazancı artırmaz — sadece daha az işlem yapar. Eğri buraya kayda geçti;
frekans 5+/ay'a çıkarsa Q82 ilk aday.

### Ölü çıktı temizliği

Ölçüm sırasında görüldü ki motorun ürettiği birçok alan hiç okunmuyordu:

| Silinen | Satır | Neden |
|---|---|---|
| `signals.detect_pullback_setup` | 157 | 3 çıktısını kimse okumuyordu; v13'te "skorlama bağlamı" diye bırakılmıştı ama skor bonus bloğu sabitlenince son tüketici de gitti. Kendi yorumu ölçümü zaten kaydediyor: R5 edge +0.29%, t=0.65 (anlamsız). **Her taranan hissede boşuna koşuyordu.** |
| `patterns.detect_vcp` | 107 | 5 çıktısının hiçbiri okunmuyordu |
| `trend_quality` composite skoru | 49 | `trend_strength` (0-100) hesaplanıyor, kimse okumuyor; `higher_highs_count` de öyle |
| Weinstein `bonus` (+10/+3/−3/−10) | 6 | Gate `stage`'i okuyor, bonusu kimse okumuyor |
| `rs_bonus_vs_spy` kademeleri | 18 | `bonus`/`sector_etf`/`ticker_5d`/`sector_5d` ölü. Fonksiyon `relative_strength_vs_spy` olarak yeniden adlandırıldı — canlı olan iki alanı (`rs_score`, `is_leader`) döndürüyor |
| `has_catalyst` dalı (Type B) | 11 | Katalizör modülü 2026-08-04'te silinince her zaman False oldu; `swing.type_b.catalyst_pts` ölü ayardı |
| 17 ölü signal-dict anahtarı | ~25 | vcp_* (5), weinstein telemetri (5), pullback_* (3), insider/news/short_interest/rsi_div_confidence (4) |
| Ölü UI dalları | ~20 | "Katalist bonus", "Squeeze (SI %)", "Insider", "News" — üreticileri silinmişti, hep boş görünüyordu |

Yerine **eklenen**: `vce_premium` ve `vce_tight_coil` artık sinyal sözlüğünde ve
UI'da görünüyor ("Premium VCE", "Sıkı yay") — skoru gerçekten kaydıran iki
işaret artık kullanıcıya da görünür.

---

## 5. tur — signals / backtest / paper_trading (2026-08-05)

Kalan büyük dosyalar denetlendi. Bu tur ölçüm değil **doğruluk** turu: burada
bulunanlar "işe yaramıyor" değil, **yanlış** kategorisindeydi.

### 🔴 PARİTE KIRIĞI — `/backtest` canlıda kullanılmayan çıkışı gösteriyordu

`smallcap_backtest.py`'nin çıkışı canlı `tracker.py`'den üç yerde ayrışmıştı:

| Konu | CANLI | BACKTEST (kırık) |
|---|---|---|
| Trailing | Chandelier: giriş sonrası **kümülatif tepe** − 3 ATR, +1.5 ATR kârdan sonra | 8 parametreli kademeli merdiven, "tepe" olarak **yalnız o günün** yükseği |
| Aynı-bar sıralaması | Stop, trail güncellemesinden **ÖNCE** | Trail önce → yükseltilmiş stop'tan çıkış → **sonuçlar iyimser** |
| Timeout | **İşlem günü** (bar) | **Takvim günü** → 20 takvim ≈ 14 işlem günü, erken çıkış |

Sıralama kırığı özellikle sinsi: tracker'da bu bir hata olarak bulunup
düzeltilmişti (kod yorumu: *"check stop BEFORE updating trail (was inverted)"*)
ama backtest ters sırayı korumuştu. Ayrıca canlıda **olmayan** bir "Time Stop"
(5 gün + %5 zarar) vardı.

→ Çıkış canlıyla **birebir** eşitlendi; 15 parametreli `backtest_exit_trailing`
bölümü (model + JSON + 15 UI slider'ı) silindi. Kilit test:
`test_backtest_exit_parity.py` — hem kaynak hem davranış seviyesinde bağlıyor
(chandelier formülü, sıralama, gün sayımı).

### 🔴 PARA HATASI — realize P/L %33 satışı %50 sanıyordu

`tracker.py` realize P/L'yi **ikinci kez** hesaplıyor ve kısmi satışı
`position_size // 2` ile bölüyordu — yani %50 varsayımı. Oysa T1 oranı
2026-08-04'te ölçümle **%33**'e indirilmişti.

`storage.close_trade()` DOĞRU hesaplıyor (kayıtlı `partial_exit_pct`'i okuyor),
ama tracker sonucu bellek üstünde **eziyordu** → DB'de doğru, API yanıtında
yanlış sayı. Kullanıcı çıkıştan hemen sonra bir P/L görüyor, sayfayı
yenileyince başkasını.

Yön **iyimser**: giriş 100, T1 108, çıkış 95 senaryosunda doğru sonuç
−0.71/hisse iken kopya **+1.50/hisse** gösteriyordu.

Aynı blokta ikinci hata: T1 sonrası **açık** işlemlerin realize-olmayan P/L'si
tam pozisyon büyüklüğüyle hesaplanıyordu (payların %33'ü satılmışken) ve T1'de
kilitlenen kâr hiç sayılmıyordu. İkisi de düzeltildi.

→ İkinci hesap silindi (otorite `storage`), realize-olmayan P/L satılan/kalan
ayrımını yapıyor.

### 🔴 KALICI BOZUK ENDPOINT — `generate_weekly_report` hiç var olmamış

`/api/performance/weekly-report` ve `/api/genai/weekly-report`,
`PaperTradeReporter.generate_weekly_report()` çağırıyordu — **böyle bir metot
yok**. Her istek `AttributeError` alıyor, geniş `except` onu yutup "Rapor
üretilemedi" fallback'i döndürüyordu. Yani iki endpoint de ilk günden beri
bozuktu ve hata hiç görünmemişti.

Hiçbir sayfa bunları çağırmıyordu (çalışan rapor `/api/genai/weekly-report-ai`,
`genai/reporter.py: WeeklyReporter`). → İki endpoint + ölü frontend export'u +
`api/deps.get_paper_reporter` silindi. `paper_trading/reporter.py` (292 satır)
**tamamen** ölü çıktı ve dosya silindi.

### 🟡 ÖLÜ AYAR TUZAĞININ YARISI GERİ GELMİŞTİ

`tracker.py` gap limitlerini `MAX_GAP_UP_PCT, MAX_GAP_DOWN_PCT = _gap_limits()`
ile **modül import anında** donduruyordu. Kullanıcı UI'dan değiştiriyor, DB'ye
yazılıyor, ama çalışan süreç import anındaki değeri kullanmaya devam ediyordu.
2026-08-04'te kapatılan tuzağın aynısı, yarım kapatılmış hali. → Çağrı anında
okunuyor; `test_gap_limits_are_not_frozen_at_import` bağlıyor.

### ⚡ PERFORMANS — `load_settings()` her çağrıda DB turu atıyordu

Ölçüldü: **5 çağrı 12.7 sn** (her çağrı JSON okuyup üstüne bir DB round-trip).
Bu fonksiyon sıcak yollarda: tracker'ın çıkış döngüsünde her T1 kısmisinde,
scoring'de her sinyalde, motorda her taramada. → 30 sn TTL'li önbellek +
yazma sonrası `invalidate_settings_cache()` (UI değişikliği hâlâ anında).
Kanıt: **test paketi 250 sn → 59 sn**.

### Ölü kod

| Silinen | Satır | Neden |
|---|---|---|
| `paper_trading/reporter.py` (dosya) | 292 | Tamamı ölü; tek "tüketicileri" var olmayan bir metodu çağıran 2 bozuk endpoint'ti |
| `signals.calculate_vwap_position` + `calculate_gap` | 97 | Hiç çağrılmıyor; 8 çıktı anahtarının hiçbiri okunmuyor |
| `signals.check_volume_surge` | 7 | Hiç çağrılmıyor → `min_volume_surge_soft` ayarı da ölmüştü, o da silindi |
| `signals.py` sınıf docstring'i | — | "3-Tier / VWAP / gap / catalyst / Volume≥1.8x" anlatıyordu — sistem VCE+RVOL. Aktif olarak **yanlış bilgi**; gerçek iki yolla yeniden yazıldı |
| 3 ölü sınıf sabiti + 12 ölü dict anahtarı | ~20 | macd ham serileri, rsi_diff/price_diff/confidence, obv_slope/rising, has_breakout/has_continuation, primary, max_single_day |
| `backtest_exit_trailing` (15 ayar + 15 UI slider'ı) | ~90 | Parite eşitlenince okuyanı kalmadı |

---

## 6. tur — SIRALAMA/BARAJ AYRIMI, uygulandı (2026-08-05)

4. turda bulunan açık sorun kapatıldı. Sorun: `+8` premium-VCE ve `+5`
tight-coil tek bir `quality_score`'a ekleniyordu, o skor da hem **sıralamada**
hem **barajda** kullanılıyordu. Sonuç: 34 sinyal yalnız bu ekleme sayesinde
Q80'in üstüne taşınıyordu ve o 34'ün EV'si **+0.89%** (taban +4.19%) — baraj
onlar için fiilen Q72 gibi davranıyordu.

### Neden "EV arttı" tek başına yeterli kanıt değildi

Atılan 34 işlem **zarar ettirmiyor**, +0.89% kazanıyor. Barajı yükseltmek işlem
başı ortalamayı her zaman artırır ama toplamı düşürebilir:

| Sermaye SINIRSIZ | işlem | EV | **toplam** |
|---|---|---|---|
| mevcut | 105 | +4.00% | **+420%** |
| ayrıştırılmış | 66 | +5.09% | +336% |

Yani slot bolsa mevcut kurgu kazanır. Karar sermaye kısıtına bağlı.

### Para ölçümü — slot kısıtlı portföy

`scripts/measure_threshold_money.py` (21 ay, 0.19 puan/işlem maliyet dahil,
slot doluysa sinyal kaçırılır):

| Eşzamanlı slot | Mevcut | Ayrıştırılmış | Fark |
|---|---|---|---|
| 2 | +28.8% | **+107.9%** | +79.1 |
| 3 | +53.7% | **+104.1%** | +50.4 |
| **4** | +47.5% | **+76.0%** | **+28.6** |
| **5** | +39.3% | **+57.5%** | **+18.2** |
| 8 | +39.2% | **+50.6%** | +11.4 |
| 12 | **+32.6%** | +31.8% | −0.8 |

Canlıda tip pozisyon tavanı %20-25 → **4-5 eşzamanlı pozisyon** sığıyor;
risk-bazlı boyutlandırma (%1.5 risk / ~%13 stop) en iyi durumda ~8 slot verir.
Her üç senaryoda da ayrıştırma kazanıyor. Mevcut kurgu ancak 12+ slotta
(imkânsız) başabaş geliyor.

**Sebep:** zayıf işlem iyi bir işlemin slotunu kapatıyor — fırsat maliyeti
+0.89%'lik brüt kazancı fazlasıyla yiyor.

### Uygulama

```
quality_score   HAM skor        → baraj (motor rejim tabanı + API eff_min) + gösterim
rank_score      ham + işaretler → YALNIZ sıralama
rank_bonus      0 | 5 | 8 | 13  → UI'da "sıralamada +8/+5" olarak görünür
```

Bilgi çöpe atılmadı: aynı sinyal sayısında `rank_score` ile sıralamak ham skorla
sıralamaktan **+6.26% vs +5.28%** daha iyi (canlı motorla doğrulandı).

Kilit test: `test_rank_vs_gate_split.py` — `quality_score +=` geri gelirse,
baraj rank_score'a kayarsa veya sıralama işaretleri bırakırsa kırılır.

---

## 7. tur — GİRİŞ ANI: gün içi mi, ertesi açılış mı? (2026-08-06)

Kullanıcı sorusu: *"gün içinde kırılırken ya da geri çekilirken alsak daha çok
kazanmaz mıyız?"* Fikirle kapatılmadı, ölçüldü. Saatlik bar (yfinance 730g)
tüm sinyal dönemimizi kapsıyor. Harness: `measure_intraday_entry.py`,
`measure_intraday_entry_v2.py`, `validate_dip_entry.py`.

### 🔴 Gün içi kırılımda giriş — ÇÜRÜDÜ (fakeout %98)

İlk ölçüm +7.58% dedi. **Geçersizdi**: yalnız KAPANIŞTA GEÇERLİ çıkmış günlerde
"gün içinde girseydik" hesabı yapıyordu — yani sonucu bilinen günler seçilmişti,
tüm fakeout'lar ölçümden düşmüştü (ileriye-bakış yanlılığı).

v2'de yalnız o an bilinebilen bilgiyle ölçüldü (t−1 sıkışma + t−1 MA50 üstü +
gün içi 20g zirve aşımı):

| Kurgu | İşlem | EV | WR | Toplam |
|---|---|---|---|---|
| A t+1 açılış (mevcut) | 65 | +5.05% | 66% | +328% |
| **C2 gün içi, tüm tetikler** | 927 | **−1.77%** | **31%** | **−1643%** |
| C2c yalnız teyitli alt küme *(yanlı)* | 21 | +15.33% | 81% | +322% |

**927 gün içi tetiğin yalnız 21'i (%2) kapanışta geçerli sinyale dönüştü.**
O 21 muhteşem, ama hangisinin o 21'den olacağı gün içinde bilinemez — "teyit"
tam olarak budur. Slot kısıtlı portföyde de her seviyede kaybettiriyor
(3 slot: +109% → +55.6%).

### 🔴 Saf dip alımı — işlem başına iyi, TOPLAMDA kaybettiriyor

Teyitli sinyalde t+1 açılış yerine dipte (limit = açılış − k×ATR) almak:

| | İşlem | EV | Toplam |
|---|---|---|---|
| A hepsi açılıştan | 66 | +5.28% | **+349%** |
| B yalnız dip −0.25 ATR | 51 | +5.45% | +278% |

Eşleştirilmiş karşılaştırmada dip kazanıyor (+5.45 vs +4.39 aynı işlemlerde),
**ama %23 sinyal hiç dolmuyor ve kaçanlar EN İYİLERİ:**

> Dip vermeyen 15 sinyal ortalama **+8.32%** — dip verenlerin (+4.39%) iki katı.

Mekanik: gerçekten patlayan hisse sana ucuzluk sunmaz. Dip beklemek en güçlü
hareketleri sistematik olarak eler.

### ✅ Melez (H) — ölçülen tek kazanan

Dipte dene, **dolmazsa t+1 kapanışta yine al**:

| Q80+ | İşlem | EV | Toplam | TRAIN | OOS |
|---|---|---|---|---|---|
| A açılış (mevcut) | 66 | +5.28% | +349% | +7.16% | +3.82% |
| **H dip −0.25, dolmazsa kapanış** | 66 | **+5.70%** | **+376%** | +7.25% | **+4.49%** |
| H dip −0.50, dolmazsa kapanış | 66 | +4.94% | +326% | +5.57% | +4.45% |

Tutarlılık kontrollerinin hepsini geçiyor: kalite bantlarında aynı (tümü +0.37,
Q80+ +0.42), TRAIN/OOS aynı yön, yıl yıl 2024 +0.65 / 2025 +1.42 / 2026 +0.54.
−0.50 ATR ise kaybettiriyor — yalnız SIĞ dip işe yarıyor.

**Durum: ölçüldü, UYGULANMADI.** Kazanç gerçek ama küçük (+0.42 puan, n=66) ve
canlı giriş akışının iki adıma çıkmasını gerektiriyor (sabah limit + kapanış
doldurma). Örneklem büyüyene kadar bekletiliyor.

### Metodolojik ders

Bu turda İKİ kez aynı hataya düştüm ve ikisini de sonraki adım yakaladı:
1. Sonucu bilinen günleri seçip "gün içinde girseydik" demek (ileriye-bakış).
2. Dolan işlemlerin ortalamasını tam örneklemle kıyaslamak (doluluk yanlılığı) —
   düzeltmesi EŞLEŞTİRİLMİŞ karşılaştırma.
Ayrıca bir fiyat hatası: açılış limitin altındaysa fill AÇILIŞTA olur, limitte
değil (dip girişini kendi aleyhine hesaplıyordum).

---

## Ölçülmemiş kalanlar (sonraki tur)

| Katman | Not |
|---|---|
| Bonus tavanının kendisi (30) | Tavanı kaldırıp bonusların ayırt etmesini sağlamak ölçülmeli — eşik rekalibrasyonu gerektirir |
| Q82 eşiği | Eğri ölçüldü (yukarıda); frekans kısıtı nedeniyle bilinçli ertelendi |
| Tip sınıflaması (C/A/B) | Boyut/stop/hedef tavanı belirliyor, sinyal üretmiyor |
| Rejim tespiti | Eşiği belirliyor; eşik ölçüldü, rejim mantığı ölçülmedi |
| Cooldown 5 gün · Ticker 2-zarar | Portföy seviyesi |
| `signal_confirmation.overext_*` | Yalnız bir log karakterini (✓/⚠) belirliyor; skor cezaları kendi sabit eşiklerini kullanıyor |

---

## Bilanço

| | Önce | Sonra |
|---|---|---|
| Hard gate | 11 | **6** |
| Ölçülmüş gate oranı | %27 | **%100** |
| Gerekçesi belirsiz gate | 8 | **0** |
| Skor bonusu (koşullu) | 14 | **0** (sabit +30) |
| Skor cezası | 21 | **5** (hepsi ölçülmüş) |
| Ölçülmüş skor-değiştirici oranı | %0 | **%100** |

Kalan her kapının ve her cezanın ya **ölçülmüş koruyucu değeri** var ya da
teknik zorunluluk. Ölçülmemiş hiçbir kural sinyal seçimine karışmıyor.
