# 🗺️ SWING TRADE — TAM PİPELİNE HARİTASI (kod-doğrulanmış)

> **Amaç:** Bu belge, sistemin A'dan Z'ye gerçek işleyişinin tek akış referansıdır.
> Ezberden değil, **2026-09-18 kodundan** satır-referanslı doğrulanmıştır.
> Ölçüm defteri (neden kaldı / neden silindi): `GATE_AUDIT.md` (son tur 2026-08-06).
>
> **Sistemin tek işi:** Kanıtlanmış edge'i (VCE volatilite-sıkışma kırılımı + RVOL thrust)
> taşıyan az sayıda YÜKSEK KALİTELİ swing adayını bulmak ve disiplinli exit'le kâra çevirmek.
> Felsefe: "çok sinyal" değil, "az ama kanıtlı edge".
>
> **Ölçülen taban** (2026-08-04, gerçek `scan_stock` + gerçek exit + slippage,
> 78 Q80+ / 21 ay): EV **+3.11%** · WR **%56** · PF **2.00**.
> Ağustos turları kapı/skor/sıralama katmanını sadeleştirdi; eşik eğrisi ve slot
> kısıtlı para ölçümü `GATE_AUDIT.md`'de.

---

## ADIM 0 — VERİ HİJYENİ (görünmez ama kritik temel)

Hiçbir sinyal, temiz olmayan veriyle üretilmez. Her fiyat serisi motora girmeden önce:

| Katman | Ne yapar | Konum | Neden |
|---|---|---|---|
| **Tamamlanmamış bar atma** | ET 16:00'dan önce bugünün yarım barını düşer | `fetcher.py:_drop_incomplete_last_bar` | Günlük-bar kuralları yalnız TAMAMLANMIŞ barda karar verilebilir; yarım hacim tüm hacim-oranlarını bozar |
| **NaN-bar guard** | Son barın OHLC'si NaN ise (yfinance glitch) o barı düşer — tarihten bağımsız | `fetcher.py:_drop_incomplete_last_bar` | 2026-07-25 olayı: NaN bar `close>ma50` gibi tüm karşılaştırmaları sessizce False yapıp gerçek sinyalleri "reddedilmiş" gösteriyordu |
| **Premarket NaN-Close temizliği** | NaN close satırlarını at | `fetcher.py:fetch_multiple_stocks_batch` | Premarket'te sağlayıcı yarım satır ekliyor |
| **Fallback zinciri** | yfinance → Tiingo → Finnhub | `fetcher.py` | Tek kaynak çökerse veri akışı sürsün |
| **%50 fetch devre kesici** | Fetch başarısı <%50 ise static evrenle 1 retry → o da tutmazsa `data_quality` hatası | `scanner.py:MIN_FETCH_SUCCESS_RATIO` | 2026-07-17 olayı: Finviz ticker bug'ı 3/146 hisseyle sessiz "0 sinyal" ürettiriyordu; artık sessiz çökme imkansız |
| **Session farkındalığı** | Pre-market/kapalı taramada UI uyarısı | `scanner.py:us_market_session` | RVOL/Change-tabanlı Finviz sorguları pre-market'te boş döner |

---

## ADIM 1 — EVREN KEŞFİ (Finviz, 4 aktif sorgu)

Binlerce hisseden, tezin oluşabileceği adayları Finviz sunucusunda ön-eler.
Parse: `_ticker_safe_overview_cls` (Finviz logo-hücresi bug'ına karşı `tab-link` okur).

| Sorgu | Kriter | Neyi hedefler |
|---|---|---|
| **Q6 (small)** | Small-cap + $7+ + avgvol>**300K** + **SMA50 üstü + 20g yeni zirve** | VCE'nin ZORUNLU ön koşulu (kırılım başlangıcı) |
| **Q6b (mid)** | Mid-cap ($2-10B) + avgvol>**500K** + SMA50 üstü + 20g yeni zirve | VCE mid-cap bandı |
| **Q7 (small)** | Small-cap + $7+ + avgvol>**300K** + **RVOL>2 + yeşil + SMA20 üstü** | RVOL thrust (ani kurumsal ilgi) |
| **Q7b (mid)** | Mid-cap + avgvol>**500K** + RVOL>2 + yeşil + SMA20 üstü | RVOL thrust mid-cap |

Hacim bantları 2026-08-03'te ölçüldü: 500K/1M → 300K/500K. Keşif katmanı gevşedi
(+20 sinyal, EV seyrelmedi); motorun **$5M/gün dolar-hacim** hard-gate'i duruyor.
Konum: `universe.py:FINVIZ_MIN_AVG_VOLUME_*`.

**Silinen sorgular:** Q1–Q5 koddan çıkarıldı (`measure_universe_recall.py`: %0–2 katkı).
Settings'te kapalı değiller — yoklar. Konum: `universe.py:get_finviz_universe`.

**Sonuç:** o gün kriterlere uyan ~45 aday (rejime göre 16–150 arası dalgalanır — sabit değil).

---

## ADIM 2 — KOD-İÇİ TEMİZLİK + LİKİDİTE SIRALAMASI

`universe.py:get_finviz_universe` (merge sonrası):
1. **Kara liste** (`EXCLUDED_TICKERS` — delisted/sorunlu ~35 isim) düşülür
2. **Fiyat post-filtresi** $7–1000 (`post_filter_price_min/max`)
3. **Dedup** (ticker normalize + ilk-görülen kazanır)
4. **Dolar-hacim sıralaması** (`fiyat × hacim`, en likit önce) + tavan **260** hisse (`max_scan_tickers`).
   Eski 4-ağırlıklı composite (RVOL/değişim/dolar-hacim/mcap + erken-birikim) ölçüldü,
   tavan 15/15 taramada bağlamadı, silindi. Yerine ölçülmüş tek satır: likidite.
   Kanıt: likit grup +3.31% / illikit −2.14%.
5. **Rank telemetrisi** (`build_rank_info`): tavanın kestiği ticker'lar stats'a yazılır.

> ⚠️ NOT: "$5M dolar-hacim" burada DEĞİL — o Adım 3'teki motor filtresinin hard-gate'i.
> Buradaki dolar-hacim yalnız **tavan bağlarsa kimi tarayacağımızı** seçer.

---

## ADIM 3–4 — MOTOR: scan_stock (her aday, 6 hard-gate)

**Bu, sistemin kalbi.** `engine.py:scan_stock`, her adayı sırayla şu kapılardan geçirir.
Herhangi biri geçilmezse sinyal İPTAL (reject sayacı artar). Canlı-birebir ölçümde
**motor adayların ~%98'ini eliyor** — asıl seçicilik burada.

2026-08-04 denetimi 11 kapıyı tek tek kapattı (bırak-birini-çıkar). ΔEV 0.00
veren 5 kapı silindi. Kalan her kapının ölçülmüş koruyucu değeri var.

| # | Gate | Reject key | Ne yapar | Kanıt |
|---|---|---|---|---|
| 1 | **Evren filtreleri** | `filter_failed` | mcap $250M–10B, **dolar-hacim ≥$5M/gün**, fiyat $7–1000, earnings ±3 gün | likit +3.31% / illikit −2.14% |
| 2 | **TETİKLEYİCİ** | `no_trigger` | **VCE** (squeeze<0.8 + 20g-breakout + green + MA50 + **ZORUNLU RVOL≥1.5x**) VEYA **RVOL thrust** (RVOL≥2.5x + green + MA20) | VCE R10 +5.2% t=2.6 OOS · RVOL EV +0.96% OOS +2.10% |
| 3 | Swing onayı | `swing_not_ready` | 5-gün momentum + MA20 üstü | ΔEV −0.50; engellediği 4 sinyal @ −7.16% |
| 4 | RSI aşırı-alım | `rsi_gate` | RSI > 70 reddet — **VCE MUAF** (sıkışmada yüksek RSI güç) | ΔEV −0.44 |
| 5 | **Weinstein Stage** | `stage_rejected` | Stage 4 (düşüş) VE Stage 3 (dağıtım) hard-reject | ΔEV −0.98; engellediği 18 @ −2.10% — en güçlü kapı |
| 6 | **Kalite eşiği** | `quality_type_*` | Ham skor < rejim floor (BULL **78** / CAUTION-BEAR **80**) reddet | Q80 +2.41% vs Q73 +1.18% |

`insufficient_data` (<20 bar) ve `scan_error` teknik zorunluluk, kaldırılamaz.
Type S (short-squeeze) 2026-08-04'te silindi — girdisi (short-interest)
backtest'te hiç yoktu, muafiyetleri de gitti.

### Silinen kapılar (kodda yok — haritada da yok)

Geç-giriş · OBV dağıtım · zayıf trend · dağıtım günü · R:R(T2).
Hepsi 21 ayda ΔEV **0.00**; çoğu yapısal olarak ateşlenemiyordu (VCE/RVOL zaten
yeşil kapanış + trend üstü istiyor). R:R, exit parametreleri materyal değişirse
yeniden ölçülmeli.

### Adım 3'ün İÇİNDEKİ hacim barajı

VCE tetikleyicisi, kırılım günü hacmi **50g ortalamanın ≥1.5 katı** değilse ateşlemez
(fakeout filtresi). RVOL≥2.0x TERS çalışır (chase) — 1.5x alt baraj, üst sınır yok.
`signals.py:check_vce_breakout:VCE_MIN_RVOL_GATE`. Hacim barajı skordan ÖNCE
(trigger içinde) — hacimsiz hisse skorlanmaya bile gelmez.

### SKORLAMA (eşiğin girdisi — `scoring.py`)

- **Katman 1 (6 ağırlık, 100 puan, `scoring_tuning`):** Float %25 + Momentum %25 + Trend %15 + ATR %13 + Volume %12 + Risk %10.
- **Katman 2 (bonus): SABİT +30.** 14 koşullu bonus (VCE-premium, tight-coil, golden-cross, sektör-RS, VCP, higher-lows…) 2026-08-05'te silindi: her sinyal tavanı dolduruyordu (`min(bonus, 30)`), ayırt etme gücü sıfırdı. `bonus = st.bonus_cap`.
- **Katman 3 (ceza, 5 kaldı — hepsi ölçülmüş):** Type A RSI>70 / >65, Type C RSI>65 / >60, tek-gün>%10. Type B'de RSI cezası yok. 16 ceza silindi (`pen_spread_risk` dahil — 21 ayda hiç ateşlemedi).
- Skor rejimsiz hesaplanır; rejim yalnız EŞİĞİ belirler.

### Sıralama / baraj ayrımı (2026-08-05 — uygulandı)

Eskiden premium-VCE (+8) ve tight-coil (+5) `quality_score`'a ekleniyordu; o skor
hem barajda hem sıralamada kullanılıyordu. 34 sinyal yalnız bu eklemeyle Q80'i
geçiyordu ve o 34'ün EV'si +0.89% (taban +4.19%) — baraj onlar için Q72 gibiydi.
Slot-kısıtlı portföyde (canlıda 4–5 eşzamanlı pozisyon) ayrıştırma +18~+29 puan kazandı.

```
quality_score   HAM skor        → baraj (rejim floor) + gösterim
rank_score      ham + işaretler → YALNIZ sıralama
rank_bonus      0 | 5 | 8 | 13  → UI'da "sıralamada +8/+5"
```

Bilgi çöpe gitmedi: aynı n'de `rank_score` ile sıralamak ham skorla sıralamaktan
daha iyi (+6.26% vs +5.28%). Kilit test: `test_rank_vs_gate_split.py`.

---

## ADIM 5 — SİNYAL PAKETİ + RİSK YÖNETİMİ

Tüm gate'leri geçen aday `signal` dict'i olur (`engine.py`): ticker, entry,
**trigger_pathway** (vce/rvol), **quality_score** (ham), **rank_score**,
**rank_bonus**, swing_type (C / A / B — Type S yok). Sonra `risk.py:add_risk_management`:
- **Risk/işlem: %1.5** (`max_risk_per_trade`)
- **Dinamik stop: entry − 2.5×ATR** (tip tavanı C %14 / A %15 / B %16). Volatil hisse → geniş stop → az lot.
- **Pozisyon cap:** tek hisse max %25 (A/C), %20 (B) — `type_position_caps`
- **T1/T2 hedefleri:** ATR-dinamik + kalite/rejim ayarı

---

## ADIM 6 — PENDING → CONFIRM (t+1 disiplini — edge'in koruyucusu)

**Sinyal ANINDA alınmaz.** `tracker.py:add_trade_from_signal` → **PENDING** kaydı:
- **Entry-window guard:** sinyal barının ertesi seans açılışı (t+1 open) geçtiyse trade AÇILMAZ (`entry_window_open`) — ölçülen edge t+1 açılış girişine ait
- **Duplicate guard:** aktif PENDING/OPEN aynı ticker varsa engelle (`check_duplicate`)
- **Cooldown:** son kapanan trade'den N gün geçmeden aynı ticker'a girme

Ertesi gün `confirm_pending_trades` (5dk'da bir arka planda çalışır):
- Giriş = **t+1 açılış fiyatı**
- **GAP FİLTRESİ:** açılış >+%5 (gap-up) veya <−%7 (gap-down) ise **REJECTED** — limitler çağrı anında ayardan okunur (import'ta dondurulmaz)
- Stop/T1/T2 gerçek açılış fiyatına göre **yeniden hesaplanır** (`CONFIRM_ATR_MULTIPLIER=2.5`)
- Status: PENDING → OPEN

### Ölçülmüş, uygulanmamış (2026-08-06)

Gün içi kırılımda giriş **çürütüldü** (927 tetik, %98 fakeout, EV −1.77%).
Saf dip alımı işlem başı iyi, toplamda kaybettiriyor (dolmayanlar en iyileri).
Melez (dip −0.25 ATR, dolmazsa t+1 kapanış) +0.42 puan kazandı ama canlı akışı
iki adıma çıkarıyor — örneklem büyüyene kadar bekletiliyor. **Canlı hâlâ t+1 açılış.**

---

## ADIM 7 — EXIT MİMARİSİ (kazananı koştur, kaybedeni küçük tut)

`tracker.py:check_exit_conditions` (her fiyat güncellemesinde, bar-bar):
1. **Stop / trailing stop** (önce — gap-down'da açılıştan gerçekçi dolum; trail güncellemesinden ÖNCE)
2. **T1 kısmi çıkış:** hedefe gelince pozisyonun **%33**'ü kapanır + stop **breakeven'a** çekilir
   (`partial_at_t1_fraction`, ölçüldü: %50 → %33 = +0.42 puan EV; sağ kuyruk kesilmesin)
3. **T2 / kalan çıkış**
4. **Chandelier trailing:** kalan, tepe−3×ATR ile takip edilir (1.5 ATR kâr sonrası aktif)
5. **Timeout:** max 20 **işlem** günü (`max_holding_days`) — takvim günü değil

Backtest çıkışı canlı tracker ile birebir eşitlendi (2026-08-05 parite turu).
Kilit test: `test_backtest_exit_parity.py`.

*(Exit ölçümü: geniş stop + trailing, dar stop'a göre EV'yi ~2 katına çıkardı — v14.)*

---

## ADIM 8 — FORWARD-RETURN TAKİBİ (canlı savaş kaydı)

`forward_returns.py:ForwardReturnTracker` (her taramada otomatik, mode=pg Supabase):
- Her sinyal kaydedilir; 3/5/10 işlem günü sonra **gerçek forward-return (R3/R5/R10)** + MFE/MAE otomatik doldurulur
- Giriş konvansiyonu backtest ile AYNI: t+1 açılış
- **Amaç:** canlı gerçek getiriyi backtest vaadiyle çarpıştırmak — "backtest tuttu mu?"

Ayrıca `signal_history_storage` her tarama koşusunu (stats + sinyaller) saklar,
`regime_storage` rejim geçmişini tutar.

---

## PİPELİNE ÖZET AKIŞ

```
[0] Veri hijyeni (NaN/incomplete bar temizliği, %50 devre kesici)
      ↓
[1] Finviz Q6/Q6b/Q7/Q7b (avgvol 300K/500K) → ~45 aday
      ↓
[2] Kara liste + fiyat filtresi + dedup + dolar-hacim sıralama (tavan 260)
      ↓
[3-4] scan_stock: 6 hard-gate
      filtre → VCE/RVOL trigger [RVOL≥1.5x] → swing → RSI → Weinstein → ham skor ≥ eşik
      rank_score (ham + VCE işaretleri) yalnız SIRALAMA     ← ~%98 eleme
      ↓
[5] Risk yönetimi (%1.5 risk, 2.5×ATR stop, pozisyon cap A/C %25 B %20)
      ↓
[6] PENDING → t+1 açılış + GAP filtresi + entry-window guard → OPEN
      ↓
[7] Exit (T1 %33 kısmi+breakeven → chandelier trailing → timeout 20 işlem günü)
      ↓
[8] ForwardReturnTracker → Supabase (canlı R3/R5/R10 vs backtest)
```

---

## DURUM — ölçülmüş, sadeleştirilmiş (Ağustos 2026 denetimi sonrası)

27 Temmuz v34 "code freeze" **kapanmıştır.** Ağustos turları freeze'i ölçümle
deldi: ölü kapı/skor/ayar silindi, sıralama-baraj ayrıldı, T1 %33'e indi,
backtest çıkışı canlıyla eşitlendi. Kalan her kuralın ya ölçülmüş koruyucu
değeri var ya da teknik zorunluluk.

**Hâlâ dokunma (ölçülmeden):** bonus tavanını kaldırmak, Q82'ye eşik çekmek,
rejim tespit mantığı, tip sınıflaması (C/A/B), cooldown. Bunlar
`GATE_AUDIT.md` "ölçülmemiş kalanlar" listesinde.

Yeni kural ekleme / parametre kaydırma yasaktır — önce harness, sonra kod.
Yalnız gerçek bug (NaN/500 gibi) ve ölçülmüş ΔEV'si kanıtlı değişiklik.

*Kanıt harness'ları: measure_gate_value, measure_remaining_gates, measure_score_modifiers,
measure_post_cap_bonuses, measure_threshold_money, measure_t1_fraction,
measure_intraday_entry, measure_signal_edge, measure_universe_recall, exit_lab_vce_rvol,
backtest_live_replica, measure_score_edge.*
