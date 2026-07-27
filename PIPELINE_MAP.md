# 🗺️ SWING TRADE — TAM PİPELİNE HARİTASI (v34, kod-doğrulanmış)

> **Amaç:** Bu belge, sistemin A'dan Z'ye gerçek işleyişinin **tek kutsal referansıdır.**
> Ezberden değil, koddan satır-referanslı doğrulanmıştır (2026-07-27, CODE FREEZE v34).
> Her katmanın *neden* var olduğu ve *hangi ölçümle* kanıtlandığı belirtilmiştir.
>
> **Sistemin tek işi:** Kanıtlanmış edge'i (VCE volatilite-sıkışma kırılımı + RVOL thrust)
> taşıyan az sayıda YÜKSEK KALİTELİ swing adayını bulmak ve disiplinli exit'le kâra çevirmek.
> Felsefe: "çok sinyal" değil, "az ama kanıtlı edge". Ölçülen backtest metrikleri: işlem başı
> beklenti (EV) ~+5% (Q80+), Profit Factor 2.13, kazanma oranı %63-64.

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

Binlerce hisseden, tezimizin oluşabileceği adayları Finviz sunucusunda ön-eler.
Parse: `_ticker_safe_overview_cls` (Finviz logo-hücresi bug'ına karşı `tab-link` okur).

| Sorgu | Kriter | Neyi hedefler |
|---|---|---|
| **Q6 (small)** | Small-cap + $7+ + avgvol>500K + **SMA50 üstü + 20g yeni zirve** | VCE'nin ZORUNLU ön koşulu (kırılım başlangıcı) |
| **Q6b (mid)** | Mid-cap ($2-10B) + avgvol>1M + SMA50 üstü + 20g yeni zirve | VCE mid-cap bandı |
| **Q7 (small)** | Small-cap + $7+ + avgvol>500K + **RVOL>2 + yeşil + SMA20 üstü** | RVOL thrust (ani kurumsal ilgi) |
| **Q7b (mid)** | Mid-cap + avgvol>1M + RVOL>2 + yeşil + SMA20 üstü | RVOL thrust mid-cap |

**Kapalı/silinen sorgular:** Q1-Q3 (momentum/setup/wider) settings'te kapalı, Q4 (RSI≤40 early) ve Q5/Q5b (VCE-day) silindi — hepsi `measure_universe_recall.py` ile ölçülüp %0-2 katkı verdiği için elendi. Konum: `universe.py:get_finviz_universe`.

**Sonuç:** o gün kriterlere uyan ~45 aday (rejime göre 16-150 arası dalgalanır — sabit değil).

---

## ADIM 2 — KOD-İÇİ TEMİZLİK + COMPOSITE SIRALAMA

`universe.py:get_finviz_universe` (merge sonrası):
1. **Kara liste** (`EXCLUDED_TICKERS` — delisted/sorunlu ~35 isim) düşülür
2. **Fiyat post-filtresi** $7-1000 (`post_filter_price_min/max`)
3. **Dedup** (ticker normalize + ilk-görülen kazanır)
4. **Composite momentum skoru** ile sıralama: RVOL %30 + Değişim %25 + Dolar-hacim %25 + Piyasa-değeri %20 + erken-birikim bonusu. **Tavan: 260 hisse** (`max_scan_tickers`).
5. **Rank telemetrisi** (`build_rank_info`): tavanın kestiği ticker'lar + sinyal sıraları stats'a yazılır (huni-kayıp ölçümü için).

> ⚠️ NOT: "$5M dolar-hacim" burada DEĞİL — o Adım 3'teki motor filtresinin hard-gate'i.

---

## ADIM 3-4 — MOTOR: scan_stock (her aday tek tek, 14 elemeli hard-gate zinciri)

**Bu, sistemin kalbi.** `engine.py:scan_stock`, her adayı sırayla şu kapılardan geçirir.
Herhangi biri geçilmezse sinyal İPTAL (reject sayacı artar). Canlı-birebir ölçümde
**motor adayların ~%98'ini eliyor** — asıl seçicilik burada.

| # | Gate | Reject key | Ne yapar |
|---|---|---|---|
| 1 | Yeterli veri | `insufficient_data` | <20 bar reddet |
| 2 | **Evren filtreleri** | `filter_failed` | mcap $250M-10B, **dolar-hacim ≥$5M/gün**, fiyat $7-1000, earnings ±3 gün (float & ATR advisory — v13'te reddetmez, skorlar) |
| 3 | **TETİKLEYİCİ** (birincil) | `no_trigger` | **VCE** (squeeze<0.8 + 20g-breakout + green + MA50 + **ZORUNLU RVOL≥1.5x**) VEYA **RVOL thrust** (RVOL≥2.5x + green + MA20). İkisi de olmazsa reddet |
| 4 | Swing onayı | `swing_not_ready` | 5-gün momentum + MA20 üstü + yükselen dipler |
| 5 | RSI aşırı-alım | `rsi_gate` | RSI > eşik (VCE MUAF — squeeze'de yüksek RSI güç demek, ölçüldü) |
| 6 | Geç-giriş | `late_entry` | Aşırı-uzamış + yüksek RSI (VCE muaf) |
| 7 | OBV dağıtım | `obv_distribution` | Akıllı-para satıyorsa reddet (Type S muaf) |
| 8 | Trend fazı | `trend_phase_weak` | Aşırı düşüş trendi reddet |
| 9 | Dağıtım günü | `distribution_day` | Yüksek hacimli düşüş günü reddet |
| 10-11 | **Weinstein Stage** | `stage_rejected` | Stage 4 (düşüş) VE Stage 3 (dağıtım) hard-reject (Type S muaf) |
| 12 | **Kalite eşiği** | `quality_type_*` | Skor < rejim floor (BULL 78 / CAUTION-BEAR 80) reddet |
| 13 | **R:R gate** | `rr_too_low` | R:R(T2) < min (BULL 1.0 / CAUTION 1.5 / BEAR 2.0) reddet |

### Adım 3'ün İÇİNDEKİ hacim barajı (v32 — Profit Factor 1.43→2.13)
VCE tetikleyicisi, kırılım günü hacmi **50g ortalamanın ≥1.5 katı** değilse ateşlemez
(fakeout filtresi). RVOL≥2.0x TERS çalışır (chase) — 1.5x alt baraj, üst sınır yok.
`signals.py:check_vce_breakout:VCE_MIN_RVOL_GATE`. **Sıralama:** hacim barajı skordan
ÖNCE (trigger içinde) — hacimsiz hisse skorlanmaya bile gelmez.

### SKORLAMA (Gate 12'nin girdisi — `scoring.py`, v33 optimal, DOKUNMA)
- **Katman 1 (6 ağırlık, 100 puan):** Float %25 + Momentum %25 + Trend %15 + ATR %13 + Volume %12 + Risk %10. *(Ağırlık çarpıştırma ölçüldü — mevcut optimal, float/mom kesme hipotezi çürütüldü.)*
- **Katman 2 (bonus, tavan +30):** VCE-premium (+8), tight-coil (+5), golden-cross, sektör-RS, VCP, higher-lows...
- **Katman 3 (ceza):** RSI aşırı-alım, aşırı-uzama, OBV dağıtım, **spread-risk** (dolar-hacim<$7M VE ATR>%8 — S1 proxy, `pen_spread_risk`), zayıf-trend.
- Skor rejimsiz hesaplanır; rejim yalnız EŞİĞİ (Gate 12) belirler.

---

## ADIM 5 — SİNYAL PAKETİ + RİSK YÖNETİMİ

Tüm gate'leri geçen aday `signal` dict'i olur (`engine.py`): ticker, entry, **trigger_pathway**
(vce/rvol), quality, swing_type. Sonra `risk.py:add_risk_management`:
- **Risk/işlem: %1.5 sabit** (`max_risk_per_trade`)
- **Dinamik stop: entry − 2.5×ATR** (type-cap %14-18 ile). Volatil hisse → geniş stop → az lot (otomatik).
- **Pozisyon cap:** tek hisse max %25 (A/C), %20 (B), %15 (S) — `type_position_caps`
- **T1/T2 hedefleri:** ATR-dinamik + kalite/rejim ayarı

---

## ADIM 6 — PENDING → CONFIRM (t+1 disiplini — edge'in koruyucusu)

**Sinyal ANINDA alınmaz.** `tracker.py:add_trade_from_signal` → **PENDING** kaydı:
- **Entry-window guard:** sinyal barının ertesi seans açılışı (t+1 open) geçtiyse trade AÇILMAZ (`entry_window_open`) — ölçülen edge t+1 açılış girişine ait, t+2 ölçülmemiş
- **Duplicate guard:** aktif PENDING/OPEN aynı ticker varsa engelle (`check_duplicate`)
- **Cooldown:** son kapanan trade'den N gün geçmeden aynı ticker'a girme

Ertesi gün `confirm_pending_trades` (5dk'da bir arka planda çalışır):
- Giriş = **t+1 açılış fiyatı**
- **GAP FİLTRESİ:** açılış >+%5 (gap-up, pump/exhaustion) veya <−%7 (gap-down) ise **REJECTED**
- Stop/T1/T2 gerçek açılış fiyatına göre **yeniden hesaplanır** (`CONFIRM_ATR_MULTIPLIER=2.5`)
- Status: PENDING → OPEN

---

## ADIM 7 — EXIT MİMARİSİ (kazananı koştur, kaybedeni küçük tut)

`tracker.py:check_exit_conditions` (her fiyat güncellemesinde, bar-bar):
1. **Stop / trailing stop** (önce — gap-down'da açılıştan gerçekçi dolum)
2. **T1 kısmi çıkış:** hedefe gelince pozisyonun %50'si kapanır + stop **breakeven'a** çekilir (risksiz mod)
3. **T2 / kalan çıkış**
4. **Chandelier trailing:** kalan %50, tepe−3×ATR ile takip edilir (1.5 ATR kâr sonrası aktif)
5. **Timeout:** max 20 işlem günü (`max_holding_days`)

*(Exit ölçümü: geniş stop + trailing, dar stop'a göre EV'yi ~2 katına çıkardı — v14.)*

---

## ADIM 8 — FORWARD-RETURN TAKİBİ (canlı savaş kaydı)

`forward_returns.py:ForwardReturnTracker` (her taramada otomatik, mode=pg Supabase):
- Her sinyal kaydedilir; 3/5/10 işlem günü sonra **gerçek forward-return (R3/R5/R10)** + MFE/MAE otomatik doldurulur
- Giriş konvansiyonu backtest ile AYNI: t+1 açılış
- **Amaç:** canlı gerçek getiriyi backtest vaadiyle (EV +5%) çarpıştırmak — "backtest tuttu mu?"

Ayrıca `signal_history_storage` her tarama koşusunu (stats + sinyaller) saklar,
`regime_storage` rejim geçmişini tutar.

---

## PİPELİNE ÖZET AKIŞ

```
[0] Veri hijyeni (NaN/incomplete bar temizliği, %50 devre kesici)
      ↓
[1] Finviz Q6/Q6b/Q7/Q7b → ~45 aday
      ↓
[2] Kara liste + fiyat filtresi + dedup + composite sıralama (tavan 260)
      ↓
[3-4] scan_stock: 14 hard-gate (filtre→VCE/RVOL trigger [RVOL≥1.5x baraj]→swing→
      RSI→late-entry→OBV→trend→dağıtım→Weinstein→SKOR≥eşik→R:R)  ← %98 eleme
      ↓
[5] Risk yönetimi (%1.5 risk, 2.5×ATR stop, pozisyon cap)
      ↓
[6] PENDING → t+1 açılış + GAP filtresi + entry-window guard → OPEN
      ↓
[7] Exit (T1 kısmi+breakeven → chandelier trailing → timeout 20g)
      ↓
[8] ForwardReturnTracker → Supabase (canlı R3/R5/R10 vs backtest)
```

---

## DURUM: CODE FREEZE (v34) — Forward-Test Dönemi

Backtest optimizasyon fazı **kapalı** (2026-07-27). Sıkılabilecek alfa sıkıldı;
örneklem 44 sinyale düşünce daha fazla parametre ayarı curve-fitting riski.
**Forward-test bitene kadar strateji parametrelerine dokunma** — yalnız gerçek
bug (NaN/500 gibi) düzeltilir. 2-3 hafta sonra ForwardReturnTracker verisiyle
"backtest canlıda tuttu mu?" sorusu parayla ölçülecek.

*Kanıt harness'ları: measure_signal_edge, measure_universe_recall, exit_lab_vce_rvol,
discover_signal_families, backtest_live_replica, measure_score_edge, measure_weight_reallocation.*
