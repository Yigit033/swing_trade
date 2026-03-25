# SmallCap: backtest vs canli scanner paritesi

Walk-forward backtest (`SmallCapBacktester` + `scan_stock(..., backtest_mode=True)`) canli `scan_universe` / API ile **ayni risk motorunu** (`SmallCapRisk.add_risk_management`) ve benzer siralama/rejim esiklerini kullanir. Asagidaki farklar **bilincli veya veri kaynaklidir**; sonuclari yorumlarken dikkate alin.

## Ayni veya hizali olanlar

- Gunluk rejim: SPY/VIX ile tarihsel `regime_from_spy_close`, `effective_scan_thresholds`.
- Sinyal uretimi: `scan_stock` backtest modunda (sentetik float/mcap, earnings atlanir).
- Giris: ertesi gun acilis + slip; stop/hedef/lot `add_risk_management` ile sinyal gunu `df` uzerinden.
- Pozisyon boyutu: `calculate_position_size` (%1.5 risk, tip bazli tavanlar).
- Komisyon: yok (sifir komisyon varsayimi). Slip: `SLIPPAGE_BPS_PER_SIDE`.

## Canlida var, backtestte yok veya farkli

| Konu | Backtest | Canli |
|------|------------|--------|
| **Earnings** | Filtre atlanir (`apply_all_filters` backtest_mode). | `check_earnings` uygulanir. |
| **Katalizor** | Short / insider / haber bonuslari 0. | `CatalystDetector` ile API/yfinance. |
| **Sektor RS** | SPY penceresi varsa proxy; yoksa 0. | `SectorRS.calculate_sector_rs` (ETF). |
| **Rejim ani** | Her gun tarihsel SPY/VIX. | Tarama anindaki rejim (`detect_market_regime`). |
| **Anlati / seviyeler** | Narrative yok. | `narrative`, `technical_levels` (canli). |

## Hedef hesaplama (v5.0 — ATR-dinamik)

- T1 = entry + ATR x tip_carpani (S: 2.5, B: 2.0, A: 1.8, C: 1.5)
- T2 = T1 carpani x T2_ATR_RATIO (BULL: 2.0, CAUTION: 1.6, BEAR: 1.05)
- Quality boost: Q >= 85 ise x 1.15, Q 75-84 ise x 1.08
- Sabit yuzde tablosu (`TYPE_TARGET_CAPS`) tavan olarak korunur.
- 1.5R taban: T1 her zaman en az 1.5 x risk mesafesindedir.
- Canli tarayicida rejim henuz `scan_stock` icinde bilinmediginden T2 rejim ayari yalnizca **backtest** icin aktiftir.

## Tip C sikilastirmasi (v5.0)

- Stop cap %6 (eskisi %8). min_quality Tip C icin 70 (diger tipler eff_min).
- ATR carpani zaten en dusuk (1.5).

## Erken trailing stop (v5.0, backtest)

- 1 ATR kazanc: stop entry - 0.5*ATR'ye cekilir.
- 1.5 ATR kazanc: stop entry'e tasinir (breakeven).
- 2+ ATR kazanc: ATR-step mantigi (eski kural korunur).
- Partial-oncesi trade'ler de korunur.

## Cikis modeli (backtest)

- T1'de **kismi cikis** (`PARTIAL_AT_T1_FRACTION`, varsayilan %50), kalan icin stop **breakeven**e yukseltilir, hedef **T2**'ye tasinir; sonra stop / T2 / timeout / zorunlu kapanis.
- `MIN_SHARES_FOR_PARTIAL` altinda tek seferde T1'de tam cikis.
- API yanitindaki **`diagnostics`**: tarama adayi sayisi, pending, gap/Tip C/risk/R:R ile elenen girisler, acilan pozisyon — funnel teshisi icin.

## Onerilen kullanim

- Ayni ticker listesi ve `min_quality` / `top_n` ile canli beklentiyi yaklastirin.
- Sonuclari "canli kopyasi" degil "ayni kurallarin gecmis simulasyonu" olarak okuyun.
- Iyilestirme onceligi: once **giris kalitesi ve cikis istatistikleri** (`diagnostics`), sonra hedef/stop ince ayari.

## Backtest sonrasi: diagnostics kontrol listesi

Her kosudan sonra once **`diagnostics`** (API JSON `diagnostics` veya backtest sayfasindaki teshis satiri), sonra **`metrics.exit_stats`** veya trade tablosuna bakin.

| Soru | Nerede | Alan / nasil |
|------|--------|----------------|
| Kac sinyal R:R yuzunden elendi? | `diagnostics` | `entry_skip_rr` |
| Kaci yukari gap yuzunden giremedi? | `diagnostics` | `entry_skip_gap_up` |
| Tip C filtresi kac girisi kesti? | `diagnostics` | `entry_skip_tip_c` |
| Kac trade T2 ile kapandi? | `metrics.exit_stats` | `TARGET_T2` -> `count` (UI: Cikis analizi **Hedef T2**) |

Tam huni (ayni yanitta):

- `signals_passed_rsi` — RSI sonrasi aday gunleri (sayac)
- `pending_queued` — kuyruga alinan giris denemeleri
- `entry_skip_gap_down` — asagi gap ile elenenler
- `entry_skip_risk` — risk/lot vb. ile elenenler
- `entries_opened` — fiilen acilan pozisyon sayisi

`entry_skip_rr` surekli 0 ise, o donemde **MIN_RR_AT_ENTRY** fiilen neredeyse hic tetiklenmiyor demektir.

## Ilgili notlar

- [BACKTEST_AB_BASELINE.md](./BACKTEST_AB_BASELINE.md) — eski vs yeni kod icin ayni donemde JSON A/B karsilastirmasi.
- [SMALLCAP_TYPE_C_REVIEW.md](./SMALLCAP_TYPE_C_REVIEW.md) — Tip C skoru, esik ve deney onerileri.
