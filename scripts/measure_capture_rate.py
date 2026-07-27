# -*- coding: utf-8 -*-
"""
KAZANDIRAN HİSSE YAKALAMA ORANI — "Kazandıran hisseyi buluyor muyuz?"
=====================================================================
Kullanıcının net sorusu: kaç günde kazandırdığı ÖNEMLİ DEĞİL — kazandıran
hisseyi sistem yakaladı mı, kaçırdı mı? (Midas'ta yükselen hisse görüp "biz
neden bulamadık" endişesi.)

İki "kazandıran fırsat" tanımı (kullanıcı ikisini de istedi):
  A) 15 işgününde %20+ yükseliş
  B) 30 işgününde %30+ yükseliş

Yöntem (canlı-birebir — gerçek ürünü test eder):
  1. Her hisse, her gün: bu günden sonra A/B fırsatı BAŞLADI mı?
     (başlangıç = önceki gün fırsat DEĞİLken bugün fırsat olan gün — art arda
      günleri tek fırsat say, şişirme yok)
  2. Fırsat başlangıç günlerinde GERÇEK sistemi koştur:
     - Finviz emülasyonu (Q6/Q6b/Q7/Q7b) geçer mi?  [aday havuzuna girer mi]
     - Geçerse scan_stock (tüm gate'ler) sinyal üretir mi?
     - Üretirse yeni eşiği (regime floor) geçer mi?
  3. Her aşamada kaç fırsat elendi → NEREDE kaçırıyoruz?

Çıktı: yakalama oranı + kaçırma aşama dökümü (Finviz / motor-gate / eşik).
Cache: _broad_data.pkl + _shares_broad.json + _edge_spy.pkl
"""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.ERROR)
logging.disable(logging.CRITICAL)
import numpy as np
import pandas as pd

from swing_trader.small_cap.engine import SmallCapEngine
from swing_trader.small_cap.settings_config import load_settings
from backtest_live_replica import enrich, finviz_hit, build_regime_map

DEFS = [("15g/%20", 15, 20.0), ("30g/%30", 30, 30.0)]


def find_opportunity_starts(df, window, pct):
    """Fırsat BAŞLANGIÇ günlerinin indekslerini döndür (art arda tekilleştirilmiş).
    Fırsat = bugünden sonraki `window` günde high, bugünün close'una göre %pct+ arttı."""
    c = df["Close"].astype(float).values
    h = df["High"].astype(float).values
    n = len(df)
    is_opp = np.zeros(n, dtype=bool)
    for t in range(60, n - 1):
        end = min(t + 1 + window, n)
        if end <= t + 1:
            continue
        fwd_max = (h[t + 1:end].max() / c[t] - 1) * 100
        is_opp[t] = fwd_max >= pct
    # başlangıç = is_opp True ve önceki gün False
    starts = [t for t in range(60, n - 1) if is_opp[t] and not is_opp[t - 1]]
    return starts


def main():
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    shares = json.load(open("output/_shares_broad.json"))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)
    data = {t: enrich(df) for t, df in raw.items()}
    engine = SmallCapEngine()
    rt = load_settings().regime_thresholds
    floor = {"BULL": rt.bull_min_quality, "CAUTION": rt.caution_other_min_quality,
             "BEAR": rt.bear_tentative_min_quality}

    for label, window, pct in DEFS:
        total = 0
        finviz_ok = 0      # Finviz emülasyonuna girdi
        signal_ok = 0      # motor sinyal üretti (gate'leri geçti)
        threshold_ok = 0   # eşiği de geçti = YAKALANDI
        miss_finviz = 0    # Finviz'e hiç girmedi
        miss_gate = 0      # Finviz'e girdi ama motor gate eledi
        miss_thresh = 0    # motor üretti ama eşik eledi

        for tk, df in data.items():
            sh = shares.get(tk, {})
            sh_out, flt = sh.get("shares"), sh.get("float")
            starts = find_opportunity_starts(df, window, pct)
            for t in starts:
                total += 1
                row = df.iloc[t]
                close = float(row["Close"])
                mcap = close * sh_out if sh_out else None
                # AŞAMA 1: Finviz emülasyonu
                if not finviz_hit(row, mcap):
                    miss_finviz += 1
                    continue
                finviz_ok += 1
                # AŞAMA 2: gerçek motor
                day = pd.to_datetime(df["Date"].iloc[t]).normalize()
                reg = rmap.get(day, "UNKNOWN")
                df_win = df.iloc[:t + 1]
                spy_slice = spy[spy["_d"] <= day].tail(60)
                stock_info = {"ticker": tk, "marketCap": int(mcap) if mcap else 0,
                              "floatShares": int(flt) if flt else 0, "shortName": tk, "sector": "Unknown"}
                try:
                    sig = engine.scan_stock(tk, df_win, stock_info=stock_info, backtest_mode=True,
                                            portfolio_value=10000,
                                            spy_df_window=spy_slice if len(spy_slice) >= 6 else None,
                                            regime=reg)
                except Exception:
                    sig = None
                if not sig:
                    miss_gate += 1
                    continue
                signal_ok += 1
                # AŞAMA 3: eşik (scan_stock zaten motor-içi floor uyguluyor ama
                # sig döndüyse floor'u geçmiş demektir; yine de netlik için kontrol)
                q = sig.get("quality_score", 0)
                if q >= floor.get(reg, 78):
                    threshold_ok += 1
                else:
                    miss_thresh += 1

        print("\n" + "=" * 76)
        print(f"  KAZANDIRAN FIRSAT YAKALAMA — tanım: {label}")
        print("=" * 76)
        print(f"  Toplam kazandıran fırsat başlangıcı: {total}")
        if total == 0:
            continue
        print(f"  → YAKALANDI (sinyal üretildi + eşik geçti): {threshold_ok} ({threshold_ok/total*100:.0f}%)")
        print(f"  → KAÇIRILDI: {total-threshold_ok} ({(total-threshold_ok)/total*100:.0f}%)")
        print(f"\n  KAÇIRMA AŞAMA DÖKÜMÜ (nerede kaybediyoruz?):")
        print(f"    Finviz'e hiç girmedi:        {miss_finviz} ({miss_finviz/total*100:.0f}%)")
        print(f"    Finviz girdi, motor-gate eledi: {miss_gate} ({miss_gate/total*100:.0f}%)")
        print(f"    Motor üretti, EŞİK eledi:    {miss_thresh} ({miss_thresh/total*100:.0f}%)")
        print(f"\n  NOT: 'Finviz'e girmedi' = hisse o gün 20g-zirve/RVOL kriterine uymuyordu")
        print(f"       (VCE/RVOL tezimizin doğası — yavaş tırmanış bu kritere uymaz).")

    print("\n" + "=" * 76)
    print("  YORUM: 'Finviz girdi ama gate/eşik eledi' YÜKSEKSE → sistemde ayar sorunu.")
    print("         'Finviz'e hiç girmedi' yüksekse → o hareketler zaten bizim")
    print("         tezimize (sıkışma-kırılım / hacim-patlaması) uymayan tiptedir.")


if __name__ == "__main__":
    main()
