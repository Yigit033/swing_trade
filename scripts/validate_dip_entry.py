# -*- coding: utf-8 -*-
"""
DİP GİRİŞİ DOĞRULAMASI — tam evren, kalite bantları, dönem tutarlılığı
================================================================================
Bulgu (measure_intraday_entry_v2): teyitli sinyalde ertesi gün AÇILIŞ yerine
DİPTE almak (limit = açılış − k×ATR) EV'yi artırıyor. Ama o ölçüm yalnız 45
ticker / 50 işlemle yapıldı — canlıya almak için fazla dar.

Bu doğrulama iki kusuru gideriyor:
  1. EVREN: dip girişi yalnız GÜNLÜK Open/Low kullanır, saatlik veri gerekmez.
     Dolayısıyla 45 ticker sınırı gereksizdi — tam 995 tickerlı tarafsız
     evrende (S&P400+600, batmışlar dahil) ölçülüyor.
  2. FİYAT HATASI: açılış limitin ALTINDAysa alış-limit emri açılışta dolar
     (daha iyi fiyat), limitte değil. v2 limitte doldurup dip girişini
     KENDİ ALEYHİNE hesaplıyordu. Düzeltildi: fill = min(açılış, limit).

DOĞRULAMA MANTIĞI — tek sayı değil, TUTARLILIK aranır:
  · Kalite bantları (tümü / Q70+ / Q80+) — etki yalnız Q80'de çıkıyorsa şüpheli
  · TRAIN/OOS — yön değişiyorsa kabul edilmez
  · Yıl yıl — tek bir yıla dayanıyorsa kabul edilmez
  · Eşleştirilmiş karşılaştırma — hep AYNI işlemlerde "açılıştan alsaydık"
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")

OOS_SPLIT = "2025-06-01"
PRICE_MAX = 200.0
DIP_KS = (0.25, 0.50, 0.75, 1.00)
CACHE = "output/dip_entry_signals.json"


def collect():
    from swing_trader.small_cap.engine import SmallCapEngine
    from backtest_live_replica import enrich, build_regime_map, EXIT_NEW, simulate_from_entry
    from collect_signal_lab import finviz_hit

    print("Motor tüm evrende koşuyor (995 ticker)...", flush=True)
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    shares = json.load(open("output/_shares_broad.json"))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)

    engine = SmallCapEngine()
    out = []
    for i, (tk, rdf) in enumerate(raw.items()):
        df = enrich(rdf)
        sh = shares.get(tk, {})
        so, fl = sh.get("shares"), sh.get("float")
        o = df["Open"].astype(float).values
        lo = df["Low"].astype(float).values
        atr = df["atr"].astype(float).values
        for t in range(60, len(df) - 21):
            row = df.iloc[t]
            close = float(row["Close"])
            if close > PRICE_MAX:
                continue
            mc = close * so if so else None
            if not finviz_hit(row, mc):
                continue
            day = pd.to_datetime(df["Date"].iloc[t]).normalize()
            sl = spy[spy["_d"] <= day].tail(60)
            try:
                s = engine.scan_stock(
                    tk, df.iloc[:t + 1],
                    stock_info={"ticker": tk, "marketCap": int(mc) if mc else 0,
                                "floatShares": int(fl) if fl else 0,
                                "shortName": tk, "sector": "Unknown"},
                    backtest_mode=True, portfolio_value=10000,
                    spy_df_window=sl if len(sl) >= 6 else None,
                    regime=rmap.get(day, "UNKNOWN"))
            except Exception:
                s = None
            if not s:
                continue
            a = simulate_from_entry(df, t, EXIT_NEW, o[t + 1])
            if a is None:
                continue
            rec = {"tk": tk, "date": str(day.date()),
                   "q": float(s.get("quality_score", 0) or 0), "A": a}
            for k in DIP_KS:
                lim = o[t + 1] - k * atr[t]
                if lo[t + 1] <= lim:
                    # DÜZELTME: açılış limitin altındaysa fill AÇILIŞTA olur
                    fill = min(o[t + 1], lim)
                    b = simulate_from_entry(df, t, EXIT_NEW, fill)
                    rec["B%.2f" % k] = b
                else:
                    rec["B%.2f" % k] = None
            out.append(rec)
        if (i + 1) % 200 == 0:
            print("  ...%d/%d (%d sinyal)" % (i + 1, len(raw), len(out)), flush=True)
    json.dump(out, open(CACHE, "w"))
    return out


def ev(xs):
    return float(np.mean(xs)) if len(xs) else 0.0


def wr(xs):
    return float((np.array(xs) > 0).mean() * 100) if len(xs) else 0.0


def report(recs, label, qmin):
    sub = [r for r in recs if r["q"] >= qmin]
    if not sub:
        return
    print("\n" + "=" * 100)
    print("  %s — n=%d sinyal" % (label, len(sub)))
    print("=" * 100)
    print("  %-22s%7s%9s%9s%7s%11s%11s" %
          ("kurgu", "dolan", "doluluk", "EV", "WR", "TRAIN EV", "OOS EV"))
    print("  " + "-" * 96)
    allA = [r["A"] for r in sub]
    print("  %-22s%7d%8.0f%%%+9.2f%%%6.0f%%%+10.2f%%%+10.2f%%" %
          ("A  t+1 açılış", len(allA), 100.0, ev(allA), wr(allA),
           ev([r["A"] for r in sub if r["date"] < OOS_SPLIT]),
           ev([r["A"] for r in sub if r["date"] >= OOS_SPLIT])))
    for k in DIP_KS:
        key = "B%.2f" % k
        fil = [r for r in sub if r.get(key) is not None]
        if not fil:
            continue
        b = [r[key] for r in fil]
        m = [r["A"] for r in fil]                       # EŞLEŞTİRİLMİŞ
        btr = [r[key] for r in fil if r["date"] < OOS_SPLIT]
        bte = [r[key] for r in fil if r["date"] >= OOS_SPLIT]
        mtr = [r["A"] for r in fil if r["date"] < OOS_SPLIT]
        mte = [r["A"] for r in fil if r["date"] >= OOS_SPLIT]
        print("  %-22s%7d%8.0f%%%+9.2f%%%6.0f%%%+10.2f%%%+10.2f%%" %
              ("B  dip −%.2f ATR" % k, len(b), 100.0 * len(b) / len(sub),
               ev(b), wr(b), ev(btr), ev(bte)))
        print("  %-22s%7s%8s%+9.2f%%%6.0f%%%+10.2f%%%+10.2f%%   Δ %+.2f (TR %+.2f / OOS %+.2f)" %
              ("     aynı işlemde A", "", "", ev(m), wr(m), ev(mtr), ev(mte),
               ev(b) - ev(m), ev(btr) - ev(mtr), ev(bte) - ev(mte)))


def per_year(recs, qmin, k):
    key = "B%.2f" % k
    sub = [r for r in recs if r["q"] >= qmin and r.get(key) is not None]
    if not sub:
        return
    print("\n  YIL YIL TUTARLILIK — Q%d+, dip −%.2f ATR" % (qmin, k))
    print("  %-8s%7s%11s%11s%9s" % ("yıl", "n", "dip EV", "açılış EV", "fark"))
    years = sorted({r["date"][:4] for r in sub})
    for y in years:
        g = [r for r in sub if r["date"][:4] == y]
        b, m = [r[key] for r in g], [r["A"] for r in g]
        print("  %-8s%7d%+10.2f%%%+10.2f%%%+9.2f" % (y, len(g), ev(b), ev(m), ev(b) - ev(m)))


def main():
    recs = json.load(open(CACHE)) if os.path.exists(CACHE) else collect()
    print("\nToplam sinyal: %d" % len(recs))
    for qmin, label in ((0, "TÜM SİNYALLER (kalite eşiği yok)"),
                        (70, "Q70+"), (80, "Q80+ (canlı eşik)")):
        report(recs, label, qmin)
    per_year(recs, 0, 0.25)
    per_year(recs, 80, 0.25)


if __name__ == "__main__":
    main()
