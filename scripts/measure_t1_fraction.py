# -*- coding: utf-8 -*-
"""
T1 KISMİ ORANI — canlı %50 vs harness %33: hangisi doğru? (PARİTE KIRIĞI)
================================================================================
BULGU (2026-08-04 denetimi): canlı tracker T1'de pozisyonun **%50**'sini satıyor
(tracker.py "T1 partial 50%"), backtest harness'ı ise **%33** (EXIT_NEW t1_frac=0.33).
Yani bugüne kadar verdiğim tüm EV sayıları (+2.36%, +3.00%, maks-fiyat kararı)
%33 varsayımıyla ölçüldü — canlı farklı davranıyor. Parite kırıkken hiçbir
ölçüme tam güvenilmez, bu yüzden önce bunu çözüyoruz.

YÖN BEKLENTİSİ (ölçüm öncesi yazıldı, sonuca göre kural uydurmamak için):
  %50 sat → T1'de daha çok kâr kilitlenir, T2'ye daha az pozisyon taşınır
            → küçük kazançlar daha güvenli, BÜYÜK kazananlardan daha az pay
  %33 sat → tersi: daha fazla pozisyon trailing'e kalır, sağ kuyruk daha güçlü
Swing trade'de getiri dağılımı sağa çarpık (birkaç büyük kazanan taşır), bu
yüzden TEORİ %33'ü favoriler. Ama teori karar vermez, ölçüm verir.

KARAR KURALI: kazanan oran hem TÜM örneklemde hem OOS'ta önde olmalı. Aksi
halde parite CANLIYA (%50) hizalanır — çünkü canlı zaten öyle çalışıyor ve
ölçülmemiş bir değişiklik yapmak yerine ölçümü gerçeğe uydurmak doğrudur.

Girdi: output/signal_lab.json'daki sinyal listesi YETMEZ (getiri zaten
hesaplanmış) → exit simülasyonunu yeniden koşmak için ham veri gerekir.
Cache: output/_broad_data.pkl + _shares_broad.json + _edge_spy.pkl
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.ERROR)
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd

from swing_trader.small_cap.engine import SmallCapEngine
from backtest_live_replica import enrich, simulate, EXIT_NEW, build_regime_map
from collect_signal_lab import finviz_hit

OOS_SPLIT = "2025-06-01"
Q_LIVE = 80.0
PRICE_MAX = 200.0
CACHE = "output/t1_fraction.json"

FRACTIONS = [0.25, 0.33, 0.50, 0.66, 1.00]   # 1.00 = T1'de tamamen çık


def stats(rows, key):
    if not rows:
        return dict(n=0, ev=0.0, wr=0.0, pf=0.0, p90=0.0)
    a = np.array([r[key] for r in rows])
    w, l = a[a > 0], a[a <= 0]
    pf = (w.sum() / abs(l.sum())) if l.size and l.sum() != 0 else float("inf")
    return dict(n=len(a), ev=float(a.mean()), wr=float((a > 0).mean() * 100),
                pf=float(pf), p90=float(np.percentile(a, 90)))


def fmt(s):
    if s["n"] == 0:
        return "n=0"
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return f"n={s['n']:<4} EV {s['ev']:+6.2f}%  WR {s['wr']:3.0f}%  PF {pf:>5}  p90 {s['p90']:+6.1f}%"


def collect():
    print("Veri yukleniyor...", flush=True)
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    shares = json.load(open("output/_shares_broad.json"))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)
    data = {t: enrich(df) for t, df in raw.items()}
    engine = SmallCapEngine()

    recs = []
    tickers = list(data.keys())
    for ti, tk in enumerate(tickers):
        df = data[tk]
        sh = shares.get(tk, {})
        sh_out, flt = sh.get("shares"), sh.get("float")
        n = len(df)
        for t in range(60, n - 21):
            row = df.iloc[t]
            close = float(row["Close"])
            if close > PRICE_MAX:
                continue
            mcap = close * sh_out if sh_out else None
            if not finviz_hit(row, mcap):
                continue
            day = pd.to_datetime(df["Date"].iloc[t]).normalize()
            reg = rmap.get(day, "UNKNOWN")
            spy_slice = spy[spy["_d"] <= day].tail(60)
            info = {"ticker": tk, "marketCap": int(mcap) if mcap else 0,
                    "floatShares": int(flt) if flt else 0,
                    "shortName": tk, "sector": "Unknown"}
            try:
                sig = engine.scan_stock(
                    tk, df.iloc[:t + 1], stock_info=info, backtest_mode=True,
                    portfolio_value=10000,
                    spy_df_window=spy_slice if len(spy_slice) >= 6 else None,
                    regime=reg)
            except Exception:
                sig = None
            if not sig:
                continue
            rec = {"tk": tk, "date": str(day.date()),
                   "q": float(sig.get("quality_score", 0) or 0)}
            ok = True
            for fr in FRACTIONS:
                cfg = dict(EXIT_NEW)
                cfg["t1_frac"] = fr
                r = simulate(df, t, cfg)
                if r is None:
                    ok = False
                    break
                rec[f"f{int(fr*100)}"] = float(r)
            if ok:
                recs.append(rec)
        if (ti + 1) % 200 == 0:
            print(f"  ...{ti+1}/{len(tickers)} ({len(recs)} sinyal)", flush=True)

    json.dump(recs, open(CACHE, "w"), default=str)
    return recs


def main():
    recs = json.load(open(CACHE)) if os.path.exists(CACHE) else collect()
    q80 = [r for r in recs if r["q"] >= Q_LIVE]
    W = 100
    print("\n" + "=" * W)
    print(f"  T1 KISMİ ORANI — {len(q80)} Q80+ sinyal | canli %50, harness %33")
    print("=" * W)
    print(f"  {'oran':<12}{'TUM ORNEKLEM':<52}{'not'}")
    print("  " + "-" * (W - 4))
    best = None
    for fr in FRACTIONS:
        k = f"f{int(fr*100)}"
        s = stats(q80, k)
        note = ""
        if abs(fr - 0.50) < 1e-9:
            note = "← CANLI"
        elif abs(fr - 0.33) < 1e-9:
            note = "← harness"
        if best is None or s["ev"] > best[1]["ev"]:
            best = (fr, s)
        print(f"  %{int(fr*100):<11}{fmt(s):<52}{note}")

    print(f"\n  {'oran':<12}{'TRAIN':<40}{'OOS (test)':<40}")
    print("  " + "-" * (W - 4))
    tr = [r for r in q80 if r["date"] < OOS_SPLIT]
    te = [r for r in q80 if r["date"] >= OOS_SPLIT]
    oos_best = None
    for fr in FRACTIONS:
        k = f"f{int(fr*100)}"
        a, b = stats(tr, k), stats(te, k)
        if oos_best is None or b["ev"] > oos_best[1]["ev"]:
            oos_best = (fr, b)
        a_s = "EV %+.2f%% WR %.0f%% (n=%d)" % (a["ev"], a["wr"], a["n"])
        b_s = "EV %+.2f%% WR %.0f%% (n=%d)" % (b["ev"], b["wr"], b["n"])
        print(f"  %{int(fr*100):<11}{a_s:<40}{b_s:<40}")

    print("\n" + "=" * W)
    live = stats(q80, "f50")
    harn = stats(q80, "f33")
    print(f"  CANLI (%50)   : {fmt(live)}")
    print(f"  HARNESS (%33) : {fmt(harn)}")
    print(f"  Fark (33-50)  : {harn['ev'] - live['ev']:+.2f} puan")
    print(f"\n  En iyi (tum)  : %{int(best[0]*100)}  EV {best[1]['ev']:+.2f}%")
    print(f"  En iyi (OOS)  : %{int(oos_best[0]*100)}  EV {oos_best[1]['ev']:+.2f}%")
    print("  " + "-" * (W - 4))
    if best[0] == oos_best[0]:
        print(f"  SONUC: %{int(best[0]*100)} hem tum ornekle hem OOS'ta kazandi → PARITE bu degere hizalanmali.")
    else:
        print("  SONUC: tum-orneklem ve OOS ayni orani secmedi → net kazanan YOK.")
        print("         Parite CANLIYA (%50) hizalanir: olculmemis degisiklik yapmak yerine")
        print("         olcumu gercege uydurmak dogrudur.")
    print("=" * W)


if __name__ == "__main__":
    main()
