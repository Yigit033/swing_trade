# -*- coding: utf-8 -*-
"""
SKOR-EDGE ÖLÇÜMÜ — "Kalite skoru gerçekten kazananı kaybedenden ayırıyor mu?"
============================================================================
Kullanıcının #1 sorusu: scoring.py'deki 6 bileşen + ~20 bonus + ~15 ceza
GERÇEKTEN para kazandırıyor mu, yoksa süslü gürültü mü (ve kazananları yanlışlıkla
mı eliyor)?

Yöntem: canlı-birebir backtest (backtest_live_replica.py) gibi GERÇEK motoru
(scan_stock, tüm gate'ler) Finviz-emüle evrende koştur — AMA her sinyal için
quality_score + ham bileşen skorlarını da kaydet. Sonra:
  1. Sinyalleri kalite skoruna göre bucket'la (Q60-70, 70-80, 80-90, 90+)
     → yüksek skorlu bucket'lar daha çok mu kazanıyor? (skor monoton mu?)
  2. Her ham bileşenin (volume/atr/float/momentum/trend/risk) forward-return
     ile korelasyonu → hangi bileşen edge'i açıklıyor, hangisi gürültü?
  3. Bonus/ceza toplamının getiriyle ilişkisi.

Kanıt: Bucket'lar arası fark yoksa/karışıksa → skor edge ayırmıyor → basitleştir.
Monoton artıyorsa → skor haklı.

Cache: _broad_data.pkl + _shares_broad.json + _edge_spy.pkl
Çıktı: output/score_edge.json (sinyal-başı skor+bileşen+getiri kayıtları)
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
from swing_trader.small_cap.regime_logic import regime_from_spy_close
from backtest_live_replica import enrich, finviz_hit, simulate, EXIT_NEW, build_regime_map


def main():
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    shares = json.load(open("output/_shares_broad.json"))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)
    data = {t: enrich(df) for t, df in raw.items()}
    engine = SmallCapEngine()

    records = []
    tickers = list(data.keys())
    for ti, tk in enumerate(tickers):
        df = data[tk]
        sh = shares.get(tk, {})
        sh_out, flt = sh.get("shares"), sh.get("float")
        n = len(df)
        for t in range(60, n - 21):
            row = df.iloc[t]
            close = float(row["Close"])
            mcap = close * sh_out if sh_out else None
            if not finviz_hit(row, mcap):
                continue
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
                continue
            r = simulate(df, t, EXIT_NEW)
            if r is None:
                continue
            records.append({
                "q": float(sig.get("quality_score", 0)),
                "r": r,
                "pw": sig.get("trigger_pathway", "vce_breakout"),
                "reg": reg,
                "date": str(day.date()),
                # ham bileşenler (scan_stock signal dict'inde bazıları var)
                "vol_surge": float(sig.get("volume_surge", 0)),
                "atr_pct": float(sig.get("atr_percent", 0)),
                "float_m": float(sig.get("float_millions", 0)),
                "rsi": float(sig.get("rsi", 50)),
                "5d": float(sig.get("five_day_return", 0)),
                "sector_rs": float(sig.get("sector_rs_score", 0)),
                "swing_type": sig.get("swing_type", "A"),
            })
        if (ti + 1) % 250 == 0:
            print(f"  ...{ti+1}/{len(tickers)} ({len(records)} sinyal)", flush=True)

    json.dump(records, open("output/score_edge.json", "w"), default=str)
    a = np.array([x["r"] for x in records])
    q = np.array([x["q"] for x in records])
    print(f"\n{'='*74}")
    print(f"  SKOR-EDGE ÖLÇÜMÜ — {len(records)} gerçek sinyal (motor + Finviz-emüle)")
    print(f"{'='*74}")
    print(f"  Genel: EV {a.mean():+.2f}%  WR {(a>0).mean()*100:.0f}%  |  skor: medyan {np.median(q):.0f}, "
          f"min {q.min():.0f}, max {q.max():.0f}")

    # [1] KALİTE BUCKET'LARI — skor monoton mu?
    print(f"\n  [1] KALİTE SKORU BUCKET'LARI (yüksek skor = daha çok kazanç MI?)")
    print(f"  {'bucket':<12}{'n':>5}{'EV':>9}{'WR':>7}{'medyan':>9}")
    buckets = [(0,60),(60,70),(70,80),(80,90),(90,200)]
    for lo,hi in buckets:
        m = (q>=lo)&(q<hi)
        if m.sum()==0: continue
        sub=a[m]
        print(f"  Q{lo}-{hi if hi<200 else '+':<7} {m.sum():>5}{sub.mean():>+8.2f}%{(sub>0).mean()*100:>6.0f}%{np.median(sub):>+8.1f}%")

    # korelasyon: skor ↔ getiri
    if len(a)>10:
        corr = np.corrcoef(q, a)[0,1]
        print(f"\n  Skor↔Getiri korelasyonu: {corr:+.3f}  "
              f"({'skor edge AYIRIYOR' if corr>0.1 else 'skor edge AYIRMIYOR (gürültü)' if abs(corr)<0.05 else 'zayıf'})")

    # [2] HAM BİLEŞEN korelasyonları
    print(f"\n  [2] HAM BİLEŞEN ↔ GETİRİ korelasyonu (hangisi edge açıklıyor?)")
    for key,lbl in [("vol_surge","Hacim patlaması"),("atr_pct","Volatilite ATR%"),
                    ("float_m","Float (M)"),("rsi","RSI"),("5d","5-gün getiri"),
                    ("sector_rs","Sektör RS")]:
        vals=np.array([x[key] for x in records])
        if vals.std()>0:
            c=np.corrcoef(vals,a)[0,1]
            print(f"  {lbl:<20}: {c:+.3f}")

    print(f"\n  📁 output/score_edge.json")


if __name__ == "__main__":
    main()
