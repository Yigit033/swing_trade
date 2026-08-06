# -*- coding: utf-8 -*-
"""
GÜN İÇİ GİRİŞ ÖLÇÜMÜ — "kırılım anında girmek, ertesi açılışta girmekten iyi mi?"
================================================================================
Kullanıcı sorusu (2026-08-06): güçlü bir hisse gün içinde geri çekilirken alsak
daha iyi para kazanmaz mıyız? Ayrıca kırılım anında girsek ertesi günün
gecelik boşluğunu ödemeyiz.

Bu soru FİKİRLE kapatılamaz — ölçülür. Elimizde saatlik bar var (yfinance 730g,
2023-09'a kadar), yani tüm sinyal dönemimizi kapsıyor.

DENEY TASARIMI — tek değişken: GİRİŞ ANI
  Sinyaller: mevcut motorun ürettiği GERÇEK VCE/RVOL sinyalleri (aynı küme).
  Çıkış: HERKESTE AYNI (canlı exit — stop 2.5xATR, T1 %33 + BE, chandelier,
         20 gün timeout, slippage dahil). Böylece fark yalnız girişten gelir.

  A  t+1 AÇILIŞ            mevcut sistem — referans
  B  t+1 GÜN İÇİ DİP       limit = t+1 açılış − k×ATR (k=0.25/0.50/1.00)
                           gün içinde değmezse İŞLEM YOK (fırsat kaybı sayılır)
  C  SİNYAL GÜNÜ GÜN İÇİ   kırılımın gerçekleştiği saatte al (TEYİTSİZ —
                           bar kapanışta geçersizleşebilir; fakeout riski dahil)
  D  t+1 KAPANIŞ           sabırlı ama dipsiz

DÜRÜSTLÜK KURALLARI
  · B'de limit değmezse işlem AÇILMAZ ve getiri 0 sayılmaz — o sinyal kaçar.
    Ortalama alırken "kaçan"ları saymamak B'yi haksız şişirir; hem doluluk
    oranı hem TOPLAM getiri raporlanır.
  · C teyitsizdir: o saatte 20g zirve aşılmış olabilir ama gün kapanışta
    altına düşebilir. Bu senaryolar C'de İŞLEM olarak sayılır (gerçek maliyet).
  · TRAIN/OOS ayrı raporlanır; yön tutmuyorsa sonuç KABUL EDİLMEZ.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")

OOS_SPLIT = "2025-06-01"
Q_LIVE = 80.0
PRICE_MAX = 200.0
SIG_CACHE = "output/intraday_signals.json"
HOUR_CACHE = "output/_hourly_bars.pkl"
DIP_KS = (0.25, 0.50, 1.00)


# ── 1) Sinyalleri MEVCUT motorla üret ────────────────────────────────────
def build_signals():
    from swing_trader.small_cap.engine import SmallCapEngine
    from backtest_live_replica import enrich, build_regime_map
    from collect_signal_lab import finviz_hit

    print("Sinyaller mevcut motorla üretiliyor...", flush=True)
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
            if s and float(s.get("quality_score", 0) or 0) >= Q_LIVE:
                out.append({"tk": tk, "date": str(day.date()), "t": t,
                            "q": float(s["quality_score"])})
        if (i + 1) % 200 == 0:
            print("  ...%d/%d (%d sinyal)" % (i + 1, len(raw), len(out)), flush=True)
    json.dump(out, open(SIG_CACHE, "w"))
    return out


# ── 2) Saatlik barları TAZE indir ────────────────────────────────────────
def fetch_hourly(tickers):
    import yfinance as yf
    print("Saatlik bar indiriliyor (%d ticker)..." % len(tickers), flush=True)
    store = {}
    for i, tk in enumerate(tickers, 1):
        for attempt in range(2):
            try:
                d = yf.Ticker(tk).history(period="730d", interval="1h", auto_adjust=False)
                if d is not None and len(d):
                    d = d[["Open", "High", "Low", "Close", "Volume"]].copy()
                    d.index = pd.to_datetime(d.index)
                    store[tk] = d
                break
            except Exception:
                time.sleep(1.0)
        if i % 25 == 0:
            print("  ...%d/%d (%d basarili)" % (i, len(tickers), len(store)), flush=True)
    pickle.dump(store, open(HOUR_CACHE, "wb"))
    return store


def main():
    sigs = json.load(open(SIG_CACHE)) if os.path.exists(SIG_CACHE) else build_signals()
    print("Q%.0f+ sinyal: %d" % (Q_LIVE, len(sigs)))
    if not sigs:
        return

    tickers = sorted({s["tk"] for s in sigs})
    hourly = (pickle.load(open(HOUR_CACHE, "rb"))
              if os.path.exists(HOUR_CACHE) else fetch_hourly(tickers))
    print("saatlik veri gelen ticker: %d/%d" % (len(hourly), len(tickers)))

    from backtest_live_replica import enrich, EXIT_NEW
    import backtest_live_replica as B
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    daily = {tk: enrich(raw[tk]) for tk in tickers if tk in raw}

    rows = []
    for s in sigs:
        tk, t = s["tk"], s["t"]
        df = daily.get(tk)
        hb = hourly.get(tk)
        if df is None or hb is None or t + 1 >= len(df):
            continue
        sig_day = pd.to_datetime(df["Date"].iloc[t]).date()
        nxt_day = pd.to_datetime(df["Date"].iloc[t + 1]).date()
        atr = float(df["atr"].iloc[t])
        if atr <= 0:
            continue

        hh = hb[hb.index.date == nxt_day]
        hsig = hb[hb.index.date == sig_day]
        if len(hh) == 0:
            continue

        nxt_open = float(df["Open"].iloc[t + 1])
        nxt_low = float(df["Low"].iloc[t + 1])
        nxt_close = float(df["Close"].iloc[t + 1])
        hi20_prev = float(df["hi20_prev"].iloc[t]) if "hi20_prev" in df.columns else np.nan

        rec = {"tk": tk, "date": s["date"], "atr": atr,
               "A": nxt_open, "D": nxt_close}
        # B: t+1 gün içi dip (limit)
        for k in DIP_KS:
            lim = nxt_open - k * atr
            rec["B%.2f" % k] = lim if nxt_low <= lim else None
        # C: sinyal günü, 20g zirvenin ilk aşıldığı saatteki fiyat (teyitsiz)
        rec["C"] = None
        if len(hsig) and not np.isnan(hi20_prev):
            cross = hsig[hsig["High"] > hi20_prev]
            if len(cross):
                rec["C"] = float(max(cross["Open"].iloc[0], hi20_prev))
        rows.append((rec, t, df))

    print("gün içi veriyle eşleşen sinyal: %d\n" % len(rows))
    if not rows:
        return

    def run(key):
        res = []
        for rec, t, df in rows:
            px = rec.get(key)
            if px is None or px <= 0:
                continue
            r = B.simulate_from_entry(df, t, EXIT_NEW, float(px)) if hasattr(
                B, "simulate_from_entry") else None
            if r is not None:
                res.append({"r": r, "date": rec["date"]})
        return res

    variants = [("A  t+1 açılış (MEVCUT)", "A"), ("D  t+1 kapanış", "D"),
                ("C  sinyal günü gün içi (TEYİTSİZ)", "C")]
    variants += [("B  t+1 dip  limit −%.2f ATR" % k, "B%.2f" % k) for k in DIP_KS]

    def stat(rs):
        if not rs:
            return (0, 0.0, 0.0, 0.0)
        a = np.array([x["r"] for x in rs])
        return (len(a), a.mean(), (a > 0).mean() * 100, a.sum())

    total = len(rows)
    print("=" * 104)
    print("  GİRİŞ ANI KARŞILAŞTIRMASI — çıkış HERKESTE AYNI, %d aday sinyal" % total)
    print("=" * 104)
    print("  %-36s%7s%9s%8s%9s%11s%11s" %
          ("kurgu", "işlem", "doluluk", "EV", "WR", "TOPLAM", "OOS EV"))
    print("  " + "-" * 100)
    base = None
    for label, key in variants:
        rs = run(key)
        n, ev, wr, tot = stat(rs)
        _, oos_ev, _, _ = stat([x for x in rs if x["date"] >= OOS_SPLIT])
        fill = 100.0 * n / total if total else 0
        mark = ""
        if base is None:
            base = (ev, tot)
        else:
            mark = "  ΔEV %+.2f  ΔTOPLAM %+.0f" % (ev - base[0], tot - base[1])
        print("  %-36s%7d%8.0f%%%+8.2f%%%8.0f%%%+10.0f%%%+10.2f%%%s"
              % (label, n, fill, ev, wr, tot, oos_ev, mark))


if __name__ == "__main__":
    main()
