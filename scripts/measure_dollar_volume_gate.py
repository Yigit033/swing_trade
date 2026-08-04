# -*- coding: utf-8 -*-
"""
DOLAR-HACİM KAPISI ÖLÇÜMÜ — "$5M/gün kapısı bize para kaybettiriyor mu?"
================================================================================
GEÇMİŞ HATA (dürüstlük kaydı): 2026-08-04'te bu kapıyı gevşetme testi (5M→3M→
2M→1M) "hiç ek sinyal yok" dedi ve ben "etkisiz" diye yorumladım. Test
GEÇERSİZDİ: kullanılan önbellek (995 ticker, S&P 400+600) neredeyse tamamen
likit isimlerden oluşuyordu — sadece 8 ticker (%1) $5M altında, sinyallerin %0'ı.
Kapının eleyeceği popülasyon veri setinde HİÇ YOKTU. Doğru ifade: "ölçülemedi".

BU ÖLÇÜM o eksiği kapatır. Evren: fetch_low_liquidity_universe.py ile çekilen
310 ticker — %64'ü $5M/gün ALTINDA, yani tam kapının kestiği bölge.

YÖNTEM: gerçek motor (scan_stock), gerçek exit (EXIT_NEW), gerçek slippage.
Kapı $0.5M'e indirilir, tüm sinyaller toplanır, sonra dolar-hacim kovalarına
göre getiri karşılaştırılır. Böylece "kapıyı $3M'e indirsem ne kazanır/kaybederim"
sorusu doğrudan cevaplanır.

SLIPPAGE KRİTİK: illikit hissede alım-satım pahalıdır. Harness'ın _slippage_bps
fonksiyonu bunu modelliyor (dvol<$3M → +25bps tek yön, yani gidiş-dönüş +50bps
ek maliyet). Kapıyı gevşetmenin bedeli buradan gelir — ölçüme dahil.

⚠️ SURVIVORSHIP (sonucu yorumlarken ZORUNLU): evren BUGÜNÜN Finviz listesi.
Düşük likiditeli small-cap'lerde delist/iflas oranı yüksektir; bugün hayatta
olanlar o dönemin popülasyonunun İYİMSER alt kümesidir. Bu önyargı GEVŞETME
LEHİNE çalışır. Dolayısıyla:
    ölçüm "gevşetme" derse  → sonuç KESİN (gerçek daha da kötü)
    ölçüm "gevşet" derse    → temkinli ol, gerçek kazanç ölçülenden AZ olur
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
COLLECT_DVOL_FLOOR = 0.5e6      # toplama sırasında kapı buraya indirilir


def stats(rows):
    if not rows:
        return dict(n=0, ev=0.0, wr=0.0, pf=0.0)
    a = np.array([r["r"] for r in rows])
    w, l = a[a > 0], a[a <= 0]
    pf = (w.sum() / abs(l.sum())) if l.size and l.sum() != 0 else float("inf")
    return dict(n=len(a), ev=float(a.mean()), wr=float((a > 0).mean() * 100), pf=float(pf))


def fmt(s):
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return f"n={s['n']:<4} EV {s['ev']:+6.2f}%  WR {s['wr']:3.0f}%  PF {pf:>5}"


def main():
    print("Veri yukleniyor...", flush=True)
    raw = pickle.load(open("output/_lowliq_data.pkl", "rb"))
    shares = json.load(open("output/_shares_lowliq.json"))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)
    data = {t: enrich(df) for t, df in raw.items()}

    engine = SmallCapEngine()
    f = engine.filters
    print(f"  Canli dolar-hacim kapisi: ${f.MIN_DOLLAR_VOLUME/1e6:.1f}M/gun")
    f.MIN_DOLLAR_VOLUME = COLLECT_DVOL_FLOOR
    f.MIN_PRICE = 5.0     # evren $5-20; fiyat kapisi ayri konu, burada engellemesin
    print(f"  Toplama kapisi          : ${COLLECT_DVOL_FLOOR/1e6:.1f}M/gun  ({len(data)} ticker)")

    recs = []
    tickers = list(data.keys())
    for ti, tk in enumerate(tickers):
        df = data[tk]
        sh = shares.get(tk, {})
        sh_out, flt = sh.get("shares"), sh.get("float")
        n = len(df)
        cl = df["Close"].astype(float)
        vol = df["Volume"].astype(float)
        dvol20 = (cl * vol).rolling(20).mean()

        for t in range(60, n - 21):
            row = df.iloc[t]
            close = float(row["Close"])
            mcap = close * sh_out if sh_out else None
            if not finviz_hit(row, mcap, min_price=5.0):
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
            r = simulate(df, t, EXIT_NEW)          # slippage DAHİL
            r_noslip = simulate(df, t, EXIT_NEW, apply_slippage=False)
            if r is None:
                continue
            dv = float(dvol20.iloc[t]) / 1e6 if not pd.isna(dvol20.iloc[t]) else 0.0
            recs.append({
                "tk": tk, "date": str(day.date()), "r": float(r),
                "r_noslip": float(r_noslip) if r_noslip is not None else float(r),
                "q": float(sig.get("quality_score", 0) or 0),
                "pw": sig.get("trigger_pathway", "?"), "reg": reg,
                "dvol_m": dv, "price": close,
            })
        if (ti + 1) % 50 == 0:
            print(f"  ...{ti+1}/{len(tickers)} ({len(recs)} sinyal)", flush=True)

    json.dump(recs, open("output/dollar_volume_gate.json", "w"), default=str)
    if not recs:
        print("\nHIC SINYAL YOK — bu evrende tetik hic ates etmemis.")
        return

    W = 96
    print("\n" + "=" * W)
    print(f"  DOLAR-HACIM KAPISI — {len(recs)} sinyal | dusuk-likidite evreni ({len(data)} ticker)")
    print("=" * W)
    print(f"  TUM sinyaller: {fmt(stats(recs))}")

    # ── Kovalar ──────────────────────────────────────────────────────────
    print(f"\n  {'dolar-hacim kovasi':<22}{'TUM kalite':<42}{'Q80+ (canli esik)'}")
    print("  " + "-" * (W - 4))
    BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 1e6)]
    for lo, hi in BUCKETS:
        sel = [r for r in recs if lo <= r["dvol_m"] < hi]
        q80 = [r for r in sel if r["q"] >= 80]
        lbl = f"${lo:.0f}M-${hi:.0f}M" if hi < 1e6 else f">${lo:.0f}M"
        mark = "  <- KAPI ALTI" if hi <= 5 else ""
        print(f"  {lbl:<22}{fmt(stats(sel)):<42}{fmt(stats(q80))}{mark}")

    # ── Kapi seviyeleri: gevsetsem ne olur? ──────────────────────────────
    print("\n" + "=" * W)
    print("  KARAR TABLOSU — kapiyi X'e indirsem (Q80+ sinyaller, slippage dahil)")
    print("=" * W)
    print(f"  {'kapi':<10}{'sinyal':<40}{'EK sinyaller ($5M altindan gelenler)'}")
    print("  " + "-" * (W - 4))
    base = [r for r in recs if r["dvol_m"] >= 5.0 and r["q"] >= 80]
    print(f"  {'$5M (su an)':<10}{fmt(stats(base)):<40}—")
    for gate in (3.0, 2.0, 1.0, 0.5):
        sel = [r for r in recs if r["dvol_m"] >= gate and r["q"] >= 80]
        extra = [r for r in sel if r["dvol_m"] < 5.0]
        se = stats(extra)
        ex = f"+{se['n']:<4} EV {se['ev']:+6.2f}% WR {se['wr']:3.0f}%" if se["n"] else "—"
        print(f"  {'$' + f'{gate:g}M':<10}{fmt(stats(sel)):<40}{ex}")

    # ── Slippage'in bedeli ───────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  SLIPPAGE'IN BEDELI — illikit hissede alim-satim maliyeti")
    print("=" * W)
    print(f"  {'kova':<14}{'slippage YOK':<22}{'slippage DAHIL':<22}{'bedel (puan)'}")
    print("  " + "-" * (W - 4))
    for lo, hi in [(0, 3), (3, 5), (5, 10), (10, 1e6)]:
        sel = [r for r in recs if lo <= r["dvol_m"] < hi]
        if not sel:
            continue
        a_no = np.array([r["r_noslip"] for r in sel]).mean()
        a_yes = np.array([r["r"] for r in sel]).mean()
        lbl = f"${lo:.0f}M-${hi:.0f}M" if hi < 1e6 else f">${lo:.0f}M"
        print(f"  {lbl:<14}{f'EV {a_no:+.2f}%':<22}{f'EV {a_yes:+.2f}%':<22}{a_yes - a_no:+.2f}")

    # ── OOS ──────────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print(f"  OOS DOGRULAMA (kesim {OOS_SPLIT}) — kapi alti sinyaller gelecekte de kazaniyor mu?")
    print("=" * W)
    print(f"  {'grup':<24}{'TRAIN':<32}{'OOS (test)'}")
    print("  " + "-" * (W - 4))
    for lbl, sel in [
        ("$5M USTU, Q80+", [r for r in recs if r["dvol_m"] >= 5.0 and r["q"] >= 80]),
        ("$3-5M, Q80+", [r for r in recs if 3.0 <= r["dvol_m"] < 5.0 and r["q"] >= 80]),
        ("$5M ALTI, Q80+", [r for r in recs if r["dvol_m"] < 5.0 and r["q"] >= 80]),
    ]:
        tr = [r for r in sel if r["date"] < OOS_SPLIT]
        te = [r for r in sel if r["date"] >= OOS_SPLIT]
        print(f"  {lbl:<24}{fmt(stats(tr)):<32}{fmt(stats(te))}")

    print("\n  -> output/dollar_volume_gate.json yazildi")


if __name__ == "__main__":
    main()
