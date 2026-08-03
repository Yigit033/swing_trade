# -*- coding: utf-8 -*-
"""
EVREN GENİŞLETME ÖLÇÜMÜ — "Hangi bant bize sinyal kaybettiriyor, ve gevşetince
gelen ek sinyaller KÂRLI mı?"
================================================================================
SORUN (2026-08-03, canlı veri): forward tracker'da 7 haftada 21 ham sinyal, bunun
yalnız 1'i Q80'i geçti (~0.6 Q80 sinyal/ay). Profesyonel swing trade pratiği
4-12 işlem/ay. Yani eşik felsefesi doğru ama HUNİ GİRDİSİ çok dar.

YAPISAL TESPİT: VCE tetiği "20g yeni zirve + MA50 üstü + yeşil + RVOL>=1.5" ister.
Finviz Q6/Q6b tam olarak bunu tarıyor → Finviz KATMANI VCE için darboğaz DEĞİL.
Darboğaz BANTLAR: motor filtresi (filters.py) mcap/hacim/float/fiyat eşikleri —
canlı red sayaçlarında filter_failed %29.7 (Finviz gönderiyor, motor eliyor).

Ayrıca bir UYUMSUZLUK var: Finviz Q6 (small) avgvol>500K istiyor ama motor
min_avg_volume=750K. Yani Finviz'in gönderdiği 500-750K bandı motorda ölüyor.

TEST: her varyantta TEK bandı gevşet, GERÇEK motoru (scan_stock) koştur, gelen
EK sinyalleri say ve GETİRİLERİNİ ölç (gerçek exit simülasyonu ile). Karar kuralı:
ek sinyaller mevcut tabanın EV'sini seyreltiyorsa gevşetme REDDEDİLİR.

ÖNEMLİ: eşiğe (quality) dokunulmuyor. Ölçülen şey yalnız KEŞİF genişliği.

Cache: output/_broad_data.pkl (995 ticker, 2024-06→2026-05) + _shares_broad.json
       + _edge_spy.pkl
Çıktı: output/universe_expansion.json + konsol tablosu
"""
import sys, os, json, pickle, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.ERROR)
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd

from swing_trader.small_cap.engine import SmallCapEngine
from backtest_live_replica import enrich, simulate, EXIT_NEW, build_regime_map

SMALL = (300e6, 2e9)
MID = (2e9, 10e9)


# ── Finviz katmanı (universe.py Q6/Q6b/Q7/Q7b birebir) ────────────────────
def finviz_hit(row, mcap, min_av_small=500e3, min_av_mid=1e6, min_price=7.0):
    price = row["Close"]
    if price <= min_price:
        return False
    small = mcap is None or (SMALL[0] <= mcap < SMALL[1])
    mid = mcap is None or (MID[0] <= mcap <= MID[1])
    av = row["avgvol_liq"]
    if pd.isna(av):
        return False
    ma50, ma20, hi20p = row["ma50"], row["ma20"], row["hi20_prev"]
    new20 = (not pd.isna(hi20p)) and row["High"] > hi20p
    above50 = (not pd.isna(ma50)) and price > ma50
    above20 = (not pd.isna(ma20)) and price > ma20
    rvol = row["Volume"] / row["avgvol50"] if row["avgvol50"] > 0 else 0
    green = row["chg"] > 0

    if small and av > min_av_small and above50 and new20:
        return True
    if mid and av > min_av_mid and above50 and new20:
        return True
    if small and av > min_av_small and rvol > 2 and green and above20:
        return True
    if mid and av > min_av_mid and rvol > 2 and green and above20:
        return True
    return False


# ── Varyantlar: TEK bandı gevşet ──────────────────────────────────────────
# None = dokunma. Motor filtresi öznitelikleri filters.py'de örnek üzerinde
# taşındığı için varyant başına doğrudan yamalanabiliyor.
VARIANTS = [
    ("0) MEVCUT (taban)",           {}),
    ("1) float 80M -> 150M",        {"MAX_FLOAT": 150e6}),
    ("2) float 80M -> 300M",        {"MAX_FLOAT": 300e6}),
    ("3) float SINIRSIZ",           {"MAX_FLOAT": 1e12}),
    ("4) hacim 750K -> 500K",       {"MIN_AVG_VOLUME": 500e3}),
    ("5) hacim 750K -> 300K",       {"MIN_AVG_VOLUME": 300e3}),
    ("6) float 150M + hacim 500K",  {"MAX_FLOAT": 150e6, "MIN_AVG_VOLUME": 500e3}),
    ("7) HEPSI GEVSEK",             {"MAX_FLOAT": 1e12, "MIN_AVG_VOLUME": 300e3}),
]


def collect(engine, data, shares, rmap, spy, patch, tag):
    """Verilen filtre yamasıyla GERÇEK motoru koştur, sinyalleri topla."""
    saved = {}
    for k, v in patch.items():
        saved[k] = getattr(engine.filters, k)
        setattr(engine.filters, k, v)
    try:
        out = []
        for tk, df in data.items():
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
                r = simulate(df, t, EXIT_NEW)
                if r is None:
                    continue
                out.append({
                    "key": f"{tk}|{day.date()}",
                    "tk": tk, "date": str(day.date()), "r": r,
                    "q": float(sig.get("quality_score", 0) or 0),
                    "pw": sig.get("trigger_pathway", "?"),
                    "reg": reg,
                    "float_m": (flt or 0) / 1e6,
                })
        return out
    finally:
        for k, v in saved.items():
            setattr(engine.filters, k, v)


def stats(rows, q_min=None):
    g = [x for x in rows if q_min is None or x["q"] >= q_min]
    if not g:
        return dict(n=0, ev=0.0, wr=0.0, pf=0.0)
    a = np.array([x["r"] for x in g])
    wins, losses = a[a > 0], a[a <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else float("inf")
    return dict(n=len(g), ev=float(a.mean()), wr=float((a > 0).mean() * 100), pf=float(pf))


def main():
    print("Veri yukleniyor...", flush=True)
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    shares = json.load(open("output/_shares_broad.json"))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)
    data = {t: enrich(df) for t, df in raw.items()}
    engine = SmallCapEngine()
    print(f"  {len(data)} ticker | motor bantlari: float<={engine.filters.MAX_FLOAT/1e6:.0f}M "
          f"avgvol>={engine.filters.MIN_AVG_VOLUME/1e3:.0f}K", flush=True)

    results = {}
    base_keys = None
    for tag, patch in VARIANTS:
        print(f"\n>>> {tag} ...", flush=True)
        rows = collect(engine, data, shares, rmap, spy, patch, tag)
        results[tag] = rows
        if base_keys is None:
            base_keys = {x["key"] for x in rows}
        print(f"    {len(rows)} sinyal", flush=True)

    # ── Rapor ─────────────────────────────────────────────────────────────
    W = 106
    print("\n" + "=" * W)
    print("  EVREN GENISLETME — tum sinyaller (esige dokunulmadi)")
    print("=" * W)
    print(f"  {'varyant':<28}{'n':>6}{'EV':>9}{'WR':>7}{'PF':>7}   |{'EK sinyal':>11}{'EK EV':>9}{'EK WR':>8}")
    print("  " + "-" * (W - 4))
    for tag, rows in results.items():
        s = stats(rows)
        extra = [x for x in rows if x["key"] not in base_keys]
        se = stats(extra)
        ex = f"{se['n']:>11}{se['ev']:>+9.2f}{se['wr']:>7.0f}%" if se["n"] else f"{'—':>11}{'—':>9}{'—':>8}"
        print(f"  {tag:<28}{s['n']:>6}{s['ev']:>+9.2f}{s['wr']:>6.0f}%{s['pf']:>7.2f}   |{ex}")

    print("\n" + "=" * W)
    print("  AYNI TABLO — SADECE Q80+ (canli esik: BULL 78 / CAUTION-BEAR 80)")
    print("=" * W)
    print(f"  {'varyant':<28}{'n':>6}{'EV':>9}{'WR':>7}{'PF':>7}   |{'EK Q80':>11}{'EK EV':>9}{'EK WR':>8}")
    print("  " + "-" * (W - 4))
    base80 = {x["key"] for x in results["0) MEVCUT (taban)"] if x["q"] >= 80}
    for tag, rows in results.items():
        s = stats(rows, q_min=80)
        extra = [x for x in rows if x["q"] >= 80 and x["key"] not in base80]
        se = stats(extra)
        ex = f"{se['n']:>11}{se['ev']:>+9.2f}{se['wr']:>7.0f}%" if se["n"] else f"{'—':>11}{'—':>9}{'—':>8}"
        print(f"  {tag:<28}{s['n']:>6}{s['ev']:>+9.2f}{s['wr']:>6.0f}%{s['pf']:>7.2f}   |{ex}")

    # ── Aylik hiz (24 ay veri) ────────────────────────────────────────────
    months = 24.0
    print("\n" + "=" * W)
    print(f"  AYLIK SINYAL HIZI (~{months:.0f} ay veri) — hedef: profesyonel pratik 4-12 islem/ay")
    print("=" * W)
    for tag, rows in results.items():
        n80 = len([x for x in rows if x["q"] >= 80])
        n78 = len([x for x in rows if x["q"] >= 78])
        print(f"  {tag:<28} ham {len(rows)/months:>5.1f}/ay   Q78+ {n78/months:>5.1f}/ay   Q80+ {n80/months:>5.1f}/ay")

    # ── Float kovasi: gevsetme hangi float bandini getiriyor? ─────────────
    loose = results["3) float SINIRSIZ"]
    print("\n" + "=" * W)
    print("  FLOAT KOVASI (float SINIRSIZ varyantinda, tum sinyaller)")
    print("=" * W)
    for lo, hi, lbl in [(0, 15, "<=15M (atomik)"), (15, 30, "15-30M"), (30, 80, "30-80M (mevcut sinir)"),
                        (80, 150, "80-150M (YENI)"), (150, 1e9, ">150M (YENI)")]:
        g = [x for x in loose if lo <= x["float_m"] < hi]
        if g:
            a = np.array([x["r"] for x in g])
            n80 = len([x for x in g if x["q"] >= 80])
            print(f"    {lbl:<24} n={len(g):<5} EV {a.mean():+6.2f}%  WR {(a>0).mean()*100:3.0f}%   Q80+: {n80}")
        else:
            print(f"    {lbl:<24} n=0")

    json.dump({k: v for k, v in results.items()},
              open("output/universe_expansion.json", "w"), default=str)
    print("\n  -> output/universe_expansion.json yazildi")


if __name__ == "__main__":
    main()
