# -*- coding: utf-8 -*-
"""
RSI KAPISI ÖLÇÜMÜ — "RSI>70 reddi bize para kaybettiriyor mu?"
================================================================================
KULLANICI HİPOTEZİ (mantıklı ve test edilmeli): "Bir hisse 20 günün zirvesini
kırıyorsa ve RVOL>2 ise, RSI zaten %70'in üstündedir veya çıkmak üzeredir.
Güçlü momentum hisseleri RSI aşırı-alım bölgesindeyken en sert yükselişlerini
yapar. RSI filtresi kaldırılmalı veya %75-80'e çekilmeli."

MEVCUT DURUM (engine.py:633):
    if rsi > max_entry_rsi and swing_type != 'S' and not _is_vce:  → REDDET
Yani VCE sinyalleri kapıdan MUAF (2026-06 ölçümü: RSI 80+ VCE sinyalleri
edge +4.53%, n=49 — sıkışma kırılımında yüksek RSI güç demek). Kapı yalnız
VCE-DIŞI girişlere (RVOL thrust vb.) uygulanıyor ve canlıda ateş ediyor
(2026-08-03 taraması: rsi_gate=3 reddetme).

SORU: RVOL thrust için de kapı kaldırılmalı mı? Yoksa orada gerçekten koruyor mu?

YÖNTEM: kapıyı devre dışı bırak (max_entry_rsi=100), gerçek motoru koştur,
her sinyalin RSI'sını + pathway'ini + gerçek exit getirisini kaydet. Sonra
pathway × RSI kovası kırılımı + OOS.

KARAR KURALI (önceden yazıldı): kapı ancak
  (a) RSI>70 grubunun EV'si RSI<=70 grubundan anlamlı DÜŞÜKSE, VE
  (b) bu TRAIN ve OOS'ta AYNI yönde ise
korunur. Aksi halde kapı para kaybettiriyor demektir ve gevşetilir/kaldırılır.
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
CACHE = "output/rsi_gate.json"


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


def collect():
    print("Veri yukleniyor...", flush=True)
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    shares = json.load(open("output/_shares_broad.json"))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)
    data = {t: enrich(df) for t, df in raw.items()}

    engine = SmallCapEngine()
    print(f"  Canli RSI kapisi: max_entry_rsi = {engine.settings.max_entry_rsi}")
    # Kapıyı devre dışı bırak — reddedilenleri de görmek için
    engine.settings.max_entry_rsi = 100
    print("  Olcum icin kapi ACILDI (max_entry_rsi=100)", flush=True)

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
            recs.append({
                "tk": tk, "date": str(day.date()), "r": float(r),
                "q": float(sig.get("quality_score", 0) or 0),
                "rsi": float(sig.get("rsi", 0) or 0),
                "pw": sig.get("trigger_pathway", "?"),
                "type": sig.get("swing_type", "?"),
                "reg": reg,
            })
        if (ti + 1) % 200 == 0:
            print(f"  ...{ti+1}/{len(tickers)} ({len(recs)} sinyal)", flush=True)

    json.dump(recs, open(CACHE, "w"), default=str)
    return recs


def main():
    recs = json.load(open(CACHE)) if os.path.exists(CACHE) else collect()
    W = 92
    print("\n" + "=" * W)
    print(f"  RSI KAPISI — {len(recs)} sinyal (kapi ACIK toplandi) | canli kapi: RSI>70 reddet")
    print("=" * W)

    rsis = np.array([r["rsi"] for r in recs if r["rsi"] > 0])
    if len(rsis):
        print(f"  RSI dagilimi: min={rsis.min():.0f} medyan={np.median(rsis):.0f} "
              f"max={rsis.max():.0f} | RSI>70 olan: {(rsis>70).sum()} ({(rsis>70).mean()*100:.0f}%)")

    # ── Kullanıcı hipotezinin 1. yarısı: kırılımda RSI gerçekten >70 mi? ──
    print("\n" + "=" * W)
    print("  HIPOTEZ TESTI 1: 'kirilim + RVOL>2 ise RSI zaten >70 olur' dogru mu?")
    print("=" * W)
    for pw in ("vce_breakout", "rvol_thrust"):
        g = [r["rsi"] for r in recs if r["pw"] == pw and r["rsi"] > 0]
        if g:
            a = np.array(g)
            print(f"  {pw:<16} n={len(a):<5} RSI medyan={np.median(a):.0f}  "
                  f">70 olan: {(a>70).sum()} ({(a>70).mean()*100:.0f}%)")

    # ── Pathway × RSI kovası ─────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  HIPOTEZ TESTI 2: yuksek RSI para kaybettiriyor mu? (pathway x RSI)")
    print("=" * W)
    BUCKETS = [(0, 50), (50, 60), (60, 70), (70, 75), (75, 80), (80, 101)]
    for pw in ("vce_breakout", "rvol_thrust"):
        sub = [r for r in recs if r["pw"] == pw]
        if not sub:
            continue
        print(f"\n  ── {pw} ({len(sub)} sinyal) ──")
        for lo, hi in BUCKETS:
            sel = [r for r in sub if lo <= r["rsi"] < hi]
            gate = " <- KAPI BUNLARI REDDEDIYOR" if lo >= 70 and pw != "vce_breakout" else ""
            if sel:
                print(f"    RSI {lo}-{hi-1:<4} {fmt(stats(sel))}{gate}")

    # ── Karar: kapı korunmalı mı? ────────────────────────────────────────
    print("\n" + "=" * W)
    print("  KARAR — kapinin ETKILEDIGI grup: VCE-DISI sinyaller (VCE muaf)")
    print("=" * W)
    nonvce = [r for r in recs if r["pw"] != "vce_breakout" and r["type"] != "S"]
    lo70 = [r for r in nonvce if r["rsi"] <= 70]
    hi70 = [r for r in nonvce if r["rsi"] > 70]
    print(f"  RSI <=70 (kapiyi gecenler) : {fmt(stats(lo70))}")
    print(f"  RSI  >70 (REDDEDILENLER)   : {fmt(stats(hi70))}")

    def split(rows):
        return ([r for r in rows if r["date"] < OOS_SPLIT],
                [r for r in rows if r["date"] >= OOS_SPLIT])

    print(f"\n  OOS kontrolu (kesim {OOS_SPLIT}):")
    for lbl, rows in (("RSI <=70", lo70), ("RSI  >70", hi70)):
        tr, te = split(rows)
        print(f"    {lbl:<10} TRAIN {fmt(stats(tr))}   OOS {fmt(stats(te))}")

    s_lo, s_hi = stats(lo70), stats(hi70)
    tr_hi, te_hi = [stats(x) for x in split(hi70)]
    tr_lo, te_lo = [stats(x) for x in split(lo70)]
    print("\n  " + "-" * (W - 4))
    if s_hi["n"] < 5:
        print(f"  SONUC: reddedilen grup cok kucuk (n={s_hi['n']}) — karar icin yetersiz veri.")
    else:
        worse_all = s_hi["ev"] < s_lo["ev"]
        worse_tr = tr_hi["ev"] < tr_lo["ev"] if tr_hi["n"] >= 3 and tr_lo["n"] >= 3 else None
        worse_te = te_hi["ev"] < te_lo["ev"] if te_hi["n"] >= 3 and te_lo["n"] >= 3 else None
        consistent = worse_all and worse_tr is not False and worse_te is not False
        if consistent:
            print("  SONUC: RSI>70 grubu TUTARLI sekilde daha kotu → KAPI KORUNUR.")
        elif not worse_all:
            print(f"  SONUC: RSI>70 grubu DAHA IYI (EV {s_hi['ev']:+.2f}% vs {s_lo['ev']:+.2f}%)")
            print("         → kapi para kaybettiriyor, GEVSETILMELI/KALDIRILMALI.")
        else:
            print("  SONUC: yon TRAIN/OOS arasinda TUTARSIZ → gurultu, mevcut kapi korunur.")

    # ── Ek: 75/80 esigi ne yapardi? ──────────────────────────────────────
    print("\n" + "=" * W)
    print("  ALTERNATIF ESIKLER (VCE-disi sinyaller, kullanicinin onerisi 75-80)")
    print("=" * W)
    for th in (70, 75, 80, 100):
        passed = [r for r in nonvce if r["rsi"] <= th]
        added = [r for r in nonvce if 70 < r["rsi"] <= th]
        ex = f"   eklenen: +{len(added)} EV {stats(added)['ev']:+.2f}%" if added else ""
        print(f"  max_entry_rsi={th:<5} {fmt(stats(passed))}{ex}")


if __name__ == "__main__":
    main()
