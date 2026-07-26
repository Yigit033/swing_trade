# -*- coding: utf-8 -*-
"""
EXIT LAB — VCE + RVOL THRUST BİRLİKTE
======================================
Amaç (kullanıcı planı): RVOL thrust'ı sisteme eklemeden ÖNCE, hem VCE hem RVOL
thrust sinyalleri için EN İYİ exit'i (stop/hedef/tutma/trailing) ölçerek bulmak.
Çünkü keşifte RVOL thrust yüksek edge ama sabit %10 stop'ta %59 stop-out veriyordu
— sabit stop, hacim patlamasının doğal oynaklığına dar geliyor olabilir.

Yöntem (measure_signal_edge.py / exit_strategy_lab.py ile aynı standart):
  - Giriş = t+1 açılış (canlı PENDING mekaniği). Lookahead YOK.
  - Bar-bar exit simülasyonu: stop → T1 kısmi (+breakeven) → T2 cap → trailing.
  - Gerçekçi stop dolumu: gap-down'da stop yerine açılış fiyatından (kötü senaryo).
  - Metrik: EV/trade (esas), WR, ort kazanç/kayıp, medyan, MFE-yakalama.
  - Sinyal tipine göre AYRI (VCE vs RVOL) — her birine kendi exit'i.
  - OOS split (2025-06-01): kazanan exit ikinci yarıda da kazanıyor mu?

Sinyal tanımları discover_signal_families.py'den birebir import edilir (drift yok).
Cache: output/_edge_data.pkl (measure_signal_edge.py).
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np
import pandas as pd

from discover_signal_families import enrich, p_vce_baseline, p_rvol_thrust

SPLIT = pd.Timestamp("2025-06-01")


# ══════════════════════════════════════════════════════════════════════
# BAR-BAR EXIT SİMÜLASYONU
# ══════════════════════════════════════════════════════════════════════
def simulate(df, t, strat):
    """
    Giriş = t+1 açılış. strat: stop_atr, t1_pct, t1_frac, be_after_t1,
    t2_pct, trail_atr, trail_after, hold. Döner: realized return % (poz ağırlıklı).
    """
    o = df["Open"].astype(float).values
    c = df["Close"].astype(float).values
    h = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    n = len(df); e = t + 1
    if e >= n:
        return None
    entry = o[e]; atr = float(df["atr"].iloc[t])
    if entry <= 0 or atr <= 0:
        return None

    stop = entry - strat["stop_atr"] * atr
    t1 = entry * (1 + strat["t1_pct"]) if strat.get("t1_pct") else None
    t2 = entry * (1 + strat["t2_pct"]) if strat.get("t2_pct") else None
    pos = 1.0; realized = 0.0; peak = entry; t1_done = False

    last = min(e + strat["hold"], n - 1)
    for j in range(e, last + 1):
        # 1. Stop / trailing (önce — kötü senaryo). Gap-down'da açılıştan dol.
        if low[j] <= stop:
            px = min(o[j], stop) if o[j] < stop else stop
            realized += pos * (px / entry - 1); pos = 0.0; break
        # 2. T1 kısmi + breakeven
        if t1 and not t1_done and h[j] >= t1:
            realized += strat["t1_frac"] * (t1 / entry - 1)
            pos -= strat["t1_frac"]; t1_done = True
            if strat.get("be_after_t1"):
                stop = max(stop, entry)
        # 3. T2 cap
        if t2 and h[j] >= t2:
            realized += pos * (t2 / entry - 1); pos = 0.0; break
        # 4. Trailing güncelle
        if h[j] > peak:
            peak = h[j]
        if strat.get("trail_atr"):
            if (peak - entry) / atr >= strat.get("trail_after", 1.0):
                stop = max(stop, peak - strat["trail_atr"] * atr)
    if pos > 0:
        realized += pos * (c[last] / entry - 1)
    return realized * 100


# ══════════════════════════════════════════════════════════════════════
# EXIT IZGARASI — sistematik, senior kapsama
# ══════════════════════════════════════════════════════════════════════
def build_grid():
    grid = {}
    # Mevcut canlı exit (referans)
    grid["MEVCUT (stop1.5, T1%10, T2cap28, hold10)"] = dict(
        stop_atr=1.5, t1_pct=0.10, t1_frac=0.5, be_after_t1=True,
        t2_pct=0.28, trail_atr=2.5, trail_after=2.0, hold=10)
    # Stop genişliği taraması (RVOL için kritik — dar stop çok kesiyordu)
    for s in (1.5, 2.0, 2.5, 3.0):
        grid[f"Trail-runner stop{s} (T1%10, cap yok, hold20)"] = dict(
            stop_atr=s, t1_pct=0.10, t1_frac=0.33, be_after_t1=True,
            t2_pct=None, trail_atr=2.5, trail_after=1.5, hold=20)
    # Saf trailing (T1 yok — kazananı tam koştur)
    for s in (2.0, 2.5):
        grid[f"Saf trail stop{s} (T1 yok, hold20)"] = dict(
            stop_atr=s, t1_pct=None, t1_frac=0.0, be_after_t1=False,
            t2_pct=None, trail_atr=2.5, trail_after=1.0, hold=20)
    # Yüksek cap varyantları (kazananı biraz daha koştur ama cap'le kilitle)
    for cap in (0.35, 0.50):
        grid[f"T1%10 + T2cap{int(cap*100)} (stop2, hold15)"] = dict(
            stop_atr=2.0, t1_pct=0.10, t1_frac=0.5, be_after_t1=True,
            t2_pct=cap, trail_atr=2.5, trail_after=2.0, hold=15)
    # Uzun tutma (keşif: sinyaller 20-30 günde olgunlaşıyor)
    grid["Uzun tut (stop2.5, T1%12, cap yok, hold30)"] = dict(
        stop_atr=2.5, t1_pct=0.12, t1_frac=0.33, be_after_t1=True,
        t2_pct=None, trail_atr=3.0, trail_after=1.5, hold=30)
    return grid


# ══════════════════════════════════════════════════════════════════════
# ÖLÇÜM
# ══════════════════════════════════════════════════════════════════════
def collect_signals(data):
    """VCE ve RVOL thrust sinyallerini ayrı topla (RVOL'de VCE ile örtüşeni düş
    → saf marjinal RVOL katkısı)."""
    vce, rvol = [], []
    for tk, df in data.items():
        for t in range(60, len(df) - 31):
            is_v = p_vce_baseline(df, t)
            is_r = p_rvol_thrust(df, t)
            if is_v:
                vce.append((tk, df, t))
            if is_r and not is_v:  # saf RVOL (VCE zaten yakalıyorsa VCE sayılır)
                rvol.append((tk, df, t))
    return vce, rvol


def eval_grid(sigs, grid, label):
    day_of = lambda s: pd.to_datetime(s[1]["Date"].iloc[s[2]])
    print(f"\n{'='*96}")
    print(f"  {label}  (n={len(sigs)} sinyal)")
    print(f"{'='*96}")

    # MFE referansı (10 ve 20 gün potansiyel tavan)
    mfe10, mfe20 = [], []
    for tk, df, t in sigs:
        h = df["High"].astype(float).values; o = df["Open"].astype(float).values
        e = t + 1; n = len(df)
        if e >= n: continue
        mfe10.append((h[e:min(e+10, n)].max() / o[e] - 1) * 100)
        mfe20.append((h[e:min(e+20, n)].max() / o[e] - 1) * 100)
    print(f"  MFE potansiyel: 10g medyan {np.median(mfe10):+.1f}% p75 {np.percentile(mfe10,75):+.1f}% | "
          f"20g medyan {np.median(mfe20):+.1f}% p75 {np.percentile(mfe20,75):+.1f}%")

    print(f"\n  {'Exit stratejisi':<44}{'EV':>8}{'WR':>6}{'kazanç':>9}{'kayıp':>8}{'medyan':>8}{'OOS EV':>8}")
    print("  " + "-" * 90)
    results = {}
    for name, strat in grid.items():
        rets, oos = [], []
        for s in sigs:
            r = simulate(s[1], s[2], strat)
            if r is None: continue
            rets.append(r)
            if day_of(s) >= SPLIT: oos.append(r)
        if len(rets) < 8:
            continue
        a = np.array(rets); wins = a[a > 0]; losses = a[a <= 0]
        ev = a.mean()
        oos_ev = np.mean(oos) if len(oos) >= 8 else None
        results[name] = {"ev": ev, "oos": oos_ev, "wr": (a > 0).mean() * 100}
        oos_s = f"{oos_ev:+.2f}" if oos_ev is not None else "  -"
        print(f"  {name:<44}{ev:>+7.2f}%{(a>0).mean()*100:>5.0f}%"
              f"{(wins.mean() if len(wins) else 0):>+8.1f}%{(losses.mean() if len(losses) else 0):>+7.1f}%"
              f"{np.median(a):>+7.1f}%{oos_s:>8}")

    best = max(results, key=lambda k: results[k]["ev"])
    ref_key = next((k for k in results if k.startswith("MEVCUT")), None)
    print(f"\n  → EN İYİ EV: '{best}' ({results[best]['ev']:+.2f}%, OOS {results[best]['oos']:+.2f}%)")
    if ref_key:
        print(f"    (mevcut exit: {results[ref_key]['ev']:+.2f}%, fark {results[best]['ev']-results[ref_key]['ev']:+.2f}%)")
    return best, results


def main():
    with open("output/_edge_data.pkl", "rb") as f:
        data = pickle.load(f)
    data = {t: enrich(df) for t, df in data.items()}
    vce, rvol = collect_signals(data)
    grid = build_grid()

    print("╔" + "═" * 94 + "╗")
    print("║  EXIT LAB — VCE + RVOL THRUST | 57 ticker, 2024-06→2026-05 | giriş t+1 açılış" + " " * 17 + "║")
    print("╚" + "═" * 94 + "╝")

    v_best, v_res = eval_grid(vce, grid, "VCE (mevcut sinyal)")
    r_best, r_res = eval_grid(rvol, grid, "RVOL THRUST (saf marjinal — VCE'nin görmediği)")

    # Birleşik: her sinyale KENDİ en iyi exit'i uygulansaydı toplam EV?
    print(f"\n{'='*96}")
    print("  SENTEZ")
    print(f"{'='*96}")
    print(f"  VCE en iyi exit    : {v_best}")
    print(f"                       EV {v_res[v_best]['ev']:+.2f}% (OOS {v_res[v_best]['oos']:+.2f}%, WR {v_res[v_best]['wr']:.0f}%)")
    print(f"  RVOL en iyi exit   : {r_best}")
    print(f"                       EV {r_res[r_best]['ev']:+.2f}% (OOS {r_res[r_best]['oos']:+.2f}%, WR {r_res[r_best]['wr']:.0f}%)")
    same = v_best == r_best
    print(f"\n  İkisi için exit AYNI mı? {'EVET — tek exit yeterli' if same else 'HAYIR — sinyal-tipine göre ayrı exit değerli'}")


if __name__ == "__main__":
    main()
