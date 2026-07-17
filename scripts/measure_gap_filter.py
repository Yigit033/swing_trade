# -*- coding: utf-8 -*-
"""
GAP FİLTRESİ EDGE ÖLÇÜMÜ
========================
Canlı tracker, pending confirm'de gap-up > +5% / gap-down < -5% olan girişleri
REDDEDIYOR (paper_trading/tracker.py: MAX_GAP_UP_PCT / MAX_GAP_DOWN_PCT).
Ama VCE edge'i (Variant B) bu filtre OLMADAN ölçüldü — yani canlı sistemin
kapıda çevirdiği sinyallerin edge'i bilinmiyor.

Bu script aynı örneklem (output/_edge_data.pkl, 2024-06→2026-05) üzerinde:
  1. Variant B sinyallerinde gap = (Open[t+1]/Close[t] - 1) dağılımını çıkarır
  2. Canlı kuralın TUTTUĞU ve REDDETTİĞİ grupların R5/R10 edge'ini ayrı ölçer
  3. Eşik duyarlılığı: farklı gap limitlerinde kalan n + R10 edge + t
  4. ±5 kuralı için train/test split (2025-06-01)

Karar kuralı: reddedilen grubun edge'i ~0 veya negatifse filtre masum/faydalı;
belirgin pozitifse filtre kazananları kesiyor → gevşet.

Cache gereksinimi: measure_signal_edge.py çalışmış olmalı (validate_volsqueeze gibi).
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np
import pandas as pd

# Kural tanımlarını doğrulama harness'ından AYNEN al (drift olmasın)
from validate_volsqueeze import add_ind, v_trend, fr, welch

# Canlı limitleri tracker'dan AYNEN al (drift olmasın)
from swing_trader.paper_trading.tracker import MAX_GAP_UP_PCT, MAX_GAP_DOWN_PCT

HZ = [5, 10]
SPLIT_DATE = pd.Timestamp('2025-06-01')


def edge(rows, bench, N):
    rv = np.array([r[f'R{N}'] for r in rows if r.get(f'R{N}') is not None])
    bv = np.array([b[f'R{N}'] for b in bench if b.get(f'R{N}') is not None])
    if len(rv) < 5:
        return None
    return {
        'n': len(rv),
        'mean': round(float(rv.mean()), 2),
        'edge': round(float(rv.mean() - bv.mean()), 2),
        't': welch(rv, bv),
        'wr': round(float((rv > 0).mean() * 100), 0),
    }


def fmt(label, e):
    if e is None:
        return f"    {label:<34} n yetersiz"
    return (f"    {label:<34} n={e['n']:<4} ort {e['mean']:+6.2f}%  "
            f"edge {e['edge']:+6.2f}%  t={e['t']}  WR {e['wr']:.0f}%")


def main():
    with open('output/_edge_data.pkl', 'rb') as f:
        data = pickle.load(f)
    data = {t: add_ind(df) for t, df in data.items()}

    sigs, bench = [], []
    for tk, df in data.items():
        o = df['Open'].astype(float).values
        c = df['Close'].astype(float).values
        n = len(df)
        for t in range(60, n - 11):
            f_ = fr(df, t + 1)  # giriş: t+1 open (ölçüm konvansiyonu)
            if not f_ or f_.get('R5') is None:
                continue
            bench.append(f_)
            if v_trend(df, t):  # Variant B: squeeze+brk+green+MA50 (canlı gate)
                if o[t + 1] <= 0 or c[t] <= 0:
                    continue
                gap = (o[t + 1] / c[t] - 1) * 100
                sigs.append({**f_, 'gap': gap, 'date': pd.to_datetime(df['Date'].iloc[t]),
                             'ticker': tk})

    print("=" * 92)
    print(f" GAP FİLTRESİ ÖLÇÜMÜ — Variant B, n={len(sigs)}, canlı kural: "
          f"-{MAX_GAP_DOWN_PCT}% / +{MAX_GAP_UP_PCT}%")
    print("=" * 92)

    gaps = np.array([s['gap'] for s in sigs])
    print(f"\n  [0] GAP DAĞILIMI: ort {gaps.mean():+.2f}%  medyan {np.median(gaps):+.2f}%  "
          f"p10 {np.percentile(gaps,10):+.2f}%  p90 {np.percentile(gaps,90):+.2f}%  "
          f"min {gaps.min():+.2f}%  max {gaps.max():+.2f}%")

    kept = [s for s in sigs if -MAX_GAP_DOWN_PCT <= s['gap'] <= MAX_GAP_UP_PCT]
    rej_up = [s for s in sigs if s['gap'] > MAX_GAP_UP_PCT]
    rej_dn = [s for s in sigs if s['gap'] < -MAX_GAP_DOWN_PCT]

    print(f"\n  [1] CANLI KURAL (±{MAX_GAP_UP_PCT:.0f}%) GRUPLARI")
    for N in HZ:
        print(f"  — R{N} —")
        print(fmt("TÜMÜ (ölçülen baz)", edge(sigs, bench, N)))
        print(fmt(f"TUTULAN (canlının trade ettiği)", edge(kept, bench, N)))
        print(fmt(f"RED gap-up > +{MAX_GAP_UP_PCT:.0f}%", edge(rej_up, bench, N)))
        print(fmt(f"RED gap-down < -{MAX_GAP_DOWN_PCT:.0f}%", edge(rej_dn, bench, N)))

    print(f"\n  [2] EŞİK DUYARLILIĞI (R10 edge; up-limit × down-limit)")
    print(f"    {'up\\dn':>8} | " + " | ".join(f"{d:>16}" for d in ['-3%', '-5%', '-7%', 'yok']))
    for up, up_v in [('+3%', 3), ('+5%', 5), ('+7%', 7), ('+10%', 10), ('yok', 1e9)]:
        cells = []
        for dn, dn_v in [('-3%', 3), ('-5%', 5), ('-7%', 7), ('yok', 1e9)]:
            grp = [s for s in sigs if -dn_v <= s['gap'] <= up_v]
            e = edge(grp, bench, 10)
            cells.append(f"n={e['n']:<4} {e['edge']:+5.2f} t={e['t']}" if e else "  —")
        print(f"    {up:>8} | " + " | ".join(f"{c:>16}" for c in cells))

    print(f"\n  [3] TRAIN/TEST — canlı ±5 kuralıyla tutulan set")
    tr = [s for s in kept if s['date'] < SPLIT_DATE]
    te = [s for s in kept if s['date'] >= SPLIT_DATE]
    for lbl, rows in [('TRAIN', tr), ('TEST ', te)]:
        print(fmt(lbl, edge(rows, bench, 10)))

    print("\n" + "=" * 92)


if __name__ == '__main__':
    main()
