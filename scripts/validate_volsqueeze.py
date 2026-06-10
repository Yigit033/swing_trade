"""
VOLSQUEEZE EDGE — DOĞRULAMA & RAFİNE
====================================
Hipotez laboratuvarında tek istatistiksel anlamlı edge çıkan 'volsqueeze_breakout'u
senior-quant titizliğiyle stres testine sokar:

  1. TRAIN/TEST split — edge ikinci yarıda da duruyor mu? (overfit/fluke testi)
  2. TICKER YOĞUNLAŞMASI — edge 1-2 hisseden mi geliyor? (en çok katkı verenleri çıkar)
  3. REGIME dağılımı — hangi rejimde çalışıyor?
  4. VARYANTLAR — trend/hacim/güçlü-kapanış filtreleri R5 edge'i ve t'yi yükseltiyor mu?

Cache gereksinimi: measure_signal_edge.py çalışmış olmalı.
"""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np
import pandas as pd
from swing_trader.small_cap.regime_logic import regime_from_spy_close

HZ = [3, 5, 10]
SPLIT_DATE = pd.Timestamp('2025-06-01')  # train: öncesi, test: sonrası


def add_ind(df):
    c = df['Close'].astype(float); h = df['High'].astype(float); l = df['Low'].astype(float)
    df = df.copy()
    df['ma50'] = c.rolling(50).mean()
    df['ma20'] = c.rolling(20).mean()
    df['hi20'] = h.rolling(20).max()
    df['vol20'] = df['Volume'].astype(float).rolling(20).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr_pct'] = tr.rolling(14).mean() / c * 100
    return df


def fr(df, ei):
    o = df['Open'].astype(float).values; c = df['Close'].astype(float).values
    n = len(df)
    if ei >= n or o[ei] <= 0:
        return None
    out = {}
    for N in HZ:
        j = ei + N - 1
        out[f'R{N}'] = (c[j] / o[ei] - 1) * 100 if j < n else None
    return out


def welch(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return None
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return round(float((a.mean() - b.mean()) / se), 2) if se > 0 else None


# ── Varyantlar: (df,t)->bool ──
def base(df, t):
    a_now = df['atr_pct'].iloc[t - 1]; a_base = df['atr_pct'].iloc[t - 20:t - 5].mean()
    c = df['Close'].iloc[t]; cp = df['Close'].iloc[t - 1]; hi = df['hi20'].iloc[t - 1]
    if pd.isna(a_now) or pd.isna(a_base) or pd.isna(hi):
        return False
    return a_now < a_base * 0.8 and c > hi > 0 and c > cp


def v_trend(df, t):
    if not base(df, t):
        return False
    c = df['Close'].iloc[t]; ma50 = df['ma50'].iloc[t]
    return not pd.isna(ma50) and c > ma50


def v_vol(df, t):
    if not base(df, t):
        return False
    return df['Volume'].iloc[t] > df['vol20'].iloc[t] * 1.5


def v_strongclose(df, t):
    if not base(df, t):
        return False
    h = df['High'].iloc[t]; l = df['Low'].iloc[t]; c = df['Close'].iloc[t]
    rng = h - l
    return rng > 0 and (c - l) / rng >= 0.6


def v_trend_vol(df, t):
    return v_trend(df, t) and df['Volume'].iloc[t] > df['vol20'].iloc[t] * 1.5


def v_all(df, t):
    if not v_trend_vol(df, t):
        return False
    h = df['High'].iloc[t]; l = df['Low'].iloc[t]; c = df['Close'].iloc[t]
    rng = h - l
    return rng > 0 and (c - l) / rng >= 0.6


VARIANTS = {
    'base(squeeze+brk+green)': base,
    '+trend(>MA50)': v_trend,
    '+volume(1.5x)': v_vol,
    '+strong_close': v_strongclose,
    '+trend+volume': v_trend_vol,
    '+trend+vol+close': v_all,
}


def collect(data, regime_map):
    """Her varyant için (forward, date, ticker, regime) topla + benchmark."""
    out = {k: [] for k in VARIANTS}
    bench = []
    for tk, df in data.items():
        n = len(df)
        for t in range(60, n - 11):
            f = fr(df, t + 1)
            if not f or f.get('R5') is None:
                continue
            bench.append(f)
            day = pd.to_datetime(df['Date'].iloc[t]).normalize()
            reg = regime_map.get(day, 'UNKNOWN')
            for k, fn in VARIANTS.items():
                try:
                    if fn(df, t):
                        out[k].append({**f, 'date': day, 'ticker': tk, 'regime': reg})
                except Exception:
                    pass
    return out, bench


def edge_row(rows, bench, N):
    rv = np.array([r[f'R{N}'] for r in rows if r.get(f'R{N}') is not None])
    bv = np.array([b[f'R{N}'] for b in bench if b.get(f'R{N}') is not None])
    if len(rv) < 10:
        return None
    return {'n': len(rv), 'mean': round(rv.mean(), 2),
            'edge': round(rv.mean() - bv.mean(), 2), 't': welch(rv, bv),
            'wr': round((rv > 0).mean() * 100, 0)}


def main():
    with open('output/_edge_data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('output/_edge_spy.pkl', 'rb') as f:
        spy = pickle.load(f)
    data = {t: add_ind(df) for t, df in data.items()}

    # regime map
    closes = spy['Close'].astype(float).reset_index(drop=True)
    dates = pd.to_datetime(spy['Date']).reset_index(drop=True)
    rmap = {}
    for i in range(len(spy)):
        if i < 50:
            rmap[dates[i].normalize()] = 'UNKNOWN'; continue
        try:
            rmap[dates[i].normalize()] = regime_from_spy_close(closes.iloc[max(0, i - 251):i + 1], None).get('regime', 'UNKNOWN')
        except Exception:
            rmap[dates[i].normalize()] = 'UNKNOWN'

    out, bench = collect(data, rmap)

    print("=" * 96)
    print(" VOLSQUEEZE DOĞRULAMA")
    print("=" * 96)
    bN = {N: np.array([b[f'R{N}'] for b in bench if b.get(f'R{N}') is not None]) for N in HZ}
    print(f"  Benchmark: R5 {bN[5].mean():+.2f}%  R10 {bN[10].mean():+.2f}%  (n={len(bench)})\n")

    # 1) Varyant karşılaştırması
    print("  [1] VARYANTLAR")
    print(f"  {'Varyant':<26}{'n':>5}{'R5 ort':>9}{'R5 edge':>9}{'t5':>6}{'R10 edge':>10}{'t10':>6}{'WR5':>6}")
    print("  " + "-" * 86)
    best = None
    for k, rows in out.items():
        e5 = edge_row(rows, bench, 5); e10 = edge_row(rows, bench, 10)
        if not e5:
            print(f"  {k:<26}{len(rows):>5}  (yetersiz)"); continue
        print(f"  {k:<26}{e5['n']:>5}{e5['mean']:>8.2f}%{e5['edge']:>+8.2f}%{e5['t']:>6}"
              f"{e10['edge']:>+9.2f}%{e10['t']:>6}{e5['wr']:>5.0f}%")
        score = (e10['t'] or 0) + (e5['t'] or 0)
        if best is None or score > best[1]:
            best = (k, score, rows)

    bk, _, brows = best
    print(f"\n  → En güçlü varyant: '{bk}'  (bunu stres testine sokuyoruz)\n")

    # 2) TRAIN/TEST split
    print("  [2] TRAIN/TEST SPLIT  (train < 2025-06-01 ≤ test)")
    tr = [r for r in brows if r['date'] < SPLIT_DATE]
    te = [r for r in brows if r['date'] >= SPLIT_DATE]
    btr = [b for b, df in zip(bench, [None]*len(bench))]  # benchmark split below
    btr = [b for b in bench if b.get('R5') is not None]
    # benchmark'ı da tarihe göre bölmek için tarih yok → tüm benchmark kullan (yaklaşık)
    for lbl, rows in [('TRAIN', tr), ('TEST ', te)]:
        e5 = edge_row(rows, bench, 5); e10 = edge_row(rows, bench, 10)
        if e5:
            print(f"    {lbl}: n={e5['n']:<4} R5 edge {e5['edge']:+.2f}% (t={e5['t']})  "
                  f"R10 edge {e10['edge']:+.2f}% (t={e10['t']})  WR5 {e5['wr']:.0f}%")
        else:
            print(f"    {lbl}: yetersiz örneklem (n={len(rows)})")

    # 3) TICKER yoğunlaşması
    print("\n  [3] TICKER YOĞUNLAŞMASI")
    from collections import Counter
    cnt = Counter(r['ticker'] for r in brows)
    top = cnt.most_common(5)
    print(f"    En çok katkı: {top}")
    top3 = {t for t, _ in cnt.most_common(3)}
    ex = [r for r in brows if r['ticker'] not in top3]
    e10ex = edge_row(ex, bench, 10)
    if e10ex:
        print(f"    Top-3 ticker çıkarılınca: n={e10ex['n']} R10 edge {e10ex['edge']:+.2f}% (t={e10ex['t']})")

    # 4) REGIME
    print("\n  [4] REGIME DAĞILIMI (R10)")
    for reg in ['BULL', 'CAUTION', 'BEAR', 'UNKNOWN']:
        rows = [r for r in brows if r['regime'] == reg]
        e10 = edge_row(rows, bench, 10)
        if e10:
            print(f"    {reg:<8}: n={e10['n']:<4} R10 ort {e10['mean']:+.2f}%  WR... edge {e10['edge']:+.2f}%")

    print("\n" + "=" * 96)
    with open('output/volsqueeze_validation.json', 'w', encoding='utf-8') as f:
        json.dump({'best_variant': bk,
                   'train_n': len(tr), 'test_n': len(te),
                   'top_tickers': top}, f, indent=2, default=str, ensure_ascii=False)
    print("  📁 output/volsqueeze_validation.json")


if __name__ == '__main__':
    main()
