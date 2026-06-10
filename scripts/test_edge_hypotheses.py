"""
EDGE HYPOTHESIS LAB
===================
measure_signal_edge.py'nin cache'lediği AYNI veri + AYNI benchmark çerçevesinde,
bilinen swing-trade edge'lerini test eder. Amaç: "Hangi kurulum rastgele girişi
İSTATİSTİKSEL ANLAMLI biçimde yeniyor?" — yeni tezin çekirdeğini VERİYLE bulmak.

Her hipotez: gün t'nin KAPANIŞINDA tetiklenir (sadece data[:t+1]), giriş t+1 AÇILIŞ.
Forward return + Welch t-test ile benchmark'a karşı kıyaslanır.

Cache gereksinimi: önce measure_signal_edge.py çalışmış olmalı (output/_edge_data.pkl).
"""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np
import pandas as pd

from swing_trader.small_cap.patterns import detect_vcp

HORIZONS = [3, 5, 10]
MFE_WINDOW = 10


# ── indikatör yardımcıları (vektörel) ──
def add_indicators(df):
    c = df['Close'].astype(float)
    df = df.copy()
    df['ma20'] = c.rolling(20).mean()
    df['ma50'] = c.rolling(50).mean()
    df['hi20'] = df['High'].astype(float).rolling(20).max()
    df['hi60'] = df['High'].astype(float).rolling(60).max()
    df['vol20'] = df['Volume'].astype(float).rolling(20).mean()
    # RSI14
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    # ATR%
    h, l = df['High'].astype(float), df['Low'].astype(float)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr_pct'] = tr.rolling(14).mean() / c * 100
    # 5-gün range (sıkışma için)
    df['range5_pct'] = (h.rolling(5).max() - l.rolling(5).min()) / c * 100
    df['range20_pct'] = (h.rolling(20).max() - l.rolling(20).min()) / c * 100
    return df


# ── HİPOTEZLER: (df, t) -> bool. Hepsi sadece t ve öncesini kullanır. ──
def h_vcp_breakout(df, t):
    """VCP tespit + bugün 20-gün pivota kırılım."""
    if t < 35:
        return False
    vcp = detect_vcp(df.iloc[:t + 1])
    if not vcp['detected']:
        return False
    c = float(df['Close'].iloc[t]); hi20_prev = float(df['hi20'].iloc[t - 1])
    return c > hi20_prev > 0


def h_pullback_uptrend(df, t):
    """Uptrend (Close>MA50, MA50 yükseliyor) + RSI<45 pullback + bugün yeşil."""
    c = float(df['Close'].iloc[t]); cp = float(df['Close'].iloc[t - 1])
    ma50 = float(df['ma50'].iloc[t]); ma50_prev = float(df['ma50'].iloc[t - 10]) if t >= 10 else np.nan
    rsi = float(df['rsi'].iloc[t])
    if np.isnan(ma50) or np.isnan(rsi) or np.isnan(ma50_prev):
        return False
    return c > ma50 and ma50 > ma50_prev and rsi < 45 and c > cp


def h_tight_base_breakout(df, t):
    """Sıkışan baz (20g range dar) + bugün 20-gün high kırılımı + hacim."""
    c = float(df['Close'].iloc[t]); hi20_prev = float(df['hi20'].iloc[t - 1])
    rng20 = float(df['range20_pct'].iloc[t - 1]) if not np.isnan(df['range20_pct'].iloc[t - 1]) else 99
    vol = float(df['Volume'].iloc[t]); vol20 = float(df['vol20'].iloc[t])
    if vol20 <= 0 or hi20_prev <= 0:
        return False
    return c > hi20_prev and rng20 < 18 and vol > vol20 * 1.5


def h_oversold_bounce(df, t):
    """RSI<30 + Close>MA50 (trend bozulmamış pullback)."""
    rsi = float(df['rsi'].iloc[t]); c = float(df['Close'].iloc[t]); ma50 = float(df['ma50'].iloc[t])
    if np.isnan(rsi) or np.isnan(ma50):
        return False
    return rsi < 30 and c > ma50


def h_early_accumulation(df, t):
    """Mevcut tezimiz: yüksek RVOL + küçük hareket (sessiz toplama)."""
    vol = float(df['Volume'].iloc[t]); vol20 = float(df['vol20'].iloc[t])
    c = float(df['Close'].iloc[t]); cp = float(df['Close'].iloc[t - 1])
    if vol20 <= 0 or cp <= 0:
        return False
    chg = (c / cp - 1) * 100
    return vol > vol20 * 2.0 and 0 <= chg <= 5


def h_new_high_momentum(df, t):
    """60-gün yeni zirve (saf momentum/breakout)."""
    c = float(df['Close'].iloc[t]); hi60_prev = float(df['hi60'].iloc[t - 1])
    return hi60_prev > 0 and c >= hi60_prev


def h_volsqueeze_breakout(df, t):
    """ATR sıkışması (düşük ATR%) sonrası genişleme + yukarı kapanış."""
    a_now = float(df['atr_pct'].iloc[t - 1]) if not np.isnan(df['atr_pct'].iloc[t - 1]) else None
    a_base = float(df['atr_pct'].iloc[t - 20:t - 5].mean()) if t >= 25 else None
    c = float(df['Close'].iloc[t]); cp = float(df['Close'].iloc[t - 1])
    hi20_prev = float(df['hi20'].iloc[t - 1])
    if a_now is None or a_base is None or np.isnan(a_base):
        return False
    return a_now < a_base * 0.8 and c > hi20_prev > 0 and c > cp


HYPOTHESES = {
    'VCP_breakout': h_vcp_breakout,
    'pullback_uptrend': h_pullback_uptrend,
    'tight_base_breakout': h_tight_base_breakout,
    'oversold_bounce': h_oversold_bounce,
    'early_accumulation': h_early_accumulation,
    'new_high_momentum': h_new_high_momentum,
    'volsqueeze_breakout': h_volsqueeze_breakout,
}


def forward_returns(df, entry_idx):
    o = df['Open'].astype(float).values; c = df['Close'].astype(float).values
    h = df['High'].astype(float).values; l = df['Low'].astype(float).values
    n = len(df)
    if entry_idx >= n:
        return None
    entry = o[entry_idx]
    if entry <= 0:
        return None
    out = {}
    for N in HORIZONS:
        j = entry_idx + N - 1
        out[f'R{N}'] = (c[j] / entry - 1) * 100 if j < n else None
    end = min(entry_idx + MFE_WINDOW, n)
    out['MFE'] = (h[entry_idx:end].max() / entry - 1) * 100 if end > entry_idx else None
    out['MAE'] = (l[entry_idx:end].min() / entry - 1) * 100 if end > entry_idx else None
    return out


def _welch_t(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return None
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return round(float((a.mean() - b.mean()) / se), 2) if se > 0 else None


def main():
    with open('output/_edge_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"  {len(data)} ticker yüklendi (cache). İndikatörler hesaplanıyor...")
    data = {t: add_indicators(df) for t, df in data.items()}

    results = {name: [] for name in HYPOTHESES}
    baseline = []

    for ticker, df in data.items():
        n = len(df)
        for t in range(60, n - MFE_WINDOW - 1):
            entry_idx = t + 1
            fr = forward_returns(df, entry_idx)
            if fr is None or fr.get('R5') is None:
                continue
            baseline.append(fr)
            for name, fn in HYPOTHESES.items():
                try:
                    if fn(df, t):
                        results[name].append(fr)
                except Exception:
                    pass

    # ── Rapor ──
    print("\n" + "=" * 100)
    print(f" EDGE HİPOTEZ LABORATUVARI   (benchmark havuzu: {len(baseline)} bar)")
    print("=" * 100)
    bench = {N: np.array([r[f'R{N}'] for r in baseline if r.get(f'R{N}') is not None]) for N in HORIZONS}
    for N in HORIZONS:
        print(f"  Benchmark R{N}: ort {bench[N].mean():+.2f}%  WR {(bench[N]>0).mean()*100:.0f}%")

    print(f"\n  {'Hipotez':<22}{'n':>6}{'R5 ort':>9}{'R5 edge':>9}{'t(R5)':>7}"
          f"{'R10 ort':>9}{'R10 edge':>10}{'t(R10)':>8}{'WR5':>6}  Karar")
    print("  " + "-" * 96)

    summary = {}
    ranked = []
    for name in HYPOTHESES:
        rows = results[name]
        if len(rows) < 15:
            print(f"  {name:<22}{len(rows):>6}   (yetersiz örneklem)")
            continue
        r5 = np.array([r['R5'] for r in rows if r.get('R5') is not None])
        r10 = np.array([r['R10'] for r in rows if r.get('R10') is not None])
        edge5 = r5.mean() - bench[5].mean()
        edge10 = r10.mean() - bench[10].mean()
        t5 = _welch_t(r5, bench[5]); t10 = _welch_t(r10, bench[10])
        wr5 = (r5 > 0).mean() * 100
        decision = "✅ EDGE" if (t5 and t5 > 2) or (t10 and t10 > 2) else \
                   ("⚠️ zayıf" if (t5 and t5 > 1.3) or (t10 and t10 > 1.3) else "❌ yok")
        ranked.append((name, edge10 if t10 else edge5, t10 or 0))
        summary[name] = {
            'n': len(rows), 'r5_mean': round(float(r5.mean()), 2), 'r5_edge': round(float(edge5), 2),
            't5': t5, 'r10_mean': round(float(r10.mean()), 2), 'r10_edge': round(float(edge10), 2),
            't10': t10, 'wr5': round(float(wr5), 1), 'decision': decision,
        }
        print(f"  {name:<22}{len(rows):>6}{r5.mean():>8.2f}%{edge5:>+8.2f}%{t5 if t5 else 0:>7}"
              f"{r10.mean():>8.2f}%{edge10:>+9.2f}%{t10 if t10 else 0:>8}{wr5:>5.0f}%  {decision}")

    print("\n" + "=" * 100)
    ranked.sort(key=lambda x: -x[2])
    if ranked:
        print("  SIRALAMA (t-stat'a göre, en güçlü edge üstte):")
        for name, edge, t in ranked:
            print(f"    {name:<24} edge {edge:+.2f}%  t={t}")
    print("=" * 100)

    os.makedirs('output', exist_ok=True)
    with open('output/edge_hypotheses.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print("  📁 output/edge_hypotheses.json")


if __name__ == '__main__':
    main()
