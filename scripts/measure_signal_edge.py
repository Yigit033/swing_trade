"""
SIGNAL EDGE MEASUREMENT HARNESS
================================
"Engine BUY dediğinde sonraki 3/5/10 günde ne oluyor — ve bu rastgele bir
small-cap'e göre daha mı iyi?"  Saf SİNYAL edge'ini ölçer (portföy/exit yönetimi
karışmadan), ve bir BENCHMARK'a karşı kıyaslar.

Tasarım (lookahead YOK):
  - Gün t'de sadece data[:t+1] ile scan_stock çalışır (gelecek bilgisi yok)
  - Sinyal ateşlenirse giriş = gün t+1 AÇILIŞ (canlı PENDING mekaniğiyle aynı)
  - Forward return: R_N = close[t+1+N] / entry_open - 1   (N = 3,5,10)
  - MFE/MAE: 10 gün içindeki max yükseliş / max düşüş
  - Regime: SPY'den günlük hesaplanır (canlı quality bar davranışıyla eşleşsin diye)

BENCHMARK (edge'in kalbi):
  - Aynı evren + aynı dönemdeki HER (ticker, gün) için forward return havuzu
  - Edge = mean(sinyal forward) - mean(baseline forward), her horizon için
  - Ayrıca Welch t-test ile istatistiksel anlamlılık

Çıktı: konsol raporu + output/signal_edge.json
"""
import sys, os, argparse, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.ERROR)

import numpy as np
import pandas as pd

from swing_trader.small_cap.engine import SmallCapEngine
from swing_trader.small_cap.regime_logic import regime_from_spy_close

# ── Temsili, sektör-çeşitli small/mid-cap evren (canlı taramayı taklit eder) ──
UNIVERSE = [
    # Tech / semis
    'AEHR','AMBA','CRDO','LSCC','RMBS','SITM','SMTC','WOLF','DIOD','ONTO',
    # Software / AI
    'BBAI','SOUN','DOCN','GTLB','MNDY','RDDT','TOST','APPF','BRZE',
    # Defense / space
    'RKLB','KTOS','ASTS','LUNR','ACHR','JOBY','RDW',
    # Industrials
    'POWL','AAON','CSWI','MLI','SSD','IBP','NVEE',
    # Energy / nuclear / clean
    'OKLO','SMR','NNE','UUUU','FLNC','SHLS','OII',
    # Healthcare
    'HIMS','ADMA','CPRX','HALO','IOVA','VKTX','NTRA',
    # Consumer
    'CAVA','CELH','ELF','WING','BROS','SHAK','PTLO',
    # Fintech / financial
    'SOFI','MGNI','STEP','TBBK','HCI',
]

HORIZONS = [3, 5, 10]
MFE_WINDOW = 10


def fetch_data(tickers, start, end, cache_path):
    """Batch yfinance download, diske cache'le."""
    if os.path.exists(cache_path):
        print(f"  Cache'ten yükleniyor: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    import yfinance as yf
    print(f"  yfinance batch download: {len(tickers)} ticker, {start} → {end} ...")
    raw = yf.download(tickers, start=start, end=end, group_by='ticker',
                      auto_adjust=True, progress=False, threads=True)
    data = {}
    for t in tickers:
        try:
            df = raw[t].dropna().reset_index()
            if len(df) >= 80:
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                data[t] = df
        except Exception:
            pass
    print(f"  {len(data)}/{len(tickers)} ticker için yeterli veri")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    return data


def fetch_spy(start, end, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    import yfinance as yf
    spy = yf.download('SPY', start=start, end=end, auto_adjust=True, progress=False)
    spy = spy.reset_index()
    spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy['Date'] = pd.to_datetime(spy['Date']).dt.tz_localize(None)
    with open(cache_path, 'wb') as f:
        pickle.dump(spy, f)
    return spy


def build_regime_map(spy_df):
    """Her tarih için SPY'den regime (BULL/CAUTION/BEAR) önceden hesapla."""
    rmap = {}
    closes = spy_df['Close'].astype(float).reset_index(drop=True)
    dates = pd.to_datetime(spy_df['Date']).reset_index(drop=True)
    for i in range(len(spy_df)):
        if i < 50:
            rmap[dates[i].normalize()] = 'UNKNOWN'
            continue
        window = closes.iloc[max(0, i - 251):i + 1]
        try:
            r = regime_from_spy_close(window, None)
            rmap[dates[i].normalize()] = r.get('regime', 'UNKNOWN')
        except Exception:
            rmap[dates[i].normalize()] = 'UNKNOWN'
    return rmap


def forward_returns(df, entry_idx):
    """entry_idx = girişin yapıldığı bar (t+1). Açılıştan forward return'ler."""
    o = df['Open'].astype(float).values
    c = df['Close'].astype(float).values
    h = df['High'].astype(float).values
    l = df['Low'].astype(float).values
    n = len(df)
    if entry_idx >= n:
        return None
    entry = o[entry_idx]
    if entry <= 0:
        return None
    out = {}
    for N in HORIZONS:
        j = entry_idx + N - 1  # N bar sonra (giriş günü dahil)
        out[f'R{N}'] = (c[j] / entry - 1) * 100 if j < n else None
    end = min(entry_idx + MFE_WINDOW, n)
    if end > entry_idx:
        out['MFE'] = (h[entry_idx:end].max() / entry - 1) * 100
        out['MAE'] = (l[entry_idx:end].min() / entry - 1) * 100
    else:
        out['MFE'] = out['MAE'] = None
    return out


def run(start, end, scan_every=1):
    engine = SmallCapEngine()
    os.makedirs('output', exist_ok=True)
    data = fetch_data(UNIVERSE, start, end, 'output/_edge_data.pkl')
    spy = fetch_spy(start, end, 'output/_edge_spy.pkl')
    regime_map = build_regime_map(spy)

    signals = []      # her ateşlenen sinyalin forward return'leri
    baseline = []     # HER (ticker, gün)'ün forward return'leri (benchmark havuzu)

    print(f"\n  Walk-forward tarama ({len(data)} ticker)...")
    for ti, (ticker, df) in enumerate(sorted(data.items())):
        n = len(df)
        # warmup 60 bar; sonu MFE_WINDOW+1 bar boşluk bırak (forward için)
        for t in range(60, n - MFE_WINDOW - 1, scan_every):
            entry_idx = t + 1  # ertesi gün açılışta giriş
            fr = forward_returns(df, entry_idx)
            if fr is None or fr.get('R5') is None:
                continue
            baseline.append(fr)

            day = pd.to_datetime(df['Date'].iloc[t]).normalize()
            regime = regime_map.get(day, 'UNKNOWN')
            df_window = df.iloc[:t + 1]
            try:
                sig = engine.scan_stock(ticker, df_window, backtest_mode=True,
                                        portfolio_value=10000, regime=regime)
            except Exception:
                sig = None
            if sig:
                signals.append({
                    'ticker': ticker,
                    'date': str(day.date()),
                    'type': sig.get('swing_type', '?'),
                    'quality': sig.get('quality_score', 0),
                    'regime': regime,
                    **fr,
                })
        print(f"    [{ti+1:2d}/{len(data)}] {ticker}: tarandı")

    report(signals, baseline, start, end)


def _stats(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    arr = np.array(vals)
    return {
        'n': len(arr),
        'mean': round(float(arr.mean()), 2),
        'median': round(float(np.median(arr)), 2),
        'win_rate': round(float((arr > 0).mean()) * 100, 1),
    }


def _welch_t(a, b):
    """Welch t-test (scipy yoksa elle)."""
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return None
    return round(float((a.mean() - b.mean()) / se), 2)


def report(signals, baseline, start, end):
    print("\n" + "=" * 88)
    print(f" SİNYAL EDGE RAPORU  ({start} → {end})")
    print("=" * 88)
    print(f"  Toplam sinyal: {len(signals)}  |  Benchmark havuzu (tüm bar): {len(baseline)}")

    if not signals:
        print("\n  ⚠️ HİÇ SİNYAL ATEŞLENMEDİ — sistem bu evrende çok seçici.")
        return

    print(f"\n  {'Horizon':<10}{'SİNYAL ort':>12}{'WR':>8}{'BENCHMARK ort':>16}{'WR':>8}"
          f"{'EDGE':>10}{'t-stat':>9}")
    print("  " + "-" * 80)
    edge_summary = {}
    for N in HORIZONS:
        k = f'R{N}'
        s = _stats(signals, k)
        b = _stats(baseline, k)
        if not s or not b:
            continue
        edge = round(s['mean'] - b['mean'], 2)
        sv = [r[k] for r in signals if r.get(k) is not None]
        bv = [r[k] for r in baseline if r.get(k) is not None]
        t = _welch_t(sv, bv)
        edge_summary[k] = {'signal': s, 'benchmark': b, 'edge': edge, 't_stat': t}
        flag = "  ✅" if (t and t > 2) else ("  ⚠️" if (t and t > 1) else "  ❌")
        print(f"  R{N:<9}{s['mean']:>11.2f}%{s['win_rate']:>7.0f}%"
              f"{b['mean']:>15.2f}%{b['win_rate']:>7.0f}%{edge:>+9.2f}%{t if t else 0:>9}{flag}")

    # MFE/MAE
    smfe, smae = _stats(signals, 'MFE'), _stats(signals, 'MAE')
    bmfe, bmae = _stats(baseline, 'MFE'), _stats(baseline, 'MAE')
    if smfe and smae:
        print(f"\n  MFE (10g max↑): sinyal {smfe['mean']:+.1f}% vs benchmark {bmfe['mean']:+.1f}%")
        print(f"  MAE (10g max↓): sinyal {smae['mean']:+.1f}% vs benchmark {bmae['mean']:+.1f}%")

    # Tip kırılımı
    print(f"\n  TİP KIRILIMI (R5):")
    for typ in ['C', 'A', 'B', 'S']:
        rows = [r for r in signals if r['type'] == typ]
        st = _stats(rows, 'R5')
        if st:
            print(f"    Type {typ}: n={st['n']:<4} R5 ort {st['mean']:+.2f}%  WR {st['win_rate']:.0f}%")

    # Quality bucket kırılımı
    print(f"\n  QUALITY BUCKET (R5):")
    for lo, hi in [(0, 60), (60, 70), (70, 80), (80, 200)]:
        rows = [r for r in signals if lo <= r['quality'] < hi]
        st = _stats(rows, 'R5')
        if st:
            print(f"    Q{lo}-{hi}: n={st['n']:<4} R5 ort {st['mean']:+.2f}%  WR {st['win_rate']:.0f}%")

    # Regime kırılımı
    print(f"\n  REGIME (R5):")
    for reg in ['BULL', 'CAUTION', 'BEAR', 'UNKNOWN']:
        rows = [r for r in signals if r['regime'] == reg]
        st = _stats(rows, 'R5')
        if st:
            print(f"    {reg:<8}: n={st['n']:<4} R5 ort {st['mean']:+.2f}%  WR {st['win_rate']:.0f}%")

    print("\n" + "=" * 88)
    print("  YORUM: EDGE = sinyal ort − benchmark ort. t-stat>2 ✅ = istatistiksel anlamlı edge.")
    print("         EDGE ≤ 0 veya t-stat<1 ❌ = sinyalin rastgele girişe üstünlüğü YOK.")
    print("=" * 88)

    with open('output/signal_edge.json', 'w', encoding='utf-8') as f:
        json.dump({
            'period': f'{start}/{end}',
            'n_signals': len(signals),
            'n_baseline': len(baseline),
            'edge_summary': edge_summary,
            'signals': signals,
        }, f, indent=2, ensure_ascii=False, default=str)
    print("  📁 output/signal_edge.json")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2024-06-01')
    ap.add_argument('--end', default='2026-05-31')
    ap.add_argument('--scan-every', type=int, default=1,
                    help='Her N günde bir tara (hız için; 1=her gün)')
    a = ap.parse_args()
    run(a.start, a.end, a.scan_every)
