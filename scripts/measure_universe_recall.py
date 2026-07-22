# -*- coding: utf-8 -*-
"""
UNIVERSE RECALL ÖLÇÜMÜ — "Aday Bulma Makinesi" Adım 1
=====================================================
Soru: Doğrulanmış 408 VCE sinyalinin (Variant B) kaçı, sinyal GÜNÜNÜN akşamında
bizim 8 Finviz sorgumuzdan en az birine takılırdı? Kaçanlar hangi kriterde ölüyor?

Para çevirisi: her kaçan sinyal = oynanmamış +2.42% (R10) beklenen edge.

Yöntem:
- Sorgu kriterleri OHLCV'den emüle edilir (avg volume ≈ 63g ort., RVOL = hacim/63g ort.,
  haftalık volatilite ≈ 5g ort. (H-L)/önceki kapanış, RSI(14) Wilder).
- Market cap / float: yfinance'ten GÜNCEL pay sayısı × o günkü kapanış (yaklaşıklık —
  2 yıl içinde dilüsyon olabilir; rapor bunu ayrı sayar).
- Post-filter ($8-200) union'dan SONRA uygulanır → "motora ulaşan" recall.

Cache gereksinimi: output/_edge_data.pkl (measure_signal_edge.py).
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np
import pandas as pd

from validate_volsqueeze import add_ind, v_trend, fr, welch

SHARES_CACHE = 'output/_shares_cache.json'
SMALL = (300e6, 2e9)
MID = (2e9, 10e9)

# Post-filter sınırlarını CANLI ayarlardan oku — script ile ürün asla ayrışmasın
from swing_trader.small_cap.settings_config import load_settings
_us = load_settings().universe_scan
POST_MIN, POST_MAX = _us.post_filter_price_min, _us.post_filter_price_max


def wilder_rsi(c, period=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_shares(tickers):
    if os.path.exists(SHARES_CACHE):
        with open(SHARES_CACHE) as f:
            return json.load(f)
    import yfinance as yf
    out = {}
    print(f"  yfinance'ten pay sayıları çekiliyor ({len(tickers)} ticker)...")
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            out[t] = {
                'shares': info.get('sharesOutstanding'),
                'float': info.get('floatShares'),
            }
        except Exception:
            out[t] = {'shares': None, 'float': None}
    with open(SHARES_CACHE, 'w') as f:
        json.dump(out, f)
    return out


def enrich(df):
    df = add_ind(df)  # ma50, hi20, vol20, atr_pct
    c = df['Close'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    v = df['Volume'].astype(float)
    df['avgvol63'] = v.rolling(63).mean()
    df['rvol'] = v / df['avgvol63']
    df['chg'] = (c / c.shift() - 1) * 100
    df['weekvol'] = ((h - l) / c.shift() * 100).rolling(5).mean()
    df['rsi'] = wilder_rsi(c)
    df['hi20_prev'] = h.rolling(20).max().shift()
    return df


def queries_hit(row, mcap, flt):
    """Sinyal gününün akşamında hangi sorgular bu satırı döndürürdü?"""
    price_ok = row.Close > 7
    small = mcap is not None and SMALL[0] <= mcap < SMALL[1]
    mid = mcap is not None and MID[0] <= mcap <= MID[1]
    band_unknown = mcap is None
    small_ = small or band_unknown  # bilinmiyorsa iyimser: geçer say
    mid_ = mid or band_unknown
    flt_ok = flt is None or flt < 100e6

    av = row.avgvol63
    hits = {}
    hits['Q1_momentum'] = small_ and flt_ok and price_ok and row.rvol > 1.5 and row.weekvol > 5 and av > 500e3
    hits['Q2_setup'] = small_ and flt_ok and price_ok and av > 750e3 and row.rsi < 60 and row.weekvol > 3
    hits['Q3_wider'] = small_ and price_ok and row.rvol > 2 and av > 1e6 and row.weekvol > 5
    hits['Q4_early'] = small_ and flt_ok and price_ok and av > 500e3 and row.rsi <= 40 and row.rvol > 1
    hits['Q5_vce_small'] = small_ and price_ok and av > 500e3 and row.rvol > 1.5 and row.chg >= 2
    hits['Q5b_vce_mid'] = mid_ and price_ok and av > 1e6 and row.rvol > 1.5 and row.chg >= 2
    new20 = row.High > row.hi20_prev if not np.isnan(row.hi20_prev) else True
    above50 = row.Close > row.ma50 if not np.isnan(row.ma50) else True
    hits['Q6_20dh_small'] = small_ and price_ok and av > 500e3 and above50 and new20
    hits['Q6b_20dh_mid'] = mid_ and price_ok and av > 1e6 and above50 and new20
    return hits


def main():
    with open('output/_edge_data.pkl', 'rb') as f:
        data = pickle.load(f)
    shares = load_shares(list(data.keys()))
    data = {t: enrich(df) for t, df in data.items()}

    sigs, bench = [], []
    for tk, df in data.items():
        sh = shares.get(tk, {})
        n = len(df)
        for t in range(60, n - 11):
            f_ = fr(df, t + 1)
            if not f_ or f_.get('R5') is None:
                continue
            bench.append(f_)
            if v_trend(df, t):
                row = df.iloc[t]
                mcap = row['Close'] * sh['shares'] if sh.get('shares') else None
                hits = queries_hit(row, mcap, sh.get('float'))
                sigs.append({
                    **f_, 'ticker': tk, 'date': pd.to_datetime(df['Date'].iloc[t]),
                    'close': float(row['Close']), 'avgvol63': float(row['avgvol63']),
                    'rvol': float(row['rvol']), 'chg': float(row['chg']),
                    'mcap': mcap, 'hits': hits, 'union': any(hits.values()),
                })

    n = len(sigs)
    print("=" * 94)
    print(f" UNIVERSE RECALL — Variant B sinyalleri, n={n} (57 ticker, 2024-06→2026-05)")
    print("=" * 94)

    # [1] Sorgu bazında yakalama
    print("\n  [1] SORGU BAZINDA YAKALAMA (sinyal gününün akşamı)")
    qnames = list(sigs[0]['hits'].keys())
    for q in qnames:
        k = sum(1 for s in sigs if s['hits'][q])
        print(f"    {q:<16} {k:>4}/{n}  ({k/n*100:5.1f}%)")

    caught = [s for s in sigs if s['union']]
    missed = [s for s in sigs if not s['union']]
    print(f"\n    UNION (herhangi biri): {len(caught)}/{n}  ({len(caught)/n*100:.1f}%)  ← RECALL")

    # Post-filter etkisi (motora ulaşan)
    pf = [s for s in caught if POST_MIN <= s['close'] <= POST_MAX]
    print(f"    + post-filter ${POST_MIN:.0f}-{POST_MAX:.0f} : {len(pf)}/{n}  ({len(pf)/n*100:.1f}%)  ← MOTORA ULAŞAN")
    pf_kill = [s for s in caught if s not in pf]

    # [2] Kaçanların otopsisi
    print(f"\n  [2] KAÇANLARIN OTOPSİSİ (union'a takılmayan {len(missed)} sinyal)")
    reasons = {'avgvol_small': 0, 'price_le7': 0, 'mcap_out': 0, 'other': 0}
    for s in missed:
        mcap = s['mcap']
        in_small = mcap is not None and SMALL[0] <= mcap < SMALL[1]
        in_mid = mcap is not None and MID[0] <= mcap <= MID[1]
        need_av = 500e3 if (in_small or mcap is None) else 1e6
        if mcap is not None and not in_small and not in_mid:
            reasons['mcap_out'] += 1
        elif s['close'] <= 7:
            reasons['price_le7'] += 1
        elif s['avgvol63'] <= need_av:
            reasons['avgvol_small'] += 1
        else:
            reasons['other'] += 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        if v:
            print(f"    {k:<14} {v:>3}  ({v/max(len(missed),1)*100:.0f}%)")
    if pf_kill:
        print(f"    (+ post-filter kurbanı: {len(pf_kill)} — union yakaladı, $8-200 bandı öldürdü; "
              f"fiyatlar: {sorted(round(s['close'],2) for s in pf_kill)[:8]})")

    # [3] Kaçan ve yakalananların EDGE'i — kaçanlar değerli mi?
    print(f"\n  [3] GRUPLARIN R10 EDGE'İ (kaçırdığımız para)")
    bv = np.array([b['R10'] for b in bench if b.get('R10') is not None])
    for lbl, grp in [('YAKALANAN (motora ulaşan)', pf), ('KAÇAN (union dışı)', missed),
                     ('post-filter kurbanı', pf_kill)]:
        rv = np.array([s['R10'] for s in grp if s.get('R10') is not None])
        if len(rv) >= 5:
            print(f"    {lbl:<28} n={len(rv):<4} R10 ort {rv.mean():+6.2f}%  "
                  f"edge {rv.mean()-bv.mean():+6.2f}%  t={welch(rv, bv)}")
        else:
            print(f"    {lbl:<28} n={len(rv)} (istatistik için az)")

    # [3b] Q5/Q5b MARJİNAL KATKI — Q6/Q6b'siz sadece Q5/Q5b'nin edge'i +
    # asıl soru: Q6/Q6b zaten yakalamışken Q5/Q5b'nin EKSTRA getirdiği var mı?
    print(f"\n  [3b] Q5/Q5b MARJİNAL KATKI (Q6/Q6b zaten yakalamışken Q5/Q5b ekstra ne katıyor?)")
    q6_group = ('Q6_20dh_small', 'Q6b_20dh_mid')
    q5_group = ('Q5_vce_small', 'Q5b_vce_mid')

    def hit_any(s, keys):
        return any(s['hits'][k] for k in keys)

    non_q5_group = ('Q1_momentum', 'Q2_setup', 'Q3_wider', 'Q4_early') + q6_group

    only_q6 = [s for s in sigs if hit_any(s, q6_group)]
    only_q5 = [s for s in sigs if hit_any(s, q5_group)]
    q5_exclusive = [s for s in sigs if hit_any(s, q5_group) and not hit_any(s, q6_group)]
    # "Q5/Q5b'yi TAMAMEN kapatsaydık" senaryosu: diğer TÜM sorguların (Q1-4, Q6/Q6b)
    # union'ı — s['union'] kullanmak YANLIŞ olurdu çünkü o zaten Q5'i içeriyor.
    union_without_q5 = [s for s in sigs if hit_any(s, non_q5_group)]

    for lbl, grp in [
        ('Q6/Q6b tek başına (recall)', only_q6),
        ('Q5/Q5b tek başına (recall)', only_q5),
        ('Q5/Q5b YALNIZ yakaladı (diğer 6 sorgu kaçırdı)', q5_exclusive),
    ]:
        print(f"    {lbl:<46} n={len(grp):>4}/{n}  ({len(grp)/n*100:5.1f}%)")

    print(f"\n    Recall Q5/Q5b KAPALI (diğer 6 sorgu)  : {len(union_without_q5)}/{n} "
          f"({len(union_without_q5)/n*100:.1f}%)")
    print(f"    Recall TÜM sorgular (Q5 DAHİL, mevcut) : {len(caught)}/{n} ({len(caught)/n*100:.1f}%)")
    print(f"    → Q5/Q5b'yi kapatırsak KAYBEDİLEN      : {len(caught)-len(union_without_q5)} sinyal")

    if q5_exclusive:
        rv = np.array([s['R10'] for s in q5_exclusive if s.get('R10') is not None])
        if len(rv) >= 3:
            print(f"\n    Q5/Q5b-yalnız yakalananların R10 edge'i: n={len(rv)} ort {rv.mean():+.2f}%  "
                  f"edge {rv.mean()-bv.mean():+.2f}%  t={welch(rv, bv)}")
        else:
            print(f"\n    Q5/Q5b-yalnız yakalananlar: n={len(rv)} (istatistik için çok az — "
                  f"tek tek incele:")
            for s in q5_exclusive:
                print(f"      {s['ticker']:<6} {str(s['date'])[:10]}  R10={s.get('R10')}")
    else:
        print(f"\n    Q5/Q5b'nin Q6/Q6b'ye kattığı TEK bir sinyal bile yok — tamamen artık (redundant).")

    # [4] Eşik duyarlılığı: avg volume barı
    print(f"\n  [4] AVG VOLUME EŞİĞİ DUYARLILIĞI (Q6 small bandı için; recall ne kazanır?)")
    for th in [250e3, 400e3, 500e3, 750e3, 1e6]:
        k = 0
        for s in sigs:
            mcap = s['mcap']
            in_mid = mcap is not None and MID[0] <= mcap <= MID[1]
            need = 1e6 if in_mid else th
            if s['close'] > 7 and s['avgvol63'] > need:
                k += 1
        print(f"    avgvol > {th/1e3:>5.0f}K → union-yaklaşık recall {k}/{n} ({k/n*100:.1f}%)")

    # [5] mcap bilinmeyenler
    unk = sum(1 for s in sigs if s['mcap'] is None)
    if unk:
        print(f"\n  NOT: {unk} sinyalde pay sayısı yok → mcap bandı 'geçti' varsayıldı (iyimser).")
    print("\n" + "=" * 94)


if __name__ == '__main__':
    main()
