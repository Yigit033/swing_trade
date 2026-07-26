# -*- coding: utf-8 -*-
"""
SİNYAL AİLESİ KEŞİF LABORATUVARI
=================================
Soru (kullanıcının #1 önceliği): VCE'nin KAÇIRDIĞI gerçek yükselişlerin
ortak, ÖLÇÜLEBİLİR bir imzası var mı? Varsa VCE'nin yanına ikinci/üçüncü
bir sinyal ailesi ekleyip recall'ü büyütebilir miyiz?

Yöntem (measure_signal_edge.py ile AYNI istatistiksel standart):
  - Her aday kalıp gün t'de sadece data[:t+1] ile değerlendirilir (lookahead YOK)
  - Giriş = t+1 açılış (canlı PENDING mekaniğiyle aynı)
  - Forward return R5/R10/R20 + gerçekçi trading sim (20 gün tut, %10 stop)
  - BENCHMARK: aynı evrendeki HER (ticker, gün) forward havuzu
  - Edge = mean(kalıp) - mean(benchmark), Welch t ile anlamlılık
  - OOS split (2025-06-01): edge ikinci yarıda da duruyor mu?
  - VCE ile örtüşme: kalıp VCE'nin GÖRMEDİĞİ fırsatları mı yakalıyor?

Bir kalıbın "kabul edilebilir" olması için (senior bar):
  1. R10 edge > 0 VE Welch t >= 2.0 (tam örneklem)
  2. OOS'ta da edge > 0 (fluke değil)
  3. Gerçek trading sim'de rastgele baseline'ı belirgin geçmeli
  4. VCE'ye net marjinal katkı (VCE'nin kaçırdığını yakalamalı)

Cache gereksinimi: output/_edge_data.pkl + _edge_spy.pkl (measure_signal_edge.py)
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np
import pandas as pd

from swing_trader.small_cap.signals import SmallCapSignals

HZ = [5, 10, 20]
SPLIT = pd.Timestamp("2025-06-01")
STOP_PCT = 10.0
HOLD = 20


# ══════════════════════════════════════════════════════════════════════
# İNDİKATÖR ZENGİNLEŞTİRME (tüm kalıplar bu kolonları kullanır)
# ══════════════════════════════════════════════════════════════════════
def enrich(df):
    df = df.copy()
    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    v = df["Volume"].astype(float)
    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()
    df["ma10"] = c.rolling(10).mean()
    df["hi20"] = h.rolling(20).max()
    df["hi50"] = h.rolling(50).max()
    df["lo20"] = l.rolling(20).min()
    df["vol20"] = v.rolling(20).mean()
    df["vol50"] = v.rolling(50).mean()
    df["rvol"] = v / df["vol50"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / c * 100
    df["chg"] = (c / c.shift() - 1) * 100
    df["chg5"] = (c / c.shift(5) - 1) * 100
    df["chg10"] = (c / c.shift(10) - 1) * 100
    df["chg20"] = (c / c.shift(20) - 1) * 100
    # RSI (Wilder)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    # dipten bu yana kaç gün (52 haftalık düşük yakınlığı için basit proxy)
    df["above_ma50"] = c > df["ma50"]
    return df


# ══════════════════════════════════════════════════════════════════════
# ADAY SİNYAL KALIPLARI — her biri (df, t) -> bool
# VCE'nin kaçırdığı farklı "yükseliş şekillerini" hedefler.
# ══════════════════════════════════════════════════════════════════════
def _ok(*vals):
    return all(not pd.isna(v) for v in vals)


def p_vce_baseline(df, t):
    """Referans: mevcut VCE (Variant B) — kıyas için."""
    a_now = df["atr_pct"].iloc[t - 1]
    a_base = df["atr_pct"].iloc[t - 20:t - 5].mean()
    c = df["Close"].iloc[t]; cp = df["Close"].iloc[t - 1]; hi = df["hi20"].iloc[t - 1]
    ma50 = df["ma50"].iloc[t]
    if not _ok(a_now, a_base, hi, ma50): return False
    return a_now < a_base * 0.8 and c > hi > 0 and c > cp and c > ma50


def p_ma50_reclaim(df, t):
    """DİP-DÖNÜŞÜ: dün MA50 altında, bugün ilk kez üstüne kapanış + hacim.
    ORIC Ekim-2024 tipi (uzun düşüşten ilk toparlanma) — VCE bunu sıkışma
    aramadığı için kaçırır."""
    c = df["Close"].iloc[t]; cp = df["Close"].iloc[t - 1]
    ma50 = df["ma50"].iloc[t]; ma50p = df["ma50"].iloc[t - 1]
    rvol = df["rvol"].iloc[t]
    if not _ok(c, cp, ma50, ma50p, rvol): return False
    return cp <= ma50p and c > ma50 and c > cp and rvol > 1.2


def p_pullback_bounce(df, t):
    """YÜKSELİŞTE PULLBACK: MA50 üstünde uptrend, MA20'ye geri çekilip
    yeşil dönüş. AMTB tipi kademeli tırmanış."""
    c = df["Close"].iloc[t]; cp = df["Close"].iloc[t - 1]
    ma20 = df["ma20"].iloc[t]; ma50 = df["ma50"].iloc[t]
    lo = df["Low"].iloc[t]
    if not _ok(c, cp, ma20, ma50): return False
    # uptrend (MA20>MA50), fiyat MA20'ye dokundu/altına indi ama yeşil kapandı
    return ma20 > ma50 and c > ma50 and lo <= ma20 and c > cp and c >= ma20 * 0.99


def p_50d_high(df, t):
    """50-GÜNLÜK YENİ ZİRVE: 20g'den daha güçlü kırılım (daha uzun direnç)."""
    c = df["Close"].iloc[t]; cp = df["Close"].iloc[t - 1]
    hi50 = df["hi50"].iloc[t - 1]; ma50 = df["ma50"].iloc[t]
    if not _ok(c, cp, hi50, ma50): return False
    return c > hi50 > 0 and c > cp and c > ma50


def p_rvol_thrust(df, t):
    """HACİM PATLAMASI: RVOL>2.5 + yeşil + MA20 üstü — momentum ignition,
    sıkışma şartı YOK (VCE'nin katı olduğu yer)."""
    c = df["Close"].iloc[t]; cp = df["Close"].iloc[t - 1]
    rvol = df["rvol"].iloc[t]; ma20 = df["ma20"].iloc[t]
    if not _ok(c, cp, rvol, ma20): return False
    return rvol > 2.5 and c > cp and c > ma20


def p_momentum_continuation(df, t):
    """TREND SÜREKLİLİĞİ: 20g'de +%15 üstü VE hâlâ yükseliyor VE aşırı-uzamamış
    (bugün tek başına <%8). 'Koşan at koşmaya devam eder' tezi."""
    c = df["Close"].iloc[t]; cp = df["Close"].iloc[t - 1]
    chg20 = df["chg20"].iloc[t]; chg = df["chg"].iloc[t]
    ma20 = df["ma20"].iloc[t]; ma50 = df["ma50"].iloc[t]
    if not _ok(c, cp, chg20, chg, ma20, ma50): return False
    return chg20 > 15 and c > ma20 > ma50 and c > cp and chg < 8


def p_tight_consolidation_break(df, t):
    """DAR KONSOLİDASYON KIRILIMI: son 10 gün dar bant (range<%8), sonra üst
    kırılım. VCE'ye benzer ama ATR yerine fiyat-bandı kullanır (farklı yakalar)."""
    c = df["Close"].iloc[t]; cp = df["Close"].iloc[t - 1]
    hi10 = df["High"].iloc[t - 10:t].max()
    lo10 = df["Low"].iloc[t - 10:t].min()
    ma50 = df["ma50"].iloc[t]
    if not _ok(c, cp, hi10, lo10, ma50) or lo10 <= 0: return False
    band = (hi10 - lo10) / lo10 * 100
    return band < 8 and c > hi10 and c > cp and c > ma50


def p_oversold_reversal(df, t):
    """AŞIRI SATIM DÖNÜŞÜ: RSI dünden <35, bugün yeşil güçlü toparlanma + hacim.
    Dip avı — riskli ama bazı en büyük yükselişler buradan."""
    c = df["Close"].iloc[t]; cp = df["Close"].iloc[t - 1]
    rsi_p = df["rsi"].iloc[t - 1]; rvol = df["rvol"].iloc[t]; chg = df["chg"].iloc[t]
    if not _ok(c, cp, rsi_p, rvol, chg): return False
    return rsi_p < 35 and chg > 3 and rvol > 1.5 and c > cp


def p_higher_low_breakout(df, t):
    """YÜKSELEN DİP + KIRILIM: son 20g'de dip yükseliyor (lo20 > önceki lo20)
    VE 20g zirve kırılımı. Yapısal uptrend teyidi."""
    c = df["Close"].iloc[t]; cp = df["Close"].iloc[t - 1]
    hi20 = df["hi20"].iloc[t - 1]
    lo_now = df["lo20"].iloc[t]; lo_prev = df["lo20"].iloc[t - 10]
    ma50 = df["ma50"].iloc[t]
    if not _ok(c, cp, hi20, lo_now, lo_prev, ma50): return False
    return c > hi20 > 0 and lo_now > lo_prev and c > cp and c > ma50


def p_gap_up_hold(df, t):
    """GAP-UP TUTMA: bugün açılış dünkü kapanışın >%3 üstünde açtı VE
    gap'i koruyup yeşil kapandı (açılışın üstünde). Katalizör günü imzası."""
    o = df["Open"].iloc[t]; c = df["Close"].iloc[t]; cp = df["Close"].iloc[t - 1]
    ma50 = df["ma50"].iloc[t]
    if not _ok(o, c, cp, ma50) or cp <= 0: return False
    gap = (o / cp - 1) * 100
    return gap > 3 and c >= o and c > cp and c > ma50


PATTERNS = {
    "VCE (mevcut)": p_vce_baseline,
    "MA50 reclaim": p_ma50_reclaim,
    "Pullback bounce": p_pullback_bounce,
    "50d new high": p_50d_high,
    "RVOL thrust": p_rvol_thrust,
    "Momentum cont.": p_momentum_continuation,
    "Tight consol. break": p_tight_consolidation_break,
    "Oversold reversal": p_oversold_reversal,
    "Higher-low breakout": p_higher_low_breakout,
    "Gap-up hold": p_gap_up_hold,
}


# ══════════════════════════════════════════════════════════════════════
# ÖLÇÜM
# ══════════════════════════════════════════════════════════════════════
def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return None
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return round(float((a.mean() - b.mean()) / se), 2) if se > 0 else None


def sim_trade(entry_idx, c, h, low):
    """Gerçekçi: t+1 açılıştan gir (burada entry_idx = giriş barı), 20 gün tut,
    %10 stop. entry = o[entry_idx]. Basitlik için close-to-close + stop low ile."""
    entry = c[entry_idx]
    if entry <= 0:
        return None
    stop = entry * (1 - STOP_PCT / 100)
    for j in range(entry_idx + 1, min(entry_idx + HOLD + 1, len(c))):
        if low[j] <= stop:
            return (stop / entry - 1) * 100
    j = min(entry_idx + HOLD, len(c) - 1)
    return (c[j] / entry - 1) * 100


def fwd(df, entry_idx):
    o = df["Open"].astype(float).values
    c = df["Close"].astype(float).values
    n = len(df)
    if entry_idx >= n or o[entry_idx] <= 0:
        return None
    entry = o[entry_idx]
    out = {}
    for N in HZ:
        j = entry_idx + N - 1
        out[f"r{N}"] = (c[j] / entry - 1) * 100 if j < n else None
    return out


def main():
    with open("output/_edge_data.pkl", "rb") as f:
        data = pickle.load(f)
    data = {t: enrich(df) for t, df in data.items()}

    # Her kalıp için: forward return listeleri + trade sim + tarih + VCE örtüşme
    hits = {k: [] for k in PATTERNS}
    vce_dates = {}   # (ticker) -> set of VCE fire indices
    bench = []

    for tk, df in data.items():
        c = df["Close"].astype(float).values
        h = df["High"].astype(float).values
        low = df["Low"].astype(float).values
        n = len(df)
        vset = set()
        for t in range(60, n - 21):
            entry_idx = t + 1
            fr = fwd(df, entry_idx)
            if not fr or fr.get("r5") is None:
                continue
            trade = sim_trade(entry_idx, c, h, low)
            bench.append({**fr, "trade": trade})
            day = pd.to_datetime(df["Date"].iloc[t])
            for name, fn in PATTERNS.items():
                try:
                    if fn(df, t):
                        hits[name].append({
                            **fr, "trade": trade, "date": day, "ticker": tk, "t": t,
                        })
                        if name == "VCE (mevcut)":
                            vset.add(t)
                except Exception:
                    pass
        vce_dates[tk] = vset

    # Benchmark referansları
    b_r10 = np.array([b["r10"] for b in bench if b.get("r10") is not None])
    b_trade = np.array([b["trade"] for b in bench if b.get("trade") is not None])
    bench_r10 = b_r10.mean()
    bench_trade = b_trade.mean()
    bench_win = (b_trade > 0).mean() * 100

    print("=" * 108)
    print(f" SİNYAL AİLESİ KEŞİF — {len(data)} ticker, 2024-06→2026-05 | benchmark: "
          f"R10 {bench_r10:+.2f}% | trade(20g,%10stop) {bench_trade:+.2f}% WR {bench_win:.0f}%")
    print("=" * 108)
    print(f"  {'Kalıp':<22}{'n':>5}{'R10 ort':>9}{'R10 edge':>10}{'t10':>6}"
          f"{'trade ort':>11}{'WR':>5}{'OOS edge':>10}{'VCE-dışı%':>11}")
    print("  " + "-" * 104)

    results = {}
    for name, fn in PATTERNS.items():
        rows = hits[name]
        r10 = np.array([r["r10"] for r in rows if r.get("r10") is not None])
        tr = np.array([r["trade"] for r in rows if r.get("trade") is not None])
        if len(r10) < 8:
            print(f"  {name:<22}{len(rows):>5}  (yetersiz örneklem)")
            continue
        edge10 = r10.mean() - bench_r10
        t10 = welch(r10, b_r10)
        # OOS
        oos = [r["r10"] for r in rows if r.get("r10") is not None and r["date"] >= SPLIT]
        oos_edge = (np.mean(oos) - bench_r10) if len(oos) >= 8 else None
        # VCE örtüşme: bu kalıbın yakaladıklarının kaçı VCE'nin GÖRMEDİĞİ?
        non_vce = sum(1 for r in rows if r["t"] not in vce_dates.get(r["ticker"], set()))
        non_vce_pct = non_vce / len(rows) * 100
        tr_win = (tr > 0).mean() * 100

        oos_str = f"{oos_edge:+.2f}%" if oos_edge is not None else "  -"
        print(f"  {name:<22}{len(rows):>5}{r10.mean():>+8.2f}%{edge10:>+9.2f}%{t10 if t10 else 0:>6}"
              f"{tr.mean():>+10.2f}%{tr_win:>4.0f}%{oos_str:>10}{non_vce_pct:>10.0f}%")
        results[name] = {
            "n": len(rows), "r10_edge": round(edge10, 2), "t10": t10,
            "trade_mean": round(float(tr.mean()), 2), "trade_win": round(tr_win, 1),
            "oos_edge": round(oos_edge, 2) if oos_edge is not None else None,
            "non_vce_pct": round(non_vce_pct, 1),
        }

    print("\n  KABUL KRİTERİ (senior bar): R10 edge>0 & t10>=2.0 & OOS edge>0 & "
          "trade WR > benchmark & VCE-dışı katkı yüksek")
    print("  " + "-" * 104)
    winners = []
    for name, r in results.items():
        if name == "VCE (mevcut)":
            continue
        ok = (r["r10_edge"] > 0 and (r["t10"] or 0) >= 2.0 and
              (r["oos_edge"] or -1) > 0 and r["trade_win"] > bench_win and
              r["non_vce_pct"] >= 50)
        verdict = "✅ KABUL" if ok else "❌ ele"
        reasons = []
        if r["r10_edge"] <= 0: reasons.append("edge≤0")
        if (r["t10"] or 0) < 2.0: reasons.append(f"t={r['t10']}<2")
        if (r["oos_edge"] or -1) <= 0: reasons.append("OOS≤0")
        if r["trade_win"] <= bench_win: reasons.append("WR≤bench")
        if r["non_vce_pct"] < 50: reasons.append("VCE-tekrarı")
        print(f"  {name:<22} {verdict}   {', '.join(reasons) if reasons else 'tüm kriterler geçti'}")
        if ok:
            winners.append(name)

    print("\n" + "=" * 108)
    if winners:
        print(f"  → EKLENMEYE DEĞER AİLELER: {winners}")
    else:
        print("  → Hiçbir aday tüm senior kriterleri geçemedi (VCE yalnız kalıyor).")
    print("=" * 108)


if __name__ == "__main__":
    main()
