# -*- coding: utf-8 -*-
"""
ÜÇÜNCÜ SİNYAL AİLESİ — aday kalıpları BUGÜNKÜ ürünle yeniden ölç
================================================================================
NEDEN YENİDEN: discover_signal_families.py 9 kalıbı ölçmüştü ama
  (a) basitleştirilmiş exit kullanıyordu (20 gün tut, %10 stop),
  (b) motorun kapılarını (dolar-hacim, RSI, OBV, Weinstein, R:R) uygulamıyordu,
  (c) kalite skoru / Q80 eşiği hiç devrede değildi.
Sonuç yanıltıcı olabiliyordu: RVOL thrust o ölçümde en güçlü çıktı (+3.34, t=2.87)
ama GERÇEK motordan geçirilince EV ~0'a düştü (project_backtest_live_replica).

BU ÖLÇÜM tetiği enjekte eder, geri kalan her şeyi CANLI bırakır:
  engine.signals.check_all_triggers sarmalanır → 'triggered' bayrağı aday
  kalıptan gelir; kapılar, skorlama, R:R, exit ve slippage aynen canlı kod.
Böylece "bu kalıp bizim ürünümüzde ne yapar?" sorusu ölçülür — "izole olarak
ne yapar?" değil. İkisi farklı sorular ve ikincisi bizi bir kez yanılttı.

KABUL KRİTERİ (önceden yazıldı — senior bar):
  1. Q80+ EV > 0 (para kazandırıyor)
  2. OOS'ta da EV > 0 (ezber değil)
  3. VCE ile ÖRTÜŞME < %50 (yeni fırsat getiriyor; aynı gün-aynı hisseyi
     tekrar bulmak net sıfır kazançtır — Q7/RVOL dersinden)
  4. Aylık katkı >= 1 sinyal (ölçülebilir fayda)
Dördünü birden geçmeyen kalıp EKLENMEZ.

Cache: output/_broad_data.pkl + _shares_broad.json + _edge_spy.pkl
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
from backtest_live_replica import simulate, EXIT_NEW, build_regime_map
from collect_signal_lab import finviz_hit
import discover_signal_families as dsf

OOS_SPLIT = "2025-06-01"
Q_LIVE = 80.0
SPAN_MONTHS = 21.0
PRICE_MAX = 200.0        # 2026-08-04 ölçümü sonrası canlı tavan

# Aday kalıplar (VCE referans olarak ilk sırada — kıyas tabanı)
CANDIDATES = [
    ("VCE (mevcut, referans)", dsf.p_vce_baseline),
    ("RVOL thrust (2. pathway)", dsf.p_rvol_thrust),
    ("MA50 reclaim", dsf.p_ma50_reclaim),
    ("Pullback bounce", dsf.p_pullback_bounce),
    ("50g yeni zirve", dsf.p_50d_high),
    ("Momentum devamı", dsf.p_momentum_continuation),
    ("Sıkı konsolidasyon kırılımı", dsf.p_tight_consolidation_break),
    ("Aşırı satım dönüşü", dsf.p_oversold_reversal),
    ("Higher-low kırılımı", dsf.p_higher_low_breakout),
    ("Gap-up tutuş", dsf.p_gap_up_hold),
]


def enrich_both(df):
    """backtest_live_replica.enrich + discover_signal_families.enrich birleşimi.
    Kalıplar dsf kolonlarını, finviz_hit ise replica kolonlarını istiyor."""
    d = dsf.enrich(df)                      # ma10/20/50, hi20, hi50, lo20, vol20/50, rvol, atr, atr_pct, chg, chg20, rsi
    c = d["Close"].astype(float)
    h = d["High"].astype(float)
    v = d["Volume"].astype(float)
    d["avgvol50"] = v.rolling(50).mean()
    d["avgvol_liq"] = d["avgvol50"]
    d["hi20_prev"] = h.rolling(20).max().shift()
    return d


def stats(rows):
    if not rows:
        return dict(n=0, ev=0.0, wr=0.0, pf=0.0)
    a = np.array([r["r"] for r in rows])
    w, l = a[a > 0], a[a <= 0]
    pf = (w.sum() / abs(l.sum())) if l.size and l.sum() != 0 else float("inf")
    return dict(n=len(a), ev=float(a.mean()), wr=float((a > 0).mean() * 100), pf=float(pf))


def fmt(s):
    if s["n"] == 0:
        return "n=0"
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return f"n={s['n']:<4} EV {s['ev']:+6.2f}%  WR {s['wr']:3.0f}%  PF {pf:>5}"


def run_pattern(engine, data, shares, rmap, spy, pattern, label):
    """Tetiği enjekte et, GERÇEK motoru koştur, sinyalleri topla."""
    sig_mod = engine.signals
    original = sig_mod.check_all_triggers
    state = {"df": None, "t": None}

    def patched(df):
        # Orijinali çağır → tüm gösterim/skor metrikleri gerçek kalsın
        _, details = original(df)
        t = state["t"]
        full = state["df"]
        fired = False
        try:
            if full is not None and t is not None and t >= 60:
                fired = bool(pattern(full, t))
        except Exception:
            fired = False
        details["triggered"] = fired
        if fired:
            details["trigger_pathway"] = "candidate"
            details["trigger_reason"] = label
        else:
            details.pop("trigger_pathway", None)
        return fired, details

    sig_mod.check_all_triggers = patched
    try:
        out = []
        for tk, df in data.items():
            sh = shares.get(tk, {})
            sh_out, flt = sh.get("shares"), sh.get("float")
            n = len(df)
            state["df"] = df
            for t in range(60, n - 21):
                row = df.iloc[t]
                close = float(row["Close"])
                if close > PRICE_MAX:
                    continue
                mcap = close * sh_out if sh_out else None
                if not finviz_hit(row, mcap):
                    continue
                state["t"] = t
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
                    "key": f"{tk}|{day.date()}", "tk": tk, "date": str(day.date()),
                    "r": float(r), "q": float(sig.get("quality_score", 0) or 0),
                })
        return out
    finally:
        sig_mod.check_all_triggers = original


def main():
    print("Veri yukleniyor...", flush=True)
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    shares = json.load(open("output/_shares_broad.json"))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)
    data = {t: enrich_both(df) for t, df in raw.items()}
    engine = SmallCapEngine()
    print(f"  {len(data)} ticker | maks fiyat ${PRICE_MAX:.0f} | esik Q{Q_LIVE:.0f}", flush=True)

    results = {}
    for label, fn in CANDIDATES:
        print(f"\n>>> {label} ...", flush=True)
        rows = run_pattern(engine, data, shares, rmap, spy, fn, label)
        results[label] = rows
        q80 = [r for r in rows if r["q"] >= Q_LIVE]
        print(f"    ham {len(rows)} | Q80+ {len(q80)}", flush=True)

    json.dump({k: v for k, v in results.items()},
              open("output/third_pathway.json", "w"), default=str)

    vce_keys = {r["key"] for r in results["VCE (mevcut, referans)"] if r["q"] >= Q_LIVE}
    W = 112

    print("\n" + "=" * W)
    print(f"  ÜÇÜNCÜ SİNYAL AİLESİ — gercek motor + gercek exit + Q{Q_LIVE:.0f} esigi")
    print("=" * W)
    print(f"  {'kalip':<30}{'Q80+ sinyal':<36}{'/ay':>6}{'VCE ortusme':>14}{'YENI sinyal':>16}")
    print("  " + "-" * (W - 4))
    rows_out = []
    for label, rows in results.items():
        q80 = [r for r in rows if r["q"] >= Q_LIVE]
        s = stats(q80)
        keys = {r["key"] for r in q80}
        overlap = len(keys & vce_keys) / len(keys) * 100 if keys else 0.0
        new = [r for r in q80 if r["key"] not in vce_keys]
        rows_out.append((label, s, overlap, new))
        print(f"  {label:<30}{fmt(s):<36}{s['n']/SPAN_MONTHS:>6.1f}{overlap:>13.0f}%{len(new):>16}")

    print("\n" + "=" * W)
    print("  YENI SINYALLERIN GETIRISI (VCE'nin GORMEDIGI gunler) + OOS")
    print("=" * W)
    print(f"  {'kalip':<30}{'YENI sinyaller':<36}{'TRAIN':<20}{'OOS'}")
    print("  " + "-" * (W - 4))
    for label, _s, _ov, new in rows_out:
        if label.startswith("VCE"):
            continue
        tr = [r for r in new if r["date"] < OOS_SPLIT]
        te = [r for r in new if r["date"] >= OOS_SPLIT]
        st_, se = stats(tr), stats(te)
        t_s = f"EV {st_['ev']:+.2f}% (n={st_['n']})" if st_["n"] else "—"
        o_s = f"EV {se['ev']:+.2f}% (n={se['n']})" if se["n"] else "—"
        print(f"  {label:<30}{fmt(stats(new)):<36}{t_s:<20}{o_s}")

    print("\n" + "=" * W)
    print("  KARAR — 4 kriter: EV>0 & OOS EV>0 & ortusme<%50 & katki>=1/ay")
    print("=" * W)
    accepted = []
    for label, _s, overlap, new in rows_out:
        if label.startswith("VCE"):
            continue
        s = stats(new)
        te = stats([r for r in new if r["date"] >= OOS_SPLIT])
        rate = s["n"] / SPAN_MONTHS
        fails = []
        if s["n"] == 0:
            fails.append("hic yeni sinyal yok")
        else:
            if s["ev"] <= 0: fails.append(f"EV {s['ev']:+.2f}%<=0")
            if te["n"] < 3: fails.append(f"OOS orneklem yok (n={te['n']})")
            elif te["ev"] <= 0: fails.append(f"OOS {te['ev']:+.2f}%<=0")
            if overlap >= 50: fails.append(f"ortusme %{overlap:.0f}")
            if rate < 1: fails.append(f"katki {rate:.1f}/ay<1")
        if fails:
            print(f"  {label:<30} RED  — {', '.join(fails)}")
        else:
            accepted.append((label, s, rate))
            print(f"  {label:<30} KABUL — EV {s['ev']:+.2f}%, OOS {te['ev']:+.2f}%, "
                  f"ortusme %{overlap:.0f}, +{rate:.1f}/ay")

    print("\n" + "=" * W)
    if accepted:
        best = max(accepted, key=lambda x: x[1]["ev"])
        print(f"  SONUC: {len(accepted)} aday gecti. En iyi: {best[0]} "
              f"(EV {best[1]['ev']:+.2f}%, +{best[2]:.1f} sinyal/ay)")
    else:
        print("  SONUC: hicbir aday 4 kriteri birden gecemedi -> UCUNCU PATHWAY EKLENMEZ.")
        print("         VCE + RVOL thrust ikilisi korunur.")
    print("=" * W)


if __name__ == "__main__":
    main()
