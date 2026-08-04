# -*- coding: utf-8 -*-
"""
KALAN KAPILAR — R:R, zayıf trend, OBV dağıtım (2. tur denetim)
================================================================================
İlk turda (measure_gate_value.py) üç kapı ölçülemedi:

  R:R          → ölçüm GEÇERSİZDİ. Kapıyı `min_rr_at_entry=0` ile kapatmaya
                 çalıştım ama gate rejime göre KODA GÖMÜLÜ değerler kullanıyordu
                 ({"BULL":1.0,"CAUTION":1.5,"BEAR":2.0}), ayar yalnız BİLİNMEYEN
                 rejim dalını etkiliyordu. Değerler artık regime_thresholds'ta
                 (bull/caution/bear_min_rr) → bu kez GERÇEKTEN kapatılabiliyor.
  Zayıf trend  → ölçüme dahil edilmemişti. Kapı `swing_details.trend_quality`
                 içindeki trend_phase/trend_strength'i okuyor; check_boosters
                 çıktısını yamayarak nötrleniyor.
  OBV dağıtım  → ilk turda ΔEV +0.15 ama etki tek sinyalden (n=1). Tekrar bakılıyor.

Yöntem ilk turla aynı: kapıyı devre dışı bırak, gerçek motoru koştur, tabanla
karşılaştır. Karar kuralı da aynı — ama bu kez kullanıcı talimatıyla
güncellendi: "işe yaramıyorsa SİL, zararlı olmasına gerek yok."

  ΔEV ≈ 0 ve hiç ek sinyal yok        → SİL (hiç ateşlenmiyor)
  ΔEV < 0 (TRAIN+OOS aynı yön)        → KAL (işe yarıyor)
  ΔEV > 0 (TRAIN+OOS aynı yön)        → SİL (zararlı)
  Yön tutarsız / örneklem yetersiz    → KAL (silmek için kanıt yok)
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
Q_LIVE = 80.0
PRICE_MAX = 200.0
CACHE = "output/remaining_gates.json"


def _off_rr(s, b):
    """Bu kez GERÇEKTEN kapat: rejim değerleri artık ayarda."""
    rt = s.regime_thresholds
    rt.bull_min_rr = 0.0
    rt.caution_min_rr = 0.0
    rt.bear_min_rr = 0.0
    s.min_rr_at_entry = 0.0
    s.min_rr_type_c = 0.0


def _off_trend(s, b):
    """Kapı trend_phase == 'markdown' arıyor; fazı nötrle."""
    b["_neutralize_trend_phase"] = True


def _off_obv(s, b):
    b["obv_distribution"] = False


GATES = [
    ("TABAN (tüm kapılar açık)", None),
    ("− R:R (rejime göre)", _off_rr),
    ("− Zayıf trend (markdown)", _off_trend),
    ("− OBV dağıtım", _off_obv),
]


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


def run(data, shares, rmap, spy, disabler):
    engine = SmallCapEngine()
    ov = {}
    if disabler is not None:
        disabler(engine.settings, ov)

    if ov:
        sig = engine.signals
        orig = sig.check_boosters

        def patched(df, _o=orig, _ov=dict(ov)):
            b = _o(df)
            if _ov.pop("_neutralize_trend_phase", False) or "_neutralize_trend_phase" in _ov:
                sd = b.get("swing_details") or {}
                tq = dict(sd.get("trend_quality") or {})
                tq["trend_phase"] = "neutral"
                tq["trend_strength"] = 50
                sd = dict(sd)
                sd["trend_quality"] = tq
                b["swing_details"] = sd
            b.update({k: v for k, v in _ov.items() if not k.startswith("_")})
            return b

        sig.check_boosters = patched

    out = []
    for tk, df in data.items():
        sh = shares.get(tk, {})
        sh_out, flt = sh.get("shares"), sh.get("float")
        n = len(df)
        for t in range(60, n - 21):
            row = df.iloc[t]
            close = float(row["Close"])
            if close > PRICE_MAX:
                continue
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
                s = engine.scan_stock(
                    tk, df.iloc[:t + 1], stock_info=info, backtest_mode=True,
                    portfolio_value=10000,
                    spy_df_window=spy_slice if len(spy_slice) >= 6 else None,
                    regime=reg)
            except Exception:
                s = None
            if not s:
                continue
            r = simulate(df, t, EXIT_NEW)
            if r is None:
                continue
            out.append({"key": f"{tk}|{day.date()}", "date": str(day.date()),
                        "r": float(r), "q": float(s.get("quality_score", 0) or 0)})
    return out


def main():
    if os.path.exists(CACHE):
        results = json.load(open(CACHE))
    else:
        print("Veri yukleniyor...", flush=True)
        raw = pickle.load(open("output/_broad_data.pkl", "rb"))
        shares = json.load(open("output/_shares_broad.json"))
        spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
        spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
        rmap = build_regime_map(spy)
        data = {t: enrich(df) for t, df in raw.items()}
        results = {}
        for label, fn in GATES:
            print(f">>> {label} ...", flush=True)
            rows = run(data, shares, rmap, spy, fn)
            results[label] = rows
            print(f"    {len(rows)} sinyal ({len([r for r in rows if r['q']>=Q_LIVE])} Q80+)", flush=True)
        json.dump(results, open(CACHE, "w"), default=str)

    base_rows = [r for r in results["TABAN (tüm kapılar açık)"] if r["q"] >= Q_LIVE]
    base = stats(base_rows)
    base_keys = {r["key"] for r in base_rows}
    b_tr = stats([r for r in base_rows if r["date"] < OOS_SPLIT])
    b_te = stats([r for r in base_rows if r["date"] >= OOS_SPLIT])

    W = 112
    print("\n" + "=" * W)
    print(f"  KALAN KAPILAR — 2. tur | Q{Q_LIVE:.0f}+ | taban {fmt(base)}")
    print("=" * W)
    print(f"  {'kapı':<28}{'kaldırınca':<40}{'ΔEV':>8}{'EK':>5}{'EK EV':>9}   karar")
    print("  " + "-" * (W - 4))
    for label, _ in GATES:
        if label.startswith("TABAN"):
            continue
        rows = [r for r in results[label] if r["q"] >= Q_LIVE]
        s = stats(rows)
        d = s["ev"] - base["ev"]
        extra = [r for r in rows if r["key"] not in base_keys]
        se = stats(extra)
        tr = stats([r for r in rows if r["date"] < OOS_SPLIT])
        te = stats([r for r in rows if r["date"] >= OOS_SPLIT])
        d_tr, d_te = tr["ev"] - b_tr["ev"], te["ev"] - b_te["ev"]

        if se["n"] == 0 and abs(d) < 0.01:
            verdict = "SİL — hiç ateşlenmiyor"
        elif d < -0.3 and d_tr <= 0 and d_te <= 0:
            verdict = "KAL — kaldırınca EV düşüyor"
        elif d > 0.3 and d_tr > 0 and d_te > 0:
            verdict = "SİL — zararlı (TRAIN+OOS aynı yön)"
        elif abs(d) <= 0.3 and se["n"] > 0:
            verdict = f"belirsiz — nötr (+{se['n']} sinyal)"
        else:
            verdict = f"KAL — yön tutarsız (TRAIN {d_tr:+.2f} / OOS {d_te:+.2f})"

        ex_n = f"+{se['n']}" if se["n"] else "0"
        ex_ev = f"{se['ev']:+.2f}%" if se["n"] else "—"
        print(f"  {label:<28}{fmt(s):<40}{d:>+8.2f}{ex_n:>5}{ex_ev:>9}   {verdict}")
        print(f"  {'':<28}{'   TRAIN ' + fmt(tr):<40}{d_tr:>+8.2f}")
        print(f"  {'':<28}{'   OOS   ' + fmt(te):<40}{d_te:>+8.2f}")


if __name__ == "__main__":
    main()
