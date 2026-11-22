"""Ek J — H6R "Profesyonel VCE": H6 adayları + canlı piyasa freni + 2.5×ATR felaket stop'u + 8 seans.

Usage: python lab/h6r.py
Output: output/lab/challengers/trades_H6R.parquet, output/lab/h6r_report.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.challengers import OUT, atr14, walk  # noqa: E402
from lab.common import panels  # noqa: E402
from lab.evaluate import evaluate  # noqa: E402
import lab.h0_signals as h0  # noqa: E402

HOLD, STOP_ATR = 8, 2.5


def simulate_h6r(cands, P, regimes, caution_max, bear_block=True):
    O, H, L, C = (P[f] for f in ("Open", "High", "Low", "Close"))
    idx = {d: i for i, d in enumerate(C.index)}
    A = atr14(P)
    pos, rows = [], []                       # pos: (ticker, entry_i, exit_i)
    skipped = dict(bear=0, caution=0, busy=0, no_atr=0, no_entry=0)
    for r in cands.itertuples():
        i = idx[r.date]
        if i + 1 >= len(C.index):
            continue
        regime = regimes.get(r.date, ("UNKNOWN", ""))[0]
        active = [p for p in pos if p[2] > i]                     # açık + bu gün kuyruğa alınanlar
        if regime == "BEAR" and bear_block:
            skipped["bear"] += 1
            continue
        if regime == "CAUTION" and len(active) >= caution_max:
            skipped["caution"] += 1
            continue
        t = r.ticker
        if any(p[0] == t for p in active):
            skipped["busy"] += 1
            continue
        a = A.at[r.date, t]
        arr = (O[t].values, H[t].values, L[t].values, C[t].values)
        o = arr[0][i + 1]
        if not np.isfinite(a) or a <= 0:
            skipped["no_atr"] += 1
            continue
        if not np.isfinite(o) or o <= 0:
            skipped["no_entry"] += 1
            continue
        xi, xp, why = walk(i + 1, arr, o - STOP_ATR * a, HOLD)
        pos.append((t, i + 1, xi))
        rows.append(dict(strategy="H6R", ticker=t, signal_date=r.date, entry_day=C.index[i + 1],
                         exit_day=C.index[xi], entry_price=o, exit_price=xp, ret_pct=100 * (xp / o - 1),
                         reason=why, key=r.key, regime=regime, censored=why == "END_OF_DATA"))
    return pd.DataFrame(rows), skipped


def regime_series(index="SPY"):
    """Canlı regime_from_spy_close, istenen endeksin 1 yıllık kapanışlarıyla (Ek J2: IWM)."""
    if index == "SPY":
        return h0.regime_series()
    from swing_trader.small_cap.regime_logic import regime_from_spy_close
    P = panels()
    px, vix = P["Close"][index], P["Close"]["^VIX"].ffill()
    out = {}
    for i, d in enumerate(px.index):
        if i < 60:
            continue
        hist = px.iloc[max(0, i - 251): i + 1].dropna()
        r = regime_from_spy_close(hist.reset_index(drop=True), float(vix.iloc[i]))
        out[d] = (r.get("regime", "UNKNOWN"), r.get("confidence", "TENTATIVE"))
    return out


def main():
    index = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    name = "H6R" if index == "SPY" else "H6S"
    S = h0.live_settings()
    P = panels()
    cands = pd.read_parquet(OUT / "h6_candidates.parquet").sort_values(["date", "key"], ascending=[True, False])
    regimes = regime_series(index)
    pd.Series({d: v[0] for d, v in regimes.items()}).to_frame("regime").to_parquet(OUT / f"regimes_{index}.parquet")
    T, skipped = simulate_h6r(cands, P, regimes, int(S.backtest_loop.caution_max_concurrent),
                              bool(S.backtest_loop.bear_block_new_entries))
    T["strategy"] = name
    T.to_parquet(OUT / f"trades_{name}.parquet", index=False)
    res = dict(candidates=len(cands), skipped=skipped)
    for cost in (0.50, 0.20):
        out, _, _ = evaluate(name, T, cost=cost)
        res[f"cost_{int(cost * 100)}bps"] = out
    m = res["cost_50bps"]
    print(f"candidates {len(cands)} skipped {skipped} → trades {m['n']} ({m['months']} months)")
    print(f"gross {m['gross_mean']}% net50 {m['net_mean']}% net20 {m['net_20bps']}% win {m['win_rate']}% "
          f"PF {m['profit_factor']} | pool {m['pool_mean']} excess {m['excess_mean']} CI{m['excess_ci95']} "
          f"| IWM/trade {m['iwm_mean']} | trim5 {m['net_trim_top5']}")
    print(f"portfolio@50 {m['portfolio_10pct']} | @20 {res['cost_20bps']['portfolio_10pct']}")
    print("criteria", m["criteria"], "PROMOTED", m["promoted"])
    print(pd.DataFrame(m["yearly"]).to_string(index=False))
    print(T.reason.value_counts().to_dict(), T.groupby("regime").ret_pct.agg(["size", "mean"]).round(2).to_dict())
    (ROOT / f"output/lab/{name.lower()}_report.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
