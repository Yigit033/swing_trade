"""Uniform evaluation of any strategy trade list against PREREGISTRATION.md §1.

Input columns: ticker, signal_date, entry_day, exit_day, entry_price, ret_pct (gross %),
optional key (same-day ordering), censored.

Usage: python lab/evaluate.py
Output: output/lab/report.json  (+ printed table)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.common import LAB, panels, pool_mask  # noqa: E402

COST = 0.50  # % round trip (primary)


def attach_benchmark(T: pd.DataFrame, pool=None, idx: str = "IWM") -> pd.DataFrame:
    """Pool mean of Open[entry] → Close[exit] for pool members on the signal day."""
    P = panels()
    pool = pool_mask() if pool is None else pool
    O, C = P["Open"], P["Close"]
    pool = pool.reindex(columns=C.columns, fill_value=False)
    iwm_o, iwm_c = O[idx], C[idx]
    cache, bm, iwm = {}, [], []
    for r in T.itertuples():
        k = (r.signal_date, r.entry_day, r.exit_day)
        if k not in cache:
            m = pool.loc[r.signal_date].values
            x = (C.loc[r.exit_day].values[m] / O.loc[r.entry_day].values[m] - 1) * 100
            x = x[np.isfinite(x)]
            cache[k] = (float(np.mean(x)) if len(x) else np.nan,
                        float((iwm_c[r.exit_day] / iwm_o[r.entry_day] - 1) * 100))
        bm.append(cache[k][0])
        iwm.append(cache[k][1])
    T = T.copy()
    T["pool_ret"] = bm
    T["iwm_ret"] = iwm          # index return (IWM by default; SPY for Ek D2)
    return T


def boot_ci(T, col, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    g = T.groupby("entry_day")[col].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    k = len(g)
    means = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        means.append(s[i].sum() / c[i].sum())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def portfolio(T, frac=0.10, cost=COST, cap0=10_000.0):
    """Daily mark-to-market, fixed fraction of current equity per trade, cash-limited.

    Ek M (2026-11-22) gün içi sıra düzeltmesi: girişler AÇILIŞTA önceki kapanış değerlemesiyle boyutlanır;
    o gün kapanışta/gün içinde çıkan işlemin nakdi aynı açılıştaki girişe kullandırılmaz. Eski sürüm H6S
    için portföyü ~+12 puan iyimser gösteriyordu (67.1% → 54.7%, L2 freniyle).
    """
    P = panels()
    C = P["Close"]
    T = T.sort_values(["entry_day", "key"], ascending=[True, False]) if "key" in T else T.sort_values("entry_day")
    days = C.index[(C.index >= T.entry_day.min()) & (C.index <= T.exit_day.max())]
    by_entry = {d: g for d, g in T.groupby("entry_day")}
    cash, pos, curve, taken = cap0, [], [], 0
    taken_ret = []
    for d in days:
        eq_open = cash + sum(p["sh"] * p["last"] for p in pos)          # önceki kapanış değerlemesi
        for r in by_entry.get(d, pd.DataFrame()).itertuples():
            alloc = frac * eq_open
            if alloc > cash or r.entry_price <= 0:
                continue
            sh = alloc / (r.entry_price * (1 + cost / 200))
            cash -= alloc
            taken += 1
            taken_ret.append(r.ret_pct - cost)
            xp = r.entry_price * (1 + r.ret_pct / 100)
            pos.append(dict(t=r.ticker, sh=sh, exit_day=r.exit_day, exit_price=xp, last=r.entry_price))
        keep = []                                     # gün içi / kapanış çıkışları — girişlerden SONRA
        for p in pos:
            if p["exit_day"] == d:
                cash += p["sh"] * p["exit_price"] * (1 - cost / 200)
            else:
                keep.append(p)
        pos = keep
        for p in pos:
            v = C.at[d, p["t"]]
            if np.isfinite(v):
                p["last"] = v
        curve.append(cash + sum(p["sh"] * p["last"] for p in pos))
    eq = pd.Series(curve, index=days)
    dd = float((eq / eq.cummax() - 1).min() * 100)
    yrs = (days[-1] - days[0]).days / 365.25
    total = float((eq.iloc[-1] / cap0 - 1) * 100)
    cagr = float(((eq.iloc[-1] / cap0) ** (1 / yrs) - 1) * 100) if yrs > 0 else np.nan
    return dict(total_pct=round(total, 1), cagr_pct=round(cagr, 1), max_dd_pct=round(dd, 1),
                trades_taken=taken, taken_net_mean=round(float(np.mean(taken_ret)), 3) if taken_ret else None,
                final_equity=round(float(eq.iloc[-1]), 0)), eq


def evaluate(name, T, cost=COST, pool=None, idx="IWM", stress=None):
    """cost: primary round trip %; stress: P5 cost (default 2× primary, = 100 bps at 50)."""
    COST = cost
    stress = 2 * cost if stress is None else stress
    T = T[T.ret_pct.notna()].copy()
    T = attach_benchmark(T, pool, idx)
    T["net"] = T.ret_pct - COST
    # Pool is charged the same cost, so net excess == gross excess.
    T["excess"] = T.ret_pct - T.pool_ret
    T["year"] = T.entry_day.dt.year
    n = len(T)
    months = T.entry_day.dt.to_period("M").nunique()
    lo, hi = boot_ci(T, "excess") if n > 10 else (np.nan, np.nan)
    yearly = T.groupby("year").agg(n=("net", "size"), net=("net", "mean"), excess=("excess", "mean"),
                                   win=("net", lambda x: 100 * (x > 0).mean())).round(2)
    trimmed = T.sort_values("ret_pct").iloc[: int(np.floor(n * 0.95))] if n else T
    pos_years = int((yearly.excess > 0).sum())
    pf = T.net[T.net > 0].sum() / max(1e-9, -T.net[T.net < 0].sum())
    port, eq = portfolio(T, cost=COST)
    crit = {
        "P1_n_and_months": bool(n >= 150 and months >= 30),
        "P2_excess_ci_gt0": bool(T.excess.mean() > 0 and lo > 0),
        "P3_years_4of6": bool(pos_years >= 4),
        "P4_trim_top5_net_gt0": bool((trimmed.ret_pct - COST).mean() > 0),
        "P5_stress_net_ge0": bool((T.ret_pct - stress).mean() >= 0),
        "P6_maxdd_lt35": bool(port["max_dd_pct"] > -35),
    }
    out = dict(
        strategy=name, n=n, months=months,
        gross_mean=round(T.ret_pct.mean(), 3), net_mean=round(T.net.mean(), 3),
        net_median=round(T.net.median(), 3), win_rate=round(100 * (T.net > 0).mean(), 1),
        profit_factor=round(float(pf), 3),
        pool_mean=round(T.pool_ret.mean(), 3), excess_mean=round(T.excess.mean(), 3),
        excess_ci95=[round(lo, 3), round(hi, 3)],
        cost_pct=COST, stress_pct=stress, index=idx,
        iwm_mean=round(T.iwm_ret.mean(), 3),
        net_10bps=round((T.ret_pct - 0.1).mean(), 3), net_20bps=round((T.ret_pct - 0.2).mean(), 3),
        net_50bps=round((T.ret_pct - 0.5).mean(), 3), net_100bps=round((T.ret_pct - 1.0).mean(), 3),
        net_trim_top5=round((trimmed.ret_pct - COST).mean(), 3),
        censored=int(T.get("censored", pd.Series(False, index=T.index)).sum()),
        yearly=yearly.reset_index().to_dict("records"),
        portfolio_10pct=port, criteria=crit, promoted=all(crit.values()),
    )
    return out, T, eq


def main():
    res, curves = [], {}
    h0 = LAB / "h0/trades_emulated.parquet"
    sets = []
    if h0.exists():
        T = pd.read_parquet(h0)
        T = T[~T.status.isin(["REJECTED", "REVIEW"])].rename(columns={"rank": "key"})
        T["censored"] = T.status.eq("OPEN")
        T.loc[T.censored, "exit_day"] = panels()["Close"].index[-1]
        sets.append(("H0", T))
    for h in ("H1", "H2", "H3", "H4", "H5"):
        f = LAB / f"challengers/trades_{h}.parquet"
        if f.exists():
            sets.append((h, pd.read_parquet(f)))
    for name, T in sets:
        out, TT, eq = evaluate(name, T)
        res.append(out)
        curves[name] = eq
        TT.to_parquet(LAB / f"eval_{name}.parquet", index=False)
        c = out["criteria"]
        print(f"\n== {name}: n={out['n']} net={out['net_mean']}% med={out['net_median']}% "
              f"win={out['win_rate']}% PF={out['profit_factor']} pool={out['pool_mean']}% "
              f"excess={out['excess_mean']} CI{out['excess_ci95']} | port {out['portfolio_10pct']} "
              f"| PROMOTED={out['promoted']} {c}")
        print(pd.DataFrame(out["yearly"]).to_string(index=False))
    (LAB / "report.json").write_text(json.dumps(res, indent=1, default=str))
    pd.DataFrame(curves).to_parquet(LAB / "equity_curves.parquet")


if __name__ == "__main__":
    main()
