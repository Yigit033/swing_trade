"""Pre-registered challengers H1–H3 (lab/PREREGISTRATION.md §2). Parameters are FIXED.

Every rule uses only bars ≤ t for the decision; entry is Open_{t+1}.
Exit convention (all): bars counted from the entry bar (= 1); stop-first on every bar,
gap through stop fills at Open; time exits at that bar's Close.

Usage: python lab/challengers.py
Output: output/lab/challengers/trades_H{1,2,3}.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.common import LAB, panels, pool_mask, EVAL_START  # noqa: E402

OUT = LAB / "challengers"


def rsi_wilder(c: pd.DataFrame, n: int) -> pd.DataFrame:
    d = c.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    r = 100 - 100 / (1 + rs)
    return r.where(ad > 0, 100.0)


def atr14(P):
    h, l, c = P["High"], P["Low"], P["Close"]
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    return tr.rolling(14).mean()


def simulate(cands: pd.DataFrame, P, exit_fn, name):
    """cands: rows (date, ticker, key, stop_spec...) sorted by date then key desc.
    exit_fn(i_entry, arrays, row) -> (exit_index, exit_price, reason)."""
    O, H, L, C = (P[f] for f in ("Open", "High", "Low", "Close"))
    idx = {d: i for i, d in enumerate(C.index)}
    busy_until = {}
    rows = []
    for r in cands.itertuples():
        i = idx[r.date]
        if i + 1 >= len(C.index):
            continue
        t = r.ticker
        if busy_until.get(t, -1) >= i + 1:
            continue                       # one position per ticker
        arr = (O[t].values, H[t].values, L[t].values, C[t].values)
        o = arr[0][i + 1]
        if not np.isfinite(o) or o <= 0:
            continue
        res = exit_fn(i + 1, arr, r)
        if res is None:
            continue
        xi, xp, why = res
        busy_until[t] = xi
        rows.append(dict(strategy=name, ticker=t, signal_date=r.date, entry_day=C.index[i + 1],
                         exit_day=C.index[xi], entry_price=o, exit_price=xp,
                         ret_pct=100 * (xp / o - 1), reason=why, key=r.key,
                         censored=why == "END_OF_DATA"))
    return pd.DataFrame(rows)


def walk(ie, arr, stop, max_bars, close_exit=None):
    O, H, L, C = arr
    n = len(C)
    for k in range(ie, min(n, ie + max_bars)):
        if not np.isfinite(C[k]):
            continue                         # missing bar: hold (no fill possible)
        if stop is not None and L[k] <= stop:
            return k, min(O[k], stop), "STOP"
        if close_exit is not None and close_exit(k):
            return k, C[k], "RULE"
    last = min(n, ie + max_bars) - 1
    while last > ie and not np.isfinite(C[last]):
        last -= 1
    why = "TIME" if ie + max_bars <= n else "END_OF_DATA"
    return last, C[last], why


def h1(P, pool):
    O, H, L, C, V = (P[f] for f in ("Open", "High", "Low", "Close", "Volume"))
    gap = O / C.shift(1) - 1
    v50 = V.shift(1).rolling(50, min_periods=40).mean()
    rng = (H - L).replace(0, np.nan)
    sig = (gap >= 0.05) & (V >= 3 * v50) & (C >= O) & ((C - L) / rng >= 0.5) & pool
    sig = sig.loc[EVAL_START:]
    st = sig.stack()
    cands = st[st].reset_index()
    cands.columns = ["date", "ticker", "_"]
    cands["key"] = [float((V / v50).at[d, t]) for d, t in zip(cands.date, cands.ticker)]
    cands["low_t"] = [float(L.at[d, t]) for d, t in zip(cands.date, cands.ticker)]
    cands = cands.sort_values(["date", "key"], ascending=[True, False])

    def ex(ie, arr, r):
        if arr[0][ie] < r.low_t:
            return None                      # pre-registered: no entry below gap-day low
        return walk(ie, arr, r.low_t, 20)
    return simulate(cands, P, ex, "H1")


def h2(P, pool):
    C = P["Close"]
    sma200, sma50, sma5 = C.rolling(200).mean(), C.rolling(50).mean(), C.rolling(5).mean()
    r2 = rsi_wilder(C, 2)
    sig = (C > sma200) & (r2 < 5) & (C > sma50 * 0.95) & pool
    sig = sig.loc[EVAL_START:]
    st = sig.stack()
    cands = st[st].reset_index()
    cands.columns = ["date", "ticker", "_"]
    cands["key"] = [-float(r2.at[d, t]) for d, t in zip(cands.date, cands.ticker)]
    cands = cands.sort_values(["date", "key"], ascending=[True, False])
    s5 = sma5

    def ex(ie, arr, r):
        o = arr[0][ie]
        col = s5[r.ticker].values
        return walk(ie, arr, o * 0.85, 10, close_exit=lambda k: arr[3][k] > col[k])
    return simulate(cands, P, ex, "H2")


def h3(P, pool):
    C, H = P["Close"], P["High"]
    hi252 = H.rolling(252, min_periods=200).max()
    sma50 = C.rolling(50).mean()
    r126 = C / C.shift(126) - 1
    a = atr14(P)
    rk = r126.where(pool).rank(axis=1, pct=True)
    cal = C.index
    week_end = pd.Series(cal, index=cal).groupby(cal.to_period("W")).transform("max") == cal
    sig = (C >= 0.95 * hi252) & (C > sma50) & (rk >= 0.8) & pool
    rows = []
    for d in cal[(cal >= EVAL_START) & week_end.values]:
        s = sig.loc[d]
        tick = s.index[s.values]
        if len(tick) == 0:
            continue
        top = r126.loc[d, tick].sort_values(ascending=False).head(10)
        for t, v in top.items():
            rows.append(dict(date=d, ticker=t, key=float(v), atr=float(a.at[d, t])))
    cands = pd.DataFrame(rows).sort_values(["date", "key"], ascending=[True, False])

    def ex(ie, arr, r):
        o = arr[0][ie]
        return walk(ie, arr, o - 2.5 * r.atr, 20)
    return simulate(cands, P, ex, "H3")


def main():
    P, pool = panels(), pool_mask()
    pool = pool.reindex(columns=P["Close"].columns, fill_value=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("H1", h1), ("H2", h2), ("H3", h3)):
        T = fn(P, pool)
        T.to_parquet(OUT / f"trades_{name}.parquet", index=False)
        print(name, len(T), f"mean {T.ret_pct.mean():.2f}% median {T.ret_pct.median():.2f}%", flush=True)


if __name__ == "__main__":
    main()
