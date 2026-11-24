"""Ek O — swing tarafı: O1 kazananı koştur, O2 pozisyon %15, O3 CAUTION sınırı yok.

Taban = canlı ürün (H6S + L2 freni), Ek M portföyü (doğru gün içi sıra) + Midas 1.5$/emir.

Usage: python lab/swing_o.py
Output: output/lab/o_swing.json
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.challengers import OUT, atr14, walk  # noqa: E402
from lab.common import panels  # noqa: E402
from lab.execution_m import CAP0, SPLIT, STREAK  # noqa: E402
from lab.execution_m import portfolio as portfolio_m  # noqa: E402
from lab.h6r import HOLD, STOP_ATR, regime_series  # noqa: E402
import lab.h0_signals as h0  # noqa: E402

FEE, RUN_MAX = 1.5, 40


def walk_runner(ie, arr, a, stop0):
    """O1: 8. kapanış girişin üstündeyse iz süren stop (en yüksek kapanış − 2.5×ATR), en çok 40 seans."""
    O, H, L, C = arr
    xi, xp, why = walk(ie, arr, stop0, HOLD)
    if why != "TIME" or not C[xi] > O[ie]:
        return xi, xp, why
    n = len(C)
    hi = float(np.nanmax(C[ie: xi + 1]))
    stop = max(stop0, hi - STOP_ATR * a)
    for k in range(xi + 1, min(n, ie + RUN_MAX)):
        if not np.isfinite(C[k]):
            continue
        if L[k] <= stop:
            return k, min(O[k], stop), "TRAIL"
        hi = max(hi, C[k])
        stop = max(stop, hi - STOP_ATR * a)
    last = min(n, ie + RUN_MAX) - 1
    while last > ie and not np.isfinite(C[last]):
        last -= 1
    return last, C[last], ("TIME_MAX" if ie + RUN_MAX <= n else "END_OF_DATA")


def simulate(cands, P, regimes, caution_max, bear_block=True, runner=False):
    """execution_m.simulate ile aynı; runner=True → O1 çıkışı."""
    O, H, L, C = (P[f] for f in ("Open", "High", "Low", "Close"))
    idx = {d: i for i, d in enumerate(C.index)}
    A = atr14(P)
    pos, rows = [], []
    for r in cands.itertuples():
        i = idx[r.date]
        if i + 1 >= len(C.index):
            continue
        regime = regimes.get(r.date, ("UNKNOWN", ""))[0]
        active = [p for p in pos if p[2] > i]
        if regime == "BEAR" and bear_block:
            continue
        if regime == "CAUTION" and len(active) >= caution_max:
            continue
        t = r.ticker
        if any(p[0] == t for p in active):
            continue
        a = A.at[r.date, t]
        arr = (O[t].values, H[t].values, L[t].values, C[t].values)
        o = arr[0][i + 1]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(o) or o <= 0:
            continue
        stop = o - STOP_ATR * a
        xi, xp, why = walk_runner(i + 1, arr, a, stop) if runner else walk(i + 1, arr, stop, HOLD)
        at_open = bool(np.isfinite(arr[0][xi]) and xp == arr[0][xi] and xi > i + 1)
        pos.append((t, i + 1, xi))
        rows.append(dict(ticker=t, signal_date=r.date, entry_day=C.index[i + 1], exit_day=C.index[xi],
                         entry_price=o, exit_price=xp, ret_pct=100 * (xp / o - 1), reason=why, key=r.key,
                         regime=regime, at_open=at_open, bars=xi - i))
    return pd.DataFrame(rows)


def portfolio(T, frac=0.10, cost=0.50, fee=FEE, cap0=CAP0, brake=True, return_curve=False):
    """execution_m.portfolio (sabit boyut) + boyut parametresi + maruziyet. frac=0.10 → birebir taban.
    return_curve=True → (özet, günlük özsermaye serisi) döner (Ek R yıllık tablo); hesap aynı."""
    C = panels()["Close"]
    T = T.sort_values(["entry_day", "key"], ascending=[True, False])
    days = C.index[(C.index >= T.entry_day.min()) & (C.index <= T.exit_day.max())]
    by_entry = {d: g for d, g in T.groupby("entry_day")}
    cash, pos, curve, closed, idle, taken = cap0, [], [], [], [], []

    def close_out(p):
        nonlocal cash
        cash += p["sh"] * p["exit_price"] * (1 - cost / 200) - fee
        closed.append(p["net"])

    for d in days:
        keep = []
        for p in pos:
            (close_out(p) if p["exit_day"] == d and p["at_open"] else keep.append(p))
        pos = keep
        eq_prev = cash + sum(p["sh"] * p["last"] for p in pos)
        for t in by_entry.get(d, pd.DataFrame()).itertuples():
            on = brake and len(closed) >= STREAK and all(x < 0 for x in closed[-STREAK:])
            alloc = (frac / 2 if on else frac) * eq_prev
            if alloc > cash:
                continue
            cash -= alloc
            taken.append(t.Index)
            pos.append(dict(t=t.ticker, sh=(alloc - fee) / (t.entry_price * (1 + cost / 200)), exit_day=t.exit_day,
                            exit_price=t.exit_price, at_open=t.at_open, last=t.entry_price,
                            net=t.ret_pct - cost - 200 * fee / alloc))
        keep = []
        for p in pos:
            (close_out(p) if p["exit_day"] == d else keep.append(p))
        pos = keep
        for p in pos:
            v = C.at[d, p["t"]]
            if np.isfinite(v):
                p["last"] = v
        eq = cash + sum(p["sh"] * p["last"] for p in pos)
        curve.append(eq)
        idle.append(cash / eq)
    eq = pd.Series(curve, index=days)
    yrs = (days[-1] - days[0]).days / 365.25
    Tk = T.loc[taken]
    out = dict(total_pct=round(float((eq.iloc[-1] / cap0 - 1) * 100), 1),
               cagr_pct=round(float(((eq.iloc[-1] / cap0) ** (1 / yrs) - 1) * 100), 1),
               max_dd_pct=round(float((eq / eq.cummax() - 1).min() * 100), 1),
               taken=len(taken), idle_cash_pct=round(100 * float(np.mean(idle)), 1),
               net_per_trade=round(float((Tk.ret_pct - cost).mean()), 3),
               mean_bars=round(float(Tk.bars.mean()), 1) if "bars" in Tk else None)
    return (out, eq) if return_curve else out


def evaluate(T, frac, cost):
    return dict(**portfolio(T, frac, cost), halves=[portfolio(T[T.entry_day < SPLIT], frac, cost),
                                                    portfolio(T[T.entry_day >= SPLIT], frac, cost)])


def decide(x, b, s_x, s_b):
    k = dict(a_toplam=bool(x["total_pct"] >= b["total_pct"] + 5),
             b_yarilar=bool(all(h["total_pct"] >= bh["total_pct"] - 2 for h, bh in zip(x["halves"], b["halves"]))),
             c_dusus=bool(x["max_dd_pct"] >= b["max_dd_pct"] - 3 and x["max_dd_pct"] > -35),
             d_20bps=bool(s_x["total_pct"] >= s_b["total_pct"] + 5))
    return k, all(k.values())


def main():
    S = h0.live_settings()
    P = panels()
    cands = pd.read_parquet(OUT / "h6_candidates.parquet").sort_values(["date", "key"], ascending=[True, False])
    regimes = regime_series("IWM")
    cm, bb = int(S.backtest_loop.caution_max_concurrent), bool(S.backtest_loop.bear_block_new_entries)
    trades = {"base": simulate(cands, P, regimes, cm, bb),
              "O1": simulate(cands, P, regimes, cm, bb, runner=True),
              "O3": simulate(cands, P, regimes, 10 ** 6, bb)}
    trades["O1+O3"] = simulate(cands, P, regimes, 10 ** 6, bb, runner=True)
    spec = {"base": ("base", 0.10), "O1": ("O1", 0.10), "O2": ("base", 0.15), "O3": ("O3", 0.10)}

    parity = portfolio_m(trades["base"].drop(columns=["bars"]), cost=0.50, fee=FEE)["total_pct"]
    res = {"prereg_sha256": hashlib.sha256((ROOT / "lab/PREREGISTRATION.md").read_bytes()).hexdigest(),
           "parite_ek_m": [parity, portfolio(trades["base"], 0.10, 0.50)["total_pct"]]}
    out = {c: {k: evaluate(trades[t], f, c) for k, (t, f) in spec.items()} for c in (0.50, 0.20)}
    p, s = out[0.50], out[0.20]
    passed = []
    for k in ("O1", "O2", "O3"):
        p[k]["karar"], p[k]["gecer"] = decide(p[k], p["base"], s[k], s["base"])
        if p[k]["gecer"]:
            passed.append(k)
    res["birincil_50bps"], res["duyarlilik_20bps"] = p, s
    res["gecenler"] = passed
    if len(passed) >= 2:                                         # birleşim (ön kayıtlı)
        best = max(passed, key=lambda k: p[k]["total_pct"])
        combos = {}
        for r in range(2, len(passed) + 1):
            for combo in itertools.combinations(passed, r):
                key = "+".join(combo)
                tk = "O1+O3" if {"O1", "O3"} <= set(combo) else ("O1" if "O1" in combo else
                                                                 ("O3" if "O3" in combo else "base"))
                f = 0.15 if "O2" in combo else 0.10
                cp, cs = evaluate(trades[tk], f, 0.50), evaluate(trades[tk], f, 0.20)
                cp["karar"], cp["gecer"] = decide(cp, p["base"], cs, s["base"])
                cp["en_iyi_tekten_iyi"] = bool(cp["total_pct"] >= p[best]["total_pct"])
                combos[key] = dict(birincil=cp, duyarlilik=cs)
        res["birlesim"] = combos
    for k, v in res.items():
        if k in ("birincil_50bps", "duyarlilik_20bps"):
            print(f"\n== {k} ==")
            for m, x in v.items():
                hv = " | ".join(f"{h['total_pct']:+.1f}% dd{h['max_dd_pct']}" for h in x["halves"])
                print(f"{m:5s} {x['total_pct']:+6.1f}% dd {x['max_dd_pct']:6.1f} | yarılar {hv} | işlem {x['taken']} "
                      f"net/işlem {x['net_per_trade']:+.3f} | nakit %{x['idle_cash_pct']} | seans {x['mean_bars']} "
                      f"{x.get('karar', '')}")
        elif k == "birlesim":
            for kk, vv in v.items():
                x = vv["birincil"]
                print(f"\nBİRLEŞİM {kk}: {x['total_pct']:+.1f}% dd {x['max_dd_pct']} yarılar "
                      f"{[h['total_pct'] for h in x['halves']]} 20bps {vv['duyarlilik']['total_pct']:+.1f}% "
                      f"{x['karar']} en_iyi_tekten_iyi={x['en_iyi_tekten_iyi']}")
        else:
            print(k, v)
    exits = trades["O1"].reason.value_counts().to_dict()
    res["O1_cikislar"] = exits
    print("O1 çıkışlar", exits)
    (ROOT / "output/lab/o_swing.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str),
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
