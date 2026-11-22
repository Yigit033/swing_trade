"""Ek L — H6S risk kuralları: L1 sektör sınırı (2), L2 kayıp serisi freni (3 zarar → yarım boyut).

Usage: python lab/risk_rules.py
Output: output/lab/l_risk_rules.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.challengers import OUT  # noqa: E402
from lab.common import info_frame, panels  # noqa: E402
import lab.h0_signals as h0  # noqa: E402
from lab.h6r import regime_series, simulate_h6r  # noqa: E402

COST = 0.50


def portfolio(T, frac_full=0.10, frac_half=0.05, brake=False, streak=3, cost=COST, cap0=10_000.0):
    """lab/evaluate.portfolio ile aynı mantık; L2 açıkken boyut hesabın KENDİ kapanışlarına göre.

    UYARI (Ek M): bu sürüm o gün kapanışta çıkan işlemin nakdini/sonucunu aynı günün açılış girişine
    kullandırıyor ve boyutu giriş günü kapanışıyla hesaplıyor (iyimser). Ek L kararını düzeltilmiş sırayla
    yeniden doğrulayan sürüm: lab/execution_m.portfolio (fren yine geçti: 50.0% → 54.7%, DD eşit).
    """
    C = panels()["Close"]
    T = T.sort_values(["entry_day", "key"], ascending=[True, False])
    days = C.index[(C.index >= T.entry_day.min()) & (C.index <= T.exit_day.max())]
    by_entry = {d: g for d, g in T.groupby("entry_day")}
    cash, pos, curve, closed_net, halved, taken = cap0, [], [], [], 0, 0
    for d in days:
        keep = []
        for p in pos:
            if p["exit_day"] == d:
                cash += p["sh"] * p["exit_price"] * (1 - cost / 200)
                closed_net.append(p["net"])
            else:
                keep.append(p)
        pos = keep
        eq_open = cash + sum(p["sh"] * (C.at[d, p["t"]] if np.isfinite(C.at[d, p["t"]]) else p["last"]) for p in pos)
        for r in by_entry.get(d, pd.DataFrame()).itertuples():
            frac = frac_full
            if brake and len(closed_net) >= streak and all(x < 0 for x in closed_net[-streak:]):
                frac = frac_half
                halved += 1
            alloc = frac * eq_open
            if alloc > cash:
                continue
            sh = alloc / (r.entry_price * (1 + cost / 200))
            cash -= alloc
            taken += 1
            xp = r.entry_price * (1 + r.ret_pct / 100)
            net = r.ret_pct - cost
            if r.exit_day == d:
                cash += sh * xp * (1 - cost / 200)
                closed_net.append(net)
            else:
                pos.append(dict(t=r.ticker, sh=sh, exit_day=r.exit_day, exit_price=xp, last=r.entry_price, net=net))
        for p in pos:
            v = C.at[d, p["t"]]
            if np.isfinite(v):
                p["last"] = v
        curve.append(cash + sum(p["sh"] * p["last"] for p in pos))
    eq = pd.Series(curve, index=days)
    yrs = (days[-1] - days[0]).days / 365.25
    return dict(total_pct=round(float((eq.iloc[-1] / cap0 - 1) * 100), 1),
                cagr_pct=round(float(((eq.iloc[-1] / cap0) ** (1 / yrs) - 1) * 100), 1),
                max_dd_pct=round(float((eq / eq.cummax() - 1).min() * 100), 1),
                taken=taken, halved=halved)


def main():
    S = h0.live_settings()
    P = panels()
    cands = pd.read_parquet(OUT / "h6_candidates.parquet").sort_values(["date", "key"], ascending=[True, False])
    regimes = regime_series("IWM")
    inf = info_frame()
    sector_of = {t: inf.at[t, "sector"] for t in cands.ticker.unique() if t in inf.index and isinstance(inf.at[t, "sector"], str)}
    cm, bb = int(S.backtest_loop.caution_max_concurrent), bool(S.backtest_loop.bear_block_new_entries)
    base, _ = simulate_h6r(cands, P, regimes, cm, bb)
    l1, sk = simulate_h6r(cands, P, regimes, cm, bb, sector_of=sector_of, sector_cap=2)
    res = {
        "H6S (taban)": dict(n=len(base), net=round(float((base.ret_pct - COST).mean()), 3), **portfolio(base)),
        "L1 sektör ≤2": dict(n=len(l1), net=round(float((l1.ret_pct - COST).mean()), 3), sector_skips=sk.get("sector", 0), **portfolio(l1)),
        "L2 kayıp serisi freni": dict(n=len(base), net=round(float((base.ret_pct - COST).mean()), 3), **portfolio(base, brake=True)),
        "L1 + L2": dict(n=len(l1), net=round(float((l1.ret_pct - COST).mean()), 3), **portfolio(l1, brake=True)),
    }
    b = res["H6S (taban)"]
    for k, v in res.items():
        if k == "H6S (taban)":
            continue
        v["adopt"] = bool(v["total_pct"] >= b["total_pct"] and v["max_dd_pct"] >= b["max_dd_pct"])
    for k, v in res.items():
        print(f"{k:24s} {v}")
    (ROOT / "output/lab/l_risk_rules.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
