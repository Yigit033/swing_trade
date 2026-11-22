"""Ek N — 2024–26 açığı: çekirdek IWM (N1 hep açık, N2 BEAR dışında) + IWM al-tut kıyası (N3).

Taban = canlı ürün (H6S + L2 freni), Ek M portföyü (doğru gün içi sıra) + Midas 1.5$/emir.
Çekirdek: boş nakit IWM'de; açılışta bant kuralıyla (1 dilim nakit, 3 dilimi aşınca IWM'e) dengelenir.

Usage: python lab/core_n.py
Output: output/lab/n_core.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.challengers import OUT  # noqa: E402
from lab.common import panels  # noqa: E402
from lab.execution_m import CAP0, SPLIT, STREAK, simulate  # noqa: E402
from lab.execution_m import portfolio as portfolio_m  # noqa: E402
from lab.h6r import regime_series  # noqa: E402
import lab.h0_signals as h0  # noqa: E402

FEE, IWM_BPS, SLOT, BAND = 1.5, 1.0, 0.10, 3


def _stats(curve, days, cap0, **extra):
    eq = pd.Series(curve, index=days)
    yrs = (days[-1] - days[0]).days / 365.25
    return dict(total_pct=round(float((eq.iloc[-1] / cap0 - 1) * 100), 1),
                cagr_pct=round(float(((eq.iloc[-1] / cap0) ** (1 / yrs) - 1) * 100), 1),
                max_dd_pct=round(float((eq / eq.cummax() - 1).min() * 100), 1), **extra)


def core_portfolio(T, mode="N1", regimes=None, cost=0.50, fee=FEE, cap0=CAP0, brake=True):
    """mode: 'N1' çekirdek hep açık | 'N2' çekirdek BEAR'de nakit | 'none' çekirdeksiz (= Ek M tabanı)."""
    P = panels()
    C, O = P["Close"], P["Open"]
    io, ic = O["IWM"], C["IWM"]
    T = T.sort_values(["entry_day", "key"], ascending=[True, False])
    days = C.index[(C.index >= T.entry_day.min()) & (C.index <= T.exit_day.max())]
    by_entry = {d: g for d, g in T.groupby("entry_day")}
    cash, core, pos, curve, closed = cap0, 0.0, [], [], []
    orders = dict(iwm=0, trade=0)
    w_core, w_pos = [], []
    prev = None
    up, dn = 1 + IWM_BPS / 1e4, 1 - IWM_BPS / 1e4

    def close_out(p):
        nonlocal cash
        cash += p["sh"] * p["exit_price"] * (1 - cost / 200) - fee
        orders["trade"] += 1
        closed.append(p["net"])

    for d in days:
        keep = []
        for p in pos:                                          # 1) açılış çıkışları
            (close_out(p) if p["exit_day"] == d and p["at_open"] else keep.append(p))
        pos = keep
        bear = mode == "N2" and prev is not None and regimes.get(prev, ("UNKNOWN", ""))[0] == "BEAR"
        if bear and core > 0:                                  # N2: BEAR → çekirdek nakde
            cash += core * io[d] * dn - fee
            core = 0.0
            orders["iwm"] += 1
        core_prev = core * (ic[prev] if prev is not None else io[d])
        eq_prev = cash + core_prev + sum(p["sh"] * p["last"] for p in pos)
        slot = SLOT * eq_prev
        on = brake and len(closed) >= STREAK and all(x < 0 for x in closed[-STREAK:])
        frac = 0.05 if on else 0.10
        todays = list(by_entry.get(d, pd.DataFrame()).itertuples())
        if mode != "none" and core > 0 and todays:             # 2) girişleri fonla (tek IWM satış emri)
            avail, need = cash + core * io[d] * dn - fee, 0.0
            for _ in todays:
                a = frac * eq_prev
                if need + a <= avail:
                    need += a
            if need > cash:
                sell_val = min(core * io[d], (need - cash + slot) / dn + fee)
                sh = sell_val / io[d]
                cash += sh * io[d] * dn - fee
                core -= sh
                orders["iwm"] += 1
        for t in todays:                                       # 3) açılış girişleri
            alloc = frac * eq_prev
            if alloc > cash:
                continue
            cash -= alloc
            orders["trade"] += 1
            pos.append(dict(t=t.ticker, sh=(alloc - fee) / (t.entry_price * (1 + cost / 200)), exit_day=t.exit_day,
                            exit_price=t.exit_price, at_open=t.at_open, last=t.entry_price,
                            net=t.ret_pct - cost - 200 * fee / alloc))
        if mode != "none" and not bear and cash > BAND * slot:  # 4) fazla nakit → IWM (tek alış emri)
            amt = cash - slot
            core += (amt - fee) / (io[d] * up)
            cash -= amt
            orders["iwm"] += 1
        keep = []
        for p in pos:                                          # 5) gün içi / kapanış çıkışları
            (close_out(p) if p["exit_day"] == d else keep.append(p))
        pos = keep
        for p in pos:
            v = C.at[d, p["t"]]
            if np.isfinite(v):
                p["last"] = v
        pv = sum(p["sh"] * p["last"] for p in pos)
        eq = cash + core * ic[d] + pv
        curve.append(eq)
        w_core.append(core * ic[d] / eq)
        w_pos.append(pv / eq)
        prev = d
    return _stats(curve, days, cap0, iwm_orders=orders["iwm"], trade_orders=orders["trade"],
                  mean_core_pct=round(100 * float(np.mean(w_core)), 1),
                  mean_positions_pct=round(100 * float(np.mean(w_pos)), 1))


def iwm_hold(d0, d1, cap0=CAP0, fee=FEE):
    P = panels()
    days = P["Close"].index[(P["Close"].index >= d0) & (P["Close"].index <= d1)]
    sh = (cap0 - fee) / (P["Open"]["IWM"][days[0]] * (1 + IWM_BPS / 1e4))
    return _stats(list(sh * P["Close"]["IWM"][days].values), days, cap0)


def run(base, regimes, cost):
    halves = [base[base.entry_day < SPLIT], base[base.entry_day >= SPLIT]]
    out = {}
    out["taban"] = dict(**portfolio_m(base, cost=cost, fee=FEE),
                        halves=[portfolio_m(h, cost=cost, fee=FEE) for h in halves])
    for m in ("none", "N1", "N2"):
        out[m] = dict(**core_portfolio(base, m, regimes, cost),
                      halves=[core_portfolio(h, m, regimes, cost) for h in halves])
    out["IWM_al_tut"] = dict(**iwm_hold(base.entry_day.min(), base.exit_day.max()),
                             halves=[iwm_hold(h.entry_day.min(), h.exit_day.max()) for h in halves])
    b = out["taban"]
    for m in ("N1", "N2"):
        x = out[m]
        x["karar"] = dict(
            a_toplam_taban_arti5=bool(x["total_pct"] >= b["total_pct"] + 5),
            b_yarilar_taban_eksi3=bool(all(h["total_pct"] >= bh["total_pct"] - 3
                                           for h, bh in zip(x["halves"], b["halves"]))),
            c_dd_gt_eksi35=bool(x["max_dd_pct"] > -35))
        x["gecer"] = all(x["karar"].values())
        x["IWM_farki"] = [round(x["total_pct"] - out["IWM_al_tut"]["total_pct"], 1)] + [
            round(h["total_pct"] - ih["total_pct"], 1) for h, ih in zip(x["halves"], out["IWM_al_tut"]["halves"])]
    return out


def main():
    S = h0.live_settings()
    P = panels()
    cands = pd.read_parquet(OUT / "h6_candidates.parquet").sort_values(["date", "key"], ascending=[True, False])
    regimes = regime_series("IWM")
    cm, bb = int(S.backtest_loop.caution_max_concurrent), bool(S.backtest_loop.bear_block_new_entries)
    base = simulate(cands, P, regimes, cm, bb)
    sha = hashlib.sha256((ROOT / "lab/PREREGISTRATION.md").read_bytes()).hexdigest()
    res = {"prereg_sha256": sha, "birincil_50bps": run(base, regimes, 0.50), "duyarlilik_20bps": run(base, regimes, 0.20)}
    p, s = res["birincil_50bps"], res["duyarlilik_20bps"]
    for m in ("N1", "N2"):
        p[m]["gecer_duyarlilik_a"] = bool(s[m]["total_pct"] >= s["taban"]["total_pct"] + 5)
        p[m]["NIHAI"] = bool(p[m]["gecer"] and p[m]["gecer_duyarlilik_a"])
    for k, v in res.items():
        if k == "prereg_sha256":
            print(k, v)
            continue
        print(f"\n== {k} ==")
        for m, x in v.items():
            hv = " | ".join(f"{h['total_pct']:+.1f}% dd{h['max_dd_pct']}" for h in x.get("halves", []))
            extra = {kk: x[kk] for kk in ("iwm_orders", "mean_core_pct", "mean_positions_pct", "karar", "IWM_farki",
                                          "NIHAI") if kk in x}
            print(f"{m:10s} {x['total_pct']:+6.1f}% dd {x['max_dd_pct']:6.1f} | yarılar {hv} | {extra}")
    (ROOT / "output/lab/n_core.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str),
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
