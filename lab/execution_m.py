"""Ek M — Kâğıttan Midas'a: M1 açılışta çıkış, M2 eşit riskli boyut, M3 gece boşluğu, boş nakit oranı.

Taban = canlı ürün (H6S + L2 freni). Portföy günü gerçek sırayla işler (Ek M düzeltmesi):
açılış çıkışları → açılış girişleri (boyut önceki kapanış değerlemesiyle) → gün içi/kapanış çıkışları.

Usage: python lab/execution_m.py
Output: output/lab/m_execution.json
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
import lab.h0_signals as h0  # noqa: E402
from lab.h6r import HOLD, STOP_ATR, regime_series  # noqa: E402
from lab.risk_rules import portfolio as portfolio_ek_l  # noqa: E402

COST, CAP0, STREAK, SPLIT = 0.50, 10_000.0, 3, pd.Timestamp("2024-01-01")


def simulate(cands, P, regimes, caution_max, bear_block=True, exit_open=False):
    """lab/h6r.simulate_h6r ile aynı; exit_open=True → süre çıkışı 9. seans AÇILIŞINDA (M1)."""
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
        xi, xp, why = walk(i + 1, arr, stop, HOLD)
        if exit_open and why == "TIME":
            j = i + 1 + HOLD                                  # 9. seans
            while j < len(C.index) and not np.isfinite(arr[0][j]):
                j += 1
            if j < len(C.index):
                xi, xp, why = j, arr[0][j], "TIME_OPEN"       # açılış stopun altındaysa da aynı fiyat
            else:
                why = "END_OF_DATA"
        at_open = bool(np.isfinite(arr[0][xi]) and xp == arr[0][xi] and xi > i + 1)
        pos.append((t, i + 1, xi))
        rows.append(dict(ticker=t, signal_date=r.date, entry_day=C.index[i + 1], exit_day=C.index[xi],
                         entry_price=o, exit_price=xp, ret_pct=100 * (xp / o - 1), reason=why, key=r.key,
                         regime=regime, at_open=at_open, atr_pct=float(a / C.at[r.date, t]),
                         signal_close=float(C.at[r.date, t])))
    return pd.DataFrame(rows)


def solve_r(atr_pct, target=0.10, cap=0.20):
    lo, hi = 1e-5, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.minimum(cap, mid / (STOP_ATR * atr_pct)).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def portfolio(T, sizing="fixed", r=None, cap=0.20, brake=True, cost=COST, cap0=CAP0, fee=0.0):
    """Gün içi gerçek sıra. sizing: 'fixed' (%10) | 'risk' (r / (2.5×ATR%), en çok cap).
    fee: kenar başına sabit $ ücret (Midas ABD: 1.5$); cost: yüzde gidiş-dönüş (spread+kayma)."""
    C = panels()["Close"]
    T = T.sort_values(["entry_day", "key"], ascending=[True, False])
    days = C.index[(C.index >= T.entry_day.min()) & (C.index <= T.exit_day.max())]
    by_entry = {d: g for d, g in T.groupby("entry_day")}
    cash, pos, curve, closed_net, idle = cap0, [], [], [], []
    taken = halved = 0
    notional = []

    def close_out(p):
        nonlocal cash
        cash += p["sh"] * p["exit_price"] * (1 - cost / 200) - fee
        closed_net.append(p["net"])

    for d in days:
        keep = []
        for p in pos:                                   # 1) açılışta gerçekleşen çıkışlar
            (close_out(p) if p["exit_day"] == d and p["at_open"] else keep.append(p))
        pos = keep
        eq_prev = cash + sum(p["sh"] * p["last"] for p in pos)   # 2) önceki kapanış değerlemesi
        for t in by_entry.get(d, pd.DataFrame()).itertuples():
            on = brake and len(closed_net) >= STREAK and all(x < 0 for x in closed_net[-STREAK:])
            if sizing == "fixed":
                frac = 0.05 if on else 0.10
            else:
                frac = min(cap, (r / 2 if on else r) / (STOP_ATR * t.atr_pct))
            alloc = frac * eq_prev
            if alloc > cash:
                continue
            halved += int(on)
            taken += 1
            notional.append(frac)
            cash -= alloc
            pos.append(dict(t=t.ticker, sh=(alloc - fee) / (t.entry_price * (1 + cost / 200)), exit_day=t.exit_day,
                            exit_price=t.exit_price, at_open=t.at_open, last=t.entry_price,
                            net=t.ret_pct - cost - 200 * fee / alloc))
        keep = []
        for p in pos:                                   # 3) gün içi / kapanış çıkışları
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
    return dict(total_pct=round(float((eq.iloc[-1] / cap0 - 1) * 100), 1),
                cagr_pct=round(float(((eq.iloc[-1] / cap0) ** (1 / yrs) - 1) * 100), 1),
                max_dd_pct=round(float((eq / eq.cummax() - 1).min() * 100), 1),
                taken=taken, halved=halved, mean_notional_pct=round(100 * float(np.mean(notional)), 1),
                idle_cash_pct=round(100 * float(np.mean(idle)), 1))


def halves(T, **kw):
    return [portfolio(T[T.entry_day < SPLIT], **kw), portfolio(T[T.entry_day >= SPLIT], **kw)]


def net_mean(T):
    return round(float((T.ret_pct - COST).mean()), 3)


def main():
    S = h0.live_settings()
    P = panels()
    cands = pd.read_parquet(OUT / "h6_candidates.parquet").sort_values(["date", "key"], ascending=[True, False])
    regimes = regime_series("IWM")
    cm, bb = int(S.backtest_loop.caution_max_concurrent), bool(S.backtest_loop.bear_block_new_entries)
    base = simulate(cands, P, regimes, cm, bb)
    m1 = simulate(cands, P, regimes, cm, bb, exit_open=True)
    res = {"prereg_sha256": "f359c1e4beab3bc6938c687ec465332c1e3d59c86a393073b27e4ba1f1ca1c36"}

    old = portfolio_ek_l(base.drop(columns=["at_open", "atr_pct", "signal_close"]), brake=True)
    b = dict(n=len(base), net=net_mean(base), **portfolio(base), halves=halves(base))
    res["taban_ek_l_eski_sira"] = old
    res["taban"] = b

    # M1 — 9. seans açılışında çıkış
    a = dict(n=len(m1), net=net_mean(m1), **portfolio(m1), halves=halves(m1),
             exits=m1.reason.value_counts().to_dict())
    a["adopt"] = bool(a["total_pct"] >= b["total_pct"] - 2 and a["max_dd_pct"] >= b["max_dd_pct"] - 1
                      and a["net"] >= b["net"] - 0.05)
    res["M1_acilista_cikis"] = a

    # M2 — eşit riskli boyut (r yalnız ATR'den)
    r = solve_r(base.atr_pct.values)
    m2 = dict(r_pct=round(100 * r, 3), n=len(base), net=net_mean(base),
              **portfolio(base, sizing="risk", r=r), halves=halves(base, sizing="risk", r=r))
    m2["adopt"] = bool(m2["total_pct"] >= b["total_pct"] and m2["max_dd_pct"] >= b["max_dd_pct"]
                       and all(h["total_pct"] >= bh["total_pct"] for h, bh in zip(m2["halves"], b["halves"])))
    res["M2_esit_risk"] = m2
    if a["adopt"] and m2["adopt"]:
        r1 = solve_r(m1.atr_pct.values)
        res["M1+M2"] = dict(**portfolio(m1, sizing="risk", r=r1))

    # M3 — gece boşluğu: sinyal kapanışında girseydik (aynı çıkış)
    gap = 100 * (base.entry_price / base.signal_close - 1)
    impr = 100 * (base.exit_price / base.signal_close - 1) - base.ret_pct
    early = base.entry_day < SPLIT
    m3 = dict(gap_mean=round(float(gap.mean()), 3), gap_median=round(float(gap.median()), 3),
              gap_pos_share=round(float((gap > 0).mean() * 100), 1),
              improvement_mean=round(float(impr.mean()), 3),
              improvement_halves=[round(float(impr[early].mean()), 3), round(float(impr[~early].mean()), 3)])
    m3["worth_building"] = bool(m3["improvement_mean"] >= 0.30 and all(x > 0 for x in m3["improvement_halves"]))
    res["M3_gece_boslugu"] = m3

    # Boş nakit oranı (bilgi). M4'ün faiz kısmı kullanıcı ilkesiyle kapatıldı (faiz yok) — silindi.
    res["bos_nakit_orani"] = dict(idle_cash_pct=b["idle_cash_pct"])

    # Ek M ek kontrol (gerçeklik): Midas sabit ücreti (1.5$/kenar) ile L2 freni hâlâ geçiyor mu?
    fees = {}
    for cap0 in (10_000.0, 25_000.0):
        for spread in (0.20, 0.50):
            k = f"{int(cap0)}$_spread{int(spread * 100)}bps"
            off = portfolio(base, brake=False, cost=spread, fee=1.5, cap0=cap0)
            on = portfolio(base, brake=True, cost=spread, fee=1.5, cap0=cap0)
            fees[k] = dict(fren_yok=off, fren_var=on, fren_gecer=bool(on["total_pct"] >= off["total_pct"]
                                                                      and on["max_dd_pct"] >= off["max_dd_pct"]))
    res["midas_ucreti_fren"] = fees

    for k, v in res.items():
        print(f"{k}: {v}")
    (ROOT / "output/lab/m_execution.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str),
                                                      encoding="utf-8")


if __name__ == "__main__":
    main()
