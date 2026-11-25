"""Gölge ileri test raporu + ön kayıtlı terfi kapısı (PREREGISTRATION Ek C, H6: Ek F, MOM: Ek U). SAF fonksiyonlar."""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .rules import PARAMS, PARAMS_H6, PARAMS_H6S, PARAMS_H6SR, PARAMS_MOM

COST = PARAMS["common"]["cost_pct"]
GATE = {
    "H5": dict(min_closed=150, min_months=6, max_months=12),
    "H4": dict(min_closed=50, min_months=6, max_months=12),
    "H6": dict(min_closed=PARAMS_H6["min_closed"], min_months=6, max_months=12),
    "H6S": dict(min_closed=PARAMS_H6S["min_closed"], min_months=6, max_months=12),
    "H6SR": dict(min_closed=PARAMS_H6SR["min_closed"], min_months=6, max_months=12),
    "MOM": dict(min_closed=PARAMS_MOM["min_closed"], min_months=PARAMS_MOM["min_months"],
                max_months=PARAMS_MOM["max_months"]),
}
# Stratejiye özel değerlendirme (Ek U): MOM büyük ve likit hisselerde → 10 bps + 1.5$/emir, kıyas SPY.
# Diğerleri ön kayıtlarındaki gibi: 50 bps, sabit ücret yok, kıyas IWM.
COST_BY = {"MOM": PARAMS_MOM["cost_pct"]}
FEE_BY = {"MOM": PARAMS_MOM["fee_per_order"]}
BENCHMARK_BY = {"MOM": PARAMS_MOM["regime_index"]}


def boot_ci(df: pd.DataFrame, col: str, n: int = 2000, seed: int = 7):
    """Giriş-günü kümeli bootstrap (lab/evaluate.boot_ci ile aynı)."""
    if len(df) < 2:
        return None, None
    g = df.groupby("entry_date")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    k = len(g)
    rng = np.random.default_rng(seed)
    means = [s[i].sum() / c[i].sum() for i in (rng.integers(0, k, k) for _ in range(n))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def portfolio(trades: List[Dict], sessions: List[date], frac: float = 0.10, cost: float = COST,
              cap0: float = 10_000.0, fee: float = 0.0) -> Dict:
    """%10/işlem, nakit sınırlı, günlük değerleme (lab/evaluate.portfolio ile aynı mantık).

    Her işlemin ``details.path`` alanı [(gün, kapanış), …] günlük değerleme serisidir.
    ``fee``: emir başına sabit $ (alışta ve satışta ayrı; lab/swing_o.portfolio ile aynı hesap).
    """
    if not sessions:
        return dict(total_pct=0.0, max_dd_pct=0.0, taken=0)
    by_entry: Dict[str, List[Dict]] = {}
    for t in trades:
        if t.get("entry_date") and t.get("entry_price"):
            by_entry.setdefault(str(t["entry_date"])[:10], []).append(t)
    for v in by_entry.values():
        v.sort(key=lambda t: -(t.get("key") or 0))
    cash, pos, curve, taken = cap0, [], [], 0
    for d in sessions:
        ds = d.isoformat()
        keep = []
        for p in pos:
            if p["exit_date"] == ds:
                cash += p["sh"] * p["exit_price"] * (1 - cost / 200) - fee
            else:
                keep.append(p)
        pos = keep
        eq_open = cash + sum(p["sh"] * p["last"] for p in pos)
        for t in by_entry.get(ds, []):
            alloc = frac * eq_open
            if alloc > cash:
                continue
            e = float(t["entry_price"])
            sh = (alloc - fee) / (e * (1 + cost / 200))
            cash -= alloc
            taken += 1
            path = {str(k)[:10]: float(v) for k, v in ((t.get("details") or {}).get("path") or [])}
            ex_d = str(t.get("exit_date") or "")[:10] or None
            ex_p = float(t["exit_price"]) if t.get("exit_price") is not None else None
            if ex_d == ds:
                cash += sh * ex_p * (1 - cost / 200) - fee
            else:
                pos.append(dict(sh=sh, path=path, last=e, exit_date=ex_d, exit_price=ex_p))
        for p in pos:
            if ds in p["path"]:
                p["last"] = p["path"][ds]
        curve.append(cash + sum(p["sh"] * p["last"] for p in pos))
    eq = pd.Series(curve)
    return dict(total_pct=round(float((eq.iloc[-1] / cap0 - 1) * 100), 2),
                max_dd_pct=round(float((eq / eq.cummax() - 1).min() * 100), 2), taken=taken)


def months_between(a: date, b: date) -> float:
    return (b - a).days / 30.4375


def strategy_report(name: str, trades: List[Dict], sessions: List[date], live_start: date,
                    asof: date, iwm_bh_pct: Optional[float]) -> Dict:
    rows = [t for t in trades if t["strategy"] == name]
    cost, fee = COST_BY.get(name, COST), FEE_BY.get(name, 0.0)
    counts = pd.Series([t["status"] for t in rows]).value_counts().to_dict() if rows else {}
    closed = [t for t in rows if t["status"] == "CLOSED"]
    eligible = [t for t in closed if not t.get("warmup") and not t.get("late_decision")]
    df = pd.DataFrame(eligible)
    out = dict(strategy=name, counts=counts, closed=len(closed), eligible_closed=len(eligible),
               late_decisions=sum(1 for t in rows if t.get("late_decision")),
               warmup=sum(1 for t in rows if t.get("warmup")),
               months_live=round(months_between(live_start, asof), 2),
               cost_pct=cost, fee_per_order=fee, benchmark=BENCHMARK_BY.get(name, "IWM"))
    if len(df):
        df["net"] = df["ret_pct"].astype(float) - cost
        df["excess_iwm"] = df["ret_pct"].astype(float) - df["iwm_ret"].astype(float)
        lo, hi = boot_ci(df, "net")
        out.update(net_mean=round(float(df.net.mean()), 3), net_median=round(float(df.net.median()), 3),
                   win_rate=round(float((df.net > 0).mean() * 100), 1),
                   net_ci95=[None if lo is None else round(lo, 3), None if hi is None else round(hi, 3)],
                   excess_iwm_mean=round(float(df.excess_iwm.mean()), 3))
    port = portfolio([t for t in rows if t["status"] in ("OPEN", "CLOSED")
                      and not t.get("warmup") and not t.get("late_decision")], sessions, cost=cost, fee=fee)
    out["portfolio_10pct"] = port
    out["iwm_buy_hold_pct"] = None if iwm_bh_pct is None else round(iwm_bh_pct, 2)
    out["gate"] = gate(name, out)
    return out


def champion_report(rows: List[Dict], since: str, cost: float = COST,
                    flags: Optional[Dict[str, List]] = None) -> Dict:
    """Aynı dönemde canlı ürünün paper işlemleri, H6 ile aynı maliyetle (Ek F kıyası).

    ``flags`` (Ek H ileri ölçümü): {işlem id: giriş öncesi 30 günde seyreltme başvuruları}. İşaretli
    ve işaretsiz işlemlerin net sonucu ayrı raporlanır — H8 filtresinin gerçek veride testi.
    """
    pct = pd.Series([float(r["realized_pnl_pct"]) for r in rows], dtype=float)
    out = dict(since=since, closed=int(len(pct)), cost_pct=cost,
               label="Canlı ürün (paper işlemleri, manuel eklenenler dahil)")
    if len(pct):
        net = pct - cost
        out.update(gross_mean=round(float(pct.mean()), 3), net_mean=round(float(net.mean()), 3),
                   win_rate=round(float((net > 0).mean() * 100), 1))
    if flags is not None:
        split = {}
        for name, want in (("dilution", True), ("clean", False)):
            sel = [float(r["realized_pnl_pct"]) - cost for r in rows
                   if str(r.get("id")) in flags and bool(flags[str(r.get("id"))]) == want]
            split[name] = dict(closed=len(sel), net_mean=round(float(np.mean(sel)), 3) if sel else None)
        split["unflagged"] = sum(1 for r in rows if str(r.get("id")) not in flags)
        out["dilution_split"] = split
    return out


def gate(name: str, r: Dict) -> Dict:
    """PREREGISTRATION Ek C. Otomatik terfi YOK — yalnız durum raporu."""
    g = GATE[name]
    f1 = r["months_live"] >= g["min_months"] and r["eligible_closed"] >= g["min_closed"]
    ci = r.get("net_ci95") or [None, None]
    f2 = bool(r.get("net_mean") is not None and r["net_mean"] > 0 and ci[0] is not None and ci[0] > 0)
    bh = r.get("iwm_buy_hold_pct")
    f3 = bool(bh is not None and r["portfolio_10pct"]["total_pct"] > bh)
    f4 = bool(r["portfolio_10pct"]["max_dd_pct"] > -35)
    if not f1:
        status = "TOPLANIYOR"
    elif f2 and f3 and f4:
        status = "TERFİ ADAYI"
    elif r.get("net_mean") is not None and r["net_mean"] <= 0:
        status = "KALDI"
    elif r["months_live"] >= g["max_months"]:
        status = "KALDI"
    else:
        status = "BELİRSİZ"
    return dict(status=status, F1_sample=f1, F2_net_ci_gt0=f2, F3_beats_iwm=f3, F4_maxdd_lt35=f4,
                need_closed=g["min_closed"], need_months=g["min_months"])
