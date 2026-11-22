"""Ek G — H0S: durgun günlerde statik liste birleşimi (canlı tasarım) vs H0 (birleşimsiz).

Canlı `SmallCapUniverse.get_universe`: Finviz < min_finviz_tickers_skip_static_merge (30) ise
evren = Finviz + statik liste (tekilleştirilmiş, max_scan_tickers tavanı). Bu betik yalnız o
günlerde EKLENEN statik hisseleri canlı motordan geçirir (h0_signals ile aynı işçi), sinyalleri
H0 sinyalleriyle birleştirip `signals_emulated_static.parquet` yazar. Ardından:
    python lab/h0_sim.py --universe emulated_static
    python lab/static_merge.py --evaluate

Usage: python lab/static_merge.py [--workers 6] | --evaluate
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lab.h0_signals as h0  # noqa: E402
from lab.common import EVAL_START, finviz_members, frames  # noqa: E402

OUT = ROOT / "output/lab/h0"


def static_additions():
    from swing_trader.small_cap.universe import SmallCapUniverse
    S = h0.live_settings()
    us = S.universe_scan
    static = SmallCapUniverse(settings=S).get_static_universe()
    members, F = finviz_members(), frames()
    add = {}
    for d, mem in members.items():
        if d < EVAL_START or len(mem) >= us.min_finviz_tickers_skip_static_merge:
            continue
        merged = list(dict.fromkeys(list(mem) + static))[: us.max_scan_tickers]
        have = set(mem)
        add[d] = [t for t in merged if t not in have and t in F]
    return add, static


def run(workers):
    t0 = time.time()
    add, static = static_additions()
    days = sorted(add)
    regimes = h0.regime_series()
    infos = h0.build_infos(add, days)
    allF = frames()
    print(f"quiet days {len(days)}, static scans {sum(len(v) for v in add.values())}, "
          f"static list {len(static)} ({sum(t in allF for t in static)} with lab data)", flush=True)
    size = -(-len(days) // (workers * 4))
    jobs = []
    for i in range(0, len(days), size):
        c = days[i:i + size]
        tick = {t for d in c for t in add[d]}
        lo = c[0] - pd.DateOffset(months=7)
        Fsub = {t: allF[t][(allF[t].Date >= lo) & (allF[t].Date <= c[-1])] for t in tick}
        jobs.append((c, {d: add[d] for d in c}, regimes,
                     {k: v for k, v in infos.items() if k[0] in set(c)}, Fsub, False))
    sigs, outs = [], []
    with ProcessPoolExecutor(workers, initializer=h0._init) as ex:
        for i, (s, o) in enumerate(ex.map(h0._work, jobs)):
            sigs += s
            outs += o
            print(f"chunk {i + 1}/{len(jobs)} static signals={len(sigs)} {time.time() - t0:.0f}s", flush=True)
    S = pd.DataFrame(sigs)
    S["source"] = "static"
    base = pd.read_parquet(OUT / "signals_emulated.parquet")
    base["source"] = "finviz"
    pd.concat([base, S], ignore_index=True).to_parquet(OUT / "signals_emulated_static.parquet", index=False)
    pd.DataFrame(outs).to_parquet(OUT / "outcomes_static_additions.parquet", index=False)
    print(f"static signals {len(S)} | written signals_emulated_static.parquet", flush=True)


def evaluate_both():
    from lab.d1_eval import load
    from lab.evaluate import evaluate
    from lab.pipeline_audit import attach, forward
    from lab.common import panels, pool_mask
    res = {}
    for name, suf in (("H0", ""), ("H0S", "_static")):
        T = load(suf)
        r, TT, _ = evaluate(name, T)
        p = r["portfolio_10pct"]
        res[name] = dict(n=r["n"], net_mean=r["net_mean"], excess=r["excess_mean"], ci=r["excess_ci95"],
                         win=r["win_rate"], port_total=p["total_pct"], max_dd=p["max_dd_pct"])
        print(f"{name:4s} n={r['n']} net={r['net_mean']:+.3f}% excess={r['excess_mean']:+.3f} CI{r['excess_ci95']} "
              f"win={r['win_rate']}% port={p['total_pct']:+.1f}% dd={p['max_dd_pct']}%")
    # Statik kaynaklı sinyaller ve işlemler
    sig = pd.read_parquet(OUT / "signals_emulated_static.parquet")
    P = panels()
    fwd = forward(P, 8)
    pool = pool_mask().reindex(columns=P["Close"].columns, fill_value=False)
    sig = attach(sig, fwd, fwd.where(pool).mean(axis=1))
    by = sig.groupby("source").agg(n=("excess", "size"), excess8=("excess", "mean"), fwd8=("fwd", "mean")).round(3)
    print(by.to_string())
    T = pd.read_parquet(OUT / "trades_emulated_static.parquet")
    src = {(pd.Timestamp(a), b): s for a, b, s in zip(sig.date, sig.ticker, sig.source)}
    T["source"] = [src.get((pd.Timestamp(a), b), "?") for a, b in zip(T.signal_date, T.ticker)]
    done = T[~T.status.isin(["REJECTED", "OPEN", "REVIEW"])]
    tr = done.groupby("source").agg(n=("ret_pct", "size"), gross=("ret_pct", "mean")).round(3)
    tr["net50"] = (tr.gross - 0.5).round(3)
    print(tr.to_string())
    a, b = res["H0"], res["H0S"]
    keep = b["net_mean"] > a["net_mean"] and b["port_total"] > a["port_total"]
    res["signals_by_source"] = by.reset_index().to_dict("records")
    res["trades_by_source"] = tr.reset_index().to_dict("records")
    res["decision"] = "KEEP_STATIC_MERGE" if keep else "REMOVE_STATIC_MERGE"
    print("Ek G decision:", res["decision"])
    (ROOT / "output/lab/g_static_merge.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    evaluate_both() if a.evaluate else run(a.workers)
