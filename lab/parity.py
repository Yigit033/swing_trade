"""Replica fidelity vs RECORDED live production runs (output/discovery_v2/recorded_runs.json).

  1. Universe : emulated Finviz membership vs the members live actually scanned
  2. Engine   : replica scan_stock outcome on the live member list vs live outcome
                (same reject reason; quality |Δ| when both produced a score)
Requires: h0_signals.py --universe recorded (outcomes_recorded.parquet)

Usage: python lab/parity.py → output/lab/parity.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.common import finviz_members, panels  # noqa: E402

LAB = ROOT / "output/lab"


def recorded():
    runs = json.loads((ROOT / "output/discovery_v2/recorded_runs.json").read_text())
    live = {}
    for r in runs:
        st = r.get("stats") or {}
        bar = st.get("bar_session")
        if not bar:
            continue
        for m in st.get("scanned_members") or []:
            if m.get("kind") not in ("universe", "signal"):
                continue
            m = dict(m, reject_reason=m.get("reject_reason") or "SIGNAL")
            live.setdefault(pd.Timestamp(bar), {})[m["ticker"]] = m
    return live


def main():
    live = recorded()
    cal = set(panels()["Close"].index)
    emu = finviz_members()
    days = sorted(d for d in live if d in cal)
    inter = lv = em = 0
    for d in days:
        L, E = set(live[d]), set(emu.get(d, []))
        inter += len(L & E)
        lv += len(L)
        em += len(E)
    uni = dict(days=len(days), live_members=lv, emulated_members=em,
               recall=round(inter / lv, 3) if lv else None,
               precision=round(inter / em, 3) if em else None)

    out = pd.read_parquet(LAB / "h0/outcomes_recorded.parquet")
    pairs, dq = Counter(), []
    same = tot = 0
    for r in out.itertuples():
        m = live.get(r.date, {}).get(r.ticker)
        if not m:
            continue
        lr = m.get("reject_reason") or "SIGNAL"
        rr = r.reject
        tot += 1
        if lr == rr or (rr == "SIGNAL" and lr in ("signal", "passed", None)):
            same += 1
        pairs[(lr, rr)] += 1
        if m.get("quality") is not None and r.quality is not None and pd.notna(r.quality):
            dq.append(abs(float(m["quality"]) - float(r.quality)))
    eng = dict(compared=tot, same_outcome=same, rate=round(same / tot, 3) if tot else None,
               quality_pairs=len(dq),
               quality_abs_diff_median=round(float(pd.Series(dq).median()), 2) if dq else None,
               quality_within_2pts=round(float((pd.Series(dq) <= 2).mean()), 3) if dq else None,
               mismatches=[dict(live=a, replica=b, n=n) for (a, b), n in pairs.most_common(15) if a != b])
    res = dict(universe=uni, engine=eng)
    (LAB / "parity.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
