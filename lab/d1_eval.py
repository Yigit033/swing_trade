"""Ek D1 — bilanço kapısı: H0 (veri yoksa geç) vs H0EC (veri yoksa ret) vs H0E0 (kapı kapalı).

Birincil: onarılmış bilanço verisi (lab.common.earnings_dates → earnings_full yedeği).
Ayrıca onarımsız (_rawearn) koşular raporlanır. Karar kuralı PREREGISTRATION.md Ek D1.

Usage: python lab/d1_eval.py
Output: output/lab/d1_report.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.common import panels  # noqa: E402
from lab.evaluate import evaluate  # noqa: E402

H0 = ROOT / "output/lab/h0"
VARIANTS = (("H0", ""), ("H0EC", "_earnclosed"), ("H0E0", "_earnoff"))


def load(suffix):
    T = pd.read_parquet(H0 / f"trades_emulated{suffix}.parquet")
    T = T[~T.status.isin(["REJECTED", "REVIEW"])].rename(columns={"rank": "key"})
    T["censored"] = T.status.eq("OPEN")
    T.loc[T.censored, "exit_day"] = panels()["Close"].index[-1]
    return T


def main():
    out = {}
    for data, tail in (("repaired", ""), ("raw", "_rawearn")):
        for name, suf in VARIANTS:
            r, _, _ = evaluate(name, load(suf + tail))
            p = r["portfolio_10pct"]
            out[f"{data}:{name}"] = dict(n=r["n"], net_mean=r["net_mean"], excess=r["excess_mean"],
                                         ci=r["excess_ci95"], win=r["win_rate"],
                                         port_total=p["total_pct"], max_dd=p["max_dd_pct"])
            print(f"{data:8s} {name:5s} n={r['n']:3d} net={r['net_mean']:+.3f}% excess={r['excess_mean']:+.3f} "
                  f"CI{r['excess_ci95']} win={r['win_rate']}% port={p['total_pct']:+.1f}% dd={p['max_dd_pct']}%")
    h0, e0 = out["repaired:H0"], out["repaired:H0E0"]
    harmful = e0["net_mean"] > h0["net_mean"] and e0["port_total"] > h0["port_total"]
    out["decision"] = ("GATE_HARMFUL_DO_NOT_FAIL_CLOSE" if harmful else "APPLY_FAIL_CLOSED_FIX")
    print("D1 decision:", out["decision"])
    (ROOT / "output/lab/d1_report.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
