"""Ek D2 — H4L / H5L (büyük hisse havuzu L, 10 bps maliyet, SPY kıyası). Parametreler ön kayıtlı.

Usage: python lab/d2_eval.py
Output: output/lab/d2_report.json, output/lab/eval_H4L.parquet, eval_H5L.parquet
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.common import panels, pool_mask_large  # noqa: E402
from lab.evaluate import evaluate  # noqa: E402

LAB = ROOT / "output/lab"
COST_L, STRESS_L = 0.10, 0.20


def spy_buy_hold(eq: pd.Series) -> float:
    C = panels()["Close"]["SPY"]
    return round(float((C[eq.index[-1]] / C[eq.index[0]] - 1) * 100), 1)


def main():
    pool = pool_mask_large()
    res = []
    for name in ("H4L", "H5L"):
        T = pd.read_parquet(LAB / f"challengers/trades_{name}.parquet")
        out, TT, eq = evaluate(name, T, cost=COST_L, pool=pool, idx="SPY", stress=STRESS_L)
        out["spy_buy_hold_same_period_pct"] = spy_buy_hold(eq)
        out["portfolio_10pct_at_50bps"] = evaluate(name, T, cost=0.50, pool=pool, idx="SPY")[0]["portfolio_10pct"]
        res.append(out)
        TT.to_parquet(LAB / f"eval_{name}.parquet", index=False)
        p, c = out["portfolio_10pct"], out["criteria"]
        print(f"\n== {name}: n={out['n']} months={out['months']} gross={out['gross_mean']}% "
              f"net10={out['net_10bps']}% net20={out['net_20bps']}% net50={out['net_50bps']}% "
              f"win={out['win_rate']}% PF={out['profit_factor']}")
        print(f"   pool L={out['pool_mean']}% excess={out['excess_mean']} CI{out['excess_ci95']} "
              f"SPY/trade={out['iwm_mean']}% trim5={out['net_trim_top5']}%")
        print(f"   portfolio@10bps {p} | @50bps total={out['portfolio_10pct_at_50bps']['total_pct']}% "
              f"| SPY buy-hold {out['spy_buy_hold_same_period_pct']}%")
        print(f"   PROMOTED={out['promoted']} {c}")
        print(pd.DataFrame(out["yearly"]).to_string(index=False))
    (LAB / "d2_report.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
