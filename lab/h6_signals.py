"""Ek F — H6 "Sade VCE" lab sağlık kontrolü (etiketli IN-SAMPLE; karar gölge ileri testte).

Canlı replikanın motor kayıtlarından tetiği geçen hisse-günler (outcomes: swing_not_ready,
rsi_gate, stage_rejected, quality_*, SIGNAL) alınır; her birinde canlı VCE fonksiyonu yeniden
çalıştırılır (aynı 6 ay pencere + OHLC onarımı, canlı ayarlar). Sert filtreler bu kayıtlarda
zaten geçilmiştir (filter_failed motorda tetikten önce). Bilanço kapısı onarılmış veriyle,
bilinmiyorsa ret (canlı kural). Giriş t+1 Open, çıkış 8. seans Close, stop/hedef yok.

Usage: python lab/h6_signals.py
Output: output/lab/challengers/trades_H6.parquet, output/lab/h6_report.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.challengers import OUT, simulate, walk  # noqa: E402
from lab.common import EVAL_START, earnings_dates, frames, indicators, panels, window_6mo  # noqa: E402
from lab.evaluate import evaluate  # noqa: E402
import lab.h0_signals as h0  # noqa: E402

PASSED_TRIGGER = ("swing_not_ready", "rsi_gate", "stage_rejected", "SIGNAL")
HOLD = 8


def main():
    h0._init()
    eng, rep = h0._ENGINE, h0._FETCH
    from swing_trader.small_cap.filters import SmallCapFilters
    filt = SmallCapFilters(None, eng.settings)
    o = pd.read_parquet(ROOT / "output/lab/h0/outcomes_emulated.parquet")
    o = o[(pd.DatetimeIndex(o.date) >= EVAL_START)
          & (o.reject.isin(PASSED_TRIGGER) | o.reject.str.startswith("quality_type"))]
    F, dv = frames(), indicators()["dv20"]
    rows, n_vce, n_earn_rej = [], 0, 0
    for r in o.itertuples():
        w = window_6mo(F[r.ticker], r.date)
        w = rep._repair_ohlc_bars(w, r.ticker)
        _, td = eng.signals.check_all_triggers(w)
        if not ((td.get("triggers") or {}).get("vce_breakout") or {}).get("passed"):
            continue
        n_vce += 1
        ed = earnings_dates(r.ticker)
        if ed is None or len(ed) == 0 or not filt.check_earnings(r.ticker, r.date.to_pydatetime(), list(ed))[0]:
            n_earn_rej += 1
            continue
        rows.append(dict(date=r.date, ticker=r.ticker, key=float(dv.at[r.date, r.ticker]), atr=0.0))
    cands = pd.DataFrame(rows).sort_values(["date", "key"], ascending=[True, False])
    P = panels()
    T = simulate(cands, P, lambda ie, arr, row: walk(ie, arr, None, HOLD), "H6")
    T.to_parquet(OUT / "trades_H6.parquet", index=False)
    res = {"trigger_rows": len(o), "vce_hits": n_vce, "earnings_rejected": n_earn_rej,
           "candidates": len(cands)}
    for cost in (0.50, 0.20):
        out, _, _ = evaluate("H6", T, cost=cost)
        res[f"cost_{int(cost * 100)}bps"] = out
    (ROOT / "output/lab/h6_report.json").write_text(json.dumps(res, indent=1, default=str))
    m = res["cost_50bps"]
    print(f"trigger rows {len(o)} → VCE {n_vce} → earnings rejected {n_earn_rej} → candidates {len(cands)} "
          f"→ trades {m['n']} over {m['months']} months")
    print(f"gross {m['gross_mean']}% net50 {m['net_mean']}% net20 {m['net_20bps']}% win {m['win_rate']}% "
          f"PF {m['profit_factor']} | pool {m['pool_mean']}% excess {m['excess_mean']} CI{m['excess_ci95']}")
    print(f"portfolio@50 {m['portfolio_10pct']} | @20 {res['cost_20bps']['portfolio_10pct']}")
    print(f"criteria {m['criteria']}")
    print(pd.DataFrame(m["yearly"]).to_string(index=False))


if __name__ == "__main__":
    main()
