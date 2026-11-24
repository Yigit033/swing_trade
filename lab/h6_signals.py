"""Ek F — H6 "Sade VCE" lab sağlık kontrolü (etiketli IN-SAMPLE; karar gölge ileri testte).

Canlı replikanın motor kayıtlarından tetiği geçen hisse-günler (outcomes: swing_not_ready,
rsi_gate, stage_rejected, quality_*, SIGNAL) alınır; her birinde canlı VCE fonksiyonu yeniden
çalıştırılır (aynı 6 ay pencere + OHLC onarımı, canlı ayarlar). Sert filtreler bu kayıtlarda
zaten geçilmiştir (filter_failed motorda tetikten önce). Bilanço kapısı onarılmış veriyle,
bilinmiyorsa ret (canlı kural). Giriş t+1 Open, çıkış 8. seans Close, stop/hedef yok.

Usage: python lab/h6_signals.py
Output: <kök>/challengers/trades_H6.parquet, <kök>/h6_report.json  (kök: LAB_DATA, varsayılan output/lab)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.challengers import OUT, simulate, walk  # noqa: E402
from lab.common import (EVAL_END, EVAL_START, LAB, asof_window, earnings_dates, frames, indicators,  # noqa: E402
                        panels, split_factor)
from lab.evaluate import evaluate  # noqa: E402
import lab.h0_signals as h0  # noqa: E402

PASSED_TRIGGER = ("swing_not_ready", "rsi_gate", "stage_rejected", "SIGNAL")
HOLD = 8


def trigger_rows() -> pd.DataFrame:
    """Motor kayıtlarından değerlendirme dönemindeki, tetiği geçmiş hisse-günler."""
    o = pd.read_parquet(LAB / "h0/outcomes_emulated.parquet")
    d = pd.DatetimeIndex(o.date)
    keep = (d >= EVAL_START) & (o.reject.isin(PASSED_TRIGGER) | o.reject.str.startswith("quality_type"))
    if EVAL_END is not None:
        keep &= d <= EVAL_END
    return o[keep]


def vce_hits(o: pd.DataFrame) -> list:
    """Canlı VCE tetiğini her kayıtta yeniden çalıştırır → [(gün, hisse)] (h0._init çağrılmış olmalı)."""
    eng, rep = h0._ENGINE, h0._FETCH
    F, SF = frames(), split_factor()
    hits = []
    for r in o.itertuples():
        w = asof_window(F[r.ticker], r.date, float(SF.at[r.date, r.ticker]))
        w = rep._repair_ohlc_bars(w, r.ticker)
        _, td = eng.signals.check_all_triggers(w)
        if ((td.get("triggers") or {}).get("vce_breakout") or {}).get("passed"):
            hits.append((r.date, r.ticker))
    return hits


def earnings_gate(hits, known=None):
    """Canlı bilanço kapısı (bilinmiyorsa ret) → (aday tablosu, reddedilen sayısı).

    known(dates, day) → o gün için takvim güvenilir mi (Ek R veri kalitesi kuralı). None = 2021–26
    davranışı: takvim boş değilse güvenilir.
    """
    from swing_trader.small_cap.filters import SmallCapFilters
    filt = SmallCapFilters(None, h0._ENGINE.settings)
    dv = indicators()["dv20"]
    rows, rejected = [], 0
    for d, t in hits:
        ed = earnings_dates(t)
        if ed is None or len(ed) == 0 or (known is not None and not known(ed, d)) \
                or not filt.check_earnings(t, d.to_pydatetime(), list(ed))[0]:
            rejected += 1
            continue
        rows.append(dict(date=d, ticker=t, key=float(dv.at[d, t]), atr=0.0))
    cands = pd.DataFrame(rows, columns=["date", "ticker", "key", "atr"])
    return cands.sort_values(["date", "key"], ascending=[True, False]), rejected


def main():
    h0._init()
    o = trigger_rows()
    hits = vce_hits(o)
    cands, n_earn_rej = earnings_gate(hits)
    n_vce = len(hits)
    cands.to_parquet(OUT / "h6_candidates.parquet", index=False)      # Ek H filtreleri için
    P = panels()
    T = simulate(cands, P, lambda ie, arr, row: walk(ie, arr, None, HOLD), "H6")
    T.to_parquet(OUT / "trades_H6.parquet", index=False)
    res = {"trigger_rows": len(o), "vce_hits": n_vce, "earnings_rejected": n_earn_rej,
           "candidates": len(cands)}
    for cost in (0.50, 0.20):
        out, _, _ = evaluate("H6", T, cost=cost)
        res[f"cost_{int(cost * 100)}bps"] = out
    (LAB / "h6_report.json").write_text(json.dumps(res, indent=1, default=str))
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
