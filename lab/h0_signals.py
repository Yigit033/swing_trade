"""H0 pass A — run the PRODUCTION scan engine over the emulated live universe.

For every trading day d and every Finviz-emulated member t:
    window  = 6-month bars ending d (live: yf.download(period='6mo')) + live OHLC repair
    info    = market cap (as-of shares × Close_d), float (current float scaled to
              as-of shares), sector  — the fields live gets from Finviz metadata
    regime  = regime_from_spy_close(SPY 1y closes ≤ d, VIX_d)  — live detect_market_regime
    signal  = SmallCapEngine.scan_stock(..., regime=regime)     — production function
Earnings are applied in pass B (pure reject filter; see h0_sim.py).

Settings: production defaults + the live Supabase patch captured 2026-10-11.

Usage: python lab/h0_signals.py [--universe recorded] [--workers 6]
Output: output/lab/h0/signals_<universe>.parquet, outcomes_<universe>.parquet
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.common import LAB  # noqa: E402  (veri kökü: LAB_DATA, varsayılan output/lab)

PATCH = ROOT / "output/discovery_v2/phase2/effective_patch.json"
OUT = LAB / "h0"

_ENGINE = None
_FETCH = None


def live_settings():
    """Production defaults overlaid with the captured live patch (deep merge)."""
    from swing_trader.small_cap import settings_config as sc
    # Same layering/pruning as production load_settings (defaults → DB patch).
    base = sc.SmallCapSettings().model_dump(mode="json")
    merged = sc._deep_merge(base, sc._prune_removed_keys(json.loads(PATCH.read_text())))
    return sc.SmallCapSettings.model_validate(merged)


def _init():
    global _ENGINE, _FETCH
    import logging
    logging.disable(logging.CRITICAL)
    from swing_trader.small_cap import settings_config as sc
    from swing_trader.small_cap.engine import SmallCapEngine
    from swing_trader.data.fetcher import DataFetcher
    s = live_settings()
    sc.load_settings = lambda *a, **k: s          # every component reads the live settings
    import swing_trader.small_cap.engine as eng
    eng.load_settings = sc.load_settings
    _ENGINE = SmallCapEngine()
    assert _ENGINE.settings.regime_thresholds.bull_min_quality == s.regime_thresholds.bull_min_quality
    _FETCH = DataFetcher.__new__(DataFetcher)      # only for _repair_ohlc_bars (pure)


def regime_series():
    from lab.common import panels
    from swing_trader.small_cap.regime_logic import regime_from_spy_close
    P = panels()
    spy, vix = P["Close"]["SPY"], P["Close"]["^VIX"].ffill()
    out = {}
    for i, d in enumerate(spy.index):
        if i < 60:
            continue
        hist = spy.iloc[max(0, i - 251): i + 1].dropna()
        r = regime_from_spy_close(hist.reset_index(drop=True), float(vix.iloc[i]))
        out[d] = (r.get("regime", "UNKNOWN"), r.get("confidence", "TENTATIVE"))
    return out


def build_infos(members, days):
    """(d, t) -> live-shaped Finviz metadata (computed once in the parent process)."""
    from lab.common import shares_panel, info_frame, actual_close
    S, info, C = shares_panel(), info_frame(), actual_close()   # o günün gerçek fiyatı (Ek R; 2021–26'da aynı)
    out = {}
    for d in days:
        for t in members[d]:
            if t not in C.columns or pd.isna(C.at[d, t]):
                continue
            sh = float(S.at[d, t]) if t in S.columns and pd.notna(S.at[d, t]) else 0.0
            cap = sh * float(C.at[d, t])
            fl_now = info.at[t, "floatShares"] if t in info.index else None
            sh_now = info.at[t, "sharesOutstanding"] if t in info.index else None
            fl_now = float(fl_now) if fl_now is not None and pd.notna(fl_now) else 0.0
            sh_now = float(sh_now) if sh_now is not None and pd.notna(sh_now) else 0.0
            fl = min(sh, fl_now * sh / sh_now) if (fl_now and sh_now and sh) else fl_now
            sec = info.at[t, "sector"] if t in info.index else None
            out[(d, t)] = {"ticker": t, "marketCap": int(cap), "floatShares": int(fl),
                           "shortName": t, "sector": sec if isinstance(sec, str) else "Unknown"}
    return out


_REAL_FLOAT = False  # PREREGISTRATION Ek B (H0F): supply approximate float instead of live 0


def _work(args):
    days, members, regimes, infos, F, real_float, *rest = args
    factors = rest[0] if rest else {}                    # (gün, hisse) → bölünme katsayısı; yoksa 1 (static_merge)
    global _REAL_FLOAT
    _REAL_FLOAT = real_float
    from lab.common import asof_window
    sig_rows, out_rows = [], []
    for d in days:
        regime, conf = regimes.get(d, ("UNKNOWN", "TENTATIVE"))
        for t in members[d]:
            df = F.get(t)
            stock_info = infos.get((d, t))
            if df is None or stock_info is None:
                continue
            w = asof_window(df, d, factors.get((d, t), 1.0))   # canlının o gün gördüğü fiyat ölçeği (Ek R)
            if len(w) < 20 or w.Date.iloc[-1] != d:
                continue
            w = _FETCH._repair_ohlc_bars(w, t)
            cap, fl = stock_info["marketCap"], stock_info["floatShares"]
            # LIVE PARITY: production reads Finviz *Overview* rows, which have no
            # Float column → live floatShares is always 0 ("unknown", score 5).
            # Verified against recorded runs (quality deltas were exact float-band
            # steps). backtest_mode=False so float 0 is NOT replaced by the 45M
            # backtest default; all inputs are supplied, so no network is touched.
            live_info = stock_info if _REAL_FLOAT else dict(stock_info, floatShares=0)
            outcomes = []
            s = _ENGINE.scan_stock(t, w, stock_info=live_info, backtest_mode=False,
                                   portfolio_value=10000, outcomes=outcomes,
                                   regime=regime, earnings_dates=[])  # [] = "bilanço yok" (açık liste); kapı pass B'de
            if s:
                td = s.get("trigger_details") or {}
                sig_rows.append(dict(
                    date=d, ticker=t, regime=regime, regime_conf=conf,
                    quality=s["quality_score"], rank=s["rank_score"],
                    pathway=s.get("trigger_pathway"), swing_type=s["swing_type"],
                    entry_price=s["entry_price"], stop_loss=s["stop_loss"],
                    target_1=s["target_1"], target_2=s["target_2"],
                    position_size=s.get("position_size", 0), hold_max=s["hold_days_max"],
                    rsi=s.get("rsi"), volume_surge=s.get("volume_surge"),
                    atr_pct=s.get("atr_percent"), cap=cap, float_sh=fl,
                    squeeze=(td.get("vce_metrics") or {}).get("squeeze_ratio")))
                out_rows.append(dict(date=d, ticker=t, reject="SIGNAL", quality=s["quality_score"]))
            else:
                o = outcomes[-1] if outcomes else {}
                out_rows.append(dict(date=d, ticker=t, reject=o.get("reject_reason", "?"),
                                     quality=o.get("quality")))
    return sig_rows, out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="emulated", choices=["emulated", "recorded"])
    ap.add_argument("--float", dest="flt", default="live", choices=["live", "real"])
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    from lab.common import finviz_members, panels, split_factor, EVAL_START, EVAL_END
    t0 = time.time()
    regimes = regime_series()
    if a.universe == "recorded":
        runs = json.loads((ROOT / "output/discovery_v2/recorded_runs.json").read_text())
        members = {}
        for r in runs:
            st = r.get("stats") or {}
            bar = st.get("bar_session")
            mem = [m["ticker"] for m in st.get("scanned_members") or []
                   if m.get("kind") in ("universe", "signal")]
            if bar and mem:
                members[pd.Timestamp(bar)] = sorted(set(members.get(pd.Timestamp(bar), [])) | set(mem))
        days = sorted(d for d in members if d in panels()["Close"].index)
    else:
        members = finviz_members()
        days = [d for d in sorted(members) if d >= EVAL_START and (EVAL_END is None or d <= EVAL_END)]
    from lab.common import frames
    infos = build_infos(members, days)
    allF = frames()
    SF = split_factor()                                   # 2021–26 kökünde ≡ 1 → factors boş
    factors = {(d, t): float(SF.at[d, t]) for d in days for t in members[d]
               if t in SF.columns and SF.at[d, t] != 1.0}
    print(f"{len(days)} days, {sum(len(members[d]) for d in days)} scans, prep {time.time()-t0:.0f}s", flush=True)
    # Contiguous date blocks keep each job's ticker set (and pickle) small.
    nb = a.workers * 6
    size = -(-len(days) // nb)
    chunks = [days[i:i + size] for i in range(0, len(days), size)]
    jobs = []
    for c in chunks:
        tick = {t for d in c for t in members[d]}
        lo = c[0] - pd.DateOffset(months=7)
        Fsub = {t: allF[t][(allF[t].Date >= lo) & (allF[t].Date <= c[-1])] for t in tick if t in allF}
        cs = set(c)
        jobs.append((c, {d: members[d] for d in c}, regimes,
                     {k: v for k, v in infos.items() if k[0] in cs}, Fsub, a.flt == "real",
                     {k: v for k, v in factors.items() if k[0] in cs}))
    sigs, outs = [], []
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for i, (s, o) in enumerate(ex.map(_work, jobs)):
            sigs += s
            outs += o
            print(f"chunk {i+1}/{len(jobs)} signals={len(sigs)} {time.time()-t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.universe + ("_floatreal" if a.flt == "real" else "")
    pd.DataFrame(sigs).to_parquet(OUT / f"signals_{tag}.parquet", index=False)
    pd.DataFrame(outs).to_parquet(OUT / f"outcomes_{tag}.parquet", index=False)
    pd.Series({str(k): v for k, v in regimes.items()}).to_json(OUT / "regimes.json")
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
