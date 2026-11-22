"""H0 pass B — live auto-scan selection, PENDING confirmation and PRODUCTION exits.

Mirrors, day by day (live daily maintenance order):
  1. confirm_pending_trades  : t+1 Open, gap +5/−7 (REJECTED rows count for cooldown),
                               stop = Open − 2.5×ATR14 floored at type cap, T1/T2 rescaled
  2. update_all_open_trades  : paper_trading.execution.evaluate (the production function)
  3. auto-scan               : filtered = Q ≥ eff_min, sorted by (rank, Q), [:eff_top],
                               capacity (BEAR 0 / CAUTION 1 − active), duplicate, cooldown 5d
Earnings gate (live: yfinance earnings_dates, 0..3 days ahead) applied here.

Usage: python lab/h0_sim.py [--universe emulated]
Output: output/lab/h0/trades_<universe>.parquet
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.common import frames, earnings_dates, panels  # noqa: E402
from lab.h0_signals import live_settings  # noqa: E402

OUT = ROOT / "output/lab/h0"


def ny_open_utc(day):
    from zoneinfo import ZoneInfo
    from datetime import datetime, timezone
    dt = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ZoneInfo("America/New_York"))
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def confirm_levels(sp, o, atr, swing_type, stop_loss, target_1, target_2):
    """Replica of PaperTracker.confirm_pending_trades level math (parity-tested)."""
    from swing_trader.paper_trading.tracker import CONFIRM_ATR_MULTIPLIER, MAX_STOP_BY_TYPE, MAX_STOP_DEFAULT
    orig_target_pct = (target_1 - sp) / sp
    if atr > 0:
        stop = round(o - atr * CONFIRM_ATR_MULTIPLIER, 2)
        max_stop = o * (1 - MAX_STOP_BY_TYPE.get(swing_type, MAX_STOP_DEFAULT))
        if stop < max_stop:
            stop = round(max_stop, 2)
        target = round(o * (1 + orig_target_pct), 2)
        min_t = round(o + (o - stop), 2)
        if target < min_t:
            target = min_t
    else:
        stop = round(o * (1 - (sp - stop_loss) / sp), 2)
        target = round(o * (1 + orig_target_pct), 2)
    t2 = round(o * (1 + (target_2 / sp - 1)), 2) if sp > 0 and target_2 > 0 else target
    return stop, target, t2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="emulated")
    # Ek D1: live = veri yok → geç (canlı fail-open), closed = veri yok → ret, off = kapı kapalı
    ap.add_argument("--earnings", default="live", choices=["live", "closed", "off"])
    # Ek H: (date, ticker) parquet — bu sinyaller seçimden önce elenir (filtre hipotezleri)
    ap.add_argument("--exclude", default=None)
    # Nakit hesabı: açık + bekleyen pozisyon tutarı + yeni pozisyon ≤ hesap (kaldıraçsız, Midas)
    ap.add_argument("--cash-limit", action="store_true")
    a = ap.parse_args()
    tag = a.universe + ("" if a.earnings == "live" else f"_earn{a.earnings}")
    if a.exclude:
        tag += "_ex" + Path(a.exclude).stem
    if a.cash_limit:
        tag += "_cash"
    import logging
    logging.disable(logging.CRITICAL)
    from swing_trader.small_cap import settings_config as sc
    S = live_settings()
    sc.load_settings = lambda *x, **k: S
    from swing_trader.small_cap.thresholds import effective_scan_thresholds
    from swing_trader.small_cap.filters import SmallCapFilters
    from swing_trader.small_cap.risk import SmallCapRisk
    from swing_trader.paper_trading import execution
    from api.auto_scan import auto_entry_capacity

    filt, risk = SmallCapFilters(None, S), SmallCapRisk(None, S)
    F = frames()
    cal = panels()["Close"].index
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    sig = pd.read_parquet(OUT / f"signals_{a.universe}.parquet")
    if a.exclude:
        ex = pd.read_parquet(a.exclude)
        drop = set(zip(pd.DatetimeIndex(ex.date), ex.ticker))
        mask = [(pd.Timestamp(d), t) not in drop for d, t in zip(sig.date, sig.ticker)]
        print(f"[exclude {Path(a.exclude).stem}] dropped {len(sig) - sum(mask)} of {len(sig)} signals")
        sig = sig[mask].copy()

    # Earnings gate (pass-A ran with no earnings; reject-only filter → exact to apply here)
    keep, missing = [], 0
    for r in sig.itertuples():
        if a.earnings == "off":
            keep.append(True)
            continue
        ed = earnings_dates(r.ticker)
        if ed is None or len(ed) == 0:
            missing += 1
            keep.append(a.earnings == "live")   # live: fetch failure/empty → "check skipped" → pass
            continue
        ok, _ = filt.check_earnings(r.ticker, r.date.to_pydatetime(), list(ed))
        keep.append(ok)
    sig = sig[keep].copy()
    print(f"[{a.earnings}] signals after earnings gate: {len(sig)} (earnings missing for {missing})")

    us = S.auto_scan
    cash_skips = 0
    trades, closed = [], []          # closed: (ticker, exit_day) for cooldown
    active = []                      # dicts with exit_day (None = still open)
    by_day = {d: g for d, g in sig.groupby("date")}
    selected_log = []
    for d in sorted(by_day):
        g = by_day[d]
        regime, conf = g.regime.iloc[0], g.regime_conf.iloc[0]
        eff_min, eff_top = effective_scan_thresholds(regime, conf, us.min_quality, us.top_n,
                                                     regime_caps=S.regime_thresholds)
        g = g.sort_values(["rank", "quality"], ascending=False)
        g = g[g.quality >= eff_min].head(eff_top)
        live_active = [t for t in active if t["entry_day"] <= d and (t["exit_day"] is None or t["exit_day"] > d)]
        cap = auto_entry_capacity(regime, S.backtest_loop, len(live_active))
        if cap is not None:
            g = g.head(cap)
        held = sum(t.get("notional", 0.0) for t in live_active)
        for r in g.itertuples():
            selected_log.append(dict(date=d, ticker=r.ticker))
            if any(t["ticker"] == r.ticker for t in live_active):
                continue                                     # duplicate (OPEN/PENDING)
            cutoff = d - pd.Timedelta(days=int(S.cooldown_days))
            if any(tk == r.ticker and xd >= cutoff for tk, xd in closed):
                continue                                     # cooldown
            if a.cash_limit:
                need = float(r.entry_price) * int(r.position_size or 0)
                if held + need > float(S.auto_scan.portfolio_value):
                    cash_skips += 1
                    continue                                 # nakit yok → giriş yok
                held += need
            e_day = nxt.get(d)
            df = F[r.ticker]
            if e_day is None or not (df.Date == e_day).any():
                continue                                     # no next session in data
            hist3 = df[(df.Date > d - pd.DateOffset(months=3)) & (df.Date <= d)].reset_index(drop=True)
            atr = risk.calculate_atr(hist3) if len(hist3) >= 14 else 0
            sp = float(r.entry_price)
            o = float(df.loc[df.Date == e_day, "Open"].iloc[0])
            gap = (o / sp - 1) * 100
            rec = dict(signal_date=d, ticker=r.ticker, regime=regime, quality=r.quality,
                       rank=r.rank, pathway=r.pathway, swing_type=r.swing_type,
                       signal_price=sp, entry_day=e_day, gap_pct=gap, atr=round(atr, 4) if atr else 0)
            if gap > S.max_gap_up_pct or gap < -S.max_gap_down_pct:
                rec.update(status="REJECTED", exit_day=e_day)
                trades.append(rec)
                closed.append((r.ticker, e_day))
                continue
            stop, target, target_2 = confirm_levels(sp, o, atr, r.swing_type, r.stop_loss,
                                                    r.target_1, r.target_2)
            size = int(r.position_size or 0)
            size_flag = size < 1
            entry = round(o, 2)
            trade = dict(status="OPEN", ticker=r.ticker, entry_date=ny_open_utc(e_day),
                         entry_price=entry, position_size=max(size, 100) if size_flag else size,
                         stop_loss=stop, initial_stop=stop, target=target, target_2=target_2,
                         max_hold_days=int(r.hold_max), atr=round(atr, 4) if atr else 0,
                         execution_state=None, partial_exit_price=None, policy_effective_at=None,
                         initial_risk=round((entry - stop) * size, 8))
            # Live tracker only ever sees bars from entry up to "today" and stops once the
            # trade closes (timeout ≤ hold bars). Feeding years of future bars would make
            # evaluate() validate unrelated sessions (e.g. 2025-01-09, which production's
            # market_calendar wrongly treats as a session) and quarantine the trade.
            hist = df[df.Date >= e_day][["Date", "Open", "High", "Low", "Close"]]
            hist = hist.head(int(r.hold_max) + 5).reset_index(drop=True)
            upd, (status, xp, xd, reason, _) = execution.evaluate(trade, hist, S.partial_at_t1_fraction)
            if reason.startswith("REVIEW_REQUIRED"):
                status = "REVIEW"
            exit_day = pd.Timestamp(xd[:10]) if xd else None
            if exit_day is not None:
                # exit_date is UTC ISO; 16:00 NY → same calendar date in UTC (20/21h)
                exit_day = pd.Timestamp(pd.Timestamp(xd).tz_convert("America/New_York").date())
            st = json.loads(upd["execution_state"]) if upd.get("execution_state") else {}
            rec.update(status=status, entry_price=entry, stop=stop, target=target, target_2=target_2,
                       hold=int(r.hold_max), exit_day=exit_day, exit_price=xp,
                       ret_pct=upd.get("realized_pnl_pct") if status not in ("OPEN", "REVIEW")
                       else upd.get("unrealized_pnl_pct"),
                       r_mult=upd.get("r_multiple"), partial=bool(st.get("partial")),
                       bars=st.get("bars"), size_flag=size_flag, reason=reason[:80])
            trades.append(rec)
            if status in ("OPEN", "REVIEW"):
                rec["exit_day"] = None
            active.append(dict(ticker=r.ticker, entry_day=d, exit_day=rec["exit_day"],
                               notional=float(entry) * int(trade["position_size"])))
            if rec["exit_day"] is not None:
                closed.append((r.ticker, rec["exit_day"]))
    T = pd.DataFrame(trades)
    T.to_parquet(OUT / f"trades_{tag}.parquet", index=False)
    pd.DataFrame(selected_log).to_parquet(OUT / f"selected_{tag}.parquet", index=False)
    if a.cash_limit:
        print(f"cash-limit skips: {cash_skips}")
    print(T.status.value_counts().to_string())
    done = T[~T.status.isin(["REJECTED", "OPEN", "REVIEW"])]
    print(f"closed trades {len(done)}: mean {done.ret_pct.mean():.2f}% median {done.ret_pct.median():.2f}% "
          f"win {100*(done.ret_pct>0).mean():.1f}%")


if __name__ == "__main__":
    main()
