"""
SmallCap Backtest router.
POST /api/backtest/smallcap  - run walk-forward SmallCap backtest
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.utils import sanitize_for_json

router = APIRouter()
logger = logging.getLogger(__name__)


class BacktestRequest(BaseModel):
    period_days: int = Field(default=90, ge=7, le=730)  # max ~2 years
    initial_capital: float = Field(default=10000, ge=100)
    max_concurrent: int = Field(default=3, ge=1, le=8)
    min_quality: int = Field(default=60, ge=30, le=100)
    top_n: int = Field(default=10, ge=1, le=50)
    tickers: Optional[List[str]] = None   # None = auto Finviz; [] invalid — rejected below


@router.post("/smallcap")
def run_smallcap_backtest(body: BacktestRequest):
    """
    Run SmallCap walk-forward backtest.

    Uses SmallCapBacktester which simulates daily scans over historical data.
    If tickers is None, fetches universe from Finviz (slower, ~1-2 min).
    If tickers provided, only those tickers are scanned.
    """
    try:
        from swing_trader.small_cap.smallcap_backtest import SmallCapBacktester
        from swing_trader.small_cap.engine import SmallCapEngine

        end_date   = datetime.now()
        start_date = end_date - timedelta(days=body.period_days)

        backtester = SmallCapBacktester(config=None)

        # Resolve ticker list: explicit [] must not fall through to Finviz
        if body.tickers is not None:
            tickers = [t.strip().upper() for t in body.tickers if t.strip()]
            if not tickers:
                return {"error": "Ticker list is empty — add symbols or choose Finviz universe", "results": None}
        else:
            # Finviz universe — same cap as live scanner (api/routers/scanner.py)
            engine = SmallCapEngine(config=None)
            tickers = engine.get_small_cap_universe()
            if not tickers:
                return {"error": "No tickers found from Finviz universe", "results": None}

        logger.info(f"Backtest: {len(tickers)} tickers, {start_date.date()} → {end_date.date()}")

        results = backtester.run_backtest(
            tickers=tickers,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=body.initial_capital,
            max_concurrent=body.max_concurrent,
            min_quality=body.min_quality,
            top_n=body.top_n,
        )

        n_trades = len(results.get("trades") or [])
        logger.info(
            "Backtest finished: %d closed trade(s). Per-trade fields (dates, prices, P/L) are in the JSON response / UI — not printed here.",
            n_trades,
        )

        return sanitize_for_json({
            "period_days":    body.period_days,
            "start_date":     start_date.strftime("%Y-%m-%d"),
            "end_date":       end_date.strftime("%Y-%m-%d"),
            "tickers_used":   tickers,
            "initial_capital": body.initial_capital,
            "min_quality":    body.min_quality,
            "top_n":          body.top_n,
            "data_stocks":    results.get("data_stocks", 0),
            "params":         results.get("params", {}),
            "diagnostics":    results.get("diagnostics", {}),
            "metrics":        results.get("metrics", {}),
            "equity_curve":   results.get("equity_curve", []),
            "trades":         results.get("trades", []),
        })

    except Exception as e:
        logger.exception("SmallCap backtest failed")
        return {"error": str(e), "results": None}
