"""
SmallCap Backtest router.
POST /api/backtest/smallcap  - run walk-forward SmallCap backtest
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import yfinance as yf
from fastapi import APIRouter
from pydantic import BaseModel

from api.utils import sanitize_for_json

router = APIRouter()
logger = logging.getLogger(__name__)


class BacktestRequest(BaseModel):
    period_days: int = 90           # 30, 60, 90, or 180
    initial_capital: float = 10000
    max_concurrent: int = 3
    tickers: Optional[List[str]] = None   # None = auto Finviz (slower)


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

        # Resolve ticker list
        if body.tickers:
            tickers = [t.strip().upper() for t in body.tickers if t.strip()]
        else:
            # Finviz universe (can be slow ~30-60 s)
            engine = SmallCapEngine(config=None)
            tickers = engine.get_small_cap_universe(use_finviz=True, max_tickers=50)
            if not tickers:
                return {"error": "No tickers found from Finviz universe", "results": None}

        logger.info(f"Backtest: {len(tickers)} tickers, {start_date.date()} → {end_date.date()}")

        results = backtester.run_backtest(
            tickers=tickers,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=body.initial_capital,
            max_concurrent=body.max_concurrent,
        )

        return sanitize_for_json({
            "period_days":    body.period_days,
            "start_date":     start_date.strftime("%Y-%m-%d"),
            "end_date":       end_date.strftime("%Y-%m-%d"),
            "tickers_used":   tickers,
            "initial_capital": body.initial_capital,
            "metrics":        results.get("metrics", {}),
            "equity_curve":   results.get("equity_curve", []),
            "trades":         results.get("trades", []),
        })

    except Exception as e:
        logger.exception("SmallCap backtest failed")
        return {"error": str(e), "results": None}
