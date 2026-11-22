"""
SmallCap Backtest router.
POST /api/backtest/smallcap  - run walk-forward SmallCap backtest
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from api.auth import require_user
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.utils import sanitize_for_json

router = APIRouter()
logger = logging.getLogger(__name__)


class BacktestRequest(BaseModel):
    period_days: int = Field(default=90, ge=7, le=730)  # max ~2 years
    initial_capital: float = Field(default=10000, ge=100)
    # None = canlı ürünle birebir: eşik/top_n canlı auto-scan ayarından, eşzamanlı
    # pozisyon sınırı yok (canlıda yalnız rejim kapasitesi var). Değer verilirse
    # canlıda OLMAYAN bir simülasyon kısıtı olarak uygulanır.
    max_concurrent: Optional[int] = Field(default=None, ge=1, le=50)
    min_quality: Optional[int] = Field(default=None, ge=30, le=100)
    top_n: Optional[int] = Field(default=None, ge=1, le=50)
    tickers: Optional[List[str]] = None   # None = auto Finviz; [] invalid — rejected below


@router.post("/smallcap", dependencies=[Depends(require_user)])
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
        survivorship_warning = None
        stock_meta = None
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
            # Canlı tarama market cap'i Finviz satırından okur — backtest de aynı kaynağı
            # kullanır (geçmiş günlere fiyat oranıyla ölçeklenir).
            stock_meta = {t: engine.universe_provider.get_ticker_metadata(t) or {} for t in tickers}
            # 2026-07-27: Finviz evreni BUGÜNKÜ listeyi geçmişe uygular →
            # survivorship bias (bugün kazanan hisselerle geçmişi test etmek
            # sonucu AŞIRI İYİMSER gösterir). Kullanıcı bunu bilmeli.
            survivorship_warning = (
                "⚠️ Bu backtest BUGÜNKÜ Finviz evrenini geçmişe uyguladı "
                "(survivorship bias). Bugün var olan/kazanan hisselerle geçmişi "
                "test etmek sonucu GERÇEKTEN İYİMSER gösterir — canlı performans "
                "daha düşük olabilir. Gerçekçi ölçüm için scripts/backtest_live_replica.py "
                "(point-in-time evren) kullanılır."
            )

        logger.info(f"Backtest: {len(tickers)} tickers, {start_date.date()} → {end_date.date()}")

        results = backtester.run_backtest(
            tickers=tickers,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=body.initial_capital,
            max_concurrent=body.max_concurrent,
            min_quality=body.min_quality,
            top_n=body.top_n,
            stock_meta=stock_meta,
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
            "min_quality":    (results.get("params") or {}).get("min_quality"),
            "top_n":          (results.get("params") or {}).get("top_n"),
            "data_stocks":    results.get("data_stocks", 0),
            "params":         results.get("params", {}),
            "diagnostics":    results.get("diagnostics", {}),
            "metrics":        results.get("metrics", {}),
            "equity_curve":   results.get("equity_curve", []),
            "trades":         results.get("trades", []),
            "survivorship_warning": survivorship_warning,
        })

    except Exception as e:
        logger.exception("SmallCap backtest failed")
        return {"error": str(e), "results": None}
