"""
SmallCap Scanner router.
GET  /api/scanner/chart          - OHLCV + indicators for any ticker
POST /api/scanner/smallcap       - run SmallCap scan (sync, legacy)
POST /api/scanner/smallcap/start - queue background scan → { job_id }
GET  /api/scanner/smallcap/job/{job_id} - poll status + result when done
POST /api/scanner/track          - add a signal to paper trades via tracker (duplicate-safe)
"""

import logging
import threading
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Callable, Optional

from api.deps import (
    get_smallcap_engine,
    get_paper_tracker,
    get_fetcher,
    get_regime_storage,
    get_signal_history_storage,
)
from api.auth import get_current_user_id
from api.utils import flatten_yf_df, sanitize_for_json, fetch_ticker_history
from api.scanner_jobs import (
    create_exclusive_scan_job,
    get_job_public,
    run_scan_worker,
    current_scan_job_id,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class TrackSignalRequest(BaseModel):
    ticker: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: Optional[float] = None
    swing_type: str = "A"
    quality_score: float = 0
    position_size: int = 100
    hold_days_max: int = 7
    atr: Optional[float] = 0
    date: Optional[str] = None


@router.post("/track")
def track_signal(
    body: TrackSignalRequest,
    user_id: Optional[str] = Depends(get_current_user_id),
):
    """
    Add a signal to paper trades using the tracker (duplicate-safe).
    Returns: {"status": "added", "trade_id": N} or {"status": "duplicate"}
    """
    tracker = get_paper_tracker()
    signal = {
        "ticker": body.ticker,
        "entry_price": body.entry_price,
        "stop_loss": body.stop_loss,
        "target_1": body.target_1,
        "target_2": body.target_2 or body.target_1,
        "swing_type": body.swing_type,
        "quality_score": body.quality_score,
        "position_size": body.position_size,
        "hold_days_max": body.hold_days_max,
        "atr": body.atr or 0,
        "date": body.date or datetime.date.today().isoformat(),
    }
    trade_id = tracker.add_trade_from_signal(signal, user_id)
    if trade_id > 0:
        return sanitize_for_json({"status": "added", "trade_id": trade_id})
    else:
        return sanitize_for_json({"status": "duplicate", "trade_id": -1})


@router.get("/chart")
def get_chart_data(ticker: str = Query(...), period: str = Query("3mo")):
    """Return OHLCV + RSI + MACD + EMAs for charting."""
    try:
        fetcher = get_fetcher()
        df = fetcher.fetch_stock_data(ticker.upper(), period=period)
        if df is None or len(df) < 5:
            return {"error": f"Veri alınamadı ({ticker}). Yahoo Finance geçici olarak rate limit uygulamış olabilir — birkaç dakika sonra tekrar deneyin.", "ticker": ticker}

        close = df["Close"].astype(float)
        vol   = df["Volume"].astype(float)

        # RSI (14)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["RSI"] = (100 - (100 / (1 + rs))).round(4)

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"]        = (ema12 - ema26).round(4)
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean().round(4)
        df["MACD_hist"]   = (df["MACD"] - df["MACD_signal"]).round(4)

        # EMA 20 / 50
        df["EMA20"] = close.ewm(span=20, adjust=False).mean().round(4)
        df["EMA50"] = close.ewm(span=50, adjust=False).mean().round(4)

        # Volume MA 20
        df["Volume_MA"] = vol.rolling(20).mean().round(0)

        def safe(v):
            try:
                f = float(v)
                return None if f != f else round(f, 4)
            except Exception:
                return None

        rows = []
        for _, row in df.iterrows():
            date = str(row.get("Date", ""))[:10]
            rows.append({
                "date":        date,
                "open":        safe(row.get("Open")),
                "high":        safe(row.get("High")),
                "low":         safe(row.get("Low")),
                "close":       safe(row.get("Close")),
                "volume":      safe(row.get("Volume")),
                "rsi":         safe(row.get("RSI")),
                "macd":        safe(row.get("MACD")),
                "macd_signal": safe(row.get("MACD_signal")),
                "macd_hist":   safe(row.get("MACD_hist")),
                "ema20":       safe(row.get("EMA20")),
                "ema50":       safe(row.get("EMA50")),
                "volume_ma":   safe(row.get("Volume_MA")),
            })

        return sanitize_for_json({"ticker": ticker.upper(), "period": period, "data": rows})

    except Exception as e:
        logger.exception(f"Chart data error for {ticker}")
        return sanitize_for_json({"error": str(e), "ticker": ticker})


class ScanRequest(BaseModel):
    portfolio_value: float = 10000
    min_quality: int = 65
    top_n: int = 10


def _scan_regime_and_thresholds(
    engine,
    body: ScanRequest,
    signals: list,
) -> tuple[str, float, str, int, int]:
    """
    Resolve regime for API response (including 0-signal scans via _last_regime)
    Resolve regime for API response (including 0-signal scans via _last_regime)
    and compute effective_min_quality / effective_top_n from regime+confidence.
    """
    last = getattr(engine, "_last_regime", None) or {}
    if signals:
        actual_regime = signals[0].get("market_regime", "UNKNOWN")
        regime_confidence = signals[0].get("regime_confidence", "CONFIRMED")
    else:
        actual_regime = last.get("regime", "UNKNOWN")
        regime_confidence = last.get("confidence", "CONFIRMED")

    from swing_trader.small_cap.thresholds import effective_scan_thresholds

    eff_min, eff_top = effective_scan_thresholds(
        actual_regime,
        regime_confidence,
        body.min_quality,
        body.top_n,
        regime_caps=engine.settings.regime_thresholds,
    )
    logger.info(
        "Effective thresholds: regime=%s conf=%s mult=%s → min_quality=%s top_n=%s (request %s/%s)",
        actual_regime,
        regime_confidence,
        "n/a",
        eff_min,
        eff_top,
        body.min_quality,
        body.top_n,
    )

    return actual_regime, 1.0, regime_confidence, eff_min, eff_top


def _execute_smallcap_scan(
    body: ScanRequest,
    on_progress: Optional[Callable[[int, str, str], None]] = None,
    *,
    job_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """Core scan logic; optional on_progress(percent, phase, message)."""
    def prog(pct: int, phase: str, message: str) -> None:
        if on_progress:
            on_progress(pct, phase, message)

    engine = get_smallcap_engine()
    fetcher = get_fetcher()

    prog(2, "universe", "Fetching small-cap universe…")
    tickers = engine.get_small_cap_universe()
    logger.info(f"SmallCap universe: {len(tickers)} tickers")
    prog(6, "universe", f"Universe: {len(tickers)} tickers")

    data_dict: dict[str, pd.DataFrame] = {}
    n = len(tickers)
    for i, ticker in enumerate(tickers):
        try:
            df = fetcher.fetch_stock_data(ticker, period="3mo")
            if df is not None and len(df) >= 20:
                data_dict[ticker] = df
        except Exception:
            continue
        if n > 0 and (i % 2 == 0 or i == n - 1):
            pct = 8 + int(75 * (i + 1) / n)
            prog(pct, "fetch", f"Price data {i + 1}/{n} ({ticker})…")

    if not data_dict:
        return {"signals": [], "stats": {"reason": "no_data"}, "market_regime": "UNKNOWN"}

    prog(84, "scan", "Running momentum engine…")
    signals = engine.scan_universe(
        tickers=list(data_dict.keys()),
        data_dict=data_dict,
        portfolio_value=body.portfolio_value,
    )

    prog(90, "filter", "Filtering signals…")

    actual_regime, regime_multiplier, regime_confidence, effective_min_quality, effective_top_n = (
        _scan_regime_and_thresholds(engine, body, signals)
    )

    filtered = [s for s in signals if s.get("quality_score", 0) >= effective_min_quality]
    filtered = filtered[: effective_top_n]

    try:
        regime_data = getattr(engine, "_last_regime", None)
        if regime_data:
            get_regime_storage().save_regime(regime_data)
    except Exception:
        logger.debug("Regime save skipped (non-critical)")

    reject_counts = getattr(engine, "_last_scan_reject_counts", None) or {}
    stats = {
        "stocks_scanned": len(tickers),
        "stocks_with_data": len(data_dict),
        "raw_signals": len(signals),
        "filtered_signals": len(filtered),
        "reject_counts": dict(sorted(reject_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "reason": "success" if filtered else "no_qualifying",
        "regime_multiplier": regime_multiplier,
        "regime_confidence": regime_confidence,
        "effective_min_quality": effective_min_quality,
        "effective_top_n": effective_top_n,
        "request_min_quality": body.min_quality,
        "request_top_n": body.top_n,
    }
    rd = getattr(engine, "_last_regime", None) or {}
    err = rd.get("detect_error")
    if err:
        stats["regime_detect_error"] = err

    # Persist scan result for future analysis (non-critical).
    try:
        get_signal_history_storage().save_run(
            job_id=job_id,
            user_id=user_id,
            portfolio_value=body.portfolio_value,
            request_min_quality=body.min_quality,
            request_top_n=body.top_n,
            effective_min_quality=effective_min_quality,
            effective_top_n=effective_top_n,
            market_regime=actual_regime,
            regime_confidence=regime_confidence,
            stats=stats,
            signals=filtered,
        )
    except Exception:
        logger.debug("Signal history save skipped (non-critical)")
    prog(97, "finalize", "Done")
    return sanitize_for_json({"signals": filtered, "stats": stats, "market_regime": actual_regime})


@router.post("/smallcap/start")
def start_smallcap_scan(
    body: ScanRequest,
    user_id: Optional[str] = Depends(get_current_user_id),
):
    """
    Start scan in a background thread. Returns immediately with job_id.
    Poll GET /api/scanner/smallcap/job/{job_id} until status is completed|failed.
    """
    job_id = create_exclusive_scan_job()
    if not job_id:
        active = current_scan_job_id()
        return JSONResponse(
            status_code=409,
            content={
                "detail": "A scan is already running",
                "active_job_id": active,
            },
        )

    def _run(body_inner: ScanRequest, progress_cb: Callable[[int, str, str], None]) -> dict:
        return _execute_smallcap_scan(body_inner, progress_cb, job_id=job_id, user_id=user_id)

    thread = threading.Thread(
        target=run_scan_worker,
        args=(job_id, body, _run),
        daemon=True,
        name=f"smallcap-scan-{job_id[:8]}",
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@router.get("/smallcap/job/{job_id}")
def get_smallcap_scan_job(job_id: str):
    """Poll scan progress / fetch result when completed."""
    data = get_job_public(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return data


@router.post("/smallcap")
def run_smallcap_scan(
    body: ScanRequest,
    user_id: Optional[str] = Depends(get_current_user_id),
):
    """Synchronous scan (legacy / scripts). Prefer POST /smallcap/start for UI."""
    try:
        return _execute_smallcap_scan(body, on_progress=None, job_id=None, user_id=user_id)
    except Exception as e:
        logger.exception("SmallCap scan failed")
        return sanitize_for_json(
            {
                "signals": [],
                "stats": {"reason": "error", "error": str(e)},
                "market_regime": "UNKNOWN",
                "error": str(e),
            }
        )


@router.get("/smallcap/history")
def list_smallcap_scan_history(
    limit: int = Query(20, ge=1, le=200),
    user_id: Optional[str] = Depends(get_current_user_id),
):
    """List recent saved scan runs (metadata only)."""
    rows = get_signal_history_storage().list_runs(limit=limit, user_id=user_id)
    return sanitize_for_json({"runs": rows, "count": len(rows)})


@router.get("/smallcap/history/{run_id}")
def get_smallcap_scan_history(
    run_id: int,
    user_id: Optional[str] = Depends(get_current_user_id),
):
    """Fetch one saved scan run including full signals payload."""
    row = get_signal_history_storage().get_run(run_id, user_id=user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return sanitize_for_json(row)
