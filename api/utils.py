"""Shared utilities for API routers."""

import logging
import time
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# In-memory TTL cache for yfinance data (avoids rate limiting)
_yf_cache: dict = {}    # key=(ticker,period) → (timestamp, df)
_YF_CACHE_TTL = 300     # 5 minutes


def fetch_ticker_history(ticker: str, period: str = "3mo", interval: str = "1d",
                         retries: int = 1, backoff: float = 1.0) -> pd.DataFrame:
    """
    Rate-limit-resilient stock data fetch using yf.Ticker().history().

    Uses in-memory TTL cache (5min) to avoid repeated Yahoo Finance requests
    for the same ticker. This mirrors Streamlit's @st.cache_data behavior.

    Returns a flat DataFrame with columns: Date, Open, High, Low, Close, Volume
    or empty DataFrame on failure.
    """
    import yfinance as yf
    import requests as req

    ticker = ticker.upper().strip()
    cache_key = (ticker, period, interval)

    # Check cache first
    if cache_key in _yf_cache:
        cached_time, cached_df = _yf_cache[cache_key]
        if time.time() - cached_time < _YF_CACHE_TTL:
            logger.debug(f"fetch_ticker_history({ticker}): serving from cache")
            return cached_df

    # Use a proper session with User-Agent to reduce rate limiting
    session = req.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    })

    for attempt in range(retries + 1):
        try:
            stock = yf.Ticker(ticker, session=session)
            df = stock.history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty and len(df) >= 2:
                df = df.reset_index()
                df.columns = [str(c) for c in df.columns]
                # Cache the result
                _yf_cache[cache_key] = (time.time(), df)
                return df
            logger.warning(f"fetch_ticker_history({ticker}): empty result (attempt {attempt+1})")
        except Exception as e:
            err_msg = str(e).lower()
            if 'rate' in err_msg or 'limit' in err_msg or 'too many' in err_msg:
                logger.warning(f"fetch_ticker_history({ticker}): rate limited (attempt {attempt+1})")
            else:
                logger.warning(f"fetch_ticker_history({ticker}) attempt {attempt+1} failed: {e}")
        if attempt < retries:
            time.sleep(backoff)

    return pd.DataFrame()


def flatten_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a yfinance DataFrame regardless of whether it has
    a MultiIndex (multi-ticker download) or a flat index (single-ticker).

    After this call df has simple string column names:
        Date, Open, High, Low, Close, Volume
    and integer index.
    """
    if df is None or df.empty:
        return df

    # If the index is a DatetimeIndex, move it to a column first
    if isinstance(df.columns, pd.MultiIndex):
        # Drop the ticker level (level 1), keep field names (level 0)
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Now reset index so Date becomes a regular column
    df = df.reset_index()

    # Rename the index column to "Date" if it was named something else
    if "index" in df.columns:
        df.rename(columns={"index": "Date"}, inplace=True)
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    # Stringify all column names (safety)
    df.columns = [str(c) for c in df.columns]
    return df


def sanitize_for_json(obj):
    """
    Recursively convert numpy / pandas types to plain Python types so
    that FastAPI's json.dumps doesn't crash on numpy.bool_, float64, etc.

    This is REQUIRED because engine methods (check_breakout, etc.) return
    numpy scalars directly inside their result dicts.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(i) for i in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (f != f) else f       # nan → None
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if hasattr(obj, 'item'):                  # any remaining numpy scalar
        return sanitize_for_json(obj.item())
    # pandas NaT / NaN
    if obj is pd.NaT:
        return None
    try:
        import math
        if isinstance(obj, float) and math.isnan(obj):
            return None
    except Exception:
        pass
    return obj
