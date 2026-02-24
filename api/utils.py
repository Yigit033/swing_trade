"""Shared utilities for API routers."""

import pandas as pd
import numpy as np


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
