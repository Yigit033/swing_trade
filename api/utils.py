"""Shared utilities for API routers."""

import pandas as pd


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
