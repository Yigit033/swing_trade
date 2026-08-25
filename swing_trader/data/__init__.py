"""Data module for fetching stock data and scan-history storage.

DataFetcher is imported lazily so `import swing_trader.data.settings_storage`
does not pull pandas/yfinance (Fly cold-boot budget is ~8s).
"""

from __future__ import annotations

__all__ = ["DataFetcher"]


def __getattr__(name: str):
    if name == "DataFetcher":
        from .fetcher import DataFetcher

        return DataFetcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
