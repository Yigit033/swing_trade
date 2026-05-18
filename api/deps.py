"""
Shared dependencies and singleton components for all routers.
"""

import yaml
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.warning(f"config.yaml not found at {CONFIG_PATH}, using defaults")
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_paper_storage():
    from swing_trader.paper_trading.storage import PaperTradeStorage
    return PaperTradeStorage()


@lru_cache(maxsize=1)
def get_paper_tracker():
    from swing_trader.paper_trading.tracker import PaperTradeTracker
    return PaperTradeTracker(get_paper_storage())


@lru_cache(maxsize=1)
def get_paper_reporter():
    from swing_trader.paper_trading.reporter import PaperTradeReporter
    return PaperTradeReporter()


@lru_cache(maxsize=1)
def get_smallcap_engine():
    from swing_trader.small_cap import SmallCapEngine
    return SmallCapEngine(get_config())


def invalidate_smallcap_engine_cache() -> None:
    """Call after small-cap JSON settings change so the next scan uses fresh parameters."""
    get_smallcap_engine.cache_clear()


@lru_cache(maxsize=1)
def get_fetcher():
    from swing_trader.data.fetcher import DataFetcher
    import os
    config = get_config()
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    keys_cfg = config.get("api_keys", {}) if isinstance(config, dict) else {}
    source = data_cfg.get("source", "yfinance")
    alpha_vantage_key = data_cfg.get("alpha_vantage_key") or data_cfg.get("alpha_vantage_api_key")
    # Env vars override config.yaml (easier for production secrets)
    tiingo_key  = os.environ.get("TIINGO_API_KEY")  or keys_cfg.get("tiingo", "")
    finnhub_key = os.environ.get("FINNHUB_API_KEY") or keys_cfg.get("finnhub", "")
    return DataFetcher(
        source=source,
        alpha_vantage_key=alpha_vantage_key,
        tiingo_key=tiingo_key,
        finnhub_key=finnhub_key,
    )


@lru_cache(maxsize=1)
def get_regime_storage():
    from swing_trader.data.regime_storage import RegimeHistoryStorage
    return RegimeHistoryStorage()


@lru_cache(maxsize=1)
def get_signal_history_storage():
    from swing_trader.data.signal_history_storage import SmallCapSignalHistoryStorage
    return SmallCapSignalHistoryStorage()
