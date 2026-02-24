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
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


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
