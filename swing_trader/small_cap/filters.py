"""
Small-cap universe filters — the four HARD gates before VCE/RVOL even run.

Live gates (settings.universe_filters, except dollar-volume which is measured
and fixed here):
  - last close in [min_price, max_price]     default $7–$200
  - market cap in [min, max]                 default $250M–$10B
  - 20-day avg (Volume × Close) ≥ $5M/day
  - earnings in the next 0..N days           default N=3; post-earnings allowed

ATR% and float are RECORDED for scoring/display. They do not reject.
v13: ATR hard floor contradicted VCE (squeeze = low ATR).
2026-08-14: absolute ATR floor also failed (measure_atr_floor.py).
Float >80M hard reject would cut the universe the VCE edge was measured on
(HIMS, SOFI, RKLB…). Tightness is scoring.score_float_tightness (weight 25%).
"""

import logging
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .settings_config import SmallCapSettings

logger = logging.getLogger(__name__)


class SmallCapFilters:
    """Hard universe filters. Defaults match SmallCapSettings; __init__ overwrites."""

    # Class-level fallbacks = live defaults (so a bare read isn't a lie).
    MIN_MARKET_CAP = 250_000_000
    MAX_MARKET_CAP = 10_000_000_000
    MIN_DOLLAR_VOLUME = 5_000_000     # not in settings — measured liquidity floor
    MIN_PRICE = 7.0
    MAX_PRICE = 200.0
    EARNINGS_EXCLUSION_DAYS = 3       # pre-event 0..+N only
    ATR_PERIOD = 10

    def __init__(self, config: Dict = None, settings: Optional["SmallCapSettings"] = None):
        from .settings_config import load_settings

        self.config = config or {}
        self._settings = settings if settings is not None else load_settings()
        uf = self._settings.universe_filters
        self.MIN_MARKET_CAP = uf.min_market_cap
        self.MAX_MARKET_CAP = uf.max_market_cap
        self.MIN_PRICE = uf.min_price
        self.MAX_PRICE = uf.max_price
        self.EARNINGS_EXCLUSION_DAYS = uf.earnings_exclusion_days
        self.ATR_PERIOD = uf.atr_period
        logger.info(
            "SmallCapFilters initialized (hard: price $%.0f–$%.0f, mcap $%.0fM–$%.0fB, "
            "dvol $%.1fM/day, earnings 0..+%dd)",
            self.MIN_PRICE, self.MAX_PRICE,
            self.MIN_MARKET_CAP / 1e6, self.MAX_MARKET_CAP / 1e9,
            self.MIN_DOLLAR_VOLUME / 1e6, self.EARNINGS_EXCLUSION_DAYS,
        )

    def calculate_atr_percent(self, df: pd.DataFrame, period: int = None) -> float:
        """ATR as a fraction of close (10-period mean true range). Never a gate."""
        if period is None:
            period = self.ATR_PERIOD

        if df is None or len(df) < period + 1:
            return 0.0

        try:
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values

            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1])
                )
            )

            atr = np.mean(tr[-period:])
            current_close = close[-1]

            return atr / current_close if current_close > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating ATR%: {e}")
            return 0.0

    def calculate_avg_dollar_volume(self, df: pd.DataFrame, period: int = 20) -> float:
        """20-day mean of Volume × Close. Share-count volume is the wrong liquidity unit."""
        if df is None or len(df) < period:
            return 0.0
        try:
            tail = df.tail(period)
            return float((tail['Volume'] * tail['Close']).mean())
        except Exception:
            return 0.0

    def check_market_cap(self, market_cap: float) -> Tuple[bool, str]:
        if market_cap is None or market_cap <= 0:
            return True, "Market cap unknown (allowing — ticker pre-screened)"

        if market_cap < self.MIN_MARKET_CAP:
            return False, (
                f"Market cap too small (${market_cap/1e6:.0f}M < "
                f"${self.MIN_MARKET_CAP/1e6:.0f}M)"
            )

        if market_cap > self.MAX_MARKET_CAP:
            return False, (
                f"Market cap too large (${market_cap/1e9:.1f}B > "
                f"${self.MAX_MARKET_CAP/1e9:.0f}B)"
            )

        return True, f"Market cap OK (${market_cap/1e6:.0f}M)"

    def check_dollar_volume(self, avg_dollar_vol: float) -> Tuple[bool, str]:
        """Real liquidity gate. $4×10M shares = $40M (ok); $15×250K = $3.75M (reject)."""
        min_dv = self.MIN_DOLLAR_VOLUME
        if avg_dollar_vol < min_dv:
            return False, (
                f"Dollar volume too low (${avg_dollar_vol/1e6:.1f}M/day < "
                f"${min_dv/1e6:.0f}M)"
            )
        return True, f"Dollar volume OK (${avg_dollar_vol/1e6:.1f}M/day)"

    def check_float(self, float_shares: float) -> Tuple[bool, str]:
        """Always passes. Size is a scoring input, not a reject."""
        if float_shares is None or float_shares <= 0:
            return True, "Float unknown (allowing)"
        return True, f"Float {float_shares/1e6:.0f}M (advisory — scoring only)"

    def check_price(self, price: float) -> Tuple[bool, str]:
        if price is None or price <= 0:
            return False, "Unknown price"

        if price < self.MIN_PRICE:
            return False, (
                f"Price too low (${price:.2f} < ${self.MIN_PRICE:.0f} — "
                f"below institutional participation floor)"
            )

        if price > self.MAX_PRICE:
            return False, (
                f"Price too high (${price:.2f} > ${self.MAX_PRICE:.0f})"
            )

        return True, f"Price OK (${price:.2f})"

    def check_earnings(self, ticker: str, signal_date, earnings_dates=None) -> Tuple[bool, str]:
        """Reject if earnings fall on signal day or within the next N days.

        Pass ``earnings_dates`` to skip yfinance (backtest/live parity).
        Negative days_diff (already reported) is allowed — gap-and-go is a
        valid swing. LIFE 2026-08-03: RVOL fired the same day as earnings;
        live rejected, a dateless backtest would have kept it.
        """
        from datetime import datetime

        try:
            if earnings_dates is None:
                import yfinance as yf
                earnings_dates = yf.Ticker(ticker).earnings_dates

            if earnings_dates is None or len(earnings_dates) == 0:
                return True, "No nearby earnings"

            if isinstance(signal_date, str):
                signal_date = datetime.strptime(signal_date, '%Y-%m-%d')

            # DataFrame/Series → dates on the index. Do NOT use getattr(x, "index")
            # on a list: list.index is a method and would silently break.
            if isinstance(earnings_dates, (pd.DataFrame, pd.Series)):
                candidates = earnings_dates.index
            else:
                candidates = earnings_dates

            for earn_date in candidates:
                earn_ts = pd.Timestamp(earn_date)
                if earn_ts.tzinfo is not None:
                    earn_ts = earn_ts.tz_localize(None)
                days_diff = (earn_ts.date() - signal_date.date()).days
                if 0 <= days_diff <= self.EARNINGS_EXCLUSION_DAYS:
                    return False, f"Earnings in {days_diff} days (pre-event risk)"

            return True, "No nearby earnings"

        except Exception as e:
            logger.debug(f"Earnings check error for {ticker}: {e}")
            return True, "Earnings check skipped"

    def apply_all_filters(
        self,
        ticker: str,
        df: pd.DataFrame,
        stock_info: Dict,
        signal_date=None,
        backtest_mode: bool = False,
        earnings_dates=None,
    ) -> Tuple[bool, Dict]:
        """Run hard gates in order. First failure returns passed=False.

        ATR% and float are always recorded with passed=True.
        """
        from datetime import datetime

        results = {
            'ticker': ticker,
            'passed': False,
            'filters': {}
        }

        if signal_date is None:
            signal_date = datetime.now()

        if df is not None and len(df) > 0:
            last_close = float(df['Close'].iloc[-1])
            passed, reason = self.check_price(last_close)
            results['filters']['price'] = {'passed': passed, 'reason': reason, 'value': last_close}
            if not passed:
                return False, results

        market_cap = stock_info.get('marketCap', 0) or stock_info.get('market_cap', 0)
        if backtest_mode and (market_cap is None or market_cap <= 0):
            market_cap = int(self.MIN_MARKET_CAP * 1.2)
        passed, reason = self.check_market_cap(market_cap)
        results['filters']['market_cap'] = {'passed': passed, 'reason': reason, 'value': market_cap}
        if not passed:
            return False, results

        # Key used to be named avg_volume; lookup showed "share count". Dollar volume.
        avg_dollar_vol = self.calculate_avg_dollar_volume(df)
        passed, reason = self.check_dollar_volume(avg_dollar_vol)
        results['filters']['dollar_volume'] = {'passed': passed, 'reason': reason, 'value': avg_dollar_vol}
        if not passed:
            return False, results

        atr_pct = self.calculate_atr_percent(df)
        results['filters']['atr_percent'] = {
            'passed': True,
            'reason': f"ATR {atr_pct*100:.1f}% (recorded, not a gate)",
            'value': atr_pct,
        }

        float_shares = stock_info.get('floatShares', 0) or stock_info.get('float_shares', 0)
        if backtest_mode and (float_shares is None or float_shares <= 0):
            float_shares = 45_000_000
        _ok, reason = self.check_float(float_shares)
        results['filters']['float'] = {
            'passed': True,
            'reason': reason,
            'value': float_shares,
        }

        if earnings_dates is not None:
            passed, reason = self.check_earnings(ticker, signal_date, earnings_dates)
            results['filters']['earnings'] = {'passed': passed, 'reason': reason}
            if not passed:
                return False, results
        elif backtest_mode:
            results['filters']['earnings'] = {
                'passed': True,
                'reason': 'Bilanço verisi geçirilmedi (canlıdan SAPMA — parite eksik)',
            }
        else:
            passed, reason = self.check_earnings(ticker, signal_date)
            results['filters']['earnings'] = {'passed': passed, 'reason': reason}
            if not passed:
                return False, results

        results['passed'] = True
        results['atr_percent'] = atr_pct
        results['market_cap'] = market_cap
        results['float_shares'] = float_shares

        return True, results
