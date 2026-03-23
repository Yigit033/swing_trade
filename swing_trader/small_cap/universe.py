"""
Small Cap Universe Provider - Dynamic universe using finvizfinance.
Provides fresh small-cap stock lists from Finviz screener.

OPTIMIZED v3.0 - "Para Kazanma Makinesi" Edition
- Pre-filters at Finviz level to avoid scanning 200+ stocks that will fail downstream
- Multi-factor momentum ranking (not just today's change%)
- Country filter to avoid ADR noise
- Float filter to focus on explosion potential
- Volatility filter to ensure tradeable setups
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class SmallCapUniverse:
    """
    Dynamic small-cap universe provider using finvizfinance.

    PRE-SCREENING STRATEGY (v3.0):
    We run TWO Finviz queries and merge results for maximum coverage:

    Query 1 - MOMENTUM HUNTERS (aggressive):
      Market Cap: Small ($300M-$2B), Float: Under 100M
      Relative Volume: Over 1.5, Price: Over $3
      Country: USA, Volatility: Week Over 5%

    Query 2 - SETUP BUILDERS (wider net):
      Market Cap: Small ($300M-$2B), Float: Under 100M
      Average Volume: Over 1M, Price: Over $3
      Country: USA, RSI: Not Overbought (<60)

    Then rank by COMPOSITE MOMENTUM SCORE and return top N.
    """

    CACHE_DURATION_MINUTES = 60

    # Known delisted/problematic tickers to exclude
    EXCLUDED_TICKERS = {
        'BCOV', 'BGFV', 'CARA', 'GNOG', 'ZIRB', 'BBIG', 'IRNT', 'OPAD',
        'SPIR', 'CLOV', 'WISH', 'HOOD', 'LCID', 'RIVN', 'NKLA', 'WKHS',
        'FSR', 'GOEV', 'FFIE', 'MULN', 'RIDE', 'HYLN', 'ARVL', 'VLDR',
        'HCP', 'SQ', 'CERE', 'FREY', 'DTC', 'FTCH', 'ZNGA', 'VORB', 'TELL'
    }

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self._cache = None
        self._cache_time = None
        logger.info("SmallCapUniverse initialized (v3.0 - Optimized)")

    def _parse_percent(self, value) -> float:
        """Parse percentage string like '5.23%' to float 5.23"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value.replace('%', '').replace(',', ''))
            return 0.0
        except Exception:
            return 0.0

    def _parse_volume(self, value) -> float:
        """Parse volume string like '1.5M' or '500K' to numeric"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                value = value.replace(',', '')
                if 'M' in value:
                    return float(value.replace('M', '')) * 1_000_000
                elif 'K' in value:
                    return float(value.replace('K', '')) * 1_000
                elif 'B' in value:
                    return float(value.replace('B', '')) * 1_000_000_000
                else:
                    return float(value)
            return 0.0
        except Exception:
            return 0.0

    def _parse_market_cap(self, value) -> float:
        """Parse market cap string to numeric"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                value = value.replace(',', '').replace('$', '').strip()
                if 'B' in value:
                    return float(value.replace('B', '')) * 1_000_000_000
                elif 'M' in value:
                    return float(value.replace('M', '')) * 1_000_000
                elif 'K' in value:
                    return float(value.replace('K', '')) * 1_000
                else:
                    return float(value)
            return 0.0
        except Exception:
            return 0.0

    def _run_finviz_query(self, filters_dict: Dict, label: str) -> pd.DataFrame:
        """Run a single Finviz screener query and return DataFrame."""
        try:
            from finvizfinance.screener.overview import Overview

            foverview = Overview()
            foverview.set_filter(filters_dict=filters_dict)
            df = foverview.screener_view()

            if df is None or len(df) == 0:
                logger.info(f"  [{label}] returned 0 results")
                return pd.DataFrame()

            logger.info(f"  [{label}] returned {len(df)} tickers")
            return df

        except Exception as e:
            logger.warning(f"  [{label}] query failed: {e}")
            return pd.DataFrame()

    def _calculate_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite momentum score using multiple factors.

        COMPOSITE SCORE = (
            Relative_Volume_Score (40%) +    # Abnormal volume = institutional interest
            Price_Change_Score (30%) +        # Today's move strength
            Volume_Weight (20%) +             # Absolute liquidity
            Close_Position_Score (10%)        # Close near high = buyer control
        )
        """
        import numpy as np

        # Parse columns
        if 'Change' in df.columns:
            df['change_pct'] = df['Change'].apply(self._parse_percent)
        else:
            df['change_pct'] = 0.0

        if 'Volume' in df.columns:
            df['vol_numeric'] = df['Volume'].apply(self._parse_volume)
        else:
            df['vol_numeric'] = 0.0

        if 'Rel Volume' in df.columns:
            df['rel_vol'] = pd.to_numeric(df['Rel Volume'], errors='coerce').fillna(1.0)
        elif 'Relative Volume' in df.columns:
            df['rel_vol'] = pd.to_numeric(df['Relative Volume'], errors='coerce').fillna(1.0)
        else:
            # Estimate relative volume from raw volume (less accurate)
            df['rel_vol'] = 1.0

        if 'Market Cap' in df.columns:
            df['mcap_numeric'] = df['Market Cap'].apply(self._parse_market_cap)
        else:
            df['mcap_numeric'] = 0.0

        # ============================================================
        # COMPONENT 1: Relative Volume Score (40% weight)
        # RVOL is the #1 signal for institutional interest
        # ============================================================
        # Scale: 1x=0, 1.5x=25, 2x=50, 3x=75, 5x+=100
        df['rvol_score'] = np.clip((df['rel_vol'] - 1.0) / 4.0 * 100, 0, 100)

        # ============================================================
        # COMPONENT 2: Price Change Score (25% weight — reduced from 30%)
        # Sweet spot: +2% to +10% (early entry zone)
        # V4: steeper chasing penalties — don't rank "already pumped" at top
        # ============================================================
        change_pct = df['change_pct']
        change_abs = change_pct.abs()
        up_bias = (change_pct > 0).astype(float) * 1.5 + 0.5
        df['change_score'] = np.clip(change_abs * up_bias * 5, 0, 100)
        chase_penalty = np.where(change_pct > 15, 50,
                        np.where(change_pct > 10, 25, 0))
        df['change_score'] = np.clip(df['change_score'] - chase_penalty, 0, 100)

        # ============================================================
        # COMPONENT 3: Volume Weight (20% weight)
        # Higher absolute volume = better liquidity & easier execution
        # ============================================================
        # Log scale: 500K=0, 1M=40, 5M=70, 10M+=100
        df['vol_score'] = np.clip(
            np.log10(df['vol_numeric'].clip(lower=1)) / np.log10(10_000_000) * 100,
            0, 100
        )

        # ============================================================
        # COMPONENT 4: Market Cap Sweet Spot (10% weight)
        # Ideal: $300M-$1.5B (true small-cap explosion zone)
        # ============================================================
        # $300M-$800M: 100, $800M-$1.5B: 70, $1.5B-$2.5B: 40
        mcap_m = df['mcap_numeric'] / 1_000_000
        df['mcap_score'] = np.where(
            mcap_m <= 800, 100,
            np.where(mcap_m <= 1500, 70,
                     np.where(mcap_m <= 2500, 40, 20))
        )

        # ============================================================
        # COMPOSITE SCORE (V4 rebalanced — less chasing, more liquidity)
        # RVOL 30% (was 40%), Change 25% (was 30%), Volume 25% (was 20%), MCap 20% (was 10%)
        # ============================================================
        df['composite_score'] = (
            df['rvol_score'] * 0.30 +
            df['change_score'] * 0.25 +
            df['vol_score'] * 0.25 +
            df['mcap_score'] * 0.20
        )

        return df

    def get_finviz_universe(self, max_tickers: int = 200) -> List[str]:
        """
        Get small-cap universe from Finviz with optimized pre-screening.

        Runs 2 queries for different profiles, merges and ranks by composite score.

        Returns:
            List of ticker symbols sorted by COMPOSITE MOMENTUM SCORE
        """
        try:
            logger.info("Fetching small-cap universe from Finviz (v3.0 optimized)...")

            # ============================================================
            # QUERY 1: MOMENTUM HUNTERS - Today's movers with tight float
            # Catches stocks currently surging (active momentum)
            # ============================================================
            q1_filters = {
                'Market Cap.': 'Small ($300mln to $2bln)',  # Strict small-cap
                'Float': 'Under 100M',                       # Tight float = explosion
                'Price': 'Over $3',                          # No penny stocks
                'Country': 'USA',                            # USA only (no ADR noise)
                'Relative Volume': 'Over 1.5',               # Abnormal volume today
                'Volatility': 'Week - Over 5%',              # Must be volatile
                'Average Volume': 'Over 200K',               # Minimum liquidity
            }
            df1 = self._run_finviz_query(q1_filters, "MOMENTUM HUNTERS")

            # ============================================================
            # QUERY 2: SETUP BUILDERS - High volume stocks not overbought
            # Catches stocks building bases before breakout
            # ============================================================
            q2_filters = {
                'Market Cap.': 'Small ($300mln to $2bln)',  # Strict small-cap
                'Float': 'Under 100M',                       # Tight float
                'Price': 'Over $3',                          # No penny stocks
                'Country': 'USA',                            # USA only
                'Average Volume': 'Over 750K',               # Good daily liquidity
                'RSI (14)': 'Not Overbought (<60)',          # Room to run
                'Volatility': 'Week - Over 3%',              # Minimum volatility
            }
            df2 = self._run_finviz_query(q2_filters, "SETUP BUILDERS")

            # ============================================================
            # QUERY 3: WIDER NET - Catch stocks with float up to 150M
            # Some good setups have 60-150M float with strong catalysts
            # ============================================================
            q3_filters = {
                'Market Cap.': 'Small ($300mln to $2bln)',
                'Price': 'Over $3',
                'Country': 'USA',
                'Relative Volume': 'Over 2',                 # Strong volume required for larger float
                'Average Volume': 'Over 1M',                 # Higher liquidity required
                'Volatility': 'Week - Over 5%',
            }
            df3 = self._run_finviz_query(q3_filters, "WIDER NET")

            # ============================================================
            # MERGE & DEDUPLICATE
            # ============================================================
            frames = [f for f in [df1, df2, df3] if len(f) > 0]
            if not frames:
                logger.warning("All Finviz queries returned empty")
                return []

            df = pd.concat(frames, ignore_index=True)
            df = df.drop_duplicates(subset='Ticker', keep='first')

            logger.info(f"Merged universe: {len(df)} unique tickers")

            # ============================================================
            # POST-FILTERS (code-level precision)
            # ============================================================
            # Remove excluded tickers
            df = df[~df['Ticker'].isin(self.EXCLUDED_TICKERS)]

            # Price filter (Finviz already filtered Over $3, add upper bound)
            if 'Price' in df.columns:
                df = df[(df['Price'] >= 3) & (df['Price'] <= 200)]

            if len(df) == 0:
                logger.warning("All tickers filtered out after post-processing")
                return []

            # ============================================================
            # COMPOSITE MOMENTUM RANKING
            # ============================================================
            df = self._calculate_composite_score(df)

            # Sort by composite score (highest first)
            df = df.sort_values('composite_score', ascending=False)

            # Get top tickers
            tickers = df['Ticker'].head(max_tickers).tolist()

            # Log diagnostics
            top_cols = ['Ticker', 'Price', 'Change', 'Volume', 'composite_score']
            available_cols = [c for c in top_cols if c in df.columns]
            top_10 = df.head(10)[available_cols]
            logger.info(f"Top 10 momentum candidates:\n{top_10.to_string()}")
            logger.info(
                f"Selected {len(tickers)} tickers by COMPOSITE SCORE "
                f"(from {len(df)} after filters)"
            )

            # Cache the results
            self._cache = tickers
            self._cache_time = datetime.now()

            return tickers

        except ImportError:
            logger.error("finvizfinance not installed. Run: pip install finvizfinance")
            return []
        except Exception as e:
            logger.error(f"Finviz screener error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_static_universe(self) -> List[str]:
        """
        Get static small-cap universe as fallback.
        REVISED v3.0: Cleaned list - no excluded tickers, no duplicates.
        Only verified small-cap stocks with Float <= 75M.
        """
        static_list = [
            # Micro-float biotech (Float < 30M) - highest explosive potential
            'IBRX', 'PXMD', 'ACRS', 'AVXL', 'CNTB', 'IMMP', 'NNOX', 'OCGN',
            'PRAX', 'TPST', 'VRPX', 'YMAB', 'ZNTL',

            # Small-float biotech (Float 30-50M)
            'ABUS', 'ADVM', 'ALEC', 'ARQT', 'BCAB', 'CRSP', 'DCPH', 'LBPH',
            'MDXH', 'NBIX', 'PCVX', 'RLAY', 'SANA', 'TSHA', 'VKTX',

            # Micro-float tech (Float < 30M)
            'BRZE', 'CXAI', 'DUOL', 'GTLB', 'IONQ', 'PCT', 'SMCI', 'SOUN',
            'WEAV', 'XPOF',

            # Small-float tech/software (Float 30-60M)
            'AMPL', 'BMBL', 'BROS', 'CRDO', 'DOCN', 'ENVX', 'FLNC', 'LSPD',
            'MNDY', 'RAMP', 'RSKD', 'TASK', 'TOST', 'VERX',

            # Micro-float energy/EV (Float < 40M) - cleaned
            'BLNK', 'EVGO', 'PTRA', 'VFS', 'XOS',

            # Small-float consumer/retail (Float 30-60M)
            'BFIT', 'CAVA', 'FIGS', 'HIMS', 'LOVE', 'OUST', 'RVLV',
            'SNBR', 'VITL',

            # Small-float industrial/space (Float 30-70M) - cleaned
            'AEHR', 'ATKR', 'JOBY', 'LUNR', 'RKLB', 'RDW', 'SPCE',

            # High-volatility momentum plays (verified small float) - cleaned
            'BBAI', 'DNA', 'QS',
        ]

        # Filter out excluded & deduplicate
        seen = set()
        unique = []
        for t in static_list:
            if t not in self.EXCLUDED_TICKERS and t not in seen:
                seen.add(t)
                unique.append(t)

        logger.info(f"Static universe: {len(unique)} tickers")
        return unique

    def get_universe(self, use_finviz: bool = True, max_tickers: int = 200, force_refresh: bool = True) -> List[str]:
        """
        Get small-cap universe from best available source.

        Args:
            use_finviz: If True, try Finviz first, then fallback to static
            max_tickers: Maximum number of tickers to return
            force_refresh: If True, skip cache and get fresh data

        Returns:
            List of ticker symbols sorted by COMPOSITE MOMENTUM SCORE
        """
        if not force_refresh and self._cache and self._cache_time:
            cache_age = datetime.now() - self._cache_time
            if cache_age < timedelta(minutes=self.CACHE_DURATION_MINUTES):
                logger.info(f"Using cached universe ({len(self._cache)} tickers)")
                return self._cache[:max_tickers]

        logger.info("Fetching FRESH momentum-ranked universe (cache bypassed)")

        if use_finviz:
            finviz_tickers = self.get_finviz_universe(max_tickers)
            if len(finviz_tickers) >= 30:
                return finviz_tickers
            else:
                logger.warning(f"Finviz returned only {len(finviz_tickers)} tickers, merging with static")
                # Merge Finviz results with static for better coverage
                static = self.get_static_universe()
                merged = list(dict.fromkeys(finviz_tickers + static))  # Preserve order, dedup
                return merged[:max_tickers]

        return self.get_static_universe()

    def validate_ticker(self, ticker: str) -> bool:
        """Check if ticker is valid (not in exclusion list)."""
        return ticker not in self.EXCLUDED_TICKERS
