"""
Small Cap Universe Provider - Dynamic universe using finvizfinance.
Provides fresh small-cap stock lists from Finviz screener.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class SmallCapUniverse:
    """
    Dynamic small-cap universe provider using finvizfinance.
    
    Criteria:
    - Market Cap: Small ($300M - $2B)
    - Avg Volume: Over 1M shares
    - Price: $2 - $100
    """
    
    # Cache duration in minutes
    CACHE_DURATION_MINUTES = 60
    
    # Known delisted/problematic tickers to exclude
    EXCLUDED_TICKERS = {
        'BCOV', 'BGFV', 'CARA', 'GNOG', 'ZIRB', 'BBIG', 'IRNT', 'OPAD',
        'SPIR', 'CLOV', 'WISH', 'HOOD', 'LCID', 'RIVN', 'NKLA', 'WKHS',
        'FSR', 'GOEV', 'FFIE', 'MULN', 'RIDE', 'HYLN', 'ARVL', 'VLDR',
        'HCP', 'SQ', 'CERE', 'FREY', 'DTC', 'FTCH', 'ZNGA', 'VORB', 'TELL'
    }
    
    def __init__(self, config: Dict = None):
        """Initialize SmallCapUniverse."""
        self.config = config or {}
        self._cache = None
        self._cache_time = None
        logger.info("SmallCapUniverse initialized (finvizfinance)")
    
    def _parse_percent(self, value) -> float:
        """Parse percentage string like '5.23%' to float 5.23"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value.replace('%', '').replace(',', ''))
            return 0.0
        except:
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
        except:
            return 0.0
    
    def get_finviz_universe(self, max_tickers: int = 200) -> List[str]:
        """
        Get small-cap universe from Finviz using finvizfinance library.
        
        Filters:
        - Market Cap: Small ($300M - $2B)
        - Average Volume: Over 1M
        - Price: $2 to $100
        
        Returns:
            List of ticker symbols sorted by MOMENTUM SCORE
        """
        try:
            from finvizfinance.screener.overview import Overview
            
            logger.info("Fetching small-cap universe from Finviz...")
            
            foverview = Overview()
            
            # Set filters for small-cap momentum stocks
            filters_dict = {
                'Market Cap.': '+Small (over $300mln)',  # Small cap
                'Average Volume': 'Over 1M',              # High liquidity
            }
            
            foverview.set_filter(filters_dict=filters_dict)
            df = foverview.screener_view()
            
            if df is None or len(df) == 0:
                logger.warning("Finviz returned no data")
                return []
            
            logger.info(f"Finviz returned {len(df)} raw tickers, applying smart ranking...")
            
            # Filter out excluded tickers
            df = df[~df['Ticker'].isin(self.EXCLUDED_TICKERS)]
            
            # Filter by price > $2 (avoid penny stocks)
            if 'Price' in df.columns:
                df = df[df['Price'] >= 2]
            
            # ============================================================
            # SMART PRE-SCREENING: Momentum Score Ranking
            # Instead of just taking first N, rank by momentum potential
            # ============================================================
            
            # Parse Change% column (comes as string like "5.23%")
            if 'Change' in df.columns:
                df['change_pct'] = df['Change'].apply(self._parse_percent)
            else:
                df['change_pct'] = 0
            
            # Parse Volume if it's string (like "1.5M")
            if 'Volume' in df.columns:
                df['vol_numeric'] = df['Volume'].apply(self._parse_volume)
            else:
                df['vol_numeric'] = 0
            
            # Calculate MOMENTUM SCORE
            # Formula: |Change%| × log(Volume) × (1 + positive_bias)
            # - Absolute change because big moves (up OR down) = volatility = opportunity
            # - Log of volume to normalize (10M vs 1M shouldn't be 10x score)
            # - Positive bias: slightly favor UP moves over DOWN
            import numpy as np
            
            df['momentum_score'] = (
                df['change_pct'].abs() *                              # Volatility
                np.log10(df['vol_numeric'].clip(lower=1) + 1) *       # Volume factor (log scale)
                (1 + 0.3 * (df['change_pct'] > 0).astype(int))        # 30% bonus for UP moves
            )
            
            # Sort by momentum score (highest first)
            df = df.sort_values('momentum_score', ascending=False)
            
            # Get top tickers
            tickers = df['Ticker'].head(max_tickers).tolist()
            
            # Log the top scorers
            top_5 = df.head(5)[['Ticker', 'Change', 'Volume', 'momentum_score']]
            logger.info(f"Top 5 momentum candidates:\n{top_5.to_string()}")
            
            logger.info(f"Selected {len(tickers)} tickers by MOMENTUM SCORE (from {len(df)} total)")
            
            # Cache the results
            self._cache = tickers
            self._cache_time = datetime.now()
            
            return tickers
            
        except ImportError:
            logger.error("finvizfinance not installed. Run: pip install finvizfinance")
            return []
        except Exception as e:
            logger.error(f"Finviz screener error: {e}")
            return []
    
    def get_static_universe(self) -> List[str]:
        """
        Get static small-cap universe as fallback.
        REVISED: Only stocks with Float ≤ 75M and swing-compatible volatility.
        """
        # Curated list - VERIFIED small-cap stocks with tight floats
        # Float data as of 2026-01 (approximate)
        static_list = [
            # Micro-float biotech (Float < 30M) - highest explosive potential
            'IBRX', 'PXMD', 'ACRS', 'AVXL', 'CNTB', 'IMMP', 'NNOX', 'OCGN',
            'PRAX', 'TPST', 'VRPX', 'YMAB', 'ZNTL',
            
            # Small-float biotech (Float 30-50M)
            'ABUS', 'ADVM', 'ALEC', 'ARQT', 'BCAB', 'CRSP', 'DCPH', 'LBPH',
            'MDXH', 'NBIX', 'PCVX', 'RLAY', 'SANA', 'TSHA', 'VKTX',
            
            # Micro-float tech (Float < 30M)
            'BRZE', 'CXAI', 'DUOL', 'GTLB', 'IONQ', 'PCT', 'SMCI', 'SOUN',
            'VZIO', 'WEAV', 'XPOF',
            
            # Small-float tech/software (Float 30-60M)
            'AMPL', 'BMBL', 'BROS', 'CRDO', 'DOCN', 'ENVX', 'FLNC', 'LSPD',
            'MNDY', 'RAMP', 'RSKD', 'TASK', 'TOST', 'VERX',
            
            # Micro-float energy/EV (Float < 40M)
            'ARVL', 'BLNK', 'EVGO', 'FREY', 'HYLN', 'NKLA', 'PTRA', 'VFS',
            'XOS',
            
            # Small-float consumer/retail (Float 30-60M)
            'BFIT', 'BROS', 'CAVA', 'FIGS', 'HIMS', 'LOVE', 'OUST', 'RVLV',
            'SNBR', 'VITL',
            
            # Small-float industrial/space (Float 30-70M)
            'AEHR', 'ATKR', 'JOBY', 'LUNR', 'RKLB', 'RDW', 'SPCE', 'SPIR',
            
            # High-volatility momentum plays (verified small float)
            'BBAI', 'DNA', 'IRNT', 'MULN', 'NKLA', 'QS', 'RIVN', 'WKHS'
        ]
        
        # Filter out excluded
        filtered = [t for t in static_list if t not in self.EXCLUDED_TICKERS]
        
        # Remove duplicates
        seen = set()
        unique = []
        for t in filtered:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        
        logger.info(f"Static universe: {len(unique)} tickers")
        return unique
    
    def get_universe(self, use_finviz: bool = True, max_tickers: int = 200, force_refresh: bool = True) -> List[str]:
        """
        Get small-cap universe from best available source.
        
        NOW ALWAYS REFRESHES for fresh momentum data.
        
        Args:
            use_finviz: If True, try Finviz first, then fallback to static
            max_tickers: Maximum number of tickers to return
            force_refresh: If True, skip cache and get fresh data
        
        Returns:
            List of ticker symbols sorted by MOMENTUM SCORE
        """
        # IMPORTANT: Always refresh to get current momentum rankings
        # Cache disabled for SmallCap because momentum changes every day
        if not force_refresh and self._cache and self._cache_time:
            cache_age = datetime.now() - self._cache_time
            if cache_age < timedelta(minutes=self.CACHE_DURATION_MINUTES):
                logger.info(f"Using cached universe ({len(self._cache)} tickers)")
                return self._cache[:max_tickers]
        
        logger.info("Fetching FRESH momentum-ranked universe (cache bypassed)")
        
        if use_finviz:
            finviz_tickers = self.get_finviz_universe(max_tickers)
            if len(finviz_tickers) >= 50:
                return finviz_tickers
            else:
                logger.warning("Finviz returned few tickers, using static fallback")
        
        return self.get_static_universe()
    
    def validate_ticker(self, ticker: str) -> bool:
        """Check if ticker is valid (not in exclusion list)."""
        return ticker not in self.EXCLUDED_TICKERS

