"""
Technical Levels Calculator - Resistance, Support, and Trendline Analysis.

Provides key price levels for professional swing trade analysis:
- Nearest resistance levels (swing highs)
- Support levels (swing lows)
- Descending trendline break detection
- Volume pattern description
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def find_pivot_highs(df: pd.DataFrame, lookback: int = 3) -> List[Tuple[int, float]]:
    """
    Find pivot highs (local maxima) in price data.
    A pivot high is a bar whose High is higher than `lookback` bars on each side.
    
    Returns list of (index_position, price) tuples.
    """
    highs = df['High'].values
    pivots = []
    
    for i in range(lookback, len(highs) - lookback):
        is_pivot = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_pivot = False
                break
        if is_pivot:
            pivots.append((i, highs[i]))
    
    return pivots


def find_pivot_lows(df: pd.DataFrame, lookback: int = 3) -> List[Tuple[int, float]]:
    """
    Find pivot lows (local minima) in price data.
    A pivot low is a bar whose Low is lower than `lookback` bars on each side.
    
    Returns list of (index_position, price) tuples.
    """
    lows = df['Low'].values
    pivots = []
    
    for i in range(lookback, len(lows) - lookback):
        is_pivot = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_pivot = False
                break
        if is_pivot:
            pivots.append((i, lows[i]))
    
    return pivots


def find_resistance_levels(df: pd.DataFrame, current_price: float, max_levels: int = 3) -> List[Dict]:
    """
    Find nearest resistance levels above current price.
    
    Uses pivot highs from last 60 bars to identify key overhead resistance.
    
    Returns list of dicts: {'price': float, 'distance_pct': float, 'strength': str}
    """
    try:
        pivot_highs = find_pivot_highs(df, lookback=3)
        
        # Filter: only levels ABOVE current price
        resistances = []
        for idx, price in pivot_highs:
            if price > current_price * 1.01:  # At least 1% above
                distance_pct = ((price / current_price) - 1) * 100
                resistances.append({
                    'price': round(price, 2),
                    'distance_pct': round(distance_pct, 1),
                    'bar_index': idx
                })
        
        # Sort by distance (closest first)
        resistances.sort(key=lambda x: x['distance_pct'])
        
        # Cluster nearby levels (within 2% of each other)
        clustered = []
        for r in resistances:
            if not clustered or abs(r['price'] - clustered[-1]['price']) / clustered[-1]['price'] > 0.02:
                # Determine strength based on how many times this level was tested
                r['strength'] = 'güçlü' if any(
                    abs(r['price'] - other['price']) / r['price'] < 0.02 
                    for other in resistances if other != r
                ) else 'orta'
                clustered.append(r)
        
        return clustered[:max_levels]
        
    except Exception as e:
        logger.debug(f"Error finding resistance levels: {e}")
        return []


def find_support_levels(df: pd.DataFrame, current_price: float, max_levels: int = 2) -> List[Dict]:
    """
    Find nearest support levels below current price.
    
    Returns list of dicts: {'price': float, 'distance_pct': float}
    """
    try:
        pivot_lows = find_pivot_lows(df, lookback=3)
        
        # Filter: only levels BELOW current price
        supports = []
        for idx, price in pivot_lows:
            if price < current_price * 0.99:  # At least 1% below
                distance_pct = ((price / current_price) - 1) * 100
                supports.append({
                    'price': round(price, 2),
                    'distance_pct': round(distance_pct, 1),
                    'bar_index': idx
                })
        
        # Sort by distance (closest first, least negative)
        supports.sort(key=lambda x: x['distance_pct'], reverse=True)
        
        # Cluster nearby levels
        clustered = []
        for s in supports:
            if not clustered or abs(s['price'] - clustered[-1]['price']) / clustered[-1]['price'] > 0.02:
                clustered.append(s)
        
        return clustered[:max_levels]
        
    except Exception as e:
        logger.debug(f"Error finding support levels: {e}")
        return []


def detect_trendline_break(df: pd.DataFrame, current_price: float) -> Dict:
    """
    Detect if price has recently broken above a descending trendline.
    
    Method:
    1. Find pivot highs over last 40 bars
    2. Check if there are 2+ declining pivot highs (descending trendline)
    3. Project trendline to current bar
    4. If current price > projected trendline → break detected
    
    Returns dict: {
        'detected': bool,
        'trendline_price': float (projected value at current bar),
        'break_pct': float (how far above trendline),
        'description': str
    }
    """
    result = {
        'detected': False,
        'trendline_price': 0.0,
        'break_pct': 0.0,
        'description': ''
    }
    
    try:
        # Need at least 20 bars
        if len(df) < 20:
            return result
        
        # Find pivot highs in the last 40 bars
        recent_df = df.tail(40).copy()
        pivot_highs = find_pivot_highs(recent_df, lookback=2)
        
        if len(pivot_highs) < 2:
            return result
        
        # Check for declining pivot highs (at least 2 declining)
        declining_pivots = []
        for i in range(len(pivot_highs)):
            for j in range(i + 1, len(pivot_highs)):
                if pivot_highs[j][1] < pivot_highs[i][1]:
                    declining_pivots.append((pivot_highs[i], pivot_highs[j]))
        
        if not declining_pivots:
            # No declining trendline found
            # Check for ASCENDING trendline (bullish continuation)
            ascending_pivots = []
            for i in range(len(pivot_highs)):
                for j in range(i + 1, len(pivot_highs)):
                    if pivot_highs[j][1] > pivot_highs[i][1]:
                        ascending_pivots.append((pivot_highs[i], pivot_highs[j]))
            
            if ascending_pivots:
                result['description'] = 'yükselen_trend'
            return result
        
        # Use the most recent declining pair
        p1, p2 = declining_pivots[-1]
        idx1, price1 = p1
        idx2, price2 = p2
        
        # Project trendline to current bar
        bar_diff = idx2 - idx1
        if bar_diff == 0:
            return result
        
        slope = (price2 - price1) / bar_diff
        current_bar = len(recent_df) - 1
        bars_from_p2 = current_bar - idx2
        projected_price = price2 + (slope * bars_from_p2)
        
        result['trendline_price'] = round(projected_price, 2)
        
        # Check if price broke above the projected trendline
        if current_price > projected_price and projected_price > 0:
            break_pct = ((current_price / projected_price) - 1) * 100
            result['detected'] = True
            result['break_pct'] = round(break_pct, 1)
            result['description'] = 'düşen_trend_kırıldı'
        else:
            result['description'] = 'düşen_trend_altında'
        
        return result
        
    except Exception as e:
        logger.debug(f"Error detecting trendline break: {e}")
        return result


def describe_volume_pattern(df: pd.DataFrame, volume_surge: float) -> str:
    """
    Generate human-readable volume pattern description.
    
    Analyzes recent volume bars to determine if volume is:
    - Surging (explosion today)
    - Building (increasing over days)
    - Declining
    - Normal
    """
    try:
        if len(df) < 5:
            return "Yetersiz veri"
        
        recent_vol = df['Volume'].tail(5).values
        avg_vol = df['Volume'].tail(20).mean()
        
        # Today vs yesterday
        vol_today = recent_vol[-1]
        vol_yesterday = recent_vol[-2]
        vol_2d_ago = recent_vol[-3]
        
        # Pattern detection
        if volume_surge >= 3.0:
            if vol_yesterday > avg_vol * 1.5:
                return f"Son 2 gündür güçlü hacim akışı var ({volume_surge:.1f}x ortalama). Kurumsal ilgi olabilir."
            else:
                return f"Bugün ani hacim patlaması ({volume_surge:.1f}x ortalama). Dikkatle takip et."
        
        elif volume_surge >= 2.0:
            if vol_yesterday > avg_vol * 1.3:
                return f"Hacim artarak devam ediyor ({volume_surge:.1f}x ortalama). Alıcılar güçlü."
            else:
                return f"Ortalamanın {volume_surge:.1f} katı hacim. Momentum başlıyor olabilir."
        
        elif volume_surge >= 1.5:
            # Check if building
            if vol_today > vol_yesterday > vol_2d_ago:
                return f"Hacim günden güne artıyor ({volume_surge:.1f}x). Sessiz birikim olabilir."
            else:
                return f"Hacim ortalamanın üstünde ({volume_surge:.1f}x)."
        
        else:
            if vol_today > vol_yesterday * 1.5:
                return "Bugün hacim artışı var ama henüz ortalamanın altında."
            else:
                return "Hacim normal seviyelerde."
        
    except Exception as e:
        logger.debug(f"Error describing volume pattern: {e}")
        return f"Hacim {volume_surge:.1f}x ortalama"


def calculate_technical_levels(df: pd.DataFrame, current_price: float, 
                                volume_surge: float = 1.0) -> Dict:
    """
    Calculate all technical levels for a stock.
    
    This is the main entry point — returns all levels in a single dict.
    
    Args:
        df: Price DataFrame with OHLCV columns
        current_price: Current close price
        volume_surge: Volume surge multiplier
    
    Returns:
        Dict with resistance_levels, support_levels, trendline, volume_pattern
    """
    result = {
        'resistance_levels': [],
        'support_levels': [],
        'nearest_resistance': None,
        'nearest_resistance_pct': 0.0,
        'nearest_support': None,
        'nearest_support_pct': 0.0,
        'trendline': detect_trendline_break(df, current_price),
        'volume_pattern': describe_volume_pattern(df, volume_surge)
    }
    
    # Find resistance levels
    resistances = find_resistance_levels(df, current_price)
    result['resistance_levels'] = resistances
    if resistances:
        result['nearest_resistance'] = resistances[0]['price']
        result['nearest_resistance_pct'] = resistances[0]['distance_pct']
    
    # Find support levels
    supports = find_support_levels(df, current_price)
    result['support_levels'] = supports
    if supports:
        result['nearest_support'] = supports[0]['price']
        result['nearest_support_pct'] = supports[0]['distance_pct']
    
    return result
