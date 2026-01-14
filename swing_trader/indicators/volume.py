"""
Volume indicators module: Volume analysis, OBV calculations.
"""

import logging
from typing import Dict
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Volume Moving Average.
    
    Args:
        volume: Series of volume values
        period: Moving average period (default: 20)
    
    Returns:
        Series with volume moving average
    
    Example:
        >>> volume_ma = calculate_volume_ma(df['Volume'], 20)
        >>> df['Volume_MA'] = volume_ma
    """
    if len(volume) < period:
        logger.warning(f"Insufficient data for Volume MA({period})")
        return pd.Series([np.nan] * len(volume), index=volume.index)
    
    try:
        return volume.rolling(window=period).mean()
    except Exception as e:
        logger.error(f"Error calculating Volume MA: {e}")
        return pd.Series([np.nan] * len(volume), index=volume.index)


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    OBV is a cumulative indicator that adds volume on up days and
    subtracts volume on down days. Rising OBV suggests accumulation,
    falling OBV suggests distribution.
    
    Args:
        close: Close prices
        volume: Volume values
    
    Returns:
        Series with OBV values
    
    Example:
        >>> obv = calculate_obv(df['Close'], df['Volume'])
        >>> df['OBV'] = obv
    """
    if len(close) < 2:
        logger.warning("Insufficient data for OBV calculation")
        return pd.Series([np.nan] * len(close), index=close.index)
    
    try:
        # Calculate price direction
        price_change = close.diff()
        
        # Create signed volume
        signed_volume = volume.copy()
        signed_volume[price_change < 0] = -signed_volume[price_change < 0]
        signed_volume[price_change == 0] = 0
        
        # Calculate cumulative OBV
        obv = signed_volume.cumsum()
        
        return obv
        
    except Exception as e:
        logger.error(f"Error calculating OBV: {e}", exc_info=True)
        return pd.Series([np.nan] * len(close), index=close.index)


def calculate_obv_slope(obv: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate OBV slope using linear regression.
    
    Positive slope indicates buying pressure, negative slope indicates
    selling pressure.
    
    Args:
        obv: OBV values
        period: Lookback period for slope calculation (default: 10)
    
    Returns:
        Series with OBV slope values
    
    Example:
        >>> obv = calculate_obv(df['Close'], df['Volume'])
        >>> obv_slope = calculate_obv_slope(obv, 10)
        >>> df['OBV_slope'] = obv_slope
    """
    if len(obv) < period:
        logger.warning(f"Insufficient data for OBV slope({period})")
        return pd.Series([np.nan] * len(obv), index=obv.index)
    
    try:
        slopes = []
        
        for i in range(len(obv)):
            if i < period - 1:
                slopes.append(np.nan)
            else:
                # Get window of OBV values
                y = obv.iloc[i - period + 1:i + 1].values
                x = np.arange(len(y))
                
                # Calculate linear regression slope
                if len(y) > 0 and not np.isnan(y).all():
                    slope, _, _, _, _ = stats.linregress(x, y)
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        
        return pd.Series(slopes, index=obv.index)
        
    except Exception as e:
        logger.error(f"Error calculating OBV slope: {e}", exc_info=True)
        return pd.Series([np.nan] * len(obv), index=obv.index)


def calculate_volume_surge(volume: pd.Series, volume_ma: pd.Series) -> pd.Series:
    """
    Calculate volume surge ratio (current volume / average volume).
    
    Ratios > 1.0 indicate above-average volume.
    
    Args:
        volume: Current volume values
        volume_ma: Volume moving average
    
    Returns:
        Series with volume surge ratios
    
    Example:
        >>> volume_ma = calculate_volume_ma(df['Volume'], 20)
        >>> surge = calculate_volume_surge(df['Volume'], volume_ma)
        >>> df['Volume_surge'] = surge
    """
    try:
        # Avoid division by zero
        surge = volume / volume_ma.replace(0, np.nan)
        return surge
    except Exception as e:
        logger.error(f"Error calculating volume surge: {e}")
        return pd.Series([np.nan] * len(volume), index=volume.index)


def calculate_volume_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Calculate all volume indicators for a DataFrame.
    
    Adds the following columns to DataFrame:
    - Volume_MA: Volume moving average
    - OBV: On-Balance Volume
    - OBV_slope: Linear regression slope of OBV
    - Volume_surge: Volume surge ratio
    
    Args:
        df: DataFrame with OHLCV data (requires: Close, Volume columns)
        config: Configuration dictionary with indicator parameters
    
    Returns:
        DataFrame with added indicator columns
    
    Raises:
        ValueError: If required columns are missing
    
    Example:
        >>> df = calculate_volume_indicators(df, config)
        >>> print(df[['Volume', 'Volume_MA', 'OBV', 'Volume_surge']].tail())
    """
    # Validate required columns
    required_cols = ['Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Need: {required_cols}")
    
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df
    
    try:
        indicator_config = config['indicators']
        
        # Calculate Volume MA
        volume_ma_period = indicator_config.get('volume_ma_period', 20)
        df['Volume_MA'] = calculate_volume_ma(df['Volume'], period=volume_ma_period)
        
        # Calculate OBV
        df['OBV'] = calculate_obv(df['Close'], df['Volume'])
        
        # Calculate OBV slope
        obv_slope_period = indicator_config.get('obv_slope_period', 10)
        df['OBV_slope'] = calculate_obv_slope(df['OBV'], period=obv_slope_period)
        
        # Calculate Volume surge
        df['Volume_surge'] = calculate_volume_surge(df['Volume'], df['Volume_MA'])
        
        logger.debug(f"Calculated volume indicators for {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating volume indicators: {e}", exc_info=True)
        return df


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    ATR measures volatility:
    - Higher ATR = Higher volatility
    - Lower ATR = Lower volatility
    
    Used for position sizing and stop-loss placement.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)
    
    Returns:
        Series with ATR values
    
    Example:
        >>> atr = calculate_atr(df['High'], df['Low'], df['Close'], 14)
        >>> df['ATR'] = atr
    """
    if len(close) < period + 1:
        logger.warning(f"Insufficient data for ATR({period})")
        return pd.Series([np.nan] * len(close), index=close.index)
    
    try:
        # Calculate True Range
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR as EMA of True Range
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
        
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}", exc_info=True)
        return pd.Series([np.nan] * len(close), index=close.index)

