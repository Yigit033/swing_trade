"""
Trend indicators module: EMA, ADX calculations.
"""

import logging
from typing import Dict
import pandas as pd
import numpy as np
import pandas_ta as ta

logger = logging.getLogger(__name__)


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        prices: Series of prices (typically Close)
        period: EMA period
    
    Returns:
        Series with EMA values
    
    Example:
        >>> ema_20 = calculate_ema(df['Close'], 20)
    """
    if len(prices) < period:
        logger.warning(f"Insufficient data for EMA({period}): {len(prices)} bars")
        return pd.Series([np.nan] * len(prices), index=prices.index)
    
    return prices.ewm(span=period, adjust=False).mean()


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).
    
    ADX measures trend strength (0-100):
    - 0-25: Weak or no trend
    - 25-50: Strong trend
    - 50-75: Very strong trend
    - 75-100: Extremely strong trend
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default: 14)
    
    Returns:
        Series with ADX values
    
    Example:
        >>> adx = calculate_adx(df['High'], df['Low'], df['Close'], 14)
    """
    if len(close) < period * 2:
        logger.warning(f"Insufficient data for ADX({period})")
        return pd.Series([np.nan] * len(close), index=close.index)
    
    try:
        # Use pandas_ta for ADX calculation
        adx_df = ta.adx(high=high, low=low, close=close, length=period)
        if adx_df is not None and f'ADX_{period}' in adx_df.columns:
            return adx_df[f'ADX_{period}']
        else:
            logger.warning("ADX calculation failed")
            return pd.Series([np.nan] * len(close), index=close.index)
    except Exception as e:
        logger.error(f"Error calculating ADX: {e}")
        return pd.Series([np.nan] * len(close), index=close.index)


def calculate_trend_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Calculate all trend indicators for a DataFrame.
    
    Adds the following columns to DataFrame:
    - EMA_20, EMA_50, EMA_200: Exponential Moving Averages
    - ADX: Average Directional Index
    
    Args:
        df: DataFrame with OHLCV data (requires: High, Low, Close columns)
        config: Configuration dictionary with indicator parameters
    
    Returns:
        DataFrame with added indicator columns
    
    Raises:
        ValueError: If required columns are missing
    
    Example:
        >>> df = calculate_trend_indicators(df, config)
        >>> print(df[['Close', 'EMA_20', 'EMA_50', 'ADX']].tail())
    """
    # Validate required columns
    required_cols = ['High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Need: {required_cols}")
    
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df
    
    try:
        indicator_config = config['indicators']
        
        # Calculate EMAs
        for period in indicator_config['ema_periods']:
            df[f'EMA_{period}'] = calculate_ema(df['Close'], period)
        
        # Calculate ADX
        adx_period = indicator_config.get('adx_period', 14)
        df['ADX'] = calculate_adx(
            df['High'],
            df['Low'],
            df['Close'],
            period=adx_period
        )
        
        logger.debug(f"Calculated trend indicators for {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating trend indicators: {e}", exc_info=True)
        return df


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Series of prices (typically Close)
        period: Period for moving average (default: 20)
        std_dev: Number of standard deviations (default: 2.0)
    
    Returns:
        Dictionary with 'upper', 'middle', 'lower' bands
    
    Example:
        >>> bb = calculate_bollinger_bands(df['Close'], 20, 2.0)
        >>> df['BB_upper'] = bb['upper']
        >>> df['BB_middle'] = bb['middle']
        >>> df['BB_lower'] = bb['lower']
    """
    if len(prices) < period:
        logger.warning(f"Insufficient data for Bollinger Bands({period})")
        nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
        return {'upper': nan_series, 'middle': nan_series, 'lower': nan_series}
    
    try:
        # Calculate middle band (SMA)
        middle = prices.rolling(window=period).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=period).std()
        
        # Calculate bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
        return {'upper': nan_series, 'middle': nan_series, 'lower': nan_series}


def calculate_support_resistance(
    high: pd.Series,
    low: pd.Series,
    period: int = 20
) -> Dict[str, pd.Series]:
    """
    Calculate support and resistance levels.
    
    Args:
        high: High prices
        low: Low prices
        period: Lookback period (default: 20)
    
    Returns:
        Dictionary with 'support' and 'resistance' levels
    
    Example:
        >>> levels = calculate_support_resistance(df['High'], df['Low'], 20)
        >>> df['Support_20'] = levels['support']
        >>> df['Resistance_20'] = levels['resistance']
    """
    if len(low) < period:
        logger.warning(f"Insufficient data for support/resistance({period})")
        nan_series = pd.Series([np.nan] * len(low), index=low.index)
        return {'support': nan_series, 'resistance': nan_series}
    
    try:
        support = low.rolling(window=period).min()
        resistance = high.rolling(window=period).max()
        
        return {
            'support': support,
            'resistance': resistance
        }
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        nan_series = pd.Series([np.nan] * len(low), index=low.index)
        return {'support': nan_series, 'resistance': nan_series}

