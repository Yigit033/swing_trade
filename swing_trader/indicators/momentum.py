"""
Momentum indicators module: RSI, MACD, Stochastic calculations.
"""

import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI is a momentum oscillator (0-100):
    - 0-30: Oversold (potential buy)
    - 30-70: Neutral
    - 70-100: Overbought (potential sell)
    
    Args:
        prices: Series of prices (typically Close)
        period: RSI period (default: 14)
    
    Returns:
        Series with RSI values
    
    Raises:
        ValueError: If period <= 0
    
    Example:
        >>> rsi = calculate_rsi(df['Close'], 14)
        >>> print(f"Current RSI: {rsi.iloc[-1]:.2f}")
    """
    if period <= 0:
        raise ValueError("RSI period must be positive")
    
    if len(prices) < period + 1:
        logger.warning(f"Insufficient data for RSI({period}): {len(prices)} bars")
        return pd.Series([np.nan] * len(prices), index=prices.index)
    
    try:
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # Calculate average gain and loss using EMA
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}", exc_info=True)
        return pd.Series([np.nan] * len(prices), index=prices.index)


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD helps identify trend changes and momentum:
    - MACD line: Fast EMA - Slow EMA
    - Signal line: EMA of MACD line
    - Histogram: MACD line - Signal line
    
    Buy signal: MACD crosses above signal (histogram > 0)
    Sell signal: MACD crosses below signal (histogram < 0)
    
    Args:
        prices: Series of prices (typically Close)
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    
    Example:
        >>> macd, signal, hist = calculate_macd(df['Close'], 12, 26, 9)
        >>> df['MACD'] = macd
        >>> df['MACD_signal'] = signal
        >>> df['MACD_hist'] = hist
    """
    if len(prices) < slow + signal:
        logger.warning(f"Insufficient data for MACD({fast},{slow},{signal})")
        nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
        return nan_series, nan_series, nan_series
    
    try:
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}", exc_info=True)
        nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
        return nan_series, nan_series, nan_series


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Stochastic compares closing price to price range (0-100):
    - 0-20: Oversold
    - 20-80: Neutral
    - 80-100: Overbought
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period (default: 14)
        d_period: %D smoothing period (default: 3)
    
    Returns:
        Tuple of (%K line, %D line)
    
    Example:
        >>> stoch_k, stoch_d = calculate_stochastic(
        ...     df['High'], df['Low'], df['Close'], 14, 3
        ... )
        >>> df['Stoch_K'] = stoch_k
        >>> df['Stoch_D'] = stoch_d
    """
    if len(close) < k_period + d_period:
        logger.warning(f"Insufficient data for Stochastic({k_period},{d_period})")
        nan_series = pd.Series([np.nan] * len(close), index=close.index)
        return nan_series, nan_series
    
    try:
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D (smoothed %K)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
        
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {e}", exc_info=True)
        nan_series = pd.Series([np.nan] * len(close), index=close.index)
        return nan_series, nan_series


def calculate_momentum_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Calculate all momentum indicators for a DataFrame.
    
    Adds the following columns to DataFrame:
    - RSI: Relative Strength Index
    - MACD: MACD line
    - MACD_signal: Signal line
    - MACD_hist: Histogram
    - Stoch_K: Stochastic %K
    - Stoch_D: Stochastic %D
    
    Args:
        df: DataFrame with OHLCV data (requires: High, Low, Close columns)
        config: Configuration dictionary with indicator parameters
    
    Returns:
        DataFrame with added indicator columns
    
    Raises:
        ValueError: If required columns are missing
    
    Example:
        >>> df = calculate_momentum_indicators(df, config)
        >>> print(df[['Close', 'RSI', 'MACD_hist']].tail())
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
        
        # Calculate RSI
        rsi_period = indicator_config.get('rsi_period', 14)
        df['RSI'] = calculate_rsi(df['Close'], period=rsi_period)
        
        # Calculate MACD
        macd_fast = indicator_config.get('macd_fast', 12)
        macd_slow = indicator_config.get('macd_slow', 26)
        macd_signal = indicator_config.get('macd_signal', 9)
        
        macd, signal, hist = calculate_macd(
            df['Close'],
            fast=macd_fast,
            slow=macd_slow,
            signal=macd_signal
        )
        df['MACD'] = macd
        df['MACD_signal'] = signal
        df['MACD_hist'] = hist
        
        # Calculate Stochastic
        stoch_k = indicator_config.get('stoch_k', 14)
        stoch_d = indicator_config.get('stoch_d', 3)
        
        stoch_k_line, stoch_d_line = calculate_stochastic(
            df['High'],
            df['Low'],
            df['Close'],
            k_period=stoch_k,
            d_period=stoch_d
        )
        df['Stoch_K'] = stoch_k_line
        df['Stoch_D'] = stoch_d_line
        
        logger.debug(f"Calculated momentum indicators for {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating momentum indicators: {e}", exc_info=True)
        return df

