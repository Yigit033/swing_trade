"""
Unit tests for indicator calculations.
"""

import pytest
import pandas as pd
import numpy as np
from swing_trader.indicators.momentum import calculate_rsi
from swing_trader.indicators.trend import calculate_ema


def test_rsi_calculation():
    """Test RSI calculation with known values."""
    # Create sample price data
    prices = pd.Series([
        44, 44.5, 45, 43.5, 44, 44.5, 45.5, 46, 45.5, 46.5,
        46, 47, 47.5, 48, 47.5, 48.5, 49, 48.5, 49.5, 50
    ])
    
    # Calculate RSI
    rsi = calculate_rsi(prices, period=14)
    
    # Assertions
    assert rsi is not None
    assert len(rsi) == len(prices)
    assert 0 <= rsi.iloc[-1] <= 100
    assert not rsi.isna().all()


def test_rsi_edge_cases():
    """Test RSI with edge cases."""
    # All same values (flat line)
    flat_prices = pd.Series([50] * 20)
    rsi = calculate_rsi(flat_prices, period=14)
    # Should handle gracefully (may be NaN or 50)
    assert len(rsi) == len(flat_prices)
    
    # Insufficient data
    short_prices = pd.Series([50, 51, 52])
    rsi = calculate_rsi(short_prices, period=14)
    assert rsi.isna().all()


def test_ema_calculation():
    """Test EMA calculation."""
    prices = pd.Series(range(1, 21))  # 1 to 20
    
    ema = calculate_ema(prices, period=10)
    
    assert ema is not None
    assert len(ema) == len(prices)
    # EMA should be smoother than raw prices
    assert not ema.isna().all()


def test_ema_insufficient_data():
    """Test EMA with insufficient data."""
    prices = pd.Series([1, 2, 3])
    ema = calculate_ema(prices, period=10)
    
    # Should return series with NaN
    assert len(ema) == len(prices)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

