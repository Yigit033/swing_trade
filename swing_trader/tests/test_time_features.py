"""
Unit tests for time-based signal features.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def test_holding_period_calculation():
    """Test that holding period is calculated correctly (not always 0)."""
    from swing_trader.backtesting.engine import BacktestEngine
    import yaml
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    engine = BacktestEngine(config)
    
    # Create a mock position
    position = {
        'ticker': 'TEST',
        'entry_date': '2024-01-01',
        'entry_price': 100.0,
        'shares': 10,
        'stop_loss': 95.0,
        'target_1': 110.0,
        'commission_paid': 1.0
    }
    
    # Add to open positions
    engine.open_positions.append(position)
    
    # Exit on a different date
    exit_date = '2024-01-15'
    engine.exit_position(position, 105.0, exit_date, 'test_exit')
    
    # Check that hold_days is correct (should be 14, not 0)
    assert len(engine.closed_trades) == 1
    trade = engine.closed_trades[0]
    assert trade['hold_days'] == 14, f"Expected 14 days, got {trade['hold_days']}"


def test_time_based_exit():
    """Test that positions exit after max_holding_days."""
    from swing_trader.strategy.risk_manager import RiskManager
    import yaml
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Override max holding days for test
    config['strategy']['max_holding_days'] = 5
    config['risk']['time_stop_enabled'] = True
    
    rm = RiskManager(config)
    
    # Create position
    position = {
        'ticker': 'TEST',
        'entry_date': '2024-01-01',
        'stop_loss': 95.0,
        'target_1': 110.0
    }
    
    # Create current data (price between stop and target - no exit yet)
    current_data = pd.Series({
        'Close': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'RSI': 50.0
    })
    
    # Test day 4 - should NOT exit
    exit_decision = rm.check_exit_conditions(position, current_data, current_date='2024-01-04')
    assert exit_decision['exit'] == False, "Should not exit on day 4"
    
    # Test day 5 - SHOULD exit (max_holding_days reached)
    exit_decision = rm.check_exit_conditions(position, current_data, current_date='2024-01-06')
    assert exit_decision['exit'] == True, "Should exit on day 6 (5 days held)"
    assert exit_decision['reason'] == 'max_hold_time'


def test_signal_has_time_metadata():
    """Test that generated signals include time-related metadata."""
    from swing_trader.strategy.signals import SignalGenerator
    import yaml
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    gen = SignalGenerator(config)
    
    # Create mock data with a signal
    dates = pd.date_range('2024-01-01', periods=250, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(95, 105, 250),
        'High': np.random.uniform(100, 110, 250),
        'Low': np.random.uniform(90, 100, 250),
        'Close': np.linspace(100, 150, 250),  # Uptrend
        'Volume': np.random.randint(1000000, 5000000, 250)
    })
    
    # Calculate indicators
    df = gen.calculate_all_indicators(df)
    
    # Try to generate signal (may or may not generate based on conditions)
    # For this test, we just verify the structure if a signal is generated
    signal = gen.generate_signal('TEST', df)
    
    # If signal was generated, check for time metadata
    if signal is not None:
        assert 'signal_date' in signal, "Signal should have signal_date"
        assert 'expiration_date' in signal, "Signal should have expiration_date"
        assert 'expected_hold_min' in signal, "Signal should have expected_hold_min"
        assert 'expected_hold_max' in signal, "Signal should have expected_hold_max"
        assert 'max_hold_date' in signal, "Signal should have max_hold_date"
        
        # Verify expiration is 3 days after signal date
        signal_dt = datetime.strptime(signal['signal_date'], '%Y-%m-%d')
        expiration_dt = datetime.strptime(signal['expiration_date'], '%Y-%m-%d')
        assert (expiration_dt - signal_dt).days == 3, "Expiration should be 3 days after signal"


def test_config_has_time_parameters():
    """Test that config.yaml has the new time parameters."""
    import yaml
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Check strategy section
    assert 'max_holding_days' in config['strategy'], "Missing max_holding_days"
    assert 'signal_expiration_days' in config['strategy'], "Missing signal_expiration_days"
    assert 'expected_hold_days_min' in config['strategy'], "Missing expected_hold_days_min"
    assert 'expected_hold_days_max' in config['strategy'], "Missing expected_hold_days_max"
    
    # Check risk section
    assert 'time_stop_enabled' in config['risk'], "Missing time_stop_enabled"
    
    # Verify default values
    assert config['strategy']['max_holding_days'] == 20
    assert config['strategy']['signal_expiration_days'] == 3
    assert config['risk']['time_stop_enabled'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
