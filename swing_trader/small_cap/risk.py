"""
Small Cap Risk Management - Independent risk rules for small-cap momentum trades.
Completely independent from LargeCap risk management.
"""

import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SmallCapRisk:
    """
    Risk management for Small Cap Momentum Engine.
    
    Rules:
    - Position size: 25-40% of large-cap allocation
    - Max risk per trade: 0.25-0.5%
    - Stop loss: 1-1.5 ATR (whichever is tighter)
    - Max holding period: 7 days
    - Target: Minimum 3R
    """
    
    # Risk constants - DIFFERENT from LargeCap
    POSITION_SIZE_FACTOR = 0.30       # 30% of normal position
    MAX_RISK_PER_TRADE = 0.005        # 0.5% of portfolio
    STOP_ATR_MULTIPLIER = 1.5         # 1.5 ATR stop (tighter than LargeCap)
    MAX_STOP_PERCENT = 0.10           # 10% max stop
    MAX_HOLDING_DAYS = 7              # 7 days max hold
    MIN_RR_RATIO = 3.0                # Minimum 3:1 risk/reward
    
    def __init__(self, config: Dict = None):
        """Initialize SmallCapRisk."""
        self.config = config or {}
        logger.info("SmallCapRisk initialized (high volatility rules)")
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR value."""
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
            
            return np.mean(tr[-period:])
            
        except Exception:
            return 0.0
    
    def calculate_stop_loss(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Calculate stop loss for small-cap trade.
        Uses tighter of: 1.5 ATR or recent swing low.
        """
        if df is None or len(df) < 5:
            return 0.0, "Insufficient data"
        
        try:
            current_close = df['Close'].iloc[-1]
            atr = self.calculate_atr(df)
            
            # Method 1: 1.5 ATR stop
            atr_stop = current_close - (self.STOP_ATR_MULTIPLIER * atr)
            atr_pct = (current_close - atr_stop) / current_close
            
            # Method 2: Recent swing low (last 5 days)
            swing_low = df['Low'].tail(5).min()
            swing_pct = (current_close - swing_low) / current_close
            
            # Use tighter stop (lower percentage)
            if atr_pct <= swing_pct and atr_pct <= self.MAX_STOP_PERCENT:
                stop = atr_stop
                method = f"ATR Stop ({atr_pct*100:.1f}%)"
            elif swing_pct <= atr_pct and swing_pct <= self.MAX_STOP_PERCENT:
                stop = swing_low * 0.995  # Slightly below swing low
                method = f"Swing Low ({swing_pct*100:.1f}%)"
            else:
                # Both too wide, use max stop percent
                stop = current_close * (1 - self.MAX_STOP_PERCENT)
                method = f"Max Stop ({self.MAX_STOP_PERCENT*100:.0f}%)"
            
            return max(stop, 0.01), method
            
        except Exception as e:
            logger.error(f"Error calculating stop: {e}")
            return 0.0, str(e)
    
    def calculate_target(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate target price based on minimum 3R.
        
        Target = Entry + (Risk × 3)
        """
        risk = entry_price - stop_loss
        target = entry_price + (risk * self.MIN_RR_RATIO)
        return target
    
    def calculate_position_size(
        self, 
        portfolio_value: float,
        entry_price: float,
        stop_loss: float
    ) -> Tuple[int, float]:
        """
        Calculate position size for small-cap trade.
        
        Uses 30% of normal allocation with max 0.5% risk per trade.
        
        Returns:
            Tuple of (shares: int, risk_amount: float)
        """
        try:
            # Calculate risk per share
            risk_per_share = entry_price - stop_loss
            if risk_per_share <= 0:
                return 0, 0.0
            
            # Calculate max risk amount (0.5% of portfolio)
            max_risk = portfolio_value * self.MAX_RISK_PER_TRADE
            
            # Calculate shares based on risk
            shares = int(max_risk / risk_per_share)
            
            # Apply position size factor (30% of normal)
            adjusted_shares = int(shares * self.POSITION_SIZE_FACTOR)
            
            # Calculate actual risk
            actual_risk = adjusted_shares * risk_per_share
            
            # Ensure minimum position
            if adjusted_shares < 1:
                adjusted_shares = 1
                actual_risk = risk_per_share
            
            return adjusted_shares, actual_risk
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0, 0.0
    
    def calculate_expected_hold(self, df: pd.DataFrame, atr_percent: float) -> Tuple[int, int]:
        """
        Calculate expected holding period based on volatility.
        
        Higher volatility = shorter expected hold.
        Max: 7 days
        """
        try:
            # Higher ATR% = shorter hold
            if atr_percent >= 0.15:  # 15%+
                return (2, 3)
            elif atr_percent >= 0.12:  # 12-15%
                return (2, 4)
            elif atr_percent >= 0.10:  # 10-12%
                return (3, 5)
            elif atr_percent >= 0.08:  # 8-10%
                return (3, 6)
            else:  # 6-8%
                return (4, 7)
                
        except Exception:
            return (3, 7)
    
    def add_risk_management(
        self, 
        signal: Dict, 
        df: pd.DataFrame,
        portfolio_value: float = 10000
    ) -> Dict:
        """
        Add risk management parameters to signal.
        
        Adds:
        - stop_loss
        - target_1 (3R minimum)
        - position_size (shares)
        - risk_amount
        - expected_hold (min, max)
        - max_hold_date
        - volatility_warning
        """
        from datetime import datetime, timedelta
        
        try:
            entry_price = signal.get('entry_price', df['Close'].iloc[-1])
            atr_percent = signal.get('atr_percent', 0.08)
            
            # Calculate stop loss
            stop_loss, stop_method = self.calculate_stop_loss(df)
            signal['stop_loss'] = round(stop_loss, 2)
            signal['stop_method'] = stop_method
            
            # Calculate target (minimum 3R)
            target = self.calculate_target(entry_price, stop_loss)
            signal['target_1'] = round(target, 2)
            
            # Calculate R:R
            risk = entry_price - stop_loss
            reward = target - entry_price
            signal['risk_reward'] = round(reward / risk, 1) if risk > 0 else 0
            
            # Calculate position size
            shares, risk_amount = self.calculate_position_size(
                portfolio_value, entry_price, stop_loss
            )
            signal['position_size'] = shares
            signal['risk_amount'] = round(risk_amount, 2)
            
            # Calculate expected hold
            hold_min, hold_max = self.calculate_expected_hold(df, atr_percent)
            signal['expected_hold_min'] = hold_min
            signal['expected_hold_max'] = hold_max
            
            # Max hold date
            signal_date = signal.get('date', datetime.now().strftime('%Y-%m-%d'))
            if isinstance(signal_date, str):
                signal_dt = datetime.strptime(signal_date, '%Y-%m-%d')
            else:
                signal_dt = signal_date
            
            max_hold_date = (signal_dt + timedelta(days=self.MAX_HOLDING_DAYS)).strftime('%Y-%m-%d')
            signal['max_hold_date'] = max_hold_date
            
            # Volatility warning - ALWAYS TRUE for small caps
            signal['volatility_warning'] = True
            
            return signal
            
        except Exception as e:
            logger.error(f"Error adding risk management: {e}")
            signal['volatility_warning'] = True
            return signal
