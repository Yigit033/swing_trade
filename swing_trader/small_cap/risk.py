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
    MAX_STOP_PERCENT = 0.15           # 15% max stop (was 10% — small-caps need wider room)
    MAX_HOLDING_DAYS = 14             # 14 days max hold (was 7 - proper swing territory)
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
        Max: 14 days (was 7 - extended for proper swing trades)
        """
        try:
            # Higher ATR% = shorter hold
            if atr_percent >= 0.15:  # 15%+ very volatile
                return (2, 5)
            elif atr_percent >= 0.12:  # 12-15%
                return (3, 7)
            elif atr_percent >= 0.10:  # 10-12%
                return (4, 10)
            elif atr_percent >= 0.08:  # 8-10%
                return (5, 12)
            else:  # <8% calmer stocks
                return (5, 14)
                
        except Exception:
            return (4, 10)
    
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
    
    # ================================================================
    # SENIOR TRADER v2.1: TRAILING STOP & PYRAMID
    # ================================================================
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        initial_stop: float,
        atr: float,
        max_trail_atr: int = 4
    ) -> Dict:
        """
        Calculate trailing stop based on unrealized gains.
        
        SENIOR TRADER LOGIC:
        - Initial stop = entry - 1.5 ATR
        - Trail: Every +1 ATR gain, move stop +1 ATR
        - Max trail: 4 ATR from current price
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            initial_stop: Initial stop loss
            atr: Current ATR value (not percent)
            max_trail_atr: Maximum trail distance in ATR units
        
        Returns:
            {
                'trailing_stop': float,
                'atr_gain': float,  # How many ATRs gained
                'stop_moved': bool,
                'trail_distance_atr': float
            }
        """
        result = {
            'trailing_stop': initial_stop,
            'atr_gain': 0.0,
            'stop_moved': False,
            'trail_distance_atr': 0.0
        }
        
        if atr <= 0 or current_price <= entry_price:
            return result
        
        try:
            # Calculate gain in ATR units
            unrealized_gain = current_price - entry_price
            atr_gain = unrealized_gain / atr
            result['atr_gain'] = round(atr_gain, 2)
            
            # Move stop if gained at least 1 ATR
            if atr_gain >= 1.0:
                # Calculate new stop: initial + (floor(atr_gain) * ATR)
                atr_steps = int(atr_gain)  # Floor
                new_stop = initial_stop + (atr_steps * atr)
                
                # Cap trail distance at max_trail_atr below current price
                min_stop = current_price - (max_trail_atr * atr)
                
                # Use whichever is higher (tighter)
                result['trailing_stop'] = max(new_stop, min_stop)
                result['stop_moved'] = True
                result['trail_distance_atr'] = round(
                    (current_price - result['trailing_stop']) / atr, 1
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {e}")
            return result
    
    def calculate_pyramid(
        self,
        entry_price: float,
        current_price: float,
        current_position_size: int,
        portfolio_value: float,
        pyramid_count: int = 0,
        max_pyramids: int = 3
    ) -> Dict:
        """
        Calculate pyramid entry based on unrealized gains.
        
        SENIOR TRADER LOGIC:
        - Trigger: +10% unrealized gain
        - Add: 50% of current position size
        - Max: 3 pyramid entries
        
        Args:
            entry_price: Average entry price
            current_price: Current market price
            current_position_size: Current shares held
            portfolio_value: Total portfolio value
            pyramid_count: Number of pyramids already done
            max_pyramids: Maximum pyramid entries allowed
        
        Returns:
            {
                'should_pyramid': bool,
                'pyramid_size': int,  # Shares to add
                'new_total_position': int,
                'unrealized_pct': float,
                'reason': str
            }
        """
        result = {
            'should_pyramid': False,
            'pyramid_size': 0,
            'new_total_position': current_position_size,
            'unrealized_pct': 0.0,
            'reason': ''
        }
        
        if current_position_size <= 0 or current_price <= entry_price:
            result['reason'] = 'No position or in loss'
            return result
        
        try:
            # Calculate unrealized gain percentage
            unrealized_pct = ((current_price - entry_price) / entry_price) * 100
            result['unrealized_pct'] = round(unrealized_pct, 2)
            
            # Check if already at max pyramids
            if pyramid_count >= max_pyramids:
                result['reason'] = f'Max pyramids reached ({max_pyramids})'
                return result
            
            # Check if threshold met (+10%)
            if unrealized_pct >= 10.0:
                # Calculate pyramid size (50% of current)
                pyramid_shares = int(current_position_size * 0.5)
                
                if pyramid_shares >= 1:
                    # Check position doesn't exceed portfolio limits
                    new_position_value = (current_position_size + pyramid_shares) * current_price
                    max_position_value = portfolio_value * 0.25  # 25% max per position
                    
                    if new_position_value <= max_position_value:
                        result['should_pyramid'] = True
                        result['pyramid_size'] = pyramid_shares
                        result['new_total_position'] = current_position_size + pyramid_shares
                        result['reason'] = f'+{unrealized_pct:.1f}% gain - pyramid #{pyramid_count + 1}'
                    else:
                        result['reason'] = 'Would exceed position limit'
                else:
                    result['reason'] = 'Position too small to pyramid'
            else:
                result['reason'] = f'Need +10%, currently at +{unrealized_pct:.1f}%'
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating pyramid: {e}")
            result['reason'] = str(e)
            return result
    
    def get_position_management(
        self,
        entry_price: float,
        current_price: float,
        initial_stop: float,
        current_position_size: int,
        portfolio_value: float,
        atr: float,
        pyramid_count: int = 0
    ) -> Dict:
        """
        Get complete position management recommendation.
        
        Combines trailing stop and pyramid calculations.
        
        Returns:
            {
                'action': str,  # 'HOLD', 'TRAIL', 'PYRAMID', 'EXIT'
                'trailing_stop': float,
                'should_pyramid': bool,
                'pyramid_size': int,
                'unrealized_pct': float,
                'recommendation': str
            }
        """
        # Get trailing stop
        trail_result = self.calculate_trailing_stop(
            entry_price, current_price, initial_stop, atr
        )
        
        # Get pyramid recommendation
        pyramid_result = self.calculate_pyramid(
            entry_price, current_price, current_position_size,
            portfolio_value, pyramid_count
        )
        
        # Determine action
        if current_price <= trail_result['trailing_stop']:
            action = 'EXIT'
            recommendation = f"STOP HIT at ${trail_result['trailing_stop']:.2f}"
        elif pyramid_result['should_pyramid']:
            action = 'PYRAMID'
            recommendation = f"ADD {pyramid_result['pyramid_size']} shares at +{pyramid_result['unrealized_pct']:.1f}%"
        elif trail_result['stop_moved']:
            action = 'TRAIL'
            recommendation = f"Move stop to ${trail_result['trailing_stop']:.2f} (+{trail_result['atr_gain']:.1f} ATR gained)"
        else:
            action = 'HOLD'
            recommendation = f"Holding, +{pyramid_result['unrealized_pct']:.1f}% unrealized"
        
        return {
            'action': action,
            'trailing_stop': trail_result['trailing_stop'],
            'should_pyramid': pyramid_result['should_pyramid'],
            'pyramid_size': pyramid_result['pyramid_size'],
            'unrealized_pct': pyramid_result['unrealized_pct'],
            'atr_gain': trail_result['atr_gain'],
            'recommendation': recommendation
        }
