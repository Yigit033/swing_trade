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
    
    v2.3 Improvements:
    - Type-specific stop loss caps (%8-12 instead of flat %15)
    - Type-specific T1/T2 dual profit targets
    - Improved position sizing (%1.0 risk, %50 factor)
    """
    
    # ── CORE CONSTANTS ──
    POSITION_SIZE_FACTOR = 0.50       # 50% of normal position (was 30% — too conservative)
    MAX_RISK_PER_TRADE = 0.010        # 1.0% of portfolio (was 0.5% — too small)
    STOP_ATR_MULTIPLIER = 1.5         # 1.5 ATR stop
    MAX_HOLDING_DAYS = 14             # 14 days max hold
    
    # ── TYPE-SPECIFIC STOP LOSS CAPS ──
    # v2.3: Replaces flat MAX_STOP_PERCENT = 0.15
    MAX_STOP_BY_TYPE = {
        'C': 0.08,   # %8  — erken giriş, düşük volatilite beklentisi
        'A': 0.10,   # %10 — standard continuation
        'B': 0.10,   # %10 — momentum ama kontrollü
        'S': 0.12,   # %12 — squeeze'de geniş hareket normal
    }
    MAX_STOP_PERCENT = 0.10  # Default fallback
    
    # ── TYPE-SPECIFIC T1/T2 PROFIT TARGETS ──
    # v2.3: Replaces single MIN_RR_RATIO = 3.0
    # Format: (T1_percent, T2_percent)
    TYPE_TARGETS = {
        'S': (0.30, 0.60),   # 🔥 Squeeze:  T1 +%30, T2 +%60
        'B': (0.30, 0.50),   # 🚀 Momentum: T1 +%30, T2 +%50
        'C': (0.18, 0.30),   # ⭐ Erken:     T1 +%18, T2 +%30
        'A': (0.25, 0.40),   # 🐢 Devam:     T1 +%25, T2 +%40
    }
    
    # ── TYPE-SPECIFIC POSITION CAPS (max % of portfolio per position) ──
    TYPE_POSITION_CAPS = {
        'C': 0.25,   # %25 max portföy
        'A': 0.25,   # %25 max portföy
        'B': 0.20,   # %20 max portföy
        'S': 0.15,   # %15 max portföy (en riskli)
    }
    
    def __init__(self, config: Dict = None):
        """Initialize SmallCapRisk."""
        self.config = config or {}
        logger.info("SmallCapRisk initialized (v2.3 — type-specific risk rules)")
    
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
    
    def calculate_stop_loss(self, df: pd.DataFrame, swing_type: str = 'A') -> Tuple[float, str]:
        """
        Calculate stop loss for small-cap trade.
        v2.3: Type-specific max stop caps.
        Uses tighter of: 1.5 ATR or recent swing low, capped per type.
        """
        if df is None or len(df) < 5:
            return 0.0, "Insufficient data"
        
        try:
            current_close = df['Close'].iloc[-1]
            atr = self.calculate_atr(df)
            max_stop_pct = self.MAX_STOP_BY_TYPE.get(swing_type, self.MAX_STOP_PERCENT)
            
            # Method 1: 1.5 ATR stop
            atr_stop = current_close - (self.STOP_ATR_MULTIPLIER * atr)
            atr_pct = (current_close - atr_stop) / current_close
            
            # Method 2: Recent swing low (last 5 days)
            swing_low = df['Low'].tail(5).min()
            swing_pct = (current_close - swing_low) / current_close
            
            # Use tighter stop (lower percentage), capped by type-specific max
            if atr_pct <= swing_pct and atr_pct <= max_stop_pct:
                stop = atr_stop
                method = f"ATR Stop ({atr_pct*100:.1f}%)"
            elif swing_pct <= atr_pct and swing_pct <= max_stop_pct:
                stop = swing_low * 0.995  # Slightly below swing low
                method = f"Swing Low ({swing_pct*100:.1f}%)"
            else:
                # Both too wide, use type-specific max
                stop = current_close * (1 - max_stop_pct)
                method = f"Max Stop ({max_stop_pct*100:.0f}%)"
            
            return max(stop, 0.01), method
            
        except Exception as e:
            logger.error(f"Error calculating stop: {e}")
            return 0.0, str(e)
    
    def calculate_targets(self, entry_price: float, stop_loss: float, swing_type: str = 'A') -> Tuple[float, float]:
        """
        Calculate T1 and T2 target prices based on swing type.
        
        v2.3: Type-specific dual targets instead of flat 3R.
        
        Returns:
            Tuple of (target_1, target_2)
        """
        t1_pct, t2_pct = self.TYPE_TARGETS.get(swing_type, (0.25, 0.40))
        
        target_1 = entry_price * (1 + t1_pct)
        target_2 = entry_price * (1 + t2_pct)
        
        # Safety: T1 must be above entry, T2 must be above T1
        risk = entry_price - stop_loss
        min_target = entry_price + (risk * 2.0)  # Minimum 2R for T1
        
        if target_1 < min_target:
            target_1 = min_target
        if target_2 <= target_1:
            target_2 = target_1 * 1.15  # T2 at least 15% above T1
        
        return round(target_1, 2), round(target_2, 2)
    
    def calculate_target(self, entry_price: float, stop_loss: float) -> float:
        """
        Legacy single target (backward compatibility).
        Returns T1 from type-specific targets.
        """
        t1, _ = self.calculate_targets(entry_price, stop_loss, 'A')
        return t1
    
    def calculate_position_size(
        self, 
        portfolio_value: float,
        entry_price: float,
        stop_loss: float,
        swing_type: str = 'A'
    ) -> Tuple[int, float]:
        """
        Calculate position size for small-cap trade.
        
        v2.3: Improved sizing with type-specific portfolio caps.
        - Risk per trade: 1.0% of portfolio (was 0.5%)
        - Position factor: 50% (was 30%)
        - Type-specific max position: S=%15, B=%20, C/A=%25
        
        Returns:
            Tuple of (shares: int, risk_amount: float)
        """
        try:
            # Calculate risk per share
            risk_per_share = entry_price - stop_loss
            if risk_per_share <= 0:
                return 0, 0.0
            
            # Calculate max risk amount (1.0% of portfolio)
            max_risk = portfolio_value * self.MAX_RISK_PER_TRADE
            
            # Calculate shares based on risk
            shares = int(max_risk / risk_per_share)
            
            # Apply position size factor (50% of normal)
            adjusted_shares = int(shares * self.POSITION_SIZE_FACTOR)
            
            # Apply type-specific portfolio cap
            max_position_pct = self.TYPE_POSITION_CAPS.get(swing_type, 0.25)
            max_shares_by_cap = int((portfolio_value * max_position_pct) / entry_price)
            adjusted_shares = min(adjusted_shares, max_shares_by_cap)
            
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
        
        v2.3 Adds:
        - stop_loss (type-specific cap)
        - target_1 (T1 — ilk hedef)
        - target_2 (T2 — ikinci hedef) 
        - position_size (shares, type-specific cap)
        - risk_amount
        - expected_hold (min, max)
        - max_hold_date
        - volatility_warning
        """
        from datetime import datetime, timedelta
        
        try:
            entry_price = signal.get('entry_price', df['Close'].iloc[-1])
            atr_percent = signal.get('atr_percent', 0.08)
            swing_type = signal.get('swing_type', 'A')
            
            # Calculate stop loss (type-specific cap)
            stop_loss, stop_method = self.calculate_stop_loss(df, swing_type)
            signal['stop_loss'] = round(stop_loss, 2)
            signal['stop_method'] = stop_method
            
            # Calculate T1 and T2 targets (type-specific)
            target_1, target_2 = self.calculate_targets(entry_price, stop_loss, swing_type)
            signal['target_1'] = target_1
            signal['target_2'] = target_2
            
            # Calculate R:R based on T1
            risk = entry_price - stop_loss
            reward_t1 = target_1 - entry_price
            reward_t2 = target_2 - entry_price
            signal['risk_reward'] = round(reward_t1 / risk, 1) if risk > 0 else 0
            signal['risk_reward_t2'] = round(reward_t2 / risk, 1) if risk > 0 else 0
            
            # Target percentages for display
            signal['target_1_pct'] = round(((target_1 / entry_price) - 1) * 100, 1)
            signal['target_2_pct'] = round(((target_2 / entry_price) - 1) * 100, 1)
            signal['stop_loss_pct'] = round(((stop_loss / entry_price) - 1) * 100, 1)
            
            # Calculate position size (type-specific cap)
            shares, risk_amount = self.calculate_position_size(
                portfolio_value, entry_price, stop_loss, swing_type
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
