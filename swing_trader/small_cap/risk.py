"""
Small Cap Risk Management - Independent risk rules for small-cap momentum trades.
Completely independent from LargeCap risk management.
"""

import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .settings_config import SmallCapSettings

logger = logging.getLogger(__name__)


class SmallCapRisk:
    """
    Risk management for Small Cap Momentum Engine.

    v3.0 Improvements:
    - Type-specific stop loss caps (%8-12)
    - Realistic T1/T2 dual profit targets (reduced for better hit rate)
    - Clean position sizing: 1.5% risk, no artificial factor, only type cap
    """
    
    # Defaults documented in settings_config.SmallCapSettings; instance values set in __init__.

    def __init__(self, config: Dict = None, settings: Optional["SmallCapSettings"] = None):
        """Initialize SmallCapRisk from SmallCapSettings (file-backed)."""
        from .settings_config import load_settings

        self.config = config or {}
        s = settings if settings is not None else load_settings()
        self.MAX_RISK_PER_TRADE = s.max_risk_per_trade
        self.STOP_ATR_MULTIPLIER = s.stop_atr_multiplier
        self.MIN_STOP_PERCENT = s.min_stop_percent
        self.MAX_STOP_PERCENT = s.max_stop_percent_fallback
        self.MAX_HOLDING_DAYS = s.max_holding_days
        self.MAX_STOP_BY_TYPE = dict(s.max_stop_by_type)
        self.TYPE_ATR_MULTIPLIERS = dict(s.type_atr_multipliers)
        self.T2_ATR_RATIO = s.t2_atr_ratio
        self.TYPE_TARGET_CAPS = {
            k: (v.t1_max_pct, v.t2_max_pct) for k, v in s.type_target_caps.items()
        }
        self.TYPE_TARGETS = self.TYPE_TARGET_CAPS
        self.TYPE_POSITION_CAPS = dict(s.type_position_caps)
        self._risk_targets = s.risk_targets
        logger.info("SmallCapRisk initialized (v3.0 — realistic targets, clean sizing)")
    
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
        v5.1: Uses WIDER of ATR stop vs swing low (not tighter!) to give
        trades room to breathe. Capped per type, floored at MIN_STOP_PERCENT.
        """
        if df is None or len(df) < 5:
            return 0.0, "Insufficient data"
        
        try:
            current_close = df['Close'].iloc[-1]
            atr = self.calculate_atr(df)
            max_stop_pct = self.MAX_STOP_BY_TYPE.get(swing_type, self.MAX_STOP_PERCENT)
            
            # Method 1: 1.5 ATR stop
            atr_stop = current_close - (self.STOP_ATR_MULTIPLIER * atr)
            atr_pct = (current_close - atr_stop) / current_close if current_close > 0 else 0
            
            # Method 2: Recent swing low (last 5 days) with buffer
            swing_low = df['Low'].tail(5).min()
            swing_stop = swing_low * 0.995
            swing_pct = (current_close - swing_stop) / current_close if current_close > 0 else 0
            
            # WIDER stop — more room below entry (pick the stop further from current close)
            if atr_pct >= swing_pct:
                stop = atr_stop
                method = f"ATR Stop ({atr_pct*100:.1f}%)"
            else:
                stop = swing_stop
                method = f"Swing Low ({swing_pct*100:.1f}%)"
            
            # Apply type-specific maximum cap
            stop_pct = (current_close - stop) / current_close if current_close > 0 else 0
            if stop_pct > max_stop_pct:
                stop = current_close * (1 - max_stop_pct)
                method = f"Max Stop ({max_stop_pct*100:.0f}%)"
                stop_pct = (current_close - stop) / current_close if current_close > 0 else 0

            # Apply minimum floor (prevents whipsaw)
            if stop_pct < self.MIN_STOP_PERCENT:
                stop = current_close * (1 - self.MIN_STOP_PERCENT)
                method = f"Min Stop ({self.MIN_STOP_PERCENT*100:.0f}%)"
            
            return max(stop, 0.01), method
            
        except Exception as e:
            logger.error(f"Error calculating stop: {e}")
            return 0.0, str(e)
    
    def calculate_targets(
        self,
        entry_price: float,
        stop_loss: float,
        swing_type: str = 'A',
        atr: float = 0.0,
        quality_score: int = 0,
        regime: str = '',
    ) -> Tuple[float, float]:
        """
        Calculate T1 and T2 target prices — ATR-dynamic with quality boost.

        v5.0: Targets scale with the stock's own volatility (ATR) instead of
        fixed percentages.  Quality score widens targets for strong signals.
        Regime narrows T2 in cautious / bear markets.

        Falls back to fixed-pct table when ATR is unavailable.
        """
        t1_cap_pct, t2_cap_pct = self.TYPE_TARGET_CAPS.get(swing_type, (0.10, 0.18))

        rt = self._risk_targets
        if atr > 0 and entry_price > 0:
            mult = self.TYPE_ATR_MULTIPLIERS.get(swing_type, 1.8)

            q_boost = 1.0
            if quality_score >= rt.quality_tier_high:
                q_boost = rt.quality_boost_high
            elif quality_score >= rt.quality_tier_mid:
                q_boost = rt.quality_boost_mid

            t1_raw = entry_price + atr * mult * q_boost
            t2_mult = self.T2_ATR_RATIO
            if regime == "CAUTION":
                t2_mult = rt.t2_atr_mult_caution
            elif regime == "BEAR":
                t2_mult = rt.t2_atr_mult_bear
            t2_raw = entry_price + atr * mult * q_boost * t2_mult

            t1_cap = entry_price * (1 + t1_cap_pct)
            t2_cap = entry_price * (1 + t2_cap_pct)
            target_1 = min(t1_raw, t1_cap)
            target_2 = min(t2_raw, t2_cap)
        else:
            target_1 = entry_price * (1 + t1_cap_pct)
            target_2 = entry_price * (1 + t2_cap_pct)

        risk = entry_price - stop_loss
        min_target = entry_price + (risk * rt.min_reward_risk_multiple_t1)
        if target_1 < min_target:
            target_1 = min_target

        t2_gap = rt.t2_min_gap_vs_t1_bull
        if regime == "BEAR":
            t2_gap = rt.t2_min_gap_vs_t1_bear
        elif regime == "CAUTION":
            t2_gap = rt.t2_min_gap_vs_t1_caution
        if target_2 < target_1 * rt.t2_vs_t1_near_cap_floor:
            target_2 = target_1 * t2_gap

        t2_hard_cap = entry_price * (1 + t2_cap_pct)
        if target_2 > t2_hard_cap:
            target_2 = max(t2_hard_cap, target_1 * 1.02)

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

        v3.0: Clean risk-based sizing. No artificial position factor.
        - Risk per trade: 1.5% of portfolio
        - Only constraint: type-specific portfolio cap (S=15%, B=20%, C/A=25%)

        Returns:
            Tuple of (shares: int, risk_amount: float)
        """
        try:
            risk_per_share = entry_price - stop_loss
            if risk_per_share <= 0:
                return 0, 0.0

            # Step 1: Calculate shares from risk budget (1.5% of portfolio)
            max_risk = portfolio_value * self.MAX_RISK_PER_TRADE
            shares_by_risk = int(max_risk / risk_per_share)

            # Step 2: Apply type-specific portfolio cap (only real constraint)
            max_position_pct = self.TYPE_POSITION_CAPS.get(swing_type, 0.25)
            max_shares_by_cap = int((portfolio_value * max_position_pct) / entry_price)

            # Use the smaller of the two
            final_shares = min(shares_by_risk, max_shares_by_cap)

            # Ensure minimum position
            if final_shares < 1:
                final_shares = 1

            actual_risk = final_shares * risk_per_share

            return final_shares, actual_risk

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
        portfolio_value: float = 10000,
        regime: str = '',
    ) -> Dict:
        """
        Add risk management parameters to signal.

        v5.0: ATR-dynamic targets, quality boost, regime-aware T2.
        """
        from datetime import datetime, timedelta
        
        try:
            entry_price = signal.get('entry_price', df['Close'].iloc[-1])
            atr_percent = signal.get('atr_percent', 0.08)
            swing_type = signal.get('swing_type', 'A')
            quality_score = int(signal.get('quality_score') or signal.get('original_quality_score') or 0)
            
            # Calculate stop loss (type-specific cap)
            stop_loss, stop_method = self.calculate_stop_loss(df, swing_type)
            signal['stop_loss'] = round(stop_loss, 2)
            signal['stop_method'] = stop_method
            
            # ATR for dynamic targets
            atr_val = self.calculate_atr(df)

            # Calculate T1 and T2 targets (ATR-dynamic + quality + regime)
            target_1, target_2 = self.calculate_targets(
                entry_price, stop_loss, swing_type,
                atr=atr_val, quality_score=quality_score, regime=regime,
            )
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
            
            # v3.0: Respect engine's type-based hold days instead of overriding.
            # Engine already sets hold_days_min/max based on type + RSI + 5d return.
            # Only fill in if engine didn't set them (standalone scan_stock call).
            if not signal.get('hold_days_min'):
                hold_min, hold_max = self.calculate_expected_hold(df, atr_percent)
                signal['expected_hold_min'] = hold_min
                signal['expected_hold_max'] = hold_max
            else:
                signal['expected_hold_min'] = signal['hold_days_min']
                signal['expected_hold_max'] = signal['hold_days_max']
            
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
