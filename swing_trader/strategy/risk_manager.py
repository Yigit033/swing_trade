"""
Risk management module for position sizing and portfolio risk control.
"""

import logging
from typing import Dict, Tuple, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages position sizing, stop-loss, take-profit, and portfolio risk.
    
    Attributes:
        config (Dict): Configuration dictionary
    """
    
    # Risk limits
    MAX_POSITION_SIZE = 0.20  # 20% of portfolio
    MAX_RISK_PER_TRADE = 0.02  # 2% risk per trade
    MAX_OPEN_POSITIONS = 5
    MAX_SECTOR_ALLOCATION = 0.30  # 30% per sector
    
    def __init__(self, config: Dict):
        """
        Initialize RiskManager.
        
        Args:
            config: Configuration dictionary with risk parameters
        """
        self.config = config
        
        # Load risk parameters from config
        risk_config = config.get('risk', {})
        self.MAX_RISK_PER_TRADE = risk_config.get('max_risk_per_trade', 0.02)
        self.MAX_POSITION_SIZE = risk_config.get('max_position_size', 0.20)
        self.MAX_OPEN_POSITIONS = risk_config.get('max_open_positions', 5)
        self.MAX_SECTOR_ALLOCATION = risk_config.get('max_sector_allocation', 0.30)
        self.stop_loss_atr_multiplier = risk_config.get('stop_loss_atr_multiplier', 2.0)
        self.take_profit_multipliers = risk_config.get('take_profit_multipliers', [1.5, 2.5])
        
        logger.info("RiskManager initialized")
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        atr: float,
        risk_percent: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate position size based on ATR and portfolio risk.
        
        Args:
            portfolio_value: Total portfolio value in dollars
            entry_price: Stock entry price
            atr: Average True Range (14-period)
            risk_percent: Risk per trade as decimal (default: from config)
        
        Returns:
            Dictionary containing:
                - shares: Number of shares to buy
                - entry_price: Entry price per share
                - stop_loss: Stop loss price
                - risk_amount: Total risk amount in dollars
                - position_value: Total position value
                - risk_percent: Actual risk percentage
        
        Raises:
            ValueError: If portfolio_value, entry_price, or atr <= 0
        
        Example:
            >>> rm = RiskManager(config)
            >>> position = rm.calculate_position_size(10000, 100, 2.5)
            >>> print(f"Buy {position['shares']} shares at ${position['entry_price']:.2f}")
            >>> print(f"Stop loss: ${position['stop_loss']:.2f}")
        """
        # Validation
        if portfolio_value <= 0:
            raise ValueError("Portfolio value must be positive")
        if entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if atr <= 0:
            raise ValueError("ATR must be positive")
        
        if risk_percent is None:
            risk_percent = self.MAX_RISK_PER_TRADE
        
        try:
            # Calculate stop loss distance and price
            stop_loss_distance = atr * self.stop_loss_atr_multiplier
            stop_loss_price = entry_price - stop_loss_distance
            
            # Ensure stop loss is positive
            if stop_loss_price <= 0:
                stop_loss_price = entry_price * 0.90  # 10% stop loss as fallback
                stop_loss_distance = entry_price - stop_loss_price
            
            # Calculate risk per share
            risk_per_share = stop_loss_distance
            
            # Calculate maximum risk amount
            max_risk_amount = portfolio_value * risk_percent
            
            # Calculate position size based on risk
            shares = int(max_risk_amount / risk_per_share)
            
            # Calculate position value
            position_value = shares * entry_price
            
            # Check maximum position size limit
            max_position_value = portfolio_value * self.MAX_POSITION_SIZE
            if position_value > max_position_value:
                # Reduce position size to meet limit
                shares = int(max_position_value / entry_price)
                position_value = shares * entry_price
            
            # Ensure at least 1 share
            if shares < 1:
                shares = 1
                position_value = entry_price
            
            # Calculate actual risk
            actual_risk_amount = shares * risk_per_share
            actual_risk_percent = actual_risk_amount / portfolio_value
            
            result = {
                'shares': shares,
                'entry_price': entry_price,
                'stop_loss': stop_loss_price,
                'risk_amount': actual_risk_amount,
                'position_value': position_value,
                'risk_percent': actual_risk_percent,
                'position_size_percent': position_value / portfolio_value
            }
            
            logger.debug(f"Position size calculated: {shares} shares, ${position_value:.2f} value")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            raise
    
    def calculate_take_profit_levels(
        self,
        entry_price: float,
        stop_loss: float
    ) -> Dict[str, float]:
        """
        Calculate take profit levels based on risk/reward ratios.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
        
        Returns:
            Dictionary with take profit levels
        
        Example:
            >>> rm = RiskManager(config)
            >>> targets = rm.calculate_take_profit_levels(100, 95)
            >>> print(f"Target 1: ${targets['target_1']:.2f}")
            >>> print(f"Target 2: ${targets['target_2']:.2f}")
        """
        try:
            risk_distance = entry_price - stop_loss
            
            targets = {}
            for i, multiplier in enumerate(self.take_profit_multipliers, 1):
                targets[f'target_{i}'] = entry_price + (risk_distance * multiplier)
            
            return targets
            
        except Exception as e:
            logger.error(f"Error calculating take profit levels: {e}")
            return {}
    
    def add_risk_management_to_signal(
        self,
        signal: Dict,
        portfolio_value: float
    ) -> Dict:
        """
        Add risk management parameters to a trading signal.
        
        Args:
            signal: Signal dictionary (must contain entry_price and atr)
            portfolio_value: Current portfolio value
        
        Returns:
            Signal dictionary with added risk management fields
        
        Example:
            >>> rm = RiskManager(config)
            >>> signal = rm.add_risk_management_to_signal(signal, 10000)
            >>> print(f"Buy {signal['shares']} shares")
            >>> print(f"Stop: ${signal['stop_loss']:.2f}, Target: ${signal['target_1']:.2f}")
        """
        try:
            entry_price = signal['entry_price']
            atr = signal.get('atr', 0)
            
            if atr <= 0:
                logger.warning(f"Invalid ATR for {signal['ticker']}, using 2% of price")
                atr = entry_price * 0.02
            
            # Calculate position size
            position = self.calculate_position_size(portfolio_value, entry_price, atr)
            
            # Calculate take profit levels
            targets = self.calculate_take_profit_levels(entry_price, position['stop_loss'])
            
            # Add to signal
            signal.update(position)
            signal.update(targets)
            
            logger.debug(f"Added risk management to {signal['ticker']}")
            return signal
            
        except Exception as e:
            logger.error(f"Error adding risk management: {e}", exc_info=True)
            return signal
    
    def validate_trade_risk(
        self,
        signal: Dict,
        portfolio: Dict
    ) -> Tuple[bool, str]:
        """
        Validate if trade meets all risk criteria.
        
        Args:
            signal: Signal dictionary with risk management fields
            portfolio: Portfolio dictionary with:
                - total_value: Total portfolio value
                - open_positions: List of open position dictionaries
                - sector_allocations: Dictionary of sector -> allocation
        
        Returns:
            Tuple of (is_valid, reason)
        
        Example:
            >>> rm = RiskManager(config)
            >>> is_valid, reason = rm.validate_trade_risk(signal, portfolio)
            >>> if not is_valid:
            ...     print(f"Trade rejected: {reason}")
        """
        try:
            total_value = portfolio.get('total_value', 0)
            open_positions = portfolio.get('open_positions', [])
            sector_allocations = portfolio.get('sector_allocations', {})
            
            # Check position size
            position_size_pct = signal.get('position_value', 0) / total_value
            if position_size_pct > self.MAX_POSITION_SIZE:
                return False, f"Position size {position_size_pct:.1%} exceeds {self.MAX_POSITION_SIZE:.1%} limit"
            
            # Check risk amount
            risk_pct = signal.get('risk_amount', 0) / total_value
            if risk_pct > self.MAX_RISK_PER_TRADE:
                return False, f"Risk {risk_pct:.1%} exceeds {self.MAX_RISK_PER_TRADE:.1%} limit"
            
            # Check open positions
            if len(open_positions) >= self.MAX_OPEN_POSITIONS:
                return False, f"Maximum {self.MAX_OPEN_POSITIONS} open positions reached"
            
            # Check sector allocation
            sector = signal.get('sector', 'Unknown')
            current_sector_allocation = sector_allocations.get(sector, 0)
            new_allocation = current_sector_allocation + position_size_pct
            
            if new_allocation > self.MAX_SECTOR_ALLOCATION:
                return False, f"Sector {sector} allocation {new_allocation:.1%} exceeds {self.MAX_SECTOR_ALLOCATION:.1%} limit"
            
            return True, "All risk checks passed"
            
        except Exception as e:
            logger.error(f"Error validating trade risk: {e}", exc_info=True)
            return False, f"Error during validation: {str(e)}"
    
    def check_exit_conditions(
        self,
        position: Dict,
        current_data: pd.Series,
        previous_data: pd.Series = None,
        current_date: str = None
    ) -> Dict:
        """
        Check if exit conditions are met for an open position.
        
        Args:
            position: Open position dictionary with entry details
            current_data: Current market data (Series with OHLC, indicators)
            previous_data: Previous bar data for crossover detection (FIXED)
            current_date: Current date string (YYYY-MM-DD) for time-based exit
        
        Returns:
            Dictionary with exit decision:
                - exit: Boolean, whether to exit
                - reason: String, reason for exit
                - price: Float, exit price
                - hold_days: Int, days held (if time-based exit)
        """
        try:
            from datetime import datetime
            
            exit_signal = False
            exit_reason = None
            exit_price = None
            
            current_close = current_data['Close']
            current_high = current_data['High']
            current_low = current_data['Low']
            
            # 0. TIME-BASED EXIT (check first, before price-based exits)
            time_stop_enabled = self.config.get('risk', {}).get('time_stop_enabled', True)
            max_holding_days = self.config.get('strategy', {}).get('max_holding_days', 20)
            
            if time_stop_enabled and current_date and 'entry_date' in position:
                try:
                    entry_dt = datetime.strptime(position['entry_date'], '%Y-%m-%d')
                    current_dt = datetime.strptime(current_date, '%Y-%m-%d')
                    hold_days = (current_dt - entry_dt).days
                    
                    if hold_days >= max_holding_days:
                        exit_signal = True
                        exit_reason = 'max_hold_time'
                        exit_price = current_close
                        logger.info(f"{position['ticker']}: Max holding time ({hold_days} days) exceeded at ${exit_price:.2f}")
                        return {'exit': exit_signal, 'reason': exit_reason, 'price': exit_price, 'hold_days': hold_days}
                except (ValueError, KeyError) as e:
                    logger.debug(f"Could not check time-based exit: {e}")
            
            # 1. Stop-loss check
            if current_low <= position['stop_loss']:
                exit_signal = True
                exit_reason = 'stop_loss'
                exit_price = position['stop_loss']
                logger.info(f"{position['ticker']}: Stop loss hit at ${exit_price:.2f}")
                return {'exit': exit_signal, 'reason': exit_reason, 'price': exit_price}
            
            # 2. Take-profit level 2 (highest target)
            if 'target_2' in position and position['target_2'] > 0 and current_high >= position['target_2']:
                exit_signal = True
                exit_reason = 'take_profit_2'
                exit_price = position['target_2']
                logger.info(f"{position['ticker']}: Target 2 reached at ${exit_price:.2f}")
                return {'exit': exit_signal, 'reason': exit_reason, 'price': exit_price}
            
            # 3. Take-profit level 1
            if 'target_1' in position and position['target_1'] > 0 and current_high >= position['target_1']:
                exit_signal = True
                exit_reason = 'take_profit_1'
                exit_price = position['target_1']
                logger.info(f"{position['ticker']}: Target 1 reached at ${exit_price:.2f}")
                return {'exit': exit_signal, 'reason': exit_reason, 'price': exit_price}
            
            # 4. RSI overbought
            rsi_exit = self.config['strategy'].get('rsi_exit', 70)
            if 'RSI' in current_data.index and not pd.isna(current_data['RSI']):
                if current_data['RSI'] > rsi_exit:
                    exit_signal = True
                    exit_reason = 'rsi_overbought'
                    exit_price = current_close
                    logger.info(f"{position['ticker']}: RSI overbought ({current_data['RSI']:.1f}) at ${exit_price:.2f}")
                    return {'exit': exit_signal, 'reason': exit_reason, 'price': exit_price}
            
            # 5. MACD bearish crossover (FIXED: now uses previous_data parameter)
            if previous_data is not None and 'MACD_hist' in current_data.index:
                curr_macd = current_data['MACD_hist']
                prev_macd = previous_data['MACD_hist'] if 'MACD_hist' in previous_data.index else None
                
                if curr_macd is not None and prev_macd is not None:
                    if not pd.isna(curr_macd) and not pd.isna(prev_macd):
                        # Bearish crossover: MACD histogram crosses from positive to negative
                        if curr_macd < 0 and prev_macd > 0:
                            exit_signal = True
                            exit_reason = 'macd_cross_down'
                            exit_price = current_close
                            logger.info(f"{position['ticker']}: MACD bearish crossover at ${exit_price:.2f}")
                            return {'exit': exit_signal, 'reason': exit_reason, 'price': exit_price}
            
            # No exit condition met
            return {'exit': False, 'reason': None, 'price': None}
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}", exc_info=True)
            return {'exit': False, 'reason': None, 'price': None}


