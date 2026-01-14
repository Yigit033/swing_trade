"""
Backtesting engine for testing trading strategies on historical data.
UPDATED: Realistic execution with next-day entry, dynamic costs, and bias warnings.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ..strategy.signals import SignalGenerator
from ..strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies.
    
    Key Features:
        - Next-day OPEN entry (no lookahead bias)
        - Gap protection (skip if gaps past stop-loss)
        - Dynamic slippage based on ATR/volume
        - Spread modeling for realistic costs
        - Survivorship bias warnings
    """
    
    def __init__(self, config: Dict):
        """Initialize BacktestEngine."""
        self.config = config
        self.signal_generator = SignalGenerator(config)
        self.risk_manager = RiskManager(config)
        
        # Backtest parameters
        bt_config = config.get('backtesting', {})
        self.initial_capital = bt_config.get('initial_capital', 10000)
        self.commission = bt_config.get('commission_per_trade', 1.0)
        self.base_slippage = bt_config.get('slippage_percent', 0.001)
        self.use_dynamic_slippage = bt_config.get('use_dynamic_slippage', True)
        self.use_spread = bt_config.get('use_spread', True)
        
        # State
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.open_positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.pending_signals = []  # NEW: Signals waiting for next-day execution
        self.skipped_trades = []   # NEW: Trades skipped due to gap
        
        logger.info(f"BacktestEngine initialized with ${self.initial_capital:,.2f}")
        logger.info(f"Dynamic slippage: {self.use_dynamic_slippage}, Spread: {self.use_spread}")
    
    def reset(self):
        """Reset backtest state to initial values."""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.open_positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.pending_signals = []
        self.skipped_trades = []
        logger.info("Backtest state reset")
    
    def calculate_dynamic_slippage(self, ticker_data: pd.DataFrame, shares: int, is_buy: bool) -> float:
        """
        Calculate slippage based on ATR and volume.
        
        Higher volatility + lower liquidity = higher slippage
        """
        if not self.use_dynamic_slippage:
            return self.base_slippage
        
        try:
            # Get recent data
            recent = ticker_data.tail(20)
            if len(recent) < 5:
                return self.base_slippage
            
            price = recent['Close'].iloc[-1]
            avg_volume = recent['Volume'].mean()
            
            # Calculate ATR if not present
            if 'ATR' in recent.columns:
                atr = recent['ATR'].iloc[-1]
            else:
                high_low = recent['High'] - recent['Low']
                atr = high_low.mean()
            
            # Base slippage
            base = self.base_slippage  # 0.1%
            
            # Volume impact: if trade is >1% of daily volume, add slippage
            trade_value = shares * price
            daily_dollar_volume = avg_volume * price
            if daily_dollar_volume > 0:
                volume_ratio = trade_value / daily_dollar_volume
                volume_slip = min(volume_ratio * 0.5, 0.01)  # Cap at 1%
            else:
                volume_slip = 0.005  # 0.5% for unknown volume
            
            # Volatility impact: higher ATR = higher slippage
            if price > 0:
                atr_ratio = atr / price
                vol_slip = atr_ratio * 0.1  # 10% of ATR ratio
            else:
                vol_slip = 0.002
            
            total_slip = base + volume_slip + vol_slip
            return min(total_slip, 0.03)  # Cap at 3%
            
        except Exception as e:
            logger.debug(f"Error calculating dynamic slippage: {e}")
            return self.base_slippage
    
    def calculate_spread(self, ticker_data: pd.DataFrame) -> float:
        """
        Calculate bid-ask spread based on liquidity.
        
        Returns half-spread (applied to each side of trade).
        """
        if not self.use_spread:
            return 0.0
        
        try:
            recent = ticker_data.tail(20)
            if len(recent) < 5:
                return 0.001  # Default 0.1%
            
            price = recent['Close'].iloc[-1]
            avg_volume = recent['Volume'].mean()
            daily_dollar_volume = avg_volume * price
            
            # Spread based on liquidity
            if daily_dollar_volume > 100_000_000:  # >$100M = very liquid
                return 0.0001  # 0.01%
            elif daily_dollar_volume > 10_000_000:  # $10-100M
                return 0.0003  # 0.03%
            elif daily_dollar_volume > 1_000_000:   # $1-10M
                return 0.0005  # 0.05%
            else:
                return 0.0015  # 0.15%
                
        except Exception:
            return 0.0005
    
    def apply_slippage(self, price: float, is_buy: bool, slippage: float = None) -> float:
        """Apply slippage to execution price."""
        if slippage is None:
            slippage = self.base_slippage
        
        if is_buy:
            return price * (1 + slippage)
        else:
            return price * (1 - slippage)
    
    def apply_costs(self, price: float, is_buy: bool, ticker_data: pd.DataFrame, shares: int) -> float:
        """Apply all costs: slippage + spread."""
        # Dynamic slippage
        slippage = self.calculate_dynamic_slippage(ticker_data, shares, is_buy)
        
        # Spread (half applied to each side)
        spread = self.calculate_spread(ticker_data)
        
        # Combined impact
        if is_buy:
            return price * (1 + slippage + spread)
        else:
            return price * (1 - slippage - spread)
    
    def enter_position(self, signal: Dict, date: str, entry_price: float, ticker_data: pd.DataFrame) -> bool:
        """
        Enter a new position based on signal.
        
        Args:
            signal: Trading signal dictionary
            date: Entry date (next day after signal)
            entry_price: The OPEN price of entry day
            ticker_data: Historical data for cost calculation
        """
        try:
            # Add risk management to signal
            signal = self.risk_manager.add_risk_management_to_signal(
                signal,
                self.portfolio_value
            )
            
            shares = signal['shares']
            
            # Apply costs (slippage + spread)
            actual_entry_price = self.apply_costs(entry_price, is_buy=True, ticker_data=ticker_data, shares=shares)
            
            # Calculate costs
            position_value = shares * actual_entry_price
            total_cost = position_value + self.commission
            
            # Check if we have enough cash
            if total_cost > self.cash:
                logger.debug(f"Insufficient cash for {signal['ticker']}: need ${total_cost:.2f}, have ${self.cash:.2f}")
                return False
            
            # Create position
            position = {
                'ticker': signal['ticker'],
                'signal_date': signal.get('signal_date', date),  # When signal was generated
                'entry_date': date,  # When actually entered (next day)
                'entry_price': actual_entry_price,
                'shares': shares,
                'stop_loss': signal['stop_loss'],
                'target_1': signal.get('target_1', 0),
                'target_2': signal.get('target_2', 0),
                'position_value': position_value,
                'commission_paid': self.commission
            }
            
            # Update cash and positions
            self.cash -= total_cost
            self.open_positions.append(position)
            
            logger.info(f"ENTER: {signal['ticker']} - {shares} shares @ ${actual_entry_price:.2f} (signal: {signal.get('signal_date', 'N/A')}, entry: {date})")
            return True
            
        except Exception as e:
            logger.error(f"Error entering position: {e}", exc_info=True)
            return False
    
    def exit_position(self, position: Dict, exit_price: float, exit_date: str, exit_reason: str, ticker_data: pd.DataFrame = None) -> bool:
        """Exit an open position with realistic costs."""
        try:
            shares = position['shares']
            entry_price = position['entry_price']
            
            # Apply costs to exit price
            if ticker_data is not None:
                actual_exit_price = self.apply_costs(exit_price, is_buy=False, ticker_data=ticker_data, shares=shares)
            else:
                actual_exit_price = self.apply_slippage(exit_price, is_buy=False)
            
            # Calculate proceeds
            gross_proceeds = shares * actual_exit_price
            net_proceeds = gross_proceeds - self.commission
            
            # Calculate P&L
            entry_cost = shares * entry_price + position['commission_paid']
            pnl = net_proceeds - entry_cost
            pnl_percent = (pnl / entry_cost) * 100
            
            # Calculate holding period
            entry_dt = datetime.strptime(position['entry_date'], '%Y-%m-%d')
            exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
            hold_days = (exit_dt - entry_dt).days
            
            # Create trade record
            trade = {
                'ticker': position['ticker'],
                'signal_date': position.get('signal_date', position['entry_date']),
                'entry_date': position['entry_date'],
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': actual_exit_price,
                'shares': shares,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'hold_days': hold_days,
                'exit_reason': exit_reason,
                'commission_total': position['commission_paid'] + self.commission
            }
            
            # Update cash
            self.cash += net_proceeds
            
            # Remove from open positions
            self.open_positions.remove(position)
            
            # Add to closed trades
            self.closed_trades.append(trade)
            
            logger.info(f"EXIT: {position['ticker']} - {shares} shares @ ${actual_exit_price:.2f} | P&L: ${pnl:.2f} ({pnl_percent:.1f}%) | Reason: {exit_reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error exiting position: {e}", exc_info=True)
            return False
    
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value based on current prices."""
        positions_value = 0
        for position in self.open_positions:
            ticker = position['ticker']
            if ticker in current_prices:
                positions_value += position['shares'] * current_prices[ticker]
        
        self.portfolio_value = self.cash + positions_value
    
    def process_day(self, date: str, data_dict: Dict[str, pd.DataFrame], previous_date: str = None):
        """
        Process a single trading day.
        
        NEW EXECUTION MODEL:
        1. Execute pending signals from yesterday at today's OPEN
        2. Check exit conditions for open positions
        3. Generate new signals (add to pending for tomorrow)
        """
        try:
            # Get current prices and OPEN prices for today
            current_prices = {}
            open_prices = {}
            
            for ticker, df in data_dict.items():
                ticker_data = df[df['Date'] <= date]
                if not ticker_data.empty:
                    current_prices[ticker] = ticker_data.iloc[-1]['Close']
                    open_prices[ticker] = ticker_data.iloc[-1]['Open']
            
            # ============================================
            # STEP 1: Execute pending signals at today's OPEN
            # ============================================
            for pending in self.pending_signals.copy():
                ticker = pending['ticker']
                
                if ticker not in data_dict or ticker not in open_prices:
                    self.pending_signals.remove(pending)
                    continue
                
                ticker_df = data_dict[ticker]
                ticker_data = ticker_df[ticker_df['Date'] <= date]
                
                if ticker_data.empty:
                    self.pending_signals.remove(pending)
                    continue
                
                today_open = open_prices[ticker]
                stop_loss = pending['stop_loss']
                
                # GAP PROTECTION: If open gaps below stop-loss, skip trade
                if today_open <= stop_loss:
                    logger.info(f"SKIP: {ticker} - Open ${today_open:.2f} <= Stop ${stop_loss:.2f} (gap protection)")
                    self.skipped_trades.append({
                        'ticker': ticker,
                        'signal_date': pending.get('signal_date', ''),
                        'planned_entry_date': date,
                        'open_price': today_open,
                        'stop_loss': stop_loss,
                        'reason': 'gap_below_stop'
                    })
                    self.pending_signals.remove(pending)
                    continue
                
                # Execute entry at today's OPEN
                pending['entry_price'] = today_open  # Override with actual OPEN
                self.enter_position(pending, date, today_open, ticker_data)
                self.pending_signals.remove(pending)
            
            # ============================================
            # STEP 2: Check exit conditions for open positions
            # ============================================
            for position in self.open_positions.copy():
                ticker = position['ticker']
                if ticker not in data_dict:
                    continue
                
                ticker_df = data_dict[ticker]
                ticker_data = ticker_df[ticker_df['Date'] <= date]
                
                if ticker_data.empty or len(ticker_data) < 2:
                    continue
                
                current_bar = ticker_data.iloc[-1]
                previous_bar = ticker_data.iloc[-2] if len(ticker_data) >= 2 else None
                
                # Check exit conditions (with previous bar for MACD)
                exit_decision = self.risk_manager.check_exit_conditions(
                    position, current_bar, 
                    previous_data=previous_bar,
                    current_date=date
                )
                
                if exit_decision['exit']:
                    self.exit_position(
                        position,
                        exit_decision['price'],
                        date,
                        exit_decision['reason'],
                        ticker_data
                    )
            
            # ============================================
            # STEP 3: Generate new signals (ADD TO PENDING, don't enter today)
            # ============================================
            max_positions = self.risk_manager.MAX_OPEN_POSITIONS
            pending_count = len(self.pending_signals)
            
            if len(self.open_positions) + pending_count < max_positions:
                signals = []
                
                for ticker, df in data_dict.items():
                    # Skip if already have position or pending signal
                    if any(p['ticker'] == ticker for p in self.open_positions):
                        continue
                    if any(p['ticker'] == ticker for p in self.pending_signals):
                        continue
                    
                    ticker_data = df[df['Date'] <= date].copy()
                    
                    if ticker_data.empty or len(ticker_data) < 50:
                        continue
                    
                    ticker_data = self.signal_generator.calculate_all_indicators(ticker_data)
                    signal = self.signal_generator.generate_signal(ticker, ticker_data)
                    
                    if signal:
                        signals.append(signal)
                
                # Sort by score and add to pending
                signals.sort(key=lambda x: x.get('score', 0), reverse=True)
                slots_available = max_positions - len(self.open_positions) - pending_count
                
                for signal in signals[:slots_available]:
                    signal['signal_date'] = date  # Record when signal was generated
                    self.pending_signals.append(signal)
                    logger.debug(f"PENDING: {signal['ticker']} signal added for next-day execution")
            
            # ============================================
            # STEP 4: Update portfolio value
            # ============================================
            self.update_portfolio_value(current_prices)
            
            self.equity_curve.append({
                'date': date,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'positions_value': self.portfolio_value - self.cash,
                'num_positions': len(self.open_positions),
                'pending_signals': len(self.pending_signals)
            })
            
        except Exception as e:
            logger.error(f"Error processing day {date}: {e}", exc_info=True)
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Run backtest over date range.
        
        Returns dictionary with backtest results including bias warnings.
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        logger.info(f"Testing on {len(data_dict)} stocks")
        
        # Reset state
        self.reset()
        
        # Generate date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get all unique dates from data
        all_dates = set()
        for df in data_dict.values():
            dates = df['Date'].dt.strftime('%Y-%m-%d').tolist() if df['Date'].dtype == 'datetime64[ns]' else df['Date'].tolist()
            all_dates.update(dates)
        
        date_range = sorted([d for d in all_dates if start_date <= d <= end_date])
        
        logger.info(f"Processing {len(date_range)} trading days")
        
        # Process each day
        previous_date = None
        for date in date_range:
            self.process_day(date, data_dict, previous_date)
            previous_date = date
        
        # Close all remaining positions at end
        if self.open_positions:
            logger.info(f"Closing {len(self.open_positions)} remaining positions")
            for position in self.open_positions.copy():
                ticker = position['ticker']
                if ticker in data_dict:
                    final_data = data_dict[ticker][data_dict[ticker]['Date'] <= end_date]
                    if not final_data.empty:
                        final_price = final_data.iloc[-1]['Close']
                        self.exit_position(position, final_price, end_date, 'backtest_end', final_data)
        
        # Final portfolio value
        self.portfolio_value = self.cash
        
        # Compile results with bias warnings
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_portfolio_value': self.portfolio_value,
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'total_pnl': self.portfolio_value - self.initial_capital,
            'num_trades': len(self.closed_trades),
            'num_stocks_tested': len(data_dict),
            'trades': self.closed_trades,
            'equity_curve': self.equity_curve,
            'skipped_trades': self.skipped_trades,
            
            # Bias warnings
            'bias_warnings': [
                "SURVIVORSHIP BIAS: Using current stock universe only. "
                "Stocks delisted during backtest period are NOT included. "
                "Estimated impact: Results may be 3-8% more optimistic than reality.",
                
                f"EXECUTION MODEL: Entry at next-day OPEN (realistic). "
                f"Skipped {len(self.skipped_trades)} trades due to gap protection.",
                
                f"COST MODEL: Dynamic slippage={self.use_dynamic_slippage}, Spread={self.use_spread}"
            ],
            
            # Cost model info
            'cost_model': {
                'dynamic_slippage': self.use_dynamic_slippage,
                'spread_model': self.use_spread,
                'base_slippage': self.base_slippage,
                'commission': self.commission
            }
        }
        
        logger.info("=" * 60)
        logger.info(f"Backtest Complete: {start_date} to {end_date}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Value: ${self.portfolio_value:,.2f}")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Total Trades: {len(self.closed_trades)}")
        logger.info(f"Skipped Trades (gap): {len(self.skipped_trades)}")
        logger.info("=" * 60)
        
        return results
