"""
SmallCap Walk-Forward Backtest Engine.

Simulates daily SmallCap scans over historical data to evaluate
signal quality, win rates, and overall system performance.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

from .engine import SmallCapEngine
from .risk import SmallCapRisk

logger = logging.getLogger(__name__)


class SmallCapBacktester:
    """
    Walk-forward backtest for SmallCap momentum system.
    
    Scans stocks day by day using historical data windows,
    opens simulated trades on signals, and tracks outcomes.
    Uses proper ATR-based stops and dynamic position sizing.
    """
    
    # Backtest parameters
    MIN_QUALITY_SCORE = 5.0      # Minimum signal quality to enter
    COOLDOWN_DAYS = 3            # Days to wait after stop-out before re-entering same ticker
    RISK_PER_TRADE_PCT = 0.005   # 0.5% of portfolio risked per trade
    MAX_GAP_UP_PCT = 5.0         # Skip entry if gap-up > 5% (momentum exhausted)
    MAX_GAP_DOWN_PCT = 3.0       # Skip entry if gap-down > 3% (bad news)
    MAX_ENTRY_RSI = 70           # Don't enter if RSI > 70 (overbought)
    
    def __init__(self, config: Dict = None):
        self.engine = SmallCapEngine(config)
        self.risk = SmallCapRisk(config)
        self.config = config or {}
        self.trades = []
        self.open_trades = []
        self.equity_curve = []
        self.initial_capital = 10000
        self.capital = self.initial_capital
        self.cooldowns = {}  # ticker -> date when cooldown expires
        self.pending_signals = []  # Signals waiting for next-day entry
    
    def run_backtest(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
        max_concurrent: int = 3,
        progress_callback=None
    ) -> Dict:
        """
        Run walk-forward backtest.
        
        Args:
            tickers: List of tickers to scan
            start_date: Backtest start (YYYY-MM-DD)
            end_date: Backtest end (YYYY-MM-DD)
            initial_capital: Starting capital
            max_concurrent: Max simultaneous open trades
            progress_callback: Callback for progress updates
        
        Returns:
            Dict with trades, equity_curve, and metrics
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.open_trades = []
        self.equity_curve = []
        self.cooldowns = {}
        self.pending_signals = []
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Step 1: Fetch all data upfront (3 months before start for indicator warmup)
        warmup_start = (start_dt - timedelta(days=90)).strftime('%Y-%m-%d')
        end_plus = (end_dt + timedelta(days=15)).strftime('%Y-%m-%d')  # Extra for trade exits
        
        if progress_callback:
            progress_callback(0, f"Downloading data for {len(tickers)} stocks...")
        
        data_dict = {}
        for i, ticker in enumerate(tickers):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=warmup_start, end=end_plus)
                if df is not None and len(df) >= 60:
                    df = df.reset_index()
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                    data_dict[ticker] = df
            except Exception as e:
                logger.debug(f"Skip {ticker}: {e}")
            
            if progress_callback and i % 5 == 0:
                pct = int((i / len(tickers)) * 30)  # 0-30% for data download
                progress_callback(pct, f"Downloaded {i+1}/{len(tickers)} stocks...")
        
        if not data_dict:
            return self._empty_results()
        
        if progress_callback:
            progress_callback(30, f"Data ready: {len(data_dict)} stocks. Starting simulation...")
        
        # Step 2: Walk forward day by day
        trading_days = pd.bdate_range(start_dt, end_dt)
        total_days = len(trading_days)
        
        for day_idx, current_date in enumerate(trading_days):
            current_date_dt = current_date.to_pydatetime()
            
            # 2a: Check exits on open trades
            self._check_exits(current_date_dt, data_dict)
            
            # 2b: Process pending signals from YESTERDAY (next-day-open entry)
            if self.pending_signals and len(self.open_trades) < max_concurrent:
                self._process_pending_entries(current_date_dt, data_dict, max_concurrent)
            
            # 2c: Record equity
            portfolio_value = self._calculate_portfolio_value(current_date_dt, data_dict)
            self.equity_curve.append({
                'date': current_date_dt.strftime('%Y-%m-%d'),
                'portfolio_value': round(portfolio_value, 2),
                'open_trades': len(self.open_trades)
            })
            
            # 2d: Scan for new signals → queue as PENDING (enter tomorrow)
            if len(self.open_trades) < max_concurrent:
                self._scan_for_signals(current_date_dt, data_dict, max_concurrent)
            
            if progress_callback and day_idx % 5 == 0:
                pct = 30 + int((day_idx / total_days) * 65)  # 30-95%
                progress_callback(pct, f"Day {day_idx+1}/{total_days} | "
                                      f"Trades: {len(self.trades)} | "
                                      f"Open: {len(self.open_trades)}")
        
        # Step 3: Force close remaining open trades at last price
        self._force_close_all(end_dt, data_dict)
        
        if progress_callback:
            progress_callback(100, f"Done! {len(self.trades)} total trades")
        
        # Step 4: Calculate metrics
        metrics = self._calculate_metrics()
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'metrics': metrics,
            'data_stocks': len(data_dict),
            'period': f"{start_date} to {end_date}"
        }
    
    def _scan_for_signals(self, current_date: datetime, data_dict: Dict, max_concurrent: int):
        """Scan all stocks — signals are QUEUED for next-day entry (not instant)."""
        open_tickers = {t['ticker'] for t in self.open_trades}
        pending_tickers = {s['signal']['ticker'] for s in self.pending_signals}
        
        for ticker, full_df in data_dict.items():
            if ticker in open_tickers or ticker in pending_tickers:
                continue
            if len(self.open_trades) + len(self.pending_signals) >= max_concurrent:
                break
            
            # Check cooldown
            if ticker in self.cooldowns:
                if current_date < self.cooldowns[ticker]:
                    continue
                else:
                    del self.cooldowns[ticker]
            
            # Slice data up to current date
            mask = full_df['Date'] <= current_date
            df_window = full_df[mask].copy()
            
            if len(df_window) < 50:
                continue
            
            try:
                signal = self.engine.scan_stock(ticker, df_window)
                if signal and signal.get('quality_score', 0) >= self.MIN_QUALITY_SCORE:
                    # ── ADIM 2: RSI filter — don't queue if RSI too high ──
                    signal_rsi = signal.get('rsi', 50)
                    if signal_rsi > self.MAX_ENTRY_RSI:
                        logger.debug(f"{ticker}: RSI {signal_rsi:.0f} > {self.MAX_ENTRY_RSI} — skipped")
                        continue
                    
                    # Apply risk management (ATR stops, targets, sizing)
                    signal = self.risk.add_risk_management(
                        signal, df_window, portfolio_value=self.capital
                    )
                    
                    # ── ADIM 1: Queue for NEXT DAY entry (don't enter today) ──
                    self.pending_signals.append({
                        'signal': signal,
                        'signal_date': current_date,
                        'signal_close': float(df_window['Close'].iloc[-1]),
                        'df_window': df_window
                    })
            except Exception as e:
                logger.debug(f"Scan error {ticker}: {e}")
    
    def _process_pending_entries(self, current_date: datetime, data_dict: Dict, max_concurrent: int):
        """Process pending signals: enter at today's Open with gap/confirmation filters."""
        remaining = []
        
        for pending in self.pending_signals:
            if len(self.open_trades) >= max_concurrent:
                remaining.append(pending)  # Keep for next day? No — expire after 1 day
                continue
            
            signal = pending['signal']
            signal_close = pending['signal_close']
            ticker = signal['ticker']
            
            # Skip if already in a trade
            if ticker in {t['ticker'] for t in self.open_trades}:
                continue
            
            # Get today's data
            full_df = data_dict.get(ticker)
            if full_df is None:
                continue
            
            # Find today's bar
            mask = full_df['Date'] <= current_date
            df_today = full_df[mask]
            if len(df_today) < 2:
                continue
            
            today_open = float(df_today['Open'].iloc[-1])
            
            # ── ADIM 1: Gap filter — skip if gap too large ──
            gap_pct = ((today_open - signal_close) / signal_close) * 100
            
            if gap_pct > self.MAX_GAP_UP_PCT:
                logger.debug(f"{ticker}: Gap UP {gap_pct:+.1f}% > {self.MAX_GAP_UP_PCT}% — skipped")
                continue
            
            if gap_pct < -self.MAX_GAP_DOWN_PCT:
                logger.debug(f"{ticker}: Gap DOWN {gap_pct:+.1f}% < -{self.MAX_GAP_DOWN_PCT}% — skipped")
                continue
            
            # ── ADIM 4: Tip C next-day confirmation ──
            # Tip C requires Open > Yesterday Close (buyers still in control)
            if signal.get('swing_type') == 'C':
                if today_open <= signal_close:
                    logger.debug(f"{ticker}: TipC confirmation FAILED (Open ${today_open:.2f} <= Close ${signal_close:.2f})")
                    continue
            
            # ── All filters passed: Enter at today's Open ──
            # Recalculate stop/target based on actual entry price
            entry_price = today_open
            signal['entry_price'] = round(entry_price, 2)
            
            # Recalculate ATR-based stop from entry (v2.3: type-specific cap)
            df_window = pending['df_window']
            swing_type = signal.get('swing_type', 'A')
            atr = self.risk.calculate_atr(df_window)
            stop_loss = entry_price - (self.risk.STOP_ATR_MULTIPLIER * atr)
            max_stop_pct = self.risk.MAX_STOP_BY_TYPE.get(swing_type, self.risk.MAX_STOP_PERCENT)
            max_stop = entry_price * (1 - max_stop_pct)
            stop_loss = max(stop_loss, max_stop)
            signal['stop_loss'] = round(stop_loss, 2)
            
            # ── ADIM 3: Type-specific T1/T2 targets (v2.3) ──
            t1_pct, t2_pct = self.risk.TYPE_TARGETS.get(swing_type, (0.25, 0.40))
            target_1 = entry_price * (1 + t1_pct)
            target_2 = entry_price * (1 + t2_pct)
            # Safety floor: T1 must be at least 2R
            risk_amount = entry_price - stop_loss
            if risk_amount > 0 and (target_1 - entry_price) < (2 * risk_amount):
                target_1 = entry_price + (2 * risk_amount)
            signal['target_1'] = round(target_1, 2)
            signal['target_2'] = round(target_2, 2)
            
            # Determine stop method label
            atr_stop_val = entry_price - (self.risk.STOP_ATR_MULTIPLIER * atr)
            if stop_loss > atr_stop_val:
                signal['stop_method'] = f"Max Stop ({max_stop_pct*100:.0f}%)"
            else:
                atr_pct = ((entry_price - stop_loss) / entry_price) * 100
                signal['stop_method'] = f"ATR Stop ({atr_pct:.1f}%)"
            
            self._open_trade(signal, current_date, df_today)
        
        # Expire all pending signals (1-day validity)
        self.pending_signals = []
    
    def _open_trade(self, signal: Dict, entry_date: datetime, df: pd.DataFrame):
        """Open a simulated trade from a signal with proper risk management."""
        entry_price = signal.get('entry_price', float(df['Close'].iloc[-1]))
        
        # ATR-based stop from risk module (no more fixed 8%!)
        stop_loss = signal.get('stop_loss', 0)
        if stop_loss <= 0 or stop_loss >= entry_price:
            # Fallback: calculate manually
            atr = self.risk.calculate_atr(df)
            stop_loss = entry_price - (self.risk.STOP_ATR_MULTIPLIER * atr)
        
        # ATR-based target from risk module (3R minimum)
        target = signal.get('target_1', 0)
        if target <= entry_price:
            risk_amount = entry_price - stop_loss
            target = entry_price + (risk_amount * self.risk.MIN_RR_RATIO)
        
        # Validate
        if entry_price <= 0 or stop_loss <= 0 or stop_loss >= entry_price:
            return
        
        # Dynamic position sizing: risk 0.5% of capital per trade
        risk_per_share = entry_price - stop_loss
        max_risk_dollar = self.capital * self.RISK_PER_TRADE_PCT
        shares = max(1, int(max_risk_dollar / risk_per_share))
        
        # Cap at 10% of capital in single position
        max_shares_by_capital = int((self.capital * 0.10) / entry_price)
        shares = min(shares, max(1, max_shares_by_capital))
        
        # Get max hold from signal
        max_hold = signal.get('hold_days_max', self.risk.MAX_HOLDING_DAYS)
        if isinstance(signal.get('hold_days'), tuple):
            max_hold = signal['hold_days'][1]
        
        # Calculate stop/target percentages for logging
        stop_pct = ((entry_price - stop_loss) / entry_price) * 100
        target_pct = ((target - entry_price) / entry_price) * 100
        
        # Calculate ATR for trailing stop
        atr = self.risk.calculate_atr(df)
        
        trade = {
            'ticker': signal['ticker'],
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'initial_stop': round(stop_loss, 2),
            'trailing_stop': round(stop_loss, 2),
            'target': round(target, 2),
            'stop_pct': round(stop_pct, 1),
            'target_pct': round(target_pct, 1),
            'shares': shares,
            'atr': round(atr, 4),
            'swing_type': signal.get('swing_type', 'A'),
            'quality_score': signal.get('quality_score', 0),
            'stop_method': signal.get('stop_method', 'ATR'),
            'max_hold_days': max_hold,
            'status': 'OPEN',
            'exit_price': 0,
            'exit_date': '',
            'exit_reason': '',
            'pnl_pct': 0,
            'pnl_dollar': 0
        }
        
        self.open_trades.append(trade)
    
    def _check_exits(self, current_date: datetime, data_dict: Dict):
        """Check exit conditions for all open trades with trailing stop."""
        still_open = []
        
        for trade in self.open_trades:
            ticker = trade['ticker']
            df = data_dict.get(ticker)
            
            if df is None:
                still_open.append(trade)
                continue
            
            # Get today's bar
            mask = df['Date'].dt.date == current_date.date() if hasattr(df['Date'].iloc[0], 'date') else df['Date'] == current_date
            today_bars = df[mask]
            
            if len(today_bars) == 0:
                still_open.append(trade)
                continue
            
            bar = today_bars.iloc[0]
            entry_date = datetime.strptime(trade['entry_date'], '%Y-%m-%d')
            days_held = (current_date - entry_date).days
            current_close = float(bar['Close'])
            current_high = float(bar['High'])
            current_low = float(bar['Low'])
            atr = trade.get('atr', 0)
            
            # ── TRAILING STOP UPDATE ──
            # Only activate after 50% of max hold time AND 2+ ATR gain
            half_hold = trade['max_hold_days'] * 0.5
            if atr > 0 and current_close > trade['entry_price'] and days_held >= half_hold:
                unrealized_gain = current_close - trade['entry_price']
                atr_gain = unrealized_gain / atr
                
                if atr_gain >= 2.0:
                    atr_steps = int(atr_gain) - 1
                    new_trail = trade['initial_stop'] + (atr_steps * atr)
                    # Only move stop UP, never down
                    if new_trail > trade['trailing_stop']:
                        trade['trailing_stop'] = round(new_trail, 2)
            
            # Use trailing stop as active stop
            active_stop = trade['trailing_stop']
            
            # Check stop loss (trailing stop)
            # CRITICAL: If today's LOW gaps through our stop, we can't exit at
            # the stop price — we exit at the Open (realistic slippage)
            if current_low <= active_stop:
                today_open = float(bar['Open'])
                if today_open <= active_stop:
                    # Gap-down through stop — exit at Open (slippage)
                    exit_price = today_open
                else:
                    # Intraday dip to stop — exit at stop price
                    exit_price = active_stop
                is_trail = active_stop > trade['initial_stop']
                trade['exit_price'] = round(exit_price, 2)
                trade['exit_date'] = current_date.strftime('%Y-%m-%d')
                trade['exit_reason'] = 'Trailing Stop' if is_trail else 'Stop Loss'
                trade['status'] = 'TRAILED' if is_trail else 'STOPPED'
                trade['days_held'] = days_held
                self._close_trade(trade)
                # Set cooldown for this ticker
                self.cooldowns[ticker] = current_date + timedelta(days=self.COOLDOWN_DAYS)
                continue
            
            # Check target (high touches target)
            if current_high >= trade['target']:
                trade['exit_price'] = trade['target']
                trade['exit_date'] = current_date.strftime('%Y-%m-%d')
                trade['exit_reason'] = 'Target Hit'
                trade['status'] = 'TARGET'
                trade['days_held'] = days_held
                self._close_trade(trade)
                continue
            
            # Check timeout
            if days_held >= trade['max_hold_days']:
                trade['exit_price'] = round(current_close, 2)
                trade['exit_date'] = current_date.strftime('%Y-%m-%d')
                trade['exit_reason'] = f'Timeout ({days_held}d)'
                trade['status'] = 'TIMEOUT'
                trade['days_held'] = days_held
                self._close_trade(trade)
                continue
            
            still_open.append(trade)
        
        self.open_trades = still_open
    
    def _close_trade(self, trade: Dict):
        """Calculate P/L using actual position size and move to closed trades."""
        entry = trade['entry_price']
        exit_p = trade['exit_price']
        shares = trade.get('shares', 100)
        
        pnl_pct = ((exit_p / entry) - 1) * 100
        pnl_dollar = (exit_p - entry) * shares
        
        trade['pnl_pct'] = round(pnl_pct, 2)
        trade['pnl_dollar'] = round(pnl_dollar, 2)
        
        self.capital += pnl_dollar
        self.trades.append(trade)
    
    def _force_close_all(self, end_date: datetime, data_dict: Dict):
        """Force close all remaining open trades at end of backtest.
        CRITICAL: Cap exit price at stop loss — can't lose more than stop allows.
        """
        for trade in self.open_trades:
            ticker = trade['ticker']
            df = data_dict.get(ticker)
            
            if df is not None and len(df) > 0:
                last_close = float(df['Close'].iloc[-1])
                stop_loss = trade.get('trailing_stop', trade.get('stop_loss', 0))
                
                # CRITICAL: If last close is below stop, cap at stop price
                # In real trading, stop would have fired before reaching this price
                if stop_loss > 0 and last_close < stop_loss:
                    trade['exit_price'] = round(stop_loss, 2)
                    trade['exit_reason'] = 'Backtest End (Stop Capped)'
                else:
                    trade['exit_price'] = round(last_close, 2)
                    trade['exit_reason'] = 'Backtest End'
            else:
                trade['exit_price'] = trade['entry_price']
                trade['exit_reason'] = 'Backtest End'
            
            entry_date = datetime.strptime(trade['entry_date'], '%Y-%m-%d')
            trade['exit_date'] = end_date.strftime('%Y-%m-%d')
            trade['status'] = 'FORCED'
            trade['days_held'] = (end_date - entry_date).days
            self._close_trade(trade)
        
        self.open_trades = []
    
    def _calculate_portfolio_value(self, current_date: datetime, data_dict: Dict) -> float:
        """Calculate total portfolio value including open positions."""
        value = self.capital
        
        for trade in self.open_trades:
            ticker = trade['ticker']
            df = data_dict.get(ticker)
            if df is None:
                continue
            
            mask = df['Date'] <= current_date
            subset = df[mask]
            if len(subset) > 0:
                current_price = float(subset['Close'].iloc[-1])
                shares = trade.get('shares', 100)
                unrealized = (current_price - trade['entry_price']) * shares
                value += unrealized
        
        return value
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics from completed trades."""
        if not self.trades:
            return self._empty_metrics()
        
        wins = [t for t in self.trades if t['pnl_pct'] > 0]
        losses = [t for t in self.trades if t['pnl_pct'] <= 0]
        
        total = len(self.trades)
        win_count = len(wins)
        loss_count = len(losses)
        
        win_rate = win_count / total if total > 0 else 0
        
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        
        avg_win_dollar = np.mean([t['pnl_dollar'] for t in wins]) if wins else 0
        avg_loss_dollar = np.mean([t['pnl_dollar'] for t in losses]) if losses else 0
        
        total_win_dollar = sum(t['pnl_dollar'] for t in wins)
        total_loss_dollar = abs(sum(t['pnl_dollar'] for t in losses))
        
        profit_factor = total_win_dollar / total_loss_dollar if total_loss_dollar > 0 else float('inf')
        
        total_pnl_dollar = sum(t['pnl_dollar'] for t in self.trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Max drawdown from equity curve
        max_dd = 0
        if self.equity_curve:
            peak = self.equity_curve[0]['portfolio_value']
            for point in self.equity_curve:
                val = point['portfolio_value']
                if val > peak:
                    peak = val
                dd = (peak - val) / peak * 100
                if dd > max_dd:
                    max_dd = dd
        
        # By swing type
        type_stats = {}
        for t in self.trades:
            st = t.get('swing_type', 'B')
            if st not in type_stats:
                type_stats[st] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
            if t['pnl_pct'] > 0:
                type_stats[st]['wins'] += 1
            else:
                type_stats[st]['losses'] += 1
            type_stats[st]['total_pnl'] += t['pnl_pct']
        
        # By exit reason
        exit_stats = {}
        for t in self.trades:
            reason = t.get('status', 'UNKNOWN')
            if reason not in exit_stats:
                exit_stats[reason] = {'count': 0, 'avg_pnl': []}
            exit_stats[reason]['count'] += 1
            exit_stats[reason]['avg_pnl'].append(t['pnl_pct'])
        
        for reason in exit_stats:
            pnls = exit_stats[reason]['avg_pnl']
            exit_stats[reason]['avg_pnl'] = round(np.mean(pnls), 2) if pnls else 0
        
        # Avg hold days
        avg_hold = np.mean([t.get('days_held', 0) for t in self.trades])
        
        return {
            'total_trades': total,
            'win_rate': round(win_rate, 4),
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'avg_win_dollar': round(avg_win_dollar, 2),
            'avg_loss_dollar': round(avg_loss_dollar, 2),
            'profit_factor': round(profit_factor, 2),
            'total_pnl_dollar': round(total_pnl_dollar, 2),
            'total_return': round(total_return, 4),
            'max_drawdown': round(max_dd, 2),
            'avg_hold_days': round(avg_hold, 1),
            'initial_capital': self.initial_capital,
            'final_capital': round(self.capital, 2),
            'type_stats': type_stats,
            'exit_stats': exit_stats
        }
    
    def _empty_results(self):
        return {
            'trades': [],
            'equity_curve': [],
            'metrics': self._empty_metrics(),
            'data_stocks': 0,
            'period': ''
        }
    
    def _empty_metrics(self):
        return {
            'total_trades': 0, 'win_rate': 0, 'winning_trades': 0,
            'losing_trades': 0, 'avg_win_pct': 0, 'avg_loss_pct': 0,
            'avg_win_dollar': 0, 'avg_loss_dollar': 0, 'profit_factor': 0,
            'total_pnl_dollar': 0, 'total_return': 0, 'max_drawdown': 0,
            'avg_hold_days': 0, 'initial_capital': 0, 'final_capital': 0,
            'type_stats': {}, 'exit_stats': {}
        }
