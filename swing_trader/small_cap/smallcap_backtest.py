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
from .regime_logic import regime_from_spy_close, regime_unknown
from .thresholds import effective_scan_thresholds

logger = logging.getLogger(__name__)


class SmallCapBacktester:
    """
    Walk-forward backtest for SmallCap momentum system.

    Scans stocks day by day using historical data windows,
    opens simulated trades on signals, and tracks outcomes.

    Live parity notes (scan_stock backtest_mode=True still differs from full live):
    - Earnings filter is skipped; catalyst / short / insider / news bonuses are zero;
      sector RS uses SPY proxy when benchmark window exists.
    - Execution: optional slippage + commission below; live API does not model fills.
    """

    # Backtest parameters (aligned with live SmallCapRisk: 1.5% risk + TYPE_POSITION_CAPS)
    COOLDOWN_DAYS = 3            # Days to wait after stop-out before re-entering same ticker
    # Adverse slippage vs mid/OHLC quote: buy worse, sell worse (0 = same as raw bar prices)
    SLIPPAGE_BPS_PER_SIDE = 5
    # Per-dollar fee on each leg (entry notional + exit notional); 0 = typical US zero-commission
    COMMISSION_BPS_PER_SIDE = 0
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
        self._spy_df: Optional[pd.DataFrame] = None
        self._vix_df: Optional[pd.DataFrame] = None
        self.min_quality = 65
        self.top_n = 10
        self._max_concurrent = 3

    def run_backtest(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
        max_concurrent: int = 3,
        min_quality: int = 65,
        top_n: int = 10,
        progress_callback=None,
    ) -> Dict:
        """
        Run walk-forward backtest (aligned with live scanner: regime, effective thresholds,
        scan_universe-style ranking, backtest_mode scan_stock).

        Args:
            min_quality / top_n: same meaning as POST /api/scanner/smallcap ScanRequest.
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.min_quality = min_quality
        self.top_n = top_n
        self._max_concurrent = max_concurrent
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

        self._spy_df, self._vix_df = self._fetch_benchmarks(warmup_start, end_plus)
        
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

            regime, eff_min, eff_top, spy_rs = self._daily_regime_bundle(current_date_dt)
            
            # 2c: Record equity (+ daily regime / thresholds for UI)
            portfolio_value = self._calculate_portfolio_value(current_date_dt, data_dict)
            self.equity_curve.append({
                'date': current_date_dt.strftime('%Y-%m-%d'),
                'portfolio_value': round(portfolio_value, 2),
                'open_trades': len(self.open_trades),
                'market_regime': regime.get('regime', 'UNKNOWN'),
                'regime_confidence': regime.get('confidence', ''),
                'regime_multiplier': regime.get('score_multiplier', 1.0),
                'effective_min_quality': eff_min,
                'effective_top_n': eff_top,
                'request_min_quality': self.min_quality,
                'request_top_n': self.top_n,
            })
            
            # 2d: Scan for new signals → queue as PENDING (enter tomorrow)
            if len(self.open_trades) < max_concurrent:
                self._scan_for_signals(
                    current_date_dt,
                    data_dict,
                    max_concurrent,
                    regime,
                    eff_min,
                    eff_top,
                    spy_rs,
                )
            
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
            'period': f"{start_date} to {end_date}",
            'params': {
                'min_quality': self.min_quality,
                'top_n': self.top_n,
                'max_concurrent': self._max_concurrent,
                'slippage_bps_per_side': self.SLIPPAGE_BPS_PER_SIDE,
                'commission_bps_per_side': self.COMMISSION_BPS_PER_SIDE,
            },
        }

    def _entry_fill_price(self, quoted: float) -> float:
        b = self.SLIPPAGE_BPS_PER_SIDE / 10000.0
        return round(quoted * (1.0 + b), 2)

    def _exit_fill_price(self, quoted: float) -> float:
        if quoted <= 0:
            return quoted
        b = self.SLIPPAGE_BPS_PER_SIDE / 10000.0
        return round(quoted * (1.0 - b), 2)

    def _commission_dollars(self, entry_px: float, exit_px: float, shares: int) -> float:
        b = self.COMMISSION_BPS_PER_SIDE / 10000.0
        if b <= 0 or shares <= 0:
            return 0.0
        return entry_px * shares * b + exit_px * shares * b

    def _normalize_benchmark_df(self, raw: Optional[pd.DataFrame]) -> pd.DataFrame:
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        d = raw.reset_index()
        date_col = "Date" if "Date" in d.columns else d.columns[0]
        d = d.rename(columns={date_col: "Date"})
        col = pd.to_datetime(d["Date"])
        if getattr(col.dt, "tz", None) is not None:
            col = col.dt.tz_convert("UTC")
        d["Date"] = pd.to_datetime(col.dt.strftime("%Y-%m-%d"))
        return d

    def _fetch_benchmarks(self, start_s: str, end_s: str) -> tuple:
        """Load SPY + VIX for point-in-time regime (same rules as live)."""
        spy_df = pd.DataFrame()
        vix_df = pd.DataFrame()
        try:
            sh = yf.Ticker("SPY").history(start=start_s, end=end_s)
            spy_df = self._normalize_benchmark_df(sh)
        except Exception as e:
            logger.warning("Backtest: SPY benchmark fetch failed: %s", e)
        try:
            vh = yf.Ticker("^VIX").history(start=start_s, end=end_s)
            vix_df = self._normalize_benchmark_df(vh)
        except Exception as e:
            logger.debug("Backtest: VIX fetch failed: %s", e)
        return spy_df, vix_df

    def _daily_regime_bundle(self, current_date: datetime):
        """Regime from SPY/VIX as of current_date + effective thresholds + SPY slice for RS."""
        if self._spy_df is None or len(self._spy_df) == 0:
            r = regime_unknown("no_spy_benchmark")
            eff_min, eff_top = effective_scan_thresholds(
                r["regime"], r["confidence"], float(r["score_multiplier"]), self.min_quality, self.top_n
            )
            return r, eff_min, eff_top, pd.DataFrame()

        m = self._spy_df[self._spy_df["Date"] <= current_date]
        if len(m) < 50:
            r = regime_unknown("insufficient_spy_history")
            eff_min, eff_top = effective_scan_thresholds(
                r["regime"], r["confidence"], float(r["score_multiplier"]), self.min_quality, self.top_n
            )
            return r, eff_min, eff_top, m.tail(60) if len(m) >= 6 else m

        spy_regime = m.tail(252)
        vx = None
        if self._vix_df is not None and len(self._vix_df) > 0:
            vm = self._vix_df[self._vix_df["Date"] <= current_date]
            if len(vm) > 0:
                vx = float(vm["Close"].iloc[-1])

        r = regime_from_spy_close(spy_regime["Close"], vx)
        eff_min, eff_top = effective_scan_thresholds(
            r.get("regime", "UNKNOWN"),
            r.get("confidence", "TENTATIVE"),
            float(r.get("score_multiplier", 1.0)),
            self.min_quality,
            self.top_n,
        )
        spy_rs = m.tail(60)
        return r, eff_min, eff_top, spy_rs

    def _scan_for_signals(
        self,
        current_date: datetime,
        data_dict: Dict,
        max_concurrent: int,
        regime: Dict,
        eff_min: int,
        eff_top: int,
        spy_rs: pd.DataFrame,
    ):
        """
        Full-universe day scan: backtest_mode scan_stock, regime multiplier,
        original_quality_score filter (live parity), sort, top_n, then queue pending.
        """
        open_tickers = {t["ticker"] for t in self.open_trades}
        pending_tickers = {s["signal"]["ticker"] for s in self.pending_signals}
        slots = max_concurrent - len(self.open_trades) - len(self.pending_signals)
        if slots <= 0:
            return

        candidates: List[Dict] = []

        for ticker in sorted(data_dict.keys()):
            if ticker in open_tickers or ticker in pending_tickers:
                continue

            if ticker in self.cooldowns:
                if current_date < self.cooldowns[ticker]:
                    continue
                del self.cooldowns[ticker]

            full_df = data_dict[ticker]
            mask = full_df["Date"] <= current_date
            df_window = full_df[mask].copy()
            if len(df_window) < 50:
                continue

            try:
                signal = self.engine.scan_stock(
                    ticker,
                    df_window,
                    backtest_mode=True,
                    portfolio_value=self.capital,
                    spy_df_window=spy_rs if len(spy_rs) >= 6 else None,
                )
                if not signal:
                    continue

                orig = float(signal["quality_score"])
                signal["original_quality_score"] = orig
                mult = float(regime.get("score_multiplier", 1.0))
                signal["market_regime"] = regime.get("regime", "UNKNOWN")
                signal["regime_multiplier"] = mult
                signal["regime_confidence"] = regime.get("confidence", "CONFIRMED")
                if mult < 1.0:
                    signal["quality_score"] = round(orig * mult, 1)

                signal_rsi = signal.get("rsi", 50)
                if signal_rsi > self.MAX_ENTRY_RSI:
                    continue

                candidates.append(
                    {"ticker": ticker, "signal": signal, "df_window": df_window}
                )
            except Exception as e:
                logger.debug(f"Scan error {ticker}: {e}")

        candidates.sort(key=lambda x: x["signal"].get("quality_score", 0), reverse=True)
        filtered = [
            c
            for c in candidates
            if c["signal"].get("original_quality_score", c["signal"]["quality_score"]) >= eff_min
        ][:eff_top]

        for item in filtered:
            if len(self.open_trades) + len(self.pending_signals) >= max_concurrent:
                break
            sig = item["signal"]
            t = item["ticker"]
            df_window = item["df_window"]
            self.pending_signals.append(
                {
                    "signal": sig,
                    "signal_date": current_date,
                    "signal_close": float(df_window["Close"].iloc[-1]),
                    "df_window": df_window,
                }
            )
    
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
            
            # ── All filters passed: Enter at today's Open (+ adverse slippage) ──
            # Live parity: stop/target/sizing from SmallCapRisk.add_risk_management using
            # signal-day df (same as scan_stock), entry_price = fill at open.
            entry_price = self._entry_fill_price(today_open)
            df_window = pending['df_window']
            sig = signal.copy()
            sig['entry_price'] = entry_price
            try:
                self.risk.add_risk_management(sig, df_window, portfolio_value=self.capital)
            except Exception as e:
                logger.debug(f"{ticker}: add_risk_management failed on entry: {e}")
                continue
            if int(sig.get('position_size') or 0) < 1:
                continue
            sl = float(sig.get('stop_loss') or 0)
            if sl <= 0 or sl >= entry_price:
                continue
            self._open_trade(sig, current_date, df_today)
        
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
            target = entry_price + (risk_amount * 2.0)  # fallback min ~2R if target_1 missing
        
        # Validate
        if entry_price <= 0 or stop_loss <= 0 or stop_loss >= entry_price:
            return
        
        swing_type = signal.get('swing_type', 'A')
        ps = int(signal.get('position_size') or 0)
        if ps >= 1:
            shares = ps
        else:
            shares, _ = self.risk.calculate_position_size(
                self.capital, entry_price, stop_loss, swing_type
            )
        if shares < 1:
            return
        
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
            'swing_type': swing_type,
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
                trade['exit_price'] = self._exit_fill_price(exit_price)
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
                trade['exit_price'] = self._exit_fill_price(float(trade['target']))
                trade['exit_date'] = current_date.strftime('%Y-%m-%d')
                trade['exit_reason'] = 'Target Hit'
                trade['status'] = 'TARGET'
                trade['days_held'] = days_held
                self._close_trade(trade)
                continue
            
            # Check timeout
            if days_held >= trade['max_hold_days']:
                trade['exit_price'] = self._exit_fill_price(current_close)
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
        fee = self._commission_dollars(entry, exit_p, shares)
        trade['commission_dollar'] = round(fee, 2)
        
        pnl_pct = ((exit_p / entry) - 1) * 100
        pnl_dollar = (exit_p - entry) * shares - fee
        
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
                    trade['exit_price'] = self._exit_fill_price(stop_loss)
                    trade['exit_reason'] = 'Backtest End (Stop Capped)'
                else:
                    trade['exit_price'] = self._exit_fill_price(last_close)
                    trade['exit_reason'] = 'Backtest End'
            else:
                trade['exit_price'] = self._exit_fill_price(trade['entry_price'])
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
            'period': '',
            'params': {
                'min_quality': getattr(self, 'min_quality', 65),
                'top_n': getattr(self, 'top_n', 10),
                'max_concurrent': getattr(self, '_max_concurrent', 3),
                'slippage_bps_per_side': self.SLIPPAGE_BPS_PER_SIDE,
                'commission_bps_per_side': self.COMMISSION_BPS_PER_SIDE,
            },
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
