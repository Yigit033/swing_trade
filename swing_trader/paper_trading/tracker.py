"""
Paper Trading Tracker - Position tracking and exit detection logic.

V3 Improvements:
- Trailing stop (activates after 50% hold + 2 ATR gain)
- Gap-down slippage (exit at Open if Open < stop)
- PENDING state → confirm at next-day Open with gap filter
- ATR-based stop/target recalculation at actual entry
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python < 3.9

from .storage import PaperTradeStorage

logger = logging.getLogger(__name__)

# NYSE timezone & market hours
_NYSE_TZ = ZoneInfo("America/New_York")
NYSE_OPEN_TIME = "09:30"  # NYSE opens 09:30 ET (= 16:30 Turkey time in winter)

# V3 Constants (match backtest engine)
MAX_GAP_UP_PCT = 5.0     # Skip if gap-up > 5%
MAX_GAP_DOWN_PCT = 3.0   # Skip if gap-down > 3%


class PaperTradeTracker:
    """
    Track paper trades and detect exits.
    
    V3 Exit Types:
    - STOPPED: Price hit stop loss (with gap-down slippage)
    - TRAILED: Trailing stop triggered (profit protection)
    - TARGET: Price hit target
    - TIMEOUT: Max hold days exceeded
    - MANUAL: User manually closed
    - REJECTED: PENDING trade failed gap filter
    
    V3 Logic:
    - New signals start as PENDING
    - Next day: confirm at Open price + gap filter
    - Trailing stop activates after 50% hold time AND 2+ ATR gain
    - Gap-down through stop → exit at Open (realistic slippage)
    """
    
    def __init__(self, storage: PaperTradeStorage = None):
        """Initialize tracker with storage."""
        self.storage = storage or PaperTradeStorage()
    
    def fetch_price_history(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch price history for a ticker.
        
        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            
            # yfinance 'end' is EXCLUSIVE — add 2 days to cover weekends/holidays
            if end_date is None:
                end_dt = datetime.now() + timedelta(days=2)
            else:
                end_dt = datetime.strptime(end_date[:10], '%Y-%m-%d') + timedelta(days=2)
            
            end_date_adj = end_dt.strftime('%Y-%m-%d')
            
            df = stock.history(start=start_date, end=end_date_adj)
            
            # Fallback: if start/end approach returns empty, use period
            if df is None or len(df) == 0:
                logger.warning(f"No data with start/end for {ticker}, trying period='1mo'")
                df = stock.history(period='1mo')
            
            if df is None or len(df) == 0:
                logger.warning(f"No price data for {ticker}")
                return None
            
            # Reset index to have Date as column
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching price history for {ticker}: {e}")
            return None
    
    def check_exit_conditions(
        self,
        trade: Dict,
        price_history: pd.DataFrame
    ) -> Tuple[str, float, str, str, float]:
        """
        Check if trade should be closed.

        V3.1: Includes trailing stop, gap-down slippage, AND partial T1/T2 exit.

        Exit priority per bar:
        1. Stop loss (with gap-down slippage)
        2. T1 partial exit — sell 50%, move stop to breakeven, target T2
        3. T2 full exit — sell remaining 50%
        4. Timeout — close at market

        Returns:
            Tuple of (status, exit_price, exit_date, reason, trailing_stop)
            status: 'OPEN' | 'STOPPED' | 'TRAILED' | 'T1_PARTIAL' | 'TARGET' | 'TIMEOUT'
        """
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        initial_stop = trade.get('initial_stop') or stop_loss
        target = trade['target']  # T1 initially; becomes T2 after partial
        target_2 = trade.get('target_2') or target
        max_hold_days = trade.get('max_hold_days', 7)
        entry_date = datetime.strptime(trade['entry_date'][:10], '%Y-%m-%d').date()
        atr = trade.get('atr') or 0
        trailing_stop = trade.get('trailing_stop') or stop_loss
        has_partial = (trade.get('partial_exit_price') or 0) > 0

        if len(price_history) <= 1:
            return ('OPEN', 0, '', '', trailing_stop)

        for _, row in price_history.iloc[1:].iterrows():
            current_date = row['Date']
            today_open = row['Open']
            low = row['Low']
            high = row['High']
            close = row['Close']
            days_held = (current_date - entry_date).days

            active_stop = trailing_stop

            # ── TRAILING STOP UPDATE ──
            if atr > 0 and close > entry_price and days_held >= max_hold_days * 0.5:
                unrealized_gain = close - entry_price
                atr_gain = unrealized_gain / atr
                if atr_gain >= 2.0:
                    atr_steps = int(atr_gain) - 1
                    new_trail = initial_stop + (atr_steps * atr)
                    if new_trail > trailing_stop:
                        trailing_stop = round(new_trail, 2)
                        active_stop = trailing_stop

            # ── CHECK STOP LOSS (with gap-down slippage) ──
            if low <= active_stop:
                if today_open <= active_stop:
                    exit_price = today_open
                else:
                    exit_price = active_stop

                is_trail = active_stop > initial_stop
                exit_date = str(current_date)
                pnl_pct = ((exit_price / entry_price) - 1) * 100

                if is_trail:
                    reason = f"Trailing stop at ${exit_price:.2f} ({pnl_pct:+.1f}%)"
                    return ('TRAILED', exit_price, exit_date, reason, trailing_stop)
                else:
                    reason = f"Stop hit at ${exit_price:.2f} ({pnl_pct:+.1f}%)"
                    return ('STOPPED', exit_price, exit_date, reason, trailing_stop)

            # ── CHECK T1 PARTIAL EXIT (v3.1) ──
            # Fire only if: no partial taken yet AND T2 is meaningfully above T1
            if not has_partial and target_2 > target * 1.01 and high >= target:
                exit_date = str(current_date)
                pnl_pct = ((target / entry_price) - 1) * 100
                reason = (
                    f"T1 partial 50% at ${target:.2f} ({pnl_pct:+.1f}%) | "
                    f"Stop->breakeven ${entry_price:.2f} | T2 ${target_2:.2f}"
                )
                return ('T1_PARTIAL', target, exit_date, reason, trailing_stop)

            # ── CHECK TARGET (T2 after partial, or T1 if no T2) ──
            if high >= target:
                exit_price = target
                exit_date = str(current_date)
                pnl_pct = ((exit_price / entry_price) - 1) * 100
                label = "T2 target" if has_partial else "Target"
                reason = f"{label} hit at ${exit_price:.2f} ({pnl_pct:+.1f}%)"
                return ('TARGET', exit_price, exit_date, reason, trailing_stop)

            # ── CHECK TIMEOUT ──
            if days_held >= max_hold_days:
                exit_price = close
                exit_date = str(current_date)
                pnl_pct = ((exit_price / entry_price) - 1) * 100
                reason = f"Timeout after {days_held} days at ${exit_price:.2f} ({pnl_pct:+.1f}%)"
                return ('TIMEOUT', exit_price, exit_date, reason, trailing_stop)

        return ('OPEN', 0, '', '', trailing_stop)
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current/last price for a ticker. Uses period='5d' to handle weekends."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')
            if hist is not None and len(hist) > 0:
                return round(float(hist['Close'].iloc[-1]), 4)
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {e}")
            return None
    
    def update_trade_status(self, trade: Dict, user_id: Optional[str] = None) -> Dict:
        """
        Update a single trade's status.
        
        V3: Also handles trailing stop updates and PENDING confirmation.
        
        Returns:
            Updated trade dict with current_price and unrealized_pnl
        """
        # Skip PENDING trades (handled by confirm_pending_trades)
        if trade.get('status') == 'PENDING':
            trade['current_price'] = trade.get('signal_price') or trade['entry_price']
            trade['unrealized_pnl'] = 0
            trade['unrealized_pnl_pct'] = 0
            trade['days_held'] = 0
            return trade
        
        ticker = trade['ticker']
        entry_date = trade['entry_date'][:10]  # Strip time part (e.g. "2026-03-11 09:30" → "2026-03-11")
        trade_id = trade['id']

        # Fetch price history since entry
        today = datetime.now().strftime('%Y-%m-%d')
        price_history = self.fetch_price_history(ticker, entry_date, today)
        
        if price_history is None or len(price_history) == 0:
            # History fetch failed — try a direct current price lookup
            cp = self.get_current_price(ticker)
            if cp and cp > 0:
                pnl = (cp - trade['entry_price']) * trade['position_size']
                pnl_pct = ((cp / trade['entry_price']) - 1) * 100
                trade['current_price'] = round(cp, 2)
                trade['unrealized_pnl'] = round(pnl, 2)
                trade['unrealized_pnl_pct'] = round(pnl_pct, 2)
            else:
                trade['current_price'] = trade['entry_price']
                trade['unrealized_pnl'] = 0
                trade['unrealized_pnl_pct'] = 0
            trade['days_held'] = 0
            # Persist so GET /api/trades returns real values
            self.storage.update_trade(trade_id, {
                'current_price': trade['current_price'],
                'unrealized_pnl': trade['unrealized_pnl'],
                'unrealized_pnl_pct': trade['unrealized_pnl_pct'],
            }, user_id)
            return trade
        
        # Check exit conditions (V3: returns trailing_stop too)
        status, exit_price, exit_date, reason, trailing_stop = self.check_exit_conditions(
            trade, price_history
        )
        
        # Always update trailing stop in storage
        if trailing_stop != (trade.get('trailing_stop') or trade['stop_loss']):
            self.storage.update_trade(trade_id, {'trailing_stop': trailing_stop}, user_id)
            trade['trailing_stop'] = trailing_stop

        if status == 'T1_PARTIAL':
            # ── v3.1: Partial exit at T1 — sell 50%, keep trade OPEN ──
            target_2 = trade.get('target_2') or trade['target']
            self.storage.update_trade(trade_id, {
                'partial_exit_price': exit_price,
                'partial_exit_pct': 50.0,
                'stop_loss': trade['entry_price'],       # breakeven
                'trailing_stop': trade['entry_price'],    # reset trail to breakeven
                'target': target_2,                       # now targeting T2
                'notes': reason,
            }, user_id)
            trade['partial_exit_price'] = exit_price
            trade['partial_exit_pct'] = 50.0
            trade['stop_loss'] = trade['entry_price']
            trade['trailing_stop'] = trade['entry_price']
            trade['target'] = target_2
            trade['notes'] = reason
            # Keep status OPEN — remaining 50% still in play
            trade['status'] = 'OPEN'

            logger.info(
                f"T1 partial exit {trade['ticker']}: sold 50% at ${exit_price:.2f}, "
                f"stop→breakeven ${trade['entry_price']:.2f}, new target T2 ${target_2:.2f}"
            )

        elif status != 'OPEN':
            # Trade should be fully closed
            self.storage.close_trade(
                trade_id, exit_price, exit_date, status, reason, user_id
            )
            trade['status'] = status
            trade['exit_price'] = exit_price
            trade['exit_date'] = exit_date
            trade['notes'] = reason

            # Calculate realized P/L (blended if partial exit exists)
            partial_price = trade.get('partial_exit_price') or 0
            if partial_price > 0:
                half = trade['position_size'] // 2
                rest = trade['position_size'] - half
                pnl = half * (partial_price - trade['entry_price']) + rest * (exit_price - trade['entry_price'])
                total_cost = trade['position_size'] * trade['entry_price']
                pnl_pct = (pnl / total_cost) * 100 if total_cost > 0 else 0
            else:
                pnl = (exit_price - trade['entry_price']) * trade['position_size']
                pnl_pct = ((exit_price / trade['entry_price']) - 1) * 100

            trade['realized_pnl'] = round(pnl, 2)
            trade['realized_pnl_pct'] = round(pnl_pct, 2)
            trade['current_price'] = exit_price
            trade['unrealized_pnl'] = 0
            trade['unrealized_pnl_pct'] = 0
        else:
            # Still open - calculate unrealized P/L
            current_price = float(price_history['Close'].iloc[-1])
            pnl = (current_price - trade['entry_price']) * trade['position_size']
            pnl_pct = ((current_price / trade['entry_price']) - 1) * 100

            trade['current_price'] = round(current_price, 2)
            trade['unrealized_pnl'] = round(pnl, 2)
            trade['unrealized_pnl_pct'] = round(pnl_pct, 2)

            # Persist to DB so GET /api/trades returns live values immediately
            self.storage.update_trade(trade_id, {
                'current_price': trade['current_price'],
                'unrealized_pnl': trade['unrealized_pnl'],
                'unrealized_pnl_pct': trade['unrealized_pnl_pct'],
            }, user_id)

            # Calculate days held
            entry_dt = datetime.strptime(entry_date[:10], '%Y-%m-%d').date()
            trade['days_held'] = (datetime.now().date() - entry_dt).days

        return trade
    
    def update_all_open_trades(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        Update all open trades.
        
        Returns:
            List of updated trade dicts
        """
        open_trades = self.storage.get_open_trades(user_id)
        updated_trades = []
        
        for trade in open_trades:
            updated = self.update_trade_status(trade, user_id)
            updated_trades.append(updated)
        
        return updated_trades
    
    def add_trade_from_signal(self, signal: Dict) -> int:
        """
        Add a paper trade from a SmallCap signal.
        
        V3: Saves as PENDING status. Entry will be confirmed at next-day Open
        when user clicks 'Update Prices' (with gap filter applied).
        
        Args:
            signal: Signal dict from SmallCapEngine
        
        Returns:
            Trade ID
        """
        signal_price = signal['entry_price']  # Today's close
        
        # Calculate ATR for trailing stop
        atr = 0
        try:
            from ..small_cap.risk import SmallCapRisk
            risk = SmallCapRisk()
            ticker = signal['ticker']
            stock = yf.Ticker(ticker)
            hist = stock.history(period='3mo')
            if hist is not None and len(hist) >= 14:
                hist = hist.reset_index()
                hist.columns = [c if c != 'Date' else 'Date' for c in hist.columns]
                atr = risk.calculate_atr(hist)
        except Exception as e:
            logger.warning(f"Could not calculate ATR for {signal['ticker']}: {e}")
        
        trade = {
            'ticker': signal['ticker'],
            'entry_date': signal.get('date', datetime.now(tz=_NYSE_TZ).strftime('%Y-%m-%d')) + ' ' + datetime.now(tz=_NYSE_TZ).strftime('%H:%M'),
            'entry_price': signal_price,  # Will be updated at confirmation
            'stop_loss': signal['stop_loss'],
            'target': signal['target_1'],
            'target_2': signal.get('target_2', signal['target_1']),  # v3.1: store T2
            'swing_type': signal.get('swing_type', 'A'),
            'quality_score': signal.get('quality_score', 0),
            'position_size': signal.get('position_size', 100),
            'max_hold_days': signal.get('hold_days_max', 7),
            'notes': f"PENDING — Ertesi gün açılışta girilecek. Sinyal: ${signal_price:.2f}",
            'trailing_stop': signal['stop_loss'],
            'initial_stop': signal['stop_loss'],
            'atr': round(atr, 4) if atr else 0,
            'signal_price': signal_price,
            'status': 'PENDING'
        }
        
        # Check for duplicate (sadece tarih kısmıyla karşılaştır)
        date_only = trade['entry_date'][:10]
        if self.storage.check_duplicate(trade['ticker'], date_only, user_id):
            logger.warning(f"Trade already exists: {trade['ticker']} on {date_only}")
            return -1
        
        return self.storage.add_trade(trade, user_id)
    
    def confirm_pending_trades(self) -> List[Dict]:
        """
        Confirm PENDING trades at next-day Open price.
        
        V3 Logic:
        1. Fetch next trading day's Open price
        2. Apply gap filter (>5% up or >3% down = REJECT)
        3. Recalculate stop/target based on actual Open entry
        4. Change status to OPEN
        
        Returns:
            List of confirmed/rejected trade dicts
        """
        results = []
        open_trades = self.storage.get_open_trades(user_id)  # Includes PENDING
        pending = [t for t in open_trades if t.get('status') == 'PENDING']
        
        today_str = datetime.now().strftime('%Y-%m-%d')
        for trade in pending:
            ticker = trade['ticker']
            signal_price = trade.get('signal_price') or trade['entry_price']
            signal_date = trade['entry_date'][:10]  # sadece YYYY-MM-DD kısmı
            trade_id = trade['id']
            
            # Fetch price data: wider range to handle weekends, holidays, and signal_date = today
            try:
                start_dt = datetime.strptime(signal_date, '%Y-%m-%d')
                fetch_start = (start_dt - timedelta(days=10)).strftime('%Y-%m-%d')
                fetch_end = (start_dt + timedelta(days=15)).strftime('%Y-%m-%d')
                price_data = self.fetch_price_history(ticker, fetch_start, fetch_end)
                
                if price_data is None or len(price_data) == 0:
                    logger.warning(
                        f"[{ticker}] PENDING kalıyor: yfinance veri döndürmedi "
                        f"(signal_date={signal_date})"
                    )
                    trade['confirm_status'] = 'waiting'
                    results.append(trade)
                    continue
                
                # Find first trading day AFTER signal_date (next-day Open)
                signal_dt = start_dt.date() if hasattr(start_dt, 'date') else start_dt
                next_days = price_data[price_data['Date'] > signal_dt]
                
                if len(next_days) == 0:
                    logger.info(
                        f"[{ticker}] PENDING kalıyor: signal_date={signal_date} sonrası henüz işlem günü yok "
                        f"(bugün son veri: {today_str})"
                    )
                    trade['confirm_status'] = 'waiting'
                    results.append(trade)
                    continue
                
                next_day = next_days.iloc[0]
                open_price = float(next_day['Open'])
                entry_date = str(next_day['Date']).split(' ')[0] + f' {NYSE_OPEN_TIME}'  # NYSE open (ET)
                
                # Gap filter
                gap_pct = ((open_price / signal_price) - 1) * 100
                
                if gap_pct > MAX_GAP_UP_PCT:
                    # Gap-up too big — momentum exhausted, reject
                    self.storage.close_trade(
                        trade_id, signal_price, entry_date, 'REJECTED',
                        f"REJECTED: Gap-up {gap_pct:+.1f}% > {MAX_GAP_UP_PCT}% limit",
                        user_id
                    )
                    trade['confirm_status'] = 'rejected'
                    trade['reject_reason'] = f"Gap-up {gap_pct:+.1f}%"
                    results.append(trade)
                    continue
                
                if gap_pct < -MAX_GAP_DOWN_PCT:
                    # Gap-down too big — bad news, reject
                    self.storage.close_trade(
                        trade_id, signal_price, entry_date, 'REJECTED',
                        f"REJECTED: Gap-down {gap_pct:+.1f}% > {MAX_GAP_DOWN_PCT}% limit",
                        user_id
                    )
                    trade['confirm_status'] = 'rejected'
                    trade['reject_reason'] = f"Gap-down {gap_pct:+.1f}%"
                    results.append(trade)
                    continue
                
                # Recalculate stop/target based on actual entry (Open)
                atr = trade.get('atr') or 0
                if atr > 0:
                    # ATR-based stop/target
                    stop_loss = round(open_price - (atr * 1.0), 2)
                    # Cap stop at MAX_STOP_PERCENT (15%)
                    max_stop = open_price * 0.85
                    if stop_loss < max_stop:
                        stop_loss = round(max_stop, 2)

                    # Target = entry + 3 × risk distance
                    risk_distance = open_price - stop_loss
                    target = round(open_price + (risk_distance * 3), 2)
                else:
                    # Fallback: use percentage-based from original signal
                    orig_stop_pct = (signal_price - trade['stop_loss']) / signal_price
                    orig_target_pct = (trade['target'] - signal_price) / signal_price
                    stop_loss = round(open_price * (1 - orig_stop_pct), 2)
                    target = round(open_price * (1 + orig_target_pct), 2)

                # v3.1: Recalculate T2 proportionally to new entry
                orig_t2 = trade.get('target_2') or trade['target']
                if signal_price > 0 and orig_t2 > 0:
                    t2_pct = (orig_t2 / signal_price) - 1
                    target_2 = round(open_price * (1 + t2_pct), 2)
                else:
                    target_2 = target

                # Confirm the trade
                self.storage.update_trade(trade_id, {
                    'status': 'OPEN',
                    'entry_price': round(open_price, 2),
                    'entry_date': entry_date,
                    'stop_loss': stop_loss,
                    'initial_stop': stop_loss,
                    'trailing_stop': stop_loss,
                    'target': target,
                    'target_2': target_2,
                    'notes': f"Confirmed at Open ${open_price:.2f} (gap {gap_pct:+.1f}%)"
                }, user_id)
                
                trade['confirm_status'] = 'confirmed'
                trade['entry_price'] = open_price
                trade['gap_pct'] = gap_pct
                results.append(trade)
                
                logger.info(
                    f"Confirmed {ticker}: Open ${open_price:.2f} "
                    f"(gap {gap_pct:+.1f}%), stop ${stop_loss:.2f}, target ${target:.2f}"
                )
                
            except Exception as e:
                logger.error(f"Error confirming {ticker}: {e}")
                trade['confirm_status'] = 'error'
                results.append(trade)
        
        return results
    
    def manual_close_trade(
        self, 
        trade_id: int, 
        exit_price: float = None,
        notes: str = "Manually closed",
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Manually close a trade.
        
        If exit_price is None, uses current price.
        """
        trade = self.storage.get_trade_by_id(trade_id, user_id)
        if not trade:
            return False
        
        if exit_price is None:
            exit_price = self.get_current_price(trade['ticker'])
            if exit_price is None:
                exit_price = trade['entry_price']
        
        exit_date = datetime.now().strftime('%Y-%m-%d')
        
        return self.storage.close_trade(
            trade_id, exit_price, exit_date, 'MANUAL', notes, user_id
        )
