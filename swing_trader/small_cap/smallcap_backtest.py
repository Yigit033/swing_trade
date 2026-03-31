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
    - Execution: optional slippage below; live API does not model fills. No commission model (US zero-commission assumption).
    """

    # Tunable backtest / entry parameters: SmallCapSettings (data/smallcap_settings.json).

    def __init__(self, config: Dict = None):
        self.engine = SmallCapEngine(config)
        self.settings = self.engine.settings
        self.risk = SmallCapRisk(config, self.settings)
        self.config = config or {}
        self.trades = []
        self.open_trades = []
        self.equity_curve = []
        self.initial_capital = 10000
        self.capital = self.initial_capital
        self.cooldowns = {}  # ticker -> date when cooldown expires
        self.ticker_losses = {}  # ticker -> count of stop-out losses (for permanent ban)
        self.banned_tickers = set()  # permanently banned after TICKER_MAX_LOSSES
        self.pending_signals = []  # Signals waiting for next-day entry
        self._peak_equity = 10000
        self._spy_df: Optional[pd.DataFrame] = None
        self._vix_df: Optional[pd.DataFrame] = None
        self.min_quality = 65
        self.top_n = 10
        self._max_concurrent = 3
        self._diag: Dict = {}

    def _reset_diagnostics(self) -> None:
        self._diag = {
            "signals_passed_rsi": 0,
            "pending_queued": 0,
            "entry_skip_gap_up": 0,
            "entry_skip_gap_down": 0,
            "entry_skip_tip_c": 0,
            "entry_skip_risk": 0,
            "entry_skip_rr": 0,
            "entry_skip_trend": 0,
            "entries_opened": 0,
        }

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
        self.ticker_losses = {}
        self.banned_tickers = set()
        self.pending_signals = []
        self._peak_equity = initial_capital
        self._reset_diagnostics()
        
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
            
            # Compute regime BEFORE entry decisions
            regime, eff_min, eff_top, spy_rs = self._daily_regime_bundle(current_date_dt)
            
            # Regime-aware concurrent limit for entries
            regime_name = regime.get('regime', 'UNKNOWN')
            regime_conf = regime.get('confidence', 'TENTATIVE')
            bl = self.settings.backtest_loop
            if regime_name == 'BEAR' and bl.bear_block_new_entries:
                entry_max = 0
            elif regime_name == 'CAUTION':
                entry_max = min(max_concurrent, bl.caution_max_concurrent)
            else:
                entry_max = max_concurrent
            
            # Drawdown circuit breaker: reduce size or pause if losing too much
            portfolio_value = self._calculate_portfolio_value(current_date_dt, data_dict)
            current_dd = (self._peak_equity - portfolio_value) / self._peak_equity if self._peak_equity > 0 else 0
            if portfolio_value > self._peak_equity:
                self._peak_equity = portfolio_value
            
            if current_dd > bl.drawdown_pause_entries_fraction:
                entry_max = 0
            elif current_dd > bl.drawdown_reduce_to_one_position_fraction:
                entry_max = min(entry_max, 1)
            
            # 2b: Process pending signals from YESTERDAY (next-day-open entry)
            if self.pending_signals and entry_max > 0 and len(self.open_trades) < entry_max:
                self._process_pending_entries(current_date_dt, data_dict, entry_max)
            
            # 2c: Record equity (+ daily regime / thresholds for UI)
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
            if entry_max > 0 and len(self.open_trades) < max_concurrent:
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
            'diagnostics': dict(self._diag),
            'params': {
                'min_quality': self.min_quality,
                'top_n': self.top_n,
                'max_concurrent': self._max_concurrent,
                'slippage_bps_per_side': self.settings.slippage_bps_per_side,
                'min_rr_at_entry': self.settings.min_rr_at_entry,
                'partial_at_t1_fraction': self.settings.partial_at_t1_fraction,
            },
        }

    def _entry_fill_price(self, quoted: float) -> float:
        b = self.settings.slippage_bps_per_side / 10000.0
        return round(quoted * (1.0 + b), 2)

    def _exit_fill_price(self, quoted: float) -> float:
        if quoted <= 0:
            return quoted
        b = self.settings.slippage_bps_per_side / 10000.0
        return round(quoted * (1.0 - b), 2)

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
                r["regime"],
                r["confidence"],
                float(r["score_multiplier"]),
                self.min_quality,
                self.top_n,
                regime_caps=self.settings.regime_thresholds,
            )
            return r, eff_min, eff_top, pd.DataFrame()

        m = self._spy_df[self._spy_df["Date"] <= current_date]
        if len(m) < 50:
            r = regime_unknown("insufficient_spy_history")
            eff_min, eff_top = effective_scan_thresholds(
                r["regime"],
                r["confidence"],
                float(r["score_multiplier"]),
                self.min_quality,
                self.top_n,
                regime_caps=self.settings.regime_thresholds,
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
            regime_caps=self.settings.regime_thresholds,
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
        regime_name = regime.get('regime', 'UNKNOWN')
        regime_conf = regime.get('confidence', 'TENTATIVE')
        
        bl = self.settings.backtest_loop
        if regime_name == 'BEAR' and bl.bear_block_new_entries:
            return
        elif regime_name == 'CAUTION':
            effective_max = min(max_concurrent, bl.caution_max_concurrent)
        else:
            effective_max = max_concurrent
        
        open_tickers = {t["ticker"] for t in self.open_trades}
        pending_tickers = {s["signal"]["ticker"] for s in self.pending_signals}
        slots = effective_max - len(self.open_trades) - len(self.pending_signals)
        if slots <= 0:
            return

        candidates: List[Dict] = []

        for ticker in sorted(data_dict.keys()):
            if ticker in open_tickers or ticker in pending_tickers:
                continue

            if ticker in self.banned_tickers:
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
                if signal_rsi > self.settings.max_entry_rsi:
                    continue

                candidates.append(
                    {"ticker": ticker, "signal": signal, "df_window": df_window}
                )
                self._diag["signals_passed_rsi"] += 1
            except Exception as e:
                logger.debug(f"Scan error {ticker}: {e}")

        candidates.sort(key=lambda x: x["signal"].get("quality_score", 0), reverse=True)
        filtered = []
        for c in candidates:
            oqs = c["signal"].get("original_quality_score", c["signal"]["quality_score"])
            if oqs < eff_min:
                continue
            stype = c["signal"].get("swing_type")
            tq = self.settings.backtest_type_quality
            if stype == "C":
                c_min = self.settings.min_quality_type_c
                if regime_name == "BEAR":
                    c_min = tq.type_c_bear
                elif regime_name == "CAUTION":
                    c_min = tq.type_c_caution
                if oqs < c_min:
                    continue
            elif stype == "A":
                a_min = self.settings.min_quality_type_a
                if regime_name == "BEAR":
                    a_min = tq.type_a_bear
                elif regime_name == "CAUTION":
                    a_min = tq.type_a_caution
                if oqs < a_min:
                    continue
            elif stype == "B":
                b_min = self.settings.min_quality_type_b
                if regime_name == "BEAR":
                    b_min = tq.type_b_bear
                elif regime_name == "CAUTION":
                    b_min = tq.type_b_caution
                if oqs < b_min:
                    continue
            filtered.append(c)
        filtered = filtered[:eff_top]

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
            self._diag["pending_queued"] += 1
    
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
            
            if gap_pct > self.settings.max_gap_up_pct:
                self._diag["entry_skip_gap_up"] += 1
                logger.debug(f"{ticker}: Gap UP {gap_pct:+.1f}% > {self.settings.max_gap_up_pct}% — skipped")
                continue
            
            if gap_pct < -self.settings.max_gap_down_pct:
                self._diag["entry_skip_gap_down"] += 1
                logger.debug(f"{ticker}: Gap DOWN {gap_pct:+.1f}% < -{self.settings.max_gap_down_pct}% — skipped")
                continue
            
            # ── Type C: open-only confirmation (no same-day close — not knowable at open)
            be = self.settings.backtest_entry
            if signal.get("swing_type") == "C":
                if today_open < signal_close * be.type_c_min_open_vs_signal_close_ratio:
                    self._diag["entry_skip_tip_c"] += 1
                    continue
            
            # ── All filters passed: Enter at today's Open (+ adverse slippage) ──
            entry_price = self._entry_fill_price(today_open)
            
            # Use entry-day data (not signal-day) for accurate stop/target
            full_df = data_dict.get(ticker)
            df_entry = full_df[full_df['Date'] <= current_date].copy() if full_df is not None else pending['df_window']
            
            # EMA trend: completed bars only (no same-day close lookahead)
            min_b = be.trend_min_bars
            df_prior = df_entry.iloc[:-1] if len(df_entry) > min_b else df_entry
            if len(df_prior) >= min_b:
                ema10 = float(
                    df_prior["Close"]
                    .ewm(span=be.trend_ema_fast_span, adjust=False)
                    .mean()
                    .iloc[-1]
                )
                ema20 = float(
                    df_prior["Close"]
                    .ewm(span=be.trend_ema_slow_span, adjust=False)
                    .mean()
                    .iloc[-1]
                )
                if entry_price < ema10 and entry_price < ema20:
                    self._diag["entry_skip_trend"] += 1
                    continue
            
            sig = signal.copy()
            sig['entry_price'] = entry_price
            regime_str = sig.get('market_regime', '')
            
            try:
                self.risk.add_risk_management(
                    sig, df_entry,
                    portfolio_value=self.capital,
                    regime=regime_str,
                )
            except Exception as e:
                self._diag["entry_skip_risk"] += 1
                logger.debug(f"{ticker}: add_risk_management failed on entry: {e}")
                continue
            
            # Recalculate stop relative to actual entry_price (not signal-day close)
            sl = float(sig.get('stop_loss') or 0)
            atr_val = self.risk.calculate_atr(df_entry)
            swing_type = sig.get('swing_type', 'A')
            max_stop_pct = self.risk.MAX_STOP_BY_TYPE.get(swing_type, self.risk.MAX_STOP_PERCENT)
            
            if atr_val > 0:
                atr_stop = entry_price - (self.risk.STOP_ATR_MULTIPLIER * atr_val)
                swing_low = float(df_entry['Low'].tail(5).min()) * 0.995
                entry_max_stop = entry_price * (1 - max_stop_pct)
                # WIDER stop: further from entry = larger % risk from ATR vs swing
                atr_dist = (entry_price - atr_stop) / entry_price if entry_price > 0 else 0.0
                swing_dist = (entry_price - swing_low) / entry_price if entry_price > 0 else 0.0
                if atr_dist >= swing_dist:
                    sl = atr_stop
                else:
                    sl = swing_low
                sl = max(sl, entry_max_stop)
                sl = min(sl, entry_price * 0.995)  # must be below entry
                hard_floor = entry_price * (1 - self.settings.max_loss_per_trade_pct)
                sl = max(sl, hard_floor)
                sig['stop_loss'] = round(sl, 2)
            
            if int(sig.get('position_size') or 0) < 1:
                self._diag["entry_skip_risk"] += 1
                continue
            if sl <= 0 or sl >= entry_price:
                self._diag["entry_skip_risk"] += 1
                continue
            
            # Recalculate targets based on actual entry_price and new stop
            quality_score = int(sig.get('original_quality_score') or sig.get('quality_score') or 0)
            t1, t2 = self.risk.calculate_targets(
                entry_price, sl, swing_type,
                atr=atr_val, quality_score=quality_score, regime=regime_str,
            )
            sig['target_1'] = t1
            sig['target_2'] = t2
            
            # Recalculate position size with new stop
            shares, _ = self.risk.calculate_position_size(
                self.capital, entry_price, sl, swing_type
            )
            # Gap-aware cap: plan 2×ATR; also vs full stop distance (gap-through-stop can exceed 2×ATR)
            if atr_val > 0 and entry_price > 0:
                atr_pct_at_entry = atr_val / entry_price
                gap_mult = self.settings.backtest_entry.gap_atr_multiplier
                two_atr = entry_price * atr_pct_at_entry * gap_mult
                stop_width = max(entry_price - sl, 1e-9)
                max_gap_loss_per_share = max(two_atr, stop_width)
                if max_gap_loss_per_share > 1e-9:
                    gap_cap = int(
                        (self.capital * self.settings.max_gap_risk_portfolio_pct)
                        / max_gap_loss_per_share
                    )
                    shares = min(shares, max(0, gap_cap))
            # Hard notional cap (survive total loss / large gap)
            max_sh_cost = int(
                (self.capital * self.settings.max_position_cost_portfolio_pct) / entry_price
            ) if entry_price > 0 else 0
            shares = min(shares, max(0, max_sh_cost))
            sig['position_size'] = shares
            if shares < 1:
                self._diag["entry_skip_risk"] += 1
                continue
            
            # R:R check with actual entry-based values
            risk_px = entry_price - sl
            reward_px = t1 - entry_price
            min_rr = self.settings.min_rr_type_c if swing_type == 'C' else self.settings.min_rr_at_entry
            if risk_px <= 0 or reward_px / risk_px < min_rr:
                self._diag["entry_skip_rr"] += 1
                logger.debug(
                    f"{ticker}: Entry R:R {reward_px/risk_px:.2f} < min {min_rr} — skipped"
                )
                continue
            self._open_trade(sig, current_date, df_today)
            self._diag["entries_opened"] += 1
        
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
        target_2 = float(signal.get("target_2") or 0)
        if target <= entry_price:
            risk_amount = entry_price - stop_loss
            target = entry_price + (risk_amount * 2.0)  # fallback min ~2R if target_1 missing
        if target_2 <= target:
            target_2 = target * self.settings.backtest_entry.partial_fallback_target_bump
        
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
            'target_2': round(target_2, 2),
            'stop_pct': round(stop_pct, 1),
            'target_pct': round(target_pct, 1),
            'shares': shares,
            'initial_shares': shares,
            'partial_done': False,
            'partial_pnl_dollar': 0.0,
            'partial_shares': 0,
            'partial_exit_price': 0.0,
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
        """Check exits: trailing stop, optional partial at T1 + BE, T2, timeout."""
        still_open = []
        
        for trade in self.open_trades:
            ticker = trade['ticker']
            df = data_dict.get(ticker)
            
            if df is None:
                still_open.append(trade)
                continue
            
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
            entry_price = trade['entry_price']
            shares = trade['shares']
            
            xt = self.settings.backtest_exit_trailing
            # Time-based exit: stale losers only after more room
            if (
                days_held >= xt.time_stop_min_days
                and current_close < entry_price
                and not trade.get("partial_done")
            ):
                loss_pct = (entry_price - current_close) / entry_price
                if loss_pct > xt.time_stop_min_loss_fraction:
                    trade['exit_price'] = self._exit_fill_price(current_close)
                    trade['exit_date'] = current_date.strftime('%Y-%m-%d')
                    trade['exit_reason'] = f'Time Stop ({days_held}d, {loss_pct*100:.1f}% loss)'
                    trade['status'] = 'STOPPED'
                    trade['days_held'] = days_held
                    self._close_trade(trade)
                    self.cooldowns[ticker] = current_date + timedelta(days=self.settings.cooldown_days)
                    self.ticker_losses[ticker] = self.ticker_losses.get(ticker, 0) + 1
                    if self.ticker_losses[ticker] >= self.settings.ticker_max_losses:
                        self.banned_tickers.add(ticker)
                    continue
            
            # Aggressive trailing stop: protect gains quickly
            if atr > 0 and current_high > entry_price:
                peak_gain = current_high - entry_price
                atr_gain_peak = peak_gain / atr
                close_gain = max(current_close - entry_price, 0)
                atr_gain_close = close_gain / atr if current_close > entry_price else 0
                new_trail = trade['trailing_stop']

                if atr_gain_peak >= xt.trail_peak_atr_25:
                    new_trail = max(new_trail, current_high - xt.trail_high_minus_atr_25 * atr)
                elif atr_gain_peak >= xt.trail_peak_atr_20:
                    new_trail = max(new_trail, entry_price + (peak_gain * xt.trail_peak_frac_20))
                elif atr_gain_peak >= xt.trail_peak_atr_15:
                    new_trail = max(new_trail, entry_price + (peak_gain * xt.trail_peak_frac_15))
                
                if atr_gain_peak >= xt.breakeven_peak_atr:
                    new_trail = max(new_trail, entry_price)
                elif atr_gain_peak >= xt.light_protect_peak_atr:
                    new_trail = max(new_trail, entry_price - xt.light_protect_below_entry_atr * atr)
                
                if atr_gain_close >= xt.close_gain_atr_20:
                    new_trail = max(new_trail, current_close - xt.close_trail_atr_20 * atr)
                elif atr_gain_close >= xt.close_gain_atr_15:
                    new_trail = max(new_trail, current_close - xt.close_trail_atr_15 * atr)

                if new_trail > trade['trailing_stop']:
                    trade['trailing_stop'] = round(new_trail, 2)
            
            active_stop = trade['trailing_stop']
            
            if current_low <= active_stop:
                today_open = float(bar['Open'])
                if today_open <= active_stop:
                    raw_exit = today_open
                else:
                    raw_exit = active_stop
                is_trail = active_stop > trade['initial_stop']
                trade['exit_price'] = self._exit_fill_price(raw_exit)
                trade['exit_date'] = current_date.strftime('%Y-%m-%d')
                if trade.get('partial_done'):
                    trade['exit_reason'] = 'Trailing Stop (kalan)' if is_trail else 'Stop Loss (kalan)'
                else:
                    trade['exit_reason'] = 'Trailing Stop' if is_trail else 'Stop Loss'
                trade['status'] = 'TRAILED' if is_trail else 'STOPPED'
                trade['days_held'] = days_held
                self._close_trade(trade)
                self.cooldowns[ticker] = current_date + timedelta(days=self.settings.cooldown_days)
                # Track losses per ticker — ban after too many
                if not is_trail:
                    self.ticker_losses[ticker] = self.ticker_losses.get(ticker, 0) + 1
                    if self.ticker_losses[ticker] >= self.settings.ticker_max_losses:
                        self.banned_tickers.add(ticker)
                continue
            
            if not trade.get('partial_done') and current_high >= trade['target']:
                if shares < self.settings.min_shares_for_partial:
                    trade['exit_price'] = self._exit_fill_price(float(trade['target']))
                    trade['exit_date'] = current_date.strftime('%Y-%m-%d')
                    trade['exit_reason'] = 'Hedef T1 (tam)'
                    trade['status'] = 'TARGET'
                    trade['days_held'] = days_held
                    self._close_trade(trade)
                    continue
                
                ps = max(1, int(shares * self.settings.partial_at_t1_fraction))
                if ps >= shares:
                    ps = max(1, shares - 1)
                px = self._exit_fill_price(float(trade['target']))
                leg_pnl = (px - entry_price) * ps
                self.capital += leg_pnl
                trade['partial_pnl_dollar'] = round(leg_pnl, 2)
                trade['partial_shares'] = ps
                trade['partial_exit_price'] = px
                trade['shares'] = shares - ps
                trade['partial_done'] = True
                trade['trailing_stop'] = max(trade['trailing_stop'], entry_price)
                trade['initial_stop'] = max(trade['initial_stop'], entry_price)
                trade['target'] = round(float(trade.get('target_2') or trade['target'] * 1.15), 2)
                still_open.append(trade)
                continue
            
            if trade.get('partial_done') and current_high >= trade['target']:
                trade['exit_price'] = self._exit_fill_price(float(trade['target']))
                trade['exit_date'] = current_date.strftime('%Y-%m-%d')
                trade['exit_reason'] = 'Hedef T2 (kısmi sonrası)'
                trade['status'] = 'TARGET_T2'
                trade['days_held'] = days_held
                self._close_trade(trade)
                continue
            
            if days_held >= trade['max_hold_days']:
                exit_px = current_close
                if not trade.get('partial_done') and current_high >= trade['target']:
                    exit_px = float(trade['target'])
                trade['exit_price'] = self._exit_fill_price(exit_px)
                trade['exit_date'] = current_date.strftime('%Y-%m-%d')
                trade['exit_reason'] = f'Timeout ({days_held}d)'
                trade['status'] = 'TIMEOUT'
                trade['days_held'] = days_held
                self._close_trade(trade)
                # Timeout on a losing trade = count as a loss for that ticker
                if exit_px < entry_price:
                    self.ticker_losses[ticker] = self.ticker_losses.get(ticker, 0) + 1
                    if self.ticker_losses[ticker] >= self.settings.ticker_max_losses:
                        self.banned_tickers.add(ticker)
                continue
            
            still_open.append(trade)
        
        self.open_trades = still_open
    
    def _close_trade(self, trade: Dict):
        """Close remaining shares; sum partial leg if any. Updates capital for remainder only."""
        entry = trade['entry_price']
        last_exit = float(trade['exit_price'])
        shares = trade.get('shares', 100)
        init_sh = int(trade.get('initial_shares') or shares)
        partial_pnl = float(trade.get('partial_pnl_dollar') or 0)
        ps = int(trade.get('partial_shares') or 0)
        ppx = float(trade.get('partial_exit_price') or 0)
        
        rem_leg = (last_exit - entry) * shares
        if ps > 0:
            self.capital += rem_leg
            total = partial_pnl + rem_leg
            trade['pnl_dollar'] = round(total, 2)
            trade['exit_price'] = round((ppx * ps + last_exit * shares) / init_sh, 2)
            trade['pnl_pct'] = round((total / (entry * init_sh)) * 100, 2) if entry * init_sh > 0 else 0
        else:
            self.capital += rem_leg
            trade['pnl_dollar'] = round(rem_leg, 2)
            trade['pnl_pct'] = round(((last_exit / entry) - 1) * 100, 2) if entry > 0 else 0
        
        trade['shares'] = init_sh
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
            'diagnostics': dict(getattr(self, '_diag', {}) or {}),
            'params': {
                'min_quality': getattr(self, 'min_quality', 65),
                'top_n': getattr(self, 'top_n', 10),
                'max_concurrent': getattr(self, '_max_concurrent', 3),
                'slippage_bps_per_side': self.settings.slippage_bps_per_side,
                'min_rr_at_entry': self.settings.min_rr_at_entry,
                'partial_at_t1_fraction': self.settings.partial_at_t1_fraction,
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
