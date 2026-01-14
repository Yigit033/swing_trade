"""
Signal generation module for buy/sell signals.
VERSION 3: Market regime, earnings filter, trend-based hold, MACD soft-only
"""

import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ..indicators.trend import calculate_trend_indicators, calculate_bollinger_bands, calculate_support_resistance
from ..indicators.momentum import calculate_momentum_indicators
from ..indicators.volume import calculate_volume_indicators, calculate_atr

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generates buy and sell signals based on technical indicators.
    
    VERSION 3 FEATURES:
    - MACD soft-only (no hard gate)
    - Trend-based hold period (ADX + EMA slope)
    - Market regime filter (SPY-based)
    - Earnings HARD gate
    - Trend-normalized extension penalty
    """
    
    def __init__(self, config: Dict):
        """Initialize SignalGenerator."""
        self.config = config
        self._earnings_cache = {}  # Cache earnings dates
        self._market_regime = 'RISK_ON'  # Default
        logger.info("SignalGenerator v3 initialized (Market Regime + Earnings Filter)")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators for signal generation."""
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        try:
            df = calculate_trend_indicators(df, self.config)
            df = calculate_momentum_indicators(df, self.config)
            df = calculate_volume_indicators(df, self.config)
            
            atr_period = self.config['indicators'].get('atr_period', 14)
            df['ATR'] = calculate_atr(
                df['High'], df['Low'], df['Close'], period=atr_period
            )
            
            bb_period = self.config['indicators'].get('bb_period', 20)
            bb_std = self.config['indicators'].get('bb_std', 2)
            bb = calculate_bollinger_bands(df['Close'], period=bb_period, std_dev=bb_std)
            df['BB_upper'] = bb['upper']
            df['BB_middle'] = bb['middle']
            df['BB_lower'] = bb['lower']
            
            sr_period = self.config['indicators'].get('support_resistance_period', 20)
            levels = calculate_support_resistance(df['High'], df['Low'], period=sr_period)
            df['Support_20'] = levels['support']
            df['Resistance_20'] = levels['resistance']
            
            # Calculate EMA slope for trend-based hold period
            if 'EMA_20' in df.columns:
                df['EMA_20_slope'] = df['EMA_20'].pct_change(5)  # 5-day slope
            
            logger.debug(f"Calculated all indicators for {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return df
    
    # =========================================================================
    # MARKET REGIME FILTER
    # =========================================================================
    def update_market_regime(self, spy_data: pd.DataFrame) -> str:
        """
        Determine market regime based on SPY.
        
        RISK_ON:  SPY > EMA_200 (normal conditions)
        RISK_OFF: SPY < EMA_200 (defensive mode)
        """
        try:
            if spy_data is None or len(spy_data) < 200:
                return 'RISK_ON'
            
            spy_data = self.calculate_all_indicators(spy_data)
            current = spy_data.iloc[-1]
            
            spy_close = current['Close']
            spy_ema200 = current['EMA_200'] if not pd.isna(current['EMA_200']) else spy_close
            
            if spy_close < spy_ema200:
                self._market_regime = 'RISK_OFF'
                logger.warning(f"MARKET REGIME: RISK_OFF (SPY ${spy_close:.2f} < EMA200 ${spy_ema200:.2f})")
            else:
                self._market_regime = 'RISK_ON'
                logger.info(f"MARKET REGIME: RISK_ON (SPY ${spy_close:.2f} > EMA200 ${spy_ema200:.2f})")
            
            return self._market_regime
            
        except Exception as e:
            logger.error(f"Error determining market regime: {e}")
            return 'RISK_ON'
    
    def get_regime_adjusted_params(self) -> Dict:
        """Get parameters adjusted for current market regime."""
        base_min_quality = self.config['strategy'].get('min_quality_score', 45)
        base_top_n = self.config['strategy'].get('top_signals_count', 6)
        
        if self._market_regime == 'RISK_OFF':
            return {
                'min_quality_score': max(base_min_quality + 15, 60),  # Raise threshold
                'top_signals_count': max(base_top_n - 3, 2)  # Fewer signals
            }
        else:
            return {
                'min_quality_score': base_min_quality,
                'top_signals_count': base_top_n
            }
    
    # =========================================================================
    # EARNINGS HARD GATE
    # =========================================================================
    def get_next_earnings_date(self, ticker: str) -> Optional[datetime]:
        """
        Get next earnings date for a ticker.
        Uses yfinance calendar if available.
        """
        try:
            # Check cache first
            cache_key = f"{ticker}_{datetime.now().strftime('%Y-%m-%d')}"
            if cache_key in self._earnings_cache:
                return self._earnings_cache[cache_key]
            
            import yfinance as yf
            stock = yf.Ticker(ticker)
            
            # Try to get earnings dates
            try:
                calendar = stock.calendar
                if calendar is not None and not calendar.empty:
                    # calendar is a DataFrame with earnings dates
                    if 'Earnings Date' in calendar.columns:
                        earnings_dates = calendar['Earnings Date']
                        if len(earnings_dates) > 0:
                            next_earnings = pd.to_datetime(earnings_dates.iloc[0])
                            self._earnings_cache[cache_key] = next_earnings.to_pydatetime()
                            return self._earnings_cache[cache_key]
            except:
                pass
            
            # Alternative: try earnings_dates property
            try:
                earnings_dates = stock.earnings_dates
                if earnings_dates is not None and len(earnings_dates) > 0:
                    future_dates = earnings_dates[earnings_dates.index > datetime.now()]
                    if len(future_dates) > 0:
                        next_earnings = future_dates.index[0].to_pydatetime()
                        self._earnings_cache[cache_key] = next_earnings
                        return next_earnings
            except:
                pass
            
            self._earnings_cache[cache_key] = None
            return None
            
        except Exception as e:
            logger.debug(f"Could not get earnings date for {ticker}: {e}")
            return None
    
    def check_earnings_gate(self, ticker: str, signal_date: datetime) -> Tuple[bool, Optional[str]]:
        """
        Check if signal is blocked by earnings.
        
        Rules:
        - Block 3 days BEFORE earnings
        - Block 1 day AFTER earnings
        """
        try:
            earnings_date = self.get_next_earnings_date(ticker)
            
            if earnings_date is None:
                return True, None  # Allow if unknown
            
            # Ensure both are datetime objects
            if isinstance(signal_date, str):
                signal_date = datetime.strptime(signal_date, '%Y-%m-%d')
            
            days_to_earnings = (earnings_date.date() - signal_date.date()).days
            
            # Block 3 days before
            if 0 <= days_to_earnings <= 3:
                return False, f"Earnings in {days_to_earnings} days"
            
            # Block 1 day after
            if -1 <= days_to_earnings < 0:
                return False, "Just had earnings yesterday"
            
            return True, None
            
        except Exception as e:
            logger.debug(f"Earnings check error for {ticker}: {e}")
            return True, None  # Allow on error
    
    # =========================================================================
    # MACD QUALITY (Soft-only, no gate)
    # =========================================================================
    def calculate_macd_quality(self, df: pd.DataFrame, row_idx: int) -> float:
        """
        Calculate MACD quality score (0-100).
        NO LONGER A GATE - only contributes to quality score.
        """
        try:
            if row_idx < 3:
                return 0
            
            row = df.iloc[row_idx]
            
            # Check for crossover in last 3 days
            for days_ago in range(0, 3):
                idx = row_idx - days_ago
                if idx < 1:
                    break
                curr_hist = df.iloc[idx]['MACD_hist']
                prev_hist = df.iloc[idx - 1]['MACD_hist']
                
                if pd.isna(curr_hist) or pd.isna(prev_hist):
                    continue
                
                if curr_hist > 0 and prev_hist <= 0:
                    return 100 - (days_ago * 15)  # 100, 85, 70
            
            # Check current bullish state
            curr_hist = row['MACD_hist']
            prev_hist = df.iloc[row_idx - 1]['MACD_hist']
            macd = row['MACD'] if 'MACD' in row else 0
            macd_signal = row['MACD_signal'] if 'MACD_signal' in row else 0
            
            if pd.isna(curr_hist) or pd.isna(prev_hist):
                return 0
            
            if curr_hist > 0 and macd > macd_signal:
                if curr_hist > prev_hist:
                    return 55
                else:
                    return 40
            
            if macd > macd_signal and curr_hist > prev_hist:
                return 30
            
            # Even no bullish signal gets some points if MACD is improving
            if curr_hist > prev_hist:
                return 15
            
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating MACD quality: {e}")
            return 0
    
    # =========================================================================
    # TREND-BASED HOLD PERIOD
    # =========================================================================
    def calculate_trend_based_hold(self, row: pd.Series) -> Tuple[int, int]:
        """
        Calculate hold period based on trend strength, not ATR.
        
        Strong trend (ADX > 30, positive EMA slope) = longer hold
        Medium trend (ADX 20-30) = medium hold
        Weak trend (ADX < 20) = short hold
        """
        try:
            adx = row['ADX'] if not pd.isna(row['ADX']) else 20
            ema_slope = row.get('EMA_20_slope', 0)
            if pd.isna(ema_slope):
                ema_slope = 0
            
            # Strong trend
            if adx > 30 and ema_slope > 0.002:
                return (8, 20)
            # Medium-strong trend
            elif adx > 25 and ema_slope > 0:
                return (6, 15)
            # Medium trend
            elif adx > 20:
                return (5, 12)
            # Weak trend
            else:
                return (3, 7)
                
        except Exception:
            return (5, 12)  # Default
    
    # =========================================================================
    # TREND-NORMALIZED EXTENSION PENALTY
    # =========================================================================
    def calculate_extension_penalty(self, row: pd.Series) -> float:
        """
        Calculate penalty for over-extended stocks.
        NORMALIZED BY TREND: Strong trend = softer penalty.
        """
        raw_penalty = 0
        
        try:
            close = row['Close']
            ema_200 = row['EMA_200']
            ema_50 = row['EMA_50'] if 'EMA_50' in row else ema_200
            rsi = row['RSI']
            adx = row['ADX'] if not pd.isna(row['ADX']) else 20
            
            if pd.isna(close) or pd.isna(ema_200):
                return 0
            
            # Distance from EMA_200
            ema_distance = (close - ema_200) / ema_200
            if ema_distance > 0.15:
                raw_penalty += min((ema_distance - 0.15) * 100, 15)
            
            # Distance from EMA_50
            if not pd.isna(ema_50):
                ema50_distance = (close - ema_50) / ema_50
                if ema50_distance > 0.10:
                    raw_penalty += min((ema50_distance - 0.10) * 50, 10)
            
            # RSI extension
            if not pd.isna(rsi) and rsi > 65:
                raw_penalty += (rsi - 65) * 0.5
            
            # TREND NORMALIZATION
            # Strong trend (ADX > 30) = 50% penalty reduction
            # Medium trend (ADX 20-30) = 25% reduction
            # Weak trend (ADX < 20) = full penalty
            if adx > 30:
                trend_factor = 0.5
            elif adx > 20:
                trend_factor = 0.75
            else:
                trend_factor = 1.0
            
            final_penalty = raw_penalty * trend_factor
            return min(final_penalty, 25)  # Reduced max from 30 to 25
            
        except Exception:
            return 0
    
    # =========================================================================
    # QUALITY SCORE CALCULATION
    # =========================================================================
    def calculate_quality_score(self, df: pd.DataFrame, row_idx: int, macd_quality: float) -> float:
        """Calculate composite quality score (0-100)."""
        try:
            row = df.iloc[row_idx]
            
            # === MOMENTUM SCORE (0-30) ===
            rsi = row['RSI'] if not pd.isna(row['RSI']) else 50
            if 35 <= rsi <= 55:
                rsi_score = 15
            elif 30 <= rsi < 35 or 55 < rsi <= 65:
                rsi_score = 10
            elif 25 <= rsi < 30:
                rsi_score = 12
            else:
                rsi_score = 5
            
            macd_score = min(macd_quality * 0.15, 15)
            momentum_score = rsi_score + macd_score
            
            # === MEAN REVERSION SCORE (0-20) ===
            close = row['Close']
            support = row['Support_20'] if not pd.isna(row['Support_20']) else close * 0.95
            support_distance = (close - support) / close
            
            if support_distance < 0.02:
                support_score = 15
            elif support_distance < 0.05:
                support_score = 10
            else:
                support_score = 5
            
            bb_lower = row['BB_lower'] if not pd.isna(row['BB_lower']) else close * 0.95
            bb_middle = row['BB_middle'] if not pd.isna(row['BB_middle']) else close
            bb_position = (close - bb_lower) / (bb_middle - bb_lower) if (bb_middle - bb_lower) > 0 else 0.5
            
            if bb_position < 0.3:
                bb_score = 5
            elif bb_position < 0.5:
                bb_score = 3
            else:
                bb_score = 0
            
            mean_reversion_score = support_score + bb_score
            
            # === VOLUME SCORE (0-20) ===
            volume_surge = row['Volume_surge'] if not pd.isna(row['Volume_surge']) else 1.0
            if volume_surge >= 1.5:
                volume_score = 15
            elif volume_surge >= 1.2:
                volume_score = 10
            elif volume_surge >= 1.0:
                volume_score = 5
            else:
                volume_score = 0
            
            obv_slope = row['OBV_slope'] if not pd.isna(row['OBV_slope']) else 0
            obv_score = 5 if obv_slope > 0 else 0
            volume_total = volume_score + obv_score
            
            # === RISK/REWARD SCORE (0-15) ===
            atr = row['ATR'] if not pd.isna(row['ATR']) else close * 0.02
            rr_ratio = 3 / 2  # 3 ATR target / 2 ATR stop
            rr_score = min(rr_ratio * 5, 15)
            
            # === TREND SCORE (0-15, CAPPED) ===
            ema_20 = row['EMA_20'] if not pd.isna(row['EMA_20']) else close
            ema_50 = row['EMA_50'] if not pd.isna(row['EMA_50']) else close
            ema_200 = row['EMA_200'] if not pd.isna(row['EMA_200']) else close
            
            if ema_20 > ema_50 > ema_200:
                ema_score = 8
            elif close > ema_200 and close > ema_50:
                ema_score = 5
            elif close > ema_200:
                ema_score = 3
            else:
                ema_score = 0
            
            adx = row['ADX'] if not pd.isna(row['ADX']) else 20
            if adx > 25:
                adx_score = min((adx - 20) * 0.3, 7)
            else:
                adx_score = 0
            
            trend_score = min(ema_score + adx_score, 15)
            
            # === TOTAL ===
            raw_score = momentum_score + mean_reversion_score + volume_total + rr_score + trend_score
            extension_penalty = self.calculate_extension_penalty(row)
            
            final_score = max(raw_score - extension_penalty, 0)
            return min(final_score, 100)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0
    
    # =========================================================================
    # HARD CONDITIONS (Simplified - MACD removed)
    # =========================================================================
    def check_hard_conditions(self, df: pd.DataFrame, row_idx: int, ticker: str, signal_date) -> Tuple[bool, Optional[str]]:
        """
        Check HARD conditions (must pass to generate signal).
        
        V3 HARD GATES:
        1. Close > EMA_200 (uptrend)
        2. RSI < 70 (not overbought)
        3. Earnings filter (3 before / 1 after)
        
        NOTE: MACD is NO LONGER a hard gate.
        """
        try:
            row = df.iloc[row_idx]
            
            # 1. TREND: Price above EMA_200
            if pd.isna(row['EMA_200']) or row['Close'] <= row['EMA_200']:
                return False, "Not in uptrend (Close <= EMA200)"
            
            # 2. NOT OVERBOUGHT: RSI < 70
            if not pd.isna(row['RSI']) and row['RSI'] >= 70:
                return False, f"Overbought (RSI={row['RSI']:.1f})"
            
            # 3. EARNINGS FILTER
            passed, reason = self.check_earnings_gate(ticker, signal_date)
            if not passed:
                return False, f"Earnings block: {reason}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking hard conditions: {e}")
            return False, str(e)
    
    def check_entry_conditions(self, row: pd.Series, prev_row: pd.Series) -> Dict:
        """Check soft entry conditions for reporting."""
        strategy_config = self.config['strategy']
        conditions = {}
        
        try:
            conditions['trend'] = (
                not pd.isna(row['Close']) and
                not pd.isna(row['EMA_200']) and
                row['Close'] > row['EMA_200']
            )
            
            rsi_min = strategy_config.get('rsi_entry_min', 25)
            rsi_max = strategy_config.get('rsi_entry_max', 65)
            conditions['rsi'] = (
                not pd.isna(row['RSI']) and
                rsi_min < row['RSI'] < rsi_max
            )
            
            conditions['macd'] = (
                not pd.isna(row['MACD_hist']) and
                row['MACD_hist'] > 0
            )
            
            volume_threshold = strategy_config.get('volume_surge_threshold', 1.1)
            conditions['volume'] = (
                not pd.isna(row['Volume_surge']) and
                row['Volume_surge'] > volume_threshold
            )
            
            conditions['obv'] = (
                not pd.isna(row['OBV_slope']) and
                row['OBV_slope'] > 0
            )
            
            support_proximity = strategy_config.get('support_proximity_percent', 0.05)
            conditions['support'] = (
                not pd.isna(row['Close']) and
                not pd.isna(row['Support_20']) and
                abs(row['Close'] - row['Support_20']) / row['Close'] < support_proximity
            )
            
        except Exception as e:
            logger.error(f"Error checking entry conditions: {e}", exc_info=True)
            conditions = {k: False for k in ['trend', 'rsi', 'macd', 'volume', 'obv', 'support']}
        
        return conditions
    
    def generate_signal(self, ticker: str, df: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal using v3 logic."""
        try:
            if len(df) < 5:
                logger.warning(f"Insufficient data for {ticker}")
                return None
            
            row_idx = len(df) - 1
            current = df.iloc[row_idx]
            previous = df.iloc[row_idx - 1]
            
            # Data validation
            if previous['Close'] > 0:
                price_change = abs(current['Close'] - previous['Close']) / previous['Close']
                if price_change > 0.20:
                    logger.warning(f"{ticker}: Data corrupt - {price_change:.1%} change")
                    return None
            
            # Get signal date
            signal_date = current['Date']
            if isinstance(signal_date, pd.Timestamp):
                signal_date_str = signal_date.strftime('%Y-%m-%d')
                signal_date_dt = signal_date.to_pydatetime()
            else:
                signal_date_str = signal_date
                signal_date_dt = datetime.strptime(signal_date, '%Y-%m-%d')
            
            # === HARD CONDITIONS ===
            passed, rejection_reason = self.check_hard_conditions(df, row_idx, ticker, signal_date_dt)
            if not passed:
                logger.debug(f"{ticker}: Hard condition failed - {rejection_reason}")
                return None
            
            # === QUALITY SCORE ===
            macd_quality = self.calculate_macd_quality(df, row_idx)
            quality_score = self.calculate_quality_score(df, row_idx, macd_quality)
            
            # Get regime-adjusted threshold
            regime_params = self.get_regime_adjusted_params()
            min_quality = regime_params['min_quality_score']
            
            if quality_score < min_quality:
                logger.debug(f"{ticker}: Quality {quality_score:.0f} < threshold {min_quality}")
                return None
            
            # Soft conditions for reporting
            conditions = self.check_entry_conditions(current, previous)
            legacy_score = sum(1 for c in conditions.values() if c) + 2
            legacy_score = max(0, min(10, legacy_score))
            
            # === TREND-BASED HOLD PERIOD ===
            expected_hold_min, expected_hold_max = self.calculate_trend_based_hold(current)
            
            # Dates
            strategy_config = self.config.get('strategy', {})
            max_holding_days = strategy_config.get('max_holding_days', 20)
            signal_expiration_days = strategy_config.get('signal_expiration_days', 3)
            
            try:
                expiration_date = (signal_date_dt + timedelta(days=signal_expiration_days)).strftime('%Y-%m-%d')
                max_hold_date = (signal_date_dt + timedelta(days=max_holding_days)).strftime('%Y-%m-%d')
            except:
                expiration_date = None
                max_hold_date = None
            
            entry_price = float(current['Close'])
            
            signal = {
                'ticker': ticker,
                'date': signal_date_str,
                'signal_type': 'BUY',
                'score': legacy_score,
                'quality_score': round(quality_score, 1),
                'macd_quality': round(macd_quality, 0),
                'market_regime': self._market_regime,
                'entry_price': entry_price,
                'atr': float(current['ATR']) if not pd.isna(current['ATR']) else 0,
                'conditions': conditions,
                'rsi': float(current['RSI']) if not pd.isna(current['RSI']) else 0,
                'macd_hist': float(current['MACD_hist']) if not pd.isna(current['MACD_hist']) else 0,
                'adx': float(current['ADX']) if not pd.isna(current['ADX']) else 0,
                'volume_surge': float(current['Volume_surge']) if not pd.isna(current['Volume_surge']) else 0,
                'support': float(current['Support_20']) if not pd.isna(current['Support_20']) else 0,
                'resistance': float(current['Resistance_20']) if not pd.isna(current['Resistance_20']) else 0,
                'signal_date': signal_date_str,
                'expiration_date': expiration_date,
                'expected_hold_min': expected_hold_min,
                'expected_hold_max': expected_hold_max,
                'max_hold_date': max_hold_date
            }
            
            logger.info(f"Signal: {ticker} | Q:{quality_score:.0f} | MACD:{macd_quality:.0f} | RSI:{signal['rsi']:.1f} | ADX:{signal['adx']:.0f} | Regime:{self._market_regime}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {ticker}: {e}", exc_info=True)
            return None
    
    def scan_stocks(self, tickers: List[str], data_dict: Dict[str, pd.DataFrame], spy_data: pd.DataFrame = None) -> List[Dict]:
        """
        Scan multiple stocks and generate signals.
        
        Args:
            tickers: List of tickers to scan
            data_dict: Dict mapping ticker to DataFrame
            spy_data: SPY data for market regime detection (optional)
        """
        signals = []
        
        logger.info(f"Scanning {len(tickers)} stocks for signals")
        
        # Update market regime if SPY data provided
        if spy_data is not None:
            self.update_market_regime(spy_data)
        elif 'SPY' in data_dict:
            self.update_market_regime(data_dict['SPY'])
        
        for ticker in tickers:
            if ticker not in data_dict:
                continue
            
            df = data_dict[ticker]
            df = self.calculate_all_indicators(df)
            signal = self.generate_signal(ticker, df)
            
            if signal:
                signals.append(signal)
        
        # Sort by quality_score
        signals.sort(key=lambda x: (x.get('quality_score', 0), x.get('score', 0)), reverse=True)
        
        # Apply regime-adjusted top_n
        regime_params = self.get_regime_adjusted_params()
        logger.info(f"Scan complete: Found {len(signals)} signals | Regime: {self._market_regime} | Min Quality: {regime_params['min_quality_score']}")
        
        return signals
