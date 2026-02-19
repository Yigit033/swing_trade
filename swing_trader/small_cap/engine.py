"""
Small Cap Momentum Engine - Main orchestrator class.
Completely independent from LargeCap Swing Engine.

This engine targets high-risk, high-volatility small cap stocks
for short-term momentum swings (2-14 days).
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from .filters import SmallCapFilters
from .signals import SmallCapSignals
from .scoring import SmallCapScoring
from .risk import SmallCapRisk
from .universe import SmallCapUniverse
from .sector_rs import SectorRS, get_sector_rs_bonus
from .catalysts import CatalystDetector, get_catalyst_bonus
from .narrative import generate_signal_narrative
from .technical_levels import calculate_technical_levels

logger = logging.getLogger(__name__)


class SmallCapEngine:
    """
    Small Cap Momentum Engine - Independent trading engine.
    
    NOT reusing any logic from LargeCap Swing Engine.
    Different philosophy: momentum ignition, not trend following.
    
    Universe:
    - Market Cap: 300M - 3B
    - Avg Volume: >= 1M shares
    - ATR%: >= 4%
    - Float: <= 150M shares
    
    Signals:
    - Volume surge >= 2.0x
    - Breakout (Close > prev High)
    - ATR% >= 6%
    
    Risk:
    - Position: 25-40% of LargeCap
    - Stop: 1-1.5 ATR
    - Max hold: 7 days
    - Target: 3R minimum
    """
    
    def __init__(self, config: Dict = None):
        """Initialize SmallCapEngine with all sub-components."""
        self.config = config or {}
        
        # Initialize independent components
        self.filters = SmallCapFilters(config)
        self.signals = SmallCapSignals(config)
        self.scoring = SmallCapScoring(config)
        self.risk = SmallCapRisk(config)
        self.universe_provider = SmallCapUniverse(config)
        
        logger.info("SmallCapEngine initialized (momentum breakout engine)")
    
    def _classify_swing_type(
        self, 
        five_day_return: float, 
        rsi: float, 
        volume_surge: float, 
        higher_lows: bool,
        close_position: float = 0.5,
        ma20_distance: float = 0.0,
        short_interest: float = 0.0,
        days_to_cover: float = 0.0,
        has_catalyst: bool = False,
        rsi_divergence: bool = False,
        macd_bullish: bool = False
    ) -> tuple:
        """
        Classify swing into Type S, C, B, or A (SENIOR TRADER 4-TYPE SYSTEM).
        
        PRIORITY ORDER: S → C → B → A
        
        TYPE S - Short Squeeze (1-4 days) - AGGRESSIVE:
        - Short Interest ≥ 20%
        - Days to Cover ≥ 5
        - Volume surge ≥ 4x
        - 5-day return: +15% to +60%
        - RSI: 60-80
        - VERY HIGH RISK, HIGH REWARD
        
        TYPE C - Early Stage (2-4 days) - BEST R/R:
        - 5-day return: -5% to +15% (pullback entry allowed!)
        - RSI: 40-60
        - Volume: 1.8x to 4x
        - RSI Divergence: BONUS
        - MA20 distance: -3% to +8%
        
        TYPE B - Momentum (2-6 days) - TIGHTENED:
        - 5-day return: +30% to +70%
        - RSI: 68-85
        - Volume: ≥ 3.5x
        
        TYPE A - Continuation (4-10 days) - STANDARD:
        - 5-day return: +10% to +35%
        - RSI: 50-68
        - Higher lows: Required
        
        Returns:
            (swing_type, (min_days, max_days), reason)
        """
        
        # ============================================================
        # EXTREME CHASING PROTECTION
        # ============================================================
        if five_day_return > 70:
            return ('B', (1, 2), f"⚠️ PARABOLIC: 5d={five_day_return:+.0f}% - EXIT FAST!")
        
        if five_day_return > 60 and rsi > 85:
            return ('B', (1, 2), f"⚠️ EXTREME: 5d={five_day_return:+.0f}%, RSI={rsi:.0f} - VERY SHORT!")
        
        # ============================================================
        # TYPE S CHECK - Short Squeeze (PRIORITY 1)
        # ============================================================
        if short_interest >= 20 and days_to_cover >= 5 and volume_surge >= 4.0:
            if 15 <= five_day_return <= 60 and 60 <= rsi <= 80:
                return ('S', (1, 4), f"🔥 SQUEEZE: SI={short_interest:.0f}%, DTC={days_to_cover:.0f}, Vol={volume_surge:.1f}x")
        
        # Also check for potential squeeze setup
        if short_interest >= 15 and days_to_cover >= 3 and volume_surge >= 3.0:
            if 10 <= five_day_return <= 40 and 55 <= rsi <= 75:
                return ('S', (2, 4), f"💥 Squeeze Setup: SI={short_interest:.0f}%, 5d={five_day_return:+.0f}%")
        
        # ============================================================
        # TYPE C CHECK - Early Stage Breakout (PRIORITY 2 - Best R/R)
        # ============================================================
        type_c_score = 0
        
        # 5-day return: -5% to +15% (pullback entry allowed!)
        if -5 <= five_day_return <= 15:
            type_c_score += 4
            if 0 <= five_day_return <= 10:
                type_c_score += 1  # Sweet spot bonus
        
        # RSI: 40-60 (room to run)
        if 40 <= rsi <= 60:
            type_c_score += 4
            if rsi <= 50:
                type_c_score += 1  # Low RSI bonus
        elif 60 < rsi <= 65:
            type_c_score += 2  # Still acceptable
        
        # Volume: 1.8x to 4.0x
        if 1.8 <= volume_surge <= 4.0:
            type_c_score += 2
            if volume_surge >= 2.5:
                type_c_score += 1
        
        # MA20 distance: -3% to +8% (pullback or just above)
        if -3 <= ma20_distance <= 8:
            type_c_score += 2
        
        # Close position: ≥ 0.55 upper half
        if close_position >= 0.55:
            type_c_score += 1
        
        # RSI Divergence: GAME CHANGER (+3)
        if rsi_divergence:
            type_c_score += 3
        
        # MACD Bullish: Bonus
        if macd_bullish:
            type_c_score += 1
        
        # Higher lows
        if higher_lows:
            type_c_score += 1
        
        # Type C threshold: 8+ points
        if type_c_score >= 8:
            if rsi_divergence:
                emoji = "🌟"
                reason = f"RSI Divergence + Early: 5d={five_day_return:+.0f}%, RSI={rsi:.0f}"
            elif five_day_return < 0:
                emoji = "⭐"
                reason = f"Pullback Entry: 5d={five_day_return:+.0f}%, RSI={rsi:.0f}"
            else:
                emoji = "⭐"
                reason = f"Early Stage: 5d={five_day_return:+.0f}%, RSI={rsi:.0f}"
            return ('C', (3, 8), f"{emoji} {reason}")
        
        # ============================================================
        # TYPE B CHECK - Momentum Swing (PRIORITY 3)
        # ============================================================
        type_b_score = 0
        
        # 5-day return: +30% to +70% (Senior Trader specs)
        if 30 <= five_day_return <= 70:
            type_b_score += 3
        elif 20 <= five_day_return < 30:
            type_b_score += 2
        elif five_day_return > 70:
            type_b_score += 1  # Extended but still momentum
        
        # RSI: 68-85 (elevated)
        if 68 <= rsi <= 85:
            type_b_score += 3
        elif 60 <= rsi < 68:
            type_b_score += 2
        elif rsi > 85:
            type_b_score += 1  # Extreme
        
        # Volume: ≥ 3.5x
        if volume_surge >= 3.5:
            type_b_score += 3
        elif volume_surge >= 2.5:
            type_b_score += 2
        
        # Close position: ≥ 0.75
        if close_position >= 0.75:
            type_b_score += 2
        
        # Catalyst bonus
        if has_catalyst:
            type_b_score += 1
        
        # Type B threshold: 6+ points
        if type_b_score >= 6:
            # ── STRICT FILTER GATE (v2.2) ──
            # Tip B is HIGH RISK (chasing extended stocks).
            # Only enter if ALL conditions met:
            #   1. Volume >= 5x (extreme conviction)
            #   2. RSI <= 75 (not too extreme)
            #   3. Has catalyst (news/squeeze/insider = real reason)
            # If filter fails → downgrade to Type A (safer hold period)
            if volume_surge < 5.0 or rsi > 75 or not has_catalyst:
                # Doesn't meet strict criteria → reclass as Type A
                pass  # Fall through to Type A
            else:
                # Meets strict criteria → genuine Tip B
                if rsi > 73:
                    hold_days = (2, 4)  # Overbought but catalyst-driven
                elif rsi > 68:
                    hold_days = (3, 5)  # Elevated
                else:
                    hold_days = (4, 6)  # Room to run
                
                return ('B', hold_days, f"🚀 Momentum: 5d={five_day_return:+.0f}%, RSI={rsi:.0f}, Cat:✓")
        
        # ============================================================
        # TYPE A - Continuation Swing (FALLBACK)
        # ============================================================
        type_a_reasons = []
        
        # 5-day return: +10% to +35%
        if 10 <= five_day_return <= 35:
            type_a_reasons.append(f"5d={five_day_return:+.0f}%")
        elif five_day_return < 10:
            type_a_reasons.append(f"5d={five_day_return:+.0f}% (building)")
        else:
            type_a_reasons.append(f"5d={five_day_return:+.0f}%")
        
        # RSI: 50-68
        if 50 <= rsi <= 68:
            type_a_reasons.append(f"RSI={rsi:.0f} (healthy)")
        else:
            type_a_reasons.append(f"RSI={rsi:.0f}")
        
        # Higher lows
        if higher_lows:
            type_a_reasons.append("HL ✓")
        
        # MACD
        if macd_bullish:
            type_a_reasons.append("MACD ✓")
        
        # Sub-classification for Type A (5-14 days, extended for proper swing)
        if five_day_return <= 15 and rsi <= 55:
            hold_days = (5, 10)  # Early continuation
        elif five_day_return <= 25 and rsi <= 62:
            hold_days = (7, 12)  # Standard
        else:
            hold_days = (8, 14)  # Extended trend
        
        return ('A', hold_days, "🐢 Continuation: " + ", ".join(type_a_reasons[:2]))
    
    def get_stock_info(self, ticker: str) -> Dict:
        """Get stock info from yfinance."""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'marketCap': info.get('marketCap', 0),
                'floatShares': info.get('floatShares', 0),
                'shortName': info.get('shortName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
        except Exception as e:
            logger.debug(f"Could not get info for {ticker}: {e}")
            return {'ticker': ticker, 'marketCap': 0, 'floatShares': 0}
    
    def scan_stock(
        self, 
        ticker: str, 
        df: pd.DataFrame,
        stock_info: Dict = None
    ) -> Optional[Dict]:
        """
        Scan a single stock for small-cap momentum signal.
        
        Returns signal dict if triggered, None otherwise.
        """
        if df is None or len(df) < 20:
            logger.debug(f"{ticker}: Insufficient data")
            return None
        
        try:
            # Get stock info if not provided
            if stock_info is None:
                stock_info = self.get_stock_info(ticker)
            
            # Get signal date - handle both index-based and column-based Date
            try:
                if 'Date' in df.columns:
                    signal_date = df['Date'].iloc[-1]
                else:
                    signal_date = df.index[-1]
                
                if isinstance(signal_date, pd.Timestamp):
                    signal_date_str = signal_date.strftime('%Y-%m-%d')
                    signal_date_dt = signal_date.to_pydatetime()
                    if signal_date_dt.tzinfo is not None:
                        signal_date_dt = signal_date_dt.replace(tzinfo=None)
                else:
                    signal_date_str = str(signal_date)[:10]
                    signal_date_dt = datetime.strptime(signal_date_str, '%Y-%m-%d')
            except Exception:
                signal_date_str = datetime.now().strftime('%Y-%m-%d')
                signal_date_dt = datetime.now()
            
            # Step 1: Apply universe filters
            filter_passed, filter_results = self.filters.apply_all_filters(
                ticker, df, stock_info, signal_date_dt
            )
            
            if not filter_passed:
                logger.debug(f"{ticker}: Failed filter - {filter_results.get('filters', {})}")
                return None
            
            # Step 2: Check signal triggers
            triggered, trigger_details = self.signals.check_all_triggers(df)
            
            if not triggered:
                logger.debug(f"{ticker}: No trigger - {trigger_details.get('triggers', {})}")
                return None
            
            # Step 3: Get boosters (includes swing confirmation)
            boosters = self.signals.check_boosters(df)
            
            # Step 3.5: SWING CONFIRMATION GATE (NEW)
            # Must pass 5-day momentum > 0 AND Close > MA20
            swing_ready = boosters.get('swing_ready', False)
            swing_details = boosters.get('swing_details', {})
            
            if not swing_ready:
                # Log why it failed
                five_day = swing_details.get('five_day_momentum', {})
                ma20 = swing_details.get('above_ma20', {})
                logger.debug(
                    f"{ticker}: Failed swing confirmation - "
                    f"5d_mom={five_day.get('passed')}, ma20={ma20.get('passed')}"
                )
                return None
            
            # Step 4: Calculate quality score (includes penalties)
            volume_surge = trigger_details.get('volume_surge', 2.0)
            atr_percent = trigger_details.get('atr_percent', filter_results.get('atr_percent', 0.06))
            float_shares = filter_results.get('float_shares', 0)
            
            # Get preliminary swing metrics for Sector RS
            five_day_return_prelim = swing_details.get('five_day_momentum', {}).get('return', 0)
            sector = stock_info.get('sector', 'Unknown')
            
            # ============================================================
            # SECTOR RS & CATALYST DATA (Senior Trader v2.1)
            # ============================================================
            # Sector Relative Strength
            sector_rs_data = SectorRS.calculate_sector_rs(ticker, sector, five_day_return_prelim)
            boosters['sector_rs_bonus'] = sector_rs_data['bonus']
            boosters['sector_rs_score'] = sector_rs_data['rs_score']
            boosters['is_sector_leader'] = sector_rs_data['is_leader']
            
            # Catalyst Data (Short Interest, Insider, News)
            catalyst_data = CatalystDetector.get_all_catalysts(ticker)
            boosters['short_interest_bonus'] = catalyst_data['short_interest']['bonus']
            boosters['short_percent'] = catalyst_data['short_interest']['short_percent']
            boosters['days_to_cover'] = catalyst_data['short_interest']['days_to_cover']
            boosters['is_squeeze_candidate'] = catalyst_data['short_interest']['is_squeeze_candidate']
            boosters['insider_bonus'] = catalyst_data['insider']['bonus']
            boosters['has_insider_buying'] = catalyst_data['insider']['has_insider_buying']
            boosters['news_bonus'] = catalyst_data['news']['bonus']
            boosters['has_recent_news'] = catalyst_data['news']['has_recent_news']
            boosters['total_catalyst_bonus'] = catalyst_data['total_catalyst_bonus']
            
            # RSI Divergence (already in signals but ensure it's in boosters)
            rsi_div = self.signals.detect_rsi_divergence(df, lookback=14)
            boosters['rsi_divergence'] = rsi_div['divergence_found']
            boosters['rsi_divergence_confidence'] = rsi_div.get('confidence', 0)
            
            # MACD check
            macd_data = self.signals.calculate_macd(df)
            boosters['macd_bullish'] = macd_data['bullish_cross'] or (macd_data['above_zero'] and macd_data['expanding'])
            
            quality_score = self.scoring.calculate_quality_score(
                df, volume_surge, atr_percent, float_shares, boosters
            )
            
            # Create signal with enhanced info
            entry_price = float(df['Close'].iloc[-1])
            
            # Get swing metrics for display
            five_day_return = swing_details.get('five_day_momentum', {}).get('return', 0)
            ma20_distance = swing_details.get('above_ma20', {}).get('distance', 0)
            rsi = boosters.get('rsi', 50)
            overext = swing_details.get('overextension', {})
            higher_lows = boosters.get('higher_lows', False)
            
            # Calculate close position (where in day's range did it close)
            today_high = float(df['High'].iloc[-1])
            today_low = float(df['Low'].iloc[-1])
            today_close = float(df['Close'].iloc[-1])
            day_range = today_high - today_low
            close_position = (today_close - today_low) / day_range if day_range > 0 else 0.5
            
            # ============================================================
            # SWING TYPE CLASSIFICATION (OPTIMIZED)
            # Type C: Early Stage (2-4 days) - check first
            # Type B: Momentum (2-5 days) - RSI-based duration
            # Type A: Continuation (4-8 days) - default
            # ============================================================
            # Pass all available data to classify (catalyst/divergence were missing before)
            has_any_catalyst = (
                boosters.get('has_recent_news', False) or
                boosters.get('is_squeeze_candidate', False) or
                boosters.get('has_insider_buying', False) or
                boosters.get('total_catalyst_bonus', 0) > 0
            )
            swing_type, hold_days, type_reason = self._classify_swing_type(
                five_day_return, rsi, volume_surge, higher_lows,
                close_position=close_position,
                ma20_distance=ma20_distance,
                short_interest=boosters.get('short_percent', 0),
                days_to_cover=boosters.get('days_to_cover', 0),
                has_catalyst=has_any_catalyst,
                rsi_divergence=boosters.get('rsi_divergence', False),
                macd_bullish=boosters.get('macd_bullish', False)
            )
            
            # Swing type labels
            type_labels = {
                'S': 'Short Squeeze',
                'A': 'Continuation',
                'B': 'Momentum', 
                'C': 'Early Stage'
            }
            
            signal = {
                'ticker': ticker,
                'date': signal_date_str,
                'signal_type': 'SMALL_CAP_SWING',
                'quality_score': round(quality_score, 1),
                'entry_price': round(entry_price, 2),
                
                # SWING TYPE (OPTIMIZED)
                'swing_type': swing_type,           # 'A', 'B', or 'C'
                'swing_type_label': type_labels.get(swing_type, 'Unknown'),
                'hold_days_min': hold_days[0],
                'hold_days_max': hold_days[1],
                'type_reason': type_reason,
                'close_position': round(close_position, 2),
                
                # Momentum Metrics
                'volume_surge': round(volume_surge, 2),
                'atr_percent': round(atr_percent * 100, 1),
                'float_millions': round(float_shares / 1e6, 1) if float_shares else 0,
                'market_cap_millions': round(filter_results.get('market_cap', 0) / 1e6, 0),
                
                # SWING METRICS
                'five_day_return': round(five_day_return, 1),
                'ma20_distance': round(ma20_distance, 1),
                'rsi': round(rsi, 0),
                'swing_ready': swing_ready,
                'higher_lows': higher_lows,
                
                # Boosters
                'high_rvol': boosters.get('high_rvol', False),
                'gap_continuation': boosters.get('gap_continuation', False),
                'higher_highs': boosters.get('higher_highs', False),
                
                # ============================================================
                # NEW SENIOR TRADER v2.1 FIELDS
                # ============================================================
                # Sector Relative Strength
                'sector_rs_score': round(boosters.get('sector_rs_score', 0), 1),
                'sector_rs_bonus': boosters.get('sector_rs_bonus', 0),
                'is_sector_leader': boosters.get('is_sector_leader', False),
                
                # Short Interest & Squeeze
                'short_percent': round(boosters.get('short_percent', 0), 1),
                'days_to_cover': round(boosters.get('days_to_cover', 0), 1),
                'is_squeeze_candidate': boosters.get('is_squeeze_candidate', False),
                'short_interest_bonus': boosters.get('short_interest_bonus', 0),
                
                # Insider & News
                'has_insider_buying': boosters.get('has_insider_buying', False),
                'insider_bonus': boosters.get('insider_bonus', 0),
                'has_recent_news': boosters.get('has_recent_news', False),
                'news_bonus': boosters.get('news_bonus', 0),
                'total_catalyst_bonus': boosters.get('total_catalyst_bonus', 0),
                
                # RSI Divergence & MACD
                'rsi_divergence': boosters.get('rsi_divergence', False),
                'rsi_divergence_confidence': boosters.get('rsi_divergence_confidence', 0),
                'macd_bullish': boosters.get('macd_bullish', False),
                
                # Filter/trigger details
                'filter_results': filter_results,
                'trigger_details': trigger_details,
                'swing_details': swing_details,
                
                # Stock info
                'company_name': stock_info.get('shortName', ticker),
                'sector': stock_info.get('sector', 'Unknown')
            }
            
            # ============================================================
            # RISK MANAGEMENT: Calculate stop_loss, target, position size
            # Must happen BEFORE narrative generation so it gets real values
            # ============================================================
            try:
                risk_signal = self.risk.add_risk_management(
                    signal.copy(), df, portfolio_value=10000
                )
                signal['stop_loss'] = risk_signal.get('stop_loss', 0)
                signal['target_1'] = risk_signal.get('target_1', 0)
                signal['target_2'] = risk_signal.get('target_2', 0)
                signal['target_1_pct'] = risk_signal.get('target_1_pct', 0)
                signal['target_2_pct'] = risk_signal.get('target_2_pct', 0)
                signal['stop_loss_pct'] = risk_signal.get('stop_loss_pct', 0)
                signal['risk_reward'] = risk_signal.get('risk_reward', 0)
                signal['risk_reward_t2'] = risk_signal.get('risk_reward_t2', 0)
                signal['position_size'] = risk_signal.get('position_size', 0)
                signal['risk_amount'] = risk_signal.get('risk_amount', 0)
                signal['expected_hold_min'] = risk_signal.get('expected_hold_min', hold_days[0])
                signal['expected_hold_max'] = risk_signal.get('expected_hold_max', hold_days[1])
                signal['max_hold_date'] = risk_signal.get('max_hold_date', '')
                signal['expiration_date'] = risk_signal.get('expiration_date', '')
                signal['volatility_warning'] = risk_signal.get('volatility_warning', False)
            except Exception as e:
                logger.warning(f"Could not add risk management for {ticker}: {e}")
                # Fallback: calculate stop/target manually with type-specific targets
                atr_val = self.risk.calculate_atr(df)
                signal['stop_loss'] = round(entry_price - (1.5 * atr_val), 2) if atr_val else round(entry_price * 0.93, 2)
                t1_pct, t2_pct = self.risk.TYPE_TARGETS.get(swing_type, (0.25, 0.40))
                signal['target_1'] = round(entry_price * (1 + t1_pct), 2)
                signal['target_2'] = round(entry_price * (1 + t2_pct), 2)
                signal['target_1_pct'] = round(t1_pct * 100, 1)
                signal['target_2_pct'] = round(t2_pct * 100, 1)
                signal['position_size'] = 0
                signal['expected_hold_min'] = hold_days[0]
                signal['expected_hold_max'] = hold_days[1]
            
            # Enhanced logging with type
            type_emoji = "🐢" if swing_type == 'A' else "🚀"
            safe_status = "✓" if overext.get('safe') else "⚠"
            logger.info(
                f"SMALL CAP SWING {type_emoji}: {ticker} | Type {swing_type} ({hold_days[0]}-{hold_days[1]}d) | "
                f"Q:{quality_score:.0f} | 5d:{five_day_return:+.0f}% | RSI:{rsi:.0f} {safe_status}"
            )
            
            # Generate narrative analysis (Cuma Çevik style)
            # Calculate technical levels for professional narrative
            try:
                tech_levels = calculate_technical_levels(
                    df, signal['entry_price'], signal.get('volume_surge', 1.0)
                )
                signal['technical_levels'] = tech_levels
            except Exception as e:
                logger.debug(f"Could not calculate technical levels for {ticker}: {e}")
                signal['technical_levels'] = None
            
            try:
                narrative = generate_signal_narrative(signal, language='tr')
                signal['narrative'] = narrative
                signal['narrative_text'] = narrative.get('full_text', '')
                signal['narrative_headline'] = narrative.get('headline', f"{ticker} - {swing_type}")
            except Exception as e:
                logger.warning(f"Could not generate narrative for {ticker}: {e}")
                signal['narrative'] = None
                signal['narrative_text'] = ''
                signal['narrative_headline'] = f"{ticker} - Type {swing_type}"
            
            return signal
            
        except Exception as e:
            logger.error(f"Error scanning {ticker}: {e}", exc_info=True)
            return None
    
    def scan_universe(
        self, 
        tickers: List[str],
        data_dict: Dict[str, pd.DataFrame],
        portfolio_value: float = 10000
    ) -> List[Dict]:
        """
        Scan multiple stocks for small-cap momentum signals.
        
        Args:
            tickers: List of ticker symbols
            data_dict: Dict mapping ticker to DataFrame
            portfolio_value: Portfolio value for position sizing
        
        Returns:
            List of signals sorted by quality_score
        """
        signals = []
        scanned = 0
        filter_failed = 0
        trigger_failed = 0
        
        logger.info(f"SmallCapEngine: Scanning {len(tickers)} stocks")
        
        for ticker in tickers:
            if ticker not in data_dict:
                continue
            
            df = data_dict[ticker]
            scanned += 1
            
            signal = self.scan_stock(ticker, df)
            
            if signal:
                # Add risk management
                signal = self.risk.add_risk_management(signal, df, portfolio_value)
                signals.append(signal)
            else:
                # Count failures for stats
                filter_failed += 1  # Simplified - could track separately
        
        # Sort by quality score (highest first)
        signals.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        logger.info(
            f"SmallCapEngine: Scanned {scanned} | "
            f"Signals: {len(signals)} | "
            f"Filtered: {filter_failed}"
        )
        
        return signals
    
    def get_small_cap_universe(self, use_finviz: bool = True, max_tickers: int = 200) -> List[str]:
        """
        Get list of potential small-cap stocks to scan.
        
        Uses SmallCapUniverse to get dynamic ticker list from:
        1. Finviz screener via finvizfinance (primary)
        2. Curated static list (fallback)
        
        Args:
            use_finviz: If True, try Finviz first
            max_tickers: Maximum number of tickers to return
        
        Returns:
            List of ticker symbols
        """
        return self.universe_provider.get_universe(use_finviz=use_finviz, max_tickers=max_tickers)

