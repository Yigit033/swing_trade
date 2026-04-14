"""
Small Cap Momentum Engine - Main orchestrator class.
Completely independent from LargeCap Swing Engine.

This engine targets high-risk, high-volatility small cap stocks
for short-term momentum swings (2-14 days).
"""

import logging
from typing import Dict, List, Optional, MutableMapping
from datetime import datetime
import pandas as pd

from .filters import SmallCapFilters
from .signals import SmallCapSignals
from .scoring import SmallCapScoring
from .risk import SmallCapRisk
from .universe import SmallCapUniverse
from .sector_rs import SectorRS
from .catalysts import CatalystDetector
from .narrative import generate_signal_narrative
from .technical_levels import calculate_technical_levels
from .regime_logic import rs_bonus_vs_spy
from .settings_config import load_settings

logger = logging.getLogger(__name__)


def _bump_scan_reject(reject_counts: Optional[MutableMapping[str, int]], key: str) -> None:
    if reject_counts is None:
        return
    reject_counts[key] = reject_counts.get(key, 0) + 1


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
        self.settings = load_settings()

        # Initialize independent components (risk/filters/signals read self.settings)
        self.filters = SmallCapFilters(config, self.settings)
        self.signals = SmallCapSignals(config, self.settings)
        self.scoring = SmallCapScoring(config, self.settings)
        self.risk = SmallCapRisk(config, self.settings)
        self.universe_provider = SmallCapUniverse(config, self.settings)

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
        
        sp = self.settings.swing.parabolic
        ts = self.settings.swing.type_s
        tc = self.settings.swing.type_c
        tb = self.settings.swing.type_b
        ta = self.settings.swing.type_a

        # ============================================================
        # EXTREME CHASING PROTECTION
        # ============================================================
        if five_day_return > sp.five_day_gt:
            return (
                "B",
                sp.hold_short,
                f"⚠️ PARABOLIC: 5d={five_day_return:+.0f}% - EXIT FAST!",
            )

        if five_day_return > sp.five_day_extreme_gt and rsi > sp.rsi_extreme_gt:
            return (
                "B",
                sp.hold_short,
                f"⚠️ EXTREME: 5d={five_day_return:+.0f}%, RSI={rsi:.0f} - VERY SHORT!",
            )

        # ============================================================
        # TYPE S CHECK - Short Squeeze (PRIORITY 1)
        # ============================================================
        if (
            short_interest >= ts.primary_si_min
            and days_to_cover >= ts.primary_dtc_min
            and volume_surge >= ts.primary_vol_min
        ):
            if (
                ts.primary_5d_min <= five_day_return <= ts.primary_5d_max
                and ts.primary_rsi_min <= rsi <= ts.primary_rsi_max
            ):
                return (
                    "S",
                    (ts.primary_hold_min, ts.primary_hold_max),
                    f"🔥 SQUEEZE: SI={short_interest:.0f}%, DTC={days_to_cover:.0f}, Vol={volume_surge:.1f}x",
                )

        if (
            short_interest >= ts.secondary_si_min
            and days_to_cover >= ts.secondary_dtc_min
            and volume_surge >= ts.secondary_vol_min
        ):
            if (
                ts.secondary_5d_min <= five_day_return <= ts.secondary_5d_max
                and ts.secondary_rsi_min <= rsi <= ts.secondary_rsi_max
            ):
                return (
                    "S",
                    (ts.secondary_hold_min, ts.secondary_hold_max),
                    f"💥 Squeeze Setup: SI={short_interest:.0f}%, 5d={five_day_return:+.0f}%",
                )

        # ============================================================
        # TYPE C CHECK - Early Stage Breakout (PRIORITY 2 - Best R/R)
        # ============================================================
        type_c_score = 0

        if tc.return_min <= five_day_return <= tc.return_max:
            type_c_score += tc.return_band_pts
            if tc.sweet_return_min <= five_day_return <= tc.sweet_return_max:
                type_c_score += tc.sweet_bonus_pts

        if tc.rsi_min <= rsi <= tc.rsi_max:
            type_c_score += tc.rsi_band_pts
            if rsi <= tc.rsi_low_max:
                type_c_score += tc.rsi_low_bonus_pts
        elif tc.rsi_max < rsi <= tc.rsi_mid_max:
            type_c_score += tc.rsi_mid_pts

        if tc.vol_min <= volume_surge <= tc.vol_max:
            type_c_score += tc.vol_band_pts
            if volume_surge >= tc.vol_high_min:
                type_c_score += tc.vol_high_bonus_pts

        if tc.ma_dist_min <= ma20_distance <= tc.ma_dist_max:
            type_c_score += tc.ma_band_pts

        if close_position >= tc.close_position_min:
            type_c_score += tc.close_position_pts

        if rsi_divergence:
            type_c_score += tc.rsi_div_pts

        if macd_bullish:
            type_c_score += tc.macd_pts

        if higher_lows:
            type_c_score += tc.higher_lows_pts

        if type_c_score >= tc.min_score:
            if rsi_divergence:
                emoji = "🌟"
                reason = f"RSI Divergence + Early: 5d={five_day_return:+.0f}%, RSI={rsi:.0f}"
            elif five_day_return < 0:
                emoji = "⭐"
                reason = f"Pullback Entry: 5d={five_day_return:+.0f}%, RSI={rsi:.0f}"
            else:
                emoji = "⭐"
                reason = f"Early Stage: 5d={five_day_return:+.0f}%, RSI={rsi:.0f}"
            return ("C", (tc.hold_min, tc.hold_max), f"{emoji} {reason}")

        # ============================================================
        # TYPE B CHECK - Momentum Swing (PRIORITY 3)
        # ============================================================
        type_b_score = 0

        if 30 <= five_day_return <= 70:
            type_b_score += tb.r_30_70_pts
        elif 20 <= five_day_return < 30:
            type_b_score += tb.r_20_30_pts
        elif five_day_return > 70:
            type_b_score += tb.r_gt_70_pts

        if 68 <= rsi <= 85:
            type_b_score += tb.rsi_68_85_pts
        elif 60 <= rsi < 68:
            type_b_score += tb.rsi_60_68_pts
        elif rsi > 85:
            type_b_score += tb.rsi_gt_85_pts

        if volume_surge >= tb.gate_vol_min:
            type_b_score += tb.vol_35_pts
        elif volume_surge >= tb.vol_surge_secondary_min:
            type_b_score += tb.vol_25_pts

        if close_position >= tb.close_pos_min:
            type_b_score += tb.close_pos_pts

        if has_catalyst:
            type_b_score += tb.catalyst_pts

        if type_b_score >= tb.min_score:
            has_safety = has_catalyst or rsi <= tb.gate_rsi_safe_max
            if volume_surge < tb.gate_vol_min or not has_safety:
                pass
            else:
                if rsi > tb.rsi_overbought_hold_gt:
                    hold_days = tb.hold_overbought
                elif rsi > tb.rsi_elevated_gt:
                    hold_days = tb.hold_elevated
                else:
                    hold_days = tb.hold_default

                cat_str = "Cat:✓" if has_catalyst else "Vol-driven"
                return (
                    "B",
                    hold_days,
                    f"🚀 Momentum: 5d={five_day_return:+.0f}%, RSI={rsi:.0f}, {cat_str}",
                )

        # ============================================================
        # TYPE A - Continuation Swing (FALLBACK)
        # ============================================================
        type_a_reasons = []

        if 10 <= five_day_return <= 35:
            type_a_reasons.append(f"5d={five_day_return:+.0f}%")
        elif five_day_return < 10:
            type_a_reasons.append(f"5d={five_day_return:+.0f}% (building)")
        else:
            type_a_reasons.append(f"5d={five_day_return:+.0f}%")

        if 50 <= rsi <= 68:
            type_a_reasons.append(f"RSI={rsi:.0f} (healthy)")
        else:
            type_a_reasons.append(f"RSI={rsi:.0f}")

        if higher_lows:
            type_a_reasons.append("HL ✓")

        if macd_bullish:
            type_a_reasons.append("MACD ✓")

        if five_day_return <= ta.five_d_max_early and rsi <= ta.rsi_max_early:
            hold_days = ta.hold_early
        elif five_day_return <= ta.five_d_max_std and rsi <= ta.rsi_max_std:
            hold_days = ta.hold_std
        else:
            hold_days = ta.hold_extended

        return ("A", hold_days, "🐢 Continuation: " + ", ".join(type_a_reasons[:2]))
    
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
        stock_info: Dict = None,
        *,
        backtest_mode: bool = False,
        portfolio_value: float = 10000,
        spy_df_window: Optional[pd.DataFrame] = None,
        reject_counts: Optional[MutableMapping[str, int]] = None,
    ) -> Optional[Dict]:
        """
        Scan a single stock for small-cap momentum signal.

        backtest_mode: skip live yfinance fundamentals, earnings API, catalysts
        (short/insider/news bonuses zero; earnings filter skipped); optional
        spy_df_window for point-in-time RS vs SPY; skip narrative/LLM.
        """
        if df is None or len(df) < 20:
            logger.debug(f"{ticker}: Insufficient data")
            _bump_scan_reject(reject_counts, "insufficient_data")
            return None
        
        try:
            if stock_info is None:
                if backtest_mode:
                    stock_info = {
                        "ticker": ticker,
                        "marketCap": int(self.filters.MIN_MARKET_CAP * 1.2),
                        "floatShares": 45_000_000,
                        "shortName": ticker,
                        "sector": "Unknown",
                    }
                else:
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
                ticker, df, stock_info, signal_date_dt, backtest_mode=backtest_mode
            )
            
            if not filter_passed:
                logger.debug(f"{ticker}: Failed filter - {filter_results.get('filters', {})}")
                _bump_scan_reject(reject_counts, "filter_failed")
                return None
            
            # Step 2: Check signal triggers
            triggered, trigger_details = self.signals.check_all_triggers(df)
            
            if not triggered:
                logger.debug(f"{ticker}: No trigger - {trigger_details.get('triggers', {})}")
                _bump_scan_reject(reject_counts, "no_trigger")
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
                _bump_scan_reject(reject_counts, "swing_not_ready")
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
            if backtest_mode:
                if (
                    spy_df_window is not None
                    and len(spy_df_window) >= 6
                    and "Close" in spy_df_window.columns
                    and len(df) >= 6
                ):
                    sector_rs_data = rs_bonus_vs_spy(df["Close"], spy_df_window["Close"])
                else:
                    sector_rs_data = {
                        "bonus": 0,
                        "rs_score": 0.0,
                        "is_leader": False,
                        "ticker_5d": 0.0,
                        "sector_5d": 0.0,
                        "sector_etf": "SPY",
                    }
                boosters["sector_rs_bonus"] = sector_rs_data.get("bonus", 0)
                boosters["sector_rs_score"] = sector_rs_data.get("rs_score", 0.0)
                boosters["is_sector_leader"] = sector_rs_data.get("is_leader", False)
                boosters["short_interest_bonus"] = 0
                boosters["short_percent"] = 0.0
                boosters["days_to_cover"] = 0.0
                boosters["is_squeeze_candidate"] = False
                boosters["insider_bonus"] = 0
                boosters["has_insider_buying"] = False
                boosters["news_bonus"] = 0
                boosters["has_recent_news"] = False
                boosters["total_catalyst_bonus"] = 0
            else:
                sector_rs_data = SectorRS.calculate_sector_rs(ticker, sector, five_day_return_prelim)
                boosters["sector_rs_bonus"] = sector_rs_data["bonus"]
                boosters["sector_rs_score"] = sector_rs_data["rs_score"]
                boosters["is_sector_leader"] = sector_rs_data["is_leader"]
                catalyst_data = CatalystDetector.get_all_catalysts(ticker)
                boosters["short_interest_bonus"] = catalyst_data["short_interest"]["bonus"]
                boosters["short_percent"] = catalyst_data["short_interest"]["short_percent"]
                boosters["days_to_cover"] = catalyst_data["short_interest"]["days_to_cover"]
                boosters["is_squeeze_candidate"] = catalyst_data["short_interest"]["is_squeeze_candidate"]
                boosters["insider_bonus"] = catalyst_data["insider"]["bonus"]
                boosters["has_insider_buying"] = catalyst_data["insider"]["has_insider_buying"]
                boosters["news_bonus"] = catalyst_data["news"]["bonus"]
                boosters["has_recent_news"] = catalyst_data["news"]["has_recent_news"]
                boosters["total_catalyst_bonus"] = catalyst_data["total_catalyst_bonus"]
            
            # RSI Divergence (already in signals but ensure it's in boosters)
            rsi_div = self.signals.detect_rsi_divergence(df, lookback=14)
            boosters['rsi_divergence'] = rsi_div['divergence_found']
            boosters['rsi_divergence_confidence'] = rsi_div.get('confidence', 0)
            
            # MACD check
            macd_data = self.signals.calculate_macd(df)
            boosters['macd_bullish'] = macd_data['bullish_cross'] or (macd_data['above_zero'] and macd_data['expanding'])
            
            # Get swing metrics for display
            entry_price = float(df['Close'].iloc[-1])
            five_day_return = swing_details.get('five_day_momentum', {}).get('return', 0)
            ma20_distance = swing_details.get('above_ma20', {}).get('distance', 0)
            rsi = boosters.get('rsi', 50)
            overext = swing_details.get('overextension', {})
            higher_lows = boosters.get('higher_lows', False)
            
            today_high = float(df['High'].iloc[-1])
            today_low = float(df['Low'].iloc[-1])
            today_close = float(df['Close'].iloc[-1])
            day_range = today_high - today_low
            close_position = (today_close - today_low) / day_range if day_range > 0 else 0.5
            
            # ── CLASSIFY SWING TYPE *BEFORE* SCORING ──
            # This way scoring penalties use the correct type-specific RSI bands.
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
            
            # V4 hard RSI gate: reject overbought signals before they become trades
            max_rsi = self.settings.max_entry_rsi
            if rsi > max_rsi and swing_type != 'S':
                logger.debug(f"{ticker}: RSI {rsi:.0f} > {max_rsi} — rejected (overbought, not squeeze)")
                _bump_scan_reject(reject_counts, "rsi_gate")
                return None

            # V4: Hard overextension gate — reject late entries (except squeeze candidates)
            overext_details = overext.get("details", {})
            five_day_total = overext_details.get("five_day_total", five_day_return)
            sg = self.settings.scan_gates
            if (
                five_day_total > sg.late_entry_five_day_total_gt
                and rsi > sg.late_entry_rsi_gt
                and swing_type != "S"
            ):
                logger.debug(
                    f"{ticker}: Late entry rejected — 5d={five_day_total:+.0f}%, RSI={rsi:.0f}"
                )
                return None

            # ================================================================
            # V5.0: OBV DISTRIBUTION HARD GATE — Smart Money Filter
            # If OBV shows distribution (smart money selling), reject signal.
            # Exception: Type S (short squeeze) — distribution is expected
            # before a squeeze.
            # ================================================================
            if boosters.get('obv_distribution', False) and swing_type != 'S':
                logger.debug(
                    f"{ticker}: OBV Distribution — hard reject "
                    f"(smart money exiting, type={swing_type})"
                )
                _bump_scan_reject(reject_counts, "obv_distribution")
                return None

            # ================================================================
            # V5.0: TREND QUALITY GATE — Reject weak trend phases
            # If trend phase is "distribution" or "markdown", reject
            # unless it's a squeeze candidate.
            # ================================================================
            trend_data = swing_details.get("trend_quality", {})
            trend_phase = trend_data.get("trend_phase", "unknown")
            if trend_phase in ("distribution", "markdown") and swing_type != 'S':
                trend_strength = trend_data.get("trend_strength", 50)
                if trend_strength < 30:
                    logger.debug(
                        f"{ticker}: Weak trend phase '{trend_phase}' "
                        f"(strength={trend_strength}) — rejected"
                    )
                    _bump_scan_reject(reject_counts, "trend_phase_weak")
                    return None

            # Inject swing_type into boosters so scoring uses correct RSI penalty bands
            boosters['swing_type'] = swing_type

            quality_score = self.scoring.calculate_quality_score(
                df, volume_surge, atr_percent, float_shares, boosters
            )

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

                # OBV Trend (v3.0)
                'obv_accumulation': boosters.get('obv_accumulation', False),
                'obv_distribution': boosters.get('obv_distribution', False),
                'obv_bonus': boosters.get('obv_bonus', 0),
                
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
                    signal.copy(), df, portfolio_value=portfolio_value
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
            
            if backtest_mode:
                signal["technical_levels"] = None
                signal["narrative"] = None
                signal["narrative_text"] = ""
                signal["narrative_headline"] = f"{ticker} - Type {swing_type}"
            else:
                try:
                    tech_levels = calculate_technical_levels(
                        df, signal["entry_price"], signal.get("volume_surge", 1.0)
                    )
                    signal["technical_levels"] = tech_levels
                except Exception as e:
                    logger.debug(f"Could not calculate technical levels for {ticker}: {e}")
                    signal["technical_levels"] = None

                try:
                    narrative = generate_signal_narrative(signal, language="tr")
                    signal["narrative"] = narrative
                    signal["narrative_text"] = narrative.get("full_text", "")
                    signal["narrative_headline"] = narrative.get(
                        "headline", f"{ticker} - {swing_type}"
                    )
                except Exception as e:
                    logger.warning(f"Could not generate narrative for {ticker}: {e}")
                    signal["narrative"] = None
                    signal["narrative_text"] = ""
                    signal["narrative_headline"] = f"{ticker} - Type {swing_type}"

            return signal
            
        except Exception as e:
            logger.error(f"Error scanning {ticker}: {e}", exc_info=True)
            _bump_scan_reject(reject_counts, "scan_error")
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
        reject_counts: Dict[str, int] = {}

        logger.info(f"SmallCapEngine: Scanning {len(tickers)} stocks")

        # v4.0: Detect market regime ONCE for all stocks
        market_regime = self.signals.detect_market_regime()
        self._last_regime = market_regime  # expose for callers (e.g. scanner API)

        for ticker in tickers:
            if ticker not in data_dict:
                continue

            df = data_dict[ticker]
            scanned += 1

            signal = self.scan_stock(ticker, df, reject_counts=reject_counts)

            if signal:
                # Keep regime info for UI, but do not modify score (regime effect removed).
                signal['market_regime'] = market_regime['regime']
                signal['regime_confidence'] = market_regime.get('confidence', 'CONFIRMED')

                # Re-apply risk with real portfolio value if different from default
                if portfolio_value != 10000:
                    signal = self.risk.add_risk_management(signal, df, portfolio_value)
                signals.append(signal)

        # Sort by quality score (highest first)
        signals.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

        self._last_scan_reject_counts = reject_counts
        no_signal = scanned - len(signals)
        top_rejects = sorted(reject_counts.items(), key=lambda kv: -kv[1])[:5]
        reject_summary = ", ".join(f"{k}={v}" for k, v in top_rejects) if top_rejects else ""
        logger.info(
            f"SmallCapEngine: Scanned {scanned} | "
            f"Signals: {len(signals)} | "
            f"No signal: {no_signal} | "
            f"Regime: {market_regime['regime']}"
            + (f" | Rejects: {reject_summary}" if reject_summary else "")
        )

        return signals
    
    def get_small_cap_universe(
        self,
        use_finviz: Optional[bool] = None,
        max_tickers: Optional[int] = None,
    ) -> List[str]:
        """
        Get list of potential small-cap stocks to scan.

        Defaults (``use_finviz``, ``max_tickers``) come from ``settings.universe_scan``.
        Pass explicit values to override (e.g. dashboard preview with a smaller cap).

        Returns:
            List of ticker symbols
        """
        us = self.settings.universe_scan
        # cache_duration_minutes == 0 → always refetch; >0 → reuse in-memory Finviz list until TTL.
        force_refresh = us.cache_duration_minutes <= 0
        return self.universe_provider.get_universe(
            use_finviz=use_finviz,
            max_tickers=max_tickers,
            force_refresh=force_refresh,
        )

