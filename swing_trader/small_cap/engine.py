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
from .narrative import generate_signal_narrative
from .technical_levels import calculate_technical_levels
from .regime_logic import relative_strength_vs_spy
from .settings_config import load_settings
from .patterns import detect_weinstein_stage

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

        # TYPE S (SHORT SQUEEZE) KALDIRILDI — 2026-08-04.
        # Sınıflandırma short_interest + days_to_cover istiyordu; bu veriler
        # katalizör modülünden geliyordu ve o modül aynı gün skordan çıkarıldı
        # (geçmişe dönük veri olmadığı için backtest'te HER ZAMAN 0'dı → hiçbir
        # ölçümde Type S oluşamamıştı, yani hiç doğrulanmamıştı). Girdi sıfır
        # olunca dal ulaşılamaz hale geldi → ölü kod olarak silindi.
        # Yan etki: Type S'e tanınan gate muafiyetleri de anlamsızlaştı ve
        # kaldırıldı (RSI, Weinstein Stage 3, kalite eşiği).

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

        if type_b_score >= tb.min_score:
            has_safety = rsi <= tb.gate_rsi_safe_max
            if volume_surge < tb.gate_vol_min or not has_safety:
                pass
            else:
                if rsi > tb.rsi_overbought_hold_gt:
                    hold_days = tb.hold_overbought
                elif rsi > tb.rsi_elevated_gt:
                    hold_days = tb.hold_elevated
                else:
                    hold_days = tb.hold_default

                return (
                    "B",
                    hold_days,
                    f"🚀 Momentum: 5d={five_day_return:+.0f}%, RSI={rsi:.0f}, Vol-driven",
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
        regime: str = '',
        earnings_dates=None,
    ) -> Optional[Dict]:
        """
        Scan a single stock for small-cap momentum signal.

        backtest_mode: skip live yfinance fundamentals and catalysts
        (short/insider/news bonuses zero); optional spy_df_window for
        point-in-time RS vs SPY; skip narrative/LLM.

        earnings_dates: önceden çekilmiş bilanço tarihleri. Verilirse bilanço
        kapısı backtest'te de CANLI ile aynı şekilde uygulanır (parite). Ölçüm
        harness'ları bunu ticker başına bir kez çekip geçirmelidir — aksi halde
        canlının reddettiği bilanço-öncesi sinyaller ölçüme sızar.
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
                    # Finviz cache is free (no API call) — use it when available
                    finviz_meta = self.universe_provider.get_ticker_metadata(ticker)
                    stock_info = finviz_meta if finviz_meta is not None else self.get_stock_info(ticker)
            
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
                ticker, df, stock_info, signal_date_dt,
                backtest_mode=backtest_mode,
                earnings_dates=earnings_dates,
            )
            
            if not filter_passed:
                logger.debug(f"{ticker}: Failed filter - {filter_results.get('filters', {})}")
                _bump_scan_reject(reject_counts, "filter_failed")
                return None
            
            # Step 2: Check signal triggers (v13: VCE squeeze-breakout is primary)
            triggered, trigger_details = self.signals.check_all_triggers(df)

            if not triggered:
                logger.debug(f"{ticker}: No trigger - {trigger_details.get('triggers', {})}")
                _bump_scan_reject(reject_counts, "no_trigger")
                return None

            # Step 3: Get boosters (includes swing confirmation)
            boosters = self.signals.check_boosters(df)

            # Step 3.5: SWING CONFIRMATION GATE (NEW)
            # Must pass 5-day momentum > 0 AND Close > MA20
            # GATE DENETİMİ (2026-08-04): ΔEV −0.50 (TRAIN −1.00 / OOS −0.10).
            # EN KESKİN kapı: yalnız 4 sinyal ekliyor ama onların EV'si −7.16%.
            # Az sayıda ama felaket işlemleri engelliyor. DOKUNMA.
            swing_ready = boosters.get('swing_ready', False)
            swing_details = boosters.get('swing_details', {})

            if not swing_ready:
                # v13: no pullback bypass — all signals pass the same gates.
                # (VCE breakouts close above the prior 20-day high, which
                # mathematically implies Close > MA20 and positive 5d momentum,
                # so this gate only rejects malformed/edge-case data.)
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
            # SEKTÖR GÖRELİ GÜCÜ — canlı ve backtest AYNI fonksiyonu kullanır.
            # 2026-08-04 parite düzeltmesi: eskiden canlı SectorRS'in gerçek-ETF
            # hesabını, backtest ise SPY proxy'sini kullanıyordu → aynı hisse iki
            # yolda farklı skor alıyordu ve ölçüm canlıyı temsil etmiyordu.
            if (
                spy_df_window is not None
                and len(spy_df_window) >= 6
                and "Close" in spy_df_window.columns
                and len(df) >= 6
            ):
                sector_rs_data = relative_strength_vs_spy(df["Close"], spy_df_window["Close"])
            else:
                sector_rs_data = {"rs_score": 0.0, "is_leader": False}
            boosters["sector_rs_score"] = sector_rs_data.get("rs_score", 0.0)
            boosters["is_sector_leader"] = sector_rs_data.get("is_leader", False)

            # KATALİZÖR BONUSLARI KALDIRILDI — 2026-08-04. short-interest /
            # insider / haber verisi geçmişe dönük mevcut olmadığı için bu
            # bileşenler backtest'te HER ZAMAN 0'dı; canlıda ise skoru ortalama
            # +5.8 puan şişiriyorlardı (29 canlı sinyalde ölçüldü, maks +17).
            # Sonuç: canlı "Q80" eşiği ölçüm diliyle Q74 gibi davranıyordu.
            # Doğrulanamaz bir bileşenin eşiği 6 puan kaydırması kabul edilemez.

            # SEKTÖR ROTASYON BONUSU KALDIRILDI — 2026-08-04. Top-3 sektöre +5,
            # bottom-3'e −10 veriyordu ama `not backtest_mode` şartıyla korumalıydı:
            # yani backtest'te HER ZAMAN 0'dı, ölçümlerimize hiç girmedi. Canlıda
            # ise skoru ±5-10 puan kaydırıyordu. Doğrulanamaz bileşen + parite
            # kırığı → silindi (bkz. katalizör bonusları, aynı gerekçe).

            # RSI Divergence (already in signals but ensure it's in boosters)
            rsi_div = self.signals.detect_rsi_divergence(df, lookback=14)
            boosters['rsi_divergence'] = rsi_div['divergence_found']
            
            # MACD check
            macd_data = self.signals.calculate_macd(df)
            boosters['macd_bullish'] = macd_data['bullish_cross'] or (macd_data['above_zero'] and macd_data['expanding'])

            # Volume direction: UP day = institutional accumulation, DOWN day = distribution
            boosters['volume_up_day'] = (
                len(df) >= 2 and float(df['Close'].iloc[-1]) > float(df['Close'].iloc[-2])
            )

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
            swing_type, hold_days, type_reason = self._classify_swing_type(
                five_day_return, rsi, volume_surge, higher_lows,
                close_position=close_position,
                ma20_distance=ma20_distance,
                rsi_divergence=boosters.get('rsi_divergence', False),
                macd_bullish=boosters.get('macd_bullish', False)
            )
            
            # V4 hard RSI gate — v13: VCE pathway is EXEMPT.
            # Edge measurement by RSI bucket on VCE signals (R10 vs 24k-bar benchmark):
            #   RSI <70:   edge +6.51% (n=24)
            #   RSI 70-80: edge +5.14% (n=50)
            #   RSI 80+:   edge +4.53% (n=49)
            # All buckets positive — at a squeeze breakout, high RSI means strength,
            # not overextension. The old gate was killing 69% of validated VCE
            # signals. RSI still feeds scoring penalties for ranking.
            max_rsi = self.settings.max_entry_rsi
            _is_vce = trigger_details.get('trigger_pathway') == 'vce_breakout'
            if rsi > max_rsi and not _is_vce:
                logger.debug(f"{ticker}: RSI {rsi:.0f} > {max_rsi} — rejected (overbought, not squeeze)")
                _bump_scan_reject(reject_counts, "rsi_gate")
                return None

            # GEÇ GİRİŞ KAPISI SİLİNDİ — 2026-08-04 (measure_gate_value.py):
            # ΔEV tam 0.00, hiç ateşlenmiyordu. VCE muafiyeti + Weinstein Stage
            # 3/4 reddi + swing onayı bu vakaları zaten eliyordu.
            sg = self.settings.scan_gates

            # DAĞITIM GÜNÜ KAPISI SİLİNDİ — 2026-08-04 (measure_gate_value.py):
            # ΔEV tam 0.00, hiç ateşlenmiyordu. Sebep yapısal: VCE ve RVOL thrust
            # İKİSİ DE "yeşil kapanış" şartı koyuyor, dolayısıyla "hacimli düşüş
            # günü" bir sinyal olarak buraya hiç ulaşamıyordu.

            # Inject swing_type into boosters so scoring uses correct RSI penalty bands
            boosters['swing_type'] = swing_type

            # Weinstein Stage — YALNIZ hard gate icin. VCP (Minervini) tespiti
            # 2026-08-05'te silindi: 5 ciktisini (detected/contractions/
            # final_range_pct/volume_declining/bonus) hicbir kod okumuyordu —
            # skor bonus blogu sabite indirilince son tuketicisi de kalmadi.
            stage_data = detect_weinstein_stage(df)

            # ================================================================
            # WEINSTEIN STAGE HARD GATE
            # Only buy Stage 2 (markup) or Stage 1 turning up (anticipatory).
            # Stage 3 (distribution topping) and Stage 4 (decline) are hard
            # rejected — high-volume spikes in these stages are dead-cat
            # bounces or seller-driven, not the start of a new trend.
            # Type S exempt from Stage 3 only (squeeze can ignite in distribution).
            # Stage 0 = insufficient data → pass through (don't penalize data gaps).
            # ================================================================
            _wstage = stage_data.get('stage', 0)
            # GATE DENETİMİ (2026-08-04, measure_gate_value.py — GATE_AUDIT.md):
            # EN GÜÇLÜ KAPI. Kaldırılınca EV +3.11% → +2.13% (ΔEV −0.98;
            # TRAIN −1.46 / OOS −0.54, aynı yön). Eklediği 18 sinyalin EV'si
            # −2.10%: dağıtım (Stage 3) / düşüş (Stage 4) fazına girmek doğrudan
            # para kaybı. DOKUNMA.
            if _wstage > 0:
                if sg.reject_stage4 and _wstage == 4:
                    logger.debug(
                        f"{ticker}: Weinstein Stage 4 (Decline) — hard reject"
                    )
                    _bump_scan_reject(reject_counts, "stage_rejected")
                    return None
                if sg.reject_stage3 and _wstage == 3:
                    logger.debug(
                        f"{ticker}: Weinstein Stage 3 (Distribution) — hard reject "
                        f"(type={swing_type})"
                    )
                    _bump_scan_reject(reject_counts, "stage_rejected")
                    return None

            quality_score = self.scoring.calculate_quality_score(
                df, volume_surge, atr_percent, float_shares, boosters
            )

            # ================================================================
            # VCE İŞARETLERİ = YALNIZ SIRALAMA (baraja karışmaz) — 2026-08-05
            # ================================================================
            # premium-VCE (+8) ve tight-coil (+5) ölçülmüş gerçek bilgi taşıyor:
            # aynı sinyal sayısında bonuslu skorla sıralamak bonussuzdan DAHA İYİ
            # sonuç veriyor (n=71'de EV +6.33% vs +5.77%). Yani sıralama özelliği
            # olarak DEĞERLİ.
            #
            # AMA eskiden `quality_score`'a ekleniyordu ve o skor hem sıralamada
            # hem BARAJDA kullanılıyor. Sonuç: 34 sinyal yalnız bu ekleme
            # sayesinde Q80'in üstüne TAŞINIYORDU ve o 34'ün EV'si +0.89%
            # (taban +4.19%) — yani baraj onlar için fiilen Q72 gibi davranıyordu.
            #
            # Para ölçümü (scripts/measure_threshold_money.py, 21 ay, maliyet
            # dahil, slot-kısıtlı portföy — canlıda tip tavanı %20-25 olduğu için
            # 4-5 eşzamanlı pozisyon sığıyor):
            #     slot 3 : mevcut +53.7%  →  ayrıştırılmış +104.1%   (+50.4 puan)
            #     slot 4 : mevcut +47.5%  →  ayrıştırılmış  +76.0%   (+28.6 puan)
            #     slot 5 : mevcut +39.3%  →  ayrıştırılmış  +57.5%   (+18.2 puan)
            #     slot 8 : mevcut +39.2%  →  ayrıştırılmış  +50.6%   (+11.4 puan)
            # Sermaye sınırsız olsaydı mevcut kurgu kazanırdı (atılan işlemler
            # zarar ettirmiyor, +0.89% kazanıyor) — ama sermaye DAR: zayıf işlem
            # iyi bir işlemin slotunu kapatıyor. Fırsat maliyeti kârı yiyor.
            #
            # Çözüm: bilgiyi KORU, ama doğru yere koy.
            #   quality_score → HAM skor: baraj + gösterim (tek eşik anlamı)
            #   rank_score    → skor + işaretler: yalnız SIRALAMA
            _vce_metrics = (trigger_details or {}).get('vce_metrics', {})
            _rank_bonus = 0
            if _vce_metrics.get('is_premium'):
                _rank_bonus += 8
                boosters['vce_premium'] = True
            _sq = _vce_metrics.get('squeeze_ratio', 1.0)
            if 0 < _sq < 0.65:
                _rank_bonus += 5
                boosters['vce_tight_coil'] = True
            rank_score = quality_score + _rank_bonus

            # Regime-aware quality floor — 2026-07-27 REVİZE: tek kaynak.
            # Eskiden burada ayrı bir "base 70 + {BULL:-10,CAUTION:0,BEAR:+5}"
            # mantığı vardı; BULL'u 60'a düşürüyordu ve API katmanındaki
            # (thresholds.effective_scan_thresholds) rejim floor'larıyla
            # çelişiyordu (aynı eşik iki yerde, farklı değerlerle). Ölçüm
            # (scripts/measure_score_edge.py) BULL Q60-70'in ~0% getiri
            # verdiğini gösterdi. Artık motor-içi floor da API ile AYNI
            # regime_thresholds değerlerinden okunuyor — scan_stock'u doğrudan
            # çağıran backtest/edge yolları da aynı korumaya sahip. Type S muaf.
            _rt = self.settings.regime_thresholds
            _regime_floor = {
                "BULL": _rt.bull_min_quality,
                "CAUTION": _rt.caution_other_min_quality,
                "BEAR": _rt.bear_tentative_min_quality,
            }.get(regime, 0)  # UNKNOWN → 0 (rejim belirsiz, floor yok)
            # Type S (short squeeze) her rejimde muaf; diğerleri regime floor.
            _type_min_q = _regime_floor
            if quality_score < _type_min_q:
                # INFO level — surface quality scores in normal logs so we can see
                # the distribution of "almost made it" stocks and tune the bar.
                logger.info(
                    f"{ticker}: Q={quality_score:.1f} < type_{swing_type} min {_type_min_q} "
                    f"(regime={regime} floor) — rejected"
                )
                _bump_scan_reject(reject_counts, f"quality_type_{swing_type.lower()}")
                return None

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
                # Hangi tetikleyici kapıdan geldi: 'vce_breakout' | 'rvol_thrust'.
                # v14: RVOL thrust ikinci pathway olarak eklendi — sinyalin
                # kaynağı UI'da ve forward-return telemetrisinde görünür olsun.
                'trigger_pathway': trigger_details.get('trigger_pathway', 'vce_breakout'),
                'trigger_reason': trigger_details.get('trigger_reason', ''),
                'quality_score': round(quality_score, 1),        # HAM — baraj + gösterim
                'rank_score': round(rank_score, 1),              # +VCE işaretleri — yalnız sıralama
                'rank_bonus': _rank_bonus,
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
                'is_sector_leader': boosters.get('is_sector_leader', False),
                
                # RSI Divergence & MACD
                'rsi_divergence': boosters.get('rsi_divergence', False),
                'macd_bullish': boosters.get('macd_bullish', False),

                # OBV Trend (v3.0)
                'obv_accumulation': boosters.get('obv_accumulation', False),
                'obv_distribution': boosters.get('obv_distribution', False),
                'obv_bonus': boosters.get('obv_bonus', 0),

                # VCE kalite isaretleri — skoru gercekten kaydiran iki ekleme.
                # 2026-08-05: bunlar boosters'a yaziliyordu ama sinyal sozlugune
                # HIC girmiyordu; ne UI gosterebiliyor ne olcum okuyabiliyordu
                # (olcum harness'i False okuyup "hic atesle miyor" sandi).
                'vce_premium': boosters.get('vce_premium', False),
                'vce_tight_coil': boosters.get('vce_tight_coil', False),

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
            
            # R:R KAPISI SİLİNDİ — 2026-08-04 (measure_remaining_gates.py):
            # ΔEV tam 0.00, TRAIN ve OOS'ta da 0.00, hiç ek sinyal yok — 21 ayda
            # HİÇ ateşlenmemiş. Sebep yapısal: 2.5×ATR stop + T1 %10 + tipe özgü
            # T2 tavanları kombinasyonu matematiksel olarak neredeyse her zaman
            # R:R(T2) > 2.0 üretiyor, yani eşik zaten sağlanmış oluyor.
            # NOT: exit parametreleri (stop çarpanı / T2 tavanları) materyal
            # olarak değişirse bu kapı yeniden ÖLÇÜLMELİ — o zaman bağlayabilir.

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
                    narrative = generate_signal_narrative(signal)
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
        portfolio_value: float = 10000,
        progress_cb=None,
    ) -> List[Dict]:
        """
        Scan multiple stocks for small-cap momentum signals.
        
        Args:
            tickers: List of ticker symbols
            data_dict: Dict mapping ticker to DataFrame
            portfolio_value: Portfolio value for position sizing
            progress_cb: opsiyonel `fn(done, total)` — döngü ilerledikçe çağrılır.
                2026-08-05: eskiden bu döngü boyunca HİÇ ilerleme yayınlanmıyordu;
                API %84'te donuyor, kullanıcı taramanın asılıp asılmadığını
                anlayamıyor ve asılı-iş bekçisi meşru taramayı iptal edebiliyordu.

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

        current_regime = market_regime.get('regime', '')

        total = len(tickers)
        for idx, ticker in enumerate(tickers, 1):
            if progress_cb is not None and (idx % 20 == 0 or idx == total):
                try:
                    progress_cb(idx, total)
                except Exception:
                    pass        # ilerleme raporu taramayı asla düşürmemeli

            if ticker not in data_dict:
                continue

            df = data_dict[ticker]
            scanned += 1

            signal = self.scan_stock(ticker, df, reject_counts=reject_counts, regime=current_regime)

            if signal:
                signal['market_regime'] = current_regime
                signal['regime_confidence'] = market_regime.get('confidence', 'CONFIRMED')

                # Re-apply risk with real portfolio value and regime for accurate T2 scaling
                if portfolio_value != 10000:
                    signal = self.risk.add_risk_management(signal, df, portfolio_value, regime=current_regime)
                signals.append(signal)

        # SIRALAMA rank_score ile (ham skor + VCE işaretleri). Baraj ise ham
        # skora uygulanıyor — bkz. scan_stock'taki 2026-08-05 ayrıştırması.
        # Eşitlikte ham skor ikincil ölçüt.
        signals.sort(key=lambda x: (x.get('rank_score', 0), x.get('quality_score', 0)),
                     reverse=True)

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

