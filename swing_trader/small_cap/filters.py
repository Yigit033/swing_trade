"""
Small Cap Universe Filters - Hard filters for stock selection.
Completely independent from LargeCap filters.

SENIOR TRADER OPTIMIZED v2.0
"""

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .settings_config import SmallCapSettings

logger = logging.getLogger(__name__)


class SmallCapFilters:
    """
    Universe selection filters for Small Cap Momentum Engine.
    
    SENIOR TRADER SPECS (Agresif):
    - Market Cap: $250M - $2.5B (true small-cap)
    - Float: ≤ 60M shares (EXPLOSION POTENTIAL!)
    - Avg Volume (20d): >= 750K shares
    - ATR%: >= 3.5% (type-specific)
    - Price: $3 - $200
    - Earnings ±7 days: REJECT
    
    Float Tiering:
    - ≤15M:  ATOMIC (+20 pts)
    - 15-30M: MICRO (+15 pts)
    - 30-45M: SMALL (+10 pts)
    - 45-60M: TIGHT (+5 pts)
    - >150M: REJECT
    """
    
    # SENIOR TRADER FILTER CONSTANTS
    MIN_MARKET_CAP = 250_000_000      # $250M
    MAX_MARKET_CAP = 2_500_000_000    # $2.5B
    MIN_DOLLAR_VOLUME = 5_000_000     # $5M/day — price×shares, real liquidity (replaces share count)
    MIN_ATR_PERCENT = 0.03            # Default; live value from SmallCapSettings.min_atr_percent
    MAX_FLOAT = 80_000_000            # 80M shares — true small-cap explosion potential (was 150M)
    MIN_PRICE = 8.00                  # $8 — institutional participation floor (was $3)
    MAX_PRICE = 200.00                # $200
    EARNINGS_EXCLUSION_DAYS = 3       # pre-earnings only: 0→+3 days (post-earnings NOT blocked)
    ATR_PERIOD = 10                   # 10-period ATR (faster reaction)
    
    # FLOAT TIERING THRESHOLDS (for scoring bonus)
    FLOAT_TIER_ATOMIC = 15_000_000    # ≤15M: +20 pts
    FLOAT_TIER_MICRO = 30_000_000     # 15-30M: +15 pts
    FLOAT_TIER_SMALL = 45_000_000     # 30-45M: +10 pts
    FLOAT_TIER_TIGHT = 60_000_000     # 45-60M: +5 pts
    
    def __init__(self, config: Dict = None, settings: Optional["SmallCapSettings"] = None):
        """Initialize SmallCapFilters."""
        from .settings_config import load_settings

        self.config = config or {}
        self._settings = settings if settings is not None else load_settings()
        uf = self._settings.universe_filters
        self.MIN_MARKET_CAP = uf.min_market_cap
        self.MAX_MARKET_CAP = uf.max_market_cap
        self.MAX_FLOAT = uf.max_float_shares
        self.MIN_PRICE = uf.min_price
        self.MAX_PRICE = uf.max_price
        self.EARNINGS_EXCLUSION_DAYS = uf.earnings_exclusion_days
        self.ATR_PERIOD = uf.atr_period
        logger.info("SmallCapFilters initialized (Senior Trader v2.0)")
    
    def calculate_atr_percent(self, df: pd.DataFrame, period: int = None) -> float:
        """Calculate ATR as percentage of close price. Uses 10-period for faster reaction."""
        if period is None:
            period = self.ATR_PERIOD  # Default 10
            
        if df is None or len(df) < period + 1:
            return 0.0
        
        try:
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1])
                )
            )
            
            atr = np.mean(tr[-period:])
            current_close = close[-1]
            
            return atr / current_close if current_close > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating ATR%: {e}")
            return 0.0
    
    def calculate_avg_dollar_volume(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate average DOLLAR volume (price × shares) over period.

        Dollar volume is the correct liquidity measure:
        - A $4 stock trading 10M shares = $40M DV → excellent
        - A $15 stock trading 250K shares = $3.75M DV → too thin
        Share-count-only filters get this backwards.
        """
        if df is None or len(df) < period:
            return 0.0
        try:
            tail = df.tail(period)
            return float((tail['Volume'] * tail['Close']).mean())
        except Exception:
            return 0.0
    
    def check_market_cap(self, market_cap: float) -> Tuple[bool, str]:
        """Check if market cap is within small-cap range."""
        if market_cap is None or market_cap <= 0:
            return True, "Market cap unknown (allowing — ticker pre-screened)"
        
        if market_cap < self.MIN_MARKET_CAP:
            return False, f"Market cap too small (${market_cap/1e6:.0f}M < $250M)"
        
        if market_cap > self.MAX_MARKET_CAP:
            return False, (
                f"Market cap too large (${market_cap/1e9:.1f}B > "
                f"${self.MAX_MARKET_CAP/1e9:.0f}B)"
            )
        
        return True, f"Market cap OK (${market_cap/1e6:.0f}M)"
    
    def check_dollar_volume(self, avg_dollar_vol: float) -> Tuple[bool, str]:
        """
        Check minimum dollar volume — the real liquidity gate.
        $5M/day ensures entries/exits don't move the price and
        confirms real institutional interest (not just penny stock noise).
        """
        min_dv = self.MIN_DOLLAR_VOLUME
        if avg_dollar_vol < min_dv:
            return False, f"Dollar volume too low (${avg_dollar_vol/1e6:.1f}M/day < $5M)"
        return True, f"Dollar volume OK (${avg_dollar_vol/1e6:.1f}M/day)"
    
    def check_atr_percent(self, atr_pct: float) -> Tuple[bool, str]:
        """Check if ATR% meets volatility threshold."""
        min_atr = self._settings.min_atr_percent
        if atr_pct < min_atr:
            return False, f"ATR% too low ({atr_pct*100:.1f}% < {min_atr*100:.1f}%)"
        return True, f"ATR% OK ({atr_pct*100:.1f}%)"
    
    def check_float(self, float_shares: float) -> Tuple[bool, str]:
        """
        Check if float is within explosion range (≤80M).

        80M is the hard ceiling. Above this, a stock behaves like mid-cap —
        institutional supply absorbs retail buying and kills the explosive move.
        Below 20M, a single catalyst can double the stock in days.
        """
        if float_shares is None or float_shares <= 0:
            return True, "Float unknown (allowing)"

        if float_shares > self.MAX_FLOAT:
            return False, f"Float too large ({float_shares/1e6:.0f}M > 80M — mid-cap behavior, no explosion)"

        if float_shares <= self.FLOAT_TIER_ATOMIC:
            return True, f"Float ATOMIC ({float_shares/1e6:.0f}M) +20pts"
        elif float_shares <= self.FLOAT_TIER_MICRO:
            return True, f"Float MICRO ({float_shares/1e6:.0f}M) +15pts"
        elif float_shares <= self.FLOAT_TIER_SMALL:
            return True, f"Float SMALL ({float_shares/1e6:.0f}M) +10pts"
        elif float_shares <= self.FLOAT_TIER_TIGHT:
            return True, f"Float TIGHT ({float_shares/1e6:.0f}M) +5pts"
        else:  # 60-80M
            return True, f"Float OK ({float_shares/1e6:.0f}M) +0pts"
    
    def check_price(self, price: float) -> Tuple[bool, str]:
        """Check if price is within acceptable range ($3-$200)."""
        if price is None or price <= 0:
            return False, "Unknown price"
        
        if price < self.MIN_PRICE:
            return False, f"Price too low (${price:.2f} < ${self.MIN_PRICE:.0f} — below institutional participation floor)"
        
        if price > self.MAX_PRICE:
            return False, f"Price too high (${price:.2f} > $200)"
        
        return True, f"Price OK (${price:.2f})"
    
    def check_earnings(self, ticker: str, signal_date, earnings_dates=None) -> Tuple[bool, str]:
        """Bilanço-öncesi pencereyi (0..+3 gün) reddet.

        earnings_dates verilirse yfinance'e HİÇ gidilmez — ölçüm harness'ları
        tarihleri ticker başına bir kez çekip buraya geçirebilir. Bu, canlı ile
        backtest'in AYNI kapıyı kullanmasını sağlar (bkz. apply_all_filters).

        Bilanço-SONRASI (negatif days_diff) bilerek engellenmez: gap + devam
        en iyi swing kurulumlarından biridir.
        """
        from datetime import datetime

        try:
            if earnings_dates is None:
                import yfinance as yf
                earnings_dates = yf.Ticker(ticker).earnings_dates

            if earnings_dates is None or len(earnings_dates) == 0:
                return True, "No nearby earnings"

            if isinstance(signal_date, str):
                signal_date = datetime.strptime(signal_date, '%Y-%m-%d')

            # DataFrame/Series ise tarihler index'te; düz liste/DatetimeIndex ise
            # doğrudan üzerinde gezilir. DİKKAT: list.index bir METOT olduğu için
            # getattr(x, "index", x) ile bakmak listeyi sessizce bozar.
            if isinstance(earnings_dates, (pd.DataFrame, pd.Series)):
                candidates = earnings_dates.index
            else:
                candidates = earnings_dates

            for earn_date in candidates:
                earn_ts = pd.Timestamp(earn_date)
                if earn_ts.tzinfo is not None:
                    earn_ts = earn_ts.tz_localize(None)
                days_diff = (earn_ts.date() - signal_date.date()).days
                if 0 <= days_diff <= self.EARNINGS_EXCLUSION_DAYS:
                    return False, f"Earnings in {days_diff} days (pre-event risk)"

            return True, "No nearby earnings"

        except Exception as e:
            logger.debug(f"Earnings check error for {ticker}: {e}")
            return True, "Earnings check skipped"
    
    def apply_all_filters(
        self,
        ticker: str,
        df: pd.DataFrame,
        stock_info: Dict,
        signal_date=None,
        backtest_mode: bool = False,
        earnings_dates=None,
    ) -> Tuple[bool, Dict]:
        """
        Apply all hard filters to a stock.
        
        Returns:
            Tuple of (passed: bool, results: dict with filter details)
        """
        from datetime import datetime
        
        results = {
            'ticker': ticker,
            'passed': False,
            'filters': {}
        }
        
        if signal_date is None:
            signal_date = datetime.now()
        
        # 0. Price range ($3–$200)
        if df is not None and len(df) > 0:
            last_close = float(df['Close'].iloc[-1])
            passed, reason = self.check_price(last_close)
            results['filters']['price'] = {'passed': passed, 'reason': reason, 'value': last_close}
            if not passed:
                return False, results
        
        # 1. Market Cap
        market_cap = stock_info.get('marketCap', 0) or stock_info.get('market_cap', 0)
        if backtest_mode and (market_cap is None or market_cap <= 0):
            market_cap = int(self.MIN_MARKET_CAP * 1.2)
        passed, reason = self.check_market_cap(market_cap)
        results['filters']['market_cap'] = {'passed': passed, 'reason': reason, 'value': market_cap}
        if not passed:
            return False, results
        
        # 2. Dolar hacim — gerçek likidite kapısı ($5M/gün)
        # Anahtar adı 2026-08-06'da 'avg_volume' → 'dollar_volume' oldu: /lookup
        # sayfası bu kapıyı "Ortalama Hacim" diye etiketliyordu, oysa hisse adedi
        # değil DOLAR hacmi ölçülüyor. Etiket kullanıcıya yanlış kapıyı gösteriyordu.
        avg_dollar_vol = self.calculate_avg_dollar_volume(df)
        passed, reason = self.check_dollar_volume(avg_dollar_vol)
        results['filters']['dollar_volume'] = {'passed': passed, 'reason': reason, 'value': avg_dollar_vol}
        if not passed:
            return False, results
        
        # 3. ATR% — ADVISORY ONLY since v13 (recorded, never rejects).
        # The old hard gate (ATR >= 3%) belonged to the momentum-chasing thesis
        # and directly contradicts the validated VCE primary trigger: a
        # volatility-squeezed stock has LOW ATR by definition, and the edge
        # measurement showed squeeze->expansion is the only entry with
        # statistically significant forward edge. Tradeability is enforced by
        # price/market-cap/dollar-volume gates above; stop/target sizing uses
        # live ATR in the risk module.
        #
        # MUTLAK TABAN (ör. ATR >= %1.5) DA REDDEDİLDİ — 2026-08-14,
        # scripts/measure_atr_floor.py. Hipotez makuldü: göreceli düşük ATR
        # (sıkışma) istediğimiz şey, ama MUTLAK düşük ATR "ölü tahta"
        # (birleşme-arbitraj: fiyat nakit teklife çivilenmiş) olabilir ve orada
        # hedefe giden yol yapısal olarak kapalıdır. Ölçüm bunu ÇÜRÜTTÜ:
        #   - 64 sinyalin (VCE + baraj sonrası RVOL) EN DÜŞÜK ATR'si %1.46;
        #     ölü-tahta kümesi diye bir şey YOK. Barajın yakalayacağı av yok.
        #   - En düşük kova bile para kazanıyor: %1.5-2.0 -> EV +5.44% (n=3).
        #   - ATR>=%1.5 tek işlem eliyor (n=1, kabul barajı n>=5 -> düşer) ve
        #     TRAIN örneklemini hiç değiştirmiyor: kanıt değil, tek veriye uyum.
        #   - Yüksek barajlar cazip görünüyor (ATR>=%3 -> EV +5.37%) AMA elenen
        #     kova hâlâ POZİTİF (+1.98%) ve 33 VCE sinyalinin 20'sini kesiyor:
        #     kârlı işlem kıymak, zarar kesmek değil.
        # Ölü tahtayı zaten mevcut yapı eliyor: RVOL barajları olay gününü,
        # VCE'nin genişleme şartı da çivilenmiş fiyatı bloke ediyor (teklife
        # kilitli tahta genişleme üretemez). Bu kapı GEREKSİZ; tavsiye kalıyor.
        atr_pct = self.calculate_atr_percent(df)
        passed, reason = self.check_atr_percent(atr_pct)
        results['filters']['atr_percent'] = {
            'passed': True, 'reason': f"{reason} (advisory)", 'value': atr_pct
        }
        
        # 4. Float — ADVISORY ONLY since v13 (recorded + scoring tier bonus,
        # never rejects). The >80M hard reject belonged to the micro-cap
        # "explosion" thesis. The VCE edge was validated on a universe that is
        # predominantly LARGE-float (HIMS, SOFI, RKLB, SOUN, TOST, RDDT...),
        # and backtest_mode bypassed this gate (45M default) — so the proven
        # pipeline never enforced it. Keeping it live would reject exactly the
        # population the edge was measured on. Float sıkılığı kalite skoruna
        # scoring.score_float_tightness ile giriyor (ağırlık %25).
        float_shares = stock_info.get('floatShares', 0) or stock_info.get('float_shares', 0)
        if backtest_mode and (float_shares is None or float_shares <= 0):
            float_shares = 45_000_000
        passed, reason = self.check_float(float_shares)
        results['filters']['float'] = {
            'passed': True,
            'reason': reason if passed else f"{reason} (advisory)",
            'value': float_shares,
        }
        
        # 5. Bilanço kapısı.
        # PARİTE (2026-08-14): eskiden backtest'te bu kapı TAMAMEN atlanıyordu,
        # yani tüm eşiklerimiz canlı sistemin işlem ETMEDİĞİ bilanço-öncesi
        # sinyalleri içeren bir örneklemde ayarlanmıştı. Kanıtlanmış vaka:
        # LIFE 2026-08-03'te RVOL thrust ateşledi ve ölçümde +52% yazdı, ama
        # bilanço aynı gün 16:00'da açıklandığı için canlı motor o sinyali
        # "Earnings in 0 days" ile reddediyordu. Ölçtüğümüz şey ürün değildi.
        # (Aynı sınıf hata katalizör bonusları ve sektör RS'inde de vardı.)
        #
        # Çözüm: harness'lar earnings_dates'i ticker başına BİR KEZ çekip
        # geçirir; kapı o zaman backtest'te de aynen çalışır. Tarih verilmezse
        # eski davranış korunur ama sebep artık dürüst yazılıyor.
        if earnings_dates is not None:
            passed, reason = self.check_earnings(ticker, signal_date, earnings_dates)
            results['filters']['earnings'] = {'passed': passed, 'reason': reason}
            if not passed:
                return False, results
        elif backtest_mode:
            results['filters']['earnings'] = {
                'passed': True,
                'reason': 'Bilanço verisi geçirilmedi (canlıdan SAPMA — parite eksik)',
            }
        else:
            passed, reason = self.check_earnings(ticker, signal_date)
            results['filters']['earnings'] = {'passed': passed, 'reason': reason}
            if not passed:
                return False, results
        
        results['passed'] = True
        results['atr_percent'] = atr_pct
        results['market_cap'] = market_cap
        results['float_shares'] = float_shares
        
        return True, results
