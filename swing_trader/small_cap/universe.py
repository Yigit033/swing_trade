"""
Small Cap Universe Provider - Dynamic universe using finvizfinance.
Provides fresh small-cap stock lists from Finviz screener.

OPTIMIZED v3.0 - "Para Kazanma Makinesi" Edition
- Pre-filters at Finviz level to avoid scanning 200+ stocks that will fail downstream
- Multi-factor momentum ranking (not just today's change%)
- Country filter to avoid ADR noise
- Float filter to focus on explosion potential
- Volatility filter to ensure tradeable setups
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd

if TYPE_CHECKING:
    from .settings_config import SmallCapSettings

logger = logging.getLogger(__name__)

_TICKER_SAFE_OVERVIEW_CLS = None


def _ticker_safe_overview_cls():
    """
    finvizfinance ``Overview``'ına ticker-güvenli parse yaması (lazy import).

    Finviz (2026-07) screener tablosunun ticker hücresine logo + ilk-harf
    fallback span'i ekledi::

        <td data-boxover-ticker="ARVN">
          <a class="company-ticker"><img .../><span>A</span></a>
          <a class="tab-link">ARVN</a>
        </td>

    Kütüphanenin kullandığı ``td.text`` iki text node'u birleştirip "AARVN"
    üretiyor — 1.2.0 ve 1.3.0 dahil upstream'de düzeltilmedi. Bu subclass
    Ticker kolonunu tab-link anchor'ından okur; sırasıyla fallback:
    ``data-boxover-ticker`` attribute'u → ``td.text``.
    """
    global _TICKER_SAFE_OVERVIEW_CLS
    if _TICKER_SAFE_OVERVIEW_CLS is not None:
        return _TICKER_SAFE_OVERVIEW_CLS

    from finvizfinance.screener.overview import Overview
    from finvizfinance.util import number_covert

    class _TickerSafeOverview(Overview):
        @staticmethod
        def _extract_ticker(col) -> str:
            link = col.find("a", class_="tab-link")
            if link is not None:
                text = link.get_text(strip=True)
                if text:
                    return text
            attr = col.get("data-boxover-ticker")
            if isinstance(attr, str) and attr.strip():
                return attr.strip()
            return col.text.strip()

        def _get_table(self, rows, df, num_col_index, table_header, limit=-1):
            rows = rows[1:]
            if limit != -1:
                rows = rows[0:limit]

            frame = []
            for row in rows:
                cols = row.find_all("td")[1:]
                info_dict = {}
                for i, col in enumerate(cols):
                    header = table_header[i]
                    if header == "Ticker":
                        info_dict[header] = self._extract_ticker(col)
                    elif i not in num_col_index:
                        info_dict[header] = col.text
                    else:
                        info_dict[header] = number_covert(col.text)
                frame.append(info_dict)
            if len(df) == 0:
                return pd.DataFrame(frame)
            return pd.concat([df, pd.DataFrame(frame)], ignore_index=True)

    _TICKER_SAFE_OVERVIEW_CLS = _TickerSafeOverview
    return _TickerSafeOverview


def build_rank_info(df: pd.DataFrame, cap: int) -> Dict:
    """
    Composite sıralamasından tavan (cap) telemetrisi üret — huni 3. aşama ölçümü.

    Sorgular bir VCE adayını getirse bile composite sıralaması onu
    ``max_scan_tickers`` tavanının altına gömüp kestirebilir. Kesilen
    ticker'lar burada kayda geçer ve tarama geçmişine yazılır; birkaç haftalık
    canlı veriyle "tavan kurbanı VCE var mı?" sorusu ek tarama maliyeti
    olmadan cevaplanır (2026-07-18, universe recall çalışmasının devamı).

    df: composite_score'a göre ÇOKTAN sıralanmış DataFrame ('Ticker' kolonu).
    """
    tickers = list(df['Ticker'])
    return {
        'ranked_total': len(tickers),
        'cap': cap,
        'ranks': {t: i + 1 for i, t in enumerate(tickers)},
        'cut_tickers': tickers[cap:],
    }


class SmallCapUniverse:
    """
    Dynamic small-cap universe provider using finvizfinance.

    PRE-SCREENING STRATEGY (v3.0):
    We run TWO Finviz queries and merge results for maximum coverage:

    Query 1 - MOMENTUM HUNTERS (aggressive):
      Market Cap: Small ($300M-$2B), Float: Under 100M
      Relative Volume: Over 1.5, Price: Over $3
      Country: USA, Volatility: Week Over 5%

    Query 2 - SETUP BUILDERS (wider net):
      Market Cap: Small ($300M-$2B), Float: Under 100M
      Average Volume: Over 1M, Price: Over $3
      Country: USA, RSI: Not Overbought (<60)

    Then rank by COMPOSITE MOMENTUM SCORE and return top N.
    """

    # Known delisted/problematic tickers to exclude
    EXCLUDED_TICKERS = {
        'BCOV', 'BGFV', 'CARA', 'GNOG', 'ZIRB', 'BBIG', 'IRNT', 'OPAD',
        'SPIR', 'CLOV', 'WISH', 'HOOD', 'LCID', 'RIVN', 'NKLA', 'WKHS',
        'FSR', 'GOEV', 'FFIE', 'MULN', 'RIDE', 'HYLN', 'ARVL', 'VLDR',
        'HCP', 'SQ', 'CERE', 'FREY', 'DTC', 'FTCH', 'ZNGA', 'VORB', 'TELL'
    }

    def __init__(self, config: Dict = None, settings: Optional[SmallCapSettings] = None):
        self.config = config or {}
        if settings is not None:
            self._settings = settings
        else:
            from .settings_config import load_settings

            self._settings = load_settings()
        self._us = self._settings.universe_scan
        self._cache = None
        self._cache_time = None
        self._cache_cap: Optional[int] = None
        self._finviz_df_cache: Optional[pd.DataFrame] = None
        self._last_rank_info: Optional[Dict] = None
        logger.info("SmallCapUniverse initialized (settings-backed scan + ranking)")

    def get_last_rank_info(self) -> Optional[Dict]:
        """Son Finviz fetch'inin sıralama/tavan telemetrisi (static path'te None)."""
        return self._last_rank_info

    def _parse_percent(self, value) -> float:
        """Parse percentage string like '5.23%' to float 5.23"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value.replace('%', '').replace(',', ''))
            return 0.0
        except Exception:
            return 0.0

    def _parse_volume(self, value) -> float:
        """Parse volume string like '1.5M' or '500K' to numeric"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                value = value.replace(',', '')
                if 'M' in value:
                    return float(value.replace('M', '')) * 1_000_000
                elif 'K' in value:
                    return float(value.replace('K', '')) * 1_000
                elif 'B' in value:
                    return float(value.replace('B', '')) * 1_000_000_000
                else:
                    return float(value)
            return 0.0
        except Exception:
            return 0.0

    def _parse_market_cap(self, value) -> float:
        """Parse market cap string to numeric"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                value = value.replace(',', '').replace('$', '').strip()
                if 'B' in value:
                    return float(value.replace('B', '')) * 1_000_000_000
                elif 'M' in value:
                    return float(value.replace('M', '')) * 1_000_000
                elif 'K' in value:
                    return float(value.replace('K', '')) * 1_000
                else:
                    return float(value)
            return 0.0
        except Exception:
            return 0.0

    def _run_finviz_query(self, filters_dict: Dict, label: str) -> pd.DataFrame:
        """Run a single Finviz screener query and return DataFrame."""
        try:
            foverview = _ticker_safe_overview_cls()()
            foverview.set_filter(filters_dict=filters_dict)
            df = foverview.screener_view()

            if df is None or len(df) == 0:
                logger.info(f"  [{label}] returned 0 results")
                return pd.DataFrame()

            logger.info(f"  [{label}] returned {len(df)} tickers")
            return df

        except Exception as e:
            logger.warning(f"  [{label}] query failed: {e}")
            return pd.DataFrame()

    def _calculate_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite momentum score using multiple factors.

        PHILOSOPHY (v4.1 — "Early Entry" Rebalance):
        A senior trader enters BEFORE the crowd, not after. The composite score
        now penalizes stocks that are already extended and rewards stocks that
        are setting up (high RVOL but modest price change = accumulation phase).

        COMPOSITE SCORE = (
            RVOL_Score         (rank_weight_rvol)   — abnormal volume = early institutional interest
            Change_Score       (rank_weight_change)  — today's move, with hard chasing penalties
            Volume_Score       (rank_weight_volume)  — absolute liquidity floor
            MCap_Score         (rank_weight_mcap)    — sweet spot $300M-$800M
        )

        KEY CHANGE vs v4.0:
        - "Early Accumulation Bonus": high RVOL + modest change (+1% to +8%) = best setup
          These are stocks being accumulated quietly before the move — highest priority.
        - Change score penalizes > +10% harder (chasing zone).
        - Source column is tracked so Setup-query stocks (RSI not overbought) get a
          small early-entry bias in tie-breaking.
        """
        import numpy as np

        # Parse columns
        if 'Change' in df.columns:
            df['change_pct'] = df['Change'].apply(self._parse_percent)
        else:
            df['change_pct'] = 0.0

        if 'Volume' in df.columns:
            df['vol_numeric'] = df['Volume'].apply(self._parse_volume)
        else:
            df['vol_numeric'] = 0.0

        if 'Rel Volume' in df.columns:
            df['rel_vol'] = pd.to_numeric(df['Rel Volume'], errors='coerce').fillna(1.0)
        elif 'Relative Volume' in df.columns:
            df['rel_vol'] = pd.to_numeric(df['Relative Volume'], errors='coerce').fillna(1.0)
        else:
            df['rel_vol'] = 1.0

        if 'Market Cap' in df.columns:
            df['mcap_numeric'] = df['Market Cap'].apply(self._parse_market_cap)
        else:
            df['mcap_numeric'] = 0.0

        # ============================================================
        # COMPONENT 1: Relative Volume Score (RVOL weight)
        # Scale: 1x=0, 1.5x=25, 2x=50, 3x=75, 5x+=100
        # ============================================================
        df['rvol_score'] = np.clip((df['rel_vol'] - 1.0) / 4.0 * 100, 0, 100)

        # ============================================================
        # COMPONENT 2: Price Change Score (change weight)
        # Sweet zone: +1% to +8% = early accumulation, not yet chased.
        # > +10%: steep penalty (stock already pumped, late entry risk).
        # Negative change on high RVOL = shakeout / accumulation (still ok).
        # ============================================================
        us = self._us
        change_pct = df['change_pct']

        # Base score: scaled by absolute change, up moves get 1.5x bias
        change_abs = change_pct.abs()
        up_bias = (change_pct > 0).astype(float) * 1.5 + 0.5
        df['change_score'] = np.clip(change_abs * up_bias * 6, 0, 100)

        # Chasing penalty: stocks that already moved big today get ranked lower
        chase_penalty = np.where(
            change_pct > us.chase_penalty_change_pct_high,   # default 15%
            us.chase_penalty_points_high,                     # default -50 pts
            np.where(
                change_pct > us.chase_penalty_change_pct_mid,  # default 10%
                us.chase_penalty_points_mid,                    # default -25 pts
                0,
            ),
        )
        df['change_score'] = np.clip(df['change_score'] - chase_penalty, 0, 100)

        # ============================================================
        # COMPONENT 3: Dollar Volume Score — REAL liquidity signal
        # Price × Shares = actual money flowing into the stock.
        # This correctly ranks a $4 stock with 10M shares ($40M DV)
        # over a $20 stock with 200K shares ($4M DV).
        # Log scale: $1M=29, $5M=50, $10M=59, $50M+=100
        # ============================================================
        if 'Price' in df.columns:
            df['price_numeric'] = pd.to_numeric(df['Price'], errors='coerce').fillna(15.0)
        else:
            df['price_numeric'] = 15.0
        df['dollar_vol_numeric'] = df['vol_numeric'] * df['price_numeric']
        df['vol_score'] = np.clip(
            np.log10(df['dollar_vol_numeric'].clip(lower=1)) / np.log10(50_000_000) * 100,
            0, 100
        )

        # ============================================================
        # COMPONENT 4: Market Cap Sweet Spot (mcap weight)
        # $300M-$800M: 100 (explosion potential), $800M-$1.5B: 70, $1.5B-$2.5B: 40
        # ============================================================
        mcap_m = df['mcap_numeric'] / 1_000_000
        df['mcap_score'] = np.where(
            mcap_m <= 800, 100,
            np.where(mcap_m <= 1500, 70,
                     np.where(mcap_m <= 2500, 40, 20))
        )

        # ============================================================
        # EARLY ACCUMULATION BONUS (v4.1 — new)
        # High RVOL (> 1.5x) + modest positive change (+1% to +8%)
        # = institutional accumulation before the crowd notices.
        # This is exactly the Type C setup we want at the top of the list.
        # Bonus: +15 pts — enough to break ties in favor of early-entry setups.
        # ============================================================
        early_accumulation = (
            (df['rel_vol'] >= 1.5) &
            (df['change_pct'] >= 1.0) &
            (df['change_pct'] <= 8.0)
        )
        df['early_bonus'] = np.where(early_accumulation, 15, 0)

        # ============================================================
        # COMPOSITE SCORE
        # ============================================================
        df['composite_score'] = (
            df['rvol_score'] * us.rank_weight_rvol
            + df['change_score'] * us.rank_weight_change
            + df['vol_score'] * us.rank_weight_volume
            + df['mcap_score'] * us.rank_weight_mcap
            + df['early_bonus']
        )

        return df

    def get_finviz_universe(self, max_tickers: Optional[int] = None) -> List[str]:
        """
        Get small-cap universe from Finviz with optimized pre-screening.

        Runs 2 queries for different profiles, merges and ranks by composite score.

        Returns:
            List of ticker symbols sorted by COMPOSITE MOMENTUM SCORE
        """
        try:
            cap = self._us.max_scan_tickers if max_tickers is None else max_tickers
            self._last_rank_info = None  # başarısız fetch'te bayat telemetri kalmasın
            logger.info("Fetching small-cap universe from Finviz (v3.0 optimized)...")

            frames: List[pd.DataFrame] = []

            # ============================================================
            # QUERY 1: MOMENTUM HUNTERS - Today's movers with tight float
            # ============================================================
            if self._us.enable_finviz_query_momentum:
                q1_filters = {
                    'Market Cap.': 'Small ($300mln to $2bln)',
                    'Float': 'Under 100M',
                    'Price': 'Over $7',
                    'Country': 'USA',
                    'Relative Volume': 'Over 1.5',
                    'Volatility': 'Week - Over 5%',
                    'Average Volume': 'Over 500K',
                }
                df1 = self._run_finviz_query(q1_filters, "MOMENTUM HUNTERS")
                if len(df1) > 0:
                    frames.append(df1)

            # ============================================================
            # QUERY 2: SETUP BUILDERS - High volume, not overbought
            # ============================================================
            if self._us.enable_finviz_query_setup:
                q2_filters = {
                    'Market Cap.': 'Small ($300mln to $2bln)',
                    'Float': 'Under 100M',
                    'Price': 'Over $7',
                    'Country': 'USA',
                    'Average Volume': 'Over 750K',
                    'RSI (14)': 'Not Overbought (<60)',
                    'Volatility': 'Week - Over 3%',
                }
                df2 = self._run_finviz_query(q2_filters, "SETUP BUILDERS")
                if len(df2) > 0:
                    frames.append(df2)

            # ============================================================
            # QUERY 3: WIDER NET - Larger float, strong volume
            # ============================================================
            if self._us.enable_finviz_query_wider:
                q3_filters = {
                    'Market Cap.': 'Small ($300mln to $2bln)',
                    'Price': 'Over $7',
                    'Country': 'USA',
                    'Relative Volume': 'Over 2',
                    'Average Volume': 'Over 1M',
                    'Volatility': 'Week - Over 5%',
                }
                df3 = self._run_finviz_query(q3_filters, "WIDER NET")
                if len(df3) > 0:
                    frames.append(df3)

            # ============================================================
            # QUERY 4 (EARLY SETUP, RSI<=40) KALDIRILDI — 2026-07-18 recall
            # ölçümü (scripts/measure_universe_recall.py): 408 doğrulanmış VCE
            # sinyalinin 0'ını yakaladı. RSI<=40 şartı, kırılım gününün doğasıyla
            # (green day + 20g yeni zirve) yapısal olarak çelişiyor. Q1-Q3 de
            # aynı ölçümle settings üzerinden kapatıldı (%0.5-2 katkı).
            # ============================================================

            # ============================================================
            # QUERY 5: VCE BREAKOUT DAY (v13 — PRIMARY THESIS ALIGNMENT)
            # The engine's primary trigger is the volatility-squeeze breakout
            # (VCE). A squeezed stock has LOW recent volatility, so queries
            # 1-3 (weekly volatility >5%) and 2/4 (RSI caps) systematically
            # miss it — exactly the stock we most want to see on the day it
            # breaks out. This query matches the VCE breakout-day signature:
            # green day + elevated volume, with NO volatility floor, NO RSI
            # cap and NO float cap (the validated rule has none of those).
            # ============================================================
            q5_filters = {
                'Market Cap.': 'Small ($300mln to $2bln)',
                'Price': 'Over $7',
                'Country': 'USA',
                'Average Volume': 'Over 500K',
                'Relative Volume': 'Over 1.5',
                'Change': 'Up 2%',
            }
            df5 = self._run_finviz_query(q5_filters, "VCE BREAKOUT DAY")
            if len(df5) > 0:
                frames.append(df5)

            # ============================================================
            # QUERY 5b: VCE BREAKOUT DAY — MID CAP ($2B-$10B) (v13.2)
            # The VCE edge was measured cap-agnostic: its strongest
            # contributors (HIMS, CELH, RDDT, TOST, MNDY) are mid-caps.
            # Finviz has no single $300M-$10B preset, so the mid band runs
            # as its own query. Higher liquidity bar (1M avg volume) keeps
            # execution quality tight for the larger names.
            # ============================================================
            q5b_filters = {
                'Market Cap.': 'Mid ($2bln to $10bln)',
                'Price': 'Over $7',
                'Country': 'USA',
                'Average Volume': 'Over 1M',
                'Relative Volume': 'Over 1.5',
                'Change': 'Up 2%',
            }
            df5b = self._run_finviz_query(q5b_filters, "VCE BREAKOUT DAY (MID)")
            if len(df5b) > 0:
                frames.append(df5b)

            # ============================================================
            # QUERY 6: 20-DAY NEW HIGH IN UPTREND (v13.3 — VCE precise feed)
            # The EXACT necessary condition for a VCE breakout is "closes at a
            # new 20-day high while above SMA50". The momentum/volatility
            # queries (1-3,5) select for stocks already MOVING — they
            # systematically miss a quiet squeeze that breaks out on LOW
            # volume / small change (which Variant B now accepts). This query
            # surfaces EVERY 20-day-high-in-uptrend stock regardless of today's
            # %change or RVOL, so the engine gets a shot at every real
            # squeeze breakout. The engine then confirms the squeeze itself.
            # No volatility/RSI/float cap — the validated rule has none.
            # ============================================================
            q6_filters = {
                'Market Cap.': 'Small ($300mln to $2bln)',
                'Price': 'Over $7',
                'Country': 'USA',
                'Average Volume': 'Over 500K',
                '50-Day Simple Moving Average': 'Price above SMA50',
                '20-Day High/Low': 'New High',
            }
            df6 = self._run_finviz_query(q6_filters, "20D NEW HIGH (small)")
            if len(df6) > 0:
                frames.append(df6)

            q6b_filters = {
                'Market Cap.': 'Mid ($2bln to $10bln)',
                'Price': 'Over $7',
                'Country': 'USA',
                'Average Volume': 'Over 1M',
                '50-Day Simple Moving Average': 'Price above SMA50',
                '20-Day High/Low': 'New High',
            }
            df6b = self._run_finviz_query(q6b_filters, "20D NEW HIGH (mid)")
            if len(df6b) > 0:
                frames.append(df6b)

            # ============================================================
            # MERGE & DEDUPLICATE
            # ============================================================
            if not frames:
                logger.warning("All Finviz queries returned empty")
                return []

            df = pd.concat(frames, ignore_index=True)
            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
            df = df.drop_duplicates(subset='Ticker', keep='first')

            logger.info(f"Merged universe: {len(df)} unique tickers")

            # ============================================================
            # POST-FILTERS (code-level precision)
            # ============================================================
            # Remove excluded tickers
            df = df[~df['Ticker'].isin(self.EXCLUDED_TICKERS)]

            pmin, pmax = self._us.post_filter_price_min, self._us.post_filter_price_max
            if 'Price' in df.columns:
                px = pd.to_numeric(df['Price'], errors='coerce')
                df = df[(px >= pmin) & (px <= pmax)]

            if len(df) == 0:
                logger.warning("All tickers filtered out after post-processing")
                return []

            # ============================================================
            # COMPOSITE MOMENTUM RANKING
            # ============================================================
            df = self._calculate_composite_score(df)

            # Sort by composite score (highest first)
            df = df.sort_values('composite_score', ascending=False)

            # Get top tickers
            tickers = df['Ticker'].head(cap).tolist()

            # Tavan telemetrisi (cap kesintisi + tam sıralama) — scanner stats'a akar
            self._last_rank_info = build_rank_info(df, cap)
            if self._last_rank_info['cut_tickers']:
                logger.info(
                    "Universe cap cut: %d ticker tavanın (%d) altında kaldı: %s",
                    len(self._last_rank_info['cut_tickers']), cap,
                    self._last_rank_info['cut_tickers'][:15],
                )

            # Log diagnostics
            top_cols = ['Ticker', 'Price', 'Change', 'Volume', 'composite_score']
            available_cols = [c for c in top_cols if c in df.columns]
            top_10 = df.head(10)[available_cols]
            logger.info(f"Top 10 momentum candidates:\n{top_10.to_string()}")
            logger.info(
                f"Selected {len(tickers)} tickers by COMPOSITE SCORE "
                f"(from {len(df)} after filters)"
            )

            # Cache full DataFrame for metadata lookup (market cap, sector, float)
            self._finviz_df_cache = df.copy()

            # Cache the results (cap must match for reuse — e.g. dashboard override 50 vs API 200)
            self._cache = tickers
            self._cache_time = datetime.now()
            self._cache_cap = cap

            return tickers

        except ImportError:
            logger.error("finvizfinance not installed. Run: pip install finvizfinance")
            return []
        except Exception as e:
            logger.error(f"Finviz screener error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_ticker_metadata(self, ticker: str) -> Optional[Dict]:
        """
        Return Finviz-sourced metadata for a ticker from the cached DataFrame.

        Eliminates per-ticker yfinance/Finnhub profile calls during scans.
        Returns None if cache is empty or ticker not found.
        """
        if self._finviz_df_cache is None or len(self._finviz_df_cache) == 0:
            return None

        df = self._finviz_df_cache
        row = df[df['Ticker'] == ticker]
        if row.empty:
            return None

        r = row.iloc[0]

        mcap = self._parse_market_cap(r.get('Market Cap', 0)) if 'Market Cap' in df.columns else 0.0
        sector = str(r.get('Sector', 'Unknown') or 'Unknown') if 'Sector' in df.columns else 'Unknown'
        industry = str(r.get('Industry', 'Unknown') or 'Unknown') if 'Industry' in df.columns else 'Unknown'
        # Finviz "Float" column uses same K/M/B notation as Volume
        float_shares = self._parse_volume(r.get('Float', 0)) if 'Float' in df.columns else 0.0

        return {
            'ticker': ticker,
            'marketCap': int(mcap),
            'floatShares': int(float_shares),
            'shortName': ticker,
            'sector': sector,
            'industry': industry,
        }

    def get_static_universe(self) -> List[str]:
        """
        Quality small-cap momentum universe — 300+ diversified names.
        Covers sectors: tech, industrial, consumer, healthcare, energy, defense.
        Criteria: market cap $250M-$2.5B, avg volume >500K, established momentum names.
        Used as fallback when Finviz is unavailable.
        """
        static_list = [
            # === TECHNOLOGY / SEMICONDUCTORS ===
            'ACLS', 'AEIS', 'AMBA', 'COHU', 'CRDO', 'ENTG', 'FORM', 'HIMX',
            'IPGP', 'IRDM', 'KLIC', 'LSCC', 'MCHP', 'MKSI', 'POWI', 'RMBS',
            'SANM', 'SMTC', 'SYNA', 'TSEM', 'VECO', 'VICR', 'WOLF', 'SITM',
            'OSIS', 'LYTS', 'DIOD', 'AOSL', 'AMAT', 'ONTO',

            # === SOFTWARE / CLOUD / AI ===
            'APPF', 'BRZE', 'CARG', 'CFLT', 'DOCN', 'DUOL', 'ESTC', 'GTLB',
            'HUBS', 'IONQ', 'MNDY', 'NCNO', 'PCTY', 'RAMP', 'RDDT', 'SMCI',
            'SOUN', 'TASK', 'TOST', 'VERX', 'WEAV', 'XPOF', 'ZI', 'AMPL',
            'BBAI', 'CXAI', 'RSKD', 'BMBL', 'LSPD', 'ENVX',

            # === DEFENSE / AEROSPACE ===
            'BWXT', 'CACI', 'DRS', 'FTAI', 'HII', 'KTOS', 'LHX', 'MRCY',
            'MOOG', 'RKLB', 'SPCE', 'TESI', 'VEC', 'ACHR', 'JOBY', 'LUNR',
            'RDW', 'ASTS', 'ASTR', 'MNTS',

            # === INDUSTRIALS / CONSTRUCTION ===
            'AAON', 'APOG', 'AWI', 'BCC', 'CSWI', 'DY', 'EPAC', 'FELE',
            'FLIR', 'GMS', 'HLIT', 'IBP', 'IESC', 'KFRC', 'LMB', 'MYRG',
            'NVT', 'POWL', 'ROAD', 'SKYW', 'SSD', 'TPIC', 'TPC', 'WLDN',
            'MLI', 'HAYW', 'GVP', 'NVEE', 'ROCK', 'SXI',

            # === ENERGY / CLEAN ENERGY / NUCLEAR ===
            'AROC', 'BORR', 'DNOW', 'FTLF', 'HLX', 'MNRL', 'NNE', 'OKLO',
            'RES', 'SMR', 'SOC', 'UUUU', 'WHD', 'OII', 'PUMP', 'TRGP',
            'SWN', 'NEXT', 'SHLS', 'NOVA', 'FLNC', 'BLNK', 'EVGO',

            # === HEALTHCARE / MEDICAL DEVICES ===
            'ACAD', 'ADMA', 'ALEC', 'CERT', 'CPRX', 'HALO', 'HIMS',
            'HRMY', 'IOVA', 'ITCI', 'LNTH', 'MDXH', 'NEOG', 'NTRA',
            'NVAX', 'PRCT', 'PRTA', 'PTGX', 'RDNT', 'ROIV', 'VCEL',
            'VRDN', 'ACLX', 'BCAB', 'CRSP', 'NBIX', 'PCVX', 'VKTX',

            # === CONSUMER / RESTAURANTS / RETAIL ===
            'BOOT', 'CAKE', 'CAVA', 'CELH', 'CHEF', 'CHUY', 'EAT', 'ELF',
            'FIGS', 'FIZZ', 'GRBK', 'JACK', 'KRUS', 'LOCO', 'PLAY', 'PTLO',
            'RVLV', 'SFM', 'SHAK', 'USPH', 'VITL', 'WING', 'XPOF', 'BROS',
            'TXRH', 'NCLH', 'HGV', 'MODV', 'OUST', 'LOVE',

            # === FINANCIAL / FINTECH ===
            'ARIS', 'AVNT', 'CATY', 'ESAB', 'EVTC', 'FCNCA', 'FULT', 'HCI',
            'HFWA', 'IIIV', 'JNPR', 'MGNI', 'NMIH', 'PAYO', 'PLMR', 'PPBI',
            'STEP', 'TBBK', 'TNET', 'TPVG', 'WSFS', 'CUBI', 'NBTB', 'FFBC',

            # === MATERIALS / SPECIALTY ===
            'AXTI', 'GATO', 'HWKN', 'KALU', 'MTRN', 'NGVT', 'PRIM', 'SXC',
            'TREC', 'USLM', 'WDFC', 'WTS', 'ZEUS', 'SLCA', 'FWRD', 'ATRI',

            # === MOMENTUM / BREAKOUT NAMES (current cycle) ===
            'AEHR', 'ATKR', 'NNE', 'SMR', 'OKLO', 'IONQ', 'RKLB',
            'SOFI', 'PLTR', 'ACHR', 'BBAI', 'SOUN', 'SMCI', 'CRDO',
            'NVAX', 'HIMS', 'CAVA', 'ELF', 'CELH', 'RDDT',
        ]

        # Filter out excluded & deduplicate
        seen = set()
        unique = []
        for t in static_list:
            if t not in self.EXCLUDED_TICKERS and t not in seen:
                seen.add(t)
                unique.append(t)

        logger.info(f"Static universe: {len(unique)} tickers")
        return unique

    def get_universe(
        self,
        use_finviz: Optional[bool] = None,
        max_tickers: Optional[int] = None,
        force_refresh: bool = True,
    ) -> List[str]:
        """
        Get small-cap universe from best available source.

        Args:
            use_finviz: If True, try Finviz first; None → ``universe_scan.use_finviz``.
            max_tickers: Cap; None → ``universe_scan.max_scan_tickers``.
            force_refresh: If False, may reuse in-memory cache when within TTL.

        Returns:
            List of ticker symbols sorted by COMPOSITE MOMENTUM SCORE
        """
        us = self._us
        uf = us.use_finviz if use_finviz is None else use_finviz
        cap = us.max_scan_tickers if max_tickers is None else max_tickers

        cache_mins = us.cache_duration_minutes
        if (
            cache_mins > 0
            and not force_refresh
            and self._cache
            and self._cache_time
            and self._cache_cap == cap
            and datetime.now() - self._cache_time < timedelta(minutes=cache_mins)
        ):
            logger.info(f"Using cached universe ({len(self._cache)} tickers, cap={cap})")
            return self._cache[:cap]

        logger.info("Fetching FRESH momentum-ranked universe (cache bypassed)")

        if uf:
            finviz_tickers = self.get_finviz_universe(cap)
            min_skip = us.min_finviz_tickers_skip_static_merge
            if len(finviz_tickers) >= min_skip:
                return finviz_tickers
            logger.warning(
                "Finviz returned only %s tickers (< min_finviz_tickers_skip_static_merge=%s), merging with static",
                len(finviz_tickers),
                min_skip,
            )
            static = self.get_static_universe()
            merged = list(dict.fromkeys(finviz_tickers + static))
            # Static merge sıralamayı bozar — rank telemetrisi bu evren için geçersiz
            self._last_rank_info = None
            return merged[:cap]

        self._last_rank_info = None  # static path: composite sıralama yok
        return self.get_static_universe()[:cap]

    def validate_ticker(self, ticker: str) -> bool:
        """Check if ticker is valid (not in exclusion list)."""
        return ticker not in self.EXCLUDED_TICKERS
