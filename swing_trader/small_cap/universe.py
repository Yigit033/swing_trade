"""
Finviz tabanlı small/mid-cap aday havuzu (evren) sağlayıcısı.

Dört sorgu koşar ve birleştirir:
  Q6/Q6b  20 günlük yeni zirve + SMA50 üstü  → VCE tetiğinin ön koşulu
  Q7/Q7b  RVOL > 2 + yeşil + SMA20 üstü      → RVOL thrust beslemesi
(small/mid ayrımı Finviz hacim eşiği bandına göre.)

Sonuç dolar-hacme göre sıralanıp ``max_scan_tickers`` tavanına kırpılır.
Finviz erişilemezse static yedek listeye düşer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd

if TYPE_CHECKING:
    from .settings_config import SmallCapSettings

logger = logging.getLogger(__name__)

# ── Finviz keşif bantları — TEK KAYNAK ───────────────────────────────────
# Q6/Q6b/Q7/Q7b sorgularının "Average Volume" eşiği. Burada durmalarının sebebi
# canlı ile backtest harness'ının (scripts/backtest_live_replica.py finviz_hit)
# AYNI bandı kullanmak zorunda olması — ayrışırlarsa backtest artık ürünü ölçmez.
#
# 2026-08-03: 500K/1M → 300K/500K (ÖLÇÜLDÜ, curve-fit değil).
#   Sebep : canlı üretim ~0.6 Q80 sinyal/ay; profesyonel pratik 4-12 işlem/ay.
#   Ölçüm : scripts/analyze_signal_lab.py — 108 sinyal / 21 ay, 8 kapı tek tek
#           gevşetildi. Dolar-hacim, mcap ve fiyat kapılarını gevşetmek HİÇ ek
#           sinyal getirmedi; yalnız bu bant getirdi.
#   Sonuç : +20 sinyal (%23), ek sinyallerin EV'si +2.16%, toplam EV +2.32% →
#           +2.29% (seyrelme yok), OOS (2025-06-01) train +2.34% / test +2.25%.
#   Not   : Likidite standardı DÜŞMEDİ — motorun $5M/gün dolar-hacim hard-gate'i
#           (filters.apply_all_filters) aynen duruyor. Bu yalnız KEŞİF katmanı.
FINVIZ_MIN_AVG_VOLUME_SMALL = "Over 300K"
FINVIZ_MIN_AVG_VOLUME_MID = "Over 500K"

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
    Tavan (cap) telemetrisi — kesilen ticker'lar tarama geçmişine yazılır ki
    "tavan bir VCE adayını kurban etti mi?" sorusu ölçülebilsin.

    df: sıralanmış DataFrame ('Ticker' kolonu zorunlu).
    """
    tickers = list(df['Ticker'])
    return {
        'ranked_total': len(tickers),
        'cap': cap,
        'ranks': {t: i + 1 for i, t in enumerate(tickers)},
        'cut_tickers': tickers[cap:],
    }


class SmallCapUniverse:
    """Finviz sorgularını koşar, birleştirir, dolar-hacme göre sıralar."""

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

    def get_finviz_universe(self, max_tickers: Optional[int] = None) -> List[str]:
        """Dört Finviz sorgusunu koş, birleştir, dolar-hacme göre sırala, tavana kırp."""
        try:
            cap = self._us.max_scan_tickers if max_tickers is None else max_tickers
            self._last_rank_info = None  # başarısız fetch'te bayat telemetri kalmasın
            logger.info("Fetching small-cap universe from Finviz (v3.0 optimized)...")

            frames: List[pd.DataFrame] = []

            # DÖRT AKTİF SORGU: Q6/Q6b (20g yeni zirve — VCE'nin ön koşulu) ve
            # Q7/Q7b (RVOL patlaması — RVOL thrust'ın beslemesi). Small ve mid
            # bantları ayrı çünkü Finviz hacim eşiği bandına göre değişiyor.
            # Kaldırılmış sorguların (Q1-Q5) ölçüm gerekçeleri: GATE_AUDIT.md +
            # scripts/measure_universe_recall.py. Kodları git geçmişinde.
            q6_filters = {
                'Market Cap.': 'Small ($300mln to $2bln)',
                'Price': 'Over $7',
                'Country': 'USA',
                'Average Volume': FINVIZ_MIN_AVG_VOLUME_SMALL,
                '50-Day Simple Moving Average': 'Price above SMA50',
                '20-Day High/Low': 'New High',
            }
            df6 = self._run_finviz_query(q6_filters, "20D NEW HIGH (small)")
            if len(df6) > 0:
                frames.append(df6)

            # Mid bandı da aynı ölçümle gevşetildi: 'Over 1M' → 'Over 500K'
            # (harness'ın topladığı bant: small 300K / mid 500K).
            q6b_filters = {
                'Market Cap.': 'Mid ($2bln to $10bln)',
                'Price': 'Over $7',
                'Country': 'USA',
                'Average Volume': FINVIZ_MIN_AVG_VOLUME_MID,
                '50-Day Simple Moving Average': 'Price above SMA50',
                '20-Day High/Low': 'New High',
            }
            df6b = self._run_finviz_query(q6b_filters, "20D NEW HIGH (mid)")
            if len(df6b) > 0:
                frames.append(df6b)

            # ============================================================
            # QUERY 7: RVOL THRUST FEED (v14 — 2026-07-26)
            # İkinci sinyal pathway'i RVOL thrust'ı besler (signals.py
            # check_rvol_thrust): anormal hacim + yeşil + MA20 üstü. Q6/Q6b
            # "20g yeni zirve" ister; RVOL thrust hisse ZİRVEDE OLMADAN da
            # ateşleyebildiği için Q6/Q6b onun %25'ini kaçırıyordu — ve ölçümde
            # o kaçırılan %25, yakalananlardan DAHA İYİ edge veriyordu (exit-EV
            # +3.08% vs +1.34%, WR %67 vs %60). Bu sorgu o boşluğu kapatır.
            # Finviz karşılığı: RelVol>2 + Change:Up + SMA20 üstü. Canlı testte
            # dar (bugün 2+9=11 ticker) — tarama maliyeti ihmal edilebilir, çöp
            # patlaması yok. Motorun RVOL thrust trigger'ı (RelVol>=2.5) son
            # kararı verir; bu sorgu yalnız aday havuzunu besler.
            # ============================================================
            # Hacim bandı Q6 ile aynı ölçüme göre gevşetildi (500K → 300K).
            q7_filters = {
                'Market Cap.': 'Small ($300mln to $2bln)',
                'Price': 'Over $7',
                'Country': 'USA',
                'Average Volume': FINVIZ_MIN_AVG_VOLUME_SMALL,
                'Relative Volume': 'Over 2',
                'Change': 'Up',
                '20-Day Simple Moving Average': 'Price above SMA20',
            }
            df7 = self._run_finviz_query(q7_filters, "RVOL THRUST (small)")
            if len(df7) > 0:
                frames.append(df7)

            q7b_filters = {
                'Market Cap.': 'Mid ($2bln to $10bln)',
                'Price': 'Over $7',
                'Country': 'USA',
                'Average Volume': FINVIZ_MIN_AVG_VOLUME_MID,
                'Relative Volume': 'Over 2',
                'Change': 'Up',
                '20-Day Simple Moving Average': 'Price above SMA20',
            }
            df7b = self._run_finviz_query(q7b_filters, "RVOL THRUST (mid)")
            if len(df7b) > 0:
                frames.append(df7b)

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

            # Sıralama: DOLAR-HACİM (fiyat × hacim), en likit önce.
            # Tek işi tavan bağladığında hangi adayların taranacağını seçmek.
            # Neden likidite: measure_price_band.py kesişim testi (fiyat sabit,
            # likidite değişken) likit grupta +3.31% / illikitte −2.14% verdi.
            df['dollar_volume'] = (
                df['Volume'].apply(self._parse_volume)
                * pd.to_numeric(df.get('Price'), errors='coerce').fillna(15.0)
            )
            df = df.sort_values('dollar_volume', ascending=False)
            tickers = df['Ticker'].head(cap).tolist()

            # Tavan telemetrisi → scanner stats. Tavan bağlaması ANORMAL
            # (15/15 taramada bağlamadı), o yüzden WARNING.
            self._last_rank_info = build_rank_info(df, cap)
            if self._last_rank_info['cut_tickers']:
                logger.warning(
                    "Universe cap BAĞLADI: %d ticker tavanın (%d) altında kaldı — "
                    "en düşük dolar-hacimliler kesildi: %s",
                    len(self._last_rank_info['cut_tickers']), cap,
                    self._last_rank_info['cut_tickers'][:15],
                )

            # Log diagnostics
            top_cols = ['Ticker', 'Price', 'Change', 'Volume', 'dollar_volume']
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

