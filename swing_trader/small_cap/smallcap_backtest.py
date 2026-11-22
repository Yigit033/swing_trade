"""
SmallCap Walk-Forward Backtest — canlı ürünün günlük akışını birebir yürütür.

2026-11-22 (Lab v3): eski motor canlıyı ölçmüyordu. Canlıda olmayan bir RSI kapısını
VCE'ye de uyguluyor, ``rank_score`` yerine kalite skoruyla sıralıyor, tipe özgü
kalite tabanları, EMA trend giriş filtresi, Tip C açılış kontrolü, %7 stop tabanı,
swing-low stop, drawdown duraklatması ve "2 kayıptan sonra kalıcı yasak" gibi canlıda
OLMAYAN kurallar işletiyordu; ``scan_stock``'a rejim geçirmiyor, sahte cap/float
(45M) kullanıyor ve çıkışları kendi kopyasıyla hesaplıyordu.

Artık her gün canlı ``run_daily_maintenance`` sırası izlenir ve her adım canlının
KENDİ fonksiyonunu çağırır:

  1. Açılış: dünkü PENDING'ler t+1 Open'da onaylanır
     → ``tracker.gap_percent`` + ``tracker.confirm_entry_levels``
  2. Çıkış: ``paper_trading.execution.evaluate`` (canlı tracker'ın çıkış motoru)
  3. Kapanış: rejim (``regime_from_spy_close``) → ``scan_stock(regime=…)`` →
     ``select_scan_signals`` (rank_score sırası, eşik, top_n) →
     ``auto_entry_capacity`` → duplicate + cooldown (çıkıştan itibaren, REJECTED dahil)

Canlıdan bilinen kalan farklar (UI'da da gösterilir):
  - Evren: bugünkü Finviz listesi geçmişe uygulanır (survivorship — iyimser).
  - Market cap: bugünkü cap × (Close_t / bugünkü Close) yaklaşımı.
  - Maliyet: ``slippage_bps_per_side`` her bacakta düşülür (canlı paper maliyet düşmez).
  - ``max_concurrent`` verilirse canlıda OLMAYAN ek bir sermaye kısıtı uygulanır.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..paper_trading import execution
from ..paper_trading.tracker import (
    MAX_STALE_PENDING_CALENDAR_DAYS,
    confirm_entry_levels,
    gap_percent,
)
from ..utils.market_calendar import is_trading_day
from .engine import SmallCapEngine
from .regime_logic import regime_from_spy_close, regime_unknown
from .risk import SmallCapRisk
from .thresholds import auto_entry_capacity, effective_scan_thresholds, select_scan_signals

logger = logging.getLogger(__name__)

# Canlı tarama 6 aylık pencereyle çalışır (api/routers/scanner.py: period="6mo").
_SCAN_WINDOW = pd.DateOffset(months=6)


class SmallCapBacktester:
    """Walk-forward backtest; canlı auto-scan + paper tracker akışının geçmişe uygulanışı."""

    def __init__(self, config: Dict = None):
        self.engine = SmallCapEngine(config)
        self.settings = self.engine.settings
        self.risk = SmallCapRisk(config, self.settings)
        self.config = config or {}
        self.initial_capital = 10000.0
        self.capital = self.initial_capital
        self.min_quality = self.settings.auto_scan.min_quality
        self.top_n = self.settings.auto_scan.top_n
        self._max_concurrent: Optional[int] = None
        self._reset_state()

    # ── durum ────────────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self.trades: List[Dict] = []          # kapanmış işlemler (UI)
        self.open_trades: List[Dict] = []     # aktif işlemler (önceden hesaplanmış sonuçla)
        self.pending_signals: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self._last_exit: Dict[str, pd.Timestamp] = {}   # ticker → son çıkış günü (cooldown)
        self._spy_df: Optional[pd.DataFrame] = None
        self._vix_df: Optional[pd.DataFrame] = None
        self._data: Dict[str, pd.DataFrame] = {}
        self._earnings: Dict[str, Optional[list]] = {}
        self._meta: Dict[str, Dict] = {}
        self._end: Optional[pd.Timestamp] = None
        self._diag = {
            "raw_signals": 0,
            "selected_signals": 0,
            "pending_queued": 0,
            "skip_capacity": 0,
            "skip_duplicate": 0,
            "skip_cooldown": 0,
            "skip_max_concurrent": 0,
            "skip_position_size": 0,
            "entry_skip_gap_up": 0,
            "entry_skip_gap_down": 0,
            "entry_skip_no_bar": 0,
            "entries_opened": 0,
            "earnings_unknown_tickers": 0,
            "skip_earnings_unknown": 0,
            "cap_unknown_tickers": 0,
        }

    # ── giriş noktası ────────────────────────────────────────────────────────

    def run_backtest(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
        max_concurrent: Optional[int] = None,
        min_quality: Optional[int] = None,
        top_n: Optional[int] = None,
        progress_callback=None,
        stock_meta: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        """
        Args:
            min_quality / top_n: None → canlı auto-scan ayarı (``settings.auto_scan``).
            max_concurrent: None → canlıdaki gibi sınırsız (yalnız rejim kapasitesi).
            stock_meta: ticker → Finviz metadata (marketCap, sector). Yoksa hisse sayısı
                yfinance'tan alınır.
        """
        self._reset_state()
        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)
        self.min_quality = int(min_quality if min_quality is not None else self.settings.auto_scan.min_quality)
        self.top_n = int(top_n if top_n is not None else self.settings.auto_scan.top_n)
        self._max_concurrent = max_concurrent

        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        self._end = end_dt
        warmup_start = (start_dt - pd.DateOffset(months=7)).strftime('%Y-%m-%d')
        fetch_end = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')

        if progress_callback:
            progress_callback(0, f"Downloading data for {len(tickers)} stocks...")
        self._load_data(list(tickers), warmup_start, fetch_end, stock_meta or {})
        if not self._data:
            return self._empty_results()
        # Canlı rejim SPY'nin 1 yıllık geçmişiyle (period="1y", MA200 için) hesaplanır.
        # Benchmark'ı yalnız fiyat ısınmasıyla (7 ay) indirmek dönemin başında MA200'ü
        # eksik bırakıp canlıdan FARKLI rejim üretiyordu (eski motorda 90 gündü).
        bench_start = (start_dt - timedelta(days=400)).strftime('%Y-%m-%d')
        self._spy_df, self._vix_df = self._fetch_benchmarks(bench_start, fetch_end)

        if progress_callback:
            progress_callback(30, f"Data ready: {len(self._data)} stocks. Starting simulation...")

        days = [d for d in pd.bdate_range(start_dt, end_dt) if is_trading_day(d.date())]
        for i, day in enumerate(days):
            # Canlı günlük bakım sırası: onay → çıkış → tarama.
            self._confirm_pending(day)
            self._settle_exits(day)
            regime, eff_min, eff_top = self._daily_regime_bundle(day)
            self.equity_curve.append({
                'date': day.strftime('%Y-%m-%d'),
                'portfolio_value': round(self._portfolio_value(day), 2),
                'open_trades': len(self.open_trades),
                'market_regime': regime.get('regime', 'UNKNOWN'),
                'regime_confidence': regime.get('confidence', ''),
                'effective_min_quality': eff_min,
                'effective_top_n': eff_top,
                'request_min_quality': self.min_quality,
                'request_top_n': self.top_n,
            })
            self._scan_and_track(day, regime, eff_min, eff_top)
            if progress_callback and i % 5 == 0:
                progress_callback(30 + int(i / max(1, len(days)) * 65),
                                  f"Day {i+1}/{len(days)} | Trades: {len(self.trades)} | "
                                  f"Open: {len(self.open_trades)}")

        self._force_close_all(end_dt)
        if progress_callback:
            progress_callback(100, f"Done! {len(self.trades)} total trades")

        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'metrics': self._calculate_metrics(),
            'data_stocks': len(self._data),
            'period': f"{start_date} to {end_date}",
            'diagnostics': dict(self._diag),
            'params': self._params(),
        }

    def _params(self) -> Dict:
        return {
            'min_quality': self.min_quality,
            'top_n': self.top_n,
            'max_concurrent': self._max_concurrent,
            'slippage_bps_per_side': self.settings.slippage_bps_per_side,
            'partial_at_t1_fraction': self.settings.partial_at_t1_fraction,
            'cooldown_days': self.settings.cooldown_days,
            'live_parity': self._max_concurrent is None,
        }

    # ── veri ─────────────────────────────────────────────────────────────────

    def _load_data(self, tickers: List[str], start: str, end: str, stock_meta: Dict[str, Dict]) -> None:
        """Canlı fetcher (aynı auto_adjust + OHLC onarımı) + bilanço tarihleri + cap kaynağı."""
        from ..data.fetcher import DataFetcher

        fetcher = DataFetcher()
        data = fetcher.fetch_multiple_stocks_batch(tickers, start_date=start, end_date=end, min_rows=60)
        self._data = {t: df.sort_values('Date').reset_index(drop=True) for t, df in data.items()}

        def extras(t):
            import yfinance as yf
            tk = yf.Ticker(t)
            earn, shares = None, None
            # Boş liste "bilanço yok" DEĞİL, bilinmiyor (Yahoo kısıtlaması boş döndürür) → None.
            for _ in range(2):
                try:
                    ed = tk.get_earnings_dates(limit=24)
                    if ed is not None and len(ed):
                        earn = list(ed.index)
                        break
                except Exception:
                    pass
            meta = stock_meta.get(t) or {}
            if not meta.get('marketCap'):
                try:
                    shares = float(tk.fast_info['shares'] or 0) or None
                except Exception:
                    shares = None
            return t, earn, shares

        with ThreadPoolExecutor(max_workers=6) as ex:
            for t, earn, shares in ex.map(extras, list(self._data)):
                self._earnings[t] = earn
                if earn is None:
                    self._diag["earnings_unknown_tickers"] += 1
                df = self._data[t]
                meta = dict(stock_meta.get(t) or {})
                meta['ref_close'] = float(df['Close'].iloc[-1])
                if shares:
                    meta['shares'] = shares
                if not meta.get('marketCap') and not meta.get('shares'):
                    self._diag["cap_unknown_tickers"] += 1
                self._meta[t] = meta

    def _stock_info(self, ticker: str, close_t: float) -> Dict:
        """Canlı Finviz metadata şekli. Float canlıda bilinmiyor (Overview'da sütun yok) → 0."""
        meta = self._meta.get(ticker) or {}
        cap = 0.0
        if meta.get('marketCap') and meta.get('ref_close'):
            cap = float(meta['marketCap']) * close_t / float(meta['ref_close'])
        elif meta.get('shares'):
            cap = float(meta['shares']) * close_t
        return {'ticker': ticker, 'marketCap': int(cap), 'floatShares': 0,
                'shortName': ticker, 'sector': meta.get('sector') or 'Unknown'}

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
        """SPY + VIX — canlı rejimle aynı kurallar için."""
        import yfinance as yf
        spy_df, vix_df = pd.DataFrame(), pd.DataFrame()
        try:
            spy_df = self._normalize_benchmark_df(yf.Ticker("SPY").history(start=start_s, end=end_s))
        except Exception as e:
            logger.warning("Backtest: SPY benchmark fetch failed: %s", e)
        try:
            vix_df = self._normalize_benchmark_df(yf.Ticker("^VIX").history(start=start_s, end=end_s))
        except Exception as e:
            logger.debug("Backtest: VIX fetch failed: %s", e)
        return spy_df, vix_df

    def _daily_regime_bundle(self, day: pd.Timestamp):
        """Günün kapanışı itibarıyla rejim + etkin eşikler (canlı scanner ile aynı fonksiyonlar)."""
        caps = self.settings.regime_thresholds
        m = self._spy_df[self._spy_df["Date"] <= day] if self._spy_df is not None and len(self._spy_df) else None
        if m is None or len(m) < 50:
            r = regime_unknown("no_spy_benchmark" if m is None else "insufficient_spy_history")
        else:
            vx = None
            if self._vix_df is not None and len(self._vix_df):
                vm = self._vix_df[self._vix_df["Date"] <= day]
                if len(vm):
                    vx = float(vm["Close"].iloc[-1])
            # Canlı: yfinance period="1y" ≈ son 252 seans.
            r = regime_from_spy_close(m.tail(252)["Close"].reset_index(drop=True), vx)
        eff_min, eff_top = effective_scan_thresholds(
            r.get("regime", "UNKNOWN"), r.get("confidence", "TENTATIVE"),
            self.min_quality, self.top_n, regime_caps=caps,
        )
        return r, eff_min, eff_top

    # ── 3. kapanış: tarama + otomatik takip ─────────────────────────────────

    def _scan_and_track(self, day: pd.Timestamp, regime: Dict, eff_min: int, eff_top: int) -> None:
        regime_name = str(regime.get("regime", "UNKNOWN")).upper()
        windows: Dict[str, pd.DataFrame] = {}
        signals: List[Dict] = []
        lo = day - _SCAN_WINDOW
        for ticker in sorted(self._data):
            df = self._data[ticker]
            w = df[(df['Date'] > lo) & (df['Date'] <= day)]
            if len(w) < 20 or w['Date'].iloc[-1] != day:
                continue          # canlı yalnız o seansın tamamlanmış barıyla tarar
            w = w.reset_index(drop=True)
            earn = self._earnings.get(ticker)
            try:
                sig = self.engine.scan_stock(
                    ticker, w,
                    stock_info=self._stock_info(ticker, float(w['Close'].iloc[-1])),
                    backtest_mode=True,                     # yalnız anlatı/LLM'i atlar
                    portfolio_value=self.initial_capital,   # canlı: sabit auto_scan.portfolio_value
                    regime=regime_name,
                    earnings_dates=earn if earn is not None else [],
                )
            except Exception as e:
                logger.debug("Scan error %s: %s", ticker, e)
                continue
            if sig and earn is None:
                # Canlı: bilanço tarihi iki kaynaktan da doğrulanamazsa sinyal reddedilir
                # (fail-closed, Lab v3 Ek D1). Backtest'te ikinci kaynak (takvim) yok.
                self._diag["skip_earnings_unknown"] += 1
                continue
            if sig:
                sig["market_regime"] = regime_name
                sig["regime_confidence"] = regime.get("confidence", "")
                signals.append(sig)
                windows[ticker] = w
        self._diag["raw_signals"] += len(signals)

        selected = select_scan_signals(signals, eff_min, eff_top)
        self._diag["selected_signals"] += len(selected)
        active = len(self.open_trades) + len(self.pending_signals)
        capacity = auto_entry_capacity(regime_name, self.settings.backtest_loop, active)
        if capacity is not None and len(selected) > capacity:
            self._diag["skip_capacity"] += len(selected) - capacity
            selected = selected[:capacity]

        cooldown = int(self.settings.cooldown_days)
        busy = {t['ticker'] for t in self.open_trades} | {p['signal']['ticker'] for p in self.pending_signals}
        for sig in selected:
            t = sig['ticker']
            if self._max_concurrent is not None and \
                    len(self.open_trades) + len(self.pending_signals) >= self._max_concurrent:
                self._diag["skip_max_concurrent"] += 1
                continue
            if t in busy:                                     # storage.check_duplicate
                self._diag["skip_duplicate"] += 1
                continue
            last = self._last_exit.get(t)                     # storage.check_cooldown_active
            if cooldown > 0 and last is not None and last >= day - pd.Timedelta(days=cooldown):
                self._diag["skip_cooldown"] += 1
                continue
            w = windows[t]
            self.pending_signals.append({
                "signal": sig,
                "signal_date": day,
                "signal_close": float(w['Close'].iloc[-1]),
                "atr": float(self.risk.calculate_atr(w)),     # tracker: ATR(14) eklenme anında
            })
            busy.add(t)
            self._diag["pending_queued"] += 1

    # ── 1. açılış: PENDING → OPEN ───────────────────────────────────────────

    def _confirm_pending(self, day: pd.Timestamp) -> None:
        keep = []
        for p in self.pending_signals:
            sig, t = p['signal'], p['signal']['ticker']
            df = self._data[t]
            after = df[df['Date'] > p['signal_date']]
            if after.empty or after['Date'].iloc[0] > day:
                # Bar yok (halt / veri boşluğu): canlı bekler, 14 takvim günü sonra reddeder.
                if (day - p['signal_date']).days > MAX_STALE_PENDING_CALENDAR_DAYS:
                    self._diag["entry_skip_no_bar"] += 1
                    self._last_exit[t] = day
                else:
                    keep.append(p)
                continue
            bar = after.iloc[0]
            open_price = float(bar['Open'])
            gap = gap_percent(open_price, p['signal_close'])
            if gap > self.settings.max_gap_up_pct or gap < -self.settings.max_gap_down_pct:
                self._diag["entry_skip_gap_up" if gap > 0 else "entry_skip_gap_down"] += 1
                self._last_exit[t] = day                  # canlı REJECTED exit_date = onay günü
                continue
            size = int(sig.get('position_size') or 0)
            if size < 1:
                self._diag["skip_position_size"] += 1
                self._last_exit[t] = day
                continue
            stop, t1, t2 = confirm_entry_levels(
                open_price, p['signal_close'], p['atr'], sig.get('swing_type', 'A'),
                float(sig['stop_loss']), float(sig['target_1']),
                float(sig.get('target_2') or sig['target_1']),
            )
            self._open_trade(sig, day, open_price, size, stop, t1, t2, p['atr'])
        self.pending_signals = keep

    def _open_trade(self, sig, day, open_price, size, stop, t1, t2, atr) -> None:
        """Canlı tracker'ın OPEN kaydını kurar ve sonucunu canlı çıkış motoruyla hesaplar."""
        entry = round(open_price, 2)
        live = dict(
            id=len(self.trades) + len(self.open_trades) + 1, ticker=sig['ticker'], status='OPEN',
            entry_date=execution.event_time(day.date(), at_open=True), entry_price=entry,
            position_size=size, stop_loss=stop, initial_stop=stop, trailing_stop=stop,
            target=t1, target_2=t2, max_hold_days=int(sig.get('hold_days_max') or 7),
            atr=round(atr, 4) if atr else 0, initial_risk=round((entry - stop) * size, 8),
            execution_state=None, partial_exit_price=None, policy_effective_at=None,
        )
        df = self._data[sig['ticker']]
        hist = df[(df['Date'] >= day) & (df['Date'] <= self._end)][['Date', 'Open', 'High', 'Low', 'Close']]
        hist = hist.head(live['max_hold_days'] + 5).reset_index(drop=True)
        updates, (status, exit_price, exit_date, reason, _stop) = execution.evaluate(
            live, hist, self.settings.partial_at_t1_fraction)
        if str(reason).startswith('REVIEW_REQUIRED'):
            # Canlıda pozisyon insan müdahalesine kadar donar; burada test sonuna kadar açık kalır.
            self._diag["review_required"] = self._diag.get("review_required", 0) + 1
        state = json.loads(updates['execution_state']) if updates.get('execution_state') else {}
        exit_day = None
        if status not in ('OPEN',) and exit_date:
            exit_day = pd.Timestamp(pd.Timestamp(exit_date).tz_convert("America/New_York").date())
        partial_day = None
        if state.get('partial') and state.get('partial_date'):
            partial_day = pd.Timestamp(pd.Timestamp(state['partial_date']).tz_convert("America/New_York").date())
        self.open_trades.append(dict(
            ticker=sig['ticker'], swing_type=sig.get('swing_type'), pathway=sig.get('trigger_pathway'),
            quality_score=sig.get('quality_score', 0), rank_score=sig.get('rank_score', 0),
            entry_day=day, entry_price=entry, size=size, stop=stop, target=t1, target_2=t2,
            max_hold_days=live['max_hold_days'], initial_risk=live['initial_risk'],
            status=status, reason=reason, exit_day=exit_day, exit_price=float(exit_price or 0),
            bars=int(state.get('bars') or 0), sold=int(state.get('sold') or 0) if state.get('partial') else 0,
            partial_price=float(state.get('partial_price') or 0), partial_day=partial_day,
            partial_realized=False,
        ))
        self._diag["entries_opened"] += 1

    # ── 2. çıkışlar ─────────────────────────────────────────────────────────

    def _slip(self) -> float:
        return self.settings.slippage_bps_per_side / 10000.0

    def _entry_fill_price(self, quoted: float) -> float:
        return round(quoted * (1.0 + self._slip()), 2)

    def _exit_fill_price(self, quoted: float) -> float:
        return round(quoted * (1.0 - self._slip()), 2) if quoted > 0 else quoted

    def _settle_exits(self, day: pd.Timestamp) -> None:
        still = []
        for tr in self.open_trades:
            if tr['sold'] and not tr['partial_realized'] and tr['partial_day'] is not None and tr['partial_day'] <= day:
                self.capital += tr['sold'] * (self._exit_fill_price(tr['partial_price'])
                                              - self._entry_fill_price(tr['entry_price']))
                tr['partial_realized'] = True
            if tr['exit_day'] is not None and tr['exit_day'] <= day:
                self._close_trade(tr, tr['exit_price'], tr['exit_day'], tr['status'], tr['reason'], tr['bars'])
            else:
                still.append(tr)
        self.open_trades = still

    def _close_trade(self, tr: Dict, raw_exit: float, exit_day: pd.Timestamp, status: str,
                     reason: str, bars: int) -> None:
        """Net P/L (her bacakta slippage). Kısmi bacak T1 gününde zaten sermayeye eklendi."""
        entry_fill = self._entry_fill_price(tr['entry_price'])
        size, sold = tr['size'], tr['sold']
        rest = size - sold
        exit_fill = self._exit_fill_price(raw_exit)
        partial_pnl = sold * (self._exit_fill_price(tr['partial_price']) - entry_fill) if sold else 0.0
        rest_pnl = rest * (exit_fill - entry_fill)
        if sold and not tr['partial_realized']:
            self.capital += partial_pnl
        self.capital += rest_pnl
        total = partial_pnl + rest_pnl
        blended = ((self._exit_fill_price(tr['partial_price']) * sold + exit_fill * rest) / size) if size else exit_fill
        self._last_exit[tr['ticker']] = exit_day
        self.trades.append({
            'ticker': tr['ticker'], 'swing_type': tr['swing_type'], 'pathway': tr['pathway'],
            'quality_score': tr['quality_score'], 'rank_score': tr['rank_score'],
            'entry_date': tr['entry_day'].strftime('%Y-%m-%d'), 'entry_price': tr['entry_price'],
            'exit_date': exit_day.strftime('%Y-%m-%d'), 'exit_price': round(blended, 2),
            'stop_loss': tr['stop'], 'target': tr['target'], 'target_2': tr['target_2'],
            'shares': size, 'initial_shares': size, 'partial_shares': sold,
            'max_hold_days': tr['max_hold_days'], 'days_held': bars,
            'status': status, 'exit_reason': reason,
            'pnl_dollar': round(total, 2),
            'pnl_pct': round(total / (entry_fill * size) * 100, 2) if size and entry_fill else 0.0,
            'r_multiple': round(total / tr['initial_risk'], 3) if tr['initial_risk'] > 0 else None,
        })

    def _force_close_all(self, end_dt: pd.Timestamp) -> None:
        """Test sonu: açık kalanlar son tamamlanmış kapanıştan kapatılır (FORCED)."""
        for tr in self.open_trades:
            df = self._data[tr['ticker']]
            df = df[df['Date'] <= end_dt]
            last = float(df['Close'].iloc[-1]) if len(df) else tr['entry_price']
            bars = int((df['Date'] >= tr['entry_day']).sum()) if len(df) else 0
            if tr['partial_day'] is None or tr['partial_day'] > end_dt:
                tr['sold'] = 0                         # kısmi henüz gerçekleşmedi
            self._close_trade(tr, last, end_dt, 'FORCED',
                              'Backtest End (last available completed close)', bars)
        self.open_trades = []

    def _portfolio_value(self, day: pd.Timestamp) -> float:
        """Realize sermaye + açık pozisyonların gün sonu değerlemesi."""
        value = self.capital
        for tr in self.open_trades:
            df = self._data[tr['ticker']]
            sub = df[df['Date'] <= day]
            if not len(sub):
                continue
            rest = tr['size'] - (tr['sold'] if tr['partial_realized'] else 0)
            value += rest * (float(sub['Close'].iloc[-1]) - self._entry_fill_price(tr['entry_price']))
        return value

    # ── metrikler ───────────────────────────────────────────────────────────

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics from completed trades."""
        if not self.trades:
            return self._empty_metrics()

        wins = [t for t in self.trades if t['pnl_pct'] > 0]
        losses = [t for t in self.trades if t['pnl_pct'] <= 0]
        total = len(self.trades)
        win_rate = len(wins) / total if total else 0
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        avg_win_dollar = np.mean([t['pnl_dollar'] for t in wins]) if wins else 0
        avg_loss_dollar = np.mean([t['pnl_dollar'] for t in losses]) if losses else 0
        total_win_dollar = sum(t['pnl_dollar'] for t in wins)
        total_loss_dollar = abs(sum(t['pnl_dollar'] for t in losses))
        profit_factor = total_win_dollar / total_loss_dollar if total_loss_dollar > 0 else float('inf')
        total_pnl_dollar = sum(t['pnl_dollar'] for t in self.trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital

        max_dd = 0
        if self.equity_curve:
            peak = self.equity_curve[0]['portfolio_value']
            for point in self.equity_curve:
                val = point['portfolio_value']
                peak = max(peak, val)
                if peak > 0:
                    max_dd = max(max_dd, (peak - val) / peak * 100)

        type_stats: Dict[str, Dict] = {}
        for t in self.trades:
            st = type_stats.setdefault(t.get('swing_type') or 'B', {'wins': 0, 'losses': 0, 'total_pnl': 0})
            st['wins' if t['pnl_pct'] > 0 else 'losses'] += 1
            st['total_pnl'] += t['pnl_pct']

        exit_stats: Dict[str, Dict] = {}
        for t in self.trades:
            es = exit_stats.setdefault(t.get('status', 'UNKNOWN'), {'count': 0, 'avg_pnl': []})
            es['count'] += 1
            es['avg_pnl'].append(t['pnl_pct'])
        for es in exit_stats.values():
            es['avg_pnl'] = round(float(np.mean(es['avg_pnl'])), 2) if es['avg_pnl'] else 0

        return {
            'total_trades': total,
            'win_rate': round(win_rate, 4),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'avg_win_pct': round(float(avg_win), 2),
            'avg_loss_pct': round(float(avg_loss), 2),
            'avg_win_dollar': round(float(avg_win_dollar), 2),
            'avg_loss_dollar': round(float(avg_loss_dollar), 2),
            'profit_factor': round(profit_factor, 2),
            'total_pnl_dollar': round(total_pnl_dollar, 2),
            'total_return': round(total_return, 4),
            'max_drawdown': round(max_dd, 2),
            'avg_hold_days': round(float(np.mean([t.get('days_held', 0) for t in self.trades])), 1),
            'initial_capital': self.initial_capital,
            'final_capital': round(self.capital, 2),
            'type_stats': type_stats,
            'exit_stats': exit_stats,
        }

    def _empty_results(self):
        return {
            'trades': [],
            'equity_curve': [],
            'metrics': self._empty_metrics(),
            'data_stocks': 0,
            'period': '',
            'diagnostics': dict(self._diag),
            'params': self._params(),
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
