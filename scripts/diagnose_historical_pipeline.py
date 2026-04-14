"""
Tarihsel point-in-time pipeline tanısı — son N iş günü için her gün scan_stock + elenme nedeni.

Kullanım (repo kökünden):
  python scripts/diagnose_historical_pipeline.py

Yahoo rate limit (429) alırsanız bir süre bekleyin veya yerel CSV kullanın:
  SWING_DIAGNOSE_DATA_DIR=C:\\path\\to\\folder
  Klasörde SGML.csv, ANRO.csv, … (sütunlar: Date,Open,High,Low,Close,Volume) olmalı.

Motor dosyalarına dokunmaz; trace_gate_failure scan_stock None iken ilk kapıyı Türkçe anahtarlarla raporlar.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("swing_trader.data.fetcher").setLevel(logging.CRITICAL)

from swing_trader.data.fetcher import DataFetcher
from swing_trader.small_cap.engine import SmallCapEngine
from swing_trader.small_cap.regime_logic import rs_bonus_vs_spy

# --- Girdi ---
TICKERS = ["SGML", "ANRO", "CAPR", "GNB"]
TRADING_DAYS = 60
MIN_QUALITY = 65  # API varsayılanı ile hizalı
FETCH_PERIOD = "2y"
BACKTEST_MODE = True
FETCH_SLEEP_SEC = 4.0
FETCH_MAX_RETRIES = 4
FETCH_RETRY_BASE_SEC = 25


def load_csv_if_configured(ticker: str) -> Optional[pd.DataFrame]:
    """SWING_DIAGNOSE_DATA_DIR/{TICKER}.csv varsa oku (Yahoo limitine alternatif)."""
    base = os.environ.get("SWING_DIAGNOSE_DATA_DIR", "").strip()
    if not base:
        return None
    path = Path(base) / f"{ticker.upper()}.csv"
    if not path.is_file():
        print(f"  [{ticker}] CSV yok: {path}")
        return None
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        print(f"  [{ticker}] CSV'de Date sütunu yok: {path}")
        return None
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].dropna(subset=["Close"])
    print(f"  [{ticker}] CSV yüklendi: {len(df)} satır ({path})")
    return df


def fetch_ohlcv_relaxed(fetcher: DataFetcher, ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Yahoo rate limit için aralıklı yeniden dene (sadece bu script)."""
    last = None
    for attempt in range(FETCH_MAX_RETRIES):
        df = fetcher.fetch_stock_data(ticker, period=period)
        if df is not None and len(df) >= 21:
            return df
        last = df
        if attempt < FETCH_MAX_RETRIES - 1:
            wait = FETCH_RETRY_BASE_SEC * (attempt + 1)
            print(
                f"  [{ticker}] veri alınamadı, {wait}s sonra tekrar (deneme {attempt + 2}/{FETCH_MAX_RETRIES})…",
                flush=True,
            )
            time.sleep(wait)
    return last


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" not in out.columns:
        out = out.reset_index()
        if "Date" not in out.columns and "index" in out.columns:
            out = out.rename(columns={"index": "Date"})
    return out


def trace_gate_failure(engine: SmallCapEngine, ticker: str, df: pd.DataFrame) -> Optional[str]:
    """
    scan_stock'un skor öncesi tüm hard gate'leri ile aynı sıra (engine.scan_stock ile uyumlu).
    İlk elenen kapıyı Türkçe kod olarak döner; hiçbiri elenmezse None.
    """
    if df is None or len(df) < 20:
        return "yetersiz_veri"

    stock_info = {
        "ticker": ticker,
        "marketCap": int(engine.filters.MIN_MARKET_CAP * 1.2),
        "floatShares": 45_000_000,
        "shortName": ticker,
        "sector": "Unknown",
    }

    try:
        if "Date" in df.columns:
            signal_date = df["Date"].iloc[-1]
        else:
            signal_date = df.index[-1]
        if isinstance(signal_date, pd.Timestamp):
            signal_date_dt = signal_date.to_pydatetime()
            if signal_date_dt.tzinfo is not None:
                signal_date_dt = signal_date_dt.replace(tzinfo=None)
        else:
            signal_date_str = str(signal_date)[:10]
            signal_date_dt = datetime.strptime(signal_date_str, "%Y-%m-%d")
    except Exception:
        signal_date_dt = datetime.now()

    filter_passed, filter_results = engine.filters.apply_all_filters(
        ticker, df, stock_info, signal_date_dt, backtest_mode=BACKTEST_MODE
    )
    if not filter_passed:
        return "evren_filtresi"

    triggered, trigger_details = engine.signals.check_all_triggers(df)
    if not triggered:
        return "no_trigger"

    boosters = engine.signals.check_boosters(df)
    swing_ready = boosters.get("swing_ready", False)
    swing_details = boosters.get("swing_details", {})
    if not swing_ready:
        return "swing_not_ready"

    volume_surge = trigger_details.get("volume_surge", 2.0)

    if BACKTEST_MODE:
        spy_df_window = None
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

    rsi_div = engine.signals.detect_rsi_divergence(df, lookback=14)
    boosters["rsi_divergence"] = rsi_div["divergence_found"]
    boosters["rsi_divergence_confidence"] = rsi_div.get("confidence", 0)

    macd_data = engine.signals.calculate_macd(df)
    boosters["macd_bullish"] = macd_data["bullish_cross"] or (
        macd_data["above_zero"] and macd_data["expanding"]
    )

    five_day_return = swing_details.get("five_day_momentum", {}).get("return", 0)
    ma20_distance = swing_details.get("above_ma20", {}).get("distance", 0)
    rsi = boosters.get("rsi", 50)
    overext = swing_details.get("overextension", {})
    higher_lows = boosters.get("higher_lows", False)

    today_high = float(df["High"].iloc[-1])
    today_low = float(df["Low"].iloc[-1])
    today_close = float(df["Close"].iloc[-1])
    day_range = today_high - today_low
    close_position = (today_close - today_low) / day_range if day_range > 0 else 0.5

    has_any_catalyst = False
    swing_type, hold_days, type_reason = engine._classify_swing_type(
        five_day_return,
        rsi,
        volume_surge,
        higher_lows,
        close_position=close_position,
        ma20_distance=ma20_distance,
        short_interest=boosters.get("short_percent", 0),
        days_to_cover=boosters.get("days_to_cover", 0),
        has_catalyst=has_any_catalyst,
        rsi_divergence=boosters.get("rsi_divergence", False),
        macd_bullish=boosters.get("macd_bullish", False),
    )

    max_rsi = engine.settings.max_entry_rsi
    if rsi > max_rsi and swing_type != "S":
        return "rsi_gate"

    overext_details = overext.get("details", {})
    five_day_total = overext_details.get("five_day_total", five_day_return)
    sg = engine.settings.scan_gates
    if (
        five_day_total > sg.late_entry_five_day_total_gt
        and rsi > sg.late_entry_rsi_gt
        and swing_type != "S"
    ):
        return "late_entry"

    if boosters.get("obv_distribution", False) and swing_type != "S":
        return "obv_gate"

    trend_data = swing_details.get("trend_quality", {})
    trend_phase = trend_data.get("trend_phase", "unknown")
    if trend_phase in ("distribution", "markdown") and swing_type != "S":
        trend_strength = trend_data.get("trend_strength", 50)
        if trend_strength < 30:
            return "trend_zayif"

    return None


def classify_day(engine: SmallCapEngine, ticker: str, df_pit: pd.DataFrame) -> str:
    """Point-in-time df ile günü sınıflandır."""
    if df_pit is None or len(df_pit) < 20:
        return "yetersiz_veri"

    try:
        signal = engine.scan_stock(
            ticker,
            df_pit,
            backtest_mode=BACKTEST_MODE,
            portfolio_value=10000,
            spy_df_window=None,
        )
    except Exception:
        return "scan_hatasi"

    if signal is not None:
        qs = float(signal.get("quality_score", 0))
        if qs < MIN_QUALITY:
            return "quality_score_low"
        return "sinyal_uretildi"

    gate = trace_gate_failure(engine, ticker, df_pit)
    if gate is not None:
        return gate
    return "scan_hatasi"


def run() -> None:
    fetcher = DataFetcher()
    engine = SmallCapEngine()

    print("=" * 72, flush=True)
    print(" diagnose_historical_pipeline.py", flush=True)
    print(f" Tickers: {', '.join(TICKERS)}", flush=True)
    print(f" Son {TRADING_DAYS} iş günü (yeterli geçmiş yoksa daha az)", flush=True)
    print(f" backtest_mode={BACKTEST_MODE}, min_quality={MIN_QUALITY}", flush=True)
    print("=" * 72, flush=True)

    for ti, ticker in enumerate(TICKERS):
        if ti > 0:
            time.sleep(FETCH_SLEEP_SEC)
        print(f"\n[{ticker}] veri yükleniyor…", flush=True)
        raw = load_csv_if_configured(ticker)
        if raw is None:
            raw = fetch_ohlcv_relaxed(fetcher, ticker, FETCH_PERIOD)
        if raw is None or len(raw) < 21:
            print(
                f"\n{ticker}: veri yok veya çok kısa ({0 if raw is None else len(raw)} bar)",
                flush=True,
            )
            continue

        df = _ensure_date_column(raw)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            df = df.sort_values("Date").reset_index(drop=True)

        n = len(df)
        start_end = max(19, n - TRADING_DAYS)
        reasons: list[str] = []

        n_days = n - start_end
        for j, end in enumerate(range(start_end, n)):
            df_pit = df.iloc[: end + 1].copy()
            reasons.append(classify_day(engine, ticker, df_pit))
            if (j + 1) % 15 == 0 or (j + 1) == n_days:
                print(f"  [{ticker}] işlendi {j + 1}/{n_days} gün…", flush=True)

        ctr = Counter(reasons)
        total = len(reasons)
        print(f"\n{ticker}: {total} gün", flush=True)
        for key in sorted(ctr.keys(), key=lambda k: (-ctr[k], k)):
            c = ctr[key]
            pct = 100.0 * c / total if total else 0
            print(f"  - {c:3d} gün: {key} ({pct:.0f}%)", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("Not: trend_zayif = distribution/markdown + düşük trend gücü (motor trend_phase_weak).", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    run()
