"""
Bayat veri devre kesicisi testleri (2026-08-04).

CANLI ARIZA: yfinance 03 Ağustos barını TÜM piyasa için OHLC=NaN / hacim dolu
döndürdü (örneklem: 14/15 ticker, SPY ve AAPL dahil). NaN-guard barı doğru
şekilde düşürüyor ama sistem sessizce 31 Temmuz'a (iki seans eski) düşüyordu:
tarama, rejim tespiti (SPY) ve çıkış kontrolü hep bayat barla koşuyordu.

Kullanıcı bunu şöyle fark etti: 3 Ağustos akşamı otomatik tarama LIFE'ı Q90 ile
PENDING'e aldı (signal_price 22.83 — gerçek bir kırılım), ertesi sabah manuel
lookup aynı hisse için "SWING HAZIR DEĞİL" dedi. İkisi de doğruydu: tarama
03 Ağustos barını görmüştü, lookup ise sağlayıcı o barı bozunca 31 Temmuz'a
düşmüştü. Sistem tutarsız DEĞİLDİ ama tutarsız GÖRÜNÜYORDU çünkü hangi bara
bakıldığı hiçbir yerde yazmıyordu.

Mevcut fetch-oranı devre kesicisi bu arızayı YAKALAMAZ: fetch başarılı
(ticker'ların %99'u veri döner), eksik olan yalnız EN SON bar.

Kural: sessizce bayat sinyal üretmek, hiç sinyal üretmemekten kötüdür —
geçmiş bir kırılımı "bugünün fırsatı" sanıp ölçülen t+1 girişini kaçırırız.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from api.routers.scanner import STALE_ABORT_RATIO, _assess_data_staleness
from swing_trader.utils.market_calendar import (
    last_completed_session, prev_trading_day, sessions_behind,
)

ET = ZoneInfo("America/New_York")
# 4 Ağustos 2026 Salı, 06:15 ET (pre-market) — arızanın yaşandığı an
NOW = datetime(2026, 8, 4, 6, 15, tzinfo=ET)


def _df(last_date: date, n: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(last_date), periods=n)
    return pd.DataFrame({
        "Date": dates,
        "Open": [10.0] * n, "High": [10.5] * n, "Low": [9.5] * n,
        "Close": [10.0] * n, "Volume": [1_000_000] * n,
    })


# ── Takvim yardımcıları ──────────────────────────────────────────────────

def test_last_completed_session_before_close():
    """Kapanıştan (16:00 ET) önce bugünün barı henüz tamamlanmamıştır."""
    assert last_completed_session(NOW) == date(2026, 8, 3)


def test_last_completed_session_after_close():
    """Kapanıştan sonra bugünün barı tamamlanmış sayılır."""
    after = datetime(2026, 8, 4, 16, 30, tzinfo=ET)
    assert last_completed_session(after) == date(2026, 8, 4)


def test_last_completed_session_skips_weekend():
    """Pazar günü bakıldığında son seans Cuma olmalı."""
    sunday = datetime(2026, 8, 2, 12, 0, tzinfo=ET)
    assert last_completed_session(sunday) == date(2026, 7, 31)


@pytest.mark.parametrize("bar,expected", [
    (date(2026, 8, 3), 0),    # güncel
    (date(2026, 7, 31), 1),   # bir seans eski (arızanın bıraktığı yer)
    (date(2026, 7, 30), 2),
])
def test_sessions_behind(bar, expected):
    assert sessions_behind(bar, NOW) == expected


def test_prev_trading_day_skips_weekend():
    assert prev_trading_day(date(2026, 8, 3)) == date(2026, 7, 31)


# ── Devre kesici ─────────────────────────────────────────────────────────

def test_fresh_data_does_not_abort():
    data = {f"T{i}": _df(date(2026, 8, 3)) for i in range(20)}
    r = _assess_data_staleness(data, now=NOW)
    assert r["abort"] is False
    assert r["stale_ratio"] == 0.0
    assert r["dominant_sessions_behind"] == 0


def test_market_wide_stale_aborts():
    """Canlı arızanın birebir tekrarı: HERKES 31 Temmuz'da kalmış."""
    data = {f"T{i}": _df(date(2026, 7, 31)) for i in range(20)}
    r = _assess_data_staleness(data, now=NOW)
    assert r["abort"] is True, r
    assert r["stale_ratio"] == 1.0
    assert r["dominant_bar_date"] == "2026-07-31"
    assert r["dominant_sessions_behind"] == 1


def test_few_stale_tickers_do_not_abort():
    """
    Tek tek gecikmeler MEŞRU olabilir (halt, yeni listing, düşük likidite).
    Çoğunluk güncelse tarama durmamalı — aksi halde sistem sürekli kilitlenir.
    """
    data = {f"T{i}": _df(date(2026, 8, 3)) for i in range(18)}
    data["HALT1"] = _df(date(2026, 7, 30))
    data["HALT2"] = _df(date(2026, 7, 31))
    r = _assess_data_staleness(data, now=NOW)
    assert r["abort"] is False
    assert r["stale_count"] == 2
    assert r["dominant_sessions_behind"] == 0


def test_abort_threshold_is_majority():
    """Eşiğin üstü durdurur, altı durdurmaz (sınır davranışı net olsun)."""
    n = 20
    n_stale = int(n * STALE_ABORT_RATIO) + 1     # çoğunluk bayat
    data = {f"S{i}": _df(date(2026, 7, 31)) for i in range(n_stale)}
    data.update({f"F{i}": _df(date(2026, 8, 3)) for i in range(n - n_stale)})
    assert _assess_data_staleness(data, now=NOW)["abort"] is True

    data2 = {f"S{i}": _df(date(2026, 7, 31)) for i in range(n_stale - 2)}
    data2.update({f"F{i}": _df(date(2026, 8, 3)) for i in range(n - n_stale + 2)})
    assert _assess_data_staleness(data2, now=NOW)["abort"] is False


def test_empty_input_is_safe():
    """Veri yoksa bu kesici karar vermez (fetch-oranı kesicisi zaten yakalar)."""
    r = _assess_data_staleness({}, now=NOW)
    assert r["abort"] is False
    assert r["total"] == 0


def test_malformed_frames_are_ignored():
    """Bozuk/boş DataFrame kesiciyi çökertmemeli."""
    data = {
        "OK": _df(date(2026, 8, 3)),
        "EMPTY": pd.DataFrame(),
        "NO_DATE": pd.DataFrame({"Close": [1.0, 2.0]}),
        "NONE": None,
    }
    r = _assess_data_staleness(data, now=NOW)
    assert r["total"] == 1
    assert r["abort"] is False
