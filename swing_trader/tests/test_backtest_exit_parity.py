"""
BACKTEST ÇIKIŞ PARİTESİ (2026-08-05).

`/backtest` sayfasının tek işi "bu strateji geçmişte ne yapardı" sorusunu
cevaplamak. Çıkışı canlıdan farklıysa cevabı YANLIŞ olur — üstelik fark
edilmez, çünkü sayılar makul görünür.

Denetimde üç ayrı kırık bulundu (hepsi smallcap_backtest tarafında):

  1. TRAILING: canlı chandelier kullanıyor (giriş sonrası görülen EN YÜKSEK
     tepe − 3 ATR, +1.5 ATR kârdan sonra devreye girer). Backtest 8
     parametreli kademeli bir merdiven kullanıyordu ve "tepe" olarak yalnız
     O GÜNÜN yükseğini alıyordu — her barda sıfırlanan bambaşka bir kural.
  2. SIRALAMA: canlı stop'u trail güncellemesinden ÖNCE kontrol ediyor
     (tracker.py'de bu bir HATA olarak bulunup düzeltilmiş: "check stop
     BEFORE updating trail (was inverted)"). Backtest ters sırayı koruyordu
     → yeni zirve yapıp düşen barlarda yükseltilmiş trail'den çıkıyor, yani
     sonuçları İYİMSER çıkıyordu.
  3. GÜN SAYIMI: canlı işlem günü (bar) sayıyor, backtest TAKVİM günü
     sayıyordu → 20 takvim günü ≈ 14 işlem günü, timeout erken tetikliyordu.

Bu test üçünü de kaynak seviyesinde ve davranış seviyesinde bağlar.
"""

import inspect
import re

import pandas as pd
import pytest

from swing_trader.paper_trading.tracker import PaperTradeTracker
from swing_trader.small_cap import smallcap_backtest as BT

BT_SRC = inspect.getsource(BT)
TRACKER_SRC = inspect.getsource(
    __import__("swing_trader.paper_trading.tracker", fromlist=["x"])
)


# ── 1. Kaynak seviyesinde: aynı formül, aynı sabitler ─────────────────────

@pytest.mark.parametrize("const,value", [("TRAIL_ATR_MULT", "3.0"),
                                         ("TRAIL_ACTIVATE_ATR", "1.5")])
def test_both_engines_use_same_trail_constants(const, value):
    for name, src in (("backtest", BT_SRC), ("tracker", TRACKER_SRC)):
        assert re.search(rf"{const}\s*=\s*{re.escape(value)}", src), (
            f"{name}'da {const} = {value} bulunamadı — iki motor ayrışmış"
        )


def test_backtest_uses_cumulative_peak_not_todays_high():
    """
    Chandelier'in tanımı "giriş sonrası görülen en yüksek tepe". Yalnız o günün
    yükseğini kullanmak farklı (ve daha gevşek) bir kuraldır.
    """
    assert "peak_high" in BT_SRC
    assert re.search(r"peak_high['\"]?\]?\s*,\s*current_high|max\(\s*float\(trade\.get\(\s*['\"]peak_high",
                     BT_SRC), "kümülatif tepe takibi yok"


def test_backtest_checks_stop_before_updating_trail():
    """Stop kontrolü, trail güncellemesinden ÖNCE gelmeli (aynı-bar sıralaması)."""
    i_stop = BT_SRC.find("if current_low <= active_stop:")
    i_trail = BT_SRC.find("TRAIL_ATR_MULT * atr")
    assert i_stop > 0 and i_trail > 0
    assert i_stop < i_trail, (
        "trail, stop kontrolünden ÖNCE güncelleniyor — tracker'da düzeltilen "
        "hatanın aynısı; sonuçlar iyimser çıkar"
    )


def test_backtest_counts_trading_days_not_calendar_days():
    """Timeout işlem günü saymalı; takvim günü 20 ≈ 14 işlem günüdür."""
    assert "bars_held" in BT_SRC
    assert not re.search(r"days_held\s*=\s*\(current_date\s*-\s*entry_date\)\.days", BT_SRC), (
        "takvim günü sayımı geri gelmiş"
    )


def test_removed_tiered_trail_settings_stay_removed():
    """15 parametreli eski merdiven geri gelmemeli."""
    for f in ("trail_peak_atr_25", "trail_peak_frac_20", "close_trail_atr_15",
              "light_protect_below_entry_atr", "time_stop_min_days"):
        assert f not in BT_SRC, f"{f} geri gelmiş (canlıda karşılığı yok)"
    from swing_trader.small_cap.settings_config import SmallCapSettings
    assert "backtest_exit_trailing" not in SmallCapSettings.model_fields


# ── 2. Davranış seviyesinde: elle kurulmuş barlarda bilinen cevap ─────────

def _bars(rows):
    """rows: (open, high, low, close) listesi; ilk satır giriş barı."""
    return pd.DataFrame({
        "Date": pd.bdate_range("2025-01-02", periods=len(rows)),
        "Open": [r[0] for r in rows], "High": [r[1] for r in rows],
        "Low": [r[2] for r in rows], "Close": [r[3] for r in rows],
    })


def _tracker_exit(bars, entry, stop, target, target_2, atr, max_hold=20):
    tr = PaperTradeTracker.__new__(PaperTradeTracker)
    trade = {
        "entry_price": entry, "stop_loss": stop, "target": target,
        "target_2": target_2, "atr": atr, "max_hold_days": max_hold,
        "trailing_stop": stop, "initial_stop": stop, "partial_exit_price": 0,
        "entry_date": "2025-01-02", "ticker": "X",
    }
    return PaperTradeTracker.check_exit_conditions(tr, trade, bars)


def test_chandelier_trail_locks_at_peak_minus_3atr():
    """
    ATR=1, giriş 100. Bar 1 zirve 104 (+4 ATR ⇒ trail devrede) → trail 101.
    Bar 2 low 100.5 trail'i kırar ⇒ TRAILED, ~101'de çıkış.
    """
    bars = _bars([
        (100, 100, 100, 100),
        (100, 104, 99.5, 103),
        (103, 103, 100.5, 100.8),
    ])
    status, px, _, _, trail = _tracker_exit(bars, 100.0, 97.5, 130.0, 140.0, 1.0)
    assert status == "TRAILED"
    assert px == pytest.approx(101.0, abs=0.05)


def test_trail_does_not_activate_below_1_5_atr_gain():
    """Zirve +1.0 ATR'de kaldıysa trail devreye GİRMEZ; başlangıç stop'u geçerli."""
    bars = _bars([
        (100, 100, 100, 100),
        (100, 101, 99.5, 100.5),
        (100.5, 100.6, 98.0, 98.2),
    ])
    status, px, _, _, trail = _tracker_exit(bars, 100.0, 97.5, 130.0, 140.0, 1.0)
    assert status == "OPEN", "trail erken devreye girmiş olmalı değil"
    assert trail == 97.5, "trail +1.5 ATR altında güncellenmemeli"


def test_stop_checked_before_trail_on_same_bar():
    """
    Aynı barda hem yeni zirve hem stop ihlali varsa: canlı ESKİ stop'tan çıkar
    (stop önce kontrol edilir). Ters sıra yükseltilmiş trail'den çıkarır ve
    sonucu iyimser gösterir — kırık tam buradaydı.
    """
    bars = _bars([
        (100, 100, 100, 100),
        (100, 106, 97.0, 98.0),   # zirve 106 (trail 103 olurdu) AMA low 97 < stop 97.5
    ])
    status, px, _, _, _ = _tracker_exit(bars, 100.0, 97.5, 130.0, 140.0, 1.0)
    assert status == "STOPPED", "yükseltilmiş trail'den çıkılmış (ters sıralama)"
    assert px == pytest.approx(97.5, abs=0.01), "eski stop fiyatından çıkmalı"
