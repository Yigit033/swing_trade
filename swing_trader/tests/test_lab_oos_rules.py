"""Lab v3 Ek R (dokunulmamış dönem sınavı) veri kuralları: bölünme düzeltmesi ve bilanço takvimi kuralı.

Yahoo fiyat/hacmi bugüne kadarki bölünmelere göre düzeltilmiş, hisse sayısı serisi ise o günün ham sayısı.
Düzeltme olmadan piyasa değeri sonradan bölünen hissede oran kadar küçülür (Tesla 2016: ~30 milyar $ →
~2 milyar $, evrene sızar). Testler sentetik bir veri kökünde koşar; gerçek lab verisine dokunmaz.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lab.common as c  # noqa: E402

SPLIT_DAY = pd.Timestamp("2020-08-31")


def _clear():
    for name in dir(c):
        fn = getattr(c, name)
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()


@pytest.fixture
def lab_root(tmp_path, monkeypatch):
    """AAA: 2020-08-31'de 4'e bölünen hisse. Düzeltilmiş fiyat hep 25 $ (o gün ekranda: önce 100, sonra 25),
    düzeltilmiş hacim hep 4M (o gün gerçekte: önce 1M, sonra 4M), ham hisse sayısı 10M → 40M. BBB bölünmedi."""
    _clear()
    for name, sub in (("LAB", ""), ("OHLCV", "ohlcv"), ("FACTS", "facts"), ("EARN", "earnings"), ("BASE", "base")):
        monkeypatch.setattr(c, name, tmp_path / sub if sub else tmp_path)
    days = pd.bdate_range("2020-03-02", "2020-12-31")
    rows = []
    for t, px, vol in (("AAA", 25.0, 4e6), ("BBB", 50.0, 1e6), ("SPY", 300.0, 5e7)):
        rows.append(pd.DataFrame(dict(ticker=t, Date=days, Open=px, High=px * 1.01, Low=px * 0.99, Close=px,
                                      Volume=vol)))
    (tmp_path / "ohlcv").mkdir()
    pd.concat(rows).to_parquet(tmp_path / "ohlcv" / "batch_000.parquet", index=False)
    (tmp_path / "facts").mkdir()
    obs = pd.bdate_range("2020-03-02", "2020-12-31", freq="10B")
    (tmp_path / "facts" / "AAA.json").write_text(json.dumps(dict(
        ticker="AAA", shares={str(d.date()): (10e6 if d < SPLIT_DAY else 40e6) for d in obs},
        info=dict(country="United States", sharesOutstanding=40e6))))
    (tmp_path / "facts" / "BBB.json").write_text(json.dumps(dict(
        ticker="BBB", shares={str(d.date()): 20e6 for d in obs}, info=dict(country="United States"))))
    yield tmp_path
    _clear()


def _write_split(root):
    (root / "splits").mkdir()
    pd.DataFrame(dict(ticker=["AAA"], Date=[SPLIT_DAY], ratio=[4.0])).to_parquet(
        root / "splits" / "batch_000.parquet", index=False)
    _clear()


def test_factor_after_counts_only_later_splits():
    ev = [(pd.Timestamp("2020-08-31"), 5.0), (pd.Timestamp("2022-08-25"), 3.0)]      # Tesla
    assert c.factor_after(ev, pd.Timestamp("2016-06-01")) == 15.0
    assert c.factor_after(ev, pd.Timestamp("2020-08-28")) == 15.0
    assert c.factor_after(ev, pd.Timestamp("2020-08-31")) == 3.0        # bölünme günü fiyat zaten bölünmüş
    assert c.factor_after(ev, pd.Timestamp("2022-08-25")) == 1.0
    assert c.factor_after([(pd.Timestamp("2023-01-10"), 0.1)], pd.Timestamp("2020-01-02")) == 0.1   # 10→1 birleşme


def test_without_split_file_everything_is_unchanged(lab_root):
    # 2021–26 kökü: bölünme dosyası yok → F ≡ 1, eski hesap (düzeltilmiş fiyat × ham sayı) aynen.
    assert c.split_events() == {}
    assert (c.split_factor().values == 1.0).all()
    cap = c.indicators()["cap"]
    assert cap.at[pd.Timestamp("2020-06-01"), "AAA"] == pytest.approx(25.0 * 10e6)      # kusurlu ama dokunulmadı
    assert c.indicators()["avgvol63"].at[pd.Timestamp("2020-08-03"), "AAA"] == pytest.approx(4e6)


def test_split_correction_restores_the_price_volume_and_cap_of_the_day(lab_root):
    _write_split(lab_root)
    before, after = pd.Timestamp("2020-08-03"), pd.Timestamp("2020-12-01")
    F = c.split_factor()
    assert F.at[before, "AAA"] == 4.0 and F.at[SPLIT_DAY, "AAA"] == 1.0 and F.at[after, "AAA"] == 1.0
    assert (F["BBB"] == 1.0).all()
    assert c.actual_close().at[before, "AAA"] == pytest.approx(100.0)      # o gün ekrandaki fiyat
    assert c.actual_close().at[after, "AAA"] == pytest.approx(25.0)
    I = c.indicators()
    assert I["avgvol63"].at[before, "AAA"] == pytest.approx(1e6)            # o günün gerçek hisse adedi
    assert I["dv20"].at[before, "AAA"] == pytest.approx(25.0 * 4e6)         # dolar hacim bölünmeden bağımsız
    for d in (before, pd.Timestamp("2020-08-28"), SPLIT_DAY, pd.Timestamp("2020-09-08"), after):
        assert I["cap"].at[d, "AAA"] == pytest.approx(1e9), d              # 100$×10M = 25$×40M, bölünme gününde de
    assert I["cap"].at[before, "BBB"] == pytest.approx(50.0 * 20e6)
    # o günün hisse sayısı (ham) korunur: actual_close × shares_panel = piyasa değeri
    assert c.shares_panel().at[before, "AAA"] == pytest.approx(10e6)
    assert c.shares_panel().at[after, "AAA"] == pytest.approx(40e6)


def test_current_share_fallback_is_converted_to_the_units_of_the_day(lab_root):
    # Serisi olmayan hisse güncel sayıya (bugünün birimi, 40M) düşer → bölünme öncesi günde 10M sayılmalı.
    f = lab_root / "facts" / "AAA.json"
    rec = json.loads(f.read_text())
    rec.pop("shares")
    f.write_text(json.dumps(rec))
    _write_split(lab_root)
    assert c.shares_panel().at[pd.Timestamp("2020-08-03"), "AAA"] == pytest.approx(10e6)
    assert c.indicators()["cap"].at[pd.Timestamp("2020-08-03"), "AAA"] == pytest.approx(1e9)


def test_asof_window_shows_the_bars_live_would_have_seen():
    days = pd.bdate_range("2020-01-02", "2020-08-28")
    df = pd.DataFrame(dict(Date=days, Open=25.0, High=26.0, Low=24.0, Close=25.0, Volume=4e6))
    d = days[-1]
    same = c.asof_window(df, d, 1.0)
    pd.testing.assert_frame_equal(same, c.window_6mo(df, d))
    w = c.asof_window(df, d, 4.0)
    assert w.Close.iloc[-1] == 100.0 and w.High.iloc[-1] == 104.0 and w.Volume.iloc[-1] == 1e6
    assert np.allclose(w.Close * w.Volume, same.Close * same.Volume)        # dolar hacim aynı
    assert df.Close.iloc[-1] == 25.0                                        # kaynak tablo değişmez


@pytest.mark.parametrize("day, known", [
    ("2018-03-01", True),     # 2018-02-10 ile 2018-05-08 arası 87 gün
    ("2018-05-08", True),     # bilanço günü: sonraki kayıt o günün kendisi
    ("2018-09-01", False),    # 2018-08-07 → 2019-02-12: 189 gün, arada bir çeyrek eksik
    ("2017-01-15", False),    # ilk kayıttan önce
    ("2019-06-01", False),    # son kayıttan sonra
])
def test_earnings_calendar_rule(day, known):
    ed = ["2017-11-07", "2018-02-10", "2018-05-08", "2018-08-07", "2019-02-12", "2019-05-07"]
    assert c.earnings_calendar_known(ed, pd.Timestamp(day)) is known


def test_earnings_calendar_rule_accepts_a_slow_fourth_quarter():
    # 3. çeyrek 1 Kasım, yıllık sonuç 31 Mart (150 gün): eksik çeyrek değil.
    assert c.earnings_calendar_known(["2018-11-01", "2019-03-31"], pd.Timestamp("2019-01-15")) is True
    assert c.earnings_calendar_known(["2018-11-01", "2019-04-10"], pd.Timestamp("2019-01-15")) is False   # 160 gün
