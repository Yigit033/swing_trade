"""Ek Z — bilanço tepki seansı ve "bilanço öncesi alma" maskesi."""
import pandas as pd
import pytest

from lab.ern_rules import WINDOW, reaction_index, skip_mask

CAL = pd.bdate_range("2024-07-01", periods=40)          # tatilsiz iş günleri yeterli


@pytest.mark.parametrize("acc,day", [
    ("2024-07-10T20:30:00.000Z", "2024-07-11"),          # 16:30 EDT kapanıştan sonra → ertesi seans
    ("2024-07-10T11:00:00.000Z", "2024-07-10"),          # 07:00 EDT açılıştan önce → aynı gün
    ("2024-07-10T13:31:00.000Z", "2024-07-11"),          # 09:31 EDT seans içinde → ertesi seans
    ("2024-07-13T14:00:00.000Z", "2024-07-15"),          # Cumartesi → Pazartesi
])
def test_reaction_session(acc, day):
    assert CAL[reaction_index(acc, CAL)] == pd.Timestamp(day)


def test_winter_utc_offset():
    cal = pd.bdate_range("2024-01-02", periods=10)
    # kış: 14:00Z = 09:00 EST → açılıştan önce → aynı gün; 14:31Z = 09:31 EST → ertesi seans
    assert cal[reaction_index("2024-01-04T14:00:00Z", cal)] == pd.Timestamp("2024-01-04")
    assert cal[reaction_index("2024-01-04T14:31:00Z", cal)] == pd.Timestamp("2024-01-05")


def test_skip_mask_window():
    r = 20                                                  # tepki seansı indeksi
    m = skip_mask(CAL, ["ABC", "XYZ"], {"ABC": [r]})
    days = [i for i in range(len(CAL)) if m["ABC"].iat[i]]
    # giriş e = i + 1; e < R ≤ e + WINDOW  ⇔  R − WINDOW − 1 ≤ i ≤ R − 2
    assert days == list(range(r - WINDOW - 1, r - 1))
    assert not m["ABC"].iat[r - 1]                         # e = R: tepki giriş açılışında fiyatta → alınır
    assert not m["XYZ"].any()
