"""Ek Z saf kuralları: bilanço tepki seansı ve "bilanço öncesi alma" maskesi (veri/ağ yok)."""
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
WINDOW = 9                    # e < R ≤ e + 9 seans → alınmaz (pozisyonun 2.–10. seansı)


def reaction_index(accepted_utc: str, cal: pd.DatetimeIndex) -> int:
    """Açıklamanın tepki seansı (takvim indeksi): 09:30 ET'den önce ve o gün seans → o gün; aksi → sonraki seans."""
    ts = pd.Timestamp(accepted_utc).tz_convert(NY) if pd.Timestamp(accepted_utc).tzinfo else \
        pd.Timestamp(accepted_utc).tz_localize("UTC").tz_convert(NY)
    d = pd.Timestamp(ts.date())
    k = int(cal.searchsorted(d, side="left"))
    if k < len(cal) and cal[k] == d and (ts.hour, ts.minute) >= (9, 30):
        k += 1
    return k


def skip_mask(cal, columns, reactions: dict) -> pd.DataFrame:
    """Sinyal günü i alınmaz ⇔ bir tepki seansı R için i+2 ≤ R ≤ i+1+WINDOW (e = i+1 < R ≤ e + WINDOW)."""
    m = np.zeros((len(cal), len(columns)), dtype=bool)
    col = {c: j for j, c in enumerate(columns)}
    for t, ks in reactions.items():
        j = col.get(t)
        if j is None:
            continue
        for k in ks:
            lo, hi = max(k - 1 - WINDOW, 0), k - 2
            if hi >= 0:
                m[lo:min(hi, len(cal) - 1) + 1, j] = True
    return pd.DataFrame(m, index=cal, columns=columns)
