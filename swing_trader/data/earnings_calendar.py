"""Yahoo bilanço takvimi — bilanço kapısının İKİNCİ, bağımsız kaynağı.

Neden: hissenin kendi bilanço tarihleri (yfinance ``earnings_dates``) ara sıra boş ya da hatalı
dönüyor. Eski kapı bunu "yakında bilanço yok" sayıp geçiriyordu (canlı SAIC, 2026-08-31 bilanço
gününde açıldı). Takvim, bir tarih aralığında açıklama yapacak bütün şirketleri tek sorguda
(sayfalı) verir; tarama başına bir kez çekilir ve önbelleğe alınır.

Dönüş sözleşmesi: ``None`` = takvim ALINAMADI (bilinmiyor). Boş sözlük asla dönmez — bir
hafta içi aralığında Yahoo'nun hiç açıklama listelememesi veri hatası kabul edilir (None).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, timedelta
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_OK_TTL_S = 6 * 3600       # başarılı takvim: aynı gün içindeki taramalar yeniden kullanır
_FAIL_TTL_S = 600          # başarısız: her hisse için tekrar denenmez, 10 dk sonra yeniden denenir
_MAX_PAGES = 60
_lock = threading.Lock()
_cache: Dict[Tuple[date, date], Tuple[float, Optional[Dict[str, Set[date]]]]] = {}


def normalize_symbol(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")


def _fetch(start: date, end_inclusive: date) -> Optional[Dict[str, Set[date]]]:
    import pandas as pd
    import yfinance as yf

    # Yahoo'da bitiş tarihi DAHİL DEĞİL (2026-08-31..09-03 sorgusu en geç 09-02'yi döndürüyor).
    cal = yf.Calendars(start=start.isoformat(), end=(end_inclusive + timedelta(days=1)).isoformat())
    out: Dict[str, Set[date]] = {}
    offset = 0
    for _ in range(_MAX_PAGES):
        df = cal.get_earnings_calendar(filter_most_active=False, limit=100, offset=offset, force=True)
        if df is None or df.empty:
            break
        for sym, ts in zip(df.index, df["Event Start Date"]):
            if ts is None or pd.isna(ts):
                continue
            ts = pd.Timestamp(ts)
            d = (ts.tz_convert("America/New_York") if ts.tzinfo else ts).date()
            if start <= d <= end_inclusive:
                out.setdefault(normalize_symbol(sym), set()).add(d)
        if len(df) < 100:
            break
        offset += 100
    return out or None


def reporters(start: date, end_inclusive: date) -> Optional[Dict[str, Set[date]]]:
    """{SEMBOL: {açıklama günü (NY)}} — [start, end_inclusive]; None = bilinmiyor."""
    key = (start, end_inclusive)
    now = time.monotonic()
    with _lock:   # tek fetch: eşzamanlı taramalar aynı sonucu bekler
        hit = _cache.get(key)
        if hit is not None:
            at, value = hit
            if now - at < (_OK_TTL_S if value is not None else _FAIL_TTL_S):
                return value
        try:
            value = _fetch(start, end_inclusive)
        except Exception as e:                       # ağ/ayrıştırma hatası → bilinmiyor
            logger.warning("Earnings calendar fetch failed (%s..%s): %s", start, end_inclusive, e)
            value = None
        if value is None:
            logger.warning("Earnings calendar unavailable for %s..%s", start, end_inclusive)
        for k in [k for k in _cache if k[1] < start - timedelta(days=7)]:
            _cache.pop(k, None)                      # eski pencereleri at
        _cache[key] = (now, value)
        return value


def clear_cache() -> None:
    with _lock:
        _cache.clear()
