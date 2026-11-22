"""Hisse ihracı (seyreltme) başvuruları — kıdemli swing trader'ın alım öncesi kontrolü.

Küçük şirketler yükselişte yeni hisse satar (kayıt beyanı S-1/S-3/F-x, fiyatlama 424B). Arz baskısı
kırılımı keser; grafik sinyali bunu görmez. Lab v3 Ek H (H8): son 30 günde böyle bir başvurusu olan
sinyalleri elemek canlı replikayı −0.24 → +0.26%/işlem taşıdı ama sağlamlık şartını geçemedi →
KAPI DEĞİL, görünür İŞARET + ileri ölçüm (gölge raporu işaretli/işaretsiz canlı işlemleri ayırır).

Kaynak: SEC EDGAR günlük form indeksi + company_tickers.json (CIK→ticker). Ağ erişimi gölge
koşucunun SEC istemcisiyle (≤8 istek/sn, kişisel bilgi içermeyen User-Agent). Geçmiş günlerin
indeksi değişmez → kalıcı süreç önbelleği; bugünün/dünün dosyası eksik olabileceği için önbelleğe
alınmaz.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DILUTION_FORMS = frozenset({"S-1", "S-3", "S-3ASR", "F-1", "F-3", "F-3ASR",
                            "424B1", "424B2", "424B3", "424B4", "424B5", "424B7"})   # Ek H, /A hariç
WINDOW_DAYS = 30
_lock = threading.Lock()
_days: Dict[date, Dict[int, List[str]]] = {}
_tickers: Dict[str, object] = {"at": 0.0, "map": {}}
_CIK_RE = re.compile(r"edgar/data/(\d+)/")


def _get(url: str) -> bytes:
    from ..shadow.sources import _sec_get
    return _sec_get(url)


def _ticker_to_cik() -> Dict[str, int]:
    with _lock:
        if _tickers["map"] and time.time() - float(_tickers["at"]) < 24 * 3600:
            return _tickers["map"]  # type: ignore[return-value]
    raw = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    m = {str(v["ticker"]).upper().replace(".", "-"): int(v["cik_str"]) for v in raw.values()}
    with _lock:
        _tickers.update(at=time.time(), map=m)
    return m


def _day_filings(day: date, today: date) -> Dict[int, List[str]]:
    """O gün dosyalanan seyreltme başvuruları: {CIK: [form, ...]}."""
    with _lock:
        if day in _days:
            return _days[day]
    from urllib.error import HTTPError
    q = (day.month - 1) // 3 + 1
    url = f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{q}/form.{day:%Y%m%d}.idx"
    out: Dict[int, List[str]] = {}
    try:
        raw = _get(url).decode("latin-1")
        for line in raw.splitlines():
            parts = line.split()
            if not parts or parts[0] not in DILUTION_FORMS:
                continue
            m = _CIK_RE.search(line)
            if m:
                out.setdefault(int(m.group(1)), []).append(parts[0])
    except HTTPError as e:
        if e.code != 404:                        # 404 = tatil / hafta sonu / henüz yayımlanmadı
            raise
    if (today - day).days >= 2:                  # geçmiş gün değişmez → önbellek
        with _lock:
            _days[day] = out
    return out


def recent_dilution(tickers: Iterable[str], asof: date, days: int = WINDOW_DAYS,
                    today: Optional[date] = None) -> Dict[str, List[Dict]]:
    """[asof − days, asof] aralığında seyreltme başvurusu olan hisseler: {ticker: [{form, date}]}."""
    want = {str(t).upper() for t in tickers}
    if not want:
        return {}
    today = today or date.today()
    cik_of = _ticker_to_cik()
    ciks = {cik_of[t]: t for t in want if t in cik_of}
    out: Dict[str, List[Dict]] = {}
    d = asof - timedelta(days=days)
    while d <= asof:
        if d.weekday() < 5:
            for cik, forms in _day_filings(d, today).items():
                if cik in ciks:
                    out.setdefault(ciks[cik], []).extend(dict(form=f, date=d.isoformat()) for f in forms)
        d += timedelta(days=1)
    return out
