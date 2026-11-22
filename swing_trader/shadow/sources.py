"""Gölge sinyaller için canlı veri kaynakları (ağ erişimi YALNIZ burada).

- SEC EDGAR günlük indeks + Form 4 XML (H4). Lab, aynı Form 4'lerin DERA çeyreklik
  tablolarını kullandı; XML ayrıştırması DERA alanlarıyla eşlenir.
- Yahoo bilanço takvimi (kim açıkladı) + ``get_earnings_dates`` (H5; lab'ın kaynağı).
- Fiyat: canlı tarayıcının fetcher'ı (yfinance auto_adjust + OHLC onarımı).
- Aday evreni: NASDAQ/NYSE resmi sembol dizini (Phase-1 CANDIDATE sınıflaması).

SEC adil kullanım: saniyede ≤10 istek; tanımlayıcı User-Agent. Varsayılan UA kişisel
bilgi içermez; ``SEC_USER_AGENT`` ortam değişkeniyle değiştirilebilir.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Dict, Iterable, List, Optional, Set
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)

SEC_UA = os.environ.get("SEC_USER_AGENT") or "swing-trade-research-client admin@research-client.invalid"
_SEC_MIN_INTERVAL = 0.125          # ≤ 8 istek/sn (SEC sınırı 10)
_sec_lock = threading.Lock()
_sec_last = [0.0]


class SecIndexMissing(Exception):
    """O gün için günlük indeks yok (henüz yayımlanmadı ya da EDGAR tatili)."""


def _sec_get(url: str, timeout: int = 30, retries: int = 4) -> bytes:
    for attempt in range(retries):
        with _sec_lock:
            wait = _sec_last[0] + _SEC_MIN_INTERVAL - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _sec_last[0] = time.monotonic()
        try:
            req = Request(url, headers={"User-Agent": SEC_UA, "Accept-Encoding": "identity"})
            with urlopen(req, timeout=timeout) as r:
                return r.read(5_000_001)
        except HTTPError as e:
            if e.code == 404:
                raise
            if e.code in (403, 429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except (URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def form4_index(day: date):
    """O günün ``form.idx`` dosyasındaki tam '4' formları: (birincil yollar, erişim → tüm yollar)."""
    q = (day.month - 1) // 3 + 1
    url = f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{q}/form.{day:%Y%m%d}.idx"
    try:
        raw = _sec_get(url).decode("latin-1")
    except HTTPError as e:
        if e.code == 404:
            raise SecIndexMissing(str(day))
        raise
    paths: Dict[str, List[str]] = {}
    for line in raw.splitlines():
        if not line.startswith("4 "):             # '4/A' dahil değil (lab: DOCUMENT_TYPE == '4')
            continue
        m = re.search(r"(edgar/data/\S+\.txt)\s*$", line)
        if m:
            # Aynı dosya birden çok CIK (ihraççı + sahip) altında listelenebilir → tek indirme.
            paths.setdefault(m.group(1).rsplit("/", 1)[-1], []).append(m.group(1))
    return [v[0] for v in paths.values()], paths


def _txt(el, path: str) -> Optional[str]:
    node = el.find(path)
    if node is None:
        return None
    val = node.find("value")
    s = (val.text if val is not None else node.text) or ""
    s = s.strip()
    return s or None


def _flag(el, tag: str) -> bool:
    s = (_txt(el, tag) or "").lower()
    return s in ("1", "true", "y", "yes")


def parse_form4(raw: str, accession: str, filing_date: date) -> Optional[Dict]:
    """Form 4 tam gönderim metninden DERA-eşlenik alanlar."""
    m = re.search(r"<XML>(.*?)</XML>", raw, re.S | re.I)
    if not m:
        return None
    body = m.group(1).strip()
    try:
        root = ET.fromstring(body.encode("utf-8"))
    except ET.ParseError:
        return None
    owners = []
    for ro in root.findall("reportingOwner"):
        rel = ro.find("reportingOwnerRelationship")
        roles = []
        if rel is not None:
            for tag, name in (("isDirector", "Director"), ("isOfficer", "Officer"),
                              ("isTenPercentOwner", "TenPercentOwner"), ("isOther", "Other")):
                if _flag(rel, tag):
                    roles.append(name)
        owners.append(dict(cik=_txt(ro, "reportingOwnerId/rptOwnerCik"), relationship=",".join(roles)))
    txs = []
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        txs.append(dict(code=_txt(tx, "transactionCoding/transactionCode"),
                        acquired=_txt(tx, "transactionAmounts/transactionAcquiredDisposedCode"),
                        shares=_txt(tx, "transactionAmounts/transactionShares"),
                        price=_txt(tx, "transactionAmounts/transactionPricePerShare")))
    return dict(form=_txt(root, "documentType"), accession=accession, filing_date=filing_date,
                symbol=_txt(root, "issuer/issuerTradingSymbol"),
                issuer_cik=_txt(root, "issuer/issuerCik"), owners=owners, transactions=txs)


def fetch_form4_filings(day: date, workers: int = 8) -> List[Dict]:
    """O günün tüm Form 4'leri (ayrıştırılmış). Hata veren tek dosya atlanır, sayılır."""
    paths, alternates = form4_index(day)

    def one(path):
        name = path.rsplit("/", 1)[-1]
        acc = name.replace(".txt", "")
        err = None
        for p in alternates.get(name, [path]):       # 404'te diğer CIK yolunu dene
            try:
                raw = _sec_get("https://www.sec.gov/Archives/" + p).decode("utf-8", "replace")
                return parse_form4(raw, acc, day)
            except HTTPError as e:
                err = e
                if e.code != 404:
                    break
            except Exception as e:                   # tek dosya hatası günü düşürmesin
                err = e
                break
        logger.debug("Form 4 fetch failed %s: %s", path, err)
        return {"error": str(err), "accession": acc, "path": path}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = [f for f in ex.map(one, paths) if f]
    # İkinci tur: geçici hatalar (zaman aşımı, 429/503) için bir kez daha.
    retry = [f["path"] for f in out if "error" in f and "404" not in f["error"]]
    if retry:
        time.sleep(5)
        again = {f.get("accession"): f for f in map(one, retry)}
        out = [again.get(f["accession"], f) if "error" in f else f for f in out]
    return out


# ── Bilanço (Yahoo) ─────────────────────────────────────────────────────────

def earnings_reporters(start: date, end: date) -> pd.DataFrame:
    """[start, end] aralığında bilanço açıklayanlar (Yahoo takvimi, sayfalı)."""
    import yfinance as yf
    cal = yf.Calendars(start=start.isoformat(), end=end.isoformat())
    frames, offset = [], 0
    while True:
        df = cal.get_earnings_calendar(filter_most_active=False, limit=100, offset=offset, force=True)
        if df is None or df.empty:
            break
        frames.append(df)
        if len(df) < 100 or offset > 5000:
            break
        offset += 100
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    out = out[~out.index.duplicated()]
    return out


def earnings_rows(ticker: str, limit: int = 8) -> List[Dict]:
    """``get_earnings_dates`` satırları — lab/fetch_earnings_full.py ile aynı alanlar."""
    import yfinance as yf
    df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    rows = []
    if df is None or df.empty:
        return rows
    for ts, r in df.iterrows():
        def num(k):
            v = r.get(k)
            return None if v is None or pd.isna(v) else float(v)
        rows.append(dict(ts=pd.Timestamp(ts).isoformat(), est=num("EPS Estimate"),
                         rep=num("Reported EPS"), surprise=num("Surprise(%)")))
    return rows


# ── Fiyat / hisse sayısı / evren ───────────────────────────────────────────

def fetch_bars(tickers: Iterable[str], start: date, end: date) -> Dict[str, pd.DataFrame]:
    from ..data.fetcher import DataFetcher
    tickers = sorted(set(tickers))
    if not tickers:
        return {}
    return DataFetcher().fetch_multiple_stocks_batch(
        tickers, start_date=start.isoformat(),
        end_date=date.fromordinal(end.toordinal() + 1).isoformat(), min_rows=1)


def shares_outstanding(ticker: str) -> Optional[float]:
    import yfinance as yf
    try:
        v = yf.Ticker(ticker).fast_info["shares"]
        return float(v) if v else None
    except Exception:
        return None


def candidate_universe() -> Set[str]:
    """NASDAQ/NYSE hisse senedi adayları (yfinance sembolleriyle) — lab havuzunun kaynağı."""
    from ..discovery.master import fetch_snapshot
    snap, _ = fetch_snapshot()
    return {r["yfinance_symbol_candidate"] for r in snap["records"]
            if r.get("discovery_status") == "CANDIDATE" and r.get("yfinance_symbol_candidate")}


# ── H6 "Sade VCE" (Ek F): canlı ürünün KENDİ parçaları, aynı sıra ────────────

def live_vce_hits(asof: date) -> Dict:
    """Canlı evren + canlı barlar → (sert filtreler + bilanço kapısı) VE VCE tetiği.

    Canlı taramayla aynı girdiler: ``get_small_cap_universe`` (Finviz, canlı ayarlar) ve
    ``fetch_multiple_stocks_batch(period="6mo")``. ``scan_stock`` ile aynı bileşenler:
    ``filters.apply_all_filters`` (bilanço iki kaynak, bilinmiyorsa ret) ve
    ``signals.check_all_triggers``. Tetikten sonraki katmanlar (RSI, Weinstein, skor, rejim,
    top N) BİLEREK yok. Sonuç mantıksal VE olduğundan sıra yalnız maliyeti etkiler: önce tetik
    (ucuz), sonra filtreler (bilanço isteği yalnız VCE'den geçenlere).
    """
    from datetime import datetime as _dt
    from ..data.fetcher import DataFetcher
    from ..small_cap.engine import SmallCapEngine

    eng = SmallCapEngine()
    tickers = eng.get_small_cap_universe()
    source = getattr(eng.universe_provider, "last_universe_source", None)
    if source == "static_fallback":       # canlı tarama da bu durumda durur (universe_unavailable)
        return dict(hits=[], stats=dict(universe=len(tickers), universe_source=source,
                                        error="UNIVERSE_UNAVAILABLE"))
    data = DataFetcher().fetch_multiple_stocks_batch(tickers, period="6mo")
    stats = dict(universe=len(tickers), universe_source=source, fetched=len(data), current_bar=0,
                 vce=0, filter_rejects={})
    hits = []
    for t, df in data.items():
        if df is None or len(df) < 20:
            continue
        dcol = df["Date"] if "Date" in df.columns else pd.Series(df.index)
        last = pd.Timestamp(dcol.iloc[-1]).date()
        if last != asof:                       # bayat/eksik son bar → bu koşuda karar yok
            continue
        stats["current_bar"] += 1
        _, td = eng.signals.check_all_triggers(df)
        if not ((td.get("triggers") or {}).get("vce_breakout") or {}).get("passed"):
            continue
        stats["vce"] += 1
        info = eng.universe_provider.get_ticker_metadata(t) or eng.get_stock_info(t)
        ok, res = eng.filters.apply_all_filters(t, df, info, _dt(asof.year, asof.month, asof.day))
        if not ok:
            failed = next((k for k, v in (res.get("filters") or {}).items() if not v.get("passed")), "?")
            stats["filter_rejects"][failed] = stats["filter_rejects"].get(failed, 0) + 1
            continue
        tail = df.tail(20)
        hits.append(dict(ticker=t, dv20=float((tail["Close"] * tail["Volume"]).mean()),
                         close=float(df["Close"].iloc[-1])))
    return dict(hits=hits, stats=stats)


def champion_trades(since: date) -> List[Dict]:
    """Canlı ürünün paper işlemleri (kapanmış, REJECTED hariç, girişi ``since`` ve sonrası).

    Salt okuma; kaynak ayrımı tutulmadığından manuel eklenen işlemler de dahildir.
    """
    from ..paper_trading.storage import PaperTradeStorage

    rows = PaperTradeStorage().get_closed_trades(limit=5000) or []
    s = since.isoformat()
    return [r for r in rows if r.get("status") != "REJECTED" and str(r.get("entry_date") or "")[:10] >= s
            and r.get("realized_pnl_pct") is not None]
