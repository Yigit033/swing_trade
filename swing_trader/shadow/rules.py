"""H4 / H5 gölge sinyal kuralları — SAF fonksiyonlar (ağ, DB, saat yok).

Kaynak: lab/PREREGISTRATION.md Ek A (kurallar) ve Ek C (gölge mod + terfi kapısı).
Parametreler sonuç görülmeden yazıldı ve DONDURULDU. Değiştirmek yeni bir ön kayıt
(yeni hipotez numarası) gerektirir; ``PARAMS_SHA256`` testi sessiz değişikliği yakalar.

Lab paritesi: tanımlar lab/challengers.py (atr14, walk) ve lab/info_signals.py
(h4_events, h5_events, to_candidates) ile birebir aynıdır;
swing_trader/tests/test_shadow_rules.py bunu lab verisi üzerinde doğrular.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time as dtime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.market_calendar import is_trading_day, next_trading_day, prev_trading_day

NY = "America/New_York"

PARAMS: Dict[str, Dict] = {
    "common": dict(
        price_min=7.0, price_max=250.0,
        avgvol63_min=300_000, avgvol63_window=63, avgvol63_min_periods=40,
        dv20_min=5e6, dv20_window=20, dv20_min_periods=15,
        cap_min=3e8, cap_max=1e10,
        atr_period=14, stop_atr=2.5, max_bars=20, cost_pct=0.50,
    ),
    "H4": dict(window_days=30, min_insiders=2, min_value=100_000, gap_days=60,
               owner_roles=("Director", "Officer"), trans_code="P", acquired="A", form="4"),
    "H5": dict(min_surprise=10.0, min_abs_estimate=0.02, open_cutoff=(9, 30)),
}
PARAMS_SHA256 = hashlib.sha256(json.dumps(PARAMS, sort_keys=True, default=list).encode()).hexdigest()

C = PARAMS["common"]

# Ek F — H6 "Sade VCE": canlı ürünün en küçük çekirdeği. H4/H5'ten AYRI hash: H6 eklemek H4/H5'in
# ön kayıt kilidini (PARAMS_SHA256) değiştirmez. Sinyal canlı ürünün kendi fonksiyonlarıyla
# üretilir (sources.live_vce_hits); burada yalnız çıkış ve kimlik parametreleri dondurulur.
PARAMS_H6: Dict = dict(
    universe="live_product_universe", bars="DataFetcher.fetch_multiple_stocks_batch(6mo)",
    filters="live_hard_filters+live_earnings_gate", trigger="check_all_triggers.vce_breakout",
    removed=("rvol_thrust", "swing_confirmation", "rsi_gate", "weinstein", "quality_threshold",
             "regime_threshold", "top_n", "capacity", "gap_filter", "stop", "target", "trail"),
    entry="next_session_open", hold_sessions=8, stop=None, rank_key="dollar_volume_20",
    cost_pct=0.50, min_closed=100,
)
PARAMS_H6_SHA256 = hashlib.sha256(json.dumps(PARAMS_H6, sort_keys=True, default=list).encode()).hexdigest()

# Ek J/J2 — H6S "Profesyonel VCE": H6 isabetleri + KÜÇÜK HİSSE (IWM) piyasa freni + felaket stop'u.
# Ayrı kilit; H4/H5/H6 kilitleri değişmez.
PARAMS_H6S: Dict = dict(
    base="H6", regime_index="IWM", regime_fn="regime_from_spy_close(IWM 1y close, VIX)",
    bear_block=True, caution_max_concurrent=1, stop_atr=2.5, hold_sessions=8,
    rank_key="dollar_volume_20", cost_pct=0.50, min_closed=100,
)
PARAMS_H6S_SHA256 = hashlib.sha256(json.dumps(PARAMS_H6S, sort_keys=True, default=list).encode()).hexdigest()

# Ek Q — H6SR "Kazananı koştur": H6S ile AYNI giriş ve fren; çıkış Ek O1 (lab/swing_o.walk_runner):
# 8. kapanış giriş açılışının üstündeyse iz süren stop (girişten beri en yüksek kapanış − 2.5×ATR14),
# en çok 40 seans. Ayrı kilit; H4/H5/H6/H6S kilitleri ve EXITS değişmez.
PARAMS_H6SR: Dict = dict(
    base="H6S", entry_and_regime="same_as_H6S", stop_atr=2.5, hold_sessions=8,
    runner_if="close_at_hold > entry_open", trail_atr=2.5, trail_ref="highest_close_since_entry",
    max_sessions=40, rank_key="dollar_volume_20", cost_pct=0.50, min_closed=100,
)
PARAMS_H6SR_SHA256 = hashlib.sha256(json.dumps(PARAMS_H6SR, sort_keys=True, default=list).encode()).hexdigest()

# Ek U — MOM "Lider rotasyonu": büyük ve likit hisselerde aylık 12-1 momentum rotasyonu (Ek T kuralı,
# lab/rotation.py ile birebir). Geçmiş sınavda KALDI (2021–26'da düşüş −71%); kullanıcı kararıyla yalnız
# gölge ileri testte. Ayrı kilit; diğer kilitler değişmez. Çıkış aylık sıraya bağlı olduğundan EXITS'te yok.
PARAMS_MOM: Dict = dict(
    pool="L", price_min=7.0, avgvol63_min=300_000, avgvol63_window=63, avgvol63_min_periods=40,
    dv20_min=5e6, dv20_window=20, dv20_min_periods=15, cap_min=1e10,
    decision="last_session_of_month", lookback=252, skip=21, sma=200,
    max_positions=10, sell_rank=20, buy_regime="BULL", regime_index="SPY",
    regime_fn="regime_from_spy_close(SPY 1y close, VIX)", entry="next_session_open", exit="next_session_open",
    stop=None, cost_pct=0.10, fee_per_order=1.5, min_closed=40, min_months=24, max_months=36,
)
PARAMS_MOM_SHA256 = hashlib.sha256(json.dumps(PARAMS_MOM, sort_keys=True, default=list).encode()).hexdigest()

# Ek W — GRW "Büyüme + kırılım": Ek V'nin "lider şartsız" kıyası (lab/pro_rules.py ile birebir). Ön kayıtlı
# paket (PRO) geçmiş sınavda KALDI; bu varyant sonuç görüldükten sonra seçildi ve endeksi geçemedi →
# kullanıcı kararıyla yalnız gölge ileri testte. Ayrı kilit; diğer kilitler değişmez. Çıkış SMA50'ye bağlı
# olduğundan EXITS'te yok (``grw_plan``).
PARAMS_GRW: Dict = dict(
    pool="G", price_min=10.0, avgvol63_min=300_000, avgvol63_window=63, avgvol63_min_periods=40,
    dv20_min=1e7, dv20_window=20, dv20_min_periods=15, cap_min=1e9,
    trigger="signals.check_vce_breakout", growth_source="SEC XBRL (first filed value, filed < signal day)",
    min_sales_growth=0.25, profit_rule="ni_q > ni_q_minus_4", max_stale_days=200, yoy_days=(350, 380),
    buy_regime="BULL", regime_index="SPY", regime_fn="regime_from_spy_close(SPY 1y close, VIX)",
    entry="next_session_open", max_stop=0.08, stop="max(signal_low, entry * (1 - max_stop))",
    exit="close_below_sma_then_next_open", sma=50, max_sessions=None,
    risk=0.01, risk_cold=0.005, cold_streak=2, max_frac=0.25,
    cost_pct=0.20, fee_per_order=1.5, min_closed=60, min_months=24, max_months=36,
)
PARAMS_GRW_SHA256 = hashlib.sha256(json.dumps(PARAMS_GRW, sort_keys=True, default=list).encode()).hexdigest()

# Strateji → (ATR stop katı, en çok seans). None stop = stop yok.
EXITS: Dict[str, Tuple[Optional[float], int]] = {
    "H4": (C["stop_atr"], C["max_bars"]),
    "H5": (C["stop_atr"], C["max_bars"]),
    "H6": (None, PARAMS_H6["hold_sessions"]),
    "H6S": (PARAMS_H6S["stop_atr"], PARAMS_H6S["hold_sessions"]),
}

# Kazananı koşturan stratejiler → (ATR stop katı, temel tutuş, iz süren stop ATR katı, en çok seans).
RUNNERS: Dict[str, Tuple[float, int, float, int]] = {
    "H6SR": (PARAMS_H6SR["stop_atr"], PARAMS_H6SR["hold_sessions"], PARAMS_H6SR["trail_atr"],
             PARAMS_H6SR["max_sessions"]),
}


# ── Takvim ──────────────────────────────────────────────────────────────────

def sessions_between(start: date, end: date) -> List[date]:
    """[start, end] aralığındaki NYSE seansları."""
    out, d = [], start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d = date.fromordinal(d.toordinal() + 1)
    return out


def last_session_on_or_before(d: date) -> date:
    return d if is_trading_day(d) else prev_trading_day(d)


# ── Fiyat çerçevesi (lab paneliyle aynı hizalama) ─────────────────────────

def align(df: pd.DataFrame, sessions: Sequence[date]) -> pd.DataFrame:
    """Hisse barlarını NYSE seans takvimine hizalar (eksik seans = NaN satır).

    Lab panelleri SPY takvimine pivotlanır; eksik bar NaN olur. Aynı davranış.
    """
    x = df.copy()
    x["Date"] = pd.to_datetime(x["Date"]).dt.tz_localize(None).dt.normalize()
    x = x.drop_duplicates("Date", keep="last").set_index("Date")
    idx = pd.DatetimeIndex(pd.to_datetime(list(sessions)))
    return x.reindex(idx)[["Open", "High", "Low", "Close", "Volume"]]


def atr14(frame: pd.DataFrame) -> pd.Series:
    """lab/challengers.atr14: TR'nin 14 seanslık basit ortalaması (Wilder DEĞİL)."""
    h, l, c = frame["High"], frame["Low"], frame["Close"]
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    return tr.rolling(C["atr_period"]).mean()


def pool_member(frame: pd.DataFrame, i: int, shares: Optional[float]) -> Tuple[bool, Dict]:
    """Ön kayıt Bölüm 0 havuzu, seans ``i`` itibarıyla (lab/common.pool_mask)."""
    c, v = frame["Close"], frame["Volume"]
    avgvol63 = v.rolling(C["avgvol63_window"], min_periods=C["avgvol63_min_periods"]).mean().iloc[i]
    dv20 = (c * v).rolling(C["dv20_window"], min_periods=C["dv20_min_periods"]).mean().iloc[i]
    close = c.iloc[i]
    cap = float(shares) * float(close) if shares and np.isfinite(close) else float("nan")
    facts = dict(close=float(close), avgvol63=float(avgvol63), dv20=float(dv20), cap=cap)
    ok = bool(np.isfinite(close) and C["price_min"] <= close <= C["price_max"]
              and np.isfinite(avgvol63) and avgvol63 > C["avgvol63_min"]
              and np.isfinite(dv20) and dv20 > C["dv20_min"]
              and np.isfinite(cap) and C["cap_min"] <= cap <= C["cap_max"])
    return ok, facts


# ── Çıkış (lab/challengers.walk + info_signals.atr_exit) ───────────────────

def walk(frame: pd.DataFrame, ie: int, stop: float, max_bars: int = None) -> Tuple[int, float, str]:
    """Stop önce; gap'te Open'dan dolum; giriş barı 1. bar; eksik barda bekle.

    Dönüş: (çıkış indeksi, çıkış fiyatı, neden) — neden STOP | TIME | OPEN
    (OPEN = henüz ``max_bars`` seans dolmadı; lab'daki END_OF_DATA).
    """
    max_bars = max_bars or C["max_bars"]
    O, L, Cl = (frame[k].to_numpy(dtype=float) for k in ("Open", "Low", "Close"))
    n = len(Cl)
    for k in range(ie, min(n, ie + max_bars)):
        if not np.isfinite(Cl[k]):
            continue
        if stop is not None and L[k] <= stop:
            return k, float(min(O[k], stop)), "STOP"
    last = min(n, ie + max_bars) - 1
    while last > ie and not np.isfinite(Cl[last]):
        last -= 1
    return last, float(Cl[last]), ("TIME" if ie + max_bars <= n else "OPEN")


def walk_runner(frame: pd.DataFrame, ie: int, stop0: float, atr: float, hold: int, max_bars: int,
                trail_atr: float) -> Tuple[int, float, str]:
    """Ek O1 / lab/swing_o.walk_runner: ``hold``. kapanış girişin üstündeyse iz süren stop ile devam.

    Neden: STOP | TIME (temel tutuşta kârda değil) | TRAIL | TIME_MAX | OPEN (henüz karar yok).
    """
    xi, xp, why = walk(frame, ie, stop0, hold)
    O, L, Cl = (frame[k].to_numpy(dtype=float) for k in ("Open", "Low", "Close"))
    if why != "TIME" or not Cl[xi] > O[ie]:
        return xi, xp, why
    n = len(Cl)
    hi = float(np.nanmax(Cl[ie: xi + 1]))
    stop = max(stop0, hi - trail_atr * atr)
    for k in range(xi + 1, min(n, ie + max_bars)):
        if not np.isfinite(Cl[k]):
            continue
        if L[k] <= stop:
            return k, float(min(O[k], stop)), "TRAIL"
        hi = max(hi, float(Cl[k]))
        stop = max(stop, hi - trail_atr * atr)
    last = min(n, ie + max_bars) - 1
    while last > ie and not np.isfinite(Cl[last]):
        last -= 1
    return last, float(Cl[last]), ("TIME_MAX" if ie + max_bars <= n else "OPEN")


def plan_trade(frame: pd.DataFrame, signal_day: date, strategy: str = "H5") -> Dict:
    """Sinyal barı ``signal_day`` → giriş/stop/çıkış (hizalanmış çerçeve üzerinde).

    Çerçeve sinyal günü ve sonrasındaki seansları içermeli. Sonuç alanları:
    status PENDING (giriş barı yok) | SKIPPED | OPEN | CLOSED. Çıkış kuralı ``EXITS[strategy]``
    (kazananı koşturanlar: ``RUNNERS[strategy]``).
    """
    runner = RUNNERS.get(strategy)
    stop_atr, max_bars = (runner[0], runner[3]) if runner else EXITS[strategy]
    idx = list(frame.index.date)
    if signal_day not in idx:
        return dict(status="SKIPPED", reason="NO_SIGNAL_BAR")
    i = idx.index(signal_day)
    a = atr14(frame).iloc[i]
    if stop_atr is None:
        a = float(a) if np.isfinite(a) and a > 0 else None   # stopsuz strateji ATR'ye muhtaç değil
    elif not np.isfinite(a) or a <= 0:
        return dict(status="SKIPPED", reason="ATR_UNAVAILABLE")
    a = None if a is None else float(a)
    if i + 1 >= len(idx):
        return dict(status="PENDING", atr=a)
    o = frame["Open"].iloc[i + 1]
    if not np.isfinite(o) or o <= 0:
        return dict(status="SKIPPED", reason="NO_ENTRY_BAR", atr=a)
    stop = None if stop_atr is None else float(o - stop_atr * a)
    if runner:
        xi, xp, why = walk_runner(frame, i + 1, stop, a, runner[1], runner[3], runner[2])
    else:
        xi, xp, why = walk(frame, i + 1, stop, max_bars)
    out = dict(atr=a, entry_day=idx[i + 1], entry_price=float(o), stop=stop,
               bars=xi - i, exit_reason=why, mark_price=xp, mark_day=idx[xi],
               ret_pct=100.0 * (xp / o - 1))
    out["status"] = "OPEN" if why == "OPEN" else "CLOSED"
    if out["status"] == "CLOSED":
        out.update(exit_day=idx[xi], exit_price=xp)
    return out


# ── H4: içeriden alım kümesi ──────────────────────────────────────────────

def h4_purchase_rows(filing: Dict) -> List[Dict]:
    """Ayrıştırılmış bir Form 4'ten uygun alım satırları (sahip başına bir satır).

    lab: TRANS_CODE == P ve edinme A, değer = adet × fiyat > 0, erişim numarası başına
    toplam; sahip rolü Director veya Officer içermeli; ortak dosyada değer bir kez sayılır.
    """
    p = PARAMS["H4"]
    if str(filing.get("form", "")).strip() != p["form"]:
        return []
    val = 0.0
    for tx in filing.get("transactions", []):
        if tx.get("code") != p["trans_code"] or tx.get("acquired") != p["acquired"]:
            continue
        try:
            v = float(tx.get("shares")) * float(tx.get("price"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 0:
            val += v
    if val <= 0:
        return []
    owners = [o for o in filing.get("owners", [])
              if any(r in (o.get("relationship") or "") for r in p["owner_roles"])]
    rows, seen = [], set()
    for o in owners:
        cik = str(o.get("cik") or "").lstrip("0")
        if not cik or cik in seen:
            continue
        rows.append(dict(accession=filing["accession"], owner_cik=cik,
                         ticker=normalize_symbol(filing.get("symbol")),
                         filing_date=filing["filing_date"],
                         value_once=val if not seen else 0.0))
        seen.add(cik)
    return rows


def normalize_symbol(sym) -> str:
    return str(sym or "").strip().upper().replace(".", "-")


def h4_events(purchases: pd.DataFrame, prior_event_dates: Dict[str, date] = None,
              observation_start: Optional[date] = None) -> List[Dict]:
    """lab/info_signals.h4_events ile aynı: her dosyalama günü F için (F−30g, F] penceresi.

    ``prior_event_dates``: ticker → defterdeki son H4 olay günü (60 gün kuralı).
    ``observation_start``: pencere gözlem öncesine taşıyorsa olay ``warmup`` işaretlenir.
    """
    p = PARAMS["H4"]
    prior_event_dates = dict(prior_event_dates or {})
    if purchases is None or purchases.empty:
        return []
    df = purchases.copy()
    df["F"] = pd.to_datetime(df["filing_date"])
    window = timedelta(days=p["window_days"])
    out = []
    for t, g in df.groupby("ticker"):
        g = g.sort_values("F")
        last = prior_event_dates.get(t)
        last = pd.Timestamp(last) if last is not None else None
        for F in sorted(g.F.unique()):
            F = pd.Timestamp(F)
            w = g[(g.F > F - window) & (g.F <= F)]
            if w.owner_cik.nunique() >= p["min_insiders"] and w.value_once.sum() >= p["min_value"]:
                if last is None or (F - last).days > p["gap_days"]:
                    warm = bool(observation_start is not None and
                                (F - window).date() < observation_start)
                    out.append(dict(ticker=t, event_date=F.date(), insiders=int(w.owner_cik.nunique()),
                                    value=float(w.value_once.sum()), warmup=warm))
                    last = F
    return out


# ── H5: bilanço sürprizi + pozitif tepki ──────────────────────────────────

def h5_qualifies(surprise, estimate, reported) -> bool:
    p = PARAMS["H5"]
    try:
        s, e, r = float(surprise), float(estimate), float(reported)
    except (TypeError, ValueError):
        return False
    if not all(np.isfinite(x) for x in (s, e, r)):
        return False
    return s >= p["min_surprise"] and r > e and abs(e) >= p["min_abs_estimate"]


def h5_reaction_session(ts) -> Optional[date]:
    """Açıklama ET 09:30'dan önceyse aynı gün (seans ise), değilse sonraki ilk seans.

    lab: ``cal.searchsorted(day, side='left' if before_open else 'right')``.
    """
    t = pd.Timestamp(ts)
    t = t.tz_convert(NY) if t.tzinfo else t.tz_localize(NY)
    d = t.date()
    before_open = (t.hour, t.minute) < PARAMS["H5"]["open_cutoff"]
    if before_open and is_trading_day(d):
        return d
    return next_trading_day(d)


def params_fingerprint() -> str:
    return PARAMS_SHA256


# ── MOM: aylık lider rotasyonu (Ek T / Ek U; lab/rotation.mom_trades ile aynı kural) ──────────

def is_month_end_session(d: date) -> bool:
    """Ayın son seansı mı (sonraki seans başka aydaysa)."""
    return is_trading_day(d) and next_trading_day(d).month != d.month


def last_month_end_session(d: date) -> date:
    """``d`` dahil geriye doğru ilk ay sonu seansı."""
    x = last_session_on_or_before(d)
    while not is_month_end_session(x):
        x = prev_trading_day(x)
    return x


def mom_facts(frame: pd.DataFrame, i: int, cap: Optional[float]) -> Dict:
    """Karar günü (hizalı çerçevede konum ``i``) için havuz L üyeliği, trend ve 12-1 skoru.

    lab: havuz = ``pool_mask_large`` (fiyat ≥ 7, ort. hacim 63 > 300K, dolar hacim 20 > 5M, değer ≥ 10 mlr);
    trend = Close > SMA200; skor = Close[t−21] / Close[t−252] − 1 (eksik bar → hesaplanamaz).
    """
    p = PARAMS_MOM
    c, v = frame["Close"].astype(float), frame["Volume"].astype(float)
    close = float(c.iloc[i])
    avgvol = float(v.rolling(p["avgvol63_window"], min_periods=p["avgvol63_min_periods"]).mean().iloc[i])
    dv = float((c * v).rolling(p["dv20_window"], min_periods=p["dv20_min_periods"]).mean().iloc[i])
    sma = float(c.rolling(p["sma"]).mean().iloc[i])
    cap_ok = cap is not None and np.isfinite(cap) and float(cap) >= p["cap_min"]
    in_pool = bool(np.isfinite(close) and close >= p["price_min"] and np.isfinite(avgvol)
                   and avgvol > p["avgvol63_min"] and np.isfinite(dv) and dv > p["dv20_min"] and cap_ok)
    above = bool(np.isfinite(close) and np.isfinite(sma) and close > sma)
    score = float("nan")
    if i - p["lookback"] >= 0:
        a, b = float(c.iloc[i - p["skip"]]), float(c.iloc[i - p["lookback"]])
        if np.isfinite(a) and np.isfinite(b):
            score = a / b - 1
    return dict(in_pool=in_pool, above=above, score=score, close=close, avgvol63=avgvol, dv20=dv, sma200=sma)


def mom_decide(facts: Dict[str, Dict], held: Iterable[str], regime: str) -> Dict:
    """Ay sonu kararı (saf). ``facts[ticker]`` = ``mom_facts`` çıktısı; ``held`` = elde tutulanlar.

    Sat: uygun değil (havuz dışı / SMA200 altı / skor yok) ya da uygunlar içinde sıra > 20.
    Al: yalnız rejim BULL ise, en yüksek skorlu elde olmayan uygun hisselerle en çok 10 pozisyona tamamla.
    """
    p = PARAMS_MOM
    elig = {t: f["score"] for t, f in facts.items() if f["in_pool"] and f["above"] and np.isfinite(f["score"])}
    ranked = sorted(elig, key=lambda t: -elig[t])
    rank = {t: k + 1 for k, t in enumerate(ranked)}
    sells = []
    for h in held:
        if h in rank and rank[h] <= p["sell_rank"]:
            continue
        f = facts.get(h)
        why = "POOL" if (f is None or not f["in_pool"]) else ("TREND" if not f["above"] else
                                                              ("RANK" if h in rank else "NA"))
        sells.append((h, why))
    sold = {h for h, _ in sells}
    keep = [h for h in held if h not in sold]
    buys: List[str] = []
    if str(regime).upper() == p["buy_regime"]:
        for t in ranked:
            if len(keep) + len(buys) >= p["max_positions"]:
                break
            if t in keep:
                continue
            buys.append(t)
    return dict(sells=sells, buys=buys, rank=rank, eligible=len(ranked))


# ── GRW: büyüme + kırılım (Ek V / Ek W; lab/pro_rules.pro_trades ile aynı kural) ───────────────

def grw_pool(df: pd.DataFrame, cap: Optional[float]) -> Tuple[bool, Dict]:
    """Havuz G üyeliği, çerçevenin SON barı itibarıyla (fiyat ≥ 10, ort. hacim 63 > 300K, dolar hacim 20 > 10M,
    piyasa değeri ≥ 1 mlr). ``df``: canlı barlar (bugüne göre düzeltilmiş = o günün gerçek fiyatı)."""
    p = PARAMS_GRW
    c, v = df["Close"].astype(float), df["Volume"].astype(float)
    close = float(c.iloc[-1])
    avgvol = float(v.rolling(p["avgvol63_window"], min_periods=p["avgvol63_min_periods"]).mean().iloc[-1])
    dv = float((c * v).rolling(p["dv20_window"], min_periods=p["dv20_min_periods"]).mean().iloc[-1])
    cap_ok = cap is not None and np.isfinite(cap) and float(cap) >= p["cap_min"]
    ok = bool(np.isfinite(close) and close >= p["price_min"] and np.isfinite(avgvol) and avgvol > p["avgvol63_min"]
              and np.isfinite(dv) and dv > p["dv20_min"] and cap_ok)
    return ok, dict(close=close, avgvol63=avgvol, dv20=dv, cap=None if cap is None else float(cap))


def grw_plan(frame: pd.DataFrame, signal_day: date) -> Dict:
    """Sinyal barı ``signal_day`` → giriş / stop / çıkış (hizalanmış çerçeve; lab ``_walk_trail``).

    Giriş t+1 açılışı; stop = max(Low_t, (1 − 0.08) × giriş), giriş ≤ Low_t → işlem yok. Her gün sırayla:
    dün Close < SMA50 idiyse açılışta sat (TRAIL) → Low ≤ stop ise min(Open, stop) (STOP) → bugün
    Close < SMA50 ise yarın açılışta satılacak. Süre sınırı yok. Dönüş alanları ``plan_trade`` ile aynı.
    """
    p = PARAMS_GRW
    idx = list(frame.index.date)
    if signal_day not in idx:
        return dict(status="SKIPPED", reason="NO_SIGNAL_BAR")
    i = idx.index(signal_day)
    if i + 1 >= len(idx):
        return dict(status="PENDING")
    O, L, Cl = (frame[k].to_numpy(dtype=float) for k in ("Open", "Low", "Close"))
    S = frame["Close"].astype(float).rolling(p["sma"]).mean().to_numpy()
    ie = i + 1
    o = O[ie]
    if not np.isfinite(o) or o <= 0:
        return dict(status="SKIPPED", reason="NO_ENTRY_BAR")
    if not np.isfinite(L[i]) or o <= L[i]:
        return dict(status="SKIPPED", reason="GAP_BELOW_STOP")
    stop = float(max(L[i], (1 - p["max_stop"]) * o))
    xi, xp, why, sell_next = None, None, "OPEN", False
    for k in range(ie, len(idx)):
        if not np.isfinite(Cl[k]):
            continue                                                   # bar yok: bekle
        if sell_next and np.isfinite(O[k]):
            xi, xp, why = k, float(O[k]), "TRAIL"
            break
        if L[k] <= stop:
            xi, xp, why = k, float(min(O[k], stop)), "STOP"
            break
        if np.isfinite(S[k]) and Cl[k] < S[k]:
            sell_next = True
    if xi is None:                                                     # hâlâ açık: son kapanışla değerle
        xi = len(idx) - 1
        while xi > ie and not np.isfinite(Cl[xi]):
            xi -= 1
        xp = float(Cl[xi])
    out = dict(atr=None, entry_day=idx[ie], entry_price=float(o), stop=stop, bars=xi - i, exit_reason=why,
               mark_price=xp, mark_day=idx[xi], ret_pct=100.0 * (xp / o - 1),
               at_open=bool(why == "TRAIL" or (why == "STOP" and xp == O[xi] and xi > ie)),
               sell_pending=bool(why == "OPEN" and sell_next))
    out["status"] = "OPEN" if why == "OPEN" else "CLOSED"
    if out["status"] == "CLOSED":
        out.update(exit_day=idx[xi], exit_price=xp)
    return out
