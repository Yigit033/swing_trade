"""Gölge sinyallerin günlük koşusu (H4, H5, H6). Ana ürünün işlemlerine DOKUNMAZ.

Sıra: yeni H6 sinyalleri → yeni H5 olayları → yeni H4 olayları → bekleyen/açık gölge işlemlerin
güncellenmesi → rapor + terfi kapısı. Her adım ayrı try/except: biri düşerse diğerleri çalışır.

Karar zamanı: sinyal ``decided_at`` ile yazılır. Bu zaman girişin açılışından (09:30 ET)
sonraysa ``late_decision=1`` — bu işlemler terfi değerlendirmesine girmez (Ek C).
Bu yüzden asıl tetik, NY gece yarısından sonra ve açılıştan ÖNCE koşan cron'dur
(.github/workflows/shadow-run.yml, 05:15/06:15/07:15 UTC — GitHub'ın saatlerce gecikmesine pay bırakır).
"""
from __future__ import annotations

import logging
import time as _time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..utils.market_calendar import (
    is_trading_day,
    last_completed_session,
    next_trading_day,
    prev_trading_day,
)
from . import rules
from .ledger import ShadowLedger
from .report import champion_report, strategy_report

logger = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
H5_LOOKBACK_SESSIONS = 3          # Yahoo EPS'i geç doldurursa son 3 tepki seansı yeniden taranır
H4_MAX_DAYS_PER_RUN = 1           # yoğun günde ~2.400 Form 4 → ~5 dk (SEC hız sınırı); sabah 2 cron
WARMUP_SESSIONS = 40              # ATR14 + seans hizalaması için sinyal öncesi pay


def _start_before(day: date, n_sessions: int = WARMUP_SESSIONS + 30) -> date:
    """``day``'den ``n_sessions`` seans önceki gün (ATR + hacim ortalamaları için pay)."""
    d, k = day, 0
    while k < n_sessions:
        d = prev_trading_day(d)
        k += 1
    return d


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _late(decided_at: datetime, signal_day: date) -> int:
    entry = next_trading_day(signal_day)
    return int(decided_at > datetime(entry.year, entry.month, entry.day, 9, 30, tzinfo=NY))


def _frame(bars: Dict[str, pd.DataFrame], t: str, start: date, end: date) -> Optional[pd.DataFrame]:
    df = bars.get(t)
    if df is None or len(df) == 0:
        return None
    f = rules.align(df, rules.sessions_between(start, end))
    # Veri gecikmesi: son 1–2 seansın barı henüz yayımlanmamış olabilir. Bunları "eksik seans"
    # (lab: tut) değil "henüz gelmedi" say — giriş kalıcı atlanmasın, zaman çıkışı bir gün
    # önceki kapanışla yapılmasın; karar bir sonraki koşuya kalsın.
    k = 0
    while k < 2 and len(f) and pd.isna(f["Close"].iloc[-1]):
        f = f.iloc[:-1]
        k += 1
    return f


def _qualify_signal(src, t: str, frame: pd.DataFrame, signal_day: date) -> Optional[str]:
    """Havuz kontrolü (Bölüm 0). None = uygun; aksi halde ret nedeni."""
    idx = list(frame.index.date)
    if signal_day not in idx:
        return "NO_SIGNAL_BAR"
    ok, facts = rules.pool_member(frame, idx.index(signal_day), src.shares_outstanding(t))
    return None if ok else "NOT_IN_POOL"


# ── H5 ───────────────────────────────────────────────────────────────────────

def run_h5(ledger: ShadowLedger, src, universe: set, asof: date, now: datetime) -> Dict:
    sessions, d = [], asof
    while len(sessions) < H5_LOOKBACK_SESSIONS:
        sessions.append(d)
        d = prev_trading_day(d)
    sessions = sorted(sessions)
    rep = src.earnings_reporters(prev_trading_day(sessions[0]) - timedelta(days=4), asof)
    if rep is None or rep.empty:
        return dict(reporters=0, candidates=0, events=0)
    cand = [t for t in rep.index if t in universe]
    if "Marketcap" in rep.columns:
        mc = rep["Marketcap"]
        cand = [t for t in cand if pd.isna(mc.get(t)) or 1.5e8 <= float(mc.get(t)) <= 2e10]
    if "Surprise(%)" in rep.columns:
        # Gevşek ön filtre (aynı Yahoo verisi): takvim sürprizi ≥ %5 ya da henüz boş. Kesin
        # kural (≥ %10 vb.) lab'ın kaynağı get_earnings_dates ile aşağıda uygulanır. Yoğun
        # sezonda yüzlerce gereksiz Yahoo isteğini (rate-limit) önler.
        sp = rep["Surprise(%)"]
        cand = [t for t in cand if pd.isna(sp.get(t)) or float(sp.get(t)) >= 5.0]
    events = []
    for t in cand:
        try:
            rows = src.earnings_rows(t)
        except Exception as e:
            logger.debug("earnings_rows %s: %s", t, e)
            continue
        for x in rows:
            if not rules.h5_qualifies(x.get("surprise"), x.get("est"), x.get("rep")):
                continue
            R = rules.h5_reaction_session(x["ts"])
            if R in sessions:
                events.append(dict(ticker=t, R=R, surprise=float(x["surprise"]), ts=x["ts"]))
    added = 0
    if events:
        start = _start_before(sessions[0])
        bars = src.fetch_bars({e["ticker"] for e in events}, start, asof)
        for e in events:
            frame = _frame(bars, e["ticker"], start, asof)
            if frame is None:
                continue
            idx = list(frame.index.date)
            if e["R"] not in idx:
                continue
            i = idx.index(e["R"])
            c = frame["Close"]
            if not (i > 0 and c.iloc[i] > c.iloc[i - 1]):      # NaN karşılaştırması False
                continue                                         # pozitif tepki yok → olay değil
            why = _qualify_signal(src, e["ticker"], frame, e["R"])
            added += ledger.add_trade(dict(
                strategy="H5", ticker=e["ticker"], event_date=e["R"].isoformat(),
                signal_date=e["R"].isoformat(), decided_at=_utc_iso(now), key=e["surprise"],
                status="SKIPPED" if why else "PENDING", reason=why,
                late_decision=_late(now, e["R"]), warmup=0,
                details=dict(report_ts=e["ts"], surprise=e["surprise"]),
            ))
    return dict(reporters=int(len(rep)), candidates=len(cand), events=len(events), added=added)


# ── H4 ───────────────────────────────────────────────────────────────────────

def run_h4(ledger: ShadowLedger, src, universe: set, today_et: date, now: datetime,
           max_days: int = H4_MAX_DAYS_PER_RUN, deadline: float = None) -> Dict:
    obs = ledger.get_meta("h4_observation_start")
    if not obs:
        obs = today_et.isoformat()
        ledger.set_meta("h4_observation_start", obs)
    obs_d = date.fromisoformat(obs)
    done = set(ledger.get_meta("h4_processed", []) or [])
    todo, d = [], max(obs_d, today_et - timedelta(days=10))
    while d < today_et:                              # bugünün dosyalaması 22:00 ET'ye kadar sürer
        if d.weekday() < 5 and d.isoformat() not in done:
            todo.append(d)
        d += timedelta(days=1)
    out = dict(observation_start=obs, pending_days=[x.isoformat() for x in todo], processed=[])
    for F in todo[:max_days]:
        if deadline and _time.monotonic() > deadline:
            break
        try:
            filings = src.fetch_form4_filings(F)
        except src.SecIndexMissing:
            if (today_et - F).days > 4:              # EDGAR tatili → kalıcı olarak boş gün
                done.add(F.isoformat())
            continue
        errors = sum(1 for f in filings if "error" in f)
        rows = [r for f in filings if "error" not in f for r in rules.h4_purchase_rows(f)]
        ledger.add_purchases(rows)
        tickers = {r["ticker"] for r in rows if r["ticker"]}
        prior = {t: date.fromisoformat(v) for t, v in ledger.last_event_dates("H4").items() if v}
        # 61 gün: pencere içindeki önceki dosyalama günlerinin de tam (30 g) penceresi olsun.
        window = ledger.purchases(tickers, (F - timedelta(days=61)).isoformat(), F.isoformat())
        evs = [e for e in rules.h4_events(pd.DataFrame(window), prior, obs_d) if e["event_date"] == F]
        added = _record_h4(ledger, src, universe, evs, F, today_et, now)
        done.add(F.isoformat())
        ledger.set_meta("h4_processed", sorted(done)[-60:])
        out["processed"].append(dict(day=F.isoformat(), filings=len(filings), fetch_errors=errors,
                                     purchases=len(rows), events=len(evs), added=added))
    return out


def _record_h4(ledger, src, universe, evs, F: date, today_et: date, now: datetime) -> int:
    if not evs:
        return 0
    t_day = rules.last_session_on_or_before(F)
    asof = last_completed_session(now)
    start = _start_before(t_day)
    in_uni = [e for e in evs if e["ticker"] in universe]
    bars = src.fetch_bars({e["ticker"] for e in in_uni}, start, asof) if in_uni else {}
    added = 0
    for e in evs:
        why = None
        if e["ticker"] not in universe:
            why = "NOT_IN_UNIVERSE"
        else:
            frame = _frame(bars, e["ticker"], start, asof)
            why = "NO_PRICE_DATA" if frame is None else _qualify_signal(src, e["ticker"], frame, t_day)
        # Olay pool dışı olsa da kaydedilir: lab'daki 60 gün kuralı TÜM olayları sayar.
        added += ledger.add_trade(dict(
            strategy="H4", ticker=e["ticker"], event_date=F.isoformat(), signal_date=t_day.isoformat(),
            decided_at=_utc_iso(now), key=e["value"], status="SKIPPED" if why else "PENDING",
            reason=why, late_decision=_late(now, t_day), warmup=int(e["warmup"]),
            details=dict(insiders=e["insiders"], value=e["value"]),
        ))
    return added


# ── H6 "Sade VCE" (Ek F) ─────────────────────────────────────────────────────

def run_h6(ledger: ShadowLedger, src, asof: date, now: datetime, res: Optional[Dict] = None) -> Dict:
    """Canlı evren + canlı parçalarla VCE (tetikten sonraki katmanlar yok). Ayrı kural kilidi."""
    if ledger.get_meta("params_h6_sha256") is None:
        ledger.set_meta("params_h6_sha256", rules.PARAMS_H6_SHA256)
        ledger.set_meta("h6_live_start", asof.isoformat())
    if ledger.get_meta("params_h6_sha256") != rules.PARAMS_H6_SHA256:
        return dict(error="H6_PARAMS_CHANGED_WITHOUT_PREREGISTRATION")
    res = res if res is not None else src.live_vce_hits(asof)
    added = 0
    for h in res["hits"]:
        added += ledger.add_trade(dict(
            strategy="H6", ticker=h["ticker"], event_date=asof.isoformat(), signal_date=asof.isoformat(),
            decided_at=_utc_iso(now), key=float(h["dv20"]), status="PENDING",
            late_decision=_late(now, asof), warmup=0,
            details=dict(dv20=float(h["dv20"]), close=float(h["close"])),
        ))
    return dict(hits=len(res["hits"]), added=added, **res.get("stats", {}))


def _small_cap_regime(src, asof: date) -> Optional[Dict]:
    """Canlı ``regime_from_spy_close`` kuralları, IWM 1 yıllık kapanış + VIX ile. Son bar asof değilse None."""
    from ..small_cap.regime_logic import regime_from_spy_close
    bars = src.fetch_bars({"IWM", "^VIX"}, asof - timedelta(days=400), asof)
    iwm = bars.get("IWM")
    if iwm is None or len(iwm) == 0:
        return None
    iwm = iwm.dropna(subset=["Close"]).sort_values("Date")
    if pd.Timestamp(iwm["Date"].iloc[-1]).date() != asof:
        return None                                    # veri gecikmesi → karar sonraki koşuya
    vix = bars.get("^VIX")
    v = float(vix.dropna(subset=["Close"])["Close"].iloc[-1]) if vix is not None and len(vix) else None
    return regime_from_spy_close(iwm["Close"].tail(252).reset_index(drop=True), v)


def run_h6s(ledger: ShadowLedger, src, asof: date, now: datetime, res: Dict, name: str = "H6S") -> Dict:
    """Ek J2: H6 isabetleri + küçük hisse piyasa freni (BEAR giriş yok, CAUTION en çok 1 açık).

    ``name="H6SR"`` (Ek Q): aynı giriş ve fren, kendi defteri/kilidi; çıkış ``rules.RUNNERS``.
    """
    p = rules.PARAMS_H6S                                  # giriş + fren iki varyantta da aynı
    sha = {"H6S": rules.PARAMS_H6S_SHA256, "H6SR": rules.PARAMS_H6SR_SHA256}[name]
    key = name.lower()
    if ledger.get_meta(f"params_{key}_sha256") is None:
        ledger.set_meta(f"params_{key}_sha256", sha)
        ledger.set_meta(f"{key}_live_start", asof.isoformat())
    if ledger.get_meta(f"params_{key}_sha256") != sha:
        return dict(error=f"{name}_PARAMS_CHANGED_WITHOUT_PREREGISTRATION")
    hits = sorted(res.get("hits") or [], key=lambda h: -float(h["dv20"]))
    if not hits:
        return dict(hits=0, added=0)
    reg = _small_cap_regime(src, asof)
    if reg is None:
        return dict(hits=len(hits), deferred="IWM_BAR_MISSING")
    regime = str(reg.get("regime", "UNKNOWN")).upper()
    active = ledger.trades(name, statuses=("PENDING", "OPEN"))
    busy = {t["ticker"] for t in active}
    n_active, added, counts = len(active), 0, {}
    for h in hits:
        why = None
        if regime == "BEAR" and p["bear_block"]:
            why = "BEAR_REGIME"
        elif regime == "CAUTION" and n_active >= p["caution_max_concurrent"]:
            why = "CAUTION_CAPACITY"
        elif h["ticker"] in busy:
            why = "TICKER_BUSY"
        n = ledger.add_trade(dict(
            strategy=name, ticker=h["ticker"], event_date=asof.isoformat(), signal_date=asof.isoformat(),
            decided_at=_utc_iso(now), key=float(h["dv20"]), status="SKIPPED" if why else "PENDING",
            reason=why, late_decision=_late(now, asof), warmup=0,
            details=dict(dv20=float(h["dv20"]), regime=regime, regime_index="IWM"),
        ))
        added += n
        if n and not why:
            n_active += 1
            busy.add(h["ticker"])
        counts[why or "PENDING"] = counts.get(why or "PENDING", 0) + 1
    return dict(hits=len(hits), regime=regime, added=added, **counts)


# ── Açık işlemlerin güncellenmesi ────────────────────────────────────────────

def update_trades(ledger: ShadowLedger, src, asof: date) -> Dict:
    active = ledger.trades(statuses=("PENDING", "OPEN"))
    if not active:
        return dict(active=0)
    first = min(date.fromisoformat(t["signal_date"]) for t in active)
    start = _start_before(first)
    bars = src.fetch_bars({t["ticker"] for t in active} | {"IWM"}, start, asof)
    iwm = _frame(bars, "IWM", start, asof)
    all_rows = ledger.trades(statuses=("OPEN", "CLOSED"))
    busy: Dict[tuple, List[tuple]] = {}
    for t in all_rows:
        if t.get("entry_date"):
            busy.setdefault((t["strategy"], t["ticker"]), []).append(
                (t["id"], t["entry_date"], t.get("exit_date") or "9999-12-31"))
    counts = dict(active=len(active), opened=0, closed=0, skipped=0, pending=0)
    for t in sorted(active, key=lambda r: (r["signal_date"], -(r.get("key") or 0))):
        frame = _frame(bars, t["ticker"], start, asof)
        if frame is None:
            counts["pending"] += 1
            continue
        plan = rules.plan_trade(frame, date.fromisoformat(t["signal_date"]), t["strategy"])
        if plan["status"] == "PENDING":
            counts["pending"] += 1
            continue
        if plan["status"] == "SKIPPED":
            ledger.update_trade(t["id"], dict(status="SKIPPED", reason=plan["reason"]))
            counts["skipped"] += 1
            continue
        entry = plan["entry_day"].isoformat()
        if t["status"] == "PENDING":
            # lab simulate: hissede açık pozisyon varken (önceki çıkış ≥ yeni giriş) yeni giriş yok
            clash = [b for b in busy.get((t["strategy"], t["ticker"]), [])
                     if b[0] != t["id"] and b[1] <= entry <= b[2]]
            if clash:
                ledger.update_trade(t["id"], dict(status="SKIPPED", reason="TICKER_BUSY"))
                counts["skipped"] += 1
                continue
            counts["opened"] += 1
        path = _path(frame, plan)
        upd = dict(entry_date=entry, entry_price=round(plan["entry_price"], 4),
                   atr=None if plan["atr"] is None else round(plan["atr"], 4),
                   stop=None if plan["stop"] is None else round(plan["stop"], 4),       # H6: stop yok
                   bars=plan["bars"], mark_price=round(plan["mark_price"], 4),
                   mark_date=plan["mark_day"].isoformat(), ret_pct=round(plan["ret_pct"], 4),
                   details=dict(t.get("details") or {}, path=path))
        if plan["status"] == "CLOSED":
            upd.update(status="CLOSED", exit_date=plan["exit_day"].isoformat(),
                       exit_price=round(plan["exit_price"], 4), exit_reason=plan["exit_reason"],
                       net_pct=round(plan["ret_pct"] - rules.C["cost_pct"], 4),
                       iwm_ret=_iwm_ret(iwm, plan["entry_day"], plan["exit_day"]))
            counts["closed"] += 1
        else:
            upd.update(status="OPEN")
        ledger.update_trade(t["id"], upd)
        busy.setdefault((t["strategy"], t["ticker"]), []).append(
            (t["id"], entry, upd.get("exit_date") or "9999-12-31"))
    return counts


def _path(frame: pd.DataFrame, plan: Dict) -> List[list]:
    """Portföy değerlemesi için günlük kapanış yolu; çıkış günü çıkış fiyatıyla."""
    seg = frame["Close"].loc[pd.Timestamp(plan["entry_day"]):pd.Timestamp(plan["mark_day"])]
    path = [[d.date().isoformat(), float(c)] for d, c in seg.items() if np.isfinite(c)]
    if plan["status"] == "CLOSED" and path:
        path[-1][1] = float(plan["exit_price"])
    return path


def _iwm_ret(iwm: Optional[pd.DataFrame], entry: date, exit_: date) -> Optional[float]:
    if iwm is None:
        return None
    try:
        o = float(iwm.loc[pd.Timestamp(entry), "Open"])
        c = float(iwm.loc[pd.Timestamp(exit_), "Close"])
        return round(100 * (c / o - 1), 4) if np.isfinite(o) and np.isfinite(c) and o > 0 else None
    except KeyError:
        return None


# ── Rapor ───────────────────────────────────────────────────────────────────

def build_report(ledger: ShadowLedger, src, asof: date) -> Dict:
    live_start = date.fromisoformat(ledger.get_meta("live_start"))
    sessions = rules.sessions_between(live_start, asof)
    iwm_bh = None
    try:
        bars = src.fetch_bars({"IWM"}, live_start - timedelta(days=7), asof)
        f = _frame(bars, "IWM", live_start, asof)
        if f is not None and f["Open"].notna().any():
            iwm_bh = 100 * (float(f["Close"].dropna().iloc[-1]) / float(f["Open"].dropna().iloc[0]) - 1)
    except Exception:
        logger.exception("IWM buy-hold failed")
    trades = ledger.trades()
    strategies = {s: strategy_report(s, trades, sessions, live_start, asof, iwm_bh) for s in ("H5", "H4")}
    h6_start = ledger.get_meta("h6_live_start")
    h6s_start = ledger.get_meta("h6s_live_start")
    if h6s_start:
        d6s = date.fromisoformat(h6s_start)
        strategies["H6S"] = strategy_report("H6S", trades, rules.sessions_between(d6s, asof), d6s, asof,
                                            iwm_bh if d6s == live_start else _buy_hold(src, "IWM", d6s, asof))
    h6sr_start = ledger.get_meta("h6sr_live_start")
    if h6sr_start:                                        # Ek Q: kazananı koştur, H6S ile yan yana
        dr = date.fromisoformat(h6sr_start)
        strategies["H6SR"] = strategy_report("H6SR", trades, rules.sessions_between(dr, asof), dr, asof,
                                             iwm_bh if dr == live_start else _buy_hold(src, "IWM", dr, asof))
    champion = None
    if h6_start:
        h6_d = date.fromisoformat(h6_start)
        h6_bh = _buy_hold(src, "IWM", h6_d, asof) if h6_d != live_start else iwm_bh
        strategies["H6"] = strategy_report("H6", trades, rules.sessions_between(h6_d, asof), h6_d, asof, h6_bh)
        try:   # aynı dönemde canlı ürünün kendi paper işlemleri, aynı 50 bps maliyetle
            rows = src.champion_trades(h6_d)
            # Ek H ileri ölçümü: her işlem giriş öncesi seyreltme başvurusuna göre BİR KEZ işaretlenir.
            flags = ledger.get_meta("champion_dilution", {}) or {}
            new = [r for r in rows if str(r.get("id")) not in flags]
            if new:
                try:
                    flags.update(src.dilution_flags(new))
                    ledger.set_meta("champion_dilution", flags)
                except Exception:
                    logger.exception("Champion dilution flags failed")
            champion = champion_report(rows, since=h6_start, flags=flags)
        except Exception:
            logger.exception("Champion report failed")
    return dict(asof=asof.isoformat(), live_start=live_start.isoformat(),
                params_sha256=rules.PARAMS_SHA256, params_h6_sha256=rules.PARAMS_H6_SHA256,
                strategies=strategies, champion=champion)


def _buy_hold(src, ticker: str, start: date, asof: date) -> Optional[float]:
    try:
        bars = src.fetch_bars({ticker}, start - timedelta(days=7), asof)
        f = _frame(bars, ticker, start, asof)
        if f is not None and f["Open"].notna().any():
            return 100 * (float(f["Close"].dropna().iloc[-1]) / float(f["Open"].dropna().iloc[0]) - 1)
    except Exception:
        logger.exception("%s buy-hold failed", ticker)
    return None


# ── Giriş noktası ───────────────────────────────────────────────────────────

def run_shadow_daily(now: Optional[datetime] = None, ledger: Optional[ShadowLedger] = None,
                     src=None, time_budget_s: int = 780) -> Dict:
    if src is None:
        from . import sources as src
    ledger = ledger or ShadowLedger()
    now = (now or datetime.now(tz=NY)).astimezone(NY)
    deadline = _time.monotonic() + time_budget_s
    asof = last_completed_session(now)
    today_et = now.date()
    if not ledger.get_meta("live_start"):
        ledger.set_meta("live_start", asof.isoformat())
        ledger.set_meta("params_sha256", rules.PARAMS_SHA256)
    out: Dict = dict(asof=asof.isoformat(), decided_at=_utc_iso(now), params_sha256=rules.PARAMS_SHA256)
    if ledger.get_meta("params_sha256") != rules.PARAMS_SHA256:
        # Kurallar ön kayıttan saptıysa yeni sinyal ÜRETME (yalnız açıkları güncelle).
        out["error"] = "PARAMS_CHANGED_WITHOUT_PREREGISTRATION"
        universe = None
    else:
        try:
            universe = src.candidate_universe()
        except Exception as e:
            logger.exception("Shadow: universe fetch failed")
            out["universe_error"] = str(e)
            universe = None
    vce = None
    try:                                  # H6 kendi kilidiyle; aday evrenine (CANDIDATE) muhtaç değil
        vce = src.live_vce_hits(asof)
        out["H6"] = run_h6(ledger, src, asof, now, vce)
    except Exception as e:
        logger.exception("Shadow H6 failed")
        out["H6"] = dict(error=str(e))
    if vce is not None:
        try:                              # aynı isabetler + küçük hisse piyasa freni (Ek J2)
            out["H6S"] = run_h6s(ledger, src, asof, now, vce)
        except Exception as e:
            logger.exception("Shadow H6S failed")
            out["H6S"] = dict(error=str(e))
        try:                              # Ek Q: aynı isabetler + aynı fren, kazananı koşturan çıkış
            out["H6SR"] = run_h6s(ledger, src, asof, now, vce, name="H6SR")
        except Exception as e:
            logger.exception("Shadow H6SR failed")
            out["H6SR"] = dict(error=str(e))
    if universe:
        for name, fn in (("H5", lambda: run_h5(ledger, src, universe, asof, now)),
                         ("H4", lambda: run_h4(ledger, src, universe, today_et, now, deadline=deadline))):
            try:
                out[name] = fn()
            except Exception as e:
                logger.exception("Shadow %s failed", name)
                out[name] = dict(error=str(e))
    try:
        out["update"] = update_trades(ledger, src, asof)
    except Exception as e:
        logger.exception("Shadow update failed")
        out["update"] = dict(error=str(e))
    try:
        report = build_report(ledger, src, asof)
        ledger.set_meta("report", report)
        out["gate"] = {s: r["gate"]["status"] for s, r in report["strategies"].items()}
    except Exception as e:
        logger.exception("Shadow report failed")
        out["report_error"] = str(e)
    ledger.set_meta("last_run", out)
    return out
