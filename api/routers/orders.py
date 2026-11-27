"""Midas Emirleri — paper işlemlerinden bugünün uygulanabilir emir listesi.

Kâğıttaki sonucun gerçek paraya dönüşmesindeki en büyük risk uygulama hatasıdır (yanlış lot, unutulan stop,
satışı kaçırmak). Bu sayfa her gün ne yapılacağını kesin olarak verir. Ana kural GRW "büyüme + kırılım"
(Lab v3 Ek W; 2026-11-28'den beri):

    AL   — PENDING GRW: ilk seansın açılışında; açılış kırılım gününün dibinde ya da altındaysa ALINMAZ.
           Stop = max(kırılım gününün dibi, alış − %8) — alır almaz girilir.
    SAT  — OPEN GRW, son kapanışı 50 günlük ortalamanın altında: bugün açılışta sat.
    TUT  — diğer OPEN GRW: geçerli stop; süre sınırı ve kâr hedefi yok.

H6S (2026-11-28'e kadar ana kuraldı) yeni pozisyon açmaz; açık H6S pozisyonları kendi kuralıyla (stop =
alış − 2.5×ATR14, 8. seans kapanışında çıkış) ``legacy_h6s`` bloğunda kapanana kadar listelenir.
Strateji kodu değişmez; bu yalnız okuma + tarih hesabı.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends

from api.auth import require_user

router = APIRouter(prefix="/api/orders", tags=["orders"])
NY = ZoneInfo("America/New_York")


def _ny_day(value: str) -> Optional[date]:
    """entry_date: PENDING 'YYYY-MM-DD HH:MM' (sinyal günü) | OPEN ISO UTC (açılış anı)."""
    if not value:
        return None
    v = str(value)
    if "T" in v:
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(NY).date()
        except ValueError:
            return None
    try:
        return date.fromisoformat(v[:10])
    except ValueError:
        return None


def _nth_session(start: date, n: int) -> date:
    """Giriş günü 1. seans olmak üzere n. seans."""
    from swing_trader.utils.market_calendar import next_trading_day
    d = start
    for _ in range(n - 1):
        d = next_trading_day(d)
    return d


def _sell_next(trade: Dict) -> bool:
    """GRW: çıkış denetimi son kapanışı 50 günlük ortalamanın altında buldu → bugün açılışta satılacak."""
    try:
        return bool(json.loads(trade.get("execution_state") or "{}").get("sell_next"))
    except (ValueError, TypeError, AttributeError):
        return False


def build_orders(trades: List[Dict], today: date, account: float, cost_pct: float = 0.0, fee: float = 0.0,
                 strategy: str = "GRW") -> Dict:
    """Tek stratejinin planı. account = başlangıç tutarı; özsermaye = başlangıç + o stratejinin kapanmış
    işlemlerinin net sonucu (Ek M)."""
    from swing_trader.paper_trading.costs import account_equity
    from swing_trader.paper_trading.risk_rules import loss_streak_active
    from swing_trader.paper_trading.storage import strategy_of
    from swing_trader.shadow.rules import PARAMS_GRW, PARAMS_H6S
    from swing_trader.utils.market_calendar import is_trading_day, next_trading_day

    grw = strategy == "GRW"
    hold, k = int(PARAMS_H6S["hold_sessions"]), float(PARAMS_H6S["stop_atr"])
    max_stop = float(PARAMS_GRW["max_stop"])
    session = today if is_trading_day(today) else next_trading_day(today)
    buy, sell, hold_rows, held = [], [], [], 0.0
    for t in trades:
        if strategy_of(t) != strategy or t.get("status") not in ("PENDING", "OPEN"):
            continue
        size = int(t.get("position_size") or 0)
        if t["status"] == "PENDING":
            sig = _ny_day(t.get("entry_date"))
            entry = next_trading_day(sig) if sig else session
            price = float(t.get("signal_price") or t.get("entry_price") or 0)
            held += size * price
            row = dict(ticker=t["ticker"], shares=size, est_price=round(price, 2), est_cost=round(size * price, 2),
                       buy_day=entry.isoformat(), overdue=entry < session, trade_id=t.get("id"))
            if grw:
                low = float(t.get("initial_stop") or 0)          # PENDING: kırılım gününün dibi
                est = max(low, (1 - max_stop) * price)
                row.update(signal_low=round(low, 2), est_stop=round(est, 2), stop_distance=round(price - est, 2),
                           sell_day=None)
            else:
                atr = float(t.get("atr") or 0)
                row.update(stop_distance=round(k * atr, 2), est_stop=round(price - k * atr, 2),
                           sell_day=_nth_session(entry, hold).isoformat())
            buy.append(row)
            continue
        entry = _ny_day(t.get("entry_date")) or session
        price = float(t.get("entry_price") or 0)
        held += size * price
        row = dict(ticker=t["ticker"], shares=size, entry_price=round(price, 2),
                   stop=round(float(t.get("stop_loss") or 0), 2), entry_day=entry.isoformat(),
                   current_price=t.get("current_price"), trade_id=t.get("id"))
        if grw:
            out = _sell_next(t)
            row.update(sell_day=session.isoformat() if out else None, overdue=False)
            (sell if out else hold_rows).append(row)
        else:
            exit_day = _nth_session(entry, hold)
            row.update(sell_day=exit_day.isoformat(), overdue=exit_day < session)
            (sell if exit_day <= session else hold_rows).append(row)
    mine = [t for t in trades if strategy_of(t) == strategy]
    brake = (loss_streak_active(mine, streak=int(PARAMS_GRW["cold_streak"]), cost=float(PARAMS_GRW["cost_pct"]))
             if grw else loss_streak_active(mine))
    equity = account_equity(account, mine, cost_pct, fee)
    free = round(equity - held, 2)
    key = lambda r: r["ticker"]
    out = dict(strategy=strategy, session=session.isoformat(), start=account, account=equity, held=round(held, 2),
               loss_streak_brake=brake, free=free, idle_pct=round(100 * free / equity, 1) if equity > 0 else 0.0,
               fee_per_order=fee, buy=sorted(buy, key=key), sell=sorted(sell, key=key),
               hold=sorted(hold_rows, key=key))
    if grw:
        risk = PARAMS_GRW["risk_cold"] if brake else PARAMS_GRW["risk"]
        out.update(max_stop_pct=round(100 * max_stop, 1), risk_pct=round(100 * float(risk), 2),
                   sma=int(PARAMS_GRW["sma"]))
    else:
        out.update(hold_sessions=hold, stop_atr=k)
    return out


@router.get("", dependencies=[Depends(require_user)])
def get_orders():
    """Bugünün Midas emirleri (GRW hesabı, auto-scan sahibi) + kapanana kadar eski H6S pozisyonları. Giriş zorunlu."""
    from api.deps import get_paper_storage
    from swing_trader.paper_trading.costs import commission_usd, round_trip_cost_pct
    from swing_trader.small_cap.settings_config import load_settings
    s = load_settings()
    rows = [t for t in (get_paper_storage().get_all_trades(None) or []) if t.get("user_id") is None]
    today = datetime.now(timezone.utc).astimezone(NY).date()
    args = (rows, today, float(s.auto_scan.portfolio_value), round_trip_cost_pct(s), commission_usd(s))
    out = build_orders(*args)
    old = build_orders(*args, strategy="H6S")
    out["legacy_h6s"] = old if (old["buy"] or old["sell"] or old["hold"]) else None
    return out
