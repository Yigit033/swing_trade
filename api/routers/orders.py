"""Midas Emirleri — H6S paper işlemlerinden bugünün uygulanabilir emir listesi.

Kâğıttaki avantajın gerçek paraya dönüşmesindeki en büyük risk uygulama hatasıdır (yanlış lot,
unutulan stop, 8. günde satmayı kaçırmak). Bu sayfa her gün ne yapılacağını kesin olarak verir:

    AL   — PENDING H6S: bugünün (ya da ilk seansın) açılışında, adet + koruyucu stop mesafesi
    SAT  — OPEN H6S, bugün 8. seansı: kapanışta sat (MOC)
    TUT  — diğer OPEN H6S: geçerli stop

Kurallar H6S ile birebir (Lab v3 Ek J2): stop = alış − 2.5×ATR14 (sinyal günü), giriş günü 1. seans
sayılarak 8. seans kapanışında çıkış. Strateji kodu değişmez; bu yalnız okuma + tarih hesabı.
"""
from __future__ import annotations

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


def build_orders(trades: List[Dict], today: date, account: float) -> Dict:
    from swing_trader.paper_trading.storage import strategy_of
    from swing_trader.shadow.rules import PARAMS_H6S
    from swing_trader.utils.market_calendar import is_trading_day, next_trading_day

    hold, k = int(PARAMS_H6S["hold_sessions"]), float(PARAMS_H6S["stop_atr"])
    session = today if is_trading_day(today) else next_trading_day(today)
    buy, sell, hold_rows, held = [], [], [], 0.0
    for t in trades:
        if strategy_of(t) != "H6S" or t.get("status") not in ("PENDING", "OPEN"):
            continue
        size = int(t.get("position_size") or 0)
        atr = float(t.get("atr") or 0)
        if t["status"] == "PENDING":
            sig = _ny_day(t.get("entry_date"))
            entry = next_trading_day(sig) if sig else session
            price = float(t.get("signal_price") or t.get("entry_price") or 0)
            held += size * price
            buy.append(dict(ticker=t["ticker"], shares=size, est_price=round(price, 2),
                            est_cost=round(size * price, 2), stop_distance=round(k * atr, 2),
                            est_stop=round(price - k * atr, 2), buy_day=entry.isoformat(),
                            sell_day=_nth_session(entry, hold).isoformat(),
                            overdue=entry < session, trade_id=t.get("id")))
            continue
        entry = _ny_day(t.get("entry_date")) or session
        exit_day = _nth_session(entry, hold)
        price = float(t.get("entry_price") or 0)
        held += size * price
        row = dict(ticker=t["ticker"], shares=size, entry_price=round(price, 2),
                   stop=round(float(t.get("stop_loss") or 0), 2), entry_day=entry.isoformat(),
                   sell_day=exit_day.isoformat(), current_price=t.get("current_price"),
                   overdue=exit_day < session, trade_id=t.get("id"))
        (sell if exit_day <= session else hold_rows).append(row)
    key = lambda r: r["ticker"]
    return dict(session=session.isoformat(), account=account, held=round(held, 2),
                free=round(account - held, 2), hold_sessions=hold, stop_atr=k,
                buy=sorted(buy, key=key), sell=sorted(sell, key=key), hold=sorted(hold_rows, key=key))


@router.get("", dependencies=[Depends(require_user)])
def get_orders():
    """Bugünün Midas emirleri (H6S hesabı, auto-scan sahibi). Giriş zorunlu."""
    from api.deps import get_paper_storage
    from swing_trader.small_cap.settings_config import load_settings
    rows = [t for t in (get_paper_storage().get_all_trades(None) or []) if t.get("user_id") is None]
    today = datetime.now(timezone.utc).astimezone(NY).date()
    return build_orders(rows, today, float(load_settings().auto_scan.portfolio_value))
