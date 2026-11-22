"""İşlem maliyeti — paper P/L'in gerçek paraya çevrilmesi.

İki kalem, ikisi de ayardan:
  * ``slippage_bps_per_side`` — spread + kayma, tutarla orantılı (her bacakta).
  * ``commission_usd_per_side`` — aracı kurumun sabit emir ücreti (Midas ABD: 1.5$/emir). Tutara bağlı
    değildir: 10.000$ hesapta 1.000$'lık pozisyonda gidiş-dönüş %0.30, 500$'lık pozisyonda %0.60.

Paper tracker fiyatları maliyetsiz kaydeder (yalnız stop'ta gap kayması var). Backtest aynı ayarları
her bacakta düşer; ekrandaki net P/L ve H6S hesabının bileşik özsermayesi de bunları kullanır.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple


def _settings(settings):
    if settings is None:
        from ..small_cap.settings_config import load_settings
        settings = load_settings()
    return settings


def round_trip_cost_pct(settings=None) -> float:
    """Gidiş-dönüş spread/kayma maliyeti, yüzde puan (ör. 10 bps/kenar → 0.20)."""
    return 2.0 * float(_settings(settings).slippage_bps_per_side) / 100.0


def commission_usd(settings=None) -> float:
    """Emir (kenar) başına sabit aracı kurum ücreti, $."""
    return float(getattr(_settings(settings), "commission_usd_per_side", 0.0) or 0.0)


def order_legs(trade: Dict) -> int:
    """Alış + satış (+ kısmi satış varsa bir emir daha)."""
    return 3 if trade.get("partial_exit_price") else 2


def net_of_cost(trade: Dict, cost_pct: float, fee_usd: float = 0.0) -> Tuple[Optional[float], Optional[float]]:
    """(net $, net %) — kapanmış işlemin maliyet sonrası sonucu. Veri yoksa (None, None).

    Yüzde: realized_pnl_pct − gidiş-dönüş maliyet − sabit ücretlerin giriş tutarına oranı. Dolar: giriş
    tutarı (giriş × lot) üzerinden aynı oran + emir sayısı × sabit ücret.
    """
    pct = trade.get("realized_pnl_pct")
    pnl = trade.get("realized_pnl")
    if pct is None or pnl is None:
        return None, None
    notional = float(trade.get("entry_price") or 0) * float(trade.get("position_size") or 0)
    fees = order_legs(trade) * float(fee_usd)
    fee_pct = 100.0 * fees / notional if notional > 0 else 0.0
    return round(float(pnl) - notional * cost_pct / 100.0 - fees, 2), round(float(pct) - cost_pct - fee_pct, 2)


def account_equity(start: float, closed_trades: Iterable[Dict], cost_pct: float, fee_usd: float = 0.0) -> float:
    """Nakit hesabın özsermayesi = başlangıç + kapanmış işlemlerin maliyet sonrası $ sonucu.

    Gerçek Midas hesabı böyle büyür/küçülür (Lab v3 Ek M: lab portföyü bileşikti, canlı sabit tutardan
    boyutluyordu). Açık pozisyonların gerçekleşmemiş kârı sayılmaz — temkinli, nakit hesabı gibi.
    """
    total = float(start)
    for t in closed_trades:
        if t.get("status") in ("OPEN", "PENDING", "REJECTED"):
            continue
        n_usd, _ = net_of_cost(t, cost_pct, fee_usd)
        if n_usd is not None:
            total += n_usd
    return round(total, 2)
