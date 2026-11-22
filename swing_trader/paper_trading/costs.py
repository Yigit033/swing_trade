"""İşlem maliyeti — paper P/L'in gerçek paraya çevrilmesi (tek kaynak: ``slippage_bps_per_side``).

Paper tracker fiyatları maliyetsiz kaydeder (yalnız stop'ta gap kayması var). Gerçek alım-satımda
her bacakta spread + kayma ödenir; Lab v3'te canlı ürünün maliyet öncesi kârı tam bu kalemle
zarara dönüyordu. Backtest aynı ayarı her bacakta düşer; ekrandaki net P/L de bunu kullanır.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple


def round_trip_cost_pct(settings=None) -> float:
    """Gidiş-dönüş maliyet, yüzde puan (ör. 10 bps/kenar → 0.20)."""
    if settings is None:
        from ..small_cap.settings_config import load_settings
        settings = load_settings()
    return 2.0 * float(settings.slippage_bps_per_side) / 100.0


def net_of_cost(trade: Dict, cost_pct: float) -> Tuple[Optional[float], Optional[float]]:
    """(net $, net %) — kapanmış işlemin maliyet sonrası sonucu. Veri yoksa (None, None).

    Yüzde: realized_pnl_pct − gidiş-dönüş maliyet. Dolar: giriş tutarı (giriş × lot) üzerinden
    aynı oran; kısmi çıkışlı işlemde de toplam lot alınıp satılmıştır.
    """
    pct = trade.get("realized_pnl_pct")
    pnl = trade.get("realized_pnl")
    if pct is None or pnl is None:
        return None, None
    notional = float(trade.get("entry_price") or 0) * float(trade.get("position_size") or 0)
    return round(float(pnl) - notional * cost_pct / 100.0, 2), round(float(pct) - cost_pct, 2)
