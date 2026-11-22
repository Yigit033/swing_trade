"""Canlı H6S hesabının risk kuralları (Lab v3 Ek L, ön kayıtlı ölçümle).

L2 — Kayıp serisi freni (EKLENDİ): kapanmış son 3 H6S işleminin hepsi 50 bps net zararlıysa yeni
pozisyon yarım boyut; ilk kârlı kapanışta tam boyuta dönülür. H6S lab replikasında portföy
+64.5 → +67.1%, max düşüş −31.5 → −29.9% (iki ölçüt de iyileşti).
L1 — Sektör sınırı (≤2) EKLENMEDİ: portföy +64.5 → −6.8% (sektör momentumunu kesiyor).
"""
from __future__ import annotations

from typing import Dict, Iterable

STREAK = 3
RULE_COST_PCT = 0.50          # ön kayıttaki tanım: 50 bps net (canlı maliyet ayarından bağımsız)


def loss_streak_active(closed_trades: Iterable[Dict], streak: int = STREAK, cost: float = RULE_COST_PCT) -> bool:
    """Hesabın kapanmış işlemleri (REJECTED hariç), çıkış tarihine göre: son ``streak`` hepsi zarar mı?"""
    rows = [t for t in closed_trades
            if t.get("status") not in ("REJECTED", "OPEN", "PENDING") and t.get("realized_pnl_pct") is not None]
    rows.sort(key=lambda t: str(t.get("exit_date") or ""))
    last = rows[-streak:]
    return len(last) == streak and all(float(t["realized_pnl_pct"]) - cost < 0 for t in last)
