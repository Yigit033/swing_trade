"""
data_collector.py — Haftalık Rapor İçin Veri Toplayıcı

HYBRID ARCHİTECTURE'ın ilk katmanı:
────────────────────────────────────
Bu dosya SQLite'dan ham veriyi alır ve düzenlenmiş,
yapılandırılmış bir context objesi üretir.

LLM bu context'i alır → insana anlatır.
Hiçbir hesaplama LLM tarafından yapılmaz, sadece okur.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class WeeklyDataCollector:
    """
    Son N günün trade verilerini toplar ve özetler.
    
    Çıktı:
    {
        "period": {"start": "2024-01-15", "end": "2024-01-21"},
        "summary": {"total": 8, "wins": 5, "losses": 3, "win_rate": 62.5, ...},
        "trades": [...],           # Her trade'in detayı
        "by_swing_type": {...},    # Tip bazında performans
        "all_time_stats": {...},   # Tüm zamanların özeti
        "top_win": {...},          # En iyi trade
        "top_loss": {...},         # En kötü trade
    }
    """

    def __init__(self, storage, days: int = 7):
        """
        Args:
            storage: PaperTradeStorage instance
            days: Kaç günlük veri? (default: 7 = haftalık)
        """
        self.storage = storage
        self.days = days

    def collect(self) -> Dict:
        """
        Tüm veriyi topla ve yapılandırılmış context döndür.
        
        Returns:
            context dict — reporter.py bunu prompt'a dönüştürür
        """
        all_closed = self.storage.get_closed_trades(limit=9999)

        # Geçerli tradeleri filtrele (REJECTED olanları atla)
        valid_trades = [
            t for t in all_closed
            if t.get("status") not in ("REJECTED", "PENDING")
        ]

        # Haftalık tradeleri filtrele
        weekly_trades = self._filter_weekly(valid_trades)

        context = {
            "period": self._period_string(),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "weekly_trades": self._format_trades(weekly_trades),
            "weekly_summary": self._compute_summary(weekly_trades),
            "all_time_summary": self._compute_summary(valid_trades),
            "by_swing_type": self._compute_by_type(valid_trades),
            "top_win": self._get_extreme(valid_trades, mode="win"),
            "top_loss": self._get_extreme(valid_trades, mode="loss"),
            "total_trade_count": len(valid_trades),
        }
        return context

    # ─────────────────────────────────────────────
    # Yardımcı Methodlar
    # ─────────────────────────────────────────────

    def _filter_weekly(self, trades: List[Dict]) -> List[Dict]:
        """Son N günde kapanan tradeleri döndür."""
        cutoff = (datetime.now() - timedelta(days=self.days)).strftime("%Y-%m-%d")
        result = []
        for t in trades:
            exit_date = t.get("exit_date", "") or ""
            if exit_date[:10] >= cutoff:
                result.append(t)
        return result

    def _period_string(self) -> Dict:
        """Dönem başlangıç/bitiş tarihleri."""
        end   = datetime.now()
        start = end - timedelta(days=self.days)
        return {
            "start": start.strftime("%Y-%m-%d"),
            "end":   end.strftime("%Y-%m-%d"),
            "label": f"{self.days} günlük dönem",
        }

    def _format_trades(self, trades: List[Dict]) -> List[Dict]:
        """Trade listesini okunabilir formata dönüştür."""
        result = []
        for t in trades:
            entry = t.get("entry_price", 0) or 0
            exit_p = t.get("exit_price", 0) or 0
            pnl_pct = t.get("realized_pnl_pct", 0) or 0
            atr = t.get("atr", 0) or 0
            risk_pct = abs((entry - (t.get("stop_loss") or entry)) / entry * 100) if entry else 0
            reward_pct = abs((t.get("target", exit_p) or exit_p) - entry) / entry * 100 if entry else 0
            rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0

            result.append({
                "ticker":      t.get("ticker", "?"),
                "status":      t.get("status", "?"),
                "outcome":     "WIN" if pnl_pct > 0 else "LOSS",
                "swing_type":  t.get("swing_type", "?"),
                "entry_price": round(entry, 2),
                "exit_price":  round(exit_p, 2),
                "pnl_pct":     round(pnl_pct, 2),
                "pnl_dollar":  round(t.get("realized_pnl", 0) or 0, 2),
                "quality":     t.get("quality_score", 0),
                "rr_ratio":    round(rr_ratio, 2),
                "entry_date":  (t.get("entry_date", "") or "")[:10],
                "exit_date":   (t.get("exit_date", "") or "")[:10],
            })

        return sorted(result, key=lambda x: x["pnl_pct"], reverse=True)

    def _compute_summary(self, trades: List[Dict]) -> Dict:
        """İstatistik özeti hesapla."""
        if not trades:
            return {
                "total": 0, "wins": 0, "losses": 0,
                "win_rate": 0, "avg_pnl_pct": 0,
                "total_pnl_pct": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
                "profit_factor": 0,
            }

        pnl_pcts = [t.get("realized_pnl_pct", 0) or 0 for t in trades]
        wins  = [p for p in pnl_pcts if p > 0]
        losses = [p for p in pnl_pcts if p <= 0]

        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))

        return {
            "total":        len(trades),
            "wins":         len(wins),
            "losses":       len(losses),
            "win_rate":     round(len(wins) / len(trades) * 100, 1),
            "avg_pnl_pct":  round(sum(pnl_pcts) / len(pnl_pcts), 2),
            "total_pnl_pct": round(sum(pnl_pcts), 2),
            "avg_win_pct":  round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss_pct": round(sum(losses) / len(losses), 2) if losses else 0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        }

    def _compute_by_type(self, trades: List[Dict]) -> Dict:
        """Swing tipi bazında performans."""
        result = {}
        for t in trades:
            st = t.get("swing_type", "?") or "?"
            pnl = t.get("realized_pnl_pct", 0) or 0
            if st not in result:
                result[st] = {"count": 0, "wins": 0, "total_pnl": 0.0}
            result[st]["count"] += 1
            result[st]["total_pnl"] = round(result[st]["total_pnl"] + pnl, 2)
            if pnl > 0:
                result[st]["wins"] += 1

        for st, data in result.items():
            data["win_rate"] = round(data["wins"] / data["count"] * 100, 1)
            data["avg_pnl"]  = round(data["total_pnl"] / data["count"], 2)

        return result

    def _get_extreme(self, trades: List[Dict], mode: str) -> Optional[Dict]:
        """En iyi veya en kötü trade."""
        valid = [t for t in trades if t.get("realized_pnl_pct") is not None]
        if not valid:
            return None

        if mode == "win":
            best = max(valid, key=lambda x: x.get("realized_pnl_pct", 0))
        else:
            best = min(valid, key=lambda x: x.get("realized_pnl_pct", 0))

        return {
            "ticker":  best.get("ticker", "?"),
            "status":  best.get("status", "?"),
            "pnl_pct": round(best.get("realized_pnl_pct", 0), 2),
            "swing_type": best.get("swing_type", "?"),
            "exit_date": (best.get("exit_date", "") or "")[:10],
        }
