"""
data_collector.py — GenAI Context Toplayıcı

HYBRID ARCHİTECTURE'ın ilk katmanı:
────────────────────────────────────
Bu dosya Supabase/PostgreSQL'den ham veriyi alır ve düzenlenmiş,
yapılandırılmış bir context objesi üretir.

LLM bu context'i alır → insana anlatır.
Hiçbir hesaplama LLM tarafından yapılmaz, sadece okur.

v2.0 İyileştirmeler:
  - Açık pozisyonlar (open trades) eklendi
  - Pending trade'ler eklendi
  - Market regime bilgisi eklendi
  - Docstring ve yorumlar güncellenip doğrulandı
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class WeeklyDataCollector:
    """
    Trade verilerini toplar ve LLM'e gönderilecek context'i oluşturur.

    Çıktı:
    {
        "period": {"start": "2024-01-15", "end": "2024-01-21"},
        "summary": {"total": 8, "wins": 5, "losses": 3, "win_rate": 62.5, ...},
        "trades": [...],            # Her trade'in detayı
        "by_swing_type": {...},     # Tip bazında performans
        "all_time_stats": {...},    # Tüm zamanların özeti
        "top_win": {...},           # En iyi trade
        "top_loss": {...},          # En kötü trade
        "open_positions": [...],    # Şu an açık pozisyonlar
        "market_regime": {...},     # Piyasa durumu (BULL/BEAR/CAUTION)
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
            context dict — reporter.py ve strategy_chat.py bunu prompt'a dönüştürür
        """
        all_closed = self.storage.get_closed_trades(limit=9999)

        # Geçerli tradeleri filtrele (REJECTED olanları atla)
        valid_trades = [
            t for t in all_closed
            if t.get("status") not in ("REJECTED", "PENDING")
        ]

        # Haftalık tradeleri filtrele
        weekly_trades = self._filter_weekly(valid_trades)

        # ── Açık pozisyonları çek ──────────────────────────────────────
        open_positions = self._collect_open_positions()

        # ── Market Regime bilgisini çek ────────────────────────────────
        market_regime = self._collect_market_regime()

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
            # v2.0: Yeni context alanları
            "open_positions": open_positions,
            "market_regime": market_regime,
        }
        return context

    # ─────────────────────────────────────────────
    # Yeni: Açık Pozisyonlar
    # ─────────────────────────────────────────────

    def _collect_open_positions(self) -> List[Dict]:
        """
        Şu an OPEN ve PENDING statüsündeki trade'leri çeker ve formatlar.

        Bu veri LLM'e gönderildiğinde kullanıcı "Açık pozisyonlarım nasıl?"
        gibi soruları cevaplayabilir hale gelir.
        """
        try:
            raw = self.storage.get_open_trades()
        except Exception as e:
            logger.warning(f"Açık pozisyonlar çekilemedi: {e}")
            return []

        positions = []
        for t in raw:
            entry = float(t.get("entry_price") or 0)
            current = float(t.get("current_price") or entry)
            unrealized_pct = float(t.get("unrealized_pnl_pct") or 0)
            stop = float(t.get("stop_loss") or 0)
            target = float(t.get("target") or 0)
            status = t.get("status", "OPEN")

            positions.append({
                "ticker": t.get("ticker", "?"),
                "status": status,
                "swing_type": t.get("swing_type", "?"),
                "entry_price": round(entry, 2),
                "current_price": round(current, 2),
                "unrealized_pnl_pct": round(unrealized_pct, 2),
                "stop_loss": round(stop, 2),
                "target": round(target, 2),
                "entry_date": (t.get("entry_date", "") or "")[:10],
                "quality_score": t.get("quality_score", 0),
            })

        return positions

    # ─────────────────────────────────────────────
    # Yeni: Market Regime
    # ─────────────────────────────────────────────

    def _collect_market_regime(self) -> Dict:
        """
        Mevcut piyasa rejimini çeker (BULL / CAUTION / BEAR / UNKNOWN).

        Regime bilgisi, LLM'in "Piyasa şu an nasıl?" gibi sorulara
        cevap verebilmesi ve analizlerini piyasa koşullarına göre
        çerçevelemesi için kritiktir.
        """
        try:
            from swing_trader.small_cap.regime_logic import regime_from_spy_close
            import yfinance as yf

            spy = yf.Ticker("SPY")
            hist = spy.history(period="6mo")
            if hist.empty:
                return {"regime": "UNKNOWN", "confidence": "TENTATIVE", "error": "SPY verisi alınamadı"}

            # regime_from_spy_close bir pd.Series bekliyor (sadece Close sütunu)
            close_series = hist["Close"]

            # VIX verisini de çek (regime kararında etkili)
            vix_val = None
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="5d")
                if not vix_hist.empty:
                    vix_val = float(vix_hist["Close"].iloc[-1])
            except Exception:
                pass  # VIX alınamazsa None kalır, regime yine hesaplanır

            result = regime_from_spy_close(close_series, vix_last=vix_val)
            return {
                "regime": result.get("regime", "UNKNOWN"),
                "confidence": result.get("confidence", "TENTATIVE"),
                "spy_price": round(result.get("spy_price", 0), 2),
                "vix": round(result.get("vix", 0), 1) if result.get("vix") else None,
            }
        except Exception as e:
            logger.warning(f"Market regime çekilemedi: {e}")
            return {"regime": "UNKNOWN", "confidence": "TENTATIVE", "error": str(e)[:100]}

    # ─────────────────────────────────────────────
    # Mevcut Yardımcı Methodlar
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

        # BUG FIX: P/L'ye gore degil, tarihe gore sirala (en yeniden en eskiye)
        return sorted(result, key=lambda x: x["exit_date"], reverse=True)

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
