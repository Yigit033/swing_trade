"""
reporter.py — Haftalık Performans Raporu Orchestrator

Bu dosya tüm parçaları bir araya getirir:
  1. DataCollector ile veriyi al
  2. Prompt'u oluştur
  3. LLM'e gönder
  4. Cevabı önbelleğe al (cache) — her page refresh'te API'a gitmez
  5. Sonucu döndür

Cache stratejisi:
  Rapor bir kez üretilince aynı gün içinde tekrar API çağrısı yapılmaz.
  "Raporu Yenile" butonuna basınca cache temizlenir.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .data_collector import WeeklyDataCollector
from .llm_client import LLMClient
from .prompts import SYSTEM_PROMPT, build_weekly_report_prompt

logger = logging.getLogger(__name__)

# Cache dizini
CACHE_DIR  = Path(__file__).parent.parent.parent / "data" / "genai_cache"
CACHE_FILE = CACHE_DIR / "weekly_report.json"


class WeeklyReporter:
    """
    Haftalık performans raporu oluşturucu.
    
    Kullanım:
        reporter = WeeklyReporter(storage)
        result = reporter.generate()
        
        if result["success"]:
            print(result["report"])        # Markdown rapor
            print(result["context"])       # Ham istatistikler  
        else:
            print(result["error"])         # Hata mesajı
    """

    def __init__(self, storage, days: int = 7, llm_provider: Optional[str] = None):
        """
        Args:
            storage: PaperTradeStorage instance
            days: Kaç günlük dönemi analiz et
            llm_provider: "openai" | "gemini" | None (.env'den okur)
        """
        self.storage  = storage
        self.days     = days
        self.collector = WeeklyDataCollector(storage, days)
        self.client    = LLMClient(provider=llm_provider)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────
    # Ana Metod
    # ─────────────────────────────────────────────────────

    def generate(self, force_refresh: bool = False) -> Dict:
        """
        Haftalık raporu üret veya önbellekten döndür.
        
        Args:
            force_refresh: True ise önbelleği yok say, yeniden üret
        
        Returns:
            {
                "success": bool,
                "report": str,          # Markdown rapor
                "context": dict,        # Ham istatistikler (deterministik)
                "from_cache": bool,
                "generated_at": str,
                "llm_available": bool,
                "error": str,           # sadece success=False'da
            }
        """
        # 1. Deterministik veriyi her zaman topla
        context = self.collector.collect()

        # 2. Önbellekte geçerli rapor var mı?
        if not force_refresh:
            cached = self._load_cache()
            if cached:
                logger.info("Rapor önbellekten yüklendi")
                return {
                    "success": True,
                    "report": cached["report"],
                    "context": context,         # Güncel istatistikler
                    "from_cache": True,
                    "generated_at": cached.get("generated_at", "?"),
                    "llm_available": self.client.is_ready(),
                }

        # 3. LLM müsait değilse → sadece istatistik raporu döndür
        if not self.client.is_ready():
            logger.info("LLM müsait değil — istatistik raporu döndürülüyor")
            return {
                "success": True,
                "report": self._build_stats_only_report(context),
                "context": context,
                "from_cache": False,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "llm_available": False,
            }

        # 4. LLM raporu üret
        try:
            prompt  = build_weekly_report_prompt(context)
            report  = self.client.complete(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=1500,
                temperature=0.5,
            )

            if not report:
                raise ValueError("LLM boş cevap döndürdü")

            # Header ekle
            period = context.get("period", {})
            header = (
                f"# 🤖 AI Haftalık Performans Raporu\n"
                f"*{period.get('start', '?')} — {period.get('end', '?')} | "
                f"Oluşturuldu: {datetime.now().strftime('%d %b %Y %H:%M')}*\n\n---\n\n"
            )
            full_report = header + report

            # 5. Önbelleğe kaydet
            self._save_cache(full_report, context)

            return {
                "success": True,
                "report": full_report,
                "context": context,
                "from_cache": False,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "llm_available": True,
            }

        except Exception as e:
            logger.error(f"Rapor üretme hatası: {e}")
            return {
                "success": False,
                "report": None,
                "context": context,
                "from_cache": False,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "llm_available": self.client.is_ready(),
                "error": str(e),
            }

    # ─────────────────────────────────────────────────────
    # LLM Olmadan Rapor (Fallback)
    # ─────────────────────────────────────────────────────

    def _build_stats_only_report(self, context: Dict) -> str:
        """
        API key yokken bile gösterilecek yapılandırılmış istatistik raporu.
        Bu rapor tamamen deterministik — LLM kullanmaz.
        """
        period  = context.get("period", {})
        weekly  = context.get("weekly_summary", {})
        by_type = context.get("by_swing_type", {})
        top_win  = context.get("top_win")
        top_loss = context.get("top_loss")

        win_rate = weekly.get("win_rate", 0)
        pf       = weekly.get("profit_factor", 0)

        # Otomatik değerlendirme
        if win_rate >= 60 and pf >= 1.5:
            verdict = "🟢 **Güçlü hafta.** Sistem beklentilerin üzerinde performans gösterdi."
        elif win_rate >= 50 and pf >= 1.0:
            verdict = "🟡 **Makul hafta.** Sistem kârlı çalışıyor, iyileştirme fırsatı var."
        elif weekly.get("total", 0) == 0:
            verdict = "ℹ️ Bu dönemde kapanan trade yok."
        else:
            verdict = "🔴 **Zor hafta.** Stop yönetimi ve setup seçimini gözden geçir."

        lines = [
            f"# 📊 Haftalık Performans Özeti",
            f"*{period.get('start','?')} — {period.get('end','?')}*",
            f"\n> 💡 AI analizi için `.env` dosyasına LLM API key ekle.\n",
            f"---",
            f"\n## Genel Tablo\n",
            f"| Metrik | Değer |",
            f"|--------|-------|",
            f"| Toplam Trade | {weekly.get('total', 0)} |",
            f"| Win Rate | %{win_rate:.1f} ({weekly.get('wins',0)}W / {weekly.get('losses',0)}L) |",
            f"| Toplam P/L | {weekly.get('total_pnl_pct',0):+.2f}% |",
            f"| Ort. P/L/Trade | {weekly.get('avg_pnl_pct',0):+.2f}% |",
            f"| Ort. Kazanç | {weekly.get('avg_win_pct',0):+.2f}% |",
            f"| Ort. Kayıp | {weekly.get('avg_loss_pct',0):+.2f}% |",
            f"| Profit Factor | {pf:.2f}x |",
            f"\n{verdict}",
        ]

        if by_type:
            lines.append("\n## Setup Analizi\n")
            lines.append("| Tip | Trade | Win Rate | Ort. P/L |")
            lines.append("|-----|-------|----------|----------|")
            for st in sorted(by_type.keys()):
                d = by_type[st]
                lines.append(f"| {st} | {d['count']} | %{d['win_rate']:.0f} | {d['avg_pnl']:+.2f}% |")

        if top_win:
            lines.append(f"\n🏆 **En İyi Trade:** {top_win['ticker']} → {top_win['pnl_pct']:+.2f}%")
        if top_loss:
            lines.append(f"🔴 **En Kötü Trade:** {top_loss['ticker']} → {top_loss['pnl_pct']:+.2f}%")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────
    # Cache Yönetimi
    # ─────────────────────────────────────────────────────

    def _load_cache(self) -> Optional[Dict]:
        """
        Bugüne ait önbelleği yükle.
        Dünkü veya daha eski önbelleği geçersiz say.
        """
        if not CACHE_FILE.exists():
            return None
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                cached = json.load(f)

            # Tarih kontrolü: bugün üretilmiş mi?
            cached_date = cached.get("generated_at", "")[:10]
            today = datetime.now().strftime("%Y-%m-%d")
            if cached_date != today:
                logger.info("Önbellek eski tarihten — yeniden üretilecek")
                return None

            return cached
        except Exception as e:
            logger.warning(f"Önbellek okuma hatası: {e}")
            return None

    def _save_cache(self, report: str, context: Dict) -> None:
        """Raporu ve bağlamı önbelleğe kaydet."""
        try:
            cache_data = {
                "report": report,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "period": context.get("period", {}),
                "summary": context.get("weekly_summary", {}),
            }
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Rapor önbelleğe kaydedildi: {CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Önbellek kaydetme hatası: {e}")

    def clear_cache(self) -> None:
        """Önbelleği temizle (yenile butonu için)."""
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            logger.info("Rapor önbelleği temizlendi")
