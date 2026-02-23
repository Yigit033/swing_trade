"""
signal_briefer.py — Sinyal Brifingi Orchestrator (A Özelliği)

Bu dosya haftalık reporter.py ile aynı kalıbı izliyor:
  1. Signal dict al (scanner'dan)
  2. Prompt'u oluştur (prompts.py)
  3. LLM'e gönder (llm_client.py)
  4. Cevabı döndür → dashboard'da göster

FARK: Rapor gibi cache yok — her sinyal anlık değerlendirme ister.
"""

import logging
from typing import Dict, Optional

from .llm_client import LLMClient
from .prompts import SIGNAL_BRIEFING_SYSTEM, build_signal_briefing_prompt

logger = logging.getLogger(__name__)


class SignalBriefer:
    """
    Tek bir scanner sinyali için 2-3 cümlelik AI brifingi üretir.

    Kullanım (dashboard'da):
        briefer = SignalBriefer()
        if briefer.is_ready():
            result = briefer.brief(signal_dict)
            st.info(result["text"])
    """

    def __init__(self, llm_provider: Optional[str] = None):
        self.client = LLMClient(provider=llm_provider)

    def is_ready(self) -> bool:
        """LLM kullanılabilir mi?"""
        return self.client.is_ready()

    def brief(self, signal: Dict) -> Dict:
        """
        Sinyal için AI brifingi üret.

        Args:
            signal: Scanner signal dict
                    (entry_price, stop_loss, target_1, atr,
                     quality_score, swing_type, ticker)

        Returns:
            {
                "success": bool,
                "text": str,         # 2-3 cümle AI yorum
                "llm_available": bool,
                "fallback": bool,    # True = LLM yok, deterministik özet
            }
        """
        # LLM yoksa deterministik özet döndür
        if not self.client.is_ready():
            return {
                "success": True,
                "text": self._fallback_summary(signal),
                "llm_available": False,
                "fallback": True,
            }

        try:
            prompt = build_signal_briefing_prompt(signal)
            response = self.client.complete(
                prompt=prompt,
                system_prompt=SIGNAL_BRIEFING_SYSTEM,
                max_tokens=200,      # Kısa brifing için 200 yeterli
                temperature=0.4,     # Biraz düşük — tutarlı yorum istiyoruz
            )

            if not response:
                raise ValueError("LLM bos cevap dondurudu")

            return {
                "success": True,
                "text": response,
                "llm_available": True,
                "fallback": False,
            }

        except Exception as e:
            logger.warning(f"SignalBriefer hatasi: {e}")
            return {
                "success": True,
                "text": self._fallback_summary(signal),
                "llm_available": False,
                "fallback": True,
            }

    def _fallback_summary(self, signal: Dict) -> str:
        """
        LLM olmadan deterministik kısa özet.
        Aynı Hybrid Architecture prensibi: hesap Python'da.
        """
        entry   = signal.get("entry_price", 0) or 0
        stop    = signal.get("stop_loss", 0) or 0
        target  = signal.get("target_1") or signal.get("target", 0) or 0
        quality = signal.get("quality_score", 0) or 0

        risk_pct   = abs(entry - stop) / entry * 100 if entry else 0
        reward_pct = abs(target - entry) / entry * 100 if entry else 0
        rr_ratio   = reward_pct / risk_pct if risk_pct > 0 else 0

        parts = []
        if rr_ratio >= 3.0:
            parts.append(f"R/R 1:{rr_ratio:.1f} ile guclu bir risk/odul dengesi.")
        elif rr_ratio >= 2.0:
            parts.append(f"R/R 1:{rr_ratio:.1f} kabul edilebilir seviyede.")
        else:
            parts.append(f"R/R 1:{rr_ratio:.1f} dusuk — dikkatli ol.")

        if quality >= 8:
            parts.append(f"Kalite skoru {quality}/10 ile yuksek kaliteli bir kurulum.")
        elif quality >= 6:
            parts.append(f"Kalite skoru {quality}/10 — orta seviye kurulum.")
        else:
            parts.append(f"Kalite skoru {quality}/10 — dusuk kaliteli, kurulumu tekrar degerlendirmek gerekebilir.")

        parts.append(
            "AI analizi icin .env dosyasina LLM API key ekle."
        )

        return " ".join(parts)
