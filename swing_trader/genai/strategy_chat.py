"""
strategy_chat.py — Strateji Soru-Cevap Orchestrator (D Özelliği)

RAG-lite mimarisi:
  Gerçek RAG → binlerce belge, vektör DB, benzerlik araması
  Buradaki   → küçük trade geçmişi direkt prompt'a giriyor (yeterli)

Veri akışı:
  1. data_collector.collect() → istatistik context
  2. kullanıcı sorusu → question str
  3. build_strategy_chat_prompt(question, context) → prompt
  4. client.complete() → cevap
  5. dashboard'da göster
"""

import logging
from typing import Dict, Optional

from .data_collector import WeeklyDataCollector
from .llm_client import LLMClient
from .prompts import STRATEGY_CHAT_SYSTEM, build_strategy_chat_prompt

logger = logging.getLogger(__name__)


class StrategyChat:
    """
    Trade geçmişine dayanan Strateji Soru-Cevap sistemi.

    Kullanım (dashboard'da):
        chat = StrategyChat(storage)
        result = chat.ask("Bu hafta neden sadece B tipi kazandı?")
        st.markdown(result["answer"])
    """

    def __init__(self, storage, llm_provider: Optional[str] = None):
        """
        Args:
            storage: PaperTradeStorage instance
            llm_provider: "openai" | "gemini" | None
        """
        self.storage   = storage
        self.collector = WeeklyDataCollector(storage, days=90)  # Geniş pencere
        self.client    = LLMClient(provider=llm_provider)

    def is_ready(self) -> bool:
        """LLM ve veri hazır mı?"""
        return self.client.is_ready()

    def ask(self, question: str) -> Dict:
        """
        Kullanıcının sorusunu tüm trade geçmişiyle birlikte LLM'e gönder.

        Args:
            question: "Bu hafta neden kaybettik?" gibi serbest soru

        Returns:
            {
                "success": bool,
                "answer": str,
                "llm_available": bool,
                "question": str,
            }
        """
        if not question or not question.strip():
            return {
                "success": False,
                "answer": "Lutfen bir soru yazin.",
                "llm_available": self.client.is_ready(),
                "question": question,
            }

        # 1. Deterministik veriyi topla (LLM'e context olarak gidecek)
        context = self.collector.collect()

        # 2. LLM yoksa → kullanıcıya bilgi ver
        if not self.client.is_ready():
            return {
                "success": False,
                "answer": (
                    "Strateji Soru-Cevap ozelligini kullanmak icin "
                    "`.env` dosyasina LLM API key eklemen gerekiyor.\n\n"
                    f"Mevcut istatistikler:\n"
                    f"- Toplam trade: {context['all_time_summary'].get('total', 0)}\n"
                    f"- Win Rate: %{context['all_time_summary'].get('win_rate', 0):.1f}\n"
                    f"- Ort. P/L: {context['all_time_summary'].get('avg_pnl_pct', 0):+.2f}%"
                ),
                "llm_available": False,
                "question": question,
            }

        try:
            # 3. Soruyu + context'i birleştir (RAG-lite)
            prompt = build_strategy_chat_prompt(question, context)

            # 4. LLM'e gönder
            answer = self.client.complete(
                prompt=prompt,
                system_prompt=STRATEGY_CHAT_SYSTEM,
                max_tokens=1200,
                temperature=0.3,   # Düşük — veriyle tutarlı cevap istiyoruz
            )

            if not answer:
                # Boş yanıt — fallback istatistik ver
                return {
                    "success": False,
                    "answer": (
                        "LLM boş yanıt döndürdü. Mevcut istatistikler:\n\n"
                        f"- Toplam trade: {context['all_time_summary'].get('total', 0)}\n"
                        f"- Win Rate: %{context['all_time_summary'].get('win_rate', 0):.1f}\n"
                        f"- Ort. P/L: {context['all_time_summary'].get('avg_pnl_pct', 0):+.2f}%"
                    ),
                    "llm_available": True,
                    "question": question,
                }

            return {
                "success": True,
                "answer": answer,
                "llm_available": True,
                "question": question,
            }

        except Exception as e:
            logger.error(f"StrategyChat hatasi: {e}")
            err_msg = str(e)
            # Kullanıcıya rate limit veya genel hata açıkça gösterilsin
            return {
                "success": False,
                "answer": err_msg if 'rate' in err_msg.lower() or 'limit' in err_msg.lower() else f"Hata olustu: {err_msg}",
                "llm_available": self.client.is_ready(),
                "question": question,
            }
