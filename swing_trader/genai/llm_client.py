"""
llm_client.py — Provider-Agnostic LLM Wrapper

NEDEN BUNU YAZIYORUZ?
──────────────────────
Direkt OpenAI veya Gemini API'ını her yerde çağırırsak,
yarın "başka bir model kullanalım" dediğinde tüm kodu değiştirmen gerekir.

Bunun yerine bir "adapter" (adaptör) katmanı oluşturuyoruz:
  - LLMClient(provider="openai")  → OpenAI GPT-4o kullanır
  - LLMClient(provider="gemini")  → Google Gemini kullanır
  
Gerisini bilmene gerek yok. Sadece şunu çağırırsın:
  client.complete(prompt)  → str (model cevabı)

KURULUM:
  .env dosyasına ekle:
    LLM_PROVIDER=openai          # veya: gemini
    OPENAI_API_KEY=sk-...        # OpenAI'dan al: platform.openai.com
    GEMINI_API_KEY=...           # Google'dan al: aistudio.google.com
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Desteklenen modeller
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",   # Daha ucuz ama güçlü
    "gemini": "gemini-1.5-flash",
}


class LLMClient:
    """
    Provider-agnostic LLM client.
    
    Örnek:
        client = LLMClient()          # .env'den okur
        response = client.complete("Bu hisseyle ilgili...")
    """

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Args:
            provider: "openai" | "gemini" | None (.env'den okur)
            model: Model adı (None ise provider'a göre default)
        """
        self.provider = (
            provider
            or os.getenv("LLM_PROVIDER", "openai")
        ).lower()

        self.model = model or DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")
        self._client = None      # Tembel yükleme (lazy load)
        self.available = False   # API key var mı?

        self._setup()

    def _setup(self):
        """Provider'a göre client'ı hazırla ve API key kontrol et."""
        if self.provider == "openai":
            self._setup_openai()
        elif self.provider == "gemini":
            self._setup_gemini()
        else:
            logger.warning(f"Bilinmeyen LLM provider: {self.provider}")

    def _setup_openai(self):
        """OpenAI client kurulumu."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("your_"):
            logger.info("OPENAI_API_KEY bulunamadı — AI rapor özelliği pasif")
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self.available = True
            logger.info(f"OpenAI client hazır: {self.model}")
        except ImportError:
            logger.warning("openai paketi yüklü değil. Kur: pip install openai")

    def _setup_gemini(self):
        """Google Gemini client kurulumu."""
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key.startswith("your_"):
            logger.info("GEMINI_API_KEY bulunamadı — AI rapor özelliği pasif")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model)
            self.available = True
            logger.info(f"Gemini client hazır: {self.model}")
        except ImportError:
            logger.warning("google-generativeai paketi yüklü değil. Kur: pip install google-generativeai")

    # ─────────────────────────────────────────────
    # Ana Method: complete()
    # ─────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1500,
        temperature: float = 0.5,
    ) -> Optional[str]:
        """
        LLM'e prompt gönder, cevap al.
        
        Args:
            prompt: Kullanıcı mesajı / analiz isteği
            system_prompt: LLM'e "kim olduğunu" söyleyen bağlam
            max_tokens: Maksimum token sayısı
            temperature: 0.0 = deterministik, 1.0 = yaratıcı (0.5 iyi denge)
        
        Returns:
            LLM'in cevabı (str) veya None (hata/API key yoksa)
        """
        if not self.available or self._client is None:
            return None

        try:
            if self.provider == "openai":
                return self._complete_openai(prompt, system_prompt, max_tokens, temperature)
            elif self.provider == "gemini":
                return self._complete_gemini(prompt, system_prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"LLM complete hatası ({self.provider}): {e}")
            return None

        return None

    def _complete_openai(self, prompt, system_prompt, max_tokens, temperature) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _complete_gemini(self, prompt, system_prompt, max_tokens, temperature) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self._client.generate_content(
            full_prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return response.text.strip()

    def is_ready(self) -> bool:
        """API key var mı ve client hazır mı?"""
        return self.available and self._client is not None
