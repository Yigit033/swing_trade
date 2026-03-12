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
    "gemini": "gemini-2.5-flash",  # Updated: 2.0-flash deprecated
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
        # Defensive: ensure .env is loaded regardless of calling context
        self._ensure_env_loaded()

        self.provider = (
            provider
            or os.getenv("LLM_PROVIDER", "openai")
        ).lower()

        self.model = model or DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")
        self._client = None      # Tembel yükleme (lazy load)
        self.available = False   # API key var mı?

        self._setup()

    @staticmethod
    def _ensure_env_loaded():
        """Load .env if not already loaded — guarantees API keys are available."""
        if os.getenv("_LLM_ENV_LOADED"):
            return
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            # Try project root (.env next to swing_trader/)
            env_path = Path(__file__).resolve().parents[2] / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=False)
                logger.info(f"LLMClient: loaded .env from {env_path}")
            os.environ["_LLM_ENV_LOADED"] = "1"
        except ImportError:
            pass  # python-dotenv not installed

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
        logger.info(f"Gemini setup: key_exists={bool(api_key)}, key_starts_with_your={api_key.startswith('your_') if api_key else 'N/A'}, key_len={len(api_key)}")
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
        except Exception as e:
            logger.error(f"Gemini setup error: {e}", exc_info=True)

    # ─────────────────────────────────────────────
    # Ana Method: complete()
    # ─────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 100000,
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
                result = self._complete_openai(prompt, system_prompt, max_tokens, temperature)
            elif self.provider == "gemini":
                result = self._complete_gemini(prompt, system_prompt, max_tokens, temperature)
            else:
                return None

            if result:
                return result
            logger.warning(f"LLM ({self.provider}) returned empty response")
            return None
        except Exception as e:
            err_msg = str(e).lower()
            if 'rate' in err_msg or 'limit' in err_msg or 'quota' in err_msg or 'resource' in err_msg:
                logger.warning(f"LLM rate limited ({self.provider}): {e}")
                raise RuntimeError("Gemini API rate limited — birkaç dakika bekleyip tekrar deneyin.") from e
            logger.error(f"LLM complete hatası ({self.provider}): {e}", exc_info=True)
            raise

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

        # Handle safety blocks and empty responses
        try:
            text = response.text
        except ValueError as e:
            # Safety filter blocked the response
            logger.warning(f"Gemini safety block: {e}")
            if hasattr(response, 'prompt_feedback'):
                logger.warning(f"Prompt feedback: {response.prompt_feedback}")
            # Try to extract from candidates
            if response.candidates:
                for c in response.candidates:
                    if c.content and c.content.parts:
                        return c.content.parts[0].text.strip()
            return ""

        if text:
            return text.strip()

        # Fallback: try candidates directly
        logger.warning(f"Gemini empty text. Candidates: {getattr(response, 'candidates', 'N/A')}")
        if hasattr(response, 'candidates') and response.candidates:
            for c in response.candidates:
                if c.content and c.content.parts:
                    return c.content.parts[0].text.strip()
        return ""

    def is_ready(self) -> bool:
        """API key var mı ve client hazır mı?"""
        return self.available and self._client is not None
