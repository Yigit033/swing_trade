"""
swing_trader/genai/__init__.py

Generative AI modülü — Hybrid Architecture
═══════════════════════════════════════════
Bu modül deterministik sistemin çıktısını alır ve
LLM ile insan-okunabilir analizler üretir.

╔══════════════════════════════════════════════════════╗
║  Deterministik             →  LLM  →  İnsan Çıktısı ║
║  (XGBoost, P/L hesabı)       (Analiz)   (Rapor)     ║
╚══════════════════════════════════════════════════════╝

LLM hiçbir zaman:
  ❌ Stop loss hesaplamaz
  ❌ Position size kararı vermez
  ❌ "İşlem yap" demez

LLM sadece:
  ✅ Haftalık sonuçları özetler
  ✅ Pattern'leri yorumlar
  ✅ İyileştirme önerileri sunar
"""

from .reporter import WeeklyReporter
from .signal_briefer import SignalBriefer
from .strategy_chat import StrategyChat

__all__ = ["WeeklyReporter", "SignalBriefer", "StrategyChat"]

