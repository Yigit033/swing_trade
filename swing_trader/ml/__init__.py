"""
swing_trader/ml/ — AI Signal Quality Predictor modülü

Bu paket, geçmiş paper trade sonuçlarından öğrenerek
yeni sinyallerin kazanıp kazanmayacağını tahmin eder.

Modüller:
    features  → Ham veriyi ML özelliklerine dönüştürür
    trainer   → XGBoost modelini eğitir ve kaydeder
    predictor → Kayıtlı modeli yükler, tahmin üretir
    explainer → TreeSHAP ile modelin kararlarını tahmin bazında açıklar
"""

# Dışa açık sınıflar — dışarıdan sadece bunları import et
from .predictor import SignalPredictor
from .trainer import SignalTrainer
from .features import FeatureEngineer
from .explainer import SignalExplainer

__all__ = ["SignalPredictor", "SignalTrainer", "FeatureEngineer", "SignalExplainer"]
