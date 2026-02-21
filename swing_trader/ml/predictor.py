"""
predictor.py — Tahmin Motoru

Kaydedilmiş modeli yükler ve yeni sinyaller için
kazanma olasılığı (win probability) tahmin eder.

─────────────────────────────────────────────────
TEMEL KAVRAM: Inference (Çıkarım)
─────────────────────────────────────────────────
Eğitim (trainer.py) → Makine öğrenir
Inference (predictor.py) → Öğrenileni kullanır

Bu sınıf model yoksa sessizce atlar — yani
dashboard'da herhangi bir hataya yol açmaz.
─────────────────────────────────────────────────
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Model yolları (trainer.py ile aynı olmalı)
_BASE = Path(__file__).parent.parent.parent
MODEL_DIR = _BASE / "data" / "ml_models"
MODEL_PATH = MODEL_DIR / "signal_predictor.pkl"
META_PATH = MODEL_DIR / "signal_predictor_meta.json"

# Confidence eşikleri — olasılık → insan dilinde açıklama
CONFIDENCE_LEVELS = [
    (0.70, "YÜKSEK 🟢"),    # %70+ → güçlü sinyal
    (0.55, "ORTA 🟡"),      # %55-70 → makul sinyal
    (0.40, "DÜŞÜK 🟠"),     # %40-55 → zayıf sinyal
    (0.00, "RİSKLİ 🔴"),    # <%40 → kaçın
]


class SignalPredictor:
    """
    Kayıtlı modeli yükler ve sinyaller için win olasılığı tahmin eder.
    
    Kullanım:
        predictor = SignalPredictor()
        
        if predictor.is_ready:
            result = predictor.predict(signal_dict)
            print(f"Win ihtimali: {result['win_probability']:.1%}")
        else:
            print("Model henüz eğitilmedi.")
    """

    def __init__(self):
        """Model ve metadata'yı yükle. Yoksa is_ready=False olur."""
        self.model = None
        self.meta = {}
        self.is_ready = False
        self._load()

    # ─────────────────────────────────────────────────────────────────
    # ANA METOD — Tahmin Üret
    # ─────────────────────────────────────────────────────────────────

    def predict(self, signal: Dict) -> Optional[Dict]:
        """
        Bir sinyal için win olasılığı tahmin eder.
        
        Args:
            signal: Scanner sinyali veya trade dict'i.
                    Şu alanlar kullanılır:
                      entry_price, stop_loss, target, atr,
                      quality_score, swing_type, max_hold_days,
                      entry_date (veya date)
        
        Returns:
            {
                'win_probability': 0.72,  # 0.0 - 1.0 arası
                'confidence': 'YÜKSEK 🟢',
                'label': 1,               # 1=WIN tahmin, 0=LOSS tahmin
                'top_features': [...]     # En etkili özellikler
            }
            veya None (model hazır değilse)
        """
        if not self.is_ready:
            return None

        try:
            from .features import FeatureEngineer

            # Sinyali feature matrix'e çevir
            engineer = FeatureEngineer()
            X = engineer.transform_signal(signal)

            # Olasılık tahmini — [LOSS olasılığı, WIN olasılığı]
            proba = self.model.predict_proba(X)[0]
            win_prob = float(proba[1])

            # Binary tahmin (threshold=0.5)
            predicted_label = int(self.model.predict(X)[0])

            # Feature importance'dan en etkilisini bul
            top_features = self._get_top_features(X)

            return {
                "win_probability": round(win_prob, 4),
                "confidence": self._get_confidence_label(win_prob),
                "label": predicted_label,
                "top_features": top_features,
            }

        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return None

    def predict_batch(self, signals: list) -> list:
        """
        Birden fazla sinyal için toplu tahmin.
        
        Args:
            signals: Sinyal dict listesi
        
        Returns:
            Sonuç dict listesi (aynı sırada)
        """
        return [self.predict(s) for s in signals]

    def get_meta(self) -> Dict:
        """
        Model hakkında bilgi döner (eğitim tarihi, metrikler vb.)
        """
        return self.meta

    # ─────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────

    def _load(self):
        """
        Model dosyasını yükle.
        Dosya yoksa is_ready=False kalır (hata vermez).
        """
        try:
            import joblib

            if not MODEL_PATH.exists():
                logger.info("Model dosyası bulunamadı. Önce trainer.run() çalıştır.")
                return

            self.model = joblib.load(MODEL_PATH)

            # Metadata yükle
            if META_PATH.exists():
                with open(META_PATH) as f:
                    self.meta = json.load(f)

            self.is_ready = True
            logger.info(
                f"Model yüklendi: {MODEL_PATH} "
                f"(ROC-AUC: {self.meta.get('roc_auc', '?')})"
            )

        except Exception as e:
            logger.warning(f"Model yüklenemedi: {e}")
            self.is_ready = False

    def _get_confidence_label(self, win_prob: float) -> str:
        """
        Olasılık değerini insan dilinde güven seviyesine çevirir.
        Örn: 0.72 → "YÜKSEK 🟢"
        """
        for threshold, label in CONFIDENCE_LEVELS:
            if win_prob >= threshold:
                return label
        return "RİSKLİ 🔴"

    def _get_top_features(self, X, top_n: int = 3) -> list:
        """
        Modelin bu tahmin için en çok hangi feature'lara
        baktığını döner (feature importance sıralaması).
        
        NOT: Bu global importance'dır (tek tahmin bazlı değil).
        Tahmin bazlı açıklama için explainer.py'de SHAP kullan.
        """
        try:
            from .features import FEATURE_COLUMNS

            importances = self.model.feature_importances_
            # [(önem, isim), ...] → büyükten küçüğe sırala
            ranked = sorted(
                zip(importances, FEATURE_COLUMNS),
                reverse=True
            )[:top_n]

            return [
                {"feature": name, "importance": round(float(imp), 4)}
                for imp, name in ranked
            ]

        except Exception:
            return []
