"""
explainer.py — Model Açıklanabilirlik Modülü (SHAP)

SHAP nedir?
───────────
Bir ML modeli "bu trade kazanır" diyebilir ama NEDENİ söylemez.
SHAP (SHapley Additive exPlanations), her özelliğin tahmine
ne kadar katkıda bulunduğunu hesaplar.

Örnek çıktı:
  risk_reward_ratio → +0.15  (kazanma ihtimalini artırıyor)
  quality_score     → +0.08  (olumlu katkı)
  atr_pct           → -0.12  (olumsuz katkı, yüksek volatilite)
  
Bu, modeli bir "kara kutu" olmaktan çıkarır.
Gerçek profesyonel ML projelerinde explainability zorunludur.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Grafik kayıt klasörü
_BASE = Path(__file__).parent.parent.parent
CHARTS_DIR = _BASE / "data" / "ml_models" / "charts"


class SignalExplainer:
    """
    SHAP ile model kararlarını açıklar ve görselleştirir.
    
    Kullanım:
        explainer = SignalExplainer(model)
        
        # Tüm test verisi üzerinde özet grafik
        explainer.plot_summary(X_test, save=True)
        
        # Tek bir tahmin için waterfall (şelale) açıklaması
        explainer.plot_single(X_single, save=True)
        
        # Sayısal açıklama (grafik olmadan)
        values = explainer.explain_single(X_single)
    """

    def __init__(self, model):
        """
        Args:
            model: Eğitilmiş XGBoost modeli (trainer.py çıktısı)
        """
        self.model = model
        self._shap_explainer = None
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_shap_explainer(self):
        """SHAP TreeExplainer'ı tembel (lazy) yükle."""
        if self._shap_explainer is None:
            import shap
            # TreeExplainer: ağaç tabanlı modeller için özel SHAP algoritması
            # XGBoost, RandomForest gibi modellerle çok hızlı çalışır
            self._shap_explainer = shap.TreeExplainer(self.model)
        return self._shap_explainer

    # ─────────────────────────────────────────────────────────────────
    # SHAP Değerleri Hesapla
    # ─────────────────────────────────────────────────────────────────

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Verilen feature matrix için SHAP değerlerini hesaplar.
        
        Returns:
            shap_values: her özellik için katkı değerleri array'i
                         pozitif → WIN'e katkı, negatif → LOSS'a katkı
        """
        explainer = self._get_shap_explainer()
        shap_values = explainer.shap_values(X)
        return shap_values

    # ─────────────────────────────────────────────────────────────────
    # Tek Tahmin Açıklaması (sayısal)
    # ─────────────────────────────────────────────────────────────────

    def explain_single(self, X_single: pd.DataFrame) -> List[Dict]:
        """
        Tek bir trade için SHAP açıklaması döner.
        
        Returns:
            [
                {"feature": "risk_reward_ratio", "shap_value": 0.15, "feature_value": 2.3},
                {"feature": "quality_score",     "shap_value": 0.08, "feature_value": 7.5},
                ...
            ]
            (SHAP değerine göre azalan sırada, en etkili feature önce)
        """
        try:
            from .features import FEATURE_COLUMNS

            shap_vals = self.compute_shap_values(X_single)

            # Binary classification'da shap_values[1] = WIN sınıfı
            # Eğer tek array dönerse doğrudan kullan
            if isinstance(shap_vals, list):
                vals = shap_vals[1][0]  # WIN sınıfı, ilk örnek
            else:
                vals = shap_vals[0]

            results = []
            for feature, sv, fv in zip(
                FEATURE_COLUMNS, vals, X_single.iloc[0].values
            ):
                results.append({
                    "feature": feature,
                    "shap_value": round(float(sv), 4),
                    "feature_value": round(float(fv), 4),
                })

            # En etkili feature'dan en az etkiliye sırala
            results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            return results

        except Exception as e:
            logger.error(f"SHAP açıklama hatası: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────
    # Grafikler
    # ─────────────────────────────────────────────────────────────────

    def plot_summary(
        self, X: pd.DataFrame, save: bool = False
    ) -> Optional[str]:
        """
        Tüm veri üzerinde feature importance özet grafiği.
        Her nokta bir trade, renk WIN/LOSS'u gösterir.
        
        Args:
            X: Feature matrix (tüm test setinden gelir)
            save: True ise dosyaya kaydet ve path döner
        
        Returns:
            Kayıt yolu (str) veya None
        """
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")  # GUI olmadan çalış
            import matplotlib.pyplot as plt

            shap_vals = self.compute_shap_values(X)

            fig, ax = plt.subplots(figsize=(10, 6))

            if isinstance(shap_vals, list):
                shap.summary_plot(
                    shap_vals[1], X,
                    show=False,
                    plot_type="bar"
                )
            else:
                shap.summary_plot(shap_vals, X, show=False, plot_type="bar")

            plt.title("Feature Importance (SHAP)", fontsize=14, fontweight="bold")
            plt.tight_layout()

            if save:
                path = CHARTS_DIR / "shap_summary.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"SHAP özet grafik kaydedildi: {path}")
                return str(path)

            plt.close()
            return None

        except Exception as e:
            logger.warning(f"SHAP özet grafik hatası: {e}")
            return None

    def plot_waterfall(
        self, X_single: pd.DataFrame, save: bool = False
    ) -> Optional[str]:
        """
        Tek bir trade için "waterfall" (şelale) grafiği.
        Her feature'ın tahmini nasıl etkilediğini bar bar gösterir.
        
        Args:
            X_single: Tek satırlık feature DataFrame
            save: True ise dosyaya kaydet
        
        Returns:
            Kayıt yolu (str) veya None
        """
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            explainer = self._get_shap_explainer()

            # Tek örnek için Explanation objesi oluştur
            shap_explanation = explainer(X_single)

            plt.figure(figsize=(10, 5))
            shap.waterfall_plot(shap_explanation[0], show=False)
            plt.title("Bu Sinyal İçin SHAP Açıklaması", fontsize=13)
            plt.tight_layout()

            if save:
                path = CHARTS_DIR / "shap_waterfall.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"SHAP waterfall kaydedildi: {path}")
                return str(path)

            plt.close()
            return None

        except Exception as e:
            logger.warning(f"SHAP waterfall hatası: {e}")
            return None
