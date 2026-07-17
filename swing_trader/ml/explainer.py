"""
explainer.py — Model Açıklanabilirlik Modülü (TreeSHAP)

SHAP nedir?
───────────
Bir ML modeli "bu trade %72 kazanır" diyebilir ama NEDENİNİ söylemez.
SHAP (SHapley Additive exPlanations), her özelliğin O TAHMİNE
ne kadar katkıda bulunduğunu hesaplar.

Örnek çıktı:
  risk_reward_ratio → +0.15  (kazanma ihtimalini artırıyor)
  quality_score     → +0.08  (olumlu katkı)
  atr_pct           → -0.12  (olumsuz katkı, yüksek volatilite)

Bu, modeli bir "kara kutu" olmaktan çıkarır — predictor'ın global
feature importance'ından farkı: bu değerler TEK BİR tahmine özeldir,
her sinyalde farklı çıkar.

Uygulama notu
─────────────
`shap` paketi KULLANILMAZ. XGBoost'un yerleşik TreeSHAP implementasyonu
(`Booster.predict(pred_contribs=True)`) aynı matematiği verir ve
shap+numba+matplotlib zincirini (production imajında ~100MB+) gerektirmez.
"""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class SignalExplainer:
    """
    Tek bir tahmin için feature katkılarını (TreeSHAP) hesaplar.

    Kullanım:
        explainer = SignalExplainer(model)   # model: eğitilmiş XGBClassifier
        contribs = explainer.explain_single(X_single)
        # [{"feature": "risk_reward_ratio", "shap_value": 0.15, "feature_value": 2.3}, ...]
    """

    def __init__(self, model):
        """
        Args:
            model: Eğitilmiş XGBoost sklearn modeli (trainer.py çıktısı)
        """
        self.model = model

    def explain_single(self, X_single: pd.DataFrame) -> List[Dict]:
        """
        Tek bir sinyal için SHAP açıklaması döner.

        Returns:
            [
                {"feature": "risk_reward_ratio", "shap_value": 0.15, "feature_value": 2.3},
                {"feature": "atr_pct",           "shap_value": -0.12, "feature_value": 4.1},
                ...
            ]
            |shap_value|'ya göre azalan sırada; pozitif değer WIN olasılığını
            artıran, negatif değer düşüren katkıdır (logit uzayında).
        """
        try:
            import xgboost as xgb

            booster = self.model.get_booster()
            dmat = xgb.DMatrix(X_single, feature_names=list(X_single.columns))
            # pred_contribs=True → satır başına [katkı_1..katkı_n, bias]
            contribs = booster.predict(dmat, pred_contribs=True)[0]

            results = []
            for feature, sv, fv in zip(
                X_single.columns, contribs[:-1], X_single.iloc[0].values
            ):
                results.append({
                    "feature": str(feature),
                    "shap_value": round(float(sv), 4),
                    "feature_value": round(float(fv), 4),
                })

            results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            return results

        except Exception as e:
            logger.error(f"SHAP açıklama hatası: {e}")
            return []
