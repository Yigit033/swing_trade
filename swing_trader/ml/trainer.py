"""
trainer.py — Model Eğitim Modülü

Bu dosya XGBoost modelini paper trade geçmişiyle eğitir ve kaydeder.

─────────────────────────────────────────────────
TEMEL KAVRAM: Supervised Learning Pipeline
─────────────────────────────────────────────────
1. Veri yükle        → SQLite'dan closed trades çek
2. Feature hazırla   → features.py ile X, y oluştur
3. Split             → %80 train, %20 test
4. Model eğit        → XGBoost (cross-validation ile)
5. Değerlendir       → Accuracy, ROC-AUC, F1
6. Kaydet            → model.pkl (predictor bunu yükler)
─────────────────────────────────────────────────
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────
# Model ve metadata kayıt yolları
# ─────────────────────────────────────────────────
# Bu dosyadan iki seviye yukarı çık → proje kökü
_BASE = Path(__file__).parent.parent.parent
MODEL_DIR = _BASE / "data" / "ml_models"
MODEL_PATH = MODEL_DIR / "signal_predictor.pkl"
META_PATH = MODEL_DIR / "signal_predictor_meta.json"

# Eğitim için minimum trade sayısı
MIN_TRADES_REQUIRED = 15


class SignalTrainer:
    """
    XGBoost modelini eğiten ve kaydeden sınıf.
    
    Kullanım (notebook veya CLI'dan):
        trainer = SignalTrainer()
        result = trainer.run()
        print(result)
    """

    def __init__(self):
        # Model kayıt klasörünü oluştur (yoksa)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # ANA METOD — Tek komutla her şeyi yapar
    # ─────────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Tüm eğitim pipeline'ını çalıştırır:
        veri yükle → feature hazırla → eğit → değerlendir → kaydet
        
        Returns:
            Sonuç dict: accuracy, roc_auc, f1, train_size, test_size, vb.
        """
        logger.info("=== AI Signal Predictor Eğitimi Başlıyor ===")

        # 1. Veriyi yükle
        X, y, raw_count = self.load_data()
        if X is None:
            return {"success": False, "error": f"Yeterli veri yok (min {MIN_TRADES_REQUIRED})"}

        # 2. Eğitim / test bölümü
        X_train, X_test, y_train, y_test = self._split(X, y)

        # 3. Modeli eğit (cross-validation ile)
        model, cv_scores = self._train(X_train, y_train)

        # 4. Test seti üzerinde değerlendir
        metrics = self._evaluate(model, X_test, y_test, cv_scores)
        metrics["train_size"] = len(X_train)
        metrics["test_size"] = len(X_test)
        metrics["total_trades"] = raw_count

        # 5. Kaydet
        self._save(model, metrics)

        logger.info(
            f"Eğitim tamamlandı → "
            f"Accuracy: {metrics['accuracy']:.3f}, "
            f"ROC-AUC: {metrics['roc_auc']:.3f}"
        )

        return {"success": True, **metrics}

    # ─────────────────────────────────────────────────────────────────
    # ADIM 1: Veri Yükleme
    # ─────────────────────────────────────────────────────────────────

    def load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], int]:
        """
        SQLite'dan closed trades çeker ve feature matrix oluşturur.
        
        Returns:
            (X, y, raw_count) veya (None, None, 0) yeterli veri yoksa
        """
        try:
            from ..paper_trading.storage import PaperTradeStorage
            from .features import FeatureEngineer

            storage = PaperTradeStorage()
            # Tüm kapatılmış trade'leri çek (limit=9999 = hepsi)
            trades = storage.get_closed_trades(limit=9999)

            # REJECTED trade'leri filtrele — bunlar hiç girmedi, anlamsız
            trades = [t for t in trades if t.get("status") != "REJECTED"]

            raw_count = len(trades)
            logger.info(f"Veritabanından {raw_count} kapatılmış trade yüklendi.")

            if raw_count < MIN_TRADES_REQUIRED:
                logger.warning(
                    f"Yeterli veri yok: {raw_count} trade "
                    f"(minimum {MIN_TRADES_REQUIRED} gerekli)"
                )
                return None, None, raw_count

            engineer = FeatureEngineer()
            X, y = engineer.transform(trades)

            return X, y, raw_count

        except Exception as e:
            logger.error(f"Veri yükleme hatası: {e}")
            return None, None, 0

    # ─────────────────────────────────────────────────────────────────
    # ADIM 2: Train / Test Split
    # ─────────────────────────────────────────────────────────────────

    def _split(self, X: pd.DataFrame, y: pd.Series):
        """
        Veriyi %80 eğitim, %20 test olarak böler.
        
        stratify=y → WIN/LOSS oranı her iki sette de eşit tutulur.
        random_state=42 → Tekrarlanabilir sonuçlar için sabit seed.
        """
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.20,       # %20 test
            stratify=y,           # Sınıf dengesini koru
            random_state=42       # Sabit seed = tekrarlanabilir
        )
        logger.info(
            f"Split: {len(X_train)} train / {len(X_test)} test | "
            f"WIN rate train: {y_train.mean():.1%} | test: {y_test.mean():.1%}"
        )
        return X_train, X_test, y_train, y_test

    # ─────────────────────────────────────────────────────────────────
    # ADIM 3: Model Eğitimi
    # ─────────────────────────────────────────────────────────────────

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        XGBoost modelini StratifiedKFold cross-validation ile eğitir.
        
        XGBoost neden?
          - Tablo verisi için son derece güçlü
          - Eksik değerlere toleranslı
          - Hızlı ve yorumlanabilir (SHAP ile)
          - Sektörde standart (Kaggle, fintech)
        
        Cross-validation neden?
          - Tek bir train/test split şansa bağlıdır
          - K=5 fold: 5 farklı bölümde test et, ortalamasını al
          - Daha güvenilir performans tahmini
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import StratifiedKFold, cross_val_score

            # ── XGBoost Parametreleri ────────────────────────────────
            # scale_pos_weight: Sınıf dengesizliğini düzeltir
            # (LOSS sayısı WIN sayısından fazlaysa modeli dengele)
            n_loss = int((y_train == 0).sum())
            n_win = int((y_train == 1).sum())
            scale_pos_weight = n_loss / n_win if n_win > 0 else 1.0

            model = xgb.XGBClassifier(
                n_estimators=200,         # Kaç ağaç? (daha fazla = daha iyi ama yavaş)
                max_depth=4,              # Her ağacın derinliği (overfitting önler)
                learning_rate=0.05,       # Her adımda ne kadar öğren
                subsample=0.8,            # Her ağaç için veri oranı (randomness)
                colsample_bytree=0.8,     # Her ağaç için feature oranı
                scale_pos_weight=scale_pos_weight,  # Sınıf dengesi
                use_label_encoder=False,
                eval_metric="logloss",    # Kayıp fonksiyonu (binary classification)
                random_state=42,
                verbosity=0               # Eğitim loglarını sustur
            )

            # ── 5-Fold Cross-Validation ──────────────────────────────
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring="roc_auc"  # ROC-AUC: 0.5=random, 1.0=mükemmel
            )

            logger.info(
                f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
            )

            # Son model tüm train verisiyle eğit
            model.fit(X_train, y_train)
            return model, cv_scores

        except ImportError:
            logger.error("xgboost kurulu değil! 'pip install xgboost' çalıştır.")
            raise

    # ─────────────────────────────────────────────────────────────────
    # ADIM 4: Değerlendirme
    # ─────────────────────────────────────────────────────────────────

    def _evaluate(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, cv_scores
    ) -> Dict:
        """
        Modeli test seti üzerinde değerlendirir.
        
        Metrikler:
          Accuracy  → Genel doğruluk (yanıltıcı olabilir, imbalanced data'da)
          ROC-AUC   → Sınıflandırma gücü (0.5=şans, 1.0=mükemmel)
          F1-Score  → Precision ve Recall'un harmonik ortalaması
          
        Confusion Matrix:
          [[TN, FP],   TN=doğru LOSS, FP=yanlış WIN dedi
           [FN, TP]]   FN=yanlış LOSS dedi, TP=doğru WIN
        """
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, f1_score, confusion_matrix
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # WIN olasılıkları

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        return {
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4),
            "f1": round(f1, 4),
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_roc_auc_std": round(float(cv_scores.std()), 4),
            "confusion_matrix": cm,
        }

    # ─────────────────────────────────────────────────────────────────
    # ADIM 5: Kaydetme
    # ─────────────────────────────────────────────────────────────────

    def _save(self, model, metrics: Dict):
        """
        Modeli .pkl formatında, metadata'yı JSON olarak kaydeder.
        
        .pkl → Python objesinin ikili (binary) formatı.
              Predictor bunu yükleyerek tahmin yapar.
        
        .json → Model hakkında meta bilgiler (ne zaman eğitildi,
               kaç trade var, metrikler) — insan okunabilir.
        """
        import joblib
        from datetime import datetime

        # Model kaydet
        joblib.dump(model, MODEL_PATH)

        # Metadata kaydet
        meta = {
            "trained_at": datetime.now().isoformat(),
            "model_path": str(MODEL_PATH),
            **metrics
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Model kaydedildi: {MODEL_PATH}")
        logger.info(f"Metadata kaydedildi: {META_PATH}")
