"""
test_ml.py — ML Modülü Unit Testleri

─────────────────────────────────────────────────
TEMEL KAVRAM: Unit Testing
─────────────────────────────────────────────────
Her fonksiyonu izole olarak test et. 
Böylece bir şey bozulunca tam olarak nerede bozulduğunu bilirsin.

Çalıştırmak için:
    python -m pytest swing_trader/tests/test_ml.py -v
─────────────────────────────────────────────────
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Proje kökünü Python path'ine ekle
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─────────────────────────────────────────────────
# Yardımcı Test Verisi
# ─────────────────────────────────────────────────

def make_trade(status="TARGET", **kwargs) -> dict:
    """Test için örnek trade dict'i üretir."""
    base = {
        "id": 1,
        "ticker": "TEST",
        "entry_price": 10.0,
        "stop_loss": 9.0,       # %10 risk
        "target": 13.0,         # %30 reward → R/R = 3.0
        "atr": 0.5,             # %5 ATR
        "quality_score": 7.0,
        "swing_type": "A",
        "max_hold_days": 7,
        "entry_date": "2024-01-15",
        "status": status,
    }
    base.update(kwargs)
    return base


def make_signal(**kwargs) -> dict:
    """Test için örnek sinyal dict'i üretir (scanner formatı)."""
    base = {
        "ticker": "SIGTEST",
        "entry_price": 15.0,
        "stop_loss": 13.5,      # %10 risk
        "target_1": 19.5,       # %30 reward
        "target": 19.5,
        "atr": 0.75,
        "quality_score": 8.0,
        "swing_type": "B",
        "hold_days_max": 8,
        "max_hold_days": 8,
        "date": "2024-03-01",
    }
    base.update(kwargs)
    return base


# ─────────────────────────────────────────────────
# Test Sınıfı: Feature Engineering
# ─────────────────────────────────────────────────

class TestFeatureEngineer:
    """features.py — FeatureEngineer testleri."""

    def test_transform_returns_correct_shape(self):
        """
        9 özellik var, 3 trade → shape (3, 9) olmalı.
        """
        from swing_trader.ml.features import FeatureEngineer, FEATURE_COLUMNS

        engineer = FeatureEngineer()
        trades = [
            make_trade("TARGET"),
            make_trade("STOPPED"),
            make_trade("TRAILED"),
        ]
        X, y = engineer.transform(trades)

        assert X.shape == (3, len(FEATURE_COLUMNS)), \
            f"Beklenen shape (3, {len(FEATURE_COLUMNS)}), alınan: {X.shape}"
        assert len(y) == 3

    def test_win_label_correct(self):
        """
        TARGET ve TRAILED → label 1 (WIN)
        STOPPED, TIMEOUT, MANUAL → label 0 (LOSS)
        """
        from swing_trader.ml.features import FeatureEngineer

        engineer = FeatureEngineer()
        trades = [
            make_trade("TARGET"),   # WIN → 1
            make_trade("TRAILED"),  # WIN → 1
            make_trade("STOPPED"),  # LOSS → 0
            make_trade("TIMEOUT"),  # LOSS → 0
            make_trade("MANUAL"),   # LOSS → 0
        ]
        _, y = engineer.transform(trades)

        assert list(y) == [1, 1, 0, 0, 0], f"Label listesi yanlış: {list(y)}"

    def test_risk_reward_calculation(self):
        """
        entry=10, stop=9, target=13 için:
        risk_pct = 10%, reward_pct = 30%, rr = 3.0
        """
        from swing_trader.ml.features import FeatureEngineer

        engineer = FeatureEngineer()
        trade = make_trade("TARGET", entry_price=10, stop_loss=9, target=13)
        X, _ = engineer.transform([trade])

        assert abs(X["risk_pct"].iloc[0] - 10.0) < 0.01, "risk_pct hatalı"
        assert abs(X["reward_pct"].iloc[0] - 30.0) < 0.01, "reward_pct hatalı"
        assert abs(X["risk_reward_ratio"].iloc[0] - 3.0) < 0.01, "rr hatalı"

    def test_empty_input(self):
        """Boş liste verilirse boş DataFrame dönmeli."""
        from swing_trader.ml.features import FeatureEngineer

        engineer = FeatureEngineer()
        X, y = engineer.transform([])

        assert X.empty, "Boş input → boş DataFrame bekleniyor"
        assert y is None

    def test_transform_signal_single_row(self):
        """
        Sinyal için transform_signal → tek satırlı DataFrame dönmeli.
        """
        from swing_trader.ml.features import FeatureEngineer

        engineer = FeatureEngineer()
        signal = make_signal()
        X = engineer.transform_signal(signal)

        assert len(X) == 1, "Tek sinyal → 1 satır bekleniyor"
        assert not X.isnull().any().any(), "NaN değer olmamalı"

    def test_swing_type_encoding(self):
        """
        A→0, B→1, C→2, S→3 encode edilmeli.
        """
        from swing_trader.ml.features import FeatureEngineer

        engineer = FeatureEngineer()
        for swing_type, expected_enc in [("A", 0), ("B", 1), ("C", 2), ("S", 3)]:
            trade = make_trade("TARGET", swing_type=swing_type)
            X, _ = engineer.transform([trade])
            assert X["swing_type_enc"].iloc[0] == expected_enc, \
                f"Swing type {swing_type} → {expected_enc} bekleniyor"

    def test_no_nan_in_output(self):
        """Feature matrix'te NaN olmamalı."""
        from swing_trader.ml.features import FeatureEngineer

        engineer = FeatureEngineer()
        # Eksik değerli trade
        trade = make_trade("TARGET", atr=None, quality_score=None)
        X, _ = engineer.transform([trade])

        assert not X.isnull().any().any(), "NaN değer var!"


# ─────────────────────────────────────────────────
# Test Sınıfı: Predictor
# ─────────────────────────────────────────────────

class TestSignalPredictor:
    """predictor.py — SignalPredictor testleri."""

    def test_predictor_graceful_no_model(self, tmp_path, monkeypatch):
        """
        Model dosyası yoksa is_ready=False olmalı, hata vermemeli.
        """
        import swing_trader.ml.predictor as pred_module
        # Model path'i geçici klasöre yönlendir (model yok)
        monkeypatch.setattr(pred_module, "MODEL_PATH", tmp_path / "no_model.pkl")

        from swing_trader.ml.predictor import SignalPredictor
        predictor = SignalPredictor()

        assert not predictor.is_ready
        assert predictor.predict(make_signal()) is None

    def test_predict_returns_expected_keys(self, tmp_path, monkeypatch):
        """
        (Model varsa) predict() şu anahtarları içermeli:
        win_probability, confidence, label, top_features
        """
        pytest.importorskip("xgboost")  # xgboost yoksa skip et

        # Küçük bir model eğitip test et
        from swing_trader.ml.features import FeatureEngineer
        from sklearn.dummy import DummyClassifier
        import joblib

        # Dummy model (her zaman WIN der)
        dummy = DummyClassifier(strategy="most_frequent")
        from sklearn.datasets import make_classification
        Xd, yd = make_classification(n_samples=20, n_features=9, random_state=42)
        dummy.fit(Xd, yd)

        model_path = tmp_path / "dummy.pkl"
        joblib.dump(dummy, model_path)

        import swing_trader.ml.predictor as pred_module
        monkeypatch.setattr(pred_module, "MODEL_PATH", model_path)
        monkeypatch.setattr(pred_module, "META_PATH", tmp_path / "meta.json")

        from swing_trader.ml.predictor import SignalPredictor
        predictor = SignalPredictor()

        if predictor.is_ready:
            result = predictor.predict(make_signal())
            assert result is not None
            for key in ("win_probability", "confidence", "label"):
                assert key in result, f"'{key}' anahtarı eksik"
            assert 0.0 <= result["win_probability"] <= 1.0


# ─────────────────────────────────────────────────
# Test Sınıfı: Trainer (hafif)
# ─────────────────────────────────────────────────

class TestSignalTrainer:
    """trainer.py — SignalTrainer testleri (xgboost gerektirir)."""

    def test_load_data_insufficient(self, monkeypatch):
        """
        Yeterli veri yoksa (< MIN_TRADES_REQUIRED) None dönmeli.
        """
        import swing_trader.ml.trainer as trainer_module
        monkeypatch.setattr(trainer_module, "MIN_TRADES_REQUIRED", 9999)

        from swing_trader.ml.trainer import SignalTrainer
        trainer = SignalTrainer()
        X, y, count = trainer.load_data()

        assert X is None, "Yetersiz veri → X=None bekleniyor"
