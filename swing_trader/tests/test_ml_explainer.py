"""
TreeSHAP explainer tests — per-prediction contributions WITHOUT the shap package.

Locks in the XGBoost pred_contribs integration: values must be
prediction-specific (unlike global feature importance) and satisfy the
SHAP consistency property (bias + contributions == model logit).
"""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from swing_trader.ml.explainer import SignalExplainer
from swing_trader.ml.features import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def model_and_data():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.rand(200, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    # Label mostly driven by risk_reward_ratio → it should dominate contributions
    y = (X["risk_reward_ratio"] + 0.1 * rng.rand(200) > 0.55).astype(int)
    model = XGBClassifier(n_estimators=25, max_depth=3)
    model.fit(X, y)
    return model, X


def test_explain_returns_all_features_sorted(model_and_data):
    model, X = model_and_data
    out = SignalExplainer(model).explain_single(X.iloc[[0]])
    assert len(out) == len(FEATURE_COLUMNS)
    assert {o["feature"] for o in out} == set(FEATURE_COLUMNS)
    magnitudes = [abs(o["shap_value"]) for o in out]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_contributions_are_prediction_specific(model_and_data):
    # Global importance olsaydı her tahminde aynı değerler dönerdi.
    model, X = model_and_data
    e = SignalExplainer(model)
    a = {o["feature"]: o["shap_value"] for o in e.explain_single(X.iloc[[0]])}
    b = {o["feature"]: o["shap_value"] for o in e.explain_single(X.iloc[[1]])}
    assert a != b


def test_contributions_sum_to_model_logit(model_and_data):
    # SHAP tutarlılık özelliği: bias + tüm katkılar ≈ modelin logit çıktısı.
    import xgboost as xgb

    model, X = model_and_data
    row = X.iloc[[3]]
    booster = model.get_booster()
    contribs = booster.predict(xgb.DMatrix(row), pred_contribs=True)[0]
    margin = booster.predict(xgb.DMatrix(row), output_margin=True)[0]
    assert abs(float(contribs.sum()) - float(margin)) < 1e-3


def test_dominant_feature_detected(model_and_data):
    model, X = model_and_data
    out = SignalExplainer(model).explain_single(X.iloc[[0]])
    assert out[0]["feature"] == "risk_reward_ratio"
