"""
sanitize_for_json — NaN/Infinity koruması (2026-07-26).

Canlı olay: POST /api/trades/update-prices 500 döndü —
"Out of range float values are not JSON compliant". NaN fiyattan hesaplanan
unrealized_pnl NaN/Infinity oluyor ve JSON standardı bunları kabul etmiyor.
Eski sanitize yalnız NaN'ı yakalıyordu (Infinity'yi değil) ve /update-prices
endpoint'i sanitize'dan hiç geçmiyordu. Bu testler her iki açığı da kapatır.
"""

import json
import math

from api.utils import sanitize_for_json


def _json_safe(obj):
    """sanitize sonrası gerçekten json.dumps edilebiliyor mu?"""
    json.dumps(sanitize_for_json(obj))
    return True


def test_native_nan_becomes_none():
    assert sanitize_for_json(float("nan")) is None


def test_native_positive_infinity_becomes_none():
    assert sanitize_for_json(float("inf")) is None


def test_native_negative_infinity_becomes_none():
    assert sanitize_for_json(float("-inf")) is None


def test_finite_floats_preserved():
    assert sanitize_for_json(3.14) == 3.14
    assert sanitize_for_json(-100.0) == -100.0
    assert sanitize_for_json(0.0) == 0.0


def test_numpy_nan_and_inf():
    import numpy as np
    assert sanitize_for_json(np.float64("nan")) is None
    assert sanitize_for_json(np.float64("inf")) is None
    assert sanitize_for_json(np.float64(2.5)) == 2.5


def test_nested_trade_dict_with_bad_pnl_is_json_serializable():
    # /update-prices'ın döndürdüğü yapıyı taklit et: bir trade'de NaN/Inf P&L
    trade = {
        "message": "Updated 2 trades",
        "trades": [
            {"ticker": "AAA", "unrealized_pnl": float("nan"),
             "unrealized_pnl_pct": float("inf"), "current_price": 12.5},
            {"ticker": "BBB", "unrealized_pnl": 42.0, "current_price": 8.0},
        ],
    }
    out = sanitize_for_json(trade)
    assert _json_safe(trade)  # ham hali sanitize'dan geçince serileşebilir
    assert out["trades"][0]["unrealized_pnl"] is None
    assert out["trades"][0]["unrealized_pnl_pct"] is None
    assert out["trades"][1]["unrealized_pnl"] == 42.0
