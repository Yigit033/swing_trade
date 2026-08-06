"""Step 1: settings JSON merge + roundtrip."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from swing_trader.small_cap.scoring import SmallCapScoring
from swing_trader.small_cap.settings_config import (
    SmallCapSettings,
    apply_settings_patch,
    default_settings,
    load_settings,
    save_settings,
)


def test_default_matches_known_engine_constants():
    d = default_settings()
    # 2026-08-04: late_entry_* / parabolic_* / extreme_* SİLİNDİ (ölü ayar +
    # ΔEV 0.00 ölçümü). Kalan iki scan_gate ÖLÇÜLDÜ ve işe yarıyor:
    assert d.scan_gates.reject_stage3 is True
    assert d.scan_gates.reject_stage4 is True
    assert d.swing.type_c.min_score == 10
    assert d.risk_targets.min_reward_risk_multiple_t1 == 1.5
    assert d.universe_filters.min_market_cap == 250_000_000
    assert d.universe_scan.max_scan_tickers == 260
    assert d.universe_scan.use_finviz is True
    assert d.universe_scan.min_finviz_tickers_skip_static_merge == 30
    # 2026-08-04: rank_weight_* / chase_penalty_* SİLİNDİ (composite sıralama
    # tek ölçüte indi — dolar-hacim). R:R kapısı da ölçülüp silindi.
    assert d.regime_thresholds.caution_other_min_quality == 80
    # 2026-08-06: kod varsayılanları ölçülmüş (dosya) değerlere hizalandı —
    # ayrışma canlıda gerçek zarar üretiyordu (dosya katmanı çökünce ürün
    # sessizce eski/gevşek varsayılanlara düşüyordu). Bkz.
    # test_defaults_match_measured.py — ayrışmayı kalıcı olarak yasaklıyor.
    assert d.backtest_type_quality.type_c_bear == 86
    assert len(d.scoring_tuning.volume_surge_tiers) == 8
    assert len(d.scoring_tuning.atr_percent_tiers) == 7
    assert len(d.scoring_tuning.float_millions_bands) == 5
    assert d.max_entry_rsi == 70
    assert d.volume_surge_trigger == 2.0
    assert d.min_atr_percent == 0.03
    # 2026-07-26: exit ölçümü sonrası geniş stop (bkz. exit_lab_vce_rvol.py)
    assert d.stop_atr_multiplier == 2.5
    assert d.max_stop_by_type["C"] == 0.14
    # Type S 2026-08-04'te kaldırıldı (girdisi geçmişe dönük yok, hiç doğrulanamadı)
    assert set(d.type_atr_multipliers) == {"A", "B", "C"}
    assert d.type_atr_multipliers["B"] == 2.0
    assert d.partial_at_t1_fraction == 0.33   # ölçüldü: monotonik, +0.42 puan
    # 2026-07-27: eşikler kanıta göre yükseltildi (measure_score_edge.py)
    assert d.regime_thresholds.bear_confirmed_min_quality == 82
    assert d.regime_thresholds.bull_min_quality == 78  # yeni: BULL floor


def test_load_partial_merge_over_defaults():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        p.write_text(json.dumps({"max_entry_rsi": 65, "volume_surge_trigger": 1.8}), encoding="utf-8")
        s = load_settings(p)
    assert s.max_entry_rsi == 65
    assert s.volume_surge_trigger == 1.8
    assert s.partial_at_t1_fraction == 0.33  # untouched default


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        a = default_settings().model_copy(update={"max_entry_rsi": 68})
        save_settings(a, path=p)
        b = load_settings(p)
    assert b.model_dump() == a.model_dump()


def test_invalid_file_falls_back_to_defaults():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.json"
        p.write_text("not json", encoding="utf-8")
        s = load_settings(p)
    assert s == default_settings()


def test_apply_settings_patch_persists_and_merges():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        save_settings(default_settings(), path=p)
        out = apply_settings_patch({"max_entry_rsi": 66, "regime_thresholds": {"bear_tentative_min_quality": 71}}, path=p)
        assert out.max_entry_rsi == 66
        assert out.regime_thresholds.bear_tentative_min_quality == 71
        # unrelated regime field preserved from previous merge
        assert out.regime_thresholds.bear_confirmed_min_quality == 82
        reloaded = load_settings(p)
        assert reloaded.max_entry_rsi == 66


def test_apply_settings_patch_validation_error():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        save_settings(default_settings(), path=p)
        with pytest.raises(ValidationError):
            apply_settings_patch({"max_entry_rsi": 999}, path=p)


def test_scoring_tier_methods_match_legacy_defaults():
    sc = SmallCapScoring(settings=default_settings())
    assert sc.score_volume_explosion(6.0) == 30
    assert sc.score_volume_explosion(5.0) == 26
    assert sc.score_volume_explosion(1.29) == 0
    assert sc.score_volatility_expansion(0.15) == 25
    assert sc.score_volatility_expansion(0.034) == 0
    assert sc.score_float_tightness(10e6) == 20
    assert sc.score_float_tightness(90e6) == -8
    assert sc.score_float_tightness(None) == 5


def test_empty_scoring_tier_lists_rejected():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        save_settings(default_settings(), path=p)
        with pytest.raises(ValidationError):
            apply_settings_patch({"scoring_tuning": {"volume_surge_tiers": []}}, path=p)


def test_universe_scan_price_band_validation():
    """
    2026-08-04: "rank ağırlıkları 1.0 olmalı" kuralı KALDIRILDI (ağırlıklar silindi).
    Kalan tek tutarlılık kuralı fiyat bandı: maks > min olmalı.
    """
    base = default_settings().model_dump(mode="json")
    with pytest.raises(ValidationError):
        SmallCapSettings.model_validate(
            {
                **base,
                "universe_scan": {
                    **base["universe_scan"],
                    "post_filter_price_min": 200.0,
                    "post_filter_price_max": 7.0,
                },
            }
        )


def test_nested_swing_patch_merges():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        save_settings(default_settings(), path=p)
        out = apply_settings_patch({"swing": {"type_c": {"min_score": 12}}}, path=p)
        assert out.swing.type_c.min_score == 12
        assert out.swing.type_c.hold_min == 3
        reloaded = load_settings(p)
        assert reloaded.swing.type_c.min_score == 12
