"""
Skor değiştiricileri ölçüldü — bu test kararı KİLİTLER (2026-08-05).

Neden test yazıyoruz: 34 bonus/ceza 21 ay boyunca hiç ölçülmeden skorun üstüne
bindi. Ölçünce çıktı ki 14 bonusun ayırt etme gücü SIFIR (tavan sinyallerin
%100'ünde bağlıyordu) ve 21 cezanın 16'sı ya hiç ateşlemiyor ya da yönü ters.
Silindiler. Bu test onların **geri sızmasını** engelliyor — kanıt
GATE_AUDIT.md "3. tur", harness scripts/measure_score_modifiers.py.

Aynı zamanda ölü ayar tuzağını kapatıyor: canlıda DB'de duran eski yamalarda bu
anahtarlar hâlâ var. `extra="forbid"` yüzünden prune çalışmazsa deploy anında
ayarlar sessizce varsayılana düşer (2026-08-03'te auto-scan böyle kapanmıştı).
"""

import inspect

import pytest

from swing_trader.small_cap import scoring as scoring_mod
from swing_trader.small_cap.settings_config import (
    _REMOVED_KEYS,
    _prune_removed_keys,
    load_settings,
)
from swing_trader.small_cap.settings_models_extra import ScoringTuningSettings

# Ölçüm sonucu KALAN cezalar — çıkarınca EV düşüyor (yani koruyorlar)
KEPT_PENALTIES = {
    "pen_a_rsi_gt_70",   # 28 ateşleme, ΔEV -0.07
    "pen_a_rsi_gt_65",   # 4  ateşleme, ΔEV -0.09
    "pen_c_rsi_gt_65",   # 17 ateşleme, ΔEV -0.16
    "pen_c_rsi_gt_60",   # 12 ateşleme, ΔEV -0.20
    "pen_today_gt_10",   # 3  ateşleme, ΔEV -0.10
}

# Ölçüm sonucu SİLİNEN değiştiriciler
DELETED = {
    # 14 bonus — bonus_cap %100 bağlıyordu, marjinal etki 0.00
    "bonus_high_rvol", "bonus_gap_continuation", "bonus_higher_highs",
    "bonus_swing_ready", "bonus_higher_lows", "bonus_multi_day_volume",
    "bonus_surge_days_3", "bonus_surge_days_2", "bonus_early_entry_lo",
    "bonus_early_entry_hi", "bonus_early_entry_pts", "bonus_very_early_hi",
    "bonus_very_early_pts", "bonus_rsi_divergence", "bonus_golden_cross",
    "bonus_confirmed_breakout", "bonus_volume_on_up_day",
    # 16 ceza — 10'u hiç ateşlemedi, 6'sının yönü tersti
    "pen_b_rsi_gt_85", "pen_b_rsi_gt_80", "pen_b_rsi_gt_75",
    "pen_ext_day_gt_25", "pen_ext_day_gt_20", "pen_today_gt_15",
    "pen_5d_gt_40", "pen_5d_gt_30", "pen_5d_gt_25",
    "pen_parabolic", "parabolic_day3_min_pct", "pen_not_swing_ready",
    "pen_obv_distribution", "pen_below_ma50", "pen_ma20_falling",
    "pen_rejection_candle", "pen_weak_trend_phase", "pen_spread_risk",
}


# ── Silinenler gerçekten gitti mi ──────────────────────────────────────────

@pytest.mark.parametrize("field", sorted(DELETED))
def test_deleted_modifier_is_not_a_setting(field):
    """Ölçülüp silinen değiştirici ayar modeline geri eklenmemeli."""
    assert field not in ScoringTuningSettings.model_fields, (
        f"{field} geri gelmiş. Geri eklemek için ÖLÇÜM gerekir — "
        "scripts/measure_score_modifiers.py, kanıt GATE_AUDIT.md '3. tur'."
    )


@pytest.mark.parametrize("field", sorted(DELETED))
def test_deleted_modifier_is_pruned_from_old_patches(field):
    """Canlıdaki eski DB/JSON yamaları deploy'u kırmamalı — prune atmalı."""
    assert f"scoring_tuning.{field}" in _REMOVED_KEYS
    pruned = _prune_removed_keys({"scoring_tuning": {field: 99}})
    assert field not in pruned["scoring_tuning"]


def test_stale_patch_with_all_deleted_keys_still_loads():
    """
    En kötü senaryo: DB'deki yama 33 ölü anahtarın hepsini içeriyor.
    Prune sonrası model doğrulaması geçmeli (aksi halde TÜM ayarlar
    varsayılana düşer — 2026-08-03 auto-scan arızasının aynısı).
    """
    stale = {"scoring_tuning": {f: 42 for f in DELETED}}
    stale["scoring_tuning"]["bonus_cap"] = 30
    pruned = _prune_removed_keys(stale)
    tuning = ScoringTuningSettings(**pruned["scoring_tuning"])
    assert tuning.bonus_cap == 30


# ── Kalanlar gerçekten okunuyor mu (ölü ayar tuzağı) ──────────────────────

@pytest.mark.parametrize("field", sorted(KEPT_PENALTIES))
def test_kept_penalty_is_read_by_scoring(field):
    """Kalan her ceza scoring.py tarafından OKUNMALI — ölü ayar bırakmıyoruz."""
    assert field in ScoringTuningSettings.model_fields
    src = inspect.getsource(scoring_mod)
    assert f"st.{field}" in src, f"{field} ayar olarak var ama kod okumuyor (ölü ayar)"


def test_scoring_reads_exactly_the_measured_penalties():
    """Ne eksik ne fazla: koddaki ceza kümesi = ölçümde 'KAL' çıkan küme."""
    import re
    src = inspect.getsource(scoring_mod)
    found = set(re.findall(r"st\.(pen_\w+)", src))
    assert found == KEPT_PENALTIES, (
        f"fazla: {sorted(found - KEPT_PENALTIES)} | eksik: {sorted(KEPT_PENALTIES - found)}"
    )


# ── Bonus artık sabit ─────────────────────────────────────────────────────

def test_bonus_is_a_flat_constant():
    """
    Bonus koşullu olmamalı: kod içinde `st.bonus_*` yalnızca bonus_cap olabilir.
    Koşullu bonus geri gelirse tavan yine %100 bağlar ve etki yine sıfır olur —
    ama okuyan kişi "ayarlanabilir" sanır. Ölçüm bunu yasakladı.
    """
    import re
    src = inspect.getsource(scoring_mod)
    assert set(re.findall(r"st\.(bonus_\w+)", src)) == {"bonus_cap"}


def test_bonus_cap_equals_the_measured_constant():
    """Sabit +30 ölçümün yapıldığı değer; değişirse tüm eşikler kayar."""
    assert load_settings().scoring_tuning.bonus_cap == 30
