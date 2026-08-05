"""
Motorun ürettiği her alanın bir tüketicisi olmalı (2026-08-05, 4. tur denetim).

Ölü çıktı sessizce birikir: engine bir alanı hesaplar, tüketicisi zamanla
silinir, alan kalır. İki somut zararı gördük:

1. `detect_pullback_setup` (157 satır) HER taranan hissede koşuyordu ama üç
   çıktısını kimse okumuyordu — saf israf.
2. `vce_premium` / `vce_tight_coil` bunun TERSİ hatasıydı: skoru gerçekten
   kaydıran iki işaret `boosters`'a yazılıp sinyal sözlüğüne hiç konmamıştı.
   Sonuç: UI gösteremiyordu ve ölçüm harness'i False okuyup "bu özellik hiç
   ateşlemiyor" sonucuna vardı — yanlış atıf. (GATE_AUDIT.md "4. tur")

Bu test iki yönü de bağlar: silinenler geri gelmesin, canlı olanlar kaybolmasın.
"""

import inspect
import re

import pytest

from swing_trader.small_cap import engine as engine_mod
from swing_trader.small_cap import patterns, regime_logic, trend_quality

ENGINE_SRC = inspect.getsource(engine_mod)

# Ölçülüp/doğrulanıp silinen çıktılar — geri gelirlerse tüketicisi de gelmeli
DELETED_OUTPUTS = [
    # VCP (Minervini): 5 çıktısının hiçbiri okunmuyordu
    "vcp_detected", "vcp_contractions", "vcp_final_range_pct",
    "vcp_volume_declining", "vcp_bonus",
    # Weinstein telemetrisi — gate `stage`'i kendi değişkeninden okuyor
    "weinstein_bonus", "weinstein_ma30", "weinstein_stage_label",
    # Pullback: 157 satırlık ölü hesap (R5 edge +0.29%, t=0.65 — anlamsız)
    "pullback_detected", "pullback_quality", "pullback_bonus",
    # Katalizör modülü 2026-08-04'te silindi → bunlar hep 0/False'tu
    "insider_bonus", "news_bonus", "short_interest_bonus",
    "is_squeeze_candidate", "has_recent_news", "has_insider_buying",
    "total_catalyst_bonus", "has_catalyst",
    # Bonus bloğu sabitlenince tüketicisi kalmadı
    "sector_rs_bonus", "rsi_divergence_confidence",
]


@pytest.mark.parametrize("key", DELETED_OUTPUTS)
def test_deleted_output_stays_deleted(key):
    assert not re.search(r"\b" + key + r"\b", ENGINE_SRC), (
        f"{key} engine.py'ye geri gelmiş. Geri eklemek için bir TÜKETİCİ gerekir "
        "(UI, narrative, tracker veya ölçüm) — yoksa her taramada boşuna hesaplanır."
    )


@pytest.mark.parametrize("fn_name, module", [
    ("detect_vcp", patterns),
    ("detect_pullback_setup", None),   # signals.py — aşağıda ayrı kontrol
])
def test_deleted_function_stays_deleted(fn_name, module):
    if module is not None:
        assert not hasattr(module, fn_name), f"{fn_name} geri gelmiş"
    else:
        from swing_trader.small_cap.signals import SmallCapSignals
        assert not hasattr(SmallCapSignals, fn_name), f"{fn_name} geri gelmiş"


# ── Canlı olan çıktılar kaybolmasın ───────────────────────────────────────

@pytest.mark.parametrize("key", [
    "vce_premium",       # +8 skor kaydırıyor → UI göstermeli, ölçüm okumalı
    "vce_tight_coil",    # +5 skor kaydırıyor
    "sector_rs_score",   # narrative + tarayıcı UI
    "is_sector_leader",  # UI "Lider!" etiketi
])
def test_live_output_is_in_signal_dict(key):
    """Sözlüğe yazılmayan alan UI'da görünmez ve ölçümde False okunur."""
    assert f"'{key}':" in ENGINE_SRC, (
        f"{key} sinyal sözlüğünden düşmüş — boosters'a yazmak YETMEZ, "
        "tüketiciler sözlüğü okur (bu tam olarak 2026-08-05'te bulunan hataydı)."
    )


def test_weinstein_gate_still_reads_stage():
    """Ölçülmüş EN GÜÇLÜ kapı (ΔEV −0.98) — telemetri temizliğinde bozulmamalı."""
    assert "detect_weinstein_stage" in ENGINE_SRC
    assert "reject_stage3" in ENGINE_SRC and "reject_stage4" in ENGINE_SRC
    out = patterns.detect_weinstein_stage(None)
    assert "stage" in out and "bonus" not in out


def test_trend_quality_keeps_only_consumed_fields():
    """score_trend_quality'nin okuduğu alanlar dursun, ölü composite gitsin."""
    out = trend_quality.calculate_trend_quality(None)
    for k in ("ma20_slope_ok", "ma20_slope_value", "ma50_distance_pct",
              "golden_cross", "trend_phase", "higher_lows_count",
              "rejection_candle", "ma50_ok"):
        assert k in out, f"{k} scoring/signals tarafından okunuyor, silinemez"
    for k in ("trend_strength", "higher_highs_count"):
        assert k not in out, f"{k} hiçbir tüketicisi olmayan ölü çıktı"


def test_relative_strength_returns_only_consumed_fields():
    out = regime_logic.relative_strength_vs_spy(None, None)
    assert set(out) == {"rs_score", "is_leader"}
    assert not hasattr(regime_logic, "rs_bonus_vs_spy"), (
        "Eski ad geri gelmiş; kademeli `bonus` çıktısının tüketicisi yok."
    )
