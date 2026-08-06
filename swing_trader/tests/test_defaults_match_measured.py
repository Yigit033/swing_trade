"""
KOD VARSAYILANI == ÖLÇÜLMÜŞ DEĞER (2026-08-06).

Neden bu test var — canlı arıza gösterdi:

Ayar dosyası (`data/smallcap_settings.json`) ölçülmüş değerleri taşıyordu ama
KOD VARSAYILANLARI eski/gevşek değerlerde kalmıştı. 16 alanda ayrışma vardı:

    universe_filters.min_price          kod $3     vs  ölçülmüş $7
    universe_filters.max_float_shares   kod 150M   vs  ölçülmüş 80M
    max_gap_down_pct                    kod 4.0    vs  ölçülmüş 7.0
    min_quality_type_a/b/c              kod 60/65  vs  ölçülmüş 70
    backtest_type_quality.type_a_bear   kod 72     vs  ölçülmüş 90
    ... (t2_atr_ratio, max_position_cost_portfolio_pct, auto_scan.min_quality)

Normalde görünmüyordu çünkü dosya katmanı varsayılanların üstüne biniyordu.
Ama 2026-08-05'te ayar doğrulaması çöktü ve sistem KOD VARSAYILANLARINA düştü
("TÜM KATMANLAR GEÇERSİZ — kalibrasyon KAYIP"). O sırada ürün $3 penny
stock'ları evrene alıyor, float tavanını 150M'e açıyor ve gap-down'ı %4'te
kesiyordu — hiçbiri ölçülmüş değer değil ve hiçbiri log'a "dikkat" diye
yazılmıyordu.

Ders: bir değerin iki kaynağı varsa, er ya da geç ayrışır ve ayrıştığında
YANLIŞ olan kazanır. Varsayılan = ölçülmüş değer olmalı; dosya yalnız
kullanıcının bilinçli sapmalarını taşımalı.
"""

import json
from pathlib import Path

import pytest

from swing_trader.small_cap.settings_config import (
    DEFAULT_SETTINGS_PATH,
    SmallCapSettings,
    _prune_removed_keys,
)

# Kullanıcı TERCİHİ olan alanlar — varsayılanla farklı olmaları normaldir.
# auto_scan.enabled varsayılanı bilinçli False (güvenli); kullanıcının açtığı
# değer DB yamasında yaşar.
USER_PREFERENCE_FIELDS = {"auto_scan.enabled"}


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, pre + k + "."))
        else:
            out[pre + k] = v
    return out


def _file_layer():
    p = Path(DEFAULT_SETTINGS_PATH)
    if not p.exists():
        pytest.skip("ayar dosyası yok")
    return _prune_removed_keys(json.loads(p.read_text(encoding="utf-8")))


def test_no_default_drifts_from_the_file():
    """
    Dosyadaki her skaler değer, kod varsayılanıyla AYNI olmalı. Farklıysa
    dosya kaybolduğunda ürün sessizce başka bir kalibrasyonla çalışır.
    """
    defaults = _flat(SmallCapSettings().model_dump(mode="json"))
    file_vals = _flat(_file_layer())

    drift = {
        k: (defaults[k], v)
        for k, v in file_vals.items()
        if k in defaults and defaults[k] != v and k not in USER_PREFERENCE_FIELDS
    }
    assert not drift, (
        "kod varsayılanı ile dosya ayrışmış — dosya katmanı kaybolursa ürün "
        f"sessizce farklı davranır:\n" +
        "\n".join(f"  {k}: kod={c!r} dosya={f!r}" for k, (c, f) in sorted(drift.items()))
    )


def test_losing_the_file_layer_does_not_change_calibration(monkeypatch):
    """
    ASIL SENARYO: dosya katmanı tamamen kaybolsa (bozuk JSON, silinmiş dosya,
    doğrulama çökmesi) ürün AYNI ayarlarla çalışmaya devam etmeli.
    """
    from swing_trader.small_cap import settings_config as sc

    monkeypatch.setattr(sc, "_db_overlay", lambda: {})
    sc.invalidate_settings_cache()
    with_file = sc.load_settings().model_dump(mode="json")

    monkeypatch.setattr(sc, "DEFAULT_SETTINGS_PATH", Path("/bu/dosya/yok.json"))
    sc.invalidate_settings_cache()
    without_file = sc.load_settings().model_dump(mode="json")
    sc.invalidate_settings_cache()

    a, b = _flat(with_file), _flat(without_file)
    diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    diff = {k: v for k, v in diff.items() if k not in USER_PREFERENCE_FIELDS}
    assert not diff, (
        "dosya kaybolunca kalibrasyon değişti:\n" +
        "\n".join(f"  {k}: dosyalı={x!r} dosyasız={y!r}" for k, (x, y) in sorted(diff.items()))
    )


# ── Ölçülmüş değerlerin kendisi (kritik olanlar açıkça kilitli) ──────────

@pytest.mark.parametrize("dotted,expected,kaynak", [
    ("universe_filters.min_price", 7.0, "measure_price_band.py — $3 penny bandı elendi"),
    ("universe_filters.max_price", 200.0, "evren bandı"),
    ("universe_filters.max_float_shares", 80_000_000, "float tavanı (advisory)"),
    ("max_gap_down_pct", 7.0, "measure_gap_filter.py — 5→7 gevşetildi, R10 +1.86→+2.24"),
    ("max_gap_up_pct", 5.0, "measure_gap_filter.py — pump-open koruması"),
    ("partial_at_t1_fraction", 0.33, "measure_t1_fraction.py — %50→%33, +0.42 puan"),
    ("stop_atr_multiplier", 2.5, "exit_lab_vce_rvol.py — 1.5→2.5, VCE EV +1.97→+2.64"),
    ("max_holding_days", 20, "exit ölçümü"),
    ("max_risk_per_trade", 0.015, "risk bütçesi"),
])
def test_measured_value_is_the_code_default(dotted, expected, kaynak):
    """Bu değerler ölçümle belirlendi; varsayılan onlardan sapmamalı."""
    got = _flat(SmallCapSettings().model_dump(mode="json"))[dotted]
    assert got == expected, f"{dotted}={got}, beklenen {expected} ({kaynak})"
