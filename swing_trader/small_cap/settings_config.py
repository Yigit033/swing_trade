"""
Small-cap runtime settings schema, defaults, and JSON persistence (Step 1).

Load merges file contents over defaults (deep). Save writes the full validated model.
Step 2+ will wire engine/risk/signals/backtest to these values.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .settings_models_extra import (
    AutoScanSettings,
    BacktestEntrySettings,
    BacktestExitTrailingSettings,
    BacktestLoopSettings,
    BacktestTypeQualityOverride,
    RiskTargetRegimeSettings,
    ScanStockGatesSettings,
    ScoringTuningSettings,
    SignalsConfirmationSettings,
    SwingEngineSettings,
    UniverseFilterSettings,
    UniverseScanSettings,
)

logger = logging.getLogger(__name__)

_SWING_TRADER_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SWING_TRADER_ROOT.parent
DEFAULT_SETTINGS_PATH = _PROJECT_ROOT / "data" / "smallcap_settings.json"

_SWING_KEYS = frozenset({"C", "A", "B", "S"})


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class TypeTargetCaps(BaseModel):
    """Max T1/T2 as fraction of entry (ceiling)."""

    model_config = ConfigDict(extra="forbid")

    t1_max_pct: float = Field(..., ge=0.01, le=0.5)
    t2_max_pct: float = Field(..., ge=0.01, le=0.8)


class RegimeThresholds(BaseModel):
    """Floors for min_quality and caps for top_n (see thresholds.effective_scan_thresholds).

    2026-07-27: BULL floor eklendi + değerler kanıta göre ayarlandı
    (scripts/measure_score_edge.py, 469 gerçek sinyal). Eski durumda BULL'un
    hiç tabanı yoktu (regime_min=0), efektif eşik UI değerine (65) ve
    engine.py'nin ayrı 60 eşiğine düşüyordu → Q60-70 bandı (~0% getiri)
    kullanıcıya gösteriliyordu. Ölçülen tatlı noktalar:
      BULL:    Q78 → EV +3.86% WR %60 (67 sinyal, hâlâ bol) [Q60: +0.81% %48]
      CAUTION: hiçbir eşik kârlı değil (Q80'de bile −0.76%) → en yükseğe çek,
               sistem temkinli piyasada işlemden korusun
      BEAR:    Q78 → EV +10.2% WR %80 (zaten güçlü; doğal olarak az sinyal)
    """

    model_config = ConfigDict(extra="forbid")

    # BULL floor — eskiden yoktu (asıl "değersiz sinyal gösterme" kaynağı buydu)
    bull_min_quality: int = Field(default=78, ge=50, le=100)
    bull_top_n_max: int = Field(default=10, ge=1, le=50)
    bear_tentative_min_quality: int = Field(default=80, ge=50, le=100)
    bear_tentative_top_n_max: int = Field(default=4, ge=1, le=50)
    bear_confirmed_min_quality: int = Field(default=82, ge=50, le=100)
    bear_confirmed_top_n_max: int = Field(default=3, ge=1, le=50)
    caution_confirmed_min_quality: int = Field(default=82, ge=50, le=100)
    caution_confirmed_top_n_max: int = Field(default=3, ge=1, le=50)
    caution_other_min_quality: int = Field(default=80, ge=50, le=100)
    caution_other_top_n_max: int = Field(default=4, ge=1, le=50)


class SmallCapSettings(BaseModel):
    """
    Serializable small-cap parameters (defaults match current hardcoded engine/backtest values).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=3, ge=1, le=999)

    # --- Signal / filter (live scan + shared semantics) ---
    max_entry_rsi: int = Field(
        default=70,
        ge=30,
        le=95,
        description="Hard reject in scan_stock when RSI above this (Type S exempt).",
    )
    volume_surge_trigger: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Minimum volume vs baseline to pass check_all_triggers.",
    )
    volume_surge_baseline_days: int = Field(
        default=50,
        ge=10,
        le=200,
        description=(
            "Lookback window for volume baseline median. "
            "50d (not 20d) captures pre-rally levels so Finviz momentum stocks "
            "aren't penalised by their own recent elevated activity."
        ),
    )
    min_volume_surge_soft: float = Field(
        default=1.2,
        ge=1.0,
        le=3.0,
        description="Helper threshold in check_volume_surge messaging.",
    )
    min_atr_percent: float = Field(
        default=0.03,
        ge=0.01,
        le=0.2,
        description="Minimum ATR/close for filters and trigger (0.03 = 3%%).",
    )

    # --- Risk ---
    max_risk_per_trade: float = Field(default=0.015, ge=0.001, le=0.1)
    # v13.4: "let winners run" exit — stop 2.0 ATR (was 1.5), 20-day hold
    # (was 14). Validated on live Finviz universe (1606 VCE signals, OOS):
    # this exit profile lifts EV/trade from -0.26% (capped) to +0.94% with
    # 56% win rate. See scripts/exit_strategy_lab.py.
    # 2026-07-26: stop 2.0→2.5, cap'ler genişletildi. Exit ölçümü
    # (scripts/exit_lab_vce_rvol.py) canlı tracker mantığıyla doğruladı: geniş
    # ATR stop iyi trade'leri gürültüde stop'lamayı bırakınca EV+WR monoton
    # artıyor (VCE +1.97→+2.64%, WR %49→%60). Cap'ler tracker.MAX_STOP_BY_TYPE
    # ile AYNI olmalı (sinyal üretimi + confirm tutarlılığı).
    stop_atr_multiplier: float = Field(default=2.5, ge=0.5, le=5.0)
    min_stop_percent: float = Field(default=0.03, ge=0.01, le=0.25)
    max_stop_percent_fallback: float = Field(default=0.15, ge=0.03, le=0.3)
    max_holding_days: int = Field(default=20, ge=1, le=60)
    max_stop_by_type: Dict[str, float] = Field(
        default_factory=lambda: {"C": 0.14, "A": 0.15, "B": 0.16, "S": 0.18}
    )
    type_position_caps: Dict[str, float] = Field(
        default_factory=lambda: {"C": 0.25, "A": 0.25, "B": 0.20, "S": 0.15}
    )

    # --- Targets (ATR-based T1/T2) ---
    type_atr_multipliers: Dict[str, float] = Field(
        default_factory=lambda: {"S": 2.5, "B": 2.0, "A": 1.8, "C": 1.5}
    )
    t2_atr_ratio: float = Field(default=2.0, ge=1.0, le=4.0)
    # v13.4: T2 caps raised so the trailing stop — not a fixed cap — decides
    # the exit on winners. The old +28% cap on Type A/B was the single biggest
    # EV drag (it guillotined the runners that pay for the losers).
    type_target_caps: Dict[str, TypeTargetCaps] = Field(
        default_factory=lambda: {
            "S": TypeTargetCaps(t1_max_pct=0.12, t2_max_pct=0.65),
            "B": TypeTargetCaps(t1_max_pct=0.10, t2_max_pct=0.55),
            "C": TypeTargetCaps(t1_max_pct=0.08, t2_max_pct=0.45),
            "A": TypeTargetCaps(t1_max_pct=0.10, t2_max_pct=0.55),
        }
    )

    # --- Backtest entry / execution (also used for parity tuning later) ---
    min_rr_at_entry: float = Field(default=1.2, ge=0.5, le=10.0)
    min_rr_type_c: float = Field(default=1.5, ge=0.5, le=10.0)
    partial_at_t1_fraction: float = Field(default=0.5, ge=0.05, le=1.0)
    min_quality_type_c: int = Field(default=65, ge=30, le=100)
    min_quality_type_a: int = Field(default=60, ge=30, le=100)
    min_quality_type_b: int = Field(default=60, ge=30, le=100)
    max_gap_up_pct: float = Field(default=5.0, ge=0.0, le=30.0)
    max_gap_down_pct: float = Field(default=4.0, ge=0.0, le=30.0)
    max_loss_per_trade_pct: float = Field(default=0.07, ge=0.02, le=0.25)
    max_gap_risk_portfolio_pct: float = Field(default=0.02, ge=0.005, le=0.1)
    max_position_cost_portfolio_pct: float = Field(default=0.15, ge=0.05, le=0.5)
    cooldown_days: int = Field(default=5, ge=0, le=30)
    ticker_max_losses: int = Field(default=2, ge=1, le=10)
    slippage_bps_per_side: int = Field(default=5, ge=0, le=100)
    min_shares_for_partial: int = Field(default=2, ge=1, le=100)

    # --- Regime-driven scan floors (scanner / backtest) ---
    regime_thresholds: RegimeThresholds = Field(default_factory=RegimeThresholds)

    # --- Extended tuning (engine / risk / filters / signals / scoring / backtest) ---
    scan_gates: ScanStockGatesSettings = Field(default_factory=ScanStockGatesSettings)
    swing: SwingEngineSettings = Field(default_factory=SwingEngineSettings)
    risk_targets: RiskTargetRegimeSettings = Field(default_factory=RiskTargetRegimeSettings)
    universe_filters: UniverseFilterSettings = Field(default_factory=UniverseFilterSettings)
    universe_scan: UniverseScanSettings = Field(default_factory=UniverseScanSettings)
    auto_scan: AutoScanSettings = Field(default_factory=AutoScanSettings)
    signal_confirmation: SignalsConfirmationSettings = Field(default_factory=SignalsConfirmationSettings)
    scoring_tuning: ScoringTuningSettings = Field(default_factory=ScoringTuningSettings)
    backtest_loop: BacktestLoopSettings = Field(default_factory=BacktestLoopSettings)
    backtest_type_quality: BacktestTypeQualityOverride = Field(default_factory=BacktestTypeQualityOverride)
    backtest_entry: BacktestEntrySettings = Field(default_factory=BacktestEntrySettings)
    backtest_exit_trailing: BacktestExitTrailingSettings = Field(
        default_factory=BacktestExitTrailingSettings
    )

    @field_validator("max_stop_by_type", "type_atr_multipliers", "type_position_caps")
    @classmethod
    def _validate_swing_type_dict(cls, v: Dict[str, float]) -> Dict[str, float]:
        keys = set(v.keys())
        if keys != _SWING_KEYS:
            raise ValueError(f"Dict keys must be exactly {sorted(_SWING_KEYS)}, got {sorted(keys)}")
        return v

    @field_validator("type_target_caps")
    @classmethod
    def _validate_target_caps(cls, v: Dict[str, TypeTargetCaps]) -> Dict[str, TypeTargetCaps]:
        keys = set(v.keys())
        if keys != _SWING_KEYS:
            raise ValueError(f"type_target_caps keys must be exactly {sorted(_SWING_KEYS)}")
        return v

    @model_validator(mode="after")
    def _t2_ge_t1_caps(self) -> SmallCapSettings:
        for t, cap in self.type_target_caps.items():
            if cap.t2_max_pct < cap.t1_max_pct:
                raise ValueError(f"type_target_caps[{t}]: t2_max_pct must be >= t1_max_pct")
        return self


def default_settings() -> SmallCapSettings:
    """Fresh defaults (no file read)."""
    return SmallCapSettings()


# ── Koddan KALDIRILMIŞ ayar anahtarları ──────────────────────────────────
# Model extra="forbid" olduğu için, bir alan koddan kaldırıldığında eski
# JSON/DB yamalarında kalan anahtar doğrulamayı PATLATIR ve (kademeli geri
# çekilmeye rağmen) kalibrasyon kaybı riski doğar. Bu liste o anahtarları
# yükleme sırasında SESSİZCE ama LOGLANARAK atar.
# Biçim: "bolum.alan" (üst seviye alanlar için sadece "alan").
_REMOVED_KEYS = {
    # 2026-08-04: Q1/Q2/Q3 Finviz sorguları silindi (recall ölçümü: %0.5-2 katkı)
    "universe_scan.enable_finviz_query_momentum",
    "universe_scan.enable_finviz_query_setup",
    "universe_scan.enable_finviz_query_wider",
}


def _prune_removed_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Koddan kaldırılmış anahtarları at (yerinde değiştirmez)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    dropped = []
    for dotted in _REMOVED_KEYS:
        if "." in dotted:
            sec, key = dotted.split(".", 1)
            if isinstance(out.get(sec), dict) and key in out[sec]:
                out[sec].pop(key, None)
                dropped.append(dotted)
        elif dotted in out:
            out.pop(dotted, None)
            dropped.append(dotted)
    if dropped:
        logger.info("Kaldırılmış ayar anahtarları yok sayıldı: %s", ", ".join(sorted(dropped)))
    return out


def _db_overlay() -> Dict[str, Any]:
    """
    Kalıcı (DB) kullanıcı yamasını getir. DB yok/erişilemiyorsa {}.

    2026-08-03: fly.io'da mount olmadığı için dosyaya yazılan UI ayarları her
    deploy'da siliniyordu (kullanıcının açtığı auto-scan sessizce kapandı).
    Kalıcılık artık DB'de; dosya katmanı git-varsayılanı olarak kalıyor.
    """
    try:
        from swing_trader.data.settings_storage import load_patch

        return load_patch()
    except Exception as e:  # modül yokluğu / beklenmeyen hata ürünü kilitlemesin
        logger.debug("Settings DB overlay atlandı: %s", e)
        return {}


def load_settings(path: Optional[Path] = None) -> SmallCapSettings:
    """
    Katmanlı yükleme (en zayıftan en güçlüye):

        kod varsayılanları  →  JSON dosyası (git)  →  DB yaması (UI değişikliği)

    Dosya yoksa varsayılanlar, DB yoksa dosya katmanı kullanılır.
    """
    p = path or DEFAULT_SETTINGS_PATH
    base = SmallCapSettings().model_dump(mode="json")

    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("settings file must be a JSON object")
            base = _deep_merge(base, _prune_removed_keys(raw))
        except Exception as e:
            logger.error("Failed to load %s: %s — using defaults", p, e)
    else:
        logger.debug("Small-cap settings file missing, using defaults: %s", p)

    # Açık bir `path` verilmişse (testler, geçici dosyalar) DB katmanı bindirilmez —
    # o çağrı bilinçli olarak izole bir dosyayı okumak istiyordur.
    if path is None:
        overlay = _db_overlay()
        if overlay:
            base = _deep_merge(base, _prune_removed_keys(overlay))

    try:
        return SmallCapSettings.model_validate(base)
    except Exception as e:
        # KADEMELİ GERİ ÇEKİLME — hepsini birden varsayılana düşürmek TEHLİKELİ.
        # Senaryo: bir ayar alanı koddan kaldırılır (model extra="forbid"), ama
        # DB'deki eski kullanıcı yaması o alanı hâlâ taşır → doğrulama patlar →
        # sistem SESSİZCE tüm kalibrasyonu (eşikler, exit, evren) varsayılana
        # düşürür. Ölçülmüş her parametre bir anda kaybolur ve kimse fark etmez.
        # Bu yüzden önce sadece DB katmanını atarak dene; o da tutmazsa dosya
        # katmanını dene; en son çare varsayılan.
        logger.error("Settings validation failed (%s) — katmanlar tek tek geri çekiliyor", e)

        if path is None:
            try:
                file_only = SmallCapSettings().model_dump(mode="json")
                if p.exists():
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        file_only = _deep_merge(file_only, raw)
                s = SmallCapSettings.model_validate(file_only)
                logger.error(
                    "DB ayar yaması geçersiz — YOK SAYILDI, dosya katmanıyla devam "
                    "ediliyor. Yamayı düzeltmek için: POST /api/settings/reset"
                )
                return s
            except Exception as e2:
                logger.error("Dosya katmanı da geçersiz (%s)", e2)

        logger.error("TÜM KATMANLAR GEÇERSİZ — kod varsayılanlarına düşüldü (kalibrasyon KAYIP)")
        return SmallCapSettings.model_validate(SmallCapSettings().model_dump(mode="json"))


def save_settings(settings: SmallCapSettings, path: Optional[Path] = None) -> None:
    """Write full validated settings to JSON (atomic replace)."""
    p = path or DEFAULT_SETTINGS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    data = settings.model_dump(mode="json")
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(p)
    logger.info("Saved small-cap settings to %s", p)


def apply_settings_patch(
    patch: Dict[str, Any], path: Optional[Path] = None
) -> SmallCapSettings:
    """
    ``patch``'i mevcut ayarların üstüne birleştir, doğrula ve KALICI olarak sakla.

    Kalıcılık iki katmanlı:
      1. DB yaması (varsa) — deploy'lardan sağ çıkar, asıl kaynak burasıdır.
      2. JSON dosyası — yerel geliştirme ve DB'siz kurulum için.

    Sadece kullanıcının GÖNDERDİĞİ alanlar yama olarak saklanır (tam anlık görüntü
    değil). Böylece kod/git varsayılanları ileride değişirse (ör. ölçümle
    yükseltilen eşikler) kullanıcının dokunmadığı alanlar yeni değeri alır —
    donmuş eski bir kopya yeni kalibrasyonu sessizce ezmez.

    Raises pydantic.ValidationError if the merged result is invalid.
    """
    if not isinstance(patch, dict):
        raise TypeError("patch must be a dict")
    current = load_settings(path=path).model_dump(mode="json")
    merged = _deep_merge(current, patch)
    validated = SmallCapSettings.model_validate(merged)

    # Dosya katmanı (yerel geliştirme + DB'siz kurulum)
    save_settings(validated, path=path)

    # DB katmanı: birikimli yama (önceki yamanın üstüne bu yama)
    if path is None:
        try:
            from swing_trader.data.settings_storage import (
                is_enabled, load_patch, save_patch,
            )

            if is_enabled():
                cumulative = _deep_merge(load_patch(), patch)
                if not save_patch(cumulative):
                    logger.warning(
                        "Ayar DB'ye yazılamadı — yalnız dosyaya kaydedildi, "
                        "bir sonraki deploy'da kaybolabilir."
                    )
            else:
                logger.info(
                    "DATABASE_URL yok — ayar yalnız dosyaya kaydedildi "
                    "(fly.io'da deploy sonrası kaybolur)."
                )
        except Exception as e:
            logger.error("Ayar kalıcılığı (DB) başarısız: %s", e)

    return validated
