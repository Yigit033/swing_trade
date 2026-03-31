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
    """Floors for min_quality and caps for top_n (see thresholds.effective_scan_thresholds)."""

    model_config = ConfigDict(extra="forbid")

    bear_tentative_min_quality: int = Field(default=70, ge=50, le=100)
    bear_tentative_top_n_max: int = Field(default=4, ge=1, le=50)
    bear_confirmed_min_quality: int = Field(default=72, ge=50, le=100)
    bear_confirmed_top_n_max: int = Field(default=3, ge=1, le=50)
    caution_confirmed_min_quality: int = Field(default=68, ge=50, le=100)
    caution_confirmed_top_n_max: int = Field(default=4, ge=1, le=50)
    caution_other_min_quality: int = Field(default=63, ge=50, le=100)
    caution_other_top_n_max: int = Field(default=5, ge=1, le=50)


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
        default=1.5,
        ge=1.0,
        le=5.0,
        description="Minimum volume vs 20d avg to pass check_all_triggers.",
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
    stop_atr_multiplier: float = Field(default=1.5, ge=0.5, le=5.0)
    min_stop_percent: float = Field(default=0.03, ge=0.01, le=0.25)
    max_stop_percent_fallback: float = Field(default=0.08, ge=0.03, le=0.3)
    max_holding_days: int = Field(default=14, ge=1, le=60)
    max_stop_by_type: Dict[str, float] = Field(
        default_factory=lambda: {"C": 0.06, "A": 0.08, "B": 0.09, "S": 0.10}
    )
    type_position_caps: Dict[str, float] = Field(
        default_factory=lambda: {"C": 0.25, "A": 0.25, "B": 0.20, "S": 0.15}
    )

    # --- Targets (ATR-based T1/T2) ---
    type_atr_multipliers: Dict[str, float] = Field(
        default_factory=lambda: {"S": 2.5, "B": 2.0, "A": 1.8, "C": 1.5}
    )
    t2_atr_ratio: float = Field(default=2.0, ge=1.0, le=4.0)
    type_target_caps: Dict[str, TypeTargetCaps] = Field(
        default_factory=lambda: {
            "S": TypeTargetCaps(t1_max_pct=0.12, t2_max_pct=0.22),
            "B": TypeTargetCaps(t1_max_pct=0.10, t2_max_pct=0.18),
            "C": TypeTargetCaps(t1_max_pct=0.08, t2_max_pct=0.15),
            "A": TypeTargetCaps(t1_max_pct=0.10, t2_max_pct=0.18),
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


def load_settings(path: Optional[Path] = None) -> SmallCapSettings:
    """
    Merge JSON file over defaults and return a validated SmallCapSettings.
    Missing file → defaults only.
    """
    p = path or DEFAULT_SETTINGS_PATH
    base = SmallCapSettings().model_dump(mode="json")
    if not p.exists():
        logger.debug("Small-cap settings file missing, using defaults: %s", p)
        return SmallCapSettings.model_validate(base)

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("settings file must be a JSON object")
        merged = _deep_merge(base, raw)
        return SmallCapSettings.model_validate(merged)
    except Exception as e:
        logger.error("Failed to load %s: %s — using defaults", p, e)
        return SmallCapSettings.model_validate(base)


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
    Deep-merge ``patch`` onto current file-backed settings, validate, and persist.

    Raises pydantic.ValidationError if the merged result is invalid.
    """
    if not isinstance(patch, dict):
        raise TypeError("patch must be a dict")
    current = load_settings(path=path).model_dump(mode="json")
    merged = _deep_merge(current, patch)
    validated = SmallCapSettings.model_validate(merged)
    save_settings(validated, path=path)
    return validated
