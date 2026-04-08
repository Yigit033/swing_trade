"""
Nested tuning models for SmallCapSettings (swing classification, gates, scoring, filters, backtest).

Defaults match pre-migration hardcoded values in engine / risk / signals / filters / scoring / backtest.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ScoringVolumeTier(BaseModel):
    """Descending lookup: first tier with volume_surge >= min_surge wins."""

    model_config = ConfigDict(extra="forbid")

    min_surge: float = Field(ge=0)
    score: float = Field()


class ScoringAtrTier(BaseModel):
    """Descending lookup on ATR/close fraction (e.g. 0.15 = 15%)."""

    model_config = ConfigDict(extra="forbid")

    min_atr_frac: float = Field(ge=0)
    score: float = Field()


class ScoringFloatBand(BaseModel):
    """Ascending lookup: first band where float_millions <= max_millions_le wins."""

    model_config = ConfigDict(extra="forbid")

    max_millions_le: float = Field(ge=0)
    score: float = Field()


class ScoringMomentumPoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    higher_highs_full: int = 6
    higher_highs_partial: int = 3
    higher_closes_full: int = 6
    higher_closes_partial: int = 3
    close_in_top_of_range_min: float = Field(default=0.8, ge=0, le=1)
    close_near_high_pts: int = 3
    raw_cap: int = Field(default=15, ge=1, le=50)
    insufficient_bars_score: int = Field(default=5, ge=0, le=20)


class ScoringRiskBands(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stop_le_05_pts: int = 10
    stop_le_08_pts: int = 7
    stop_le_10_pts: int = 5
    stop_else_pts: int = 3
    range_le_05_pts: int = 5
    range_le_08_pts: int = 3
    raw_cap: int = Field(default=15, ge=1, le=50)
    insufficient_bars_score: int = Field(default=5, ge=0, le=20)


def _default_volume_surge_tiers() -> List[ScoringVolumeTier]:
    return [
        ScoringVolumeTier(min_surge=6.0, score=30),
        ScoringVolumeTier(min_surge=5.0, score=26),
        ScoringVolumeTier(min_surge=4.0, score=22),
        ScoringVolumeTier(min_surge=3.0, score=18),
        ScoringVolumeTier(min_surge=2.5, score=14),
        ScoringVolumeTier(min_surge=2.0, score=10),
        ScoringVolumeTier(min_surge=1.5, score=6),
        ScoringVolumeTier(min_surge=1.3, score=3),
    ]


def _default_atr_percent_tiers() -> List[ScoringAtrTier]:
    return [
        ScoringAtrTier(min_atr_frac=0.15, score=25),
        ScoringAtrTier(min_atr_frac=0.12, score=22),
        ScoringAtrTier(min_atr_frac=0.10, score=18),
        ScoringAtrTier(min_atr_frac=0.08, score=14),
        ScoringAtrTier(min_atr_frac=0.06, score=10),
        ScoringAtrTier(min_atr_frac=0.04, score=7),
        ScoringAtrTier(min_atr_frac=0.035, score=5),
    ]


def _default_float_millions_bands() -> List[ScoringFloatBand]:
    return [
        ScoringFloatBand(max_millions_le=15, score=20),
        ScoringFloatBand(max_millions_le=30, score=15),
        ScoringFloatBand(max_millions_le=45, score=10),
        ScoringFloatBand(max_millions_le=60, score=5),
        ScoringFloatBand(max_millions_le=80, score=0),
    ]


class ScanStockGatesSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parabolic_five_day_return_gt: float = Field(default=70.0, ge=10, le=200)
    extreme_five_day_return_gt: float = Field(default=60.0, ge=10, le=200)
    extreme_rsi_gt: float = Field(default=85.0, ge=50, le=100)
    late_entry_five_day_total_gt: float = Field(default=30.0, ge=0, le=100)
    late_entry_rsi_gt: float = Field(default=65.0, ge=40, le=95)


class SwingParabolicSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    five_day_gt: float = 70
    five_day_extreme_gt: float = 60
    rsi_extreme_gt: float = 85
    hold_short: tuple[int, int] = (1, 2)


class SwingTypeSSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_si_min: float = 20
    primary_dtc_min: float = 5
    primary_vol_min: float = 4.0
    primary_5d_min: float = 15
    primary_5d_max: float = 60
    primary_rsi_min: float = 60
    primary_rsi_max: float = 80
    primary_hold_min: int = 1
    primary_hold_max: int = 4
    secondary_si_min: float = 15
    secondary_dtc_min: float = 3
    secondary_vol_min: float = 3.0
    secondary_5d_min: float = 10
    secondary_5d_max: float = 40
    secondary_rsi_min: float = 55
    secondary_rsi_max: float = 75
    secondary_hold_min: int = 2
    secondary_hold_max: int = 4


class SwingTypeCSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    return_min: float = -5
    return_max: float = 15
    return_band_pts: int = 4
    sweet_return_min: float = 0
    sweet_return_max: float = 10
    sweet_bonus_pts: int = 1
    rsi_min: float = 40
    rsi_max: float = 60
    rsi_band_pts: int = 4
    rsi_low_max: float = 50
    rsi_low_bonus_pts: int = 1
    rsi_mid_max: float = 65
    rsi_mid_pts: int = 2
    vol_min: float = 1.8
    vol_max: float = 4.0
    vol_band_pts: int = 2
    vol_high_min: float = 2.5
    vol_high_bonus_pts: int = 1
    ma_dist_min: float = -3
    ma_dist_max: float = 8
    ma_band_pts: int = 2
    close_position_min: float = 0.55
    close_position_pts: int = 1
    rsi_div_pts: int = 3
    macd_pts: int = 1
    higher_lows_pts: int = 1
    min_score: int = 10
    hold_min: int = 3
    hold_max: int = 8


class SwingTypeBSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r_30_70_pts: int = 3
    r_20_30_pts: int = 2
    r_gt_70_pts: int = 1
    rsi_68_85_pts: int = 3
    rsi_60_68_pts: int = 2
    rsi_gt_85_pts: int = 1
    vol_35_pts: int = 3
    vol_25_pts: int = 2
    close_pos_min: float = 0.75
    close_pos_pts: int = 2
    catalyst_pts: int = 1
    min_score: int = 6
    gate_vol_min: float = 3.5
    vol_surge_secondary_min: float = 2.5
    gate_rsi_safe_max: float = 72
    rsi_overbought_hold_gt: float = 73
    hold_overbought: tuple[int, int] = (2, 4)
    rsi_elevated_gt: float = 68
    hold_elevated: tuple[int, int] = (3, 5)
    hold_default: tuple[int, int] = (4, 6)


class SwingTypeASettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    five_d_max_early: float = 15
    rsi_max_early: float = 55
    hold_early: tuple[int, int] = (5, 10)
    five_d_max_std: float = 25
    rsi_max_std: float = 62
    hold_std: tuple[int, int] = (7, 12)
    hold_extended: tuple[int, int] = (8, 14)


class SwingEngineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parabolic: SwingParabolicSettings = Field(default_factory=SwingParabolicSettings)
    type_s: SwingTypeSSettings = Field(default_factory=SwingTypeSSettings)
    type_c: SwingTypeCSettings = Field(default_factory=SwingTypeCSettings)
    type_b: SwingTypeBSettings = Field(default_factory=SwingTypeBSettings)
    type_a: SwingTypeASettings = Field(default_factory=SwingTypeASettings)


class RiskTargetRegimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality_tier_high: int = 85
    quality_boost_high: float = 1.15
    quality_tier_mid: int = 75
    quality_boost_mid: float = 1.08
    t2_atr_mult_caution: float = 1.6
    t2_atr_mult_bear: float = 1.05
    min_reward_risk_multiple_t1: float = 1.5
    t2_min_gap_vs_t1_bull: float = 1.15
    t2_min_gap_vs_t1_bear: float = 1.05
    t2_min_gap_vs_t1_caution: float = 1.10
    t2_vs_t1_near_cap_floor: float = 1.005


class UniverseFilterSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_market_cap: int = 250_000_000
    max_market_cap: int = 2_500_000_000
    min_avg_volume: int = 750_000
    min_price: float = 3.0
    max_price: float = 200.0
    max_float_shares: int = 150_000_000
    earnings_exclusion_days: int = 3
    atr_period: int = 10


class UniverseScanSettings(BaseModel):
    """
    Finviz-based daily universe build: which tickers enter the ~N-name scan list.

    Screen URLs stay in code; this model only toggles layers and numeric knobs.
    """

    model_config = ConfigDict(extra="forbid")

    max_scan_tickers: int = Field(default=200, ge=20, le=500)
    use_finviz: bool = True
    cache_duration_minutes: int = Field(
        default=60,
        ge=0,
        le=10080,
        description="0 = do not reuse cached Finviz results by age (always refetch when invoked).",
    )
    min_finviz_tickers_skip_static_merge: int = Field(
        default=30,
        ge=0,
        le=500,
        description="If Finviz returns at least this many names, skip merging static_seed / tier lists.",
    )

    enable_finviz_query_momentum: bool = True
    enable_finviz_query_setup: bool = True
    enable_finviz_query_wider: bool = True

    post_filter_price_min: float = Field(default=3.0, ge=0.5, le=500.0)
    post_filter_price_max: float = Field(default=200.0, ge=1.0, le=50000.0)

    rank_weight_rvol: float = Field(default=0.30, ge=0.0, le=1.0)
    rank_weight_change: float = Field(default=0.25, ge=0.0, le=1.0)
    rank_weight_volume: float = Field(default=0.25, ge=0.0, le=1.0)
    rank_weight_mcap: float = Field(default=0.20, ge=0.0, le=1.0)

    chase_penalty_change_pct_high: float = Field(default=15.0, ge=5.0, le=50.0)
    chase_penalty_change_pct_mid: float = Field(default=10.0, ge=1.0, le=40.0)
    chase_penalty_points_high: int = Field(default=50, ge=0, le=100)
    chase_penalty_points_mid: int = Field(default=25, ge=0, le=100)

    @model_validator(mode="after")
    def _validate_universe_scan(self) -> "UniverseScanSettings":
        if self.use_finviz:
            if not (
                self.enable_finviz_query_momentum
                or self.enable_finviz_query_setup
                or self.enable_finviz_query_wider
            ):
                raise ValueError(
                    "At least one Finviz query must be enabled when use_finviz is True."
                )
        wsum = (
            self.rank_weight_rvol
            + self.rank_weight_change
            + self.rank_weight_volume
            + self.rank_weight_mcap
        )
        if abs(wsum - 1.0) > 0.02:
            raise ValueError(
                f"Rank weights must sum to 1.0 (±0.02); got {wsum:.4f}"
            )
        if self.post_filter_price_max <= self.post_filter_price_min:
            raise ValueError("post_filter_price_max must be > post_filter_price_min")
        if self.chase_penalty_change_pct_mid >= self.chase_penalty_change_pct_high:
            raise ValueError(
                "chase_penalty_change_pct_mid must be < chase_penalty_change_pct_high"
            )
        return self


class SignalsConfirmationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ma20_max_distance_below_pct: float = 3.0
    ma50_max_below_pct: float = Field(
        default=8.0,
        ge=1.0,
        le=25.0,
        description="Max %% below MA50 before rejecting swing confirmation.",
    )
    overext_today_change_max: float = 15.0
    overext_single_day_max: float = 25.0
    overext_five_day_total_max: float = 40.0


class ScoringTuningSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    volume_surge_tiers: List[ScoringVolumeTier] = Field(
        default_factory=_default_volume_surge_tiers,
        min_length=1,
    )
    atr_percent_tiers: List[ScoringAtrTier] = Field(
        default_factory=_default_atr_percent_tiers,
        min_length=1,
    )
    float_millions_bands: List[ScoringFloatBand] = Field(
        default_factory=_default_float_millions_bands,
        min_length=1,
    )
    float_score_unknown: float = Field(default=5.0, ge=-50, le=50)
    float_score_above_max_band: float = Field(default=-8.0, ge=-50, le=50)
    momentum_points: ScoringMomentumPoints = Field(default_factory=ScoringMomentumPoints)
    risk_bands: ScoringRiskBands = Field(default_factory=ScoringRiskBands)

    weight_volume: float = 0.30
    weight_volatility: float = 0.20
    weight_float: float = 0.20
    weight_momentum: float = 0.15
    weight_risk: float = 0.15
    max_volume_score: float = 30
    max_volatility_score: float = 25
    max_float_score: float = 20
    max_momentum_score: float = 15
    max_risk_score: float = 15
    bonus_cap: int = 40
    final_score_max: int = 140
    risk_score_atr_mult: float = 1.5
    # Booster bonuses (subset — catalyst/sector still from live data)
    bonus_high_rvol: int = 3
    bonus_gap_continuation: int = 4
    bonus_higher_highs: int = 3
    bonus_swing_ready: int = 10
    bonus_higher_lows: int = 5
    bonus_multi_day_volume: int = 3
    bonus_surge_days_3: int = 5
    bonus_surge_days_2: int = 3
    bonus_early_entry_lo: float = 5
    bonus_early_entry_hi: float = 15
    bonus_early_entry_pts: int = 8
    bonus_very_early_hi: float = 5
    bonus_very_early_pts: int = 5
    bonus_rsi_divergence: int = 8
    # RSI penalties by type
    pen_a_rsi_gt_70: int = 10
    pen_a_rsi_gt_65: int = 5
    pen_b_rsi_gt_85: int = 15
    pen_b_rsi_gt_80: int = 10
    pen_b_rsi_gt_75: int = 5
    pen_c_rsi_gt_65: int = 10
    pen_c_rsi_gt_60: int = 5
    pen_ext_day_gt_25: int = 15
    pen_ext_day_gt_20: int = 8
    pen_today_gt_15: int = 10
    pen_today_gt_10: int = 5
    pen_5d_gt_40: int = 15
    pen_5d_gt_30: int = 10
    pen_5d_gt_25: int = 5
    pen_parabolic: int = 15
    parabolic_day3_min_pct: float = 10.0
    pen_not_swing_ready: int = 5
    # v5.0: Directional scoring fields
    weight_trend: float = Field(default=0.20, ge=0.0, le=1.0)
    max_trend_score: float = Field(default=25.0, ge=1.0, le=100.0)
    bonus_golden_cross: int = Field(default=5, ge=0, le=20)
    bonus_confirmed_breakout: int = Field(default=8, ge=0, le=20)
    # v5.0: Directional penalties
    pen_obv_distribution: int = Field(default=15, ge=0, le=30)
    pen_below_ma50: int = Field(default=10, ge=0, le=25)
    pen_ma20_falling: int = Field(default=8, ge=0, le=20)
    pen_rejection_candle: int = Field(default=12, ge=0, le=25)
    pen_weak_trend_phase: int = Field(default=8, ge=0, le=20)


class BacktestLoopSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drawdown_pause_entries_fraction: float = 0.25
    drawdown_reduce_to_one_position_fraction: float = 0.15
    caution_max_concurrent: int = 1
    bear_block_new_entries: bool = True


class BacktestTypeQualityOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type_c_bear: int = 82
    type_c_caution: int = 75
    type_a_bear: int = 72
    type_a_caution: int = 66
    type_b_bear: int = 67
    type_b_caution: int = 60


class BacktestEntrySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type_c_min_open_vs_signal_close_ratio: float = 0.98
    trend_ema_fast_span: int = 10
    trend_ema_slow_span: int = 20
    trend_min_bars: int = 21
    gap_atr_multiplier: float = 2.0
    partial_fallback_target_bump: float = 1.15


class BacktestExitTrailingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_stop_min_days: int = 5
    time_stop_min_loss_fraction: float = 0.05
    trail_peak_atr_25: float = 2.5
    trail_high_minus_atr_25: float = 0.8
    trail_peak_atr_20: float = 2.0
    trail_peak_frac_20: float = 0.5
    trail_peak_atr_15: float = 1.5
    trail_peak_frac_15: float = 0.3
    breakeven_peak_atr: float = 1.5
    light_protect_peak_atr: float = 1.0
    light_protect_below_entry_atr: float = 0.2
    close_gain_atr_20: float = 2.0
    close_trail_atr_20: float = 1.0
    close_gain_atr_15: float = 1.5
    close_trail_atr_15: float = 1.2
