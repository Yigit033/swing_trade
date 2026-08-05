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

    # 2026-08-04 TEMİZLİK — aşağıdaki alanlar SİLİNDİ:
    #   parabolic_five_day_return_gt / extreme_five_day_return_gt / extreme_rsi_gt
    #     → HİÇBİR KOD OKUMUYORDU (%100 ölü ayar; grep ile doğrulandı)
    #   late_entry_five_day_total_gt / late_entry_rsi_gt
    #     → gate ölçüldü, ΔEV tam 0.00, hiç ateşlenmiyordu (measure_gate_value.py);
    #       VCE muafiyeti + Weinstein + swing onayı bu vakaları zaten eliyordu
    #   distribution_day_min_vol / distribution_day_max_change_pct
    #     → gate ölçüldü, ΔEV 0.00; VCE ve RVOL İKİSİ DE yeşil kapanış şartı
    #       koyduğu için "hacimli düşüş günü" bir sinyal olarak hiç oluşamıyordu
    # Kalan iki alan ÖLÇÜLDÜ ve İŞE YARIYOR: Weinstein reddi ΔEV −0.98
    # (eklediği 18 sinyalin EV'si −2.10%). Bkz. GATE_AUDIT.md.
    reject_stage3: bool = Field(default=True, description="Hard reject Weinstein Stage 3 (distribution) — ΔEV −0.98 ölçüldü")
    reject_stage4: bool = Field(default=True, description="Hard reject Weinstein Stage 4 (decline) — all types")


class SwingParabolicSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    five_day_gt: float = 70
    five_day_extreme_gt: float = 60
    rsi_extreme_gt: float = 85
    hold_short: tuple[int, int] = (1, 2)


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


class AutoScanSettings(BaseModel):
    """
    Zamanlanmış (kullanıcı etkileşimi gerektirmeyen) günlük tarama.

    Kapanış sonrası tek seferlik otomatik tarama — motorun VCE tetikleyicisi
    zaten dünün TAMAMLANMIŞ barına göre karar veriyor (fetcher.py
    _drop_incomplete_last_bar) ve evren artık saatten bağımsız (Q5/Q5b
    kaldırıldı, 2026-07-22) — yani bu, manuel taramanın otomatikleştirilmiş
    hali, ayrı bir mantık değil.

    min_quality BİLEREK Scanner sayfasındaki "Auto-Track" slider'ından
    (kullanıcı unutup değiştirebilir, UI state) AYRI ve sabit tutulur —
    gece kimse izlemezken hangi eşiğin kullanıldığı, ekranda o an ne
    görünüyor olduğuna bağlı olmamalı.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # NYSE kapanışından sonraki hedef saat (ET). 16:30 ET = kapanıştan 60dk
    # sonra — Finviz'in günlük Change/Volume/20D-High kolonları o ana kadar
    # sindirilmiş olur (ölçümlerin yapıldığı pencereyle aynı varsayım).
    target_hour_et: int = Field(default=16, ge=0, le=23)
    target_minute_et: int = Field(default=30, ge=0, le=59)
    min_quality: int = Field(default=70, ge=0, le=100)
    top_n: int = Field(default=15, ge=1, le=100)
    portfolio_value: float = Field(default=10_000.0, gt=0)


class UniverseFilterSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_market_cap: int = 250_000_000
    # v13.2: raised 2.5B → 10B. The VCE edge was measured cap-agnostic and its
    # strongest contributors (HIMS, CELH, RDDT, TOST, MNDY) are mid-caps.
    max_market_cap: int = 10_000_000_000
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

    max_scan_tickers: int = Field(default=260, ge=20, le=500)
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

    # 2026-08-04: enable_finviz_query_momentum / _setup / _wider KALDIRILDI.
    # Üç sorgu 2026-07-18 recall ölçümünde %0.5-2 marjinal katkı verdiği için
    # kapatılmıştı; kodu da silindi. Alanları burada bırakmak "ölü ayar" olurdu.
    # NOT: eski JSON/DB yamaları bu anahtarları taşıyabilir → model extra="forbid"
    # olduğu için doğrulama patlar. load_settings kademeli geri çekilme yapıyor
    # (DB katmanını atıp dosyayla devam eder) ve _prune_removed_keys aşağıdaki
    # listeyi sessizce temizler.

    post_filter_price_min: float = Field(default=3.0, ge=0.5, le=500.0)
    post_filter_price_max: float = Field(default=200.0, ge=1.0, le=50000.0)

    # 2026-08-04 TEMİZLİK — rank_weight_* ve chase_penalty_* SİLİNDİ.
    # Composite sıralama tek ölçüte indirildi (dolar-hacim; gerekçe ölçülmüş:
    # measure_price_band.py kesişim testi, fiyat SABİT/likidite değişken →
    # likit +3.31% WR62 / illikit −2.14% WR44). 4 ağırlık + kovalama cezası +
    # erken-birikim bonusunun HİÇBİRİ ölçülmemişti ve tavan 15/15 taramada
    # bağlamadığı için etkisi de doğrulanamıyordu.

    @model_validator(mode="after")
    def _validate_universe_scan(self) -> "UniverseScanSettings":
        # 2026-08-04: rank-weight toplamı ve kovalama-cezası tutarlılık kuralları
        # kaldırıldı — dayandıkları alanlar silindi (sıralama tek ölçüte indi).
        if self.post_filter_price_max <= self.post_filter_price_min:
            raise ValueError("post_filter_price_max must be > post_filter_price_min")
        return self


class SignalsConfirmationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ma20_max_distance_below_pct: float = 0.0
    ma50_max_below_pct: float = Field(
        default=3.0,
        ge=0.0,
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

    weight_volume: float = 0.12
    weight_volatility: float = 0.13
    weight_float: float = 0.25
    weight_momentum: float = 0.25
    weight_risk: float = 0.10
    max_volume_score: float = 30
    max_volatility_score: float = 25
    max_float_score: float = 20
    max_momentum_score: float = 15
    max_risk_score: float = 15
    final_score_max: int = 140
    risk_score_atr_mult: float = 1.5
    weight_trend: float = Field(default=0.15, ge=0.0, le=1.0)
    max_trend_score: float = Field(default=25.0, ge=1.0, le=100.0)

    # Bonus artık koşullu değil, SABİT. 14 koşullu bonus (high_rvol,
    # gap_continuation, higher_highs, swing_ready, higher_lows, multi_day_volume,
    # surge_days, early_entry, rsi_divergence, golden_cross, confirmed_breakout,
    # volume_on_up_day + sector_rs + obv) toplanıp buraya kırpılıyordu; ölçüm
    # gösterdi ki tavan sinyallerin %100'ünde bağlıyor (ham toplam ~60), yani
    # 14 koşulun net çıktısı herkese aynı +30. Ayarları kaldırdık, sabit kaldı.
    bonus_cap: int = 30

    # CEZALAR — yalnız ölçülüp "çıkarınca EV düşüyor" çıkanlar kaldı (5/21).
    # Silinen 16'nın kanıtı GATE_AUDIT.md'de.
    pen_a_rsi_gt_70: int = 10
    pen_a_rsi_gt_65: int = 5
    pen_c_rsi_gt_65: int = 10
    pen_c_rsi_gt_60: int = 5
    pen_today_gt_10: int = 5


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


