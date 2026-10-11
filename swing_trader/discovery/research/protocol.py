"""Fixed before inspecting Phase 3 outcomes. No parameter search."""
PROTOCOL = {
    "version": "phase3-preregistered-v1",
    "cohort": "HISTORICAL_RESEARCH_CURRENT_UNIVERSE_SURVIVORSHIP_BIASED",
    "sessions": 90, "research_sessions": 30, "purge_sessions": 30, "evaluation_sessions": 30,
    "min_history": 130, "horizons": [5, 10, 20, 30], "cost_bps": [20, 50, 100],
    "primary_horizon": 20, "primary_cost_bps": 50,
    "opportunity": "NEXT_OPEN_TO_CLOSE_GE10PCT_AT_20_SESSIONS;MFE_SEPARATE",
    "independence": "FIRST_CHRONOLOGICAL_EVENT_PER_TICKER_THEN_30_SESSION_COOLDOWN",
    "confidence": "95% TICKER_CLUSTER_BOOTSTRAP;DATE_CLUSTER_SENSITIVITY;1000_DRAWS_SEED7301",
    "minimum_evaluation_clusters": 30,
    "challenger_1": {
        "name": "PREBREAK_ACCUMULATION", "ma20_gt_ma50": True,
        "close_to_prior20high": [0.97, 1.0], "range5_to_range20_max": 0.5,
        "up_down_dollar_volume10_min": 1.2, "median_volume5_to20_max": 0.8,
        "economic_hypothesis": "Supply contracts near resistance before confirmed breakout; not old unqualified volume ignition.",
    },
    "challenger_2": {
        "name": "IMPULSE_BASE_RELAUNCH", "impulse_tminus10_to_tminus5_pct": [10, 40],
        "prior5_range_pct_max": 8, "ma20_gt_ma50": True,
        "close_gt_prior5high": True, "rvol20_min": 1.2, "day_return_pct_max": 8,
        "economic_hypothesis": "A bounded prior impulse is digested in a five-session base, then demand returns; not old generic continuation or ten-day tight breakout.",
    },
    "no_optimization": True,
    "full_trade_eligibility": "UNKNOWN_WITHOUT_HISTORICAL_CAP_FLOAT_EARNINGS",
    "comparison": "RAW_DISCOVERY_ON_IDENTICAL_PRICE_LIQUIDITY_HISTORY_COHORT_NOT_PRODUCTION_EXITS",
    "regime": "ACTUAL_SPY_AND_VIX_IF_AVAILABLE_ELSE_UNKNOWN;SPY_ONLY_CONTEXT_SEPARATE",
}
