"""
Effective min_quality / top_n from regime — must match api/routers/scanner.py logic.
"""

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .settings_config import RegimeThresholds


def effective_scan_thresholds(
    regime: str,
    regime_confidence: str,
    request_min_quality: int,
    request_top_n: int,
    regime_caps: Optional["RegimeThresholds"] = None,
) -> Tuple[int, int]:
    rq = request_min_quality
    rt = request_top_n

    # Regime-driven quality floor: bear/caution markets require higher quality signals.
    # Takes max(request, regime_floor) so user can never lower below regime minimum.
    if regime_caps is not None:
        if regime == "BEAR":
            regime_min = (
                regime_caps.bear_tentative_min_quality
                if regime_confidence == "TENTATIVE"
                else regime_caps.bear_confirmed_min_quality
            )
        elif regime == "CAUTION":
            regime_min = (
                regime_caps.caution_confirmed_min_quality
                if regime_confidence == "CONFIRMED"
                else regime_caps.caution_other_min_quality
            )
        else:
            regime_min = 0  # BULL / UNKNOWN — no floor applied
        eff_min = max(rq, regime_min)
    else:
        eff_min = rq

    # Hard top_n caps per regime (explicit product rule).
    if regime == "BEAR":
        if regime_confidence == "TENTATIVE":
            eff_top = min(rt, 4)
        else:
            eff_top = min(rt, 3)
    elif regime == "CAUTION":
        if regime_confidence == "TENTATIVE":
            eff_top = min(rt, 5)
        else:
            eff_top = min(rt, 4)
    else:
        # BULL / UNKNOWN / other → user top_n
        eff_top = rt

    return eff_min, eff_top
