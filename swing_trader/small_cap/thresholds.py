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
        elif regime == "BULL":
            # 2026-07-27: BULL artık taban uyguluyor (eskiden 0'dı → Q60-70
            # değersiz sinyaller geçiyordu). Ölçülen tatlı nokta Q78.
            regime_min = regime_caps.bull_min_quality
        else:
            regime_min = 0  # UNKNOWN — taban yok (rejim belirsiz)
        eff_min = max(rq, regime_min)
    else:
        eff_min = rq

    # Hard top_n caps per regime. 2026-07-27: hardcoded sayılar yerine
    # regime_caps model değerlerinden okunuyor (tek kaynak — eskiden bu caps
    # ile RegimeThresholds değerleri ayrışabiliyordu).
    if regime_caps is not None:
        if regime == "BEAR":
            cap = (regime_caps.bear_tentative_top_n_max
                   if regime_confidence == "TENTATIVE"
                   else regime_caps.bear_confirmed_top_n_max)
            eff_top = min(rt, cap)
        elif regime == "CAUTION":
            cap = (regime_caps.caution_other_top_n_max
                   if regime_confidence != "CONFIRMED"
                   else regime_caps.caution_confirmed_top_n_max)
            eff_top = min(rt, cap)
        elif regime == "BULL":
            eff_top = min(rt, regime_caps.bull_top_n_max)
        else:
            eff_top = rt  # UNKNOWN → user top_n
    else:
        eff_top = rt

    return eff_min, eff_top
