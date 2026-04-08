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

    # Rejim yalnızca bilgi ve top_n kısıtlaması içindir.
    # Min_quality kullanıcıdan geldiği gibi ham skora uygulanır (rejim bunu değiştirmez).
    eff_min = rq

    # Hard-coded caps (explicit product rule).
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
