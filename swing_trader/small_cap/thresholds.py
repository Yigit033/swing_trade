"""
Effective min_quality / top_n from regime — must match api/routers/scanner.py logic.
"""

import math
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .settings_config import RegimeThresholds


def effective_scan_thresholds(
    regime: str,
    regime_confidence: str,
    regime_multiplier: float,
    request_min_quality: int,
    request_top_n: int,
    regime_caps: Optional["RegimeThresholds"] = None,
) -> Tuple[int, int]:
    from .settings_config import load_settings

    rq = request_min_quality
    rt = request_top_n
    caps = regime_caps if regime_caps is not None else load_settings().regime_thresholds

    if regime == "BEAR":
        if regime_confidence == "TENTATIVE":
            eff_min = max(rq, caps.bear_tentative_min_quality)
            eff_top = min(rt, caps.bear_tentative_top_n_max)
        else:
            eff_min = max(rq, caps.bear_confirmed_min_quality)
            eff_top = min(rt, caps.bear_confirmed_top_n_max)
    elif regime == "CAUTION":
        if regime_confidence == "CONFIRMED":
            eff_min = max(rq, caps.caution_confirmed_min_quality)
            eff_top = min(rt, caps.caution_confirmed_top_n_max)
        else:
            eff_min = max(rq, caps.caution_other_min_quality)
            eff_top = min(rt, caps.caution_other_top_n_max)
    else:
        eff_min = rq
        eff_top = rt

    if 0 < regime_multiplier < 1.0:
        adj = int(math.ceil(rq / regime_multiplier - 1e-9))
        eff_min = max(eff_min, adj)

    return eff_min, eff_top
