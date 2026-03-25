"""
Effective min_quality / top_n from regime — must match api/routers/scanner.py logic.
"""

import math
from typing import Tuple


def effective_scan_thresholds(
    regime: str,
    regime_confidence: str,
    regime_multiplier: float,
    request_min_quality: int,
    request_top_n: int,
) -> Tuple[int, int]:
    rq = request_min_quality
    rt = request_top_n

    # Relaxed vs legacy (deep backtest fix): ~10–15pt lower floors so more trades pass in CAUTION/BEAR scans
    if regime == "BEAR":
        if regime_confidence == "TENTATIVE":
            eff_min = max(rq, 70)
            eff_top = min(rt, 4)
        else:
            eff_min = max(rq, 72)
            eff_top = min(rt, 3)
    elif regime == "CAUTION":
        if regime_confidence == "CONFIRMED":
            eff_min = max(rq, 68)
            eff_top = min(rt, 4)
        else:
            eff_min = max(rq, 63)
            eff_top = min(rt, 5)
    else:
        eff_min = rq
        eff_top = rt

    if 0 < regime_multiplier < 1.0:
        adj = int(math.ceil(rq / regime_multiplier - 1e-9))
        eff_min = max(eff_min, adj)

    return eff_min, eff_top
