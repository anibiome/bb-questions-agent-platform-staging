from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "completion": 1.0,
    "burden_penalty": 0.2,
    "unlock_value": 0.1,
    "uncertainty_reduction": 0.3,
}


@dataclass(frozen=True)
class RewardResult:
    components: Dict[str, float]
    total: float
    weights: Dict[str, float]


def compute_reward(
    *,
    completion_rate: float,
    response_time_ms_median: Optional[float],
    unlock_count: int,
    uncertainty_before_mean: Optional[float] = None,
    uncertainty_after_mean: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> RewardResult:
    """
    v1 reward proxy for policy training/evaluation.

    This intentionally avoids clinical interpretation. It optimizes for:
    - adherence/completion,
    - low burden,
    - progress through validated scales (unlock/retest readiness),
    - information gain proxy (uncertainty reduction) when available.

    You can later replace/extend this with richer components (forecast error,
    biomarker correlation, etc.) without changing the decision logging contract.
    """
    w = dict(DEFAULT_REWARD_WEIGHTS)
    if weights:
        for k, v in weights.items():
            w[str(k)] = float(v)

    if isinstance(overrides, dict):
        ow = overrides.get("weights")
        if isinstance(ow, dict):
            for k, v in ow.items():
                w[str(k)] = float(v)

    completion = float(max(0.0, min(1.0, completion_rate)))

    burden = 0.0
    if response_time_ms_median is not None:
        # Normalize to ~1 min median response time (cap at 1.0).
        burden = min(1.0, max(0.0, float(response_time_ms_median)) / 60_000.0)

    unlock_value = min(1.0, max(0.0, int(unlock_count)) / 3.0)

    uncertainty_reduction = 0.0
    if uncertainty_before_mean is not None and uncertainty_after_mean is not None:
        uncertainty_reduction = max(0.0, float(uncertainty_before_mean) - float(uncertainty_after_mean))

    components = {
        "completion": completion,
        "burden_penalty": burden,
        "unlock_value": unlock_value,
        "uncertainty_reduction": uncertainty_reduction,
    }

    if isinstance(overrides, dict):
        oc = overrides.get("components")
        if isinstance(oc, dict):
            for k, v in oc.items():
                try:
                    components[str(k)] = float(v)
                except Exception:
                    continue

    total = 0.0
    total += float(w.get("completion", 1.0)) * components["completion"]
    total -= float(w.get("burden_penalty", 0.2)) * components["burden_penalty"]
    total += float(w.get("unlock_value", 0.1)) * components["unlock_value"]
    total += float(w.get("uncertainty_reduction", 0.3)) * components["uncertainty_reduction"]

    return RewardResult(components=components, total=float(total), weights=w)

