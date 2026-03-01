from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PolicyContext:
    """
    Context for a single daily decision.

    Most fields are optional so the policy can run in shadow mode even when the
    upstream fusion/Anifold Z is not yet wired in.
    """

    anifold_z: Optional[Sequence[float]] = None
    z_uncertainty_diag: Optional[Sequence[float]] = None
    z_velocity: Optional[Sequence[float]] = None
    z_distance_to_attractor: Optional[float] = None

    # Behavior / burden signals (optional, can be computed upstream)
    completion_rate_7d: Optional[float] = None
    completion_rate_14d: Optional[float] = None
    completion_rate_30d: Optional[float] = None
    burden_ms_median_14d: Optional[float] = None

    # Time context
    day_of_week: Optional[int] = None  # 0=Mon .. 6=Sun

    # Protocol state
    safety_trigger_active: bool = False
    allow_context_batches: bool = True
    identity_mask_id: Optional[str] = None


@dataclass(frozen=True)
class CandidateItem:
    item_id: str
    item_type: str
    scale_ids: Tuple[str, ...]
    deterministic_score: float
    constraint_tags: Tuple[str, ...]
    reason_codes: Tuple[str, ...]
    features: Dict[str, Any]


@dataclass(frozen=True)
class CandidateSet:
    """
    Candidate set C produced by the deterministic selector adapter.
    """

    user_id: str
    day: date
    k_core: int
    candidates: Tuple[CandidateItem, ...]
    mandatory_item_ids: Tuple[str, ...]
    deterministic_baseline_selected: Tuple[str, ...]


@dataclass(frozen=True)
class SelectedItemExplanation:
    item_id: str
    selection_rank: int
    policy_score: float
    deterministic_score: float
    reason_codes: Tuple[str, ...]
    constraint_justification: Tuple[str, ...]
    top_feature_contributions: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class CounterfactualExplanation:
    item_id: str
    policy_score: float
    reason_codes: Tuple[str, ...]


@dataclass(frozen=True)
class PolicyDecision:
    decision_id: str
    user_id: str
    day: date
    policy_version: str
    feature_version: str
    mode: str  # live | shadow | safe_fallback | replay
    selected_item_ids: Tuple[str, ...]
    propensities: Optional[Dict[str, float]]
    explanations: Tuple[SelectedItemExplanation, ...]
    counterfactuals: Tuple[CounterfactualExplanation, ...]
    deterministic_baseline_selected: Tuple[str, ...]
    # Hashes for auditability
    context_hash: str
    candidate_set_hash: str
