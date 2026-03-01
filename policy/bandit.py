from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from questions_agent_platform.policy.features import FeatureMapping, FEATURE_VERSION, default_stats, featurize_v1, top_contributions
from questions_agent_platform.policy.hashing import stable_hash_hex, stable_hash_floats
from questions_agent_platform.policy.types import (
    CandidateItem,
    CandidateSet,
    CounterfactualExplanation,
    PolicyContext,
    PolicyDecision,
    SelectedItemExplanation,
)


@dataclass(frozen=True)
class PolicyParams:
    policy_version: str
    feature_version: str = FEATURE_VERSION
    lambda_reg: float = 1.0
    # Posterior parameters for Bayesian linear regression (A, b).
    # A is symmetric positive definite (d x d). b is length d.
    A: Tuple[Tuple[float, ...], ...] = ()
    b: Tuple[float, ...] = ()
    # Feature normalization stats (running stats from training; standardized at inference).
    feature_means: Tuple[float, ...] = ()
    feature_stds: Tuple[float, ...] = ()


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    epsilon_explore: float = 0.05
    max_counterfactuals: int = 10
    # If None, a stable seed derived from the decision context is used.
    rng_seed: Optional[int] = None


def make_policy_decision(
    *,
    candidate_set: CandidateSet,
    context: PolicyContext,
    params: PolicyParams,
    mapping: FeatureMapping,
    mode: str,
    runtime: PolicyRuntimeConfig,
) -> PolicyDecision:
    """
    Constrained contextual bandit policy.

    Invariants:
    - Never select outside candidate_set.candidates (action-space constraint).
    - Mandatory items must be included.
    - Blocked items must never be selected (unless the candidate generator removed the "blocked" tag).
    - If any invariant fails, fall back to deterministic baseline and record mode="safe_fallback".

    Notes:
    - We provide exact propensities only for explicit epsilon exploration steps.
    - Thompson sampling is used for scoring; selection is greedy by sampled score subject to constraints.
    """
    if params.feature_version != mapping.feature_version:
        raise ValueError(
            f"Feature version mismatch: params={params.feature_version} mapping={mapping.feature_version}"
        )

    candidate_items_by_id: Dict[str, CandidateItem] = {c.item_id: c for c in candidate_set.candidates}
    candidate_ids = list(candidate_items_by_id.keys())
    if len(candidate_ids) != len(candidate_set.candidates):
        # Duplicate ids -> unsafe; fall back.
        return _fallback_decision(candidate_set=candidate_set, context=context, params=params, mapping=mapping, mode=mode)

    # Hashes for auditability (do not include item text / raw answers).
    context_hash = _hash_context(context)
    candidate_set_hash = _hash_candidate_set(candidate_set)

    # Deterministic seed for reproducibility if caller doesn't provide one.
    seed = runtime.rng_seed if runtime.rng_seed is not None else int(context_hash[:12], 16)
    rng = random.Random(seed)

    # Safety protocol override (no exploration).
    if bool(context.safety_trigger_active):
        selected = _safe_protocol(candidate_set, candidate_items_by_id)
        return _build_decision(
            candidate_set=candidate_set,
            context=context,
            params=params,
            mapping=mapping,
            mode="safe_fallback",
            selected_item_ids=selected,
            propensities=None,
            sampled_weights=None,
            context_hash=context_hash,
            candidate_set_hash=candidate_set_hash,
            counterfactuals=(),
        )

    # Mandatory items (anchors) must always be included.
    mandatory = [i for i in candidate_set.mandatory_item_ids if i in candidate_items_by_id]
    mandatory = [i for i in mandatory if "blocked" not in candidate_items_by_id[i].constraint_tags]
    if len(mandatory) > int(candidate_set.k_core):
        return _fallback_decision(candidate_set=candidate_set, context=context, params=params, mapping=mapping, mode=mode)

    # Eligible optionals (must be inside C, not blocked, not mandatory).
    optionals: List[CandidateItem] = []
    for cid, c in candidate_items_by_id.items():
        if cid in mandatory:
            continue
        if "blocked" in c.constraint_tags:
            continue
        optionals.append(c)

    k_remaining = max(0, int(candidate_set.k_core) - len(mandatory))

    # Exploration is restricted to the safe subset: low/medium sensitivity only.
    safe_optionals = [c for c in optionals if str(c.features.get("sensitivity") or "") in ("low", "medium")]
    explore = rng.random() < float(max(0.0, min(1.0, runtime.epsilon_explore)))

    selected_optional_ids: List[str] = []
    propensities: Optional[Dict[str, float]] = None
    sampled_weights: Optional[List[float]] = None
    counterfactuals: Tuple[CounterfactualExplanation, ...] = ()

    if k_remaining > 0:
        if explore and safe_optionals:
            selected_optional_ids, propensities = _epsilon_explore(
                rng=rng, optionals=safe_optionals, k=k_remaining, epsilon=float(runtime.epsilon_explore)
            )
        else:
            sampled_weights = _sample_weights(params=params, rng=rng, dim=len(mapping.names))
            means, stds = _stats_from_params(params=params, dim=len(mapping.names))
            selected_optional_ids, counterfactuals = _rank_and_select(
                context=context,
                optionals=optionals,
                mapping=mapping,
                weights=sampled_weights,
                k=k_remaining,
                max_counterfactuals=int(runtime.max_counterfactuals),
                feature_means=means,
                feature_stds=stds,
            )

    selected = tuple(mandatory + selected_optional_ids)

    # Final validation: must be within C, unique, and sized exactly k_core.
    ok = True
    ok = ok and len(selected) == len(set(selected))
    ok = ok and all(i in candidate_items_by_id for i in selected)
    ok = ok and len(selected) == int(candidate_set.k_core)
    ok = ok and all("blocked" not in candidate_items_by_id[i].constraint_tags for i in selected)
    if not ok:
        return _fallback_decision(candidate_set=candidate_set, context=context, params=params, mapping=mapping, mode=mode)

    return _build_decision(
        candidate_set=candidate_set,
        context=context,
        params=params,
        mapping=mapping,
        mode=mode,
        selected_item_ids=selected,
        propensities=propensities,
        sampled_weights=sampled_weights,
        context_hash=context_hash,
        candidate_set_hash=candidate_set_hash,
        counterfactuals=counterfactuals,
    )


def _fallback_decision(
    *,
    candidate_set: CandidateSet,
    context: PolicyContext,
    params: PolicyParams,
    mapping: FeatureMapping,
    mode: str,
) -> PolicyDecision:
    # Fallback to deterministic baseline. Still produce a PolicyDecision for auditability.
    context_hash = _hash_context(context)
    candidate_set_hash = _hash_candidate_set(candidate_set)
    by_id: Dict[str, CandidateItem] = {c.item_id: c for c in candidate_set.candidates}
    selected: List[str] = []
    for item_id in candidate_set.mandatory_item_ids:
        c = by_id.get(str(item_id))
        if not c:
            continue
        if "blocked" in c.constraint_tags:
            continue
        selected.append(str(item_id))
    for item_id in candidate_set.deterministic_baseline_selected:
        c = by_id.get(str(item_id))
        if not c:
            continue
        if "blocked" in c.constraint_tags:
            continue
        if str(item_id) in selected:
            continue
        selected.append(str(item_id))

    if len(selected) < int(candidate_set.k_core):
        remaining = [
            c
            for c in candidate_set.candidates
            if c.item_id not in selected and "blocked" not in c.constraint_tags
        ]
        remaining.sort(key=lambda c: (float(c.deterministic_score), c.item_id), reverse=True)
        for c in remaining:
            if len(selected) >= int(candidate_set.k_core):
                break
            selected.append(c.item_id)

    if len(selected) != int(candidate_set.k_core):
        selected = list(_safe_protocol(candidate_set, by_id))

    return _build_decision(
        candidate_set=candidate_set,
        context=context,
        params=params,
        mapping=mapping,
        mode="safe_fallback" if mode != "replay" else mode,
        selected_item_ids=tuple(selected),
        propensities=None,
        sampled_weights=None,
        context_hash=context_hash,
        candidate_set_hash=candidate_set_hash,
        counterfactuals=(),
    )


def _build_decision(
    *,
    candidate_set: CandidateSet,
    context: PolicyContext,
    params: PolicyParams,
    mapping: FeatureMapping,
    mode: str,
    selected_item_ids: Sequence[str],
    propensities: Optional[Dict[str, float]],
    sampled_weights: Optional[Sequence[float]],
    context_hash: str,
    candidate_set_hash: str,
    counterfactuals: Sequence[CounterfactualExplanation],
) -> PolicyDecision:
    candidate_items_by_id: Dict[str, CandidateItem] = {c.item_id: c for c in candidate_set.candidates}
    explanations: List[SelectedItemExplanation] = []

    weights_for_explain = list(sampled_weights) if sampled_weights is not None else [0.0] * len(mapping.names)
    means, stds = _stats_from_params(params=params, dim=len(mapping.names))

    for rank, item_id in enumerate(selected_item_ids, start=1):
        c = candidate_items_by_id.get(str(item_id))
        if c is None:
            continue
        phi = featurize_v1(context, c, mapping, feature_means=means, feature_stds=stds)
        policy_score = float(_dot(weights_for_explain, phi))

        constraint_justification = ["in_candidate_set"]
        if "mandatory" in c.constraint_tags:
            constraint_justification.append("mandatory")
        if "override_allowed" in c.constraint_tags:
            constraint_justification.append("override_allowed")
        if "blocked" in c.constraint_tags:
            constraint_justification.append("blocked")  # should never happen for served items

        explanations.append(
            SelectedItemExplanation(
                item_id=item_id,
                selection_rank=int(rank),
                policy_score=policy_score,
                deterministic_score=float(c.deterministic_score),
                reason_codes=tuple(c.reason_codes),
                constraint_justification=tuple(constraint_justification),
                top_feature_contributions=tuple(top_contributions(mapping, weights_for_explain, phi, n=8)),
            )
        )

    return PolicyDecision(
        decision_id=str(uuid.uuid4()),
        user_id=candidate_set.user_id,
        day=candidate_set.day,
        policy_version=params.policy_version,
        feature_version=params.feature_version,
        mode=str(mode),
        selected_item_ids=tuple(selected_item_ids),
        propensities=propensities,
        explanations=tuple(explanations),
        counterfactuals=tuple(counterfactuals),
        deterministic_baseline_selected=tuple(candidate_set.deterministic_baseline_selected),
        context_hash=context_hash,
        candidate_set_hash=candidate_set_hash,
    )


def _safe_protocol(candidate_set: CandidateSet, by_id: Dict[str, CandidateItem]) -> Tuple[str, ...]:
    """
    Minimal safe protocol when safety triggers are active.

    v1: mandatory anchors + lowest-sensitivity, lowest-burden candidates until k_core.
    """
    selected: List[str] = []
    for item_id in candidate_set.mandatory_item_ids:
        c = by_id.get(item_id)
        if not c:
            continue
        if "blocked" in c.constraint_tags:
            continue
        selected.append(item_id)

    def key(c: CandidateItem) -> Tuple[int, float, float, str]:
        sensitivity = str(c.features.get("sensitivity") or "")
        sens_rank = 0 if sensitivity == "low" else 1 if sensitivity == "medium" else 2
        burden = float(c.features.get("expected_burden") or 1.0)
        return (sens_rank, burden, -float(c.deterministic_score), c.item_id)

    remaining = [c for c in by_id.values() if c.item_id not in selected and "blocked" not in c.constraint_tags]
    remaining.sort(key=key)
    for c in remaining:
        if len(selected) >= int(candidate_set.k_core):
            break
        selected.append(c.item_id)

    return tuple(selected[: int(candidate_set.k_core)])


def _epsilon_explore(
    *, rng: random.Random, optionals: Sequence[CandidateItem], k: int, epsilon: float
) -> Tuple[List[str], Dict[str, float]]:
    """
    Uniform exploration without replacement from the safe subset.

    Returns:
      (selected_ids, propensities)
    propensities include:
      - per-item step probability (when chosen)
      - "__slate__" joint probability of the sampled slate under this procedure
    """
    remaining = list(optionals)
    selected: List[str] = []
    prop: Dict[str, float] = {}
    slate_p = float(max(0.0, min(1.0, epsilon)))  # logging policy mixture; used by eval scripts
    for _ in range(max(0, int(k))):
        if not remaining:
            break
        p_step = 1.0 / float(len(remaining))
        idx = rng.randrange(len(remaining))
        chosen = remaining.pop(idx)
        selected.append(chosen.item_id)
        prop[chosen.item_id] = float(p_step)
        slate_p *= float(p_step)
    prop["__slate__"] = float(slate_p)
    return selected, prop


def _rank_and_select(
    *,
    context: PolicyContext,
    optionals: Sequence[CandidateItem],
    mapping: FeatureMapping,
    weights: Sequence[float],
    k: int,
    max_counterfactuals: int,
    feature_means: Optional[Sequence[float]],
    feature_stds: Optional[Sequence[float]],
) -> Tuple[List[str], Tuple[CounterfactualExplanation, ...]]:
    scored: List[Tuple[float, float, str]] = []
    by_id: Dict[str, CandidateItem] = {}
    for c in optionals:
        by_id[c.item_id] = c
        phi = featurize_v1(context, c, mapping, feature_means=feature_means, feature_stds=feature_stds)
        s = float(_dot(weights, phi))
        scored.append((s, float(c.deterministic_score), c.item_id))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)

    chosen = [item_id for _, _, item_id in scored[: max(0, int(k))]]
    counter = []
    for score, _, item_id in scored[max(0, int(k)) :][: max(0, int(max_counterfactuals))]:
        candidate = by_id[item_id]
        counter.append(
            CounterfactualExplanation(
                item_id=str(item_id),
                policy_score=float(score),
                reason_codes=tuple(candidate.reason_codes),
            )
        )
    counter_tuple = tuple(counter)
    return chosen, counter_tuple


def _stats_from_params(*, params: PolicyParams, dim: int) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if len(params.feature_means) == dim and len(params.feature_stds) == dim:
        return tuple(float(x) for x in params.feature_means), tuple(float(x) for x in params.feature_stds)
    return default_stats(dim)


def _sample_weights(*, params: PolicyParams, rng: random.Random, dim: int) -> List[float]:
    d = int(dim)
    if not params.A or not params.b:
        return [0.0] * d
    A = [list(row) for row in params.A]
    b = list(params.b)
    if len(b) != d or any(len(r) != d for r in A):
        raise ValueError("Invalid policy params dimensions")
    mean, L = _solve_posterior_mean_and_cholesky(A, b)
    g = [rng.gauss(0.0, 1.0) for _ in range(d)]
    # Sample from covariance inv(A): x = solve(A, g)
    y = _solve_lower(L, g)
    x = _solve_upper_transpose(L, y)
    return [float(m) + float(v) for m, v in zip(mean, x)]


def _solve_posterior_mean_and_cholesky(A: List[List[float]], b: List[float]) -> Tuple[List[float], List[List[float]]]:
    L = _cholesky(A)
    y = _solve_lower(L, b)
    mean = _solve_upper_transpose(L, y)
    return mean, L


def _cholesky(A: List[List[float]]) -> List[List[float]]:
    d = len(A)
    L = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i][k] * L[j][k]
            if i == j:
                val = float(A[i][i]) - s
                if val <= 1e-12:
                    raise ValueError("Posterior matrix not positive definite")
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (float(A[i][j]) - s) / max(1e-12, L[j][j])
    return L


def _solve_lower(L: List[List[float]], b: Sequence[float]) -> List[float]:
    d = len(L)
    x = [0.0] * d
    for i in range(d):
        s = float(b[i])
        for j in range(i):
            s -= L[i][j] * x[j]
        x[i] = s / max(1e-12, L[i][i])
    return x


def _solve_upper_transpose(L: List[List[float]], b: Sequence[float]) -> List[float]:
    """
    Solve L^T x = b where L is lower triangular.
    """
    d = len(L)
    x = [0.0] * d
    for i in reversed(range(d)):
        s = float(b[i])
        for j in range(i + 1, d):
            s -= L[j][i] * x[j]
        x[i] = s / max(1e-12, L[i][i])
    return x


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _hash_context(context: PolicyContext) -> str:
    payload = {
        "anifold_z_hash": stable_hash_floats(context.anifold_z, quantize=3) if context.anifold_z else None,
        "z_uncertainty_hash": stable_hash_floats(context.z_uncertainty_diag, quantize=3) if context.z_uncertainty_diag else None,
        "z_velocity_hash": stable_hash_floats(context.z_velocity, quantize=3) if context.z_velocity else None,
        "z_distance_to_attractor": round(float(context.z_distance_to_attractor or 0.0), 4),
        "completion_rate_7d": round(float(context.completion_rate_7d or 0.0), 4),
        "completion_rate_14d": round(float(context.completion_rate_14d or 0.0), 4),
        "completion_rate_30d": round(float(context.completion_rate_30d or 0.0), 4),
        "burden_ms_median_14d": round(float(context.burden_ms_median_14d or 0.0), 2),
        "day_of_week": int(context.day_of_week) if context.day_of_week is not None else None,
        "safety_trigger_active": bool(context.safety_trigger_active),
        "allow_context_batches": bool(context.allow_context_batches),
        "identity_mask_id": str(context.identity_mask_id) if context.identity_mask_id else None,
    }
    return stable_hash_hex(payload)


def _hash_candidate_set(candidate_set: CandidateSet) -> str:
    payload = {
        "user_id": candidate_set.user_id,
        "day": candidate_set.day.isoformat(),
        "k_core": int(candidate_set.k_core),
        "mandatory_item_ids": list(candidate_set.mandatory_item_ids),
        "candidates": [
            {
                "item_id": c.item_id,
                "item_type": c.item_type,
                "scale_ids": list(c.scale_ids),
                "deterministic_score": float(c.deterministic_score),
                "constraint_tags": list(c.constraint_tags),
                "reason_codes": list(c.reason_codes),
                "features": dict(c.features),
            }
            for c in candidate_set.candidates
        ],
    }
    return stable_hash_hex(payload)
