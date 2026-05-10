from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


COHERENCE_TIER_LADDER = (
    {
        "tier": "optimal",
        "label": "Optimal",
        "min_score": 0.85,
        "max_score": 1.0,
        "clinical_equivalent": "healthy_flexible",
        "agent_action": "baseline_only_quarterly_rescan",
        "user_message": "Core systems look coordinated.",
    },
    {
        "tier": "good",
        "label": "Good",
        "min_score": 0.65,
        "max_score": 0.85,
        "clinical_equivalent": "early_drift",
        "agent_action": "targeted_questions_within_48h",
        "user_message": "Some shifts are present and usually recoverable.",
    },
    {
        "tier": "moderate",
        "label": "Moderate",
        "min_score": 0.45,
        "max_score": 0.65,
        "clinical_equivalent": "metabolic_syndrome_range",
        "agent_action": "active_targeted_tracking",
        "user_message": "Multiple systems may be decoupling.",
    },
    {
        "tier": "elevated",
        "label": "Elevated",
        "min_score": 0.2,
        "max_score": 0.45,
        "clinical_equivalent": "active_cardiometabolic_risk",
        "agent_action": "full_rescan_plus_clinician_guidance",
        "user_message": "Strongly consider clinician review.",
    },
    {
        "tier": "critical",
        "label": "Critical",
        "min_score": 0.0,
        "max_score": 0.2,
        "clinical_equivalent": "multi_system_decoherence",
        "agent_action": "safety_protocol_and_urgent_clinician_guidance",
        "user_message": "Urgent clinician review is recommended.",
    },
)

COHERENCE_CIRCLE_GEOMETRY_CONTRACT = "ani.coherence_circle_geometry.v1"


def coherence_tier_for_radius(radius: float) -> str:
    r = max(0.0, float(radius))
    if r >= 0.85:
        return "critical"
    if r >= 0.65:
        return "elevated"
    if r >= 0.35:
        return "moderate"
    if r >= 0.15:
        return "good"
    return "optimal"


CANONICAL_STATE_DIMENSIONS: Tuple[str, ...] = (
    "energy_vitality",
    "sleep_quality",
    "gut_gi",
    "glycemic_risk",
    "cardiovascular_load",
    "mood_affect",
    "cognitive_control",
    "agency_purpose",
    "social_connectedness",
)

CANONICAL_AXIS_LOADINGS: Dict[str, Dict[str, float]] = {
    "metabolic": {
        "energy_vitality": 0.20,
        "glycemic_risk": 0.45,
        "cardiovascular_load": 0.35,
    },
    "autonomic": {
        "sleep_quality": 0.35,
        "cardiovascular_load": 0.30,
        "cognitive_control": 0.35,
    },
    "immune": {
        "gut_gi": 0.65,
        "energy_vitality": 0.20,
        "cardiovascular_load": 0.15,
    },
    "affective": {
        "mood_affect": 0.35,
        "cognitive_control": 0.15,
        "agency_purpose": 0.25,
        "social_connectedness": 0.25,
    },
}

CANONICAL_AXIS_ANGLES: Dict[str, float] = {
    "metabolic": math.radians(45.0),
    "autonomic": math.radians(135.0),
    "immune": math.radians(225.0),
    "affective": math.radians(315.0),
}

CANONICAL_COMPONENT_WEIGHTS: Dict[str, float] = {
    "distance": 0.40,
    "stability": 0.15,
    "concordance": 0.20,
    "recovery": 0.15,
    "alignment": 0.10,
}

CANONICAL_IDEAL_STATE: Dict[str, float] = {
    "energy_vitality": 0.82,
    "sleep_quality": 0.82,
    "gut_gi": 0.78,
    "glycemic_risk": 0.18,
    "cardiovascular_load": 0.18,
    "mood_affect": 0.82,
    "cognitive_control": 0.80,
    "agency_purpose": 0.78,
    "social_connectedness": 0.76,
}

_EPS = 1e-9
_CANONICAL_DISTANCE_SCALE = 0.75
_CANONICAL_VELOCITY_SCALE = 0.15
_CANONICAL_SIGMA_REFERENCE = 0.25
_CANONICAL_IDEAL_SIGMA = 0.10


def coherence_score_from_radius(radius: float, *, radius_at_zero: float = 1.25) -> float:
    r = max(0.0, float(radius))
    scale = max(0.25, float(radius_at_zero))
    score = 1.0 - (r / scale)
    return round(max(0.0, min(1.0, score)), 6)


def canonical_prior_state(
    *,
    state_dimensions: Sequence[str] = CANONICAL_STATE_DIMENSIONS,
) -> Dict[str, float]:
    return {str(dim): 0.5 for dim in state_dimensions}


def canonical_prior_sigma_diag(
    *,
    state_dimensions: Sequence[str] = CANONICAL_STATE_DIMENSIONS,
    sigma: float = _CANONICAL_SIGMA_REFERENCE,
) -> Dict[str, float]:
    return {str(dim): float(sigma) for dim in state_dimensions}


def canonical_ideal_state(
    *,
    state_dimensions: Sequence[str] = CANONICAL_STATE_DIMENSIONS,
) -> Dict[str, float]:
    return {
        str(dim): float(CANONICAL_IDEAL_STATE.get(str(dim), 0.5))
        for dim in state_dimensions
    }


def canonical_ideal_sigma_diag(
    *,
    state_dimensions: Sequence[str] = CANONICAL_STATE_DIMENSIONS,
    sigma: float = _CANONICAL_IDEAL_SIGMA,
) -> Dict[str, float]:
    return {str(dim): float(sigma) for dim in state_dimensions}


def build_feasible_reference(
    *,
    current_mu: Mapping[str, float],
    current_sigma_diag: Mapping[str, float],
    previous_attractor_mu: Optional[Mapping[str, float]] = None,
    previous_attractor_sigma_diag: Optional[Mapping[str, float]] = None,
    state_dimensions: Sequence[str] = CANONICAL_STATE_DIMENSIONS,
    prior_strength: float = 4.0,
    base_alpha: float = 0.15,
    stable_distance: float = 0.50,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Estimate the best currently reachable stable state.

    This is the personalized feasible reference, not the final destination.
    Supercoherence is handled separately by ``canonical_ideal_state``.
    """
    prior_mu = canonical_prior_state(state_dimensions=state_dimensions)
    prior_sigma = canonical_prior_sigma_diag(state_dimensions=state_dimensions)

    if previous_attractor_mu and previous_attractor_sigma_diag:
        distance = canonical_distance_from_state(
            mu=current_mu,
            sigma_diag=current_sigma_diag,
            attractor_mu=previous_attractor_mu,
            attractor_sigma_diag=previous_attractor_sigma_diag,
            state_dimensions=state_dimensions,
        )
        stability_gate = math.exp(-((distance / max(stable_distance, _EPS)) ** 2))
        alpha = float(base_alpha) * stability_gate
        attractor_mu: Dict[str, float] = {}
        attractor_sigma: Dict[str, float] = {}
        for dim in state_dimensions:
            key = str(dim)
            prev_mu = float(previous_attractor_mu.get(key, prior_mu[key]))
            prev_sigma = float(previous_attractor_sigma_diag.get(key, prior_sigma[key]))
            cur_mu = float(current_mu.get(key, prev_mu))
            cur_sigma = float(current_sigma_diag.get(key, prev_sigma))
            attractor_mu[key] = (1.0 - alpha) * prev_mu + alpha * cur_mu
            attractor_sigma[key] = (1.0 - alpha) * prev_sigma + alpha * cur_sigma
        return attractor_mu, attractor_sigma

    weight = 0.0 / max(_EPS, 0.0 + float(prior_strength))
    attractor_mu = {}
    attractor_sigma = {}
    for dim in state_dimensions:
        key = str(dim)
        prior_mu_val = float(prior_mu[key])
        prior_sigma_val = float(prior_sigma[key])
        cur_mu = float(current_mu.get(key, prior_mu_val))
        cur_sigma = float(current_sigma_diag.get(key, prior_sigma_val))
        attractor_mu[key] = (1.0 - weight) * prior_mu_val + weight * cur_mu
        attractor_sigma[key] = (1.0 - weight) * prior_sigma_val + weight * cur_sigma
    return attractor_mu, attractor_sigma


def build_personal_attractor(
    *,
    current_mu: Mapping[str, float],
    current_sigma_diag: Mapping[str, float],
    previous_attractor_mu: Optional[Mapping[str, float]] = None,
    previous_attractor_sigma_diag: Optional[Mapping[str, float]] = None,
    state_dimensions: Sequence[str] = CANONICAL_STATE_DIMENSIONS,
    prior_strength: float = 4.0,
    base_alpha: float = 0.15,
    stable_distance: float = 0.50,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Backward-compatible alias for the older personalized-attractor API."""
    return build_feasible_reference(
        current_mu=current_mu,
        current_sigma_diag=current_sigma_diag,
        previous_attractor_mu=previous_attractor_mu,
        previous_attractor_sigma_diag=previous_attractor_sigma_diag,
        state_dimensions=state_dimensions,
        prior_strength=prior_strength,
        base_alpha=base_alpha,
        stable_distance=stable_distance,
    )


def build_feasible_optimum(
    *,
    current_mu: Mapping[str, float],
    current_sigma_diag: Mapping[str, float],
    previous_feasible_mu: Optional[Mapping[str, float]] = None,
    previous_feasible_sigma_diag: Optional[Mapping[str, float]] = None,
    state_dimensions: Sequence[str] = CANONICAL_STATE_DIMENSIONS,
    prior_strength: float = 4.0,
    base_alpha: float = 0.15,
    stable_distance: float = 0.50,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Backward-compatible wrapper around the feasible-reference machinery.

    The old attractor becomes the feasible optimum: the best state the person
    can stably occupy right now. The absolute destination is handled separately
    via ``canonical_ideal_state``.
    """
    return build_feasible_reference(
        current_mu=current_mu,
        current_sigma_diag=current_sigma_diag,
        previous_attractor_mu=previous_feasible_mu,
        previous_attractor_sigma_diag=previous_feasible_sigma_diag,
        state_dimensions=state_dimensions,
        prior_strength=prior_strength,
        base_alpha=base_alpha,
        stable_distance=stable_distance,
    )


def canonical_distance_from_state(
    *,
    mu: Mapping[str, float],
    sigma_diag: Mapping[str, float],
    attractor_mu: Mapping[str, float],
    attractor_sigma_diag: Mapping[str, float],
    state_dimensions: Sequence[str] = CANONICAL_STATE_DIMENSIONS,
) -> float:
    total = 0.0
    count = 0
    for dim in state_dimensions:
        key = str(dim)
        delta = float(mu.get(key, 0.0)) - float(attractor_mu.get(key, 0.0))
        variance = (
            float(sigma_diag.get(key, _CANONICAL_SIGMA_REFERENCE))
            + float(attractor_sigma_diag.get(key, _CANONICAL_SIGMA_REFERENCE))
            + _EPS
        )
        total += (delta * delta) / variance
        count += 1
    return math.sqrt(total / max(1, count))


def decompose_rejuvenation_geometry(
    *,
    current_mu: Mapping[str, float],
    current_sigma_diag: Mapping[str, float],
    previous_feasible_mu: Optional[Mapping[str, float]] = None,
    previous_feasible_sigma_diag: Optional[Mapping[str, float]] = None,
    state_dimensions: Sequence[str] = CANONICAL_STATE_DIMENSIONS,
    velocity: Optional[float] = None,
    concordance: Optional[float] = None,
    recovery: Optional[float] = None,
) -> Dict[str, Any]:
    """Split state error into acute dysregulation, structural gap, and total gap."""
    feasible_mu, feasible_sigma = build_feasible_optimum(
        current_mu=current_mu,
        current_sigma_diag=current_sigma_diag,
        previous_feasible_mu=previous_feasible_mu,
        previous_feasible_sigma_diag=previous_feasible_sigma_diag,
        state_dimensions=state_dimensions,
    )
    ideal_mu = canonical_ideal_state(state_dimensions=state_dimensions)
    ideal_sigma = canonical_ideal_sigma_diag(state_dimensions=state_dimensions)

    acute_distance = canonical_distance_from_state(
        mu=current_mu,
        sigma_diag=current_sigma_diag,
        attractor_mu=feasible_mu,
        attractor_sigma_diag=feasible_sigma,
        state_dimensions=state_dimensions,
    )
    structural_distance = canonical_distance_from_state(
        mu=feasible_mu,
        sigma_diag=feasible_sigma,
        attractor_mu=ideal_mu,
        attractor_sigma_diag=ideal_sigma,
        state_dimensions=state_dimensions,
    )
    absolute_distance = canonical_distance_from_state(
        mu=current_mu,
        sigma_diag=current_sigma_diag,
        attractor_mu=ideal_mu,
        attractor_sigma_diag=ideal_sigma,
        state_dimensions=state_dimensions,
    )

    local_axis_scores = semantic_axis_scores_from_state(
        mu=current_mu,
        sigma_diag=current_sigma_diag,
        attractor_mu=feasible_mu,
        attractor_sigma_diag=feasible_sigma,
        axis_loadings=CANONICAL_AXIS_LOADINGS,
    )
    structural_axis_scores = semantic_axis_scores_from_state(
        mu=feasible_mu,
        sigma_diag=feasible_sigma,
        attractor_mu=ideal_mu,
        attractor_sigma_diag=ideal_sigma,
        axis_loadings=CANONICAL_AXIS_LOADINGS,
    )
    absolute_axis_scores = semantic_axis_scores_from_state(
        mu=current_mu,
        sigma_diag=current_sigma_diag,
        attractor_mu=ideal_mu,
        attractor_sigma_diag=ideal_sigma,
        axis_loadings=CANONICAL_AXIS_LOADINGS,
    )

    local_theta, local_theta_defined, local_concentration = semantic_direction(local_axis_scores)
    structural_theta, structural_theta_defined, structural_concentration = semantic_direction(structural_axis_scores)
    absolute_theta, absolute_theta_defined, absolute_concentration = semantic_direction(absolute_axis_scores)

    local_coherence = canonical_coherence_score(
        distance=acute_distance,
        velocity=velocity,
        concordance=concordance,
        recovery=recovery,
        semantic_concentration=local_concentration if local_theta_defined else None,
    )
    absolute_coherence = canonical_coherence_score(
        distance=absolute_distance,
        velocity=velocity,
        concordance=concordance,
        recovery=recovery,
        semantic_concentration=absolute_concentration if absolute_theta_defined else None,
    )

    return {
        "feasible_state": feasible_mu,
        "feasible_sigma_diag": feasible_sigma,
        "ideal_state": ideal_mu,
        "ideal_sigma_diag": ideal_sigma,
        "acute_vector": _delta_map(current_mu, feasible_mu, state_dimensions=state_dimensions),
        "structural_vector": _delta_map(feasible_mu, ideal_mu, state_dimensions=state_dimensions),
        "absolute_vector": _delta_map(current_mu, ideal_mu, state_dimensions=state_dimensions),
        "acute_distance": round(float(acute_distance), 6),
        "structural_distance": round(float(structural_distance), 6),
        "absolute_distance": round(float(absolute_distance), 6),
        "local_coherence": round(float(local_coherence), 6),
        "absolute_coherence": round(float(absolute_coherence), 6),
        "local_axis_scores": {k: round(float(v), 6) for k, v in local_axis_scores.items()},
        "structural_axis_scores": {k: round(float(v), 6) for k, v in structural_axis_scores.items()},
        "absolute_axis_scores": {k: round(float(v), 6) for k, v in absolute_axis_scores.items()},
        "local_theta": round(float(local_theta), 6),
        "structural_theta": round(float(structural_theta), 6),
        "absolute_theta": round(float(absolute_theta), 6),
        "local_theta_defined": bool(local_theta_defined),
        "structural_theta_defined": bool(structural_theta_defined),
        "absolute_theta_defined": bool(absolute_theta_defined),
        "local_concentration": round(float(local_concentration), 6),
        "structural_concentration": round(float(structural_concentration), 6),
        "absolute_concentration": round(float(absolute_concentration), 6),
        "viable_volume_log_proxy": round(
            sum(math.log(1.0 / max(float(feasible_sigma.get(str(dim), _CANONICAL_SIGMA_REFERENCE)), _EPS)) for dim in state_dimensions)
            / max(1, len(state_dimensions)),
            6,
        ),
        "uncertainty_trace_proxy": round(
            (
                sum(float(current_sigma_diag.get(str(dim), _CANONICAL_SIGMA_REFERENCE)) for dim in state_dimensions)
                + sum(float(feasible_sigma.get(str(dim), _CANONICAL_SIGMA_REFERENCE)) for dim in state_dimensions)
                + sum(float(ideal_sigma.get(str(dim), _CANONICAL_SIGMA_REFERENCE)) for dim in state_dimensions)
            )
            / max(1, len(state_dimensions)),
            6,
        ),
    }


def semantic_axis_scores_from_state(
    *,
    mu: Mapping[str, float],
    sigma_diag: Mapping[str, float],
    attractor_mu: Mapping[str, float],
    attractor_sigma_diag: Mapping[str, float],
    axis_loadings: Mapping[str, Mapping[str, float]] = CANONICAL_AXIS_LOADINGS,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for axis, loadings in axis_loadings.items():
        weighted = 0.0
        total_weight = 0.0
        for dim, weight in loadings.items():
            abs_weight = abs(float(weight))
            if abs_weight <= 0.0:
                continue
            delta = float(mu.get(dim, 0.0)) - float(attractor_mu.get(dim, 0.0))
            variance = (
                float(sigma_diag.get(dim, _CANONICAL_SIGMA_REFERENCE))
                + float(attractor_sigma_diag.get(dim, _CANONICAL_SIGMA_REFERENCE))
                + _EPS
            )
            weighted += abs_weight * ((delta * delta) / variance)
            total_weight += abs_weight
        scores[str(axis)] = math.sqrt(weighted / max(total_weight, _EPS)) if total_weight > 0.0 else 0.0
    return scores


def semantic_direction(
    axis_scores: Mapping[str, float],
) -> Tuple[float, bool, float]:
    total = sum(max(0.0, float(value)) for value in axis_scores.values())
    if total <= _EPS:
        return 0.0, False, 0.0

    x = 0.0
    y = 0.0
    for axis, score in axis_scores.items():
        angle = CANONICAL_AXIS_ANGLES.get(str(axis))
        if angle is None:
            continue
        x += float(score) * math.cos(angle)
        y += float(score) * math.sin(angle)

    theta = math.atan2(y, x) if abs(x) > _EPS or abs(y) > _EPS else 0.0
    concentration = max(0.0, min(1.0, math.sqrt(x * x + y * y) / total))
    return theta, True, concentration


def canonical_coherence_score(
    *,
    distance: float,
    velocity: Optional[float] = None,
    concordance: Optional[float] = None,
    recovery: Optional[float] = None,
    semantic_concentration: Optional[float] = None,
) -> float:
    components: Dict[str, Optional[float]] = {
        "distance": _clamp01(math.exp(-max(0.0, float(distance)) / _CANONICAL_DISTANCE_SCALE)),
        "stability": None if velocity is None else _clamp01(math.exp(-abs(float(velocity)) / _CANONICAL_VELOCITY_SCALE)),
        "concordance": None if concordance is None else _clamp01(float(concordance)),
        "recovery": None if recovery is None else _clamp01(float(recovery)),
        "alignment": None if semantic_concentration is None else _clamp01(float(semantic_concentration)),
    }

    total_weight = 0.0
    log_sum = 0.0
    for name, value in components.items():
        if value is None:
            continue
        weight = float(CANONICAL_COMPONENT_WEIGHTS.get(name, 0.0))
        if weight <= 0.0:
            continue
        total_weight += weight
        log_sum += weight * math.log(max(float(value), _EPS))
    if total_weight <= _EPS:
        return 0.0
    return round(_clamp01(math.exp(log_sum / total_weight)), 6)


def coherence_uncertainty_from_state(
    *,
    sigma_diag: Mapping[str, float],
    attractor_sigma_diag: Mapping[str, float],
    concordance: Optional[float] = None,
    state_dimensions: Sequence[str] = CANONICAL_STATE_DIMENSIONS,
) -> float:
    state_component = _scaled_uncertainty(sigma_diag, state_dimensions=state_dimensions)
    attractor_component = _scaled_uncertainty(attractor_sigma_diag, state_dimensions=state_dimensions)
    modality_component = None if concordance is None else _clamp01(1.0 - float(concordance))
    weights = {"state": 0.50, "attractor": 0.25, "modality": 0.25}
    total = 0.0
    total_weight = 0.0
    for name, value in (
        ("state", state_component),
        ("attractor", attractor_component),
        ("modality", modality_component),
    ):
        if value is None:
            continue
        total += weights[name] * float(value)
        total_weight += weights[name]
    if total_weight <= _EPS:
        return 1.0
    return round(_clamp01(total / total_weight), 6)


def coherence_tier_for_score(score: float) -> Dict[str, Any]:
    s = max(0.0, min(1.0, float(score)))
    for row in COHERENCE_TIER_LADDER:
        lo = float(row["min_score"])
        hi = float(row["max_score"])
        if row["tier"] == "optimal":
            if lo <= s <= hi:
                return _with_score(row, s)
            continue
        if lo <= s < hi:
            return _with_score(row, s)
    return _with_score(COHERENCE_TIER_LADDER[-1], s)


def coherence_tier_contract() -> Dict[str, Any]:
    return {
        "version": "coherence_tier_contract_v1",
        "score_range": {"min": 0.0, "max": 1.0},
        "tiers": [dict(row) for row in COHERENCE_TIER_LADDER],
        "disclaimer": "Decision support only, not diagnosis.",
    }


def compute_ews_features(
    *,
    circle_history: Sequence[Mapping[str, Any]],
    window_size_days: int = 14,
) -> Dict[str, float]:
    rows = list(circle_history)
    r_vals = [float(r.get("r") or 0.0) for r in rows]
    vel_vals = [float(r.get("velocity") or 0.0) for r in rows]
    acc_vals = [float(r.get("acceleration") or 0.0) for r in rows]
    days = [_extract_date(r) for r in rows]

    var_r = _variance(r_vals)
    ac1_r = _autocorr_lag1(r_vals)
    trend_speed = float(vel_vals[-1]) if vel_vals else 0.0
    trend_accel = float(acc_vals[-1]) if acc_vals else 0.0
    recovery_rate = _recovery_rate(r_vals, days)

    var_n = min(1.0, var_r / 0.02) if var_r > 0.0 else 0.0
    ac_n = max(0.0, min(1.0, (ac1_r + 1.0) / 2.0))
    speed_n = min(1.0, abs(trend_speed) / 0.20)
    accel_n = min(1.0, abs(trend_accel) / 0.10)
    non_recovery_n = max(0.0, 1.0 - min(1.0, max(0.0, recovery_rate) / 0.05))
    ews_score = (
        0.35 * var_n
        + 0.25 * ac_n
        + 0.20 * speed_n
        + 0.10 * accel_n
        + 0.10 * non_recovery_n
    )

    return {
        "window_size_days": float(max(1, int(window_size_days))),
        "var_r": round(var_r, 6),
        "ac1_r": round(ac1_r, 6),
        "trend_speed": round(trend_speed, 6),
        "trend_accel": round(trend_accel, 6),
        "recovery_rate": round(recovery_rate, 6),
        "ews_score": round(max(0.0, min(1.0, ews_score)), 6),
    }


def _with_score(row: Mapping[str, Any], score: float) -> Dict[str, Any]:
    out = dict(row)
    out["score"] = round(float(score), 6)
    return out


def _variance(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = sum(float(v) for v in values) / float(n)
    return sum((float(v) - mu) ** 2 for v in values) / float(n - 1)


def _autocorr_lag1(values: Sequence[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mu = sum(float(v) for v in values) / float(n)
    num = 0.0
    den = 0.0
    for i in range(1, n):
        num += (float(values[i]) - mu) * (float(values[i - 1]) - mu)
    for v in values:
        den += (float(v) - mu) ** 2
    if den <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def _recovery_rate(r_vals: Sequence[float], days: Sequence[Optional[date]]) -> float:
    if len(r_vals) < 2:
        return 0.0
    peak_idx = max(range(len(r_vals)), key=lambda i: float(r_vals[i]))
    last_idx = len(r_vals) - 1
    if peak_idx >= last_idx:
        return 0.0
    peak = float(r_vals[peak_idx])
    last = float(r_vals[last_idx])
    if peak <= last:
        return 0.0
    peak_day = days[peak_idx]
    last_day = days[last_idx]
    if peak_day and last_day:
        delta_days = max(1, int((last_day - peak_day).days))
    else:
        delta_days = max(1, int(last_idx - peak_idx))
    return (peak - last) / float(delta_days)


def _extract_date(row: Mapping[str, Any]) -> Optional[date]:
    raw = row.get("date")
    if isinstance(raw, date):
        return raw
    s = str(raw or "")
    if len(s) < 10:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / math.sqrt(na * nb)))


_MODALITY_TARGET_DIMS: Dict[str, Sequence[str]] = {
    "wearable": ("energy_vitality", "sleep_quality", "cardiovascular_load"),
    "voice": ("mood_affect", "energy_vitality", "cognitive_control"),
    "imaging": ("glycemic_risk", "cardiovascular_load", "gut_gi"),
    "multispectral": ("glycemic_risk", "cardiovascular_load", "gut_gi"),
    "omics": tuple((
        "energy_vitality",
        "sleep_quality",
        "gut_gi",
        "glycemic_risk",
        "cardiovascular_load",
        "mood_affect",
        "cognitive_control",
        "agency_purpose",
        "social_connectedness",
    )),
    "labs": ("glycemic_risk", "cardiovascular_load", "gut_gi"),
    "questionnaire": ("mood_affect", "sleep_quality", "energy_vitality"),
}


def confidence_to_quality(confidence: str, explicit_quality: Optional[float] = None) -> float:
    if explicit_quality is not None:
        return round(max(0.0, min(1.0, float(explicit_quality))), 6)
    c = str(confidence or "medium").strip().lower()
    if c == "high":
        return 0.9
    if c == "low":
        return 0.35
    return 0.65


def modality_target_dimensions(modality: str) -> Sequence[str]:
    key = str(modality or "").strip().lower()
    return tuple(_MODALITY_TARGET_DIMS.get(key) or ("energy_vitality", "sleep_quality", "cardiovascular_load"))


def build_uncertainty_coupling(
    *,
    modality: str,
    features: Mapping[str, Any],
    quality_score: float,
    x_hat: Mapping[str, float],
    x_uncertainty: Mapping[str, float],
) -> Dict[str, Any]:
    target_dims = tuple(modality_target_dimensions(modality))
    state_vec = [float(x_hat.get(dim, 0.5)) for dim in target_dims]
    observed_vec = _extract_observed_vector(features, target_len=len(state_vec))
    if observed_vec:
        agreement_score = (cosine_similarity(observed_vec, state_vec) + 1.0) / 2.0
        agreement_score = max(0.0, min(1.0, agreement_score))
    else:
        agreement_score = None

    if agreement_score is None:
        agreement_label = "insufficient_signal"
        reason_codes = ["missing_numeric_features"]
    elif agreement_score >= 0.75:
        agreement_label = "agreement"
        reason_codes = ["cross_modal_agreement"]
    elif agreement_score <= 0.40:
        agreement_label = "disagreement"
        reason_codes = ["cross_modal_disagreement"]
    else:
        agreement_label = "mixed"
        reason_codes = ["cross_modal_partial_agreement"]

    uncertainty_delta: Dict[str, float] = {}
    for dim in target_dims:
        if agreement_score is None:
            delta = 0.0
        else:
            delta = (0.5 - float(agreement_score)) * float(quality_score) * 0.2
        uncertainty_delta[str(dim)] = round(float(delta), 6)

    unc_before = [float(x_uncertainty.get(dim, 1.0)) for dim in target_dims]
    unc_after = [
        round(max(0.0, min(1.0, float(u) + float(uncertainty_delta.get(dim, 0.0)))), 6)
        for dim, u in zip(target_dims, unc_before)
    ]

    return {
        "modality": str(modality),
        "target_dimensions": list(target_dims),
        "quality_score": round(float(quality_score), 6),
        "agreement_score": round(float(agreement_score), 6) if agreement_score is not None else None,
        "agreement_label": agreement_label,
        "uncertainty_delta_recommendation": uncertainty_delta,
        "mean_uncertainty_before": round(sum(unc_before) / max(1, len(unc_before)), 6),
        "mean_uncertainty_after_recommended": round(sum(unc_after) / max(1, len(unc_after)), 6),
        "reason_codes": reason_codes,
    }


def _extract_observed_vector(features: Mapping[str, Any], *, target_len: int) -> Sequence[float]:
    if target_len <= 0:
        return []
    if isinstance(features.get("projection"), list):
        vals = [float(v) for v in features.get("projection", []) if isinstance(v, (int, float))]
        if vals:
            return _normalize_vector(vals, target_len)

    if isinstance(features.get("vector"), list):
        vals = [float(v) for v in features.get("vector", []) if isinstance(v, (int, float))]
        if vals:
            return _normalize_vector(vals, target_len)

    numeric = [float(v) for v in features.values() if isinstance(v, (int, float))]
    if not numeric:
        return []
    return _normalize_vector(numeric, target_len)


def _normalize_vector(values: Sequence[float], target_len: int) -> Sequence[float]:
    if not values:
        return [0.5] * int(target_len)
    vals = [float(v) for v in values]
    # Clamp and rescale into [0,1] if values look percentage-like; otherwise sigmoid.
    out: list[float] = []
    for v in vals:
        if 0.0 <= v <= 1.0:
            out.append(v)
        elif 0.0 <= v <= 100.0:
            out.append(v / 100.0)
        else:
            out.append(1.0 / (1.0 + math.exp(-v / 10.0)))
    if len(out) >= int(target_len):
        return [round(float(v), 6) for v in out[: int(target_len)]]
    padded = list(out)
    while len(padded) < int(target_len):
        padded.append(float(out[-1]))
    return [round(float(v), 6) for v in padded]


def _scaled_uncertainty(
    sigma_diag: Mapping[str, float],
    *,
    state_dimensions: Sequence[str],
) -> float:
    values = [
        _clamp01(float(sigma_diag.get(str(dim), _CANONICAL_SIGMA_REFERENCE)) / _CANONICAL_SIGMA_REFERENCE)
        for dim in state_dimensions
    ]
    return sum(values) / max(1, len(values))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _delta_map(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    state_dimensions: Sequence[str],
) -> Dict[str, float]:
    return {
        str(dim): round(float(left.get(str(dim), 0.0)) - float(right.get(str(dim), 0.0)), 6)
        for dim in state_dimensions
    }
