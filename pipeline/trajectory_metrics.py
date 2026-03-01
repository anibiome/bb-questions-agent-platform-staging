from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, Mapping, Optional, Sequence


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


def coherence_score_from_radius(radius: float, *, radius_at_zero: float = 1.25) -> float:
    r = max(0.0, float(radius))
    scale = max(0.25, float(radius_at_zero))
    score = 1.0 - (r / scale)
    return round(max(0.0, min(1.0, score)), 6)


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
