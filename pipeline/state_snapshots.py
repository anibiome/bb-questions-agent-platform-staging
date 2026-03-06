from __future__ import annotations

import math
from datetime import date
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from questions_agent_platform.pipeline.registry import Registry
from questions_agent_platform.pipeline.trajectory_metrics import (
    CANONICAL_AXIS_LOADINGS,
    build_personal_attractor,
    canonical_coherence_score,
    canonical_distance_from_state,
    coherence_uncertainty_from_state,
    semantic_axis_scores_from_state,
    semantic_direction,
)


STATE_SCHEMA_VERSION_DEFAULT = "v1_state_schema"
STATE_MODEL_VERSION = "state_service_v1"
CIRCLE_PROJECTION_VERSION = "circle_projection_v2_canonical"
CIRCLE_ANCHOR_VERSION = "personal_attractor_state_v2"

STATE_DIMENSIONS: Tuple[str, ...] = (
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

_TAG_TO_DIMS: Dict[str, Tuple[str, ...]] = {
    "energy": ("energy_vitality",),
    "fatigue": ("energy_vitality",),
    "vitality": ("energy_vitality",),
    "wellbeing": ("energy_vitality", "mood_affect"),
    "activity": ("energy_vitality", "cardiovascular_load"),
    "exercise": ("energy_vitality", "cardiovascular_load"),
    "sleep": ("sleep_quality",),
    "rest": ("sleep_quality",),
    "circadian": ("sleep_quality",),
    "gut": ("gut_gi",),
    "gi": ("gut_gi",),
    "digestion": ("gut_gi",),
    "stool": ("gut_gi",),
    "ibs": ("gut_gi",),
    "metabolic": ("glycemic_risk", "cardiovascular_load"),
    "glucose": ("glycemic_risk",),
    "diabetes": ("glycemic_risk",),
    "liver": ("glycemic_risk", "cardiovascular_load"),
    "nafld": ("glycemic_risk", "cardiovascular_load"),
    "cardio": ("cardiovascular_load",),
    "cardiovascular": ("cardiovascular_load",),
    "cvd": ("cardiovascular_load",),
    "kidney": ("cardiovascular_load",),
    "renal": ("cardiovascular_load",),
    "bp": ("cardiovascular_load",),
    "hypertension": ("cardiovascular_load",),
    "alcohol": ("cardiovascular_load", "mood_affect"),
    "mood": ("mood_affect",),
    "anxiety": ("mood_affect",),
    "depression": ("mood_affect",),
    "stress": ("mood_affect",),
    "affect": ("mood_affect",),
    "cognition": ("cognitive_control",),
    "focus": ("cognitive_control",),
    "executive": ("cognitive_control",),
    "agency": ("agency_purpose",),
    "purpose": ("agency_purpose",),
    "control": ("agency_purpose",),
    "social": ("social_connectedness",),
    "isolation": ("social_connectedness",),
    "support": ("social_connectedness",),
}

_STATE_TO_CIRCLE_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "energy_vitality": (0.92, 0.18),
    "sleep_quality": (0.55, 0.76),
    "gut_gi": (-0.44, 0.70),
    "glycemic_risk": (-0.87, 0.20),
    "cardiovascular_load": (-0.71, -0.55),
    "mood_affect": (0.20, -0.84),
    "cognitive_control": (0.76, -0.42),
    "agency_purpose": (0.68, 0.47),
    "social_connectedness": (0.28, 0.63),
}


def compute_state_snapshot(
    *,
    registry: Registry,
    latest_scores: Dict[str, Dict[str, Any]],
    previous_x_hat: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    values: Dict[str, List[float]] = {d: [] for d in STATE_DIMENSIONS}
    abs_z: Dict[str, List[float]] = {d: [] for d in STATE_DIMENSIONS}
    scale_ids_by_dim: Dict[str, Set[str]] = {d: set() for d in STATE_DIMENSIONS}

    prev = previous_x_hat or {}
    for scale_id, score in latest_scores.items():
        scale = registry.scales.get(scale_id)
        if not scale:
            continue
        dims = infer_state_dimensions_from_tags(scale.tags)
        if not dims:
            continue
        val = _clip01(float(score.get("normalized_score", 50.0)) / 100.0)
        z_abs = abs(float(score.get("personal_z") or 0.0))
        for dim in dims:
            values[dim].append(val)
            abs_z[dim].append(z_abs)
            scale_ids_by_dim[dim].add(scale_id)

    x_hat: Dict[str, float] = {}
    x_uncertainty: Dict[str, float] = {}
    coverage: Dict[str, Dict[str, Any]] = {}
    for dim in STATE_DIMENSIONS:
        dim_vals = values[dim]
        if dim_vals:
            x_hat_val = float(mean(dim_vals))
        else:
            x_hat_val = float(prev.get(dim, 0.5))
        x_hat[dim] = round(_clip01(x_hat_val), 6)

        if dim_vals:
            base_unc = 1.0 / (1.0 + float(len(dim_vals)))
            conflict_unc = min(0.35, float(mean(abs_z[dim])) / 10.0) if abs_z[dim] else 0.0
            unc = min(1.0, base_unc + conflict_unc)
        else:
            unc = 1.0
        x_uncertainty[dim] = round(float(unc), 6)

        coverage[dim] = {
            "scale_count": int(len(scale_ids_by_dim[dim])),
            "scales": sorted(scale_ids_by_dim[dim]),
        }

    return {
        "x_hat": x_hat,
        "x_uncertainty": x_uncertainty,
        "coverage": coverage,
    }


def compute_circle_snapshot(
    *,
    day: date,
    x_hat: Dict[str, float],
    x_uncertainty: Dict[str, float],
    previous_circle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prev = previous_circle or {}
    prev_attractor_state = prev.get("attractor_state")
    prev_attractor_sigma_diag = prev.get("attractor_sigma_diag")
    if not isinstance(prev_attractor_state, dict):
        prev_uncertainty = prev.get("uncertainty")
        if isinstance(prev_uncertainty, dict):
            state_candidate = prev_uncertainty.get("attractor_state")
            sigma_candidate = prev_uncertainty.get("attractor_sigma_diag")
            if isinstance(state_candidate, dict):
                prev_attractor_state = state_candidate
            if isinstance(sigma_candidate, dict):
                prev_attractor_sigma_diag = sigma_candidate

    attractor_state, attractor_sigma_diag = build_personal_attractor(
        current_mu=x_hat,
        current_sigma_diag=x_uncertainty,
        previous_attractor_mu=prev_attractor_state if isinstance(prev_attractor_state, dict) else None,
        previous_attractor_sigma_diag=prev_attractor_sigma_diag if isinstance(prev_attractor_sigma_diag, dict) else None,
        state_dimensions=STATE_DIMENSIONS,
    )
    r = canonical_distance_from_state(
        mu=x_hat,
        sigma_diag=x_uncertainty,
        attractor_mu=attractor_state,
        attractor_sigma_diag=attractor_sigma_diag,
        state_dimensions=STATE_DIMENSIONS,
    )
    semantic_axis_scores = semantic_axis_scores_from_state(
        mu=x_hat,
        sigma_diag=x_uncertainty,
        attractor_mu=attractor_state,
        attractor_sigma_diag=attractor_sigma_diag,
        axis_loadings=CANONICAL_AXIS_LOADINGS,
    )
    theta, theta_defined, semantic_concentration = semantic_direction(semantic_axis_scores)
    z_star = _project_state_to_2d(attractor_state)
    z = [
        float(z_star[0]) + float(r) * math.cos(theta),
        float(z_star[1]) + float(r) * math.sin(theta),
    ]

    velocity = 0.0
    acceleration = 0.0
    prev_r = prev.get("r")
    observed_velocity: Optional[float] = None
    if prev_r is not None:
        prev_day = _coerce_date(prev.get("date"))
        delta_days = max(1.0, float((day - prev_day).days)) if prev_day else 1.0
        velocity = abs(float(r) - float(prev_r)) / delta_days
        observed_velocity = float(velocity)
        prev_v = float(prev.get("velocity", 0.0) or 0.0)
        acceleration = (float(velocity) - prev_v) / delta_days

    unc_vals = [float(x_uncertainty.get(dim, 1.0)) for dim in STATE_DIMENSIONS]
    unc_mean = float(sum(unc_vals) / max(1, len(unc_vals)))
    unc_max = float(max(unc_vals) if unc_vals else 1.0)
    circle_unc = coherence_uncertainty_from_state(
        sigma_diag=x_uncertainty,
        attractor_sigma_diag=attractor_sigma_diag,
        state_dimensions=STATE_DIMENSIONS,
    )
    coherence = canonical_coherence_score(
        distance=r,
        velocity=observed_velocity,
        semantic_concentration=semantic_concentration if theta_defined else None,
    )
    coherence_components = {
        "distance": round(float(math.exp(-max(0.0, float(r)) / 0.75)), 6),
        "stability": (
            round(float(math.exp(-abs(float(velocity)) / 0.15)), 6)
            if observed_velocity is not None
            else None
        ),
        "alignment": round(float(semantic_concentration), 6) if theta_defined else None,
    }
    rounded_axis_scores = {
        key: round(float(value), 6) for key, value in semantic_axis_scores.items()
    }
    rounded_attractor_state = {
        key: round(float(value), 6) for key, value in attractor_state.items()
    }
    rounded_attractor_sigma_diag = {
        key: round(float(value), 6) for key, value in attractor_sigma_diag.items()
    }

    return {
        "z": [round(float(z[0]), 6), round(float(z[1]), 6)],
        "z_star": [round(float(z_star[0]), 6), round(float(z_star[1]), 6)],
        "r": round(float(r), 6),
        "theta": round(float(theta), 6),
        "theta_defined": bool(theta_defined),
        "semantic_axis_scores": rounded_axis_scores,
        "semantic_concentration": round(float(semantic_concentration), 6),
        "coherence": round(float(coherence), 6),
        "coherence_components": coherence_components,
        "attractor_state": rounded_attractor_state,
        "attractor_sigma_diag": rounded_attractor_sigma_diag,
        "velocity": round(float(velocity), 6),
        "acceleration": round(float(acceleration), 6),
        "uncertainty": {
            "mean_state_uncertainty": round(unc_mean, 6),
            "max_state_uncertainty": round(unc_max, 6),
            "circle_uncertainty": round(float(circle_unc), 6),
            "projection_kind": "canonical_state_space",
            "theta_defined": bool(theta_defined),
            "semantic_axis_scores": rounded_axis_scores,
            "semantic_concentration": round(float(semantic_concentration), 6),
            "coherence_components": coherence_components,
            "attractor_state": rounded_attractor_state,
            "attractor_sigma_diag": rounded_attractor_sigma_diag,
        },
    }


def infer_state_dimensions_from_tags(tags: Sequence[str]) -> Set[str]:
    dims: Set[str] = set()
    for tag in tags:
        mapped = _TAG_TO_DIMS.get(str(tag).lower())
        if mapped:
            dims.update(mapped)
    return dims


def _project_state_to_2d(x_hat: Dict[str, float]) -> List[float]:
    x = 0.0
    y = 0.0
    for dim in STATE_DIMENSIONS:
        val = float(x_hat.get(dim, 0.5))
        centered = val - 0.5
        wx, wy = _STATE_TO_CIRCLE_WEIGHTS[dim]
        x += centered * float(wx)
        y += centered * float(wy)
    return [float(x), float(y)]


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def _coerce_date(value: Any) -> Optional[date]:
    if isinstance(value, date):
        return value
    s = str(value or "")
    if len(s) < 10:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MiniFold Sub-Circles: per-decoherence-mode projections
# ---------------------------------------------------------------------------

# Map decoherence modes → state dimensions they govern
_MODE_DIMENSIONS: Dict[str, Tuple[str, ...]] = {
    "metabolic": ("energy_vitality", "glycemic_risk", "cardiovascular_load"),
    "autonomic": ("sleep_quality", "cognitive_control"),
    "immune": ("gut_gi",),
    "affective": ("mood_affect", "agency_purpose", "social_connectedness"),
}

# Per-mode 2D projection weights (unit vectors in the sub-space)
# Each mode gets its own local coordinate system for its MiniFold circle
_MODE_PROJECTION_WEIGHTS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "metabolic": {
        "energy_vitality": (0.85, 0.35),
        "glycemic_risk": (-0.70, 0.55),
        "cardiovascular_load": (-0.50, -0.75),
    },
    "autonomic": {
        "sleep_quality": (0.80, 0.45),
        "cognitive_control": (0.45, -0.80),
    },
    "immune": {
        "gut_gi": (1.0, 0.0),
    },
    "affective": {
        "mood_affect": (0.30, -0.85),
        "agency_purpose": (0.80, 0.40),
        "social_connectedness": (0.35, 0.70),
    },
}

MINIFOLD_MODES: Tuple[str, ...] = ("metabolic", "autonomic", "immune", "affective")
MINIFOLD_VERSION = "minifold_questions_v1"


def compute_minifold_circles(
    *,
    x_hat: Dict[str, float],
    x_uncertainty: Dict[str, float],
    previous_minifolds: Optional[Dict[str, Dict[str, Any]]] = None,
    day: Optional[date] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-decoherence-mode MiniFold sub-circles.

    Each mode gets its own local 2D Coherence Circle projection using
    only the state dimensions relevant to that mode.

    Returns a dict keyed by mode name (metabolic/autonomic/immune/affective),
    each containing: z, z_star, r, theta, uncertainty, velocity, acceleration.

    This is the Questions Agent's DigitalFold decomposed into 4 MiniFolds
    that align with Ivan's hierarchical architecture.
    """
    prev_all = previous_minifolds or {}
    result: Dict[str, Dict[str, Any]] = {}

    for mode in MINIFOLD_MODES:
        dims = _MODE_DIMENSIONS[mode]
        weights = _MODE_PROJECTION_WEIGHTS[mode]
        prev_mode = prev_all.get(mode, {})

        # Project mode-specific state dimensions to 2D
        z_mode = _project_mode_to_2d(x_hat, dims, weights)

        # EMA attractor estimation — initial z_star is population prior [0,0]
        # (not the observed z), matching Ivan's architecture: "Day 1: x* = population mean"
        prev_z_star = prev_mode.get("z_star")
        if isinstance(prev_z_star, list) and len(prev_z_star) == 2:
            z_star = [
                0.85 * float(prev_z_star[0]) + 0.15 * z_mode[0],
                0.85 * float(prev_z_star[1]) + 0.15 * z_mode[1],
            ]
        else:
            # First observation: z_star = population prior (center)
            z_star = [0.0, 0.0]

        r = _l2(z_mode, z_star)
        theta = float(math.atan2(z_mode[1], z_mode[0]))

        # Velocity / acceleration
        velocity = 0.0
        acceleration = 0.0
        prev_z = prev_mode.get("z")
        if isinstance(prev_z, list) and len(prev_z) == 2:
            prev_day = prev_mode.get("date")
            delta_days = 1.0
            if day is not None and isinstance(prev_day, str):
                from questions_agent_platform.pipeline.time_utils import parse_date
                pd = parse_date(prev_day)
                if pd is not None:
                    delta_days = max(1.0, float((day - pd).days))
            velocity = _l2(z_mode, prev_z) / delta_days
            prev_v = float(prev_mode.get("velocity", 0.0))
            acceleration = (velocity - prev_v) / delta_days

        # Uncertainty for this mode's dimensions
        mode_unc_vals = [float(x_uncertainty.get(d, 1.0)) for d in dims]
        mode_unc = float(sum(mode_unc_vals) / max(1, len(mode_unc_vals)))

        # Coverage: how many dimensions have data
        dims_with_data = sum(1 for d in dims if x_hat.get(d) is not None and abs(x_hat.get(d, 0.5) - 0.5) > 1e-6)
        coverage_ratio = dims_with_data / max(1, len(dims))

        # Coherence score for this mode (same formula as main circle)
        mode_coherence = max(0.0, min(1.0, 1.0 - float(r)))

        result[mode] = {
            "mode": mode,
            "z": [round(z_mode[0], 6), round(z_mode[1], 6)],
            "z_star": [round(z_star[0], 6), round(z_star[1], 6)],
            "r": round(r, 6),
            "theta": round(theta, 6),
            "velocity": round(velocity, 6),
            "acceleration": round(acceleration, 6),
            "coherence": round(mode_coherence, 6),
            "uncertainty": round(mode_unc, 6),
            "coverage_ratio": round(coverage_ratio, 4),
            "dimensions": list(dims),
            "date": day.isoformat() if day else None,
            "version": MINIFOLD_VERSION,
        }

    return result


def dominant_decoherence_mode(
    minifolds: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Return the mode with the largest radius (most decoherence).

    Used by the progressive measurement engine to decide which axis
    package to prioritize next.
    """
    if not minifolds:
        return None
    return max(minifolds, key=lambda m: float(minifolds[m].get("r", 0.0)))


def _project_mode_to_2d(
    x_hat: Dict[str, float],
    dims: Tuple[str, ...],
    weights: Dict[str, Tuple[float, float]],
) -> List[float]:
    """Project mode-specific state dimensions to local 2D coordinates."""
    x = 0.0
    y = 0.0
    for dim in dims:
        val = float(x_hat.get(dim, 0.5))
        centered = val - 0.5
        wx, wy = weights.get(dim, (0.0, 0.0))
        x += centered * float(wx)
        y += centered * float(wy)
    return [x, y]
