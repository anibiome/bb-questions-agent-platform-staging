"""QuestionsFold — the Questions Agent's MiniFold contribution to AniFold.

This module is the standalone computational core that converts questionnaire
evidence into a structured latent-state representation compatible with
AniFold's hierarchical fusion architecture.

Architecture Position
=====================

    ┌─────────────────────────────────────────────────────┐
    │                     AniFold                          │
    │                                                     │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
    │  │ PhenoFold│  │VoiceFold │  │ WearFold │   ...    │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
    │       │              │              │               │
    │       └──────┬───────┴──────┬───────┘               │
    │              │              │                        │
    │      Cross-Attention Fusion (Perceiver-like 1D)      │
    │              │                                       │
    │              ▼                                       │
    │     ┌─────────────────┐                             │
    │     │  DigitalFold    │                             │
    │     └────────┬────────┘                             │
    │              │                                       │
    │     ┌────────┴────────┐                             │
    │     │ QuestionsFold   │  ◄── THIS MODULE            │
    │     │ (MindFold)      │                             │
    │     └─────────────────┘                             │
    └─────────────────────────────────────────────────────┘

The QuestionsFold produces an **Identity Mask fragment** for
AniFold's Cross-Attention Fusion (Perceiver-like 1D) layer::

    m_questions(t) = {
        μ_t, Σ_t,                      # 9D latent posterior
        z_t, z*_t, z†_t,               # current position, feasible reference, Supercoherence
        r_t, s_t, q_t, θ_t,            # acute, structural, absolute distances + angle
        κ_local,t, κ_abs,t, σ_κ,t,     # local coherence, absolute coherence, uncertainty
        c_t,                           # coverage vector (per-dimension scale counts)
        h_t,                           # history envelope (velocity, acceleration, EWS)
        minifolds,                     # 4 per-mode sub-circles
    }

This fragment is self-contained: it can be used standalone for
questionnaire-only products, or fused with other modality encoders
via AniFold's Cross-Attention Fusion layer.

Usage::

    from questions_agent_platform.pipeline.questions_fold import (
        compute_questions_fold,
        QuestionsFoldResult,
    )

    fold = compute_questions_fold(
        x_hat=state["x_hat"],
        x_uncertainty=state["x_uncertainty"],
        coverage=state["coverage"],
        day=date.today(),
        previous_fold=yesterday_fold,  # optional
    )

    # Standalone API response
    payload = fold.to_identity_mask()

    # For AniFold cross-attention fusion
    mu, sigma = fold.mu, fold.sigma_diag
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

from questions_agent_platform.pipeline.state_snapshots import (
    CIRCLE_ANCHOR_VERSION,
    CIRCLE_PROJECTION_VERSION,
    MINIFOLD_MODES,
    MINIFOLD_VERSION,
    STATE_DIMENSIONS,
    STATE_MODEL_VERSION,
    compute_circle_snapshot,
    compute_minifold_circles,
    dominant_decoherence_mode,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUESTIONS_FOLD_VERSION = "questions_fold_v1"

# Coherence score is a nonlinear mapping from radius to [0, 1].
# radius_at_zero is the radius where coherence = 0. Beyond this,
# the system is fully decoherent. Calibrated from population data:
# 95th percentile healthy population radius ≈ 0.4, clinical ≈ 1.0.
_RADIUS_AT_ZERO = 1.25

# EWS (Early Warning System) thresholds for critical slowing down.
# These are tuned to balance sensitivity/specificity for predicting
# regime transitions in the Coherence Circle.
_EWS_VARIANCE_WARN = 0.04    # var(r) > 0.04 → elevated variability
_EWS_AUTOCORR_WARN = 0.6     # AC1(r) > 0.6 → critical slowing down
_EWS_TREND_WARN = 0.02       # |trend_speed| > 0.02/day → systematic drift


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MiniFoldCircle:
    """A single MiniFold sub-circle for one decoherence mode."""
    mode: str
    z: Tuple[float, float]
    z_star: Tuple[float, float]
    r: float
    theta: float
    velocity: float
    acceleration: float
    coherence: float
    uncertainty: float
    coverage_ratio: float
    dimensions: Tuple[str, ...]
    z_dagger: Optional[Tuple[float, float]] = None
    r_struct: Optional[float] = None
    r_abs: Optional[float] = None
    local_coherence: Optional[float] = None
    absolute_coherence: Optional[float] = None


@dataclass(frozen=True)
class HistoryEnvelope:
    """Temporal dynamics of the Coherence Circle trajectory.

    Captures velocity, acceleration, and early warning signals.
    These are the "how fast is the person changing?" features
    that distinguish acute from chronic decoherence.
    """
    velocity: float          # |dz/dt| per day
    acceleration: float      # d²z/dt² per day²
    ews_variance: float      # var(r) over window
    ews_autocorrelation: float  # lag-1 AC of r
    ews_trend_speed: float   # linear trend slope of r
    ews_score: float         # composite EWS risk [0, 1]
    recovery_rate: float     # how fast r returns to baseline after perturbation
    trajectory_days: int     # number of days with circle data


@dataclass(frozen=True)
class QuestionsFoldResult:
    """Complete QuestionsFold output — an Identity Mask fragment.

    This is the primary output of the Questions Agent's fold
    computation. It contains everything needed for:
    1. Standalone questionnaire product (display coherence circle)
    2. PoE fusion with other MiniFolds in AniFold
    3. Drift detection and anamnesis triggering
    4. Progressive measurement plan updates
    """
    # --- Latent state (μ, Σ) ---
    mu: Dict[str, float]                    # 9D state vector x_hat
    sigma_diag: Dict[str, float]            # 9D uncertainty diagonal

    # --- Rejuvenation Geometry (z, z*, z†, r, s, q, θ) ---
    z: Tuple[float, float]                  # 2D current position
    z_star: Tuple[float, float]             # 2D feasible reference (best currently reachable stable state)
    r: float                                # acute dysregulation radius
    theta: float                            # decoherence angle (radians)
    theta_defined: bool                     # semantic angle validity
    semantic_axis_scores: Dict[str, float]  # anchored mode magnitudes
    semantic_concentration: float           # how directional the drift is
    attractor_state: Dict[str, float]       # legacy alias for feasible_state
    attractor_sigma_diag: Dict[str, float]  # legacy alias for feasible_sigma_diag
    coherence: float                        # legacy alias for local_coherence
    coherence_uncertainty: float            # σ_κ from state uncertainty

    # --- Coverage ---
    coverage: Dict[str, int]                # per-dimension scale count
    total_scales_used: int
    coverage_ratio: float                   # fraction of dimensions with data

    # --- Temporal dynamics ---
    history: HistoryEnvelope

    # --- MiniFold sub-circles ---
    minifolds: Dict[str, MiniFoldCircle]    # 4 per-mode circles
    dominant_mode: Optional[str]            # mode with highest r

    # --- Metadata ---
    day: str                                # ISO date
    version: str = QUESTIONS_FOLD_VERSION
    circle_projection_version: str = CIRCLE_PROJECTION_VERSION
    minifold_version: str = MINIFOLD_VERSION
    z_dagger: Optional[Tuple[float, float]] = None
    r_struct: Optional[float] = None
    r_abs: Optional[float] = None
    structural_axis_scores: Dict[str, float] = field(default_factory=dict)
    structural_concentration: float = 0.0
    absolute_axis_scores: Dict[str, float] = field(default_factory=dict)
    absolute_concentration: float = 0.0
    feasible_state: Dict[str, float] = field(default_factory=dict)
    feasible_sigma_diag: Dict[str, float] = field(default_factory=dict)
    ideal_state: Dict[str, float] = field(default_factory=dict)
    ideal_sigma_diag: Dict[str, float] = field(default_factory=dict)
    supercoherence_state: Dict[str, float] = field(default_factory=dict)
    supercoherence_sigma_diag: Dict[str, float] = field(default_factory=dict)
    local_coherence: Optional[float] = None
    absolute_coherence: Optional[float] = None

    def to_identity_mask(self) -> Dict[str, Any]:
        """Serialise to Identity Mask fragment for API / PoE fusion.

        The output follows the Identity Mask schema from the
        AAAI 2026 position paper (Balen et al.):

            m_t = {μ_t, Σ_t; z_t, z*_t, z†_t; r_t, s_t, q_t, θ_t; κ_local,t, κ_abs,t, σ_κ,t; c_t; h_t}

        All numeric values are rounded to 6 decimal places for
        deterministic JSON serialisation.
        """
        z_dagger = self.z_dagger if self.z_dagger is not None else (0.0, 0.0)
        r_struct = float(self.r_struct if self.r_struct is not None else 0.0)
        r_abs = float(self.r_abs if self.r_abs is not None else self.r)
        local_coherence = float(self.local_coherence if self.local_coherence is not None else self.coherence)
        absolute_coherence = float(self.absolute_coherence if self.absolute_coherence is not None else self.coherence)
        feasible_state = self.feasible_state or self.attractor_state
        feasible_sigma_diag = self.feasible_sigma_diag or self.attractor_sigma_diag
        ideal_state = self.ideal_state or self.supercoherence_state
        ideal_sigma_diag = self.ideal_sigma_diag or self.supercoherence_sigma_diag
        supercoherence_state = self.supercoherence_state or ideal_state
        supercoherence_sigma_diag = self.supercoherence_sigma_diag or ideal_sigma_diag
        return {
            "schema": "identity_mask_fragment",
            "version": self.version,
            "modality": "questionnaire",
            "day": self.day,

            # Belief state: μ, Σ (diagonal)
            "mu": {k: round(v, 6) for k, v in self.mu.items()},
            "sigma_diag": {k: round(v, 6) for k, v in self.sigma_diag.items()},

            # Coherence Circle geometry
            "circle": {
                "z": [round(self.z[0], 6), round(self.z[1], 6)],
                "z_star": [round(self.z_star[0], 6), round(self.z_star[1], 6)],
                "z_dagger": [round(z_dagger[0], 6), round(z_dagger[1], 6)],
                "r": round(self.r, 6),
                "structural_distance": round(r_struct, 6),
                "absolute_distance": round(r_abs, 6),
                "theta": round(self.theta, 6),
                "velocity": round(self.history.velocity, 6),
                "acceleration": round(self.history.acceleration, 6),
                "theta_defined": self.theta_defined,
                "semantic_axis_scores": {
                    k: round(v, 6) for k, v in self.semantic_axis_scores.items()
                },
                "semantic_concentration": round(self.semantic_concentration, 6),
                "structural_axis_scores": {
                    k: round(v, 6) for k, v in self.structural_axis_scores.items()
                },
                "structural_concentration": round(self.structural_concentration, 6),
                "absolute_axis_scores": {
                    k: round(v, 6) for k, v in self.absolute_axis_scores.items()
                },
                "absolute_concentration": round(self.absolute_concentration, 6),
                "feasible_state": {
                    k: round(v, 6) for k, v in feasible_state.items()
                },
                "feasible_sigma_diag": {
                    k: round(v, 6) for k, v in feasible_sigma_diag.items()
                },
                "ideal_state": {
                    k: round(v, 6) for k, v in ideal_state.items()
                },
                "ideal_sigma_diag": {
                    k: round(v, 6) for k, v in ideal_sigma_diag.items()
                },
                "local_coherence": round(local_coherence, 6),
                "absolute_coherence": round(absolute_coherence, 6),
                "coherence_uncertainty": round(self.coherence_uncertainty, 6),
                "projection_version": self.circle_projection_version,
                "feasible_reference": {
                    "z": [round(self.z_star[0], 6), round(self.z_star[1], 6)],
                    "state": {
                        k: round(v, 6) for k, v in feasible_state.items()
                    },
                    "sigma_diag": {
                        k: round(v, 6) for k, v in feasible_sigma_diag.items()
                    },
                },
                "supercoherence": {
                    "z": [round(z_dagger[0], 6), round(z_dagger[1], 6)],
                    "state": {
                        k: round(v, 6) for k, v in supercoherence_state.items()
                    },
                    "sigma_diag": {
                        k: round(v, 6) for k, v in supercoherence_sigma_diag.items()
                    },
                },
                "rejuvenation_geometry": {
                    "acute_distance": round(self.r, 6),
                    "structural_distance": round(r_struct, 6),
                    "absolute_distance": round(r_abs, 6),
                    "local_coherence": round(local_coherence, 6),
                    "absolute_coherence": round(absolute_coherence, 6),
                    "feasible_reference": {
                        "z": [round(self.z_star[0], 6), round(self.z_star[1], 6)],
                        "state": {
                            k: round(v, 6) for k, v in feasible_state.items()
                        },
                        "sigma_diag": {
                            k: round(v, 6) for k, v in feasible_sigma_diag.items()
                        },
                    },
                    "supercoherence": {
                        "z": [round(z_dagger[0], 6), round(z_dagger[1], 6)],
                        "state": {
                            k: round(v, 6) for k, v in supercoherence_state.items()
                        },
                        "sigma_diag": {
                            k: round(v, 6) for k, v in supercoherence_sigma_diag.items()
                        },
                    },
                },
            },

            # Coverage quality
            "coverage": {
                "per_dimension": self.coverage,
                "total_scales": self.total_scales_used,
                "coverage_ratio": round(self.coverage_ratio, 4),
            },

            # Temporal dynamics
            "history": {
                "velocity": round(self.history.velocity, 6),
                "acceleration": round(self.history.acceleration, 6),
                "ews": {
                    "variance": round(self.history.ews_variance, 6),
                    "autocorrelation": round(self.history.ews_autocorrelation, 6),
                    "trend_speed": round(self.history.ews_trend_speed, 6),
                    "score": round(self.history.ews_score, 6),
                    "recovery_rate": round(self.history.recovery_rate, 6),
                },
                "trajectory_days": self.history.trajectory_days,
            },

            # 4 MiniFold sub-circles
            "minifolds": {
                mode: {
                    "z": [round(mf.z[0], 6), round(mf.z[1], 6)],
                    "z_star": [round(mf.z_star[0], 6), round(mf.z_star[1], 6)],
                    "z_dagger": [round((mf.z_dagger or (0.0, 0.0))[0], 6), round((mf.z_dagger or (0.0, 0.0))[1], 6)],
                    "r": round(mf.r, 6),
                    "structural_distance": round(float(mf.r_struct if mf.r_struct is not None else 0.0), 6),
                    "absolute_distance": round(float(mf.r_abs if mf.r_abs is not None else mf.r), 6),
                    "theta": round(mf.theta, 6),
                    "velocity": round(mf.velocity, 6),
                    "acceleration": round(mf.acceleration, 6),
                    "local_coherence": round(float(mf.local_coherence if mf.local_coherence is not None else mf.coherence), 6),
                    "absolute_coherence": round(float(mf.absolute_coherence if mf.absolute_coherence is not None else mf.coherence), 6),
                    "uncertainty": round(mf.uncertainty, 6),
                    "coverage_ratio": round(mf.coverage_ratio, 4),
                    "dimensions": list(mf.dimensions),
                }
                for mode, mf in self.minifolds.items()
            },
            "dominant_mode": self.dominant_mode,
        }

    def for_anifold_fusion(self) -> Dict[str, Any]:
        """Extract the minimal fusion-ready payload for AniFold integration.

        AniFold's Cross-Attention Fusion layer consumes per-modality
        token vectors.  We export:
          - μ (mean vector, 9D)
          - precision (1/σ² diagonal — encodes confidence per dimension)
          - coherence + coverage metadata

        This is consumed by the DigitalFold encoder (e_affect) which
        tokenises it for the Perceiver-like cross-attention block.
        """
        precision = {}
        for dim, sigma in self.sigma_diag.items():
            # Precision = 1/σ². Clamp sigma to avoid division by zero.
            s = max(0.001, sigma)
            precision[dim] = round(1.0 / (s * s), 6)

        local_coherence = float(self.local_coherence if self.local_coherence is not None else self.coherence)
        absolute_coherence = float(self.absolute_coherence if self.absolute_coherence is not None else self.coherence)
        r_struct = float(self.r_struct if self.r_struct is not None else 0.0)
        r_abs = float(self.r_abs if self.r_abs is not None else self.r)
        return {
            "modality": "questionnaire",
            "mu": {k: round(v, 6) for k, v in self.mu.items()},
            "precision": precision,
            "local_coherence": round(local_coherence, 6),
            "absolute_coherence": round(absolute_coherence, 6),
            "structural_distance": round(r_struct, 6),
            "absolute_distance": round(r_abs, 6),
            "coverage_ratio": round(self.coverage_ratio, 4),
            "day": self.day,
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_questions_fold(
    *,
    x_hat: Dict[str, float],
    x_uncertainty: Dict[str, float],
    coverage: Optional[Dict[str, Dict[str, Any]]] = None,
    day: Optional[date] = None,
    previous_fold: Optional[Dict[str, Any]] = None,
    circle_history: Optional[Sequence[Dict[str, Any]]] = None,
) -> QuestionsFoldResult:
    """Compute the complete QuestionsFold from state vectors.

    This is the single entry point for the fold computation. It takes
    the 9D state vector (from ``compute_state_snapshot``) and produces
    the full Identity Mask fragment.

    Args:
        x_hat: 9D state vector, each dimension in [0, 1].
        x_uncertainty: 9D uncertainty per dimension.
        coverage: Per-dimension coverage info (from state snapshot).
            If None, coverage is inferred from x_hat (non-0.5 = has data).
        day: Date for this computation. Defaults to today.
        previous_fold: Previous fold output (for EMA attractor and velocity).
            Expected keys: ``circle`` (with z, z_star, velocity, date)
            and ``minifolds`` (with per-mode z, z_star, velocity, date).
        circle_history: List of previous circle snapshots for EWS
            computation. Each should have at least ``r`` and ``date``.

    Returns:
        QuestionsFoldResult with full Identity Mask fragment.
    """
    today = day or date.today()

    # 1. Extract previous circle for EMA attractor
    prev_circle = _extract_previous_circle(previous_fold)
    prev_minifolds = _extract_previous_minifolds(previous_fold)

    # 2. Compute main Coherence Circle
    circle = compute_circle_snapshot(
        day=today,
        x_hat=x_hat,
        x_uncertainty=x_uncertainty,
        previous_circle=prev_circle,
    )

    # 3. Compute 4 MiniFold sub-circles
    minifold_raw = compute_minifold_circles(
        x_hat=x_hat,
        x_uncertainty=x_uncertainty,
        previous_minifolds=prev_minifolds,
        day=today,
    )

    # 4. Coherence score: prefer canonical circle score when available
    coherence = float(circle.get("local_coherence", circle.get("coherence", _coherence_from_radius(circle["r"]))))
    local_coherence = float(circle.get("local_coherence", coherence))
    absolute_coherence = float(circle.get("absolute_coherence", local_coherence))

    # 5. Coherence uncertainty: propagated from state uncertainty
    coherence_unc = _coherence_uncertainty(x_uncertainty, circle)

    # 6. Coverage analysis
    cov_info = _compute_coverage(x_hat, x_uncertainty, coverage)

    # 7. History envelope (EWS features)
    history = _compute_history_envelope(circle, circle_history)

    # 8. Build MiniFoldCircle objects
    minifolds = {}
    for mode in MINIFOLD_MODES:
        mf = minifold_raw[mode]
        z_dagger = mf.get("z_dagger")
        local_mode_coherence = float(mf.get("local_coherence", mf.get("coherence", _coherence_from_radius(mf["r"]))))
        minifolds[mode] = MiniFoldCircle(
            mode=mode,
            z=(mf["z"][0], mf["z"][1]),
            z_star=(mf["z_star"][0], mf["z_star"][1]),
            r=mf["r"],
            theta=mf["theta"],
            velocity=mf["velocity"],
            acceleration=mf["acceleration"],
            coherence=local_mode_coherence,
            uncertainty=mf["uncertainty"],
            coverage_ratio=mf["coverage_ratio"],
            dimensions=tuple(mf["dimensions"]),
            z_dagger=(z_dagger[0], z_dagger[1]) if isinstance(z_dagger, list) and len(z_dagger) == 2 else None,
            r_struct=float(mf.get("structural_distance", mf.get("r_struct", 0.0))),
            r_abs=float(mf.get("absolute_distance", mf.get("r_abs", mf["r"]))),
            local_coherence=local_mode_coherence,
            absolute_coherence=float(mf.get("absolute_coherence", mf.get("coherence", local_mode_coherence))),
        )

    # 9. Dominant decoherence mode
    dominant = dominant_decoherence_mode(minifold_raw)

    return QuestionsFoldResult(
        mu=dict(x_hat),
        sigma_diag=dict(x_uncertainty),
        z=(circle["z"][0], circle["z"][1]),
        z_star=(circle["z_star"][0], circle["z_star"][1]),
        r=circle["r"],
        theta=circle["theta"],
        theta_defined=bool(circle.get("theta_defined", True)),
        semantic_axis_scores=dict(circle.get("semantic_axis_scores", {})),
        semantic_concentration=float(circle.get("semantic_concentration", 0.0)),
        attractor_state=dict(circle.get("feasible_state", circle.get("attractor_state", {}))),
        attractor_sigma_diag=dict(circle.get("feasible_sigma_diag", circle.get("attractor_sigma_diag", {}))),
        coherence=coherence,
        coherence_uncertainty=coherence_unc,
        coverage=cov_info["per_dimension"],
        total_scales_used=cov_info["total_scales"],
        coverage_ratio=cov_info["ratio"],
        history=history,
        minifolds=minifolds,
        dominant_mode=dominant,
        day=today.isoformat(),
        z_dagger=(circle["z_dagger"][0], circle["z_dagger"][1]) if isinstance(circle.get("z_dagger"), list) and len(circle.get("z_dagger", [])) == 2 else None,
        r_struct=float(circle.get("structural_distance", circle.get("r_struct", 0.0))),
        r_abs=float(circle.get("absolute_distance", circle.get("r_abs", circle["r"]))),
        structural_axis_scores=dict(circle.get("structural_axis_scores", {})),
        structural_concentration=float(circle.get("structural_concentration", 0.0)),
        absolute_axis_scores=dict(circle.get("absolute_axis_scores", {})),
        absolute_concentration=float(circle.get("absolute_concentration", 0.0)),
        feasible_state=dict(circle.get("feasible_state", circle.get("attractor_state", {}))),
        feasible_sigma_diag=dict(circle.get("feasible_sigma_diag", circle.get("attractor_sigma_diag", {}))),
        ideal_state=dict(circle.get("ideal_state", circle.get("supercoherence_state", {}))),
        ideal_sigma_diag=dict(circle.get("ideal_sigma_diag", circle.get("supercoherence_sigma_diag", {}))),
        supercoherence_state=dict(circle.get("supercoherence_state", circle.get("ideal_state", (circle.get("supercoherence") or {}).get("state", {})))),
        supercoherence_sigma_diag=dict(circle.get("supercoherence_sigma_diag", circle.get("ideal_sigma_diag", (circle.get("supercoherence") or {}).get("sigma_diag", {})))),
        local_coherence=local_coherence,
        absolute_coherence=absolute_coherence,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coherence_from_radius(r: float) -> float:
    """Legacy radius-only fallback used when richer circle metadata is absent.

    Uses a soft-clipped linear mapping:
        κ̂ = max(0, 1 - r / R₀)

    where R₀ = 1.25 is the radius at which coherence hits zero.
    This is calibrated so that:
    - r = 0.0  → κ̂ = 1.0  (perfect coherence)
    - r = 0.4  → κ̂ = 0.68 (healthy range)
    - r = 1.0  → κ̂ = 0.20 (clinical concern)
    - r ≥ 1.25 → κ̂ = 0.0  (fully decoherent)
    """
    return max(0.0, min(1.0, 1.0 - float(r) / _RADIUS_AT_ZERO))


def _coherence_uncertainty(
    x_uncertainty: Dict[str, float],
    circle: Dict[str, Any],
) -> float:
    """Estimate uncertainty in the coherence score.

    Combines two sources:
    1. State uncertainty: high Σ means we're unsure about the state
    2. Circle uncertainty: propagated through the projection

    Returns σ_κ ∈ [0, 1].
    """
    # State contribution: mean uncertainty across dimensions
    unc_vals = [float(x_uncertainty.get(d, 1.0)) for d in STATE_DIMENSIONS]
    mean_unc = sum(unc_vals) / max(1, len(unc_vals))

    # Circle contribution (already computed in circle snapshot)
    circle_unc_obj = circle.get("uncertainty", {})
    if isinstance(circle_unc_obj, dict):
        circle_unc = float(
            circle_unc_obj.get(
                "circle_uncertainty",
                circle_unc_obj.get("coherence_uncertainty", mean_unc),
            )
        )
    else:
        circle_unc = float(circle_unc_obj)

    # Combine: weighted average (state uncertainty is the primary driver)
    sigma_kappa = 0.6 * mean_unc + 0.4 * circle_unc
    return max(0.0, min(1.0, sigma_kappa))


def _compute_coverage(
    x_hat: Dict[str, float],
    x_uncertainty: Dict[str, float],
    coverage: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Analyse measurement coverage across state dimensions."""
    per_dim: Dict[str, int] = {}
    total_scales = 0

    if coverage:
        for dim in STATE_DIMENSIONS:
            cov = coverage.get(dim, {})
            count = int(cov.get("scale_count", 0))
            per_dim[dim] = count
            total_scales += count
    else:
        # Infer from uncertainty: unc < 1.0 implies some data
        for dim in STATE_DIMENSIONS:
            has_data = float(x_uncertainty.get(dim, 1.0)) < 1.0
            per_dim[dim] = 1 if has_data else 0
            total_scales += per_dim[dim]

    dims_with_data = sum(1 for c in per_dim.values() if c > 0)
    ratio = dims_with_data / max(1, len(STATE_DIMENSIONS))

    return {
        "per_dimension": per_dim,
        "total_scales": total_scales,
        "ratio": ratio,
    }


def _compute_history_envelope(
    circle: Dict[str, Any],
    circle_history: Optional[Sequence[Dict[str, Any]]],
) -> HistoryEnvelope:
    """Compute temporal dynamics from circle trajectory history.

    If no history is provided, returns a zero-state envelope
    (appropriate for first observation).
    """
    velocity = float(circle.get("velocity", 0.0))
    acceleration = float(circle.get("acceleration", 0.0))

    if not circle_history or len(circle_history) < 2:
        return HistoryEnvelope(
            velocity=velocity,
            acceleration=acceleration,
            ews_variance=0.0,
            ews_autocorrelation=0.0,
            ews_trend_speed=0.0,
            ews_score=0.0,
            recovery_rate=0.0,
            trajectory_days=len(circle_history) if circle_history else 0,
        )

    # Extract radius timeseries
    r_vals = [float(c.get("r", 0.0)) for c in circle_history]
    n = len(r_vals)

    # Variance of r (indicator of critical fluctuations)
    mean_r = sum(r_vals) / n
    var_r = sum((rv - mean_r) ** 2 for rv in r_vals) / max(1, n - 1)

    # Lag-1 autocorrelation (critical slowing down signature)
    ac1 = _lag1_autocorrelation(r_vals)

    # Linear trend (systematic drift)
    trend = _linear_trend_slope(r_vals)

    # Recovery rate: how fast does r return after peak?
    recovery = _recovery_rate(r_vals)

    # Composite EWS score: normalised combination
    ews_score = _composite_ews_score(var_r, ac1, abs(trend))

    return HistoryEnvelope(
        velocity=velocity,
        acceleration=acceleration,
        ews_variance=var_r,
        ews_autocorrelation=ac1,
        ews_trend_speed=trend,
        ews_score=ews_score,
        recovery_rate=recovery,
        trajectory_days=n,
    )


def _lag1_autocorrelation(values: Sequence[float]) -> float:
    """Compute lag-1 autocorrelation coefficient."""
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    if var < 1e-12:
        return 0.0
    cov = sum((values[i] - mean) * (values[i + 1] - mean) for i in range(n - 1)) / (n - 1)
    return max(-1.0, min(1.0, cov / var))


def _linear_trend_slope(values: Sequence[float]) -> float:
    """Compute simple linear regression slope (OLS)."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if abs(den) < 1e-12:
        return 0.0
    return num / den


def _recovery_rate(r_vals: Sequence[float]) -> float:
    """Estimate how fast the system recovers from peak decoherence.

    Recovery rate = (peak_r - final_r) / days_since_peak.
    Higher is better (faster return to baseline).
    """
    if len(r_vals) < 3:
        return 0.0
    peak_idx = max(range(len(r_vals)), key=lambda i: r_vals[i])
    if peak_idx >= len(r_vals) - 1:
        return 0.0  # peak is at the end — no recovery yet
    peak_val = r_vals[peak_idx]
    final_val = r_vals[-1]
    days_since_peak = len(r_vals) - 1 - peak_idx
    if days_since_peak < 1:
        return 0.0
    return max(0.0, (peak_val - final_val) / days_since_peak)


def _composite_ews_score(var_r: float, ac1: float, trend_abs: float) -> float:
    """Combine EWS indicators into a single risk score ∈ [0, 1].

    Each indicator contributes proportionally to its warning threshold:
    - Variance: high variance = larger fluctuations = more risk
    - Autocorrelation: high AC1 = critical slowing down
    - Trend: systematic drift away from attractor

    Weights: variance 0.35, autocorrelation 0.40, trend 0.25
    (AC1 is the strongest predictor of regime transitions in
    dynamical systems, hence highest weight)
    """
    v_score = min(1.0, var_r / _EWS_VARIANCE_WARN) if _EWS_VARIANCE_WARN > 0 else 0.0
    a_score = min(1.0, max(0.0, ac1) / _EWS_AUTOCORR_WARN) if _EWS_AUTOCORR_WARN > 0 else 0.0
    t_score = min(1.0, trend_abs / _EWS_TREND_WARN) if _EWS_TREND_WARN > 0 else 0.0

    return max(0.0, min(1.0, 0.35 * v_score + 0.40 * a_score + 0.25 * t_score))


def _extract_previous_circle(
    previous_fold: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Extract previous circle data from a fold output or raw dict."""
    if previous_fold is None:
        return None
    # Support both QuestionsFoldResult.to_identity_mask() output
    # and raw circle snapshot dicts
    circle = previous_fold.get("circle")
    if circle and isinstance(circle, dict):
        geometry = circle.get("rejuvenation_geometry") if isinstance(circle.get("rejuvenation_geometry"), dict) else {}
        feasible_reference = circle.get("feasible_reference") if isinstance(circle.get("feasible_reference"), dict) else {}
        if not feasible_reference and isinstance(geometry.get("feasible_reference"), dict):
            feasible_reference = dict(geometry.get("feasible_reference") or {})
        supercoherence = circle.get("supercoherence") if isinstance(circle.get("supercoherence"), dict) else {}
        if not supercoherence and isinstance(geometry.get("supercoherence"), dict):
            supercoherence = dict(geometry.get("supercoherence") or {})
        return {
            "z": circle.get("z"),
            "z_star": circle.get("z_star"),
            "z_dagger": circle.get("z_dagger"),
            "r": float(circle.get("r", 0.0)),
            "r_struct": float(circle.get("structural_distance", circle.get("r_struct", geometry.get("structural_distance", 0.0)))),
            "r_abs": float(circle.get("absolute_distance", circle.get("r_abs", geometry.get("absolute_distance", circle.get("r", 0.0))))),
            "velocity": float(circle.get("velocity", 0.0)),
            "acceleration": float(circle.get("acceleration", 0.0)),
            "date": previous_fold.get("day"),
            "attractor_state": circle.get("attractor_state") or circle.get("feasible_state") or feasible_reference.get("state"),
            "attractor_sigma_diag": circle.get("attractor_sigma_diag") or circle.get("feasible_sigma_diag") or feasible_reference.get("sigma_diag"),
            "feasible_state": circle.get("feasible_state") or feasible_reference.get("state"),
            "feasible_sigma_diag": circle.get("feasible_sigma_diag") or feasible_reference.get("sigma_diag"),
            "ideal_state": circle.get("ideal_state") or supercoherence.get("state"),
            "ideal_sigma_diag": circle.get("ideal_sigma_diag") or supercoherence.get("sigma_diag"),
            "supercoherence_state": circle.get("supercoherence_state") or circle.get("ideal_state") or supercoherence.get("state"),
            "supercoherence_sigma_diag": circle.get("supercoherence_sigma_diag") or circle.get("ideal_sigma_diag") or supercoherence.get("sigma_diag"),
        }
    # Raw circle snapshot format
    if "z" in previous_fold and "z_star" in previous_fold:
        return previous_fold
    return None


def _extract_previous_minifolds(
    previous_fold: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Extract previous MiniFold data from a fold output."""
    if previous_fold is None:
        return None
    mfs = previous_fold.get("minifolds")
    if mfs and isinstance(mfs, dict):
        result = {}
        for mode, data in mfs.items():
            if isinstance(data, dict) and "z" in data:
                result[mode] = {
                    "z": data["z"],
                    "z_star": data["z_star"],
                    "z_dagger": data.get("z_dagger"),
                    "r": float(data.get("r", 0.0)),
                    "r_struct": float(data.get("structural_distance", data.get("r_struct", 0.0))),
                    "r_abs": float(data.get("absolute_distance", data.get("r_abs", data.get("r", 0.0)))),
                    "theta": float(data.get("theta", 0.0)),
                    "velocity": float(data.get("velocity", 0.0)),
                    "acceleration": float(data.get("acceleration", 0.0)),
                    "date": previous_fold.get("day"),
                }
        return result if result else None
    return None
