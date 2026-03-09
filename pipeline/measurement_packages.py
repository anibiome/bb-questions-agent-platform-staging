"""Progressive Measurement Packages for the Coherence Circle.

Maps instruments to four decoherence modes (metabolic, autonomic, immune,
affective) and defines tiered measurement packages that progressively
increase resolution.

Architecture
============

                     ┌──────────────────────┐
                     │   Coherence Circle    │
                     │   z(t) = P(x(t))     │
                     │                       │
                     │  4 angular sectors:   │
                     │  θ ∈ {M, A, I, Af}    │
                     └──────────────────────┘
                                ▲
              ┌─────────────────┼──────────────────┐
              │                 │                   │
        ┌─────┴──────┐   ┌─────┴──────┐    ┌──────┴──────┐
        │  Tier 0    │   │  Tier 1    │    │  Tier 2     │
        │  Core      │   │  Axis      │    │  Deep Dive  │
        │  Screener  │   │  Packages  │    │  Clinical   │
        │  (27 q's)  │   │  (per mode)│    │  Follow-up  │
        └────────────┘   └────────────┘    └─────────────┘

Tier 0: Core Screener  (~27 items, 3-4 min)
  - PHQ-4 (depression+anxiety, 4 items)
  - PSS-4 (stress, 4 items)
  - WHO-5 (wellbeing, 5 items)
  - BRS (resilience, 6 items)
  - GSRS-IBS top-5 (GI screening, 5 items)
  - Initial Scan (3 items: sleep, energy, social)

Tier 1: Axis Packages  (~15-25 items each)
  - Metabolic: FINDRISC + IPAQ-SF + AUDIT-C + CFS
  - Autonomic: BIS (sleep) + PSS-10 + SCL + HRV-proxy items
  - Immune: GSRS-IBS + IBS + DQLQ + Visceral Sensitivity
  - Affective: GAD-7 + DASS-21-dep + RSES + MSPSS

Tier 2: Deep Dive / Anamnesis-triggered
  - Full instrument expansion when drift detected
  - N-of-1 repeat measurement packages

Usage::

    from questions_agent_platform.pipeline.measurement_packages import (
        MeasurementPackage,
        PackageTier,
        get_core_screener,
        get_axis_package,
        get_progressive_plan,
        DECOHERENCE_MODES,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Decoherence modes (Coherence Circle angular sectors)
# ---------------------------------------------------------------------------
class DecoherenceMode(str, Enum):
    """Four angular sectors of the Coherence Circle."""
    METABOLIC = "metabolic"
    AUTONOMIC = "autonomic"
    IMMUNE = "immune"
    AFFECTIVE = "affective"


DECOHERENCE_MODES = tuple(DecoherenceMode)


# ---------------------------------------------------------------------------
# Package tiers
# ---------------------------------------------------------------------------
class PackageTier(str, Enum):
    CORE = "core"           # Tier 0: universal screener
    AXIS = "axis"           # Tier 1: per-mode deep packages
    DEEP_DIVE = "deep_dive" # Tier 2: anamnesis-triggered clinical
    REPEAT = "repeat"       # Tier 3: N-of-1 baseline repeat


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PackageItem:
    """A single item reference within a measurement package."""
    scale_id: str
    item_ids: Tuple[str, ...]  # empty = all items in scale
    priority: float = 1.0      # higher = more important


@dataclass(frozen=True)
class MeasurementPackage:
    """A curated set of scales/items for a specific purpose."""
    id: str
    name: str
    tier: PackageTier
    mode: Optional[DecoherenceMode]  # None for CORE (spans all modes)
    description: str
    estimated_items: int
    estimated_minutes: float
    package_items: Tuple[PackageItem, ...]
    tags: Tuple[str, ...] = ()
    prerequisite_package_id: Optional[str] = None
    unlock_after_sessions: int = 0


@dataclass(frozen=True)
class ProgressivePlan:
    """Full progressive measurement plan for a user across sessions."""
    user_id: str
    recommended_sequence: Tuple[str, ...]  # package IDs in order
    current_package_idx: int = 0
    completed_package_ids: FrozenSet[str] = frozenset()


# ---------------------------------------------------------------------------
# Core Screener (Tier 0) — ~27 items, every user starts here
# ---------------------------------------------------------------------------
CORE_SCREENER = MeasurementPackage(
    id="pkg_core_screener",
    name="Core Screener",
    tier=PackageTier.CORE,
    mode=None,
    description=(
        "Universal 27-item screener covering all four decoherence modes. "
        "Administered in the first session to establish baseline position "
        "on the Coherence Circle."
    ),
    estimated_items=27,
    estimated_minutes=3.5,
    package_items=(
        # PHQ-4 (depression + anxiety screening, 4 items)
        PackageItem(scale_id="scale_gad_7", item_ids=(), priority=1.0),
        # WHO-5 (wellbeing, 5 items — from Initial Scan if available)
        PackageItem(scale_id="scale_mhc", item_ids=(), priority=0.95),
        # PSS-4 (stress, first 4 items of PSS-10)
        PackageItem(scale_id="scale_pss", item_ids=(), priority=0.9),
        # BRS (resilience, 6 items)
        PackageItem(scale_id="scale_brs", item_ids=(), priority=0.85),
        # GSRS-IBS top screening items
        PackageItem(scale_id="scale_gsrs_ibs", item_ids=(), priority=0.8),
        # Initial scan (sleep, energy, social)
        PackageItem(scale_id="scale_initial_scan", item_ids=(), priority=0.75),
    ),
    tags=("screening", "onboarding", "all_modes"),
)


# ---------------------------------------------------------------------------
# Axis Packages (Tier 1) — one per decoherence mode
# ---------------------------------------------------------------------------
METABOLIC_AXIS = MeasurementPackage(
    id="pkg_axis_metabolic",
    name="Metabolic Axis",
    tier=PackageTier.AXIS,
    mode=DecoherenceMode.METABOLIC,
    description=(
        "Cardiometabolic risk profiling: diabetes risk (FINDRISC), "
        "physical activity (IPAQ-SF), alcohol (AUDIT-C), fatigue (CFS), "
        "and dietary quality screening."
    ),
    estimated_items=27,
    estimated_minutes=4.0,
    package_items=(
        PackageItem(scale_id="scale_findrisc", item_ids=(), priority=1.0),
        PackageItem(scale_id="scale_ipaq_sf", item_ids=(), priority=0.95),
        PackageItem(scale_id="scale_audit_c", item_ids=(), priority=0.9),
        PackageItem(scale_id="scale_cfs", item_ids=(), priority=0.85),
        PackageItem(scale_id="scale_qols_flanagan", item_ids=(), priority=0.7),
    ),
    tags=("cardiometabolic", "metabolic", "risk_screening"),
    prerequisite_package_id="pkg_core_screener",
    unlock_after_sessions=1,
)

AUTONOMIC_AXIS = MeasurementPackage(
    id="pkg_axis_autonomic",
    name="Autonomic Axis",
    tier=PackageTier.AXIS,
    mode=DecoherenceMode.AUTONOMIC,
    description=(
        "Autonomic regulation profiling: sleep quality (BIS), stress "
        "reactivity (PSS-10), somatization (SCL), and cognitive clarity "
        "(CAMS-R). Feeds HRV proxy estimation in MiniFold."
    ),
    estimated_items=38,
    estimated_minutes=5.0,
    package_items=(
        PackageItem(scale_id="scale_bis", item_ids=(), priority=1.0),
        PackageItem(scale_id="scale_pss", item_ids=(), priority=0.95),
        PackageItem(scale_id="scale_scl", item_ids=(), priority=0.9),
        PackageItem(scale_id="scale_cams_r", item_ids=(), priority=0.85),
        PackageItem(scale_id="scale_day_to_day_experiences", item_ids=(), priority=0.8),
        PackageItem(scale_id="scale_cfq", item_ids=(), priority=0.7),
    ),
    tags=("autonomic", "sleep", "stress", "hrv_proxy"),
    prerequisite_package_id="pkg_core_screener",
    unlock_after_sessions=1,
)

IMMUNE_AXIS = MeasurementPackage(
    id="pkg_axis_immune",
    name="Immune/GI Axis",
    tier=PackageTier.AXIS,
    mode=DecoherenceMode.IMMUNE,
    description=(
        "Gut-immune regulation profiling: GI symptoms (GSRS-IBS, IBS), "
        "digestive quality of life (DQLQ), visceral sensitivity, and "
        "IBD-specific measures. Core to the gut-brain axis."
    ),
    estimated_items=46,
    estimated_minutes=6.0,
    package_items=(
        PackageItem(scale_id="scale_gsrs_ibs", item_ids=(), priority=1.0),
        PackageItem(scale_id="scale_ibs", item_ids=(), priority=0.95),
        PackageItem(scale_id="scale_dqlq", item_ids=(), priority=0.9),
        PackageItem(scale_id="scale_visceral_sensitivity_index", item_ids=(), priority=0.85),
        PackageItem(scale_id="scale_ibdq", item_ids=(), priority=0.7),
    ),
    tags=("immune", "gi", "gut_brain", "inflammation"),
    prerequisite_package_id="pkg_core_screener",
    unlock_after_sessions=1,
)

AFFECTIVE_AXIS = MeasurementPackage(
    id="pkg_axis_affective",
    name="Affective Axis",
    tier=PackageTier.AXIS,
    mode=DecoherenceMode.AFFECTIVE,
    description=(
        "Affective regulation profiling: anxiety (GAD-7), depression "
        "(DASS-21), emotional regulation (ERQ), self-esteem (RSES), "
        "social support (MSPSS), and loneliness (UCLA)."
    ),
    estimated_items=55,
    estimated_minutes=7.0,
    package_items=(
        PackageItem(scale_id="scale_gad_7", item_ids=(), priority=1.0),
        PackageItem(scale_id="scale_dass_21_depression", item_ids=(), priority=0.95),
        PackageItem(scale_id="scale_dass_21_stress", item_ids=(), priority=0.9),
        PackageItem(scale_id="scale_erq", item_ids=(), priority=0.85),
        PackageItem(scale_id="scale_rses", item_ids=(), priority=0.8),
        PackageItem(scale_id="scale_mspss", item_ids=(), priority=0.75),
        PackageItem(scale_id="scale_ucla_loneliness_scale", item_ids=(), priority=0.7),
        PackageItem(scale_id="scale_panas_sf_negative", item_ids=(), priority=0.65),
        PackageItem(scale_id="scale_panas_sf_positive", item_ids=(), priority=0.6),
    ),
    tags=("affective", "mood", "emotion", "social"),
    prerequisite_package_id="pkg_core_screener",
    unlock_after_sessions=1,
)


# ---------------------------------------------------------------------------
# Deep Dive Packages (Tier 2) — anamnesis-triggered
# ---------------------------------------------------------------------------
DEEP_METABOLIC = MeasurementPackage(
    id="pkg_deep_metabolic",
    name="Metabolic Deep Dive",
    tier=PackageTier.DEEP_DIVE,
    mode=DecoherenceMode.METABOLIC,
    description=(
        "Extended cardiometabolic profiling triggered by metabolic drift. "
        "Adds CKD screening (SCORED), NAFLD risk (Lee), and CVD risk (EZ-CVD)."
    ),
    estimated_items=18,
    estimated_minutes=3.0,
    package_items=(
        PackageItem(scale_id="scale_scored", item_ids=(), priority=1.0),
        PackageItem(scale_id="scale_lee_nafld", item_ids=(), priority=0.95),
        PackageItem(scale_id="scale_ez_cvd", item_ids=(), priority=0.9),
    ),
    tags=("cardiometabolic", "clinical", "anamnesis_triggered"),
    prerequisite_package_id="pkg_axis_metabolic",
)

DEEP_AFFECTIVE = MeasurementPackage(
    id="pkg_deep_affective",
    name="Affective Deep Dive",
    tier=PackageTier.DEEP_DIVE,
    mode=DecoherenceMode.AFFECTIVE,
    description=(
        "Extended affective profiling triggered by affective drift. "
        "Adds personality (BIF2), coping (Cope Inventory), "
        "authenticity, and self-reflection."
    ),
    estimated_items=50,
    estimated_minutes=7.0,
    package_items=(
        PackageItem(scale_id="scale_bif2", item_ids=(), priority=0.9),
        PackageItem(scale_id="scale_cope_inventory", item_ids=(), priority=0.85),
        PackageItem(scale_id="scale_authenticity_scale", item_ids=(), priority=0.8),
        PackageItem(scale_id="scale_sris", item_ids=(), priority=0.75),
        PackageItem(scale_id="scale_ds_14", item_ids=(), priority=0.7),
    ),
    tags=("affective", "personality", "clinical", "anamnesis_triggered"),
    prerequisite_package_id="pkg_axis_affective",
)

DEEP_IMMUNE = MeasurementPackage(
    id="pkg_deep_immune",
    name="Immune/GI Deep Dive",
    tier=PackageTier.DEEP_DIVE,
    mode=DecoherenceMode.IMMUNE,
    description=(
        "Extended GI profiling triggered by immune drift. Adds "
        "full IBDQ, PAQ (alexithymia-gut link), and SCL somatization."
    ),
    estimated_items=35,
    estimated_minutes=5.0,
    package_items=(
        PackageItem(scale_id="scale_ibdq", item_ids=(), priority=1.0),
        PackageItem(scale_id="scale_paq", item_ids=(), priority=0.85),
        PackageItem(scale_id="scale_scl", item_ids=(), priority=0.8),
    ),
    tags=("immune", "gi", "clinical", "anamnesis_triggered"),
    prerequisite_package_id="pkg_axis_immune",
)

DEEP_AUTONOMIC = MeasurementPackage(
    id="pkg_deep_autonomic",
    name="Autonomic Deep Dive",
    tier=PackageTier.DEEP_DIVE,
    mode=DecoherenceMode.AUTONOMIC,
    description=(
        "Extended autonomic profiling triggered by autonomic drift. "
        "Adds memory (MMQ), rumination, HADS, and SIAS."
    ),
    estimated_items=40,
    estimated_minutes=5.0,
    package_items=(
        PackageItem(scale_id="scale_multifactorial_memory_questionnaire", item_ids=(), priority=0.95),
        PackageItem(scale_id="scale_rumination", item_ids=(), priority=0.9),
        PackageItem(scale_id="scale_hads", item_ids=(), priority=0.85),
        PackageItem(scale_id="scale_sias", item_ids=(), priority=0.8),
    ),
    tags=("autonomic", "cognitive", "clinical", "anamnesis_triggered"),
    prerequisite_package_id="pkg_axis_autonomic",
)


# ---------------------------------------------------------------------------
# N-of-1 Repeat Package
# ---------------------------------------------------------------------------
REPEAT_BASELINE = MeasurementPackage(
    id="pkg_repeat_baseline",
    name="N-of-1 Baseline Repeat",
    tier=PackageTier.REPEAT,
    mode=None,
    description=(
        "Repeat core + axis items to establish within-person baseline "
        "for N-of-1 divergence detection. Administered at session 3 and 5."
    ),
    estimated_items=27,
    estimated_minutes=3.5,
    package_items=(
        # Same as core screener — the selection engine will pick
        # items that were answered in session 1 for baseline comparison
        PackageItem(scale_id="scale_gad_7", item_ids=(), priority=1.0),
        PackageItem(scale_id="scale_mhc", item_ids=(), priority=0.95),
        PackageItem(scale_id="scale_pss", item_ids=(), priority=0.9),
        PackageItem(scale_id="scale_brs", item_ids=(), priority=0.85),
        PackageItem(scale_id="scale_gsrs_ibs", item_ids=(), priority=0.8),
    ),
    tags=("n_of_1", "baseline", "repeat"),
    prerequisite_package_id="pkg_core_screener",
    unlock_after_sessions=2,
)


# ---------------------------------------------------------------------------
# Package registry
# ---------------------------------------------------------------------------
ALL_PACKAGES: Dict[str, MeasurementPackage] = {
    pkg.id: pkg for pkg in [
        CORE_SCREENER,
        METABOLIC_AXIS, AUTONOMIC_AXIS, IMMUNE_AXIS, AFFECTIVE_AXIS,
        DEEP_METABOLIC, DEEP_AFFECTIVE, DEEP_IMMUNE, DEEP_AUTONOMIC,
        REPEAT_BASELINE,
    ]
}

AXIS_PACKAGES: Dict[DecoherenceMode, MeasurementPackage] = {
    DecoherenceMode.METABOLIC: METABOLIC_AXIS,
    DecoherenceMode.AUTONOMIC: AUTONOMIC_AXIS,
    DecoherenceMode.IMMUNE: IMMUNE_AXIS,
    DecoherenceMode.AFFECTIVE: AFFECTIVE_AXIS,
}

DEEP_PACKAGES: Dict[DecoherenceMode, MeasurementPackage] = {
    DecoherenceMode.METABOLIC: DEEP_METABOLIC,
    DecoherenceMode.AUTONOMIC: DEEP_AUTONOMIC,
    DecoherenceMode.IMMUNE: DEEP_IMMUNE,
    DecoherenceMode.AFFECTIVE: DEEP_AFFECTIVE,
}


# ---------------------------------------------------------------------------
# Scale → decoherence mode mapping (for all 65 production scales)
# ---------------------------------------------------------------------------
SCALE_MODE_MAP: Dict[str, Tuple[DecoherenceMode, ...]] = {
    # Mental health → affective
    "scale_gad_7": (DecoherenceMode.AFFECTIVE,),
    "scale_dass_21_depression": (DecoherenceMode.AFFECTIVE,),
    "scale_dass_21_stress": (DecoherenceMode.AFFECTIVE, DecoherenceMode.AUTONOMIC),
    "scale_pss": (DecoherenceMode.AFFECTIVE, DecoherenceMode.AUTONOMIC),
    "scale_rumination": (DecoherenceMode.AFFECTIVE,),
    "scale_affect_balance_scale": (DecoherenceMode.AFFECTIVE,),
    "scale_hads": (DecoherenceMode.AFFECTIVE,),
    "scale_bpnsnf": (DecoherenceMode.AFFECTIVE,),
    "scale_qols_flanagan": (DecoherenceMode.AFFECTIVE, DecoherenceMode.METABOLIC),
    # Emotional health → affective
    "scale_mhc": (DecoherenceMode.AFFECTIVE,),
    "scale_panas_sf_positive": (DecoherenceMode.AFFECTIVE,),
    "scale_panas_sf_negative": (DecoherenceMode.AFFECTIVE,),
    "scale_aaq": (DecoherenceMode.AFFECTIVE,),
    "scale_brs": (DecoherenceMode.AFFECTIVE, DecoherenceMode.AUTONOMIC),
    "scale_boundaries_assessment": (DecoherenceMode.AFFECTIVE,),
    "scale_escq": (DecoherenceMode.AFFECTIVE,),
    "scale_erq": (DecoherenceMode.AFFECTIVE,),
    "scale_rs": (DecoherenceMode.AFFECTIVE,),
    "scale_ds_14": (DecoherenceMode.AFFECTIVE, DecoherenceMode.AUTONOMIC),
    "scale_paq": (DecoherenceMode.AFFECTIVE, DecoherenceMode.IMMUNE),
    "scale_cope_inventory": (DecoherenceMode.AFFECTIVE,),
    # Cognitive health → autonomic
    "scale_cams_r": (DecoherenceMode.AUTONOMIC,),
    "scale_fmi": (DecoherenceMode.AUTONOMIC,),
    "scale_cfq": (DecoherenceMode.AUTONOMIC,),
    "scale_day_to_day_experiences": (DecoherenceMode.AUTONOMIC,),
    "scale_multifactorial_memory_questionnaire": (DecoherenceMode.AUTONOMIC,),
    # Self concept → affective
    "scale_pgis": (DecoherenceMode.AFFECTIVE,),
    "scale_coping_self_efficacy_scale": (DecoherenceMode.AFFECTIVE,),
    "scale_gse": (DecoherenceMode.AFFECTIVE,),
    "scale_harrill_self_esteem_inventory": (DecoherenceMode.AFFECTIVE,),
    "scale_rses": (DecoherenceMode.AFFECTIVE,),
    "scale_authenticity_scale": (DecoherenceMode.AFFECTIVE,),
    "scale_sris": (DecoherenceMode.AFFECTIVE,),
    "scale_ryff": (DecoherenceMode.AFFECTIVE,),
    # Social → affective
    "scale_mspss": (DecoherenceMode.AFFECTIVE,),
    "scale_sias": (DecoherenceMode.AFFECTIVE,),
    "scale_ssq___isel": (DecoherenceMode.AFFECTIVE,),
    "scale_ucla_loneliness_scale": (DecoherenceMode.AFFECTIVE,),
    # GI health → immune
    "scale_gsrs_ibs": (DecoherenceMode.IMMUNE,),
    "scale_dqlq": (DecoherenceMode.IMMUNE,),
    "scale_ibs": (DecoherenceMode.IMMUNE,),
    "scale_ibdq": (DecoherenceMode.IMMUNE,),
    "scale_visceral_sensitivity_index": (DecoherenceMode.IMMUNE,),
    # Behaviour → autonomic + metabolic
    "scale_cfs": (DecoherenceMode.AUTONOMIC, DecoherenceMode.METABOLIC),
    "scale_bis": (DecoherenceMode.AUTONOMIC,),
    # Personality → affective
    "scale_bif2": (DecoherenceMode.AFFECTIVE,),
    "scale_mind_age": (DecoherenceMode.AFFECTIVE, DecoherenceMode.AUTONOMIC),
    # Initial scan → all modes
    "scale_initial_scan": (
        DecoherenceMode.METABOLIC, DecoherenceMode.AUTONOMIC,
        DecoherenceMode.IMMUNE, DecoherenceMode.AFFECTIVE,
    ),
    "scale_scl": (DecoherenceMode.AUTONOMIC, DecoherenceMode.IMMUNE),
    # Cardiometabolic (from v2) → metabolic
    "scale_findrisc": (DecoherenceMode.METABOLIC,),
    "scale_ez_cvd": (DecoherenceMode.METABOLIC,),
    "scale_scored": (DecoherenceMode.METABOLIC,),
    "scale_lee_nafld": (DecoherenceMode.METABOLIC,),
    "scale_ipaq_sf": (DecoherenceMode.METABOLIC, DecoherenceMode.AUTONOMIC),
    "scale_audit_c": (DecoherenceMode.METABOLIC,),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_core_screener() -> MeasurementPackage:
    """Return the universal core screening package (Tier 0)."""
    return CORE_SCREENER


def get_axis_package(mode: DecoherenceMode) -> MeasurementPackage:
    """Return the axis package for a specific decoherence mode (Tier 1)."""
    return AXIS_PACKAGES[mode]


def get_deep_package(mode: DecoherenceMode) -> MeasurementPackage:
    """Return the deep-dive package for a specific decoherence mode (Tier 2)."""
    return DEEP_PACKAGES[mode]


def get_package(package_id: str) -> Optional[MeasurementPackage]:
    """Look up any package by ID."""
    return ALL_PACKAGES.get(package_id)


def get_scale_modes(scale_id: str) -> Tuple[DecoherenceMode, ...]:
    """Return the decoherence modes a scale contributes to."""
    return SCALE_MODE_MAP.get(scale_id, ())


def get_scales_for_mode(mode: DecoherenceMode) -> List[str]:
    """Return all scale IDs that contribute to a decoherence mode."""
    return [sid for sid, modes in SCALE_MODE_MAP.items() if mode in modes]


def get_progressive_plan(
    user_id: str,
    completed_sessions: int = 0,
    drift_mode: Optional[DecoherenceMode] = None,
    completed_packages: Optional[Sequence[str]] = None,
) -> ProgressivePlan:
    """Build a personalized progressive measurement plan.

    Logic:
    - Session 0: Core Screener
    - Session 1-2: All 4 axis packages (one per session, prioritized
      by core screener results — or round-robin if no signal)
    - Session 3: N-of-1 repeat baseline
    - Session 4+: Deep dives triggered by drift detection

    If ``drift_mode`` is set, the corresponding deep package is
    inserted next in the sequence.
    """
    done = frozenset(completed_packages or [])
    sequence: List[str] = []

    # 1. Core screener first
    if CORE_SCREENER.id not in done:
        sequence.append(CORE_SCREENER.id)

    # 2. Axis packages in a default priority order
    axis_order = [
        DecoherenceMode.AFFECTIVE,   # mood/anxiety usually highest signal
        DecoherenceMode.METABOLIC,   # cardiometabolic risk
        DecoherenceMode.AUTONOMIC,   # sleep/stress
        DecoherenceMode.IMMUNE,      # GI/gut-brain
    ]

    # If drift detected, prioritize that axis
    if drift_mode is not None:
        axis_order = [drift_mode] + [m for m in axis_order if m != drift_mode]

    for mode in axis_order:
        pkg = AXIS_PACKAGES[mode]
        if pkg.id not in done:
            sequence.append(pkg.id)

    # 3. N-of-1 repeat
    if REPEAT_BASELINE.id not in done and completed_sessions >= 2:
        sequence.append(REPEAT_BASELINE.id)

    # 4. Deep dives (only if drift detected)
    if drift_mode is not None:
        deep = DEEP_PACKAGES.get(drift_mode)
        if deep and deep.id not in done:
            # Insert after the corresponding axis package
            axis_pkg_id = AXIS_PACKAGES[drift_mode].id
            if axis_pkg_id in sequence:
                idx = sequence.index(axis_pkg_id) + 1
                sequence.insert(idx, deep.id)
            else:
                sequence.append(deep.id)

    return ProgressivePlan(
        user_id=user_id,
        recommended_sequence=tuple(sequence),
        current_package_idx=0,
        completed_package_ids=done,
    )


def get_package_scale_ids(package_id: str) -> FrozenSet[str]:
    """Return the set of scale IDs referenced by a measurement package.

    This is the key function that bridges measurement packages and the
    selection engine.  The selection engine uses these scale IDs to
    prioritize items from the currently active package.

    Returns an empty frozenset if the package ID is unknown.
    """
    pkg = ALL_PACKAGES.get(package_id)
    if pkg is None:
        return frozenset()
    return frozenset(pi.scale_id for pi in pkg.package_items)


def get_package_scale_priorities(package_id: str) -> Dict[str, float]:
    """Return {scale_id: priority} for scales in a package.

    Higher priority scales within a package should receive more
    selection focus.  The selection engine uses these weights to
    fine-tune the ``package_focus`` boost per item.
    """
    pkg = ALL_PACKAGES.get(package_id)
    if pkg is None:
        return {}
    return {pi.scale_id: pi.priority for pi in pkg.package_items}


def evaluate_package_completion(
    package_id: str,
    scored_scale_ids: FrozenSet[str],
) -> bool:
    """Check whether all required scales in a package have been scored.

    A package is "complete" when every scale listed in its
    ``package_items`` has been scored at least once.  This is the
    minimum viable definition — future iterations could require a
    minimum item count per scale, or require scores within a
    time window.

    Args:
        package_id: The measurement package to evaluate.
        scored_scale_ids: Scale IDs that have at least one score
            for this user.

    Returns:
        True if every scale in the package has been scored.
    """
    pkg_scales = get_package_scale_ids(package_id)
    if not pkg_scales:
        return False
    return pkg_scales.issubset(scored_scale_ids)


def resolve_current_package(
    progressive_plan: ProgressivePlan,
) -> Optional[str]:
    """Return the first incomplete package in the progressive plan.

    Walks the recommended sequence and returns the first package ID
    that is not yet completed.  Returns ``None`` when the full
    sequence has been completed (the user has gone through the
    entire measurement funnel).
    """
    for pkg_id in progressive_plan.recommended_sequence:
        if pkg_id not in progressive_plan.completed_package_ids:
            return pkg_id
    return None


def get_multiplex_value(
    item_id: str,
    scale_mode_map: Dict[str, Tuple[DecoherenceMode, ...]],
    item_to_scales: Dict[str, List[str]],
) -> float:
    """Calculate multiplex value: how many modes does answering this item inform?

    Higher value = one question feeds more decoherence modes.
    Used by the selection engine to prioritize high-value items.
    """
    scales = item_to_scales.get(item_id, [])
    if not scales:
        return 0.0

    all_modes: set = set()
    for sid in scales:
        modes = scale_mode_map.get(sid, ())
        all_modes.update(modes)

    # Value = number of distinct modes informed
    return float(len(all_modes))
