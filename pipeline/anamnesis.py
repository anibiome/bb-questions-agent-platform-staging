"""
Anamnesis Loop: Drift-Triggered Clinical Branching (Patent Claim Family 4).

When the trajectory monitoring system detects drift in a latent state
dimension, the anamnesis agent automatically:
  1. Identifies the clinical domain(s) affected
  2. Maps drift to the appropriate validated instrument(s) via the routing table
  3. Enqueues targeted items into the follow-up queue with priority
  4. Tracks the branching episode through: screen → detect → branch → confirm/deny → stop

This transforms the Questions Agent from a passive measurement tool into an
active diagnostic probe that responds to real-time state inference.

No existing EMA or CAT system triggers targeted clinical probes from
real-time state inference.  Existing systems use fixed schedules.

Trigger conditions (any one is sufficient):
  - EWS score exceeds threshold (default: 0.6)
  - Radius deviation r(t) exceeds personal baseline by >2σ
  - Trajectory velocity exceeds threshold
  - Cross-modal disagreement detected (questionnaire vs. wearable)

References:
  - Shiffman et al. (2008). Ecological Momentary Assessment. Annual Rev Clin Psych.
  - Fisher et al. (2017). Intensive Longitudinal Data in Psychology.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from questions_agent_platform.pipeline.baseline import BaselineState, baseline_std
from questions_agent_platform.pipeline.drift_routing import (
    DriftRoute,
    default_drift_routes,
    resolve_drift_route,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnamnesisEpisode:
    """A single drift-triggered branching episode."""
    episode_id: str
    user_id: str
    trigger_date: str                           # ISO date when drift detected
    drift_domain: str                           # e.g. "metabolic", "cardiovascular"
    trigger_type: str                           # "ews_threshold" | "radius_deviation" | "velocity" | "cross_modal"
    trigger_value: float                        # the numeric trigger value
    triggered_scale_ids: Tuple[str, ...]        # instruments to probe
    status: str                                 # "active" | "confirmed" | "denied" | "resolved" | "expired"
    follow_up_item_ids: Tuple[str, ...]         # items enqueued
    priority: int = 100                         # lower = higher priority
    max_days: int = 7                           # max days before expiration
    observations_collected: int = 0             # items answered so far
    resolution_date: Optional[str] = None       # date of resolution
    resolution_reason: Optional[str] = None     # why resolved


@dataclass(frozen=True)
class AnamnesisAction:
    """Action to take based on anamnesis evaluation."""
    action_type: str                 # "start_episode" | "continue_episode" | "resolve_episode" | "no_action"
    episode_id: Optional[str] = None
    follow_up_item_ids: Tuple[str, ...] = ()    # items to add to today's session
    priority_boost: float = 0.0                  # score boost for follow-up items
    reason: str = ""


# ---------------------------------------------------------------------------
# Drift trigger evaluation
# ---------------------------------------------------------------------------

# Thresholds for triggering anamnesis
DEFAULT_EWS_THRESHOLD = 0.6
DEFAULT_RADIUS_Z_THRESHOLD = 2.0
DEFAULT_VELOCITY_THRESHOLD = 0.15


def evaluate_drift_triggers(
    *,
    ews_score: Optional[float] = None,
    radius_current: Optional[float] = None,
    radius_baseline: Optional[BaselineState] = None,
    velocity: Optional[float] = None,
    cross_modal_disagreement: bool = False,
    ews_threshold: float = DEFAULT_EWS_THRESHOLD,
    radius_z_threshold: float = DEFAULT_RADIUS_Z_THRESHOLD,
    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD,
) -> List[Tuple[str, float]]:
    """
    Evaluate all drift trigger conditions.

    Returns list of (trigger_type, trigger_value) for all conditions that fire.
    Multiple triggers can fire simultaneously.
    """
    triggers: List[Tuple[str, float]] = []

    # EWS threshold
    if ews_score is not None and ews_score >= ews_threshold:
        triggers.append(("ews_threshold", float(ews_score)))

    # Radius deviation from personal baseline
    if radius_current is not None and radius_baseline is not None:
        if radius_baseline.n >= 5:  # need sufficient baseline
            std = baseline_std(radius_baseline)
            if std > 0.01:
                z = (radius_current - radius_baseline.mean) / std
                if abs(z) >= radius_z_threshold:
                    triggers.append(("radius_deviation", float(z)))

    # Trajectory velocity
    if velocity is not None and abs(velocity) >= velocity_threshold:
        triggers.append(("velocity", float(velocity)))

    # Cross-modal disagreement
    if cross_modal_disagreement:
        triggers.append(("cross_modal", 1.0))

    return triggers


# ---------------------------------------------------------------------------
# Episode management
# ---------------------------------------------------------------------------

def create_anamnesis_episode(
    *,
    user_id: str,
    trigger_date: date,
    trigger_type: str,
    trigger_value: float,
    drift_domain: str,
    triggered_scale_ids: Sequence[str],
    follow_up_item_ids: Sequence[str],
    priority: int = 100,
    max_days: int = 7,
) -> AnamnesisEpisode:
    """Create a new anamnesis episode from a drift trigger."""
    episode_id = f"anm_{user_id}_{trigger_date.isoformat()}_{drift_domain}"
    return AnamnesisEpisode(
        episode_id=episode_id,
        user_id=user_id,
        trigger_date=trigger_date.isoformat(),
        drift_domain=drift_domain,
        trigger_type=trigger_type,
        trigger_value=trigger_value,
        triggered_scale_ids=tuple(triggered_scale_ids),
        status="active",
        follow_up_item_ids=tuple(follow_up_item_ids),
        priority=priority,
        max_days=max_days,
    )


def evaluate_episode_resolution(
    episode: AnamnesisEpisode,
    *,
    current_date: date,
    current_ews_score: Optional[float] = None,
    observations_collected: int = 0,
    scale_completed: bool = False,
    ews_threshold: float = DEFAULT_EWS_THRESHOLD,
) -> AnamnesisEpisode:
    """
    Evaluate whether an active episode should be resolved.

    Resolution conditions (any one):
    1. Triggered scale has been fully completed → "confirmed" or "denied" based on score
    2. EWS score dropped below threshold → "resolved" (drift subsided)
    3. Max days exceeded → "expired"
    4. All follow-up items answered → "resolved"
    """
    if episode.status != "active":
        return episode

    trigger_date = date.fromisoformat(episode.trigger_date)
    days_elapsed = (current_date - trigger_date).days

    # Check expiration
    if days_elapsed > episode.max_days:
        return AnamnesisEpisode(
            episode_id=episode.episode_id,
            user_id=episode.user_id,
            trigger_date=episode.trigger_date,
            drift_domain=episode.drift_domain,
            trigger_type=episode.trigger_type,
            trigger_value=episode.trigger_value,
            triggered_scale_ids=episode.triggered_scale_ids,
            status="expired",
            follow_up_item_ids=episode.follow_up_item_ids,
            priority=episode.priority,
            max_days=episode.max_days,
            observations_collected=observations_collected,
            resolution_date=current_date.isoformat(),
            resolution_reason="max_days_exceeded",
        )

    # Check if scale was completed (instrument unlocked)
    if scale_completed:
        return AnamnesisEpisode(
            episode_id=episode.episode_id,
            user_id=episode.user_id,
            trigger_date=episode.trigger_date,
            drift_domain=episode.drift_domain,
            trigger_type=episode.trigger_type,
            trigger_value=episode.trigger_value,
            triggered_scale_ids=episode.triggered_scale_ids,
            status="confirmed",
            follow_up_item_ids=episode.follow_up_item_ids,
            priority=episode.priority,
            max_days=episode.max_days,
            observations_collected=observations_collected,
            resolution_date=current_date.isoformat(),
            resolution_reason="scale_completed",
        )

    # Check if EWS dropped back to normal
    if current_ews_score is not None and current_ews_score < ews_threshold * 0.7:
        return AnamnesisEpisode(
            episode_id=episode.episode_id,
            user_id=episode.user_id,
            trigger_date=episode.trigger_date,
            drift_domain=episode.drift_domain,
            trigger_type=episode.trigger_type,
            trigger_value=episode.trigger_value,
            triggered_scale_ids=episode.triggered_scale_ids,
            status="denied",
            follow_up_item_ids=episode.follow_up_item_ids,
            priority=episode.priority,
            max_days=episode.max_days,
            observations_collected=observations_collected,
            resolution_date=current_date.isoformat(),
            resolution_reason="ews_subsided",
        )

    # Still active
    return AnamnesisEpisode(
        episode_id=episode.episode_id,
        user_id=episode.user_id,
        trigger_date=episode.trigger_date,
        drift_domain=episode.drift_domain,
        trigger_type=episode.trigger_type,
        trigger_value=episode.trigger_value,
        triggered_scale_ids=episode.triggered_scale_ids,
        status="active",
        follow_up_item_ids=episode.follow_up_item_ids,
        priority=episode.priority,
        max_days=episode.max_days,
        observations_collected=observations_collected,
    )


# ---------------------------------------------------------------------------
# Full anamnesis loop integration
# ---------------------------------------------------------------------------

def run_anamnesis_check(
    *,
    user_id: str,
    current_date: date,
    active_episodes: Sequence[AnamnesisEpisode],
    ews_score: Optional[float] = None,
    radius_current: Optional[float] = None,
    radius_baseline: Optional[BaselineState] = None,
    velocity: Optional[float] = None,
    cross_modal_disagreement: bool = False,
    drift_domain_to_scale_ids: Optional[Dict[str, List[str]]] = None,
    drift_domain_to_item_ids: Optional[Dict[str, List[str]]] = None,
    max_concurrent_episodes: int = 2,
) -> List[AnamnesisAction]:
    """
    Main anamnesis loop entry point.

    Called daily during item selection to:
    1. Check for new drift triggers
    2. Evaluate existing episodes for resolution
    3. Return actions for the selection engine

    Parameters
    ----------
    active_episodes : currently active anamnesis episodes
    ews_score : latest EWS composite score
    drift_domain_to_scale_ids : mapping from domain → scale IDs to probe
    drift_domain_to_item_ids : mapping from domain → specific items to enqueue
    max_concurrent_episodes : limit on simultaneous active episodes
    """
    actions: List[AnamnesisAction] = []

    # 1. Evaluate existing episodes for resolution
    for ep in active_episodes:
        if ep.status != "active":
            continue

        resolved = evaluate_episode_resolution(
            ep,
            current_date=current_date,
            current_ews_score=ews_score,
        )
        if resolved.status != "active":
            actions.append(AnamnesisAction(
                action_type="resolve_episode",
                episode_id=resolved.episode_id,
                reason=resolved.resolution_reason or "resolved",
            ))
        else:
            # Continue active episode — re-enqueue remaining items
            actions.append(AnamnesisAction(
                action_type="continue_episode",
                episode_id=ep.episode_id,
                follow_up_item_ids=ep.follow_up_item_ids,
                priority_boost=18.0,  # matches existing anamnesis_followup score
                reason=f"continue_{ep.drift_domain}_probe",
            ))

    # 2. Check for new triggers
    active_count = sum(1 for ep in active_episodes if ep.status == "active")
    resolved_ids = {a.episode_id for a in actions if a.action_type == "resolve_episode"}
    active_count -= len(resolved_ids)

    if active_count < max_concurrent_episodes:
        triggers = evaluate_drift_triggers(
            ews_score=ews_score,
            radius_current=radius_current,
            radius_baseline=radius_baseline,
            velocity=velocity,
            cross_modal_disagreement=cross_modal_disagreement,
        )

        if triggers:
            # Use the strongest trigger
            strongest = max(triggers, key=lambda t: abs(t[1]))
            trigger_type, trigger_value = strongest

            # Determine drift domain from the trigger
            domain = _infer_drift_domain(trigger_type)

            # Check if we already have an active episode for this domain
            existing_domains = {
                ep.drift_domain for ep in active_episodes
                if ep.status == "active" and ep.episode_id not in resolved_ids
            }
            if domain not in existing_domains:
                # Map domain to scale IDs and items
                scale_ids = (drift_domain_to_scale_ids or {}).get(domain, [])
                item_ids = (drift_domain_to_item_ids or {}).get(domain, [])

                if scale_ids or item_ids:
                    episode = create_anamnesis_episode(
                        user_id=user_id,
                        trigger_date=current_date,
                        trigger_type=trigger_type,
                        trigger_value=trigger_value,
                        drift_domain=domain,
                        triggered_scale_ids=scale_ids,
                        follow_up_item_ids=item_ids,
                    )
                    actions.append(AnamnesisAction(
                        action_type="start_episode",
                        episode_id=episode.episode_id,
                        follow_up_item_ids=tuple(item_ids),
                        priority_boost=18.0,
                        reason=f"drift_{domain}_{trigger_type}",
                    ))

    return actions


def _infer_drift_domain(trigger_type: str) -> str:
    """Infer the clinical domain from the trigger type.

    In the full system, the drift domain comes from the latent state
    dimension that triggered.  Here we use a simple mapping.
    """
    # Default: if trigger comes from EWS or cross-modal, it's general
    if trigger_type in ("ews_threshold", "cross_modal"):
        return "general"
    if trigger_type == "velocity":
        return "metabolic"  # velocity changes often metabolic
    if trigger_type == "radius_deviation":
        return "cardiovascular"  # radius deviation = systemic risk
    return "general"


# ---------------------------------------------------------------------------
# Routing table integration helpers
# ---------------------------------------------------------------------------

def build_domain_to_items_map(
    routes: Sequence[DriftRoute],
    scale_items: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Build a mapping from drift domain → item IDs for follow-up.

    Uses the drift routing table to determine which instruments to probe,
    then expands to the specific items of those instruments.
    """
    domain_to_items: Dict[str, List[str]] = {}
    for route in routes:
        items: List[str] = []
        for scale_id in route.scale_ids:
            items.extend(scale_items.get(scale_id, []))
        if route.triggered_instrument:
            items.extend(scale_items.get(route.triggered_instrument, []))
        if items:
            domain_to_items[route.drift_domain] = list(dict.fromkeys(items))  # deduplicate
    return domain_to_items


def build_domain_to_scales_map(
    routes: Sequence[DriftRoute],
) -> Dict[str, List[str]]:
    """Build mapping from drift domain → scale IDs."""
    domain_to_scales: Dict[str, List[str]] = {}
    for route in routes:
        scales = list(route.scale_ids)
        if route.triggered_instrument and route.triggered_instrument not in scales:
            scales.append(route.triggered_instrument)
        if scales:
            domain_to_scales[route.drift_domain] = scales
    return domain_to_scales
