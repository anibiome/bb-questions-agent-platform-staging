"""
Behavioural Metadata as Uncertainty Modifiers (Patent Claim Family 5).

The insight: HOW someone answers a question is informative independently of
WHAT they answer.  A fast, consistent response indicates high confidence;
a hesitant, edited, or skipped response indicates ambiguity or avoidance.

This module captures per-response behavioural metadata and computes
uncertainty modifiers that adjust SE(theta) for the corresponding scale
dimensions.  No existing questionnaire or CAT system uses response-level
behavioural metadata to modulate state uncertainty.

Metadata captured per response event:
  - response_latency_ms: time from question display to final submission
  - edit_count: number of times the user changed their answer before submitting
  - was_skipped: whether the user initially skipped, then came back
  - was_declined: whether the user permanently declined this item
  - channel: input channel (tap, voice, keyboard, swipe)
  - voice_hesitation_ms: if voice channel, pause duration before answer
  - time_of_day_hour: local hour of response (circadian context)

Uncertainty modifier model:
  - Base uncertainty multiplier = 1.0 (neutral)
  - Fast + no edits → multiplier < 1.0 (reduce uncertainty)
  - Slow + edits → multiplier > 1.0 (increase uncertainty)
  - Skip → multiplier >> 1.0 (high uncertainty signal)
  - Decline → treated as missing data with avoidance flag

The modifier is applied as:
  SE_adjusted = SE_irt * uncertainty_multiplier

This does NOT change the theta estimate — it modulates HOW CONFIDENT
we are in that estimate.  A user who answers hesitantly may still give
the same numerical response, but we trust it less.

References:
  - Wise & Kong (2005). Response time effort in psychometric measurement.
  - Ranger & Kuhn (2012). Response time modelling in psychometrics.
  - De Boeck & Jeon (2019). Joint modelling of response times and accuracy.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResponseMetadata:
    """Behavioural metadata captured for a single response event."""
    item_id: str
    response_latency_ms: Optional[float] = None   # milliseconds from display to submit
    edit_count: int = 0                            # answer changes before final submit
    was_skipped: bool = False                      # initially skipped, returned later
    was_declined: bool = False                     # permanently declined (no response)
    channel: str = "tap"                           # tap | voice | keyboard | swipe
    voice_hesitation_ms: Optional[float] = None    # voice-only: pause before answer
    time_of_day_hour: Optional[int] = None         # local hour 0-23 (circadian context)


@dataclass(frozen=True)
class UncertaintyModifier:
    """Computed uncertainty modifier for a single response."""
    item_id: str
    multiplier: float             # ≥ 0.5 (high confidence) to ≤ 3.0 (low confidence)
    confidence_label: str         # "high" | "medium" | "low" | "very_low" | "declined"
    contributing_factors: Tuple[str, ...]   # which metadata signals contributed
    raw_components: Dict[str, float]        # individual factor values for auditing


@dataclass(frozen=True)
class SessionUncertaintyProfile:
    """Aggregated uncertainty profile for a full daily session."""
    session_date: str
    user_id: str
    item_modifiers: Tuple[UncertaintyModifier, ...]
    session_multiplier: float       # geometric mean of item multipliers
    engagement_quality: str         # "focused" | "normal" | "distracted" | "fatigued"
    median_latency_ms: float
    total_edits: int
    skip_count: int
    decline_count: int


# ---------------------------------------------------------------------------
# Reference latency model (item-level)
# ---------------------------------------------------------------------------

# Expected response latency by item complexity.
# These are population medians from psychometric timing literature.
# In a calibrated system, these would be learned per-item from data.

_EXPECTED_LATENCY_MS: Dict[str, float] = {
    "likert_0_4": 4000.0,           # 4s for a standard 5-point Likert
    "likert_0_3": 3500.0,           # 3.5s for a 4-point
    "bool_0_1": 2500.0,             # 2.5s for binary
    "sex_female_male": 2000.0,      # 2s for demographic (fast recall)
    "age_band_0_4": 2000.0,         # 2s for demographic
    "bmi_band_0_3": 3000.0,         # 3s (may need to think)
    "waist_points_0_4": 4000.0,     # 4s (may not know off-hand)
    "family_diabetes_points_0_5": 5000.0,  # 5s (family history = recall effort)
    "audit_frequency_0_4": 4500.0,  # 4.5s (sensitive, may hesitate)
    "audit_typical_0_4": 4500.0,    # 4.5s (sensitive)
    "audit_binge_0_4": 5000.0,      # 5s (most sensitive of AUDIT-C)
    "days_0_7": 3500.0,             # 3.5s
    "minutes_0_180": 5000.0,        # 5s (estimation required)
    "minutes_0_960": 5000.0,        # 5s
}

DEFAULT_EXPECTED_LATENCY_MS = 4000.0


def expected_latency_for_type(response_type: str) -> float:
    """Expected population median response latency for an item type."""
    return _EXPECTED_LATENCY_MS.get(response_type, DEFAULT_EXPECTED_LATENCY_MS)


# ---------------------------------------------------------------------------
# Core uncertainty modifier computation
# ---------------------------------------------------------------------------

def compute_uncertainty_modifier(
    metadata: ResponseMetadata,
    *,
    response_type: str = "likert_0_4",
    is_sensitive: bool = False,
) -> UncertaintyModifier:
    """
    Compute an uncertainty multiplier from behavioural metadata.

    The multiplier adjusts SE(theta): SE_adjusted = SE_irt * multiplier.

    Parameters
    ----------
    metadata : ResponseMetadata for one item response
    response_type : the item's response type (affects expected latency)
    is_sensitive : whether the item is tagged as sensitive (alcohol, etc.)

    Returns
    -------
    UncertaintyModifier with multiplier in [0.5, 3.0]
    """
    if metadata.was_declined:
        return UncertaintyModifier(
            item_id=metadata.item_id,
            multiplier=3.0,
            confidence_label="declined",
            contributing_factors=("declined",),
            raw_components={"declined": 3.0},
        )

    factors: List[str] = []
    components: Dict[str, float] = {}
    multiplier = 1.0

    # --- Factor 1: Response latency ---
    if metadata.response_latency_ms is not None and metadata.response_latency_ms > 0:
        expected = expected_latency_for_type(response_type)
        # Sensitive items get 50% more expected time (legitimate hesitation)
        if is_sensitive:
            expected *= 1.5

        ratio = metadata.response_latency_ms / expected

        if ratio < 0.3:
            # Suspiciously fast — possible inattentive rapid-fire
            latency_mod = 1.3
            factors.append("suspiciously_fast")
        elif ratio < 0.7:
            # Fast but plausible — high confidence
            latency_mod = 0.85
            factors.append("fast_response")
        elif ratio <= 1.5:
            # Normal range — neutral
            latency_mod = 1.0
        elif ratio <= 3.0:
            # Slow — possible uncertainty or deliberation
            latency_mod = 1.15
            factors.append("slow_response")
        else:
            # Very slow — possible distraction or high difficulty
            latency_mod = 1.3
            factors.append("very_slow_response")

        components["latency_modifier"] = latency_mod
        multiplier *= latency_mod

    # --- Factor 2: Edit count ---
    edits = max(0, metadata.edit_count)
    if edits == 0:
        edit_mod = 0.95  # no changes = slight confidence boost
        factors.append("no_edits")
    elif edits == 1:
        edit_mod = 1.1  # one change = minor reconsideration
        factors.append("single_edit")
    elif edits <= 3:
        edit_mod = 1.25  # multiple changes = genuine uncertainty
        factors.append("multiple_edits")
    else:
        edit_mod = 1.4  # excessive changes = high ambivalence
        factors.append("excessive_edits")

    components["edit_modifier"] = edit_mod
    multiplier *= edit_mod

    # --- Factor 3: Skip-then-return ---
    if metadata.was_skipped:
        skip_mod = 1.35  # skipping indicates avoidance or difficulty
        factors.append("skip_then_return")
        components["skip_modifier"] = skip_mod
        multiplier *= skip_mod

    # --- Factor 4: Voice channel hesitation ---
    if metadata.channel == "voice" and metadata.voice_hesitation_ms is not None:
        if metadata.voice_hesitation_ms < 500:
            voice_mod = 0.9  # immediate verbal response = confident
            factors.append("voice_immediate")
        elif metadata.voice_hesitation_ms < 2000:
            voice_mod = 1.0  # normal pause
        elif metadata.voice_hesitation_ms < 5000:
            voice_mod = 1.15  # long pause = thinking/uncertain
            factors.append("voice_hesitation")
        else:
            voice_mod = 1.3  # very long pause
            factors.append("voice_long_hesitation")

        components["voice_modifier"] = voice_mod
        multiplier *= voice_mod

    # --- Factor 5: Circadian context ---
    if metadata.time_of_day_hour is not None:
        hour = metadata.time_of_day_hour % 24
        # Late night / very early morning responses less reliable
        if 2 <= hour <= 5:
            circadian_mod = 1.15
            factors.append("late_night_response")
        elif 23 <= hour or hour <= 1:
            circadian_mod = 1.08
            factors.append("night_response")
        else:
            circadian_mod = 1.0  # daytime = neutral

        components["circadian_modifier"] = circadian_mod
        multiplier *= circadian_mod

    # Clamp to valid range
    multiplier = max(0.5, min(3.0, multiplier))

    # Determine confidence label
    if multiplier <= 0.8:
        label = "high"
    elif multiplier <= 1.1:
        label = "medium"
    elif multiplier <= 1.5:
        label = "low"
    else:
        label = "very_low"

    return UncertaintyModifier(
        item_id=metadata.item_id,
        multiplier=round(multiplier, 4),
        confidence_label=label,
        contributing_factors=tuple(factors),
        raw_components=components,
    )


def compute_session_uncertainty_profile(
    modifiers: Sequence[UncertaintyModifier],
    metadata_list: Sequence[ResponseMetadata],
    *,
    session_date: str,
    user_id: str,
) -> SessionUncertaintyProfile:
    """
    Aggregate item-level uncertainty modifiers into a session profile.

    The session multiplier is the geometric mean of item multipliers
    (geometric mean because multipliers are multiplicative factors).
    """
    if not modifiers:
        return SessionUncertaintyProfile(
            session_date=session_date,
            user_id=user_id,
            item_modifiers=(),
            session_multiplier=1.0,
            engagement_quality="normal",
            median_latency_ms=0.0,
            total_edits=0,
            skip_count=0,
            decline_count=0,
        )

    # Geometric mean of multipliers
    log_sum = sum(math.log(max(0.5, m.multiplier)) for m in modifiers)
    session_mult = math.exp(log_sum / len(modifiers))
    session_mult = round(max(0.5, min(3.0, session_mult)), 4)

    # Aggregate metadata
    latencies = [
        m.response_latency_ms
        for m in metadata_list
        if m.response_latency_ms is not None and m.response_latency_ms > 0
    ]
    median_lat = _median(latencies) if latencies else 0.0
    total_edits = sum(m.edit_count for m in metadata_list)
    skip_count = sum(1 for m in metadata_list if m.was_skipped)
    decline_count = sum(1 for m in metadata_list if m.was_declined)

    # Engagement quality classification
    if session_mult <= 0.85 and total_edits <= 1 and skip_count == 0:
        quality = "focused"
    elif session_mult > 1.4 or skip_count >= 2 or decline_count >= 1:
        quality = "distracted"
    elif session_mult > 1.6 or total_edits >= 8:
        quality = "fatigued"
    else:
        quality = "normal"

    return SessionUncertaintyProfile(
        session_date=session_date,
        user_id=user_id,
        item_modifiers=tuple(modifiers),
        session_multiplier=session_mult,
        engagement_quality=quality,
        median_latency_ms=round(median_lat, 1),
        total_edits=total_edits,
        skip_count=skip_count,
        decline_count=decline_count,
    )


# ---------------------------------------------------------------------------
# Adjusted SE computation
# ---------------------------------------------------------------------------

def adjust_se_with_metadata(
    se_theta: float,
    item_modifiers: Sequence[UncertaintyModifier],
) -> float:
    """
    Adjust IRT SE(theta) using behavioural metadata uncertainty modifiers.

    SE_adjusted = SE_irt * session_uncertainty_multiplier

    The session multiplier is the geometric mean of item-level multipliers
    for all items that contributed to the scale score.

    Parameters
    ----------
    se_theta : SE from IRT scoring (before metadata adjustment)
    item_modifiers : uncertainty modifiers for items used in this scale

    Returns
    -------
    Adjusted SE(theta), always >= se_theta * 0.5 and <= se_theta * 3.0
    """
    if not item_modifiers:
        return se_theta

    log_sum = sum(math.log(max(0.5, m.multiplier)) for m in item_modifiers)
    geo_mean = math.exp(log_sum / len(item_modifiers))
    geo_mean = max(0.5, min(3.0, geo_mean))

    return round(se_theta * geo_mean, 4)


# ---------------------------------------------------------------------------
# Engagement-based item selection priority
# ---------------------------------------------------------------------------

def engagement_selection_bonus(
    session_profile: SessionUncertaintyProfile,
    item_id: str,
    *,
    response_type: str = "likert_0_4",
) -> float:
    """
    Compute a selection bonus/penalty for an item based on user engagement.

    Focused users → no change (already optimal).
    Distracted users → prefer simpler (binary, Likert) items, avoid complex items.
    Fatigued users → prefer novel items (variety to re-engage).

    Returns a score adjustment in [-1.0, +1.0] range.
    """
    quality = session_profile.engagement_quality

    if quality == "focused":
        return 0.0  # already optimal engagement

    if quality == "distracted":
        # Prefer simpler response types for distracted users
        simple_types = {"bool_0_1", "sex_female_male", "age_band_0_4"}
        if response_type in simple_types:
            return 0.3  # bonus for easy items
        complex_types = {"minutes_0_180", "minutes_0_960", "family_diabetes_points_0_5"}
        if response_type in complex_types:
            return -0.3  # penalty for complex items
        return 0.0

    if quality == "fatigued":
        # Slight penalty for all items (user burden awareness)
        return -0.1

    return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median(values: Sequence[float]) -> float:
    """Simple median without numpy dependency."""
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
