"""Active Coherence Detection from Behavioral Metadata.

Detects careless/inattentive responding in real time using multiple
converging behavioral signals captured during questionnaire administration.

Implements the multiple-hurdle approach (DeSimone et al., 2015, JAP):
flag a session when >= 2 independent indicators suggest low-quality
responding.  Six validated signals are computed:

  1. Speed index       — proportion of items below 2 s (Huang et al., 2012)
  2. Longstring        — longest run of identical consecutive responses
                         (Johnson, 2005; Meade & Craig, 2012)
  3. Response-time CV  — coefficient of variation of RTs; very low CV
                         implies mechanical/scripted responding
  4. Intra-scale var   — mean within-scale response variance; near-zero
                         with multiple response options implies straightlining
  5. Fatigue slope     — linear RT trend over session; negative log-slope
                         (speeding up) implies progressive disengagement
  6. Skip rate         — proportion of skipped/declined items

Each signal computes a continuous severity score [0, 1] and a binary
flagged/not-flagged status.  The composite coherence_score is the
severity-weighted complement:
    coherence_score = 1.0 - sum(severity_i * weight_i)

Tier assignment follows the multiple-hurdle:
    valid:    <  2 flags
    suspect:  >= 2 flags (DeSimone et al., 2015)
    invalid:  >= 4 flags

All functions are pure: no I/O, no side effects, no global state.
Input is a sequence of ItemResponse objects; output is a frozen
CoherenceAssessment dataclass.

References
----------
DeSimone et al., 2015.  Best practice recommendations for data
    screening.  Journal of Organizational Behavior, 36, 171-181.
Huang et al., 2012.  Detecting and deterring insufficient effort
    responding.  J Business and Psychology, 27, 99-114.
Johnson, 2005.  Ascertaining the validity of individual protocols.
    Multivariate Behavioral Research, 40, 169-187.
Meade & Craig, 2012.  Identifying careless responses in survey data.
    Psychological Methods, 17(3), 437-455.
Curran, 2016.  Methods for detection of carelessly invalid responses.
    J Experimental Social Psychology, 66, 4-19.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


COHERENCE_VERSION = "coherence_v1"

# ---------------------------------------------------------------------------
# Thresholds (literature-backed)
# ---------------------------------------------------------------------------
FAST_RT_MS = 2000.0                # Huang et al. 2012: minimum reading time
SPEED_INDEX_THRESHOLD = 0.50       # > 50% items below fast threshold
LONGSTRING_THRESHOLD = 8           # absolute max consecutive identical
LONGSTRING_RATIO_THRESHOLD = 0.60  # or > 60% of item count
RT_CV_THRESHOLD = 0.20             # CV(RT) below this is suspect
INTRA_SCALE_VAR_THRESHOLD = 0.10   # normalized within-scale variance
FATIGUE_SLOPE_THRESHOLD = -0.50    # log(RT) slope: ~39% RT decrease
SKIP_RATE_THRESHOLD = 0.30         # > 30% items skipped/declined

# Composite scoring
SUSPECT_MIN_FLAGS = 2              # DeSimone et al. 2015
INVALID_MIN_FLAGS = 4

# Signal weights for composite score (sum to 1.0)
SIGNAL_WEIGHTS: Dict[str, float] = {
    "speed_index":         0.20,
    "longstring":          0.20,
    "rt_variability":      0.15,
    "intra_scale_variance": 0.15,
    "fatigue_slope":       0.15,
    "skip_rate":           0.15,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemResponse:
    """Single item response with behavioral metadata.

    This is the minimal behavioral record needed for coherence assessment.
    Maps directly to the response_metadata table schema.
    """
    item_id: str
    scale_id: str
    response_value: int          # ordinal response (0-indexed within scale)
    response_time_ms: float      # time from display to submission
    item_position: int           # 0-indexed position in session
    n_response_options: int      # number of response categories
    edit_count: int = 0
    was_skipped: bool = False


@dataclass(frozen=True)
class CoherenceSignal:
    """Result of a single coherence indicator."""
    name: str                    # signal identifier
    value: float                 # raw signal value
    threshold: float             # flagging threshold
    flagged: bool                # crossed threshold
    severity: float              # 0.0 (no concern) to 1.0 (maximum concern)
    description: str             # human-readable interpretation


@dataclass(frozen=True)
class CoherenceAssessment:
    """Comprehensive coherence assessment for a response session."""
    signals: Tuple[CoherenceSignal, ...]
    n_flagged: int               # count of flagged signals
    coherence_score: float       # 0 (incoherent) to 1 (coherent)
    tier: str                    # "valid" | "suspect" | "invalid"
    n_items: int                 # total items assessed
    version: str = COHERENCE_VERSION


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assess_coherence(
    responses: Sequence[ItemResponse],
    *,
    fast_rt_ms: float = FAST_RT_MS,
) -> CoherenceAssessment:
    """Assess response coherence from behavioral metadata.

    Pure function: takes a sequence of item responses and returns a
    comprehensive coherence assessment with six independent signals
    and a composite score.

    Parameters
    ----------
    responses : sequence of ItemResponse
        All item responses from a single session, in presentation order.
    fast_rt_ms : float
        Threshold for "too fast" in milliseconds (default 2000).

    Returns
    -------
    CoherenceAssessment
        Full assessment with per-signal detail and composite score.
    """
    answered = [r for r in responses if not r.was_skipped]
    all_items = list(responses)
    n_total = len(all_items)

    if n_total == 0:
        return CoherenceAssessment(
            signals=(),
            n_flagged=0,
            coherence_score=1.0,
            tier="valid",
            n_items=0,
        )

    signals: List[CoherenceSignal] = [
        _speed_index(answered, fast_rt_ms),
        _longstring_index(answered),
        _rt_variability(answered),
        _intra_scale_variance(answered),
        _fatigue_slope(answered),
        _skip_rate(all_items),
    ]

    signals_tuple = tuple(signals)
    n_flagged = sum(1 for s in signals_tuple if s.flagged)

    # Composite score: 1.0 minus weighted severity
    weighted_severity = sum(
        s.severity * SIGNAL_WEIGHTS.get(s.name, 1.0 / len(signals_tuple))
        for s in signals_tuple
    )
    coherence_score = max(0.0, min(1.0, 1.0 - weighted_severity))

    # Tier assignment (DeSimone multiple-hurdle)
    if n_flagged >= INVALID_MIN_FLAGS:
        tier = "invalid"
    elif n_flagged >= SUSPECT_MIN_FLAGS:
        tier = "suspect"
    else:
        tier = "valid"

    return CoherenceAssessment(
        signals=signals_tuple,
        n_flagged=n_flagged,
        coherence_score=round(coherence_score, 4),
        tier=tier,
        n_items=n_total,
    )


# ---------------------------------------------------------------------------
# Signal 1: Speed index (Huang et al., 2012)
# ---------------------------------------------------------------------------

def _speed_index(
    items: List[ItemResponse],
    fast_threshold_ms: float,
) -> CoherenceSignal:
    """Proportion of items answered below the fast-response threshold.

    Huang et al. (2012): 2 seconds per item is the minimum time for
    reading and processing a typical Likert-scale item.  Responses
    faster than this suggest the item was not read.

    Severity ramps linearly from 0 at threshold to 1 at 100%.
    """
    if not items:
        return CoherenceSignal(
            "speed_index", 0.0, SPEED_INDEX_THRESHOLD, False, 0.0,
            "No answered items",
        )

    fast_count = sum(1 for r in items if r.response_time_ms < fast_threshold_ms)
    proportion = fast_count / len(items)
    flagged = proportion > SPEED_INDEX_THRESHOLD

    severity = _ramp_severity(proportion, SPEED_INDEX_THRESHOLD, 1.0)

    return CoherenceSignal(
        name="speed_index",
        value=round(proportion, 4),
        threshold=SPEED_INDEX_THRESHOLD,
        flagged=flagged,
        severity=round(severity, 4),
        description=f"{fast_count}/{len(items)} items below {fast_threshold_ms:.0f}ms",
    )


# ---------------------------------------------------------------------------
# Signal 2: Longstring index (Johnson, 2005)
# ---------------------------------------------------------------------------

def _longstring_index(items: List[ItemResponse]) -> CoherenceSignal:
    """Longest run of identical consecutive responses.

    Johnson (2005): the longstring index counts the maximum number of
    consecutive identical responses.  Long runs suggest the respondent
    is selecting the same option without reading items.

    Two criteria (flagged if EITHER is exceeded):
      - Absolute: max_run >= LONGSTRING_THRESHOLD (8)
      - Relative: max_run / n_items >= LONGSTRING_RATIO_THRESHOLD (60%)
    """
    if len(items) < 3:
        return CoherenceSignal(
            "longstring", 0, LONGSTRING_THRESHOLD, False, 0.0,
            "Too few items",
        )

    max_run = 1
    current_run = 1
    for i in range(1, len(items)):
        if items[i].response_value == items[i - 1].response_value:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    # Flag if absolute threshold crossed OR relative threshold crossed
    ratio = max_run / len(items)
    flagged = max_run >= LONGSTRING_THRESHOLD or ratio >= LONGSTRING_RATIO_THRESHOLD

    # Severity: ramp from the lower of the two thresholds
    effective_threshold = min(
        LONGSTRING_THRESHOLD,
        max(3, int(math.ceil(len(items) * LONGSTRING_RATIO_THRESHOLD))),
    )
    severity = _ramp_severity(
        float(max_run), float(effective_threshold),
        float(effective_threshold) * 1.5,
    )

    return CoherenceSignal(
        name="longstring",
        value=float(max_run),
        threshold=float(LONGSTRING_THRESHOLD),
        flagged=flagged,
        severity=round(severity, 4),
        description=f"Longest identical run: {max_run}/{len(items)} items",
    )


# ---------------------------------------------------------------------------
# Signal 3: Response time variability
# ---------------------------------------------------------------------------

def _rt_variability(items: List[ItemResponse]) -> CoherenceSignal:
    """Coefficient of variation of response times.

    Very low RT variability (CV < 0.20) suggests mechanical or scripted
    responding, where the person is clicking at a constant rate without
    engaging with item content.  Attentive respondents show natural
    RT variation from item difficulty and content engagement.

    Severity is inverse: lower CV = higher severity.
    """
    if len(items) < 3:
        return CoherenceSignal(
            "rt_variability", 1.0, RT_CV_THRESHOLD, False, 0.0,
            "Too few items",
        )

    rts = [r.response_time_ms for r in items if r.response_time_ms > 0]
    if len(rts) < 3:
        return CoherenceSignal(
            "rt_variability", 1.0, RT_CV_THRESHOLD, False, 0.0,
            "No valid RTs",
        )

    mean_rt = sum(rts) / len(rts)
    if mean_rt < 1.0:
        return CoherenceSignal(
            "rt_variability", 0.0, RT_CV_THRESHOLD, True, 1.0,
            "Near-zero mean RT",
        )

    var_rt = sum((t - mean_rt) ** 2 for t in rts) / (len(rts) - 1)
    sd_rt = math.sqrt(max(0.0, var_rt))
    cv = sd_rt / mean_rt

    flagged = cv < RT_CV_THRESHOLD
    # Inverse: lower CV = higher severity
    severity = _ramp_severity_inverse(cv, 0.0, RT_CV_THRESHOLD)

    return CoherenceSignal(
        name="rt_variability",
        value=round(cv, 4),
        threshold=RT_CV_THRESHOLD,
        flagged=flagged,
        severity=round(severity, 4),
        description=f"RT coefficient of variation: {cv:.3f}",
    )


# ---------------------------------------------------------------------------
# Signal 4: Intra-scale variance
# ---------------------------------------------------------------------------

def _intra_scale_variance(items: List[ItemResponse]) -> CoherenceSignal:
    """Average within-scale response variance, normalized by scale range.

    For multi-item scales, some response variance is expected (different
    items tap different facets).  Near-zero variance with multiple
    response options strongly suggests straightlining.

    Computed per-scale, then averaged across scales (weighted by item
    count).  Normalized by the variance of a uniform distribution over
    the response range: (n_options - 1)^2 / 12.

    Known limitation: legitimate extreme scorers (all-high or all-low)
    will have low variance.  This is why we use the multiple-hurdle
    approach — a single flag does not change the tier.
    """
    # Group by scale
    by_scale: Dict[str, List[ItemResponse]] = {}
    for r in items:
        by_scale.setdefault(r.scale_id, []).append(r)

    if not by_scale:
        return CoherenceSignal(
            "intra_scale_variance", 1.0, INTRA_SCALE_VAR_THRESHOLD,
            False, 0.0, "No scales",
        )

    weighted_var_sum = 0.0
    total_weight = 0.0

    for scale_id, scale_items in by_scale.items():
        if len(scale_items) < 2:
            continue

        values = [float(r.response_value) for r in scale_items]
        n = len(values)
        mean_val = sum(values) / n
        var_val = sum((v - mean_val) ** 2 for v in values) / (n - 1)

        # Normalize by expected variance of uniform distribution
        max_options = max(r.n_response_options for r in scale_items)
        uniform_var = (max_options - 1) ** 2 / 12.0 if max_options > 1 else 1.0
        normalized_var = var_val / max(1e-12, uniform_var)

        weighted_var_sum += normalized_var * n
        total_weight += n

    if total_weight < 2.0:
        return CoherenceSignal(
            "intra_scale_variance", 1.0, INTRA_SCALE_VAR_THRESHOLD,
            False, 0.0, "Insufficient scale data",
        )

    avg_norm_var = weighted_var_sum / total_weight
    flagged = avg_norm_var < INTRA_SCALE_VAR_THRESHOLD

    # Inverse: lower variance = higher severity
    severity = _ramp_severity_inverse(avg_norm_var, 0.0, INTRA_SCALE_VAR_THRESHOLD)

    return CoherenceSignal(
        name="intra_scale_variance",
        value=round(avg_norm_var, 4),
        threshold=INTRA_SCALE_VAR_THRESHOLD,
        flagged=flagged,
        severity=round(severity, 4),
        description=f"Normalized intra-scale variance: {avg_norm_var:.4f}",
    )


# ---------------------------------------------------------------------------
# Signal 5: Fatigue slope
# ---------------------------------------------------------------------------

def _fatigue_slope(items: List[ItemResponse]) -> CoherenceSignal:
    """Linear trend of log(RT) over session position.

    Negative slope (RT decreasing over items) suggests the respondent
    is speeding up as they disengage.  Computed as the OLS slope of
    log(RT) on normalized item position [0, 1].

    Interpretation of the slope:
      slope = -0.5: RT decreases by ~39% from start to end
      slope = -1.0: RT decreases by ~63% from start to end
      slope = +0.3: RT increases by ~35% (fatigue-slowing, usually benign)

    Log-transform dampens the influence of RT outliers (right-skewed).
    Only negative slopes (speeding up) are flagged, since slowing down
    is typically legitimate fatigue rather than disengagement.
    """
    valid = [
        (r.item_position, r.response_time_ms)
        for r in items
        if r.response_time_ms > 0
    ]
    if len(valid) < 5:
        return CoherenceSignal(
            "fatigue_slope", 0.0, FATIGUE_SLOPE_THRESHOLD, False, 0.0,
            "Too few items for trend",
        )

    # Normalize positions to [0, 1]
    positions = [float(p) for p, _ in valid]
    min_pos = min(positions)
    max_pos = max(positions)
    pos_range = max_pos - min_pos
    if pos_range < 1.0:
        return CoherenceSignal(
            "fatigue_slope", 0.0, FATIGUE_SLOPE_THRESHOLD, False, 0.0,
            "No position range",
        )

    x = [(p - min_pos) / pos_range for p in positions]
    y = [math.log(max(1.0, rt)) for _, rt in valid]

    # OLS slope: beta = Cov(x,y) / Var(x)
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)

    if var_x < 1e-12:
        slope = 0.0
    else:
        slope = cov_xy / var_x

    flagged = slope < FATIGUE_SLOPE_THRESHOLD
    # Only negative slopes get severity (positive slope = slowing down = not flagged)
    severity = _ramp_severity_inverse(
        slope, 2.0 * FATIGUE_SLOPE_THRESHOLD, FATIGUE_SLOPE_THRESHOLD,
    )

    return CoherenceSignal(
        name="fatigue_slope",
        value=round(slope, 4),
        threshold=FATIGUE_SLOPE_THRESHOLD,
        flagged=flagged,
        severity=round(severity, 4),
        description=f"Log(RT) slope over session: {slope:.4f}",
    )


# ---------------------------------------------------------------------------
# Signal 6: Skip rate
# ---------------------------------------------------------------------------

def _skip_rate(all_items: List[ItemResponse]) -> CoherenceSignal:
    """Proportion of items skipped or declined.

    High skip rate suggests disengagement or refusal to engage with
    the assessment content.
    """
    if not all_items:
        return CoherenceSignal(
            "skip_rate", 0.0, SKIP_RATE_THRESHOLD, False, 0.0,
            "No items",
        )

    skipped = sum(1 for r in all_items if r.was_skipped)
    proportion = skipped / len(all_items)
    flagged = proportion > SKIP_RATE_THRESHOLD

    severity = _ramp_severity(proportion, SKIP_RATE_THRESHOLD, 1.0)

    return CoherenceSignal(
        name="skip_rate",
        value=round(proportion, 4),
        threshold=SKIP_RATE_THRESHOLD,
        flagged=flagged,
        severity=round(severity, 4),
        description=f"{skipped}/{len(all_items)} items skipped",
    )


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

def _ramp_severity(value: float, threshold: float, maximum: float) -> float:
    """Linear severity ramp: 0 at threshold, 1 at maximum.  Higher = worse.

    Returns 0.0 for values at or below threshold.
    Returns 1.0 for values at or above maximum.
    """
    if value <= threshold:
        return 0.0
    if maximum <= threshold:
        return 1.0 if value > threshold else 0.0
    return min(1.0, (value - threshold) / (maximum - threshold))


def _ramp_severity_inverse(
    value: float, minimum: float, threshold: float,
) -> float:
    """Inverse severity ramp: 1 at minimum, 0 at threshold.  Lower = worse.

    Returns 0.0 for values at or above threshold.
    Returns 1.0 for values at or below minimum.
    """
    if value >= threshold:
        return 0.0
    if threshold <= minimum:
        return 1.0 if value < threshold else 0.0
    return min(1.0, (threshold - value) / (threshold - minimum))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def coherence_to_dict(assessment: CoherenceAssessment) -> Dict[str, Any]:
    """Serialize CoherenceAssessment for JSON storage."""
    return {
        "signals": [
            {
                "name": s.name,
                "value": s.value,
                "threshold": s.threshold,
                "flagged": s.flagged,
                "severity": s.severity,
                "description": s.description,
            }
            for s in assessment.signals
        ],
        "n_flagged": assessment.n_flagged,
        "coherence_score": assessment.coherence_score,
        "tier": assessment.tier,
        "n_items": assessment.n_items,
        "version": assessment.version,
    }


def coherence_from_dict(d: Dict[str, Any]) -> CoherenceAssessment:
    """Deserialize CoherenceAssessment from JSON storage."""
    signals = tuple(
        CoherenceSignal(
            name=str(s["name"]),
            value=float(s["value"]),
            threshold=float(s["threshold"]),
            flagged=bool(s["flagged"]),
            severity=float(s["severity"]),
            description=str(s["description"]),
        )
        for s in d.get("signals", ())
    )
    return CoherenceAssessment(
        signals=signals,
        n_flagged=int(d["n_flagged"]),
        coherence_score=float(d["coherence_score"]),
        tier=str(d["tier"]),
        n_items=int(d["n_items"]),
        version=str(d.get("version", COHERENCE_VERSION)),
    )
