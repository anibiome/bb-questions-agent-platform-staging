from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from statistics import NormalDist as _NormalDist
from typing import Any, Dict, List, Optional, Set, Tuple

from questions_agent_platform.pipeline.baseline import BaselineState, baseline_std
from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.irt import (
    GRMItemParams,
    categories_for_response_type,
    default_item_params,
    grm_item_information,
)
from questions_agent_platform.pipeline.registry import Item, Registry

logger = logging.getLogger("questions_agent.selection")

DEFAULT_SESSION_TIMEFRAME = "last_7_days"
TIMEFRAME_PRIORITY: Tuple[str, ...] = ("today", "last_7_days", "last_30_days", "last_90_days")


@dataclass(frozen=True)
class Candidate:
    item_id: str
    item_type: str
    scale_ids: Tuple[str, ...]
    deterministic_score: float
    deterministic_rank: int
    constraint_tags: Tuple[str, ...]
    reason_codes: Tuple[str, ...]
    features: Dict[str, Any]


@dataclass(frozen=True)
class CandidateSet:
    date: str
    k_core: int
    extra_batch_size: int
    max_extra_batches: int
    mandatory_item_ids: Tuple[str, ...]
    candidates: Tuple[Candidate, ...]
    ordered_candidate_item_ids: Tuple[str, ...]
    excluded_item_ids: Tuple[str, ...]
    near_unlock_scales: Tuple[str, ...]
    due_retest_scales: Tuple[str, ...]
    drift_scales: Tuple[str, ...]
    timeframe: str = DEFAULT_SESSION_TIMEFRAME


@dataclass(frozen=True)
class SelectionPlan:
    date: str
    core_item_ids: Tuple[str, ...]
    extra_batches: Tuple[Tuple[str, ...], ...]
    reasons_by_item_id: Dict[str, Tuple[str, ...]]
    primary_scale_by_item_id: Dict[str, str]
    near_unlock_scales: Tuple[str, ...]
    due_retest_scales: Tuple[str, ...]
    drift_scales: Tuple[str, ...]
    timeframe: str = DEFAULT_SESSION_TIMEFRAME


def build_selection_plan(
    *,
    registry: Registry,
    cfg: QuestionsAgentConfig,
    user_id: str,
    day: date,
    answered_item_ids_by_scale: Dict[str, Set[str]],
    last_asked_date_by_item_id: Dict[str, date],
    last_score_date_by_scale_id: Dict[str, date],
    baselines_by_scale_id: Dict[str, BaselineState],
    rolling_normalized_by_scale_id: Dict[str, float],
    allowed_item_ids: Optional[Set[str]] = None,
    declined_item_ids: Optional[Set[str]] = None,
    priority_item_ids: Optional[Set[str]] = None,
    follow_up_item_ids: Optional[Set[str]] = None,
    force_timeframe: Optional[str] = None,
    previous_engagement_quality: Optional[str] = None,
    active_package_scale_ids: Optional[Set[str]] = None,
    active_package_scale_priorities: Optional[Dict[str, float]] = None,
    efficiency_scores: Optional[Dict[str, float]] = None,
) -> SelectionPlan:
    candidate_set, missing_item_ids_by_scale, primary_scale = build_candidate_set(
        registry=registry,
        cfg=cfg,
        user_id=user_id,
        day=day,
        answered_item_ids_by_scale=answered_item_ids_by_scale,
        last_asked_date_by_item_id=last_asked_date_by_item_id,
        last_score_date_by_scale_id=last_score_date_by_scale_id,
        baselines_by_scale_id=baselines_by_scale_id,
        rolling_normalized_by_scale_id=rolling_normalized_by_scale_id,
        allowed_item_ids=allowed_item_ids,
        declined_item_ids=declined_item_ids,
        priority_item_ids=priority_item_ids,
        follow_up_item_ids=follow_up_item_ids,
        force_timeframe=force_timeframe,
        previous_engagement_quality=previous_engagement_quality,
        active_package_scale_ids=active_package_scale_ids,
        active_package_scale_priorities=active_package_scale_priorities,
        efficiency_scores=efficiency_scores,
    )

    anchors = list(candidate_set.mandatory_item_ids)
    excluded = set(candidate_set.excluded_item_ids)
    ordered_candidates = list(candidate_set.ordered_candidate_item_ids)
    allowed_set = {str(i) for i in (allowed_item_ids or set())}
    declined_set = {str(i) for i in (declined_item_ids or set())}

    core: List[str] = []
    per_scale_count: Dict[str, int] = {}
    high_intrusiveness_count = 0

    def can_take(item_id: str) -> bool:
        ps = primary_scale.get(item_id)
        if not ps:
            return True
        return per_scale_count.get(ps, 0) < 2

    def is_allowed(item_id: str) -> bool:
        if item_id in declined_set:
            return False
        if allowed_set and item_id not in allowed_set:
            return False
        return True

    def can_take_intrusiveness(item_id: str) -> bool:
        nonlocal high_intrusiveness_count
        item = registry.items.get(item_id)
        if not item:
            return False
        if str(item.intrusiveness) == "high" and high_intrusiveness_count >= 1:
            return False
        return True

    def on_take(item_id: str) -> None:
        nonlocal high_intrusiveness_count
        item = registry.items.get(item_id)
        if item and str(item.intrusiveness) == "high":
            high_intrusiveness_count += 1

    for item_id in anchors:
        if len(core) >= int(cfg.core_questions_per_day):
            break
        if item_id in excluded or item_id not in registry.items:
            continue
        if not is_allowed(item_id):
            continue
        if item_id in core:
            continue
        if not can_take(item_id):
            continue
        if not can_take_intrusiveness(item_id):
            continue
        core.append(item_id)
        on_take(item_id)
        ps = primary_scale.get(item_id)
        if ps:
            per_scale_count[ps] = per_scale_count.get(ps, 0) + 1

    for item_id in ordered_candidates:
        if len(core) >= int(cfg.core_questions_per_day):
            break
        if item_id in core:
            continue
        if item_id not in registry.items:
            continue
        if not is_allowed(item_id):
            continue
        if not can_take(item_id):
            continue
        if not can_take_intrusiveness(item_id):
            continue
        core.append(item_id)
        on_take(item_id)
        ps = primary_scale.get(item_id)
        if ps:
            per_scale_count[ps] = per_scale_count.get(ps, 0) + 1

    if len(core) < int(cfg.core_questions_per_day):
        for item_id in _fallback_items(
            registry,
            last_asked_date_by_item_id,
            day,
            timeframe=candidate_set.timeframe,
            allowed_item_ids=allowed_item_ids,
            declined_item_ids=declined_item_ids,
        ):
            if len(core) >= int(cfg.core_questions_per_day):
                break
            if item_id in excluded or item_id in core:
                continue
            if not is_allowed(item_id):
                continue
            if not can_take_intrusiveness(item_id):
                continue
            core.append(item_id)
            on_take(item_id)

    core = _enforce_minimum_tag_diversity(
        registry=registry,
        selected_item_ids=core,
        ordered_candidates=ordered_candidates,
        excluded_item_ids=excluded,
        mandatory_item_ids=set(candidate_set.mandatory_item_ids),
        min_distinct_tags=2,
    )

    if len(core) < int(cfg.core_questions_per_day):
        logger.warning(
            "Undersized selection: got %d items, expected %d. "
            "User %s on %s may have too many declined/cooldown items.",
            len(core), int(cfg.core_questions_per_day), user_id, day.isoformat(),
        )

    already = set(core)
    extra_batches: List[List[str]] = []
    near_unlock_scales = set(candidate_set.near_unlock_scales)
    due_retest_scales = set(candidate_set.due_retest_scales)

    def can_append_extra(item_id: str) -> bool:
        if item_id in already or item_id in excluded or item_id not in registry.items:
            return False
        if not is_allowed(item_id):
            return False
        if not can_take_intrusiveness(item_id):
            return False
        return True

    def append_extra(batch: List[str], item_id: str) -> bool:
        if len(batch) >= int(cfg.extra_batch_size):
            return False
        if not can_append_extra(item_id):
            return False
        batch.append(item_id)
        already.add(item_id)
        on_take(item_id)
        return True

    for _ in range(int(cfg.extra_batches_max_per_day)):
        batch: List[str] = []

        for scale_id in sorted(near_unlock_scales):
            for item_id in missing_item_ids_by_scale.get(scale_id, []):
                if len(batch) >= int(cfg.extra_batch_size):
                    break
                append_extra(batch, item_id)
            if len(batch) >= int(cfg.extra_batch_size):
                break

        if len(batch) < int(cfg.extra_batch_size):
            for scale_id in sorted(due_retest_scales):
                scale = registry.scales[scale_id]
                for si in scale.items:
                    if len(batch) >= int(cfg.extra_batch_size):
                        break
                    append_extra(batch, si.item_id)
                if len(batch) >= int(cfg.extra_batch_size):
                    break

        if len(batch) < int(cfg.extra_batch_size):
            for item_id in ordered_candidates:
                if len(batch) >= int(cfg.extra_batch_size):
                    break
                append_extra(batch, item_id)

        if not batch:
            break
        extra_batches.append(batch)

    reasons_final = {c.item_id: c.reason_codes for c in candidate_set.candidates if c.reason_codes}

    logger.info(
        "selection.plan user=%s day=%s core=%d extras=%d near_unlock=%d drift=%d tf=%s",
        user_id, day.isoformat(), len(core), sum(len(b) for b in extra_batches),
        len(candidate_set.near_unlock_scales), len(candidate_set.drift_scales),
        candidate_set.timeframe,
    )

    return SelectionPlan(
        date=day.isoformat(),
        core_item_ids=tuple(core),
        extra_batches=tuple(tuple(b) for b in extra_batches),
        reasons_by_item_id=reasons_final,
        primary_scale_by_item_id=dict(primary_scale),
        near_unlock_scales=tuple(candidate_set.near_unlock_scales),
        due_retest_scales=tuple(candidate_set.due_retest_scales),
        drift_scales=tuple(candidate_set.drift_scales),
        timeframe=str(candidate_set.timeframe),
    )


def build_candidate_set(
    *,
    registry: Registry,
    cfg: QuestionsAgentConfig,
    user_id: str,
    day: date,
    answered_item_ids_by_scale: Dict[str, Set[str]],
    last_asked_date_by_item_id: Dict[str, date],
    last_score_date_by_scale_id: Dict[str, date],
    baselines_by_scale_id: Dict[str, BaselineState],
    rolling_normalized_by_scale_id: Dict[str, float],
    allowed_item_ids: Optional[Set[str]] = None,
    declined_item_ids: Optional[Set[str]] = None,
    priority_item_ids: Optional[Set[str]] = None,
    follow_up_item_ids: Optional[Set[str]] = None,
    force_timeframe: Optional[str] = None,
    previous_engagement_quality: Optional[str] = None,
    active_package_scale_ids: Optional[Set[str]] = None,
    active_package_scale_priorities: Optional[Dict[str, float]] = None,
    efficiency_scores: Optional[Dict[str, float]] = None,
) -> Tuple[CandidateSet, Dict[str, List[str]], Dict[str, str]]:
    """
    Builds the candidate set C for a user+day selection.

    The policy layer must only ever select from this set, and must honor constraints
    provided here (mandatory, blocked, override_allowed).

    Returns:
      (CandidateSet, missing_item_ids_by_scale, primary_scale_by_item_id)
    """
    allowed_set = {str(i) for i in (allowed_item_ids or set())}
    declined_set = {str(i) for i in (declined_item_ids or set())}
    priority_set = {str(i) for i in (priority_item_ids or set())}
    follow_up_set = {str(i) for i in (follow_up_item_ids or set())}

    def is_allowed(item_id: str) -> bool:
        if item_id in declined_set:
            return False
        if allowed_set and item_id not in allowed_set:
            return False
        return True

    # Precompute multiplex counts: item_id -> number of scales that include it
    scales_by_item: Dict[str, List[str]] = {}
    for scale in registry.scales.values():
        for si in scale.items:
            scales_by_item.setdefault(si.item_id, []).append(scale.id)
    multiplex_count = {item_id: len(scales) for item_id, scales in scales_by_item.items()}

    near_unlock_scales: Set[str] = set()
    due_retest_scales: Set[str] = set()
    drift_scales: Set[str] = set()

    missing_item_ids_by_scale: Dict[str, List[str]] = {}
    missing_count_by_scale: Dict[str, int] = {}
    for scale in registry.scales.values():
        answered = answered_item_ids_by_scale.get(scale.id, set())
        required_items = [si.item_id for si in scale.items]
        answered_count = len(set(required_items) & set(answered))
        missing = max(0, scale.min_items_required - answered_count)
        missing_items = [item_id for item_id in required_items if item_id not in answered]
        missing_item_ids_by_scale[scale.id] = missing_items
        missing_count_by_scale[scale.id] = missing

        last_score = last_score_date_by_scale_id.get(scale.id)
        is_due_retest = False
        if last_score is not None and (day - last_score).days >= int(scale.retest_interval_days):
            is_due_retest = True
            due_retest_scales.add(scale.id)

        if 0 < missing <= 2 and (last_score is None or is_due_retest):
            near_unlock_scales.add(scale.id)

        baseline = baselines_by_scale_id.get(scale.id)
        rolling = rolling_normalized_by_scale_id.get(scale.id)
        if baseline and rolling is not None:
            # Burn-in gate: require >=5 observations before drift detection
            if baseline.n < 5:
                continue
            std = baseline_std(baseline)
            # Bonferroni correction: adjust z-threshold for number of scales tested
            n_scales_tested = max(1, sum(
                1 for s in registry.scales.values()
                if baselines_by_scale_id.get(s.id) is not None
                and baselines_by_scale_id[s.id].n >= 5
                and rolling_normalized_by_scale_id.get(s.id) is not None
            ))
            # Bonferroni-adjusted threshold: z for alpha=0.05/n_scales (two-tailed)
            bonferroni_alpha = 0.05 / n_scales_tested
            z_threshold = _NormalDist().inv_cdf(1.0 - bonferroni_alpha / 2.0)

            if std >= 1.0:
                z = (rolling - baseline.mean) / std
                if abs(z) >= z_threshold:
                    drift_scales.add(scale.id)
            else:
                # Reliable Change Index (Jacobson-Truax): RCI = delta / SE_diff
                # SE_diff = std * sqrt(2) for EWMA (using last known std as proxy)
                # When std is near-zero, use absolute threshold scaled to score range
                score_range = float(scale.normalize_max - scale.normalize_min) or 100.0
                # RCI fallback: flag if absolute change > 10% of scale range
                rci_threshold = max(5.0, score_range * 0.10)
                if abs(rolling - baseline.mean) >= rci_threshold:
                    drift_scales.add(scale.id)

    candidate_score: Dict[str, float] = {}
    reasons_by_item: Dict[str, List[str]] = {}
    primary_scale: Dict[str, str] = {}

    def add_reason(item_id: str, score: float, reason: str, scale_id: Optional[str] = None) -> None:
        if item_id not in registry.items:
            return
        if not is_allowed(item_id):
            return
        candidate_score[item_id] = candidate_score.get(item_id, 0.0) + float(score)
        reasons_by_item.setdefault(item_id, []).append(reason)
        if scale_id and item_id not in primary_scale:
            primary_scale[item_id] = scale_id

    for scale_id in near_unlock_scales:
        for item_id in missing_item_ids_by_scale.get(scale_id, []):
            add_reason(item_id, 4.0, f"near_unlock:{scale_id}", scale_id=scale_id)

    for scale_id in due_retest_scales:
        scale = registry.scales[scale_id]
        answered = answered_item_ids_by_scale.get(scale_id, set())
        for si in scale.items:
            bump = 3.0 if si.item_id not in answered else 1.0
            add_reason(si.item_id, bump, f"retest_due:{scale_id}", scale_id=scale_id)

    for scale_id in drift_scales:
        scale = registry.scales[scale_id]
        for si in scale.items:
            add_reason(si.item_id, 3.0, f"drift_probe:{scale_id}", scale_id=scale_id)

    for item_id, count in multiplex_count.items():
        if count <= 1:
            continue
        add_reason(item_id, 0.5 * float(count - 1), "multiplex", scale_id=None)

    # --- IRT information-gain scoring (Fisher information at current θ̂) ---
    # For each candidate item that already has a clinical reason (near-unlock,
    # retest, drift, multiplex), compute its total Fisher information across
    # ALL scales that use it.  Clinical priorities determine WHICH domains to
    # measure; IRT information gain selects the MOST INFORMATIVE items within
    # those domains.  This is the multiplexed active sensing innovation.
    _irt_info_scores = _compute_irt_information_scores(
        registry=registry,
        scales_by_item=scales_by_item,
        baselines_by_scale_id=baselines_by_scale_id,
    )
    for item_id, info_score in _irt_info_scores.items():
        # Only boost items that already have at least one clinical reason
        if item_id in candidate_score and info_score > 0.01:
            add_reason(item_id, info_score, "irt_information_gain", scale_id=None)

    # --- Active measurement package focus (Progressive Plan) ---
    # Items belonging to scales in the currently active measurement
    # package receive a strong priority boost.  This is the mechanism
    # that makes the progressive measurement funnel actually constrain
    # selection: each session focuses on the scales needed to complete
    # the current package before advancing to the next.
    #
    # Design decision: *soft boost, not hard filter*.  Clinical signals
    # (drift probes, anamnesis follow-ups, near-unlock) can still win.
    # The boost is high enough (default 5.0) to dominate over generic
    # signals (multiplex ~0.5, retest ~3.0) but below onboarding
    # priority (50-400).
    _pkg_scale_ids = set(active_package_scale_ids or set())
    _pkg_scale_priorities = active_package_scale_priorities or {}
    if _pkg_scale_ids:
        base_boost = float(cfg.package_focus_boost)
        for item_id, item_scales in scales_by_item.items():
            # Check which of this item's scales overlap with the active package
            matching_scales = [s for s in item_scales if s in _pkg_scale_ids]
            if not matching_scales:
                continue
            # Boost = base × max priority among matching scales.
            # A scale with priority 1.0 gets the full boost; 0.7 gets 70%.
            max_priority = max(
                float(_pkg_scale_priorities.get(s, 1.0)) for s in matching_scales
            )
            pkg_boost = round(base_boost * max_priority, 4)
            # Pick the highest-priority matching scale for primary_scale tracking
            best_scale = max(matching_scales, key=lambda s: float(_pkg_scale_priorities.get(s, 1.0)))
            add_reason(item_id, pkg_boost, f"package_focus:{best_scale}", scale_id=best_scale)

    def _onboarding_sort_key(item_id: str) -> Tuple[int, str]:
        item = registry.items.get(item_id)
        order = item.onboarding_order if item and item.onboarding_order is not None else 10_000
        return (int(order), str(item_id))

    for item_id in sorted(priority_set, key=_onboarding_sort_key):
        item = registry.items.get(item_id)
        order = int(item.onboarding_order) if item and item.onboarding_order is not None else 10_000
        priority_bonus = max(50.0, 400.0 - min(float(order), 350.0))
        add_reason(item_id, priority_bonus, f"onboarding_priority:o{order}", scale_id=None)

    for item_id in sorted(follow_up_set):
        add_reason(item_id, 18.0, "anamnesis_followup", scale_id=primary_scale.get(item_id))

    # --- Engagement quality feedback (Claim Family 5) ---
    # If previous session was distracted/fatigued, adjust item preferences
    if previous_engagement_quality and previous_engagement_quality in ("distracted", "fatigued"):
        for item_id in list(candidate_score.keys()):
            item = registry.items.get(item_id)
            if not item:
                continue
            resp_type = item.response_type
            if previous_engagement_quality == "distracted":
                # Prefer simpler items for distracted users
                simple_types = {"bool_0_1", "sex_female_male", "age_band_0_4"}
                complex_types = {"minutes_0_180", "minutes_0_960", "family_diabetes_points_0_5"}
                if resp_type in simple_types:
                    candidate_score[item_id] += 0.3
                    reasons_by_item.setdefault(item_id, []).append("engagement_bonus_simple")
                elif resp_type in complex_types:
                    candidate_score[item_id] -= 0.3
                    reasons_by_item.setdefault(item_id, []).append("engagement_penalty_complex")
            elif previous_engagement_quality == "fatigued":
                candidate_score[item_id] -= 0.1
                reasons_by_item.setdefault(item_id, []).append("engagement_penalty_fatigued")

    # --- Shadow efficiency scoring (Item Efficiency Analyzer) ---
    # Applies a soft penalty to items flagged as redundant/low-efficiency by
    # the cross-instrument concordance analyzer. This does NOT block items —
    # it nudges selection toward more informative items when alternatives exist.
    # Clinical priorities (near_unlock, drift, retest, anamnesis) always override.
    if efficiency_scores is not None:
        _EFFICIENCY_PENALTY_MAX = 1.5  # maximum penalty for lowest-efficiency items
        _EFFICIENCY_THRESHOLD = 0.40   # only penalize items below this threshold
        n_penalized = 0
        for item_id, eff_score in efficiency_scores.items():
            if item_id not in candidate_score:
                continue
            if eff_score >= _EFFICIENCY_THRESHOLD:
                continue
            # Skip items with clinical priority — those must be measured regardless
            has_clinical = any(
                r.startswith("near_unlock:")
                or r.startswith("retest_due:")
                or r.startswith("drift_probe:")
                or r.startswith("anamnesis_followup")
                for r in reasons_by_item.get(item_id, [])
            )
            if has_clinical:
                continue
            # Penalty proportional to how far below threshold: 0.0 → full penalty, 0.40 → no penalty
            penalty = _EFFICIENCY_PENALTY_MAX * (1.0 - eff_score / _EFFICIENCY_THRESHOLD)
            candidate_score[item_id] -= penalty
            reasons_by_item.setdefault(item_id, []).append(
                f"efficiency_penalty:{eff_score:.2f}"
            )
            n_penalized += 1
        if n_penalized > 0:
            logger.info("efficiency_shadow: penalized %d/%d items below threshold %.2f",
                        n_penalized, len(efficiency_scores), _EFFICIENCY_THRESHOLD)

    # Anchors are mandatory in the candidate set contract (can be changed via config later).
    if priority_set:
        anchors = [item_id for item_id in sorted(priority_set, key=_onboarding_sort_key)[:2]]
    else:
        anchors = _choose_anchor_items(registry=registry, multiplex_count=multiplex_count, k=2)
    anchors = [item_id for item_id in anchors if is_allowed(item_id)]
    for item_id in anchors:
        add_reason(item_id, 2.0, "anchor_continuity", scale_id=None)

    session_timeframe = _choose_session_timeframe(
        registry=registry,
        candidate_item_ids=set(candidate_score.keys()) | set(anchors),
        force_timeframe=force_timeframe,
    )

    excluded: Set[str] = set()
    for item_id in list(candidate_score.keys()):
        item = registry.items.get(item_id)
        if not item:
            continue
        if session_timeframe not in set(item.timeframes_allowed):
            excluded.add(item_id)
            reasons_by_item.setdefault(item_id, []).append("timeframe_mismatch")

    anchors = [
        item_id
        for item_id in anchors
        if item_id in registry.items and session_timeframe in set(registry.items[item_id].timeframes_allowed)
    ]
    cooldown = int(cfg.item_repeat_cooldown_days)
    override_allowed: Set[str] = set()
    anchor_set: Set[str] = set(anchors)
    for item_id, last_asked in last_asked_date_by_item_id.items():
        if item_id not in candidate_score:
            continue
        if (day - last_asked).days < cooldown:
            has_priority = any(
                r.startswith("near_unlock:")
                or r.startswith("retest_due:")
                or r.startswith("drift_probe:")
                or r.startswith("onboarding_priority")
                or r.startswith("anamnesis_followup")
                or r.startswith("package_focus:")
                for r in reasons_by_item.get(item_id, [])
            )
            if item_id in anchor_set:
                # Anchors remain feasible by contract. Keep them selectable and
                # apply only a soft cooldown penalty when not already prioritized.
                override_allowed.add(item_id)
                if not has_priority:
                    candidate_score[item_id] -= 1.0
                    reasons_by_item.setdefault(item_id, []).append("cooldown_penalty_anchor")
                continue
            if not has_priority:
                excluded.add(item_id)
            else:
                override_allowed.add(item_id)
                candidate_score[item_id] -= 2.0
                reasons_by_item.setdefault(item_id, []).append("cooldown_penalty")

    # Keep the candidate set feasible (>= k_core non-blocked items).
    required_core = int(cfg.core_questions_per_day)
    available_count = sum(1 for item_id in candidate_score.keys() if item_id not in excluded)
    if available_count < required_core:
        blocked_ranked = sorted(
            [item_id for item_id in excluded if item_id in candidate_score],
            key=lambda i: (candidate_score.get(i, 0.0), multiplex_count.get(i, 0), i),
            reverse=True,
        )
        for item_id in blocked_ranked:
            excluded.discard(item_id)
            override_allowed.add(item_id)
            reasons_by_item.setdefault(item_id, []).append("cooldown_relaxed_for_coverage")
            available_count += 1
            if available_count >= required_core:
                break

    ordered_candidates = sorted(
        [item_id for item_id in candidate_score.keys() if item_id not in excluded],
        key=lambda i: (candidate_score.get(i, 0.0), multiplex_count.get(i, 0), i),
        reverse=True,
    )

    # Build candidate objects (include excluded items as blocked for auditability)
    all_candidate_ids: Set[str] = (
        set(candidate_score.keys())
        | set(excluded)
        | set(anchors)
        | ({i for i in declined_set if i in registry.items})
    )
    ranked_ids = ordered_candidates + [i for i in sorted(all_candidate_ids) if i not in ordered_candidates]
    rank_by_id = {item_id: idx for idx, item_id in enumerate(ranked_ids, start=1)}

    candidates_out: List[Candidate] = []
    for item_id in ranked_ids:
        if item_id not in registry.items:
            continue
        reasons = list(reasons_by_item.get(item_id, []))
        if item_id in declined_set and "permanently_declined" not in reasons:
            reasons.append("permanently_declined")
        scale_ids = tuple(sorted(scales_by_item.get(item_id, [])))
        missing_to_unlock = None
        if scale_ids:
            missing_to_unlock = min(int(missing_count_by_scale.get(s, 9999)) for s in scale_ids)

        # Heuristic burden: all demo items are Likert; keep as simple numeric.
        expected_burden = 1.0
        item_obj = registry.items[item_id]
        sensitivity = item_obj.sensitivity

        last_asked = last_asked_date_by_item_id.get(item_id)
        novelty_days = 3650 if last_asked is None else max(0, (day - last_asked).days)

        item_type = "other"
        if item_id in anchors:
            item_type = "anchor"
        elif any(r.startswith("anamnesis_followup") for r in reasons):
            item_type = "follow_up"
        elif any(r.startswith("near_unlock:") for r in reasons):
            item_type = "unlock_item"
        elif any(r.startswith("retest_due:") for r in reasons):
            item_type = "retest_item"
        elif any(r.startswith("drift_probe:") for r in reasons):
            item_type = "drift_probe"

        tags: List[str] = []
        if item_id in anchors:
            tags.append("mandatory")
        else:
            tags.append("optional")

        if item_id in excluded:
            tags.append("blocked")
        if item_id in override_allowed:
            tags.append("override_allowed")
        if item_id in declined_set:
            tags.append("blocked")
            tags.append("permanently_declined")

        candidates_out.append(
            Candidate(
                item_id=item_id,
                item_type=item_type,
                scale_ids=scale_ids,
                deterministic_score=float(candidate_score.get(item_id, 0.0)),
                deterministic_rank=int(rank_by_id.get(item_id, 999999)),
                constraint_tags=tuple(tags),
                reason_codes=tuple(reasons),
                features={
                    "multiplex_count": int(multiplex_count.get(item_id, 0)),
                    "novelty_days": int(novelty_days),
                    "missing_to_unlock": int(missing_to_unlock) if missing_to_unlock is not None else None,
                    "is_due_retest": bool(any(s in due_retest_scales for s in scale_ids)),
                    "is_drift_probe": bool(any(s in drift_scales for s in scale_ids)),
                    "is_follow_up": bool(any(r.startswith("anamnesis_followup") for r in reasons)),
                    "is_package_focus": bool(any(r.startswith("package_focus:") for r in reasons)),
                    "expected_burden": float(expected_burden),
                    "sensitivity": str(sensitivity),
                    "intrusiveness": str(item_obj.intrusiveness),
                    "timeframes_allowed": list(item_obj.timeframes_allowed),
                    "irt_information_gain": float(_irt_info_scores.get(item_id, 0.0)),
                    "efficiency_score": float(efficiency_scores.get(item_id, 1.0)) if efficiency_scores else None,
                },
            )
        )

    candidate_set = CandidateSet(
        date=day.isoformat(),
        k_core=int(cfg.core_questions_per_day),
        extra_batch_size=int(cfg.extra_batch_size),
        max_extra_batches=int(cfg.extra_batches_max_per_day),
        mandatory_item_ids=tuple(anchors),
        candidates=tuple(candidates_out),
        ordered_candidate_item_ids=tuple(ordered_candidates),
        excluded_item_ids=tuple(sorted(excluded)),
        near_unlock_scales=tuple(sorted(near_unlock_scales)),
        due_retest_scales=tuple(sorted(due_retest_scales)),
        drift_scales=tuple(sorted(drift_scales)),
        timeframe=str(session_timeframe),
    )
    return candidate_set, missing_item_ids_by_scale, dict(primary_scale)


def _choose_anchor_items(*, registry: Registry, multiplex_count: Dict[str, int], k: int) -> List[str]:
    # Prefer low-sensitivity items tagged for general state continuity.
    # These tags are configurable via the registry; defaults cover common domains.
    preferred_tags = {"mood", "sleep", "energy", "stress", "metabolic", "cardiovascular"}

    def score(item: Item) -> Tuple[int, int, str]:
        return (
            int(multiplex_count.get(item.id, 0)),
            1 if any(t in preferred_tags for t in item.tags) else 0,
            item.id,
        )

    items = [it for it in registry.items.values() if it.sensitivity in ("low", "medium")]
    items_sorted = sorted(items, key=score, reverse=True)
    return [it.id for it in items_sorted[: max(0, int(k))]]


def _fallback_items(
    registry: Registry,
    last_asked: Dict[str, date],
    day: date,
    *,
    timeframe: str,
    allowed_item_ids: Optional[Set[str]] = None,
    declined_item_ids: Optional[Set[str]] = None,
) -> List[str]:
    # Prefer never-asked, low-sensitivity items first.
    allowed_set = {str(i) for i in (allowed_item_ids or set())}
    declined_set = {str(i) for i in (declined_item_ids or set())}
    candidates = []
    for item in registry.items.values():
        if item.sensitivity not in ("low", "medium"):
            continue
        if timeframe not in set(item.timeframes_allowed):
            continue
        if item.id in declined_set:
            continue
        if allowed_set and item.id not in allowed_set:
            continue
        last = last_asked.get(item.id)
        days_ago = 10_000 if last is None else (day - last).days
        candidates.append((days_ago, item.id))
    candidates.sort(reverse=True)
    return [item_id for _, item_id in candidates]


def _choose_session_timeframe(
    *,
    registry: Registry,
    candidate_item_ids: Set[str],
    force_timeframe: Optional[str] = None,
) -> str:
    if force_timeframe:
        return str(force_timeframe)
    if not candidate_item_ids:
        return DEFAULT_SESSION_TIMEFRAME
    by_tf: Dict[str, int] = {tf: 0 for tf in TIMEFRAME_PRIORITY}
    for item_id in candidate_item_ids:
        item = registry.items.get(item_id)
        if not item:
            continue
        for tf in item.timeframes_allowed:
            if tf not in by_tf:
                by_tf[tf] = 0
            by_tf[tf] += 1
    if not by_tf:
        return DEFAULT_SESSION_TIMEFRAME
    return max(by_tf.keys(), key=lambda tf: (int(by_tf.get(tf, 0)), -TIMEFRAME_PRIORITY.index(tf) if tf in TIMEFRAME_PRIORITY else -999))


def _enforce_minimum_tag_diversity(
    *,
    registry: Registry,
    selected_item_ids: List[str],
    ordered_candidates: List[str],
    excluded_item_ids: Set[str],
    mandatory_item_ids: Set[str],
    min_distinct_tags: int,
) -> List[str]:
    if min_distinct_tags <= 1:
        return list(selected_item_ids)
    selected = list(selected_item_ids)
    tags: Set[str] = set()
    for item_id in selected:
        tags.update(registry.items[item_id].tags)
    if len(tags) >= min_distinct_tags:
        return selected

    for candidate_id in ordered_candidates:
        if candidate_id in selected:
            continue
        if candidate_id in excluded_item_ids:
            continue
        candidate_tags = set(registry.items[candidate_id].tags)
        if not candidate_tags:
            continue
        if len(tags | candidate_tags) < min_distinct_tags:
            continue
        replace_idx = None
        for idx in range(len(selected) - 1, -1, -1):
            if selected[idx] not in mandatory_item_ids:
                replace_idx = idx
                break
        if replace_idx is not None:
            selected[replace_idx] = candidate_id
        elif not selected:
            selected.append(candidate_id)
        else:
            continue
        break
    return selected


# ---------------------------------------------------------------------------
# IRT information-gain scoring for multiplexed active sensing
# ---------------------------------------------------------------------------

def _compute_irt_information_scores(
    *,
    registry: Registry,
    scales_by_item: Dict[str, List[str]],
    baselines_by_scale_id: Dict[str, BaselineState],
) -> Dict[str, float]:
    """
    Compute a multiplexed Fisher information score for each candidate item.

    For each item, we sum its GRM Fisher information across ALL scales that
    use it, evaluated at the user's current θ̂ per scale (derived from the
    EWMA baseline via a linear mapping from [0-100] normalized to [-3, 3] θ).

    This is the core of cross-instrument multiplexed active sensing:
    a single question answer simultaneously reduces measurement uncertainty
    across multiple latent constructs.

    Returns {item_id: information_gain_score} where the score is scaled to
    be comparable to other heuristic scores (0-5 range typically).
    """
    # Cache: scale_id -> (theta_hat, GRM item params by item_id)
    scale_irt_cache: Dict[str, Tuple[float, Dict[str, GRMItemParams]]] = {}

    for scale in registry.scales.values():
        baseline = baselines_by_scale_id.get(scale.id)
        if baseline is None or baseline.n < 2:
            # No baseline yet — use prior mean θ=0
            theta_hat = 0.0
        else:
            # Map normalized score [0, 100] → θ [-3, 3]
            theta_hat = (baseline.mean - 50.0) / 50.0 * 3.0
            theta_hat = max(-3.5, min(3.5, theta_hat))

        n_cat = categories_for_response_type(scale.response_type)
        item_params: Dict[str, GRMItemParams] = {}
        for si in scale.items:
            ip = default_item_params(
                si.item_id, n_cat, weight=si.weight, base_discrimination=1.0,
            )
            item_params[si.item_id] = ip
        scale_irt_cache[scale.id] = (theta_hat, item_params)

    # Now compute multiplexed information for each item
    info_by_item: Dict[str, float] = {}
    for item_id, scale_ids in scales_by_item.items():
        total_info = 0.0
        for sid in scale_ids:
            entry = scale_irt_cache.get(sid)
            if entry is None:
                continue
            theta_hat, item_params = entry
            ip = item_params.get(item_id)
            if ip is None:
                continue
            total_info += grm_item_information(theta_hat, ip)

        if total_info > 0.01:
            # Scale to act as a tiebreaker within the heuristic ranking,
            # not to override clinical signals (near_unlock=4, drift=3, etc.).
            # Raw Fisher info for a typical likert item at a=1.0 is ~0.5-0.8.
            # Single-scale item → ~0.3-0.5, multiplexed (3 scales) → ~1.0-1.5.
            info_by_item[item_id] = min(3.0, total_info * 0.6)

    return info_by_item
