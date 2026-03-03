from __future__ import annotations

import json
import hashlib
import uuid
from statistics import median
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sqlalchemy import and_, func, or_, select, text, update
from sqlalchemy.orm import Session

from questions_agent_platform.pipeline.anamnesis import (
    AnamnesisEpisode,
    create_anamnesis_episode,
    evaluate_drift_triggers,
    evaluate_episode_resolution,
    run_anamnesis_check,
)
from questions_agent_platform.pipeline.baseline import BaselineState, baseline_std, update_ewma_baseline
from questions_agent_platform.pipeline.drift_routing import load_drift_routes, resolve_drift_route, route_contract
from questions_agent_platform.pipeline.governance import (
    CARDIOMETABOLIC_DOMAIN,
    EMOTION_TAGS,
    canonical_domain_id,
    encode_json_str_list,
    parse_json_str_list,
    registry_domains,
    registry_item_domains,
    registry_scale_domains,
    sync_user_domain_state,
)
from questions_agent_platform.pipeline.registry import Registry, load_registry, save_registry, validate_registry_bundle
from questions_agent_platform.pipeline.response_metadata import (
    ResponseMetadata,
    SessionUncertaintyProfile,
    UncertaintyModifier,
    adjust_se_with_metadata,
    compute_session_uncertainty_profile,
    compute_uncertainty_modifier,
)
from questions_agent_platform.pipeline.response_types import get_response_type
from questions_agent_platform.pipeline.site_config import SiteConfig, default_site_registry
from questions_agent_platform.pipeline.scoring import (
    compute_scale_progress,
    compute_scale_score,
    confidence_tier,
    infer_risk_tier_from_scale,
)
from questions_agent_platform.pipeline.selection import build_candidate_set, build_selection_plan
from questions_agent_platform.pipeline.session_helpers import (
    POLICY_SELECTION_MODES,
    allowed_item_ids_for_domains,
    build_selection_explain_payload,
    evaluate_onboarding_cardiometabolic_gate,
    filter_extra_batches,
    merge_allowed_item_ids,
    normalize_selection_mode,
    policy_runtime_mode,
)
from questions_agent_platform.policy.runtime import execute_policy_selection
from questions_agent_platform.pipeline.state_snapshots import (
    CIRCLE_ANCHOR_VERSION,
    CIRCLE_PROJECTION_VERSION,
    STATE_MODEL_VERSION,
    STATE_SCHEMA_VERSION_DEFAULT,
    compute_circle_snapshot,
    compute_state_snapshot,
)
from questions_agent_platform.pipeline.trajectory_metrics import (
    build_uncertainty_coupling,
    coherence_tier_contract,
    coherence_tier_for_score,
    coherence_score_from_radius,
    confidence_to_quality,
    compute_ews_features,
)
from questions_agent_platform.pipeline.time_utils import date_to_start_iso, now_iso, parse_date

from questions_agent_platform.prod.models import (
    AnamnesisEpisodeRow,
    AnswerEvent,
    CircleSnapshot,
    DailySession,
    DriftEvent,
    EwsFeature,
    FollowUpQueueItem,
    ObservationEvent,
    OperationalMetric,
    PolicyDecision as PolicyDecisionRow,
    PolicyOutcome as PolicyOutcomeRow,
    ResponseMetadataRow,
    SafetyEvent,
    ScaleEvidence,
    RegistryVersion,
    SessionUncertaintyProfileRow,
    StateSnapshot,
    ScaleBaseline,
    ScaleScore,
    Experiment,
    User,
    UserProfile,
)


class SnapshotWriteError(RuntimeError):
    pass


def ensure_user(session: Session, user_id: str) -> None:
    if session.get(User, user_id):
        return
    session.add(User(user_id=user_id, created_at=now_iso()))


def ensure_user_profile(session: Session, user_id: str) -> None:
    row = session.get(UserProfile, user_id)
    if row:
        return
    session.add(
        UserProfile(
            user_id=user_id,
            mode="consumer",
            site_config_id="consumer",
            permanently_declined_items_json="[]",
            active_domains_json='["cardiometabolic"]',
            queued_domains_json="[]",
            promoted_domains_json='["cardiometabolic"]',
            onboarding_complete=False,
            state_schema_version="v1_state_schema",
            updated_at=now_iso(),
        )
    )


def get_user_profile(session: Session, user_id: str) -> Dict[str, Any]:
    ensure_user_profile(session, user_id)
    row = session.get(UserProfile, user_id)
    if not row:
        return {
            "mode": "consumer",
            "permanently_declined_item_ids": [],
            "onboarding_complete": False,
            "state_schema_version": "v1_state_schema",
        }
    declined: List[str] = []
    try:
        raw = json.loads(str(row.permanently_declined_items_json or "[]"))
        if isinstance(raw, list):
            declined = [str(x) for x in raw]
    except Exception:
        declined = []
    return {
        "mode": str(row.mode or "consumer"),
        "site_config_id": str(getattr(row, "site_config_id", None) or "consumer"),
        "permanently_declined_item_ids": declined,
        "active_domains": parse_json_str_list(row.active_domains_json, fallback=[CARDIOMETABOLIC_DOMAIN]),
        "queued_domains": parse_json_str_list(row.queued_domains_json, fallback=[]),
        "promoted_domains": parse_json_str_list(row.promoted_domains_json, fallback=[CARDIOMETABOLIC_DOMAIN]),
        "onboarding_complete": bool(row.onboarding_complete),
        "state_schema_version": str(row.state_schema_version or "v1_state_schema"),
    }


def upsert_user_profile(
    session: Session,
    *,
    user_id: str,
    patch: Dict[str, Any],
    registry: Registry,
    max_active_new_domains: int = 2,
) -> Dict[str, Any]:
    ensure_user(session, user_id)
    ensure_user_profile(session, user_id)
    profile = get_user_profile(session, user_id)

    mode = str(patch.get("mode") or profile.get("mode") or "consumer").strip().lower()
    if mode not in {"trial", "consumer"}:
        raise ValueError("mode must be one of: trial, consumer")

    active_domains = profile.get("active_domains") or [CARDIOMETABOLIC_DOMAIN]
    queued_domains = profile.get("queued_domains") or []
    promoted_domains = profile.get("promoted_domains") or [CARDIOMETABOLIC_DOMAIN]
    onboarding_complete = bool(profile.get("onboarding_complete", False))
    if "onboarding_complete" in patch:
        onboarding_complete = bool(patch.get("onboarding_complete"))
    if isinstance(patch.get("active_domains"), list):
        active_domains = [canonical_domain_id(str(x)) for x in patch.get("active_domains") or []]
    if isinstance(patch.get("queued_domains"), list):
        queued_domains = [canonical_domain_id(str(x)) for x in patch.get("queued_domains") or []]
    if isinstance(patch.get("promoted_domains"), list):
        promoted_domains = [canonical_domain_id(str(x)) for x in patch.get("promoted_domains") or []]

    # Site config
    site_config_id = str(
        patch.get("site_config_id") or profile.get("site_config_id") or "consumer"
    ).strip()

    row = session.get(UserProfile, user_id)
    if not row:
        raise ValueError("Unknown user profile")
    row.mode = mode
    row.site_config_id = site_config_id
    row.onboarding_complete = bool(onboarding_complete)
    row.active_domains_json = encode_json_str_list(active_domains)
    row.queued_domains_json = encode_json_str_list(queued_domains)
    row.promoted_domains_json = encode_json_str_list(promoted_domains)
    row.updated_at = now_iso()
    session.flush()

    refreshed = get_user_profile(session, user_id)
    return _sync_user_domain_profile_pg(
        session,
        user_id=user_id,
        profile=refreshed,
        registry=registry,
        max_active_new_domains=max_active_new_domains,
    )


def list_safety_events(
    session: Session,
    *,
    user_id: str,
    day: date,
    lookback_days: int,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    start = day - timedelta(days=max(1, int(lookback_days)) - 1)
    conditions = [
        SafetyEvent.user_id == user_id,
        SafetyEvent.date >= start,
        SafetyEvent.date <= day,
    ]
    if status:
        conditions.append(SafetyEvent.status == str(status))
    rows = (
        session.execute(
            select(SafetyEvent)
            .where(and_(*conditions))
            .order_by(SafetyEvent.date.desc(), SafetyEvent.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [_row_to_safety_event(r) for r in rows]


def resolve_safety_event(
    session: Session,
    *,
    user_id: str,
    event_id: str,
    resolved_by: str,
    resolved_reason: str,
) -> Dict[str, Any]:
    row = session.get(SafetyEvent, event_id)
    if not row:
        raise ValueError("Unknown safety event")
    if str(row.user_id) != str(user_id):
        raise ValueError("safety event does not belong to user")
    if str(row.status) != "open":
        return {"ok": True, "event_id": event_id, "status": str(row.status)}
    row.status = "resolved"
    row.resolved_at = now_iso()
    row.resolved_by = str(resolved_by)
    row.resolved_reason = str(resolved_reason)
    return {"ok": True, "event_id": event_id, "status": "resolved"}


def _row_to_safety_event(row: SafetyEvent) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    try:
        parsed = json.loads(str(row.details_json or "{}"))
        if isinstance(parsed, dict):
            details = parsed
    except Exception:
        details = {}
    return {
        "event_id": str(row.event_id),
        "user_id": str(row.user_id),
        "session_id": str(row.session_id) if row.session_id else None,
        "date": row.date.isoformat(),
        "trigger_source": str(row.trigger_source),
        "severity": str(row.severity),
        "status": str(row.status),
        "item_id": str(row.item_id) if row.item_id else None,
        "reason_code": str(row.reason_code),
        "details": details,
        "created_at": str(row.created_at),
        "resolved_at": str(row.resolved_at) if row.resolved_at else None,
        "resolved_by": str(row.resolved_by) if row.resolved_by else None,
        "resolved_reason": str(row.resolved_reason) if row.resolved_reason else None,
    }


def mark_previous_day_incomplete_sessions_abandoned(session: Session, *, user_id: str, day: date) -> None:
    session.execute(
        text(
            """
            UPDATE daily_sessions
            SET status='abandoned'
            WHERE user_id=:user_id
              AND date<:day
              AND status IN ('created','started');
            """
        ),
        {"user_id": user_id, "day": day},
    )


def _sync_user_domain_profile_pg(
    session: Session,
    *,
    user_id: str,
    profile: Dict[str, Any],
    registry: Registry,
    max_active_new_domains: int,
) -> Dict[str, Any]:
    mode = str(profile.get("mode") or "consumer")
    onboarding_complete = bool(profile.get("onboarding_complete", False))
    active, queued, promoted, allowed_domains = sync_user_domain_state(
        mode=mode,
        onboarding_complete=onboarding_complete,
        active_domains=profile.get("active_domains") or [CARDIOMETABOLIC_DOMAIN],
        queued_domains=profile.get("queued_domains") or [],
        promoted_domains=profile.get("promoted_domains") or [CARDIOMETABOLIC_DOMAIN],
        registry_domains_all=registry_domains(registry),
        max_active_new_domains=int(max_active_new_domains),
    )
    row = session.get(UserProfile, user_id)
    if row:
        row.active_domains_json = encode_json_str_list(active)
        row.queued_domains_json = encode_json_str_list(queued)
        row.promoted_domains_json = encode_json_str_list(promoted)
        row.updated_at = now_iso()
    out = dict(profile)
    out["active_domains"] = list(active)
    out["queued_domains"] = list(queued)
    out["promoted_domains"] = list(promoted)
    out["allowed_domains"] = sorted(set(allowed_domains))
    return out


def _create_safety_event_pg(
    session: Session,
    *,
    user_id: str,
    day: date,
    session_id: Optional[str],
    trigger_source: str,
    severity: str,
    reason_code: str,
    item_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> str:
    event_id = str(uuid.uuid4())
    session.add(
        SafetyEvent(
            event_id=event_id,
            user_id=user_id,
            session_id=session_id,
            date=day,
            trigger_source=str(trigger_source or "runtime"),
            severity=str(severity or "medium"),
            status="open",
            item_id=item_id,
            reason_code=str(reason_code or "safety_trigger"),
            details_json=json.dumps(details or {}, ensure_ascii=False),
            created_at=now_iso(),
            resolved_at=None,
            resolved_by=None,
            resolved_reason=None,
        )
    )
    session.flush()
    return event_id


def _list_open_safety_events_pg(
    session: Session,
    *,
    user_id: str,
    day: date,
    lookback_days: int,
) -> List[Dict[str, Any]]:
    return list_safety_events(
        session,
        user_id=user_id,
        day=day,
        lookback_days=lookback_days,
        status="open",
    )


def get_active_registry_version(session: Session) -> Optional[str]:
    row = session.execute(
        select(RegistryVersion.version).where(RegistryVersion.status == "active").order_by(RegistryVersion.created_at.desc())
    ).first()
    return str(row[0]) if row else None


def set_registry_version_active(session: Session, version: str) -> None:
    session.execute(text("UPDATE registry_versions SET status='inactive' WHERE status='active';"))
    session.merge(RegistryVersion(version=version, status="active", created_at=now_iso()))


def upsert_registry_version(session: Session, version: str, status: str = "inactive") -> None:
    existing = session.get(RegistryVersion, version)
    if existing:
        return
    session.add(RegistryVersion(version=version, status=status, created_at=now_iso()))


def ensure_registry_active(session: Session, registry_root: str) -> str:
    active = get_active_registry_version(session)
    if active:
        return active
    versions = _list_versions(registry_root)
    if not versions:
        raise ValueError("No registry versions found")
    active = versions[-1]
    set_registry_version_active(session, active)
    return active


def upload_registry_bundle(session: Session, registry_root: str, bundle: Dict[str, Any]) -> str:
    registry = validate_registry_bundle(bundle)
    save_registry(registry_root, registry)
    upsert_registry_version(session, registry.version, status="inactive")
    return registry.version


def activate_registry_version(session: Session, registry_root: str, version: str) -> None:
    versions = set(_list_versions(registry_root))
    if version not in versions:
        raise ValueError(f"Unknown registry version: {version}")
    set_registry_version_active(session, version)


def get_or_create_daily_session(
    session: Session,
    *,
    registry_root: str,
    user_id: str,
    day: date,
    core_questions_per_day: int,
    extra_batch_size: int,
    extra_batches_max_per_day: int,
    item_repeat_cooldown_days: int,
    max_active_new_domains: int = 2,
    safety_event_lookback_days: int = 30,
    safety_min_questions_override: int = 3,
    selection_mode: str = "deterministic",
    policy_context: Optional[Dict[str, Any]] = None,
    identity_mask_id: Optional[str] = None,
    policy_root: Optional[str] = None,
    policy_default_version: str = "v1",
    policy_epsilon_explore: float = 0.05,
    policy_log_context_snapshot: bool = False,
    policy_log_candidate_set_snapshot: bool = False,
    policy_auto_rollback_enabled: bool = True,
    policy_rollback_window_days: int = 14,
    policy_rollback_min_outcomes: int = 40,
    policy_rollback_max_safety_violations: int = 0,
    policy_rollback_min_completion_drop: float = 0.03,
    policy_rollback_burden_increase_ratio: float = 1.10,
) -> DailySession:
    ensure_user(session, user_id)
    ensure_user_profile(session, user_id)
    reg_version = ensure_registry_active(session, registry_root)
    registry = load_registry(registry_root, reg_version)
    mark_previous_day_incomplete_sessions_abandoned(session, user_id=user_id, day=day)
    session.flush()

    existing = session.execute(
        select(DailySession).where(and_(DailySession.user_id == user_id, DailySession.date == day))
    ).scalars().first()
    if existing:
        return existing

    requested_selection_mode = normalize_selection_mode(selection_mode or "deterministic")
    selection_mode_final = requested_selection_mode
    rollback_guard: Optional[Dict[str, Any]] = None
    if requested_selection_mode == "policy_live" and bool(policy_auto_rollback_enabled):
        rollback_guard = evaluate_policy_live_rollback_guard_pg(
            session,
            day=day,
            window_days=policy_rollback_window_days,
            min_outcomes=policy_rollback_min_outcomes,
            max_safety_violations=policy_rollback_max_safety_violations,
            min_completion_drop=policy_rollback_min_completion_drop,
            burden_increase_ratio=policy_rollback_burden_increase_ratio,
        )
        if bool(rollback_guard.get("rollback", False)):
            selection_mode_final = "deterministic"

    last_asked = _get_last_asked_dates(session, user_id=user_id, day=day, lookback_days=max(30, item_repeat_cooldown_days + 2))
    last_scores = _get_last_score_dates(session, user_id=user_id)
    baselines = _get_baselines(session, user_id=user_id)
    answered_by_scale, rolling_by_scale = _compute_scale_windows(session, registry=registry, user_id=user_id, day=day)
    ctx_obj = policy_context or {}
    if bool(ctx_obj.get("safety_trigger_active", False)):
        _create_safety_event_pg(
            session,
            user_id=user_id,
            day=day,
            session_id=None,
            trigger_source="fusion_context",
            severity=str(ctx_obj.get("safety_severity") or "high"),
            reason_code=str(ctx_obj.get("safety_reason_code") or "safety_trigger_active"),
            details={"source": "context", "payload_keys": sorted(list(ctx_obj.keys()))},
        )

    profile = get_user_profile(session, user_id)
    onboarding_complete = bool(profile.get("onboarding_complete", False))
    declined_item_ids = {str(i) for i in profile.get("permanently_declined_item_ids", [])}
    allowed_item_ids: Optional[Set[str]] = None
    priority_item_ids: Optional[Set[str]] = None
    follow_up_item_ids = _get_due_follow_up_item_ids_pg(session, user_id=user_id, day=day)

    answered_item_ids: Set[str] = set()
    if not onboarding_complete:
        answered_item_ids = _get_all_answered_item_ids_pg(session, user_id=user_id)
    onboarding_gate = evaluate_onboarding_cardiometabolic_gate(
        onboarding_complete=onboarding_complete,
        registry=registry,
        declined_item_ids=declined_item_ids,
        answered_item_ids=answered_item_ids,
    )
    allowed_item_ids = onboarding_gate.allowed_item_ids
    priority_item_ids = onboarding_gate.priority_item_ids
    if onboarding_gate.mark_onboarding_complete:
        prof = session.get(UserProfile, user_id)
        if prof:
            prof.onboarding_complete = True
            prof.updated_at = now_iso()

    profile = _sync_user_domain_profile_pg(
        session,
        user_id=user_id,
        profile=get_user_profile(session, user_id),
        registry=registry,
        max_active_new_domains=max_active_new_domains,
    )
    declined_item_ids = {str(i) for i in profile.get("permanently_declined_item_ids", [])}
    allowed_domains = {canonical_domain_id(d) for d in profile.get("allowed_domains", [])}
    domain_allowed_item_ids = allowed_item_ids_for_domains(
        registry=registry,
        allowed_domains=allowed_domains,
    )
    allowed_item_ids = merge_allowed_item_ids(
        current_allowed_item_ids=allowed_item_ids,
        domain_allowed_item_ids=domain_allowed_item_ids,
    )

    open_safety_events = _list_open_safety_events_pg(
        session,
        user_id=user_id,
        day=day,
        lookback_days=int(safety_event_lookback_days),
    )
    safety_mode_active = bool(open_safety_events)
    if safety_mode_active:
        selection_mode_final = "deterministic"

    cfg = _cfg_stub(
        core_questions_per_day=core_questions_per_day,
        extra_batch_size=extra_batch_size,
        extra_batches_max_per_day=extra_batches_max_per_day,
        item_repeat_cooldown_days=item_repeat_cooldown_days,
        safety_min_questions_override=safety_min_questions_override,
    )

    # Load previous session engagement quality for selection feedback (Claim Family 5)
    _prev_engagement: Optional[str] = None
    prev_sup = session.execute(
        select(SessionUncertaintyProfileRow.engagement_quality)
        .where(
            and_(
                SessionUncertaintyProfileRow.user_id == user_id,
                SessionUncertaintyProfileRow.session_date < day.isoformat(),
            )
        )
        .order_by(SessionUncertaintyProfileRow.session_date.desc())
        .limit(1)
    ).first()
    if prev_sup:
        _prev_engagement = str(prev_sup[0])

    selection = build_selection_plan(
        registry=registry,
        cfg=cfg,
        user_id=user_id,
        day=day,
        answered_item_ids_by_scale=answered_by_scale,
        last_asked_date_by_item_id=last_asked,
        last_score_date_by_scale_id=last_scores,
        baselines_by_scale_id=baselines,
        rolling_normalized_by_scale_id=rolling_by_scale,
        allowed_item_ids=allowed_item_ids,
        declined_item_ids=declined_item_ids,
        priority_item_ids=priority_item_ids,
        follow_up_item_ids=follow_up_item_ids,
        previous_engagement_quality=_prev_engagement,
    )
    if safety_mode_active:
        selection = _apply_safety_mode_selection_pg(
            registry=registry,
            cfg=cfg,
            user_id=user_id,
            day=day,
            base_selection=selection,
            answered_item_ids_by_scale=answered_by_scale,
            last_asked_date_by_item_id=last_asked,
            last_score_date_by_scale_id=last_scores,
            baselines_by_scale_id=baselines,
            rolling_normalized_by_scale_id=rolling_by_scale,
            allowed_item_ids=allowed_item_ids,
            declined_item_ids=declined_item_ids,
            priority_item_ids=priority_item_ids,
            follow_up_item_ids=follow_up_item_ids,
        )

    policy_decision_id: Optional[str] = None
    served_core_item_ids: Tuple[str, ...] = tuple(selection.core_item_ids)
    policy_summary: Optional[Dict[str, Any]] = None
    if selection_mode_final in POLICY_SELECTION_MODES:
        pipeline_candidate_set, _, _ = build_candidate_set(
            registry=registry,
            cfg=cfg,
            user_id=user_id,
            day=day,
            answered_item_ids_by_scale=answered_by_scale,
            last_asked_date_by_item_id=last_asked,
            last_score_date_by_scale_id=last_scores,
            baselines_by_scale_id=baselines,
            rolling_normalized_by_scale_id=rolling_by_scale,
            allowed_item_ids=allowed_item_ids,
            declined_item_ids=declined_item_ids,
            priority_item_ids=priority_item_ids,
            follow_up_item_ids=follow_up_item_ids,
            force_timeframe=selection.timeframe,
        )
        policy_exec = execute_policy_selection(
            pipeline_candidate_set=pipeline_candidate_set,
            user_id=user_id,
            day=day,
            deterministic_baseline_selected=selection.core_item_ids,
            selection_mode=selection_mode_final,
            runtime_mode=policy_runtime_mode(selection_mode_final),
            identity_mask_id=identity_mask_id,
            context_obj=ctx_obj,
            policy_root=str(policy_root or ""),
            policy_default_version=str(policy_default_version),
            epsilon_explore=float(policy_epsilon_explore),
            include_context_snapshot=bool(policy_log_context_snapshot),
            include_candidate_set_snapshot=bool(policy_log_candidate_set_snapshot),
        )
        row = policy_exec.decision_row_payload
        session.add(
            PolicyDecisionRow(
                decision_id=row["decision_id"],
                user_id=user_id,
                date=day,
                identity_mask_id=row["identity_mask_id"],
                selection_mode=row["selection_mode"],
                mode=row["mode"],
                policy_version=row["policy_version"],
                feature_version=row["feature_version"],
                context_hash=row["context_hash"],
                candidate_set_hash=row["candidate_set_hash"],
                context_json=row["context_json"],
                candidate_set_json=row["candidate_set_json"],
                selected_item_ids_json=row["selected_item_ids_json"],
                deterministic_baseline_selected_json=row["deterministic_baseline_selected_json"],
                propensities_json=row["propensities_json"],
                explanations_json=row["explanations_json"],
                counterfactuals_json=row["counterfactuals_json"],
                created_at=now_iso(),
            )
        )
        policy_decision_id = policy_exec.decision.decision_id
        served_core_item_ids = policy_exec.served_core_item_ids
        policy_summary = policy_exec.summary

    session_id = str(uuid.uuid4())
    explain = build_selection_explain_payload(
        reasons_by_item_id=selection.reasons_by_item_id,
        primary_scale_by_item_id=selection.primary_scale_by_item_id,
        near_unlock_scales=selection.near_unlock_scales,
        due_retest_scales=selection.due_retest_scales,
        drift_scales=selection.drift_scales,
        follow_up_item_ids=follow_up_item_ids,
        profile=profile,
        allowed_domains=allowed_domains,
        requested_selection_mode=requested_selection_mode,
        selection_mode=selection_mode_final,
        policy_summary=policy_summary,
        deterministic_baseline_core_item_ids=selection.core_item_ids,
        served_core_item_ids=served_core_item_ids,
        rollback_guard=rollback_guard,
        safety_mode_active=safety_mode_active,
        open_safety_events=open_safety_events,
        timeframe=selection.timeframe,
    )
    if _prev_engagement:
        explain["previous_engagement_quality"] = _prev_engagement

    extra_batches_filtered = filter_extra_batches(
        extra_batches=selection.extra_batches,
        core_item_ids=served_core_item_ids,
        declined_item_ids=declined_item_ids,
        allowed_item_ids=allowed_item_ids,
    )

    ds = DailySession(
        session_id=session_id,
        user_id=user_id,
        date=day,
        registry_version=reg_version,
        timeframe=str(selection.timeframe),
        status="created",
        selection_mode=selection_mode_final,
        policy_decision_id=policy_decision_id,
        core_questions_json=json.dumps(list(served_core_item_ids)),
        extra_batches_json=json.dumps([list(batch) for batch in extra_batches_filtered]),
        extra_batches_used=0,
        selection_explain_json=json.dumps(explain),
        created_at=now_iso(),
    )
    session.add(ds)
    return ds


def take_next_extra_batch(
    session: Session,
    session_id: str,
    *,
    user_id: Optional[str] = None,
    extra_batches_max_per_day: int = 3,
) -> Optional[List[str]]:
    ds = session.get(DailySession, session_id)
    if not ds:
        return None
    if user_id is not None and str(ds.user_id) != str(user_id):
        raise PermissionError("session_id does not belong to user_id")
    batches = json.loads(ds.extra_batches_json)
    pool: List[str] = []
    for batch in batches:
        if isinstance(batch, list):
            pool.extend(str(item_id) for item_id in batch)
    max_extras = min(max(0, int(extra_batches_max_per_day)), len(pool))
    used = int(ds.extra_batches_used or 0)
    if used >= max_extras:
        return None
    batch = [pool[used]]
    ds.extra_batches_used = used + 1
    return list(batch)


def _get_due_follow_up_item_ids_pg(session: Session, *, user_id: str, day: date) -> Set[str]:
    rows = session.execute(
        select(FollowUpQueueItem.item_id)
        .where(
            and_(
                FollowUpQueueItem.user_id == user_id,
                FollowUpQueueItem.status == "pending",
                FollowUpQueueItem.earliest_date <= day,
            )
        )
        .order_by(FollowUpQueueItem.priority.desc(), FollowUpQueueItem.created_at.asc())
    ).all()
    return {str(r[0]) for r in rows}


def _enqueue_follow_up_queue_item_pg(
    session: Session,
    *,
    user_id: str,
    item_id: str,
    scale_id: Optional[str],
    reason_code: str,
    reason_detail: Optional[Dict[str, Any]],
    priority: float,
    earliest_date: date,
    timeframe: Optional[str],
) -> None:
    existing = session.execute(
        select(FollowUpQueueItem)
        .where(
            and_(
                FollowUpQueueItem.user_id == user_id,
                FollowUpQueueItem.item_id == item_id,
                FollowUpQueueItem.status == "pending",
            )
        )
        .order_by(FollowUpQueueItem.created_at.asc())
        .limit(1)
    ).scalars().first()
    if existing:
        existing.scale_id = scale_id
        existing.reason_code = reason_code
        existing.reason_detail_json = json.dumps(reason_detail or {}, ensure_ascii=False)
        existing.priority = float(max(float(existing.priority or 0.0), float(priority)))
        existing.earliest_date = min(existing.earliest_date, earliest_date)
        existing.timeframe = timeframe
        return
    session.add(
        FollowUpQueueItem(
            queue_id=str(uuid.uuid4()),
            user_id=user_id,
            item_id=item_id,
            scale_id=scale_id,
            reason_code=reason_code,
            reason_detail_json=json.dumps(reason_detail or {}, ensure_ascii=False),
            priority=float(priority),
            earliest_date=earliest_date,
            timeframe=timeframe,
            status="pending",
            created_at=now_iso(),
            resolved_at=None,
            resolved_reason=None,
        )
    )


def _enqueue_follow_up_items_for_scale_pg(
    session: Session,
    *,
    registry: Registry,
    user_id: str,
    day: date,
    scale_id: str,
    answered_item_ids: Set[str],
    reason_code: str,
    priority: float,
    reason_detail: Optional[Dict[str, Any]] = None,
    max_items: int = 2,
) -> None:
    scale = registry.scales.get(scale_id)
    if not scale:
        return
    profile = get_user_profile(session, user_id)
    declined = {str(i) for i in profile.get("permanently_declined_item_ids", [])}

    missing = [
        si.item_id
        for si in scale.items
        if si.item_id in registry.items and si.item_id not in answered_item_ids and si.item_id not in declined
    ]
    fallback = [
        si.item_id
        for si in scale.items
        if si.item_id in registry.items and si.item_id not in declined
    ]

    def _sort_key(item_id: str) -> Tuple[int, int, str]:
        item = registry.items[item_id]
        intrusiveness_rank = {"low": 0, "medium": 1, "high": 2}.get(str(item.intrusiveness), 1)
        sensitivity_rank = {"low": 0, "medium": 1, "high": 2}.get(str(item.sensitivity), 1)
        return (intrusiveness_rank, sensitivity_rank, str(item_id))

    selected = sorted(missing, key=_sort_key)
    if not selected:
        selected = sorted(fallback, key=_sort_key)
    selected = selected[: max(0, int(max_items))]

    for item_id in selected:
        item = registry.items[item_id]
        timeframe = str(item.timeframes_allowed[0]) if item.timeframes_allowed else "last_7_days"
        _enqueue_follow_up_queue_item_pg(
            session,
            user_id=user_id,
            item_id=item_id,
            scale_id=scale_id,
            reason_code=reason_code,
            reason_detail=reason_detail,
            priority=float(priority),
            earliest_date=day,
            timeframe=timeframe,
        )


def _resolve_follow_up_queue_for_item_pg(
    session: Session,
    *,
    user_id: str,
    item_id: str,
    resolution_day: date,
    resolved_reason: str,
    status: str = "resolved",
) -> None:
    session.execute(
        text(
            """
            UPDATE follow_up_queue
            SET status=:status,
                resolved_at=:resolved_at,
                resolved_reason=:resolved_reason
            WHERE user_id=:user_id
              AND item_id=:item_id
              AND status='pending'
              AND earliest_date<=:resolution_day;
            """
        ),
        {
            "status": str(status),
            "resolved_at": date_to_start_iso(resolution_day),
            "resolved_reason": str(resolved_reason),
            "user_id": user_id,
            "item_id": item_id,
            "resolution_day": resolution_day,
        },
    )


def _resolve_follow_up_queue_for_scale_pg(
    session: Session,
    *,
    user_id: str,
    scale_id: str,
    resolution_day: date,
    resolved_reason: str,
) -> None:
    session.execute(
        text(
            """
            UPDATE follow_up_queue
            SET status='resolved',
                resolved_at=:resolved_at,
                resolved_reason=:resolved_reason
            WHERE user_id=:user_id
              AND scale_id=:scale_id
              AND status='pending'
              AND earliest_date<=:resolution_day;
            """
        ),
        {
            "resolved_at": date_to_start_iso(resolution_day),
            "resolved_reason": str(resolved_reason),
            "user_id": user_id,
            "scale_id": scale_id,
            "resolution_day": resolution_day,
        },
    )


def _mark_item_permanently_declined_pg(session: Session, *, user_id: str, item_id: str) -> None:
    ensure_user_profile(session, user_id)
    profile = session.get(UserProfile, user_id)
    if not profile:
        return
    declined: List[str] = []
    try:
        raw = json.loads(str(profile.permanently_declined_items_json or "[]"))
        if isinstance(raw, list):
            declined = [str(x) for x in raw]
    except Exception:
        declined = []
    if item_id in declined:
        return
    declined.append(str(item_id))
    profile.permanently_declined_items_json = json.dumps(sorted(set(declined)), ensure_ascii=False)
    profile.updated_at = now_iso()
    _resolve_follow_up_queue_for_item_pg(
        session,
        user_id=user_id,
        item_id=item_id,
        resolution_day=date.today(),
        resolved_reason="permanently_declined",
        status="cancelled",
    )


def _insert_scale_evidence_rows_pg(
    session: Session,
    *,
    registry: Registry,
    user_id: str,
    session_id: str,
    answer_event_id: str,
    item_id: str,
    answer_value: float,
    timeframe: str,
    window_date: date,
) -> None:
    for scale in registry.scales.values():
        for scale_item in scale.items:
            if scale_item.item_id != item_id:
                continue
            contribution = float(answer_value)
            if bool(scale_item.reverse):
                response_type = get_response_type(scale.response_type)
                contribution = response_type.min_value + response_type.max_value - contribution
            contribution *= float(scale_item.weight)
            session.add(
                ScaleEvidence(
                    evidence_id=str(uuid.uuid4()),
                    user_id=user_id,
                    answer_event_id=answer_event_id,
                    session_id=session_id,
                    item_id=item_id,
                    scale_id=scale.id,
                    weight=float(scale_item.weight),
                    contribution_value=float(contribution),
                    timeframe=str(timeframe),
                    window_date=window_date,
                    created_at=now_iso(),
                )
            )


def _apply_safety_mode_selection_pg(
    *,
    registry: Registry,
    cfg: Any,
    user_id: str,
    day: date,
    base_selection,
    answered_item_ids_by_scale: Dict[str, Set[str]],
    last_asked_date_by_item_id: Dict[str, date],
    last_score_date_by_scale_id: Dict[str, date],
    baselines_by_scale_id: Dict[str, BaselineState],
    rolling_normalized_by_scale_id: Dict[str, float],
    allowed_item_ids: Optional[Set[str]],
    declined_item_ids: Set[str],
    priority_item_ids: Optional[Set[str]],
    follow_up_item_ids: Set[str],
):
    candidate_set, _, _ = build_candidate_set(
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
        force_timeframe=base_selection.timeframe,
    )
    target = max(1, min(int(cfg.core_questions_per_day), int(max(1, cfg.safety_min_questions_override))))
    selected: List[str] = []
    blocked = set(candidate_set.excluded_item_ids)

    def _is_safe_item(item_id: str) -> bool:
        item = registry.items.get(item_id)
        if not item:
            return False
        if str(item.intrusiveness) == "high":
            return False
        if str(item.sensitivity) == "high":
            return False
        if item_id in declined_item_ids:
            return False
        return True

    for item_id in candidate_set.mandatory_item_ids:
        if len(selected) >= target:
            break
        if item_id in blocked:
            continue
        if not _is_safe_item(item_id):
            continue
        if item_id not in selected:
            selected.append(item_id)

    for item_id in candidate_set.ordered_candidate_item_ids:
        if len(selected) >= target:
            break
        if item_id in selected:
            continue
        if item_id in blocked:
            continue
        if not _is_safe_item(item_id):
            continue
        selected.append(item_id)

    if not selected:
        for item_id in base_selection.core_item_ids:
            if item_id in registry.items and item_id not in declined_item_ids:
                selected.append(item_id)
                break

    reasons = dict(base_selection.reasons_by_item_id)
    for item_id in selected:
        current = list(reasons.get(item_id, ()))
        if "safety_protocol" not in current:
            current.append("safety_protocol")
        reasons[item_id] = tuple(current)

    return type(base_selection)(
        date=base_selection.date,
        core_item_ids=tuple(selected[:target]),
        extra_batches=tuple(),
        reasons_by_item_id=reasons,
        primary_scale_by_item_id=dict(base_selection.primary_scale_by_item_id),
        near_unlock_scales=tuple(),
        due_retest_scales=tuple(),
        drift_scales=tuple(),
        timeframe=base_selection.timeframe,
    )


def evaluate_policy_live_rollback_guard_pg(
    session: Session,
    *,
    day: date,
    window_days: int,
    min_outcomes: int,
    max_safety_violations: int,
    min_completion_drop: float,
    burden_increase_ratio: float,
) -> Dict[str, Any]:
    w = int(max(1, window_days))
    min_n = int(max(1, min_outcomes))
    max_safety = int(max(0, max_safety_violations))
    completion_drop = float(max(0.0, min_completion_drop))
    burden_ratio = float(max(1.0, burden_increase_ratio))

    recent_start = day - timedelta(days=w - 1)
    recent_end = day
    prior_start = day - timedelta(days=(2 * w) - 1)
    prior_end = day - timedelta(days=w)

    def _agg(start_day: date, end_day: date) -> Tuple[int, Optional[float], Optional[float]]:
        row = session.execute(
            select(
                func.count(PolicyOutcomeRow.decision_id),
                func.avg(PolicyOutcomeRow.completion_rate),
                func.avg(PolicyOutcomeRow.response_time_ms_median),
            )
            .join(PolicyDecisionRow, PolicyDecisionRow.decision_id == PolicyOutcomeRow.decision_id)
            .where(
                and_(
                    PolicyDecisionRow.selection_mode == "policy_live",
                    PolicyDecisionRow.mode == "live",
                    PolicyDecisionRow.date >= start_day,
                    PolicyDecisionRow.date <= end_day,
                )
            )
        ).first()
        if row is None:
            return 0, None, None
        return int(row[0] or 0), (float(row[1]) if row[1] is not None else None), (float(row[2]) if row[2] is not None else None)

    recent_n, recent_completion, recent_burden = _agg(recent_start, recent_end)
    prior_n, prior_completion, prior_burden = _agg(prior_start, prior_end)
    safety_violations = _count_policy_live_safety_violations_pg(session, start_day=recent_start, end_day=recent_end)

    degradation = False
    if (
        recent_n >= min_n
        and prior_n >= min_n
        and recent_completion is not None
        and prior_completion is not None
        and recent_burden is not None
        and prior_burden is not None
        and prior_burden > 0.0
    ):
        completion_down = float(recent_completion) <= float(prior_completion) - float(completion_drop)
        burden_up = float(recent_burden) >= float(prior_burden) * float(burden_ratio)
        degradation = bool(completion_down and burden_up)

    rollback = bool(safety_violations > max_safety or degradation)
    reasons: List[str] = []
    if safety_violations > max_safety:
        reasons.append("safety_violations")
    if degradation:
        reasons.append("burden_up_completion_down")

    return {
        "enabled": True,
        "rollback": rollback,
        "reasons": reasons,
        "window_days": w,
        "min_outcomes": min_n,
        "max_safety_violations": max_safety,
        "completion_drop_threshold": completion_drop,
        "burden_increase_ratio_threshold": burden_ratio,
        "recent": {
            "start_date": recent_start.isoformat(),
            "end_date": recent_end.isoformat(),
            "outcomes": recent_n,
            "completion_rate": recent_completion,
            "burden_ms_median": recent_burden,
            "safety_violations": int(safety_violations),
        },
        "prior": {
            "start_date": prior_start.isoformat(),
            "end_date": prior_end.isoformat(),
            "outcomes": prior_n,
            "completion_rate": prior_completion,
            "burden_ms_median": prior_burden,
        },
    }


def _count_policy_live_safety_violations_pg(session: Session, *, start_day: date, end_day: date) -> int:
    rows = session.execute(
        select(PolicyDecisionRow.selected_item_ids_json, PolicyDecisionRow.candidate_set_json).where(
            and_(
                PolicyDecisionRow.selection_mode == "policy_live",
                PolicyDecisionRow.mode == "live",
                PolicyDecisionRow.date >= start_day,
                PolicyDecisionRow.date <= end_day,
            )
        )
    ).all()
    violations = 0
    for selected_json, candidate_set_json in rows:
        try:
            selected = json.loads(str(selected_json or "[]"))
            cand_obj = json.loads(str(candidate_set_json or "{}"))
            if not isinstance(selected, list) or not isinstance(cand_obj, dict):
                violations += 1
                continue
            candidates = cand_obj.get("candidates") if isinstance(cand_obj.get("candidates"), list) else []
            candidate_ids = {str(c.get("item_id")) for c in candidates}
            blocked_ids = {
                str(c.get("item_id"))
                for c in candidates
                if isinstance(c.get("constraint_tags"), list) and "blocked" in c.get("constraint_tags")
            }
            mandatory_ids = {
                str(x)
                for x in (
                    cand_obj.get("mandatory_item_ids")
                    if isinstance(cand_obj.get("mandatory_item_ids"), list)
                    else []
                )
            }
            required_mandatory = mandatory_ids - blocked_ids
            selected_set = {str(i) for i in selected}
            expected_k = int(cand_obj.get("k_core") or len(selected))
            if any(str(i) not in candidate_ids for i in selected):
                violations += 1
                continue
            if any(str(i) in blocked_ids for i in selected):
                violations += 1
                continue
            if not required_mandatory.issubset(selected_set):
                violations += 1
                continue
            if len(selected) != expected_k:
                violations += 1
                continue
        except Exception:
            violations += 1
    return int(violations)


def submit_answers(
    session: Session,
    *,
    registry_root: str,
    user_id: str,
    session_id: str,
    answers: List[Dict[str, Any]],
    drift_routing_table_path: Optional[str] = None,
    site_config_id: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_user(session, user_id)
    ds = session.get(DailySession, session_id)
    if not ds:
        raise ValueError("Unknown session_id")
    if ds.user_id != user_id:
        raise ValueError("session_id does not belong to user_id")
    reg_version = ds.registry_version
    registry = load_registry(registry_root, reg_version)
    session_timeframe = str(ds.timeframe or "last_7_days")

    # Resolve site config for feature gating
    resolved_config = _resolve_site_config_obj(session, user_id, site_config_id=site_config_id)
    metadata_enabled = bool(resolved_config and resolved_config.behavioural_metadata_enabled)

    # Accumulators for response metadata (Claim Family 5)
    session_modifiers: List[UncertaintyModifier] = []
    session_metadata_list: List[ResponseMetadata] = []

    inserted = 0
    for a in answers:
        item_id = str(a.get("item_id") or "")
        if item_id not in registry.items:
            raise ValueError(f"Unknown item_id: {item_id}")
        raw_obj = a.get("raw")
        if isinstance(raw_obj, dict) and bool(raw_obj.get("permanently_declined")):
            _mark_item_permanently_declined_pg(session, user_id=user_id, item_id=item_id)
            continue
        if isinstance(raw_obj, dict) and bool(raw_obj.get("safety_trigger")):
            _create_safety_event_pg(
                session,
                user_id=user_id,
                day=ds.date,
                session_id=session_id,
                trigger_source="answer",
                severity=str(raw_obj.get("safety_severity") or "high"),
                reason_code=str(raw_obj.get("safety_reason_code") or "answer_safety_trigger"),
                item_id=item_id,
                details={
                    "raw_flags": {k: v for k, v in raw_obj.items() if str(k).startswith("safety_")},
                    "answered_at": str(a.get("answered_at") or now_iso()),
                },
            )
        client_event_id = str(a.get("client_event_id") or "")
        if not client_event_id:
            raise ValueError("Missing client_event_id")
        value = float(a.get("value"))
        answered_at = str(a.get("answered_at") or now_iso())
        response_type = get_response_type(registry.items[item_id].response_type)
        value = max(response_type.min_value, min(response_type.max_value, value))

        existing = session.execute(
            select(AnswerEvent).where(
                and_(AnswerEvent.user_id == user_id, AnswerEvent.client_event_id == client_event_id)
            )
        ).first()
        if existing:
            continue
        event_id = str(uuid.uuid4())
        session.add(
            AnswerEvent(
                event_id=event_id,
                client_event_id=client_event_id,
                user_id=user_id,
                session_id=session_id,
                item_id=item_id,
                answered_at=answered_at,
                value=value,
                raw_json=json.dumps(a.get("raw"), ensure_ascii=False) if a.get("raw") is not None else None,
            )
        )
        _insert_scale_evidence_rows_pg(
            session,
            registry=registry,
            user_id=user_id,
            session_id=session_id,
            answer_event_id=event_id,
            item_id=item_id,
            answer_value=float(value),
            timeframe=session_timeframe,
            window_date=ds.date,
        )
        _resolve_follow_up_queue_for_item_pg(
            session,
            user_id=user_id,
            item_id=item_id,
            resolution_day=ds.date,
            resolved_reason="answered",
        )

        # --- Behavioural metadata capture (Claim Family 5) ---
        meta_raw = a.get("metadata") if isinstance(a.get("metadata"), dict) else None
        if metadata_enabled and meta_raw is not None:
            resp_meta = ResponseMetadata(
                item_id=item_id,
                response_latency_ms=float(meta_raw["response_latency_ms"]) if meta_raw.get("response_latency_ms") is not None else None,
                edit_count=int(meta_raw.get("edit_count", 0)),
                was_skipped=bool(meta_raw.get("was_skipped", False)),
                was_declined=bool(meta_raw.get("was_declined", False)),
                channel=str(meta_raw.get("channel", "tap")),
                voice_hesitation_ms=float(meta_raw["voice_hesitation_ms"]) if meta_raw.get("voice_hesitation_ms") is not None else None,
                time_of_day_hour=int(meta_raw["time_of_day_hour"]) if meta_raw.get("time_of_day_hour") is not None else None,
            )
            item_spec = registry.items.get(item_id)
            resp_type_str = item_spec.response_type if item_spec else "likert_0_4"
            is_sensitive = bool(getattr(item_spec, "sensitivity", "low") in ("high",)) if item_spec else False
            modifier = compute_uncertainty_modifier(
                resp_meta, response_type=resp_type_str, is_sensitive=is_sensitive,
            )
            _insert_response_metadata_pg(
                session,
                user_id=user_id,
                session_id=session_id,
                answer_event_id=event_id,
                metadata=resp_meta,
                modifier=modifier,
            )
            session_modifiers.append(modifier)
            session_metadata_list.append(resp_meta)

        inserted += 1

    # Ensure newly inserted AnswerEvent rows are visible to subsequent SELECTs (autoflush is disabled).
    session.flush()

    # --- Session uncertainty profile ---
    session_engagement_out = None
    if metadata_enabled and session_modifiers:
        sp = compute_session_uncertainty_profile(
            session_modifiers,
            session_metadata_list,
            session_date=ds.date.isoformat(),
            user_id=user_id,
        )
        _upsert_session_uncertainty_profile_pg(session, session_id=session_id, profile=sp)
        session_engagement_out = {
            "session_id": session_id,
            "user_id": user_id,
            "session_date": ds.date.isoformat(),
            "session_multiplier": round(sp.session_multiplier, 4),
            "engagement_quality": sp.engagement_quality,
            "median_latency_ms": round(sp.median_latency_ms, 1),
            "total_edits": sp.total_edits,
            "skip_count": sp.skip_count,
            "decline_count": sp.decline_count,
            "item_count": len(session_modifiers),
        }

    core_item_ids = tuple(json.loads(ds.core_questions_json or "[]"))
    answered_rows = session.execute(
        select(AnswerEvent.item_id).where(AnswerEvent.session_id == session_id).distinct()
    ).all()
    answered_set = {str(r[0]) for r in answered_rows}
    core_answered_count = sum(1 for item_id in core_item_ids if item_id in answered_set)
    ds.status = "created"
    if answered_set:
        ds.status = "started"
    if core_item_ids and core_answered_count >= len(core_item_ids):
        ds.status = "completed"

    new_scores = _compute_and_store_scores(
        session, registry=registry, user_id=user_id, day=ds.date,
        session_id=session_id, site_config=resolved_config,
    )
    profile = get_user_profile(session, user_id)
    try:
        snapshot_write = _upsert_daily_state_and_circle_snapshots_pg(
            session,
            registry=registry,
            user_id=user_id,
            day=ds.date,
            state_schema_version=str(profile.get("state_schema_version") or STATE_SCHEMA_VERSION_DEFAULT),
            drift_routing_table_path=drift_routing_table_path,
            site_config=resolved_config,
        )
    except Exception as exc:
        raise SnapshotWriteError(f"Failed to write state/circle snapshots: {exc}") from exc
    progress = compute_user_scale_progress(session, registry=registry, user_id=user_id, day=ds.date)

    if ds.policy_decision_id:
        _upsert_policy_outcome_from_answers_pg(
            session,
            user_id=user_id,
            day=ds.date,
            session_id=session_id,
            decision_id=str(ds.policy_decision_id),
            core_item_ids=tuple(json.loads(ds.core_questions_json)),
            unlock_count=len(new_scores or []),
        )

    return {
        "inserted_answer_events": inserted,
        "new_scale_scores": new_scores,
        "scale_progress": progress,
        "snapshot_write": snapshot_write,
        "session_engagement": session_engagement_out,
    }


def get_state_snapshots(
    session: Session,
    *,
    user_id: str,
    day: date,
    window_days: int = 30,
) -> List[Dict[str, Any]]:
    start = day - timedelta(days=max(1, int(window_days)) - 1)
    rows = session.execute(
        select(StateSnapshot)
        .where(and_(StateSnapshot.user_id == user_id, StateSnapshot.date >= start))
        .order_by(StateSnapshot.date.asc(), StateSnapshot.created_at.asc())
    ).scalars().all()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "snapshot_id": str(r.snapshot_id),
                "user_id": str(r.user_id),
                "date": str(r.date.isoformat()),
                "timestamp": str(r.timestamp),
                "state_schema_version": str(r.state_schema_version),
                "model_version": str(r.model_version),
                "x_hat": json.loads(str(r.x_hat_json)),
                "x_uncertainty": json.loads(str(r.x_uncertainty_json)),
                "coverage": json.loads(str(r.coverage_json)),
                "source": str(r.source),
            }
        )
    return out


def get_circle_snapshots(
    session: Session,
    *,
    user_id: str,
    day: date,
    window_days: int = 30,
) -> List[Dict[str, Any]]:
    start = day - timedelta(days=max(1, int(window_days)) - 1)
    rows = session.execute(
        select(CircleSnapshot)
        .where(and_(CircleSnapshot.user_id == user_id, CircleSnapshot.date >= start))
        .order_by(CircleSnapshot.date.asc(), CircleSnapshot.created_at.asc())
    ).scalars().all()
    out: List[Dict[str, Any]] = []
    for r in rows:
        score = (
            float(r.coherence_score)
            if r.coherence_score is not None
            else coherence_score_from_radius(float(r.r))
        )
        tier = (
            _parse_json_or_none(str(r.coherence_tier_json) if r.coherence_tier_json else None)
            if r.coherence_tier_json
            else coherence_tier_for_score(score)
        )
        if not isinstance(tier, dict):
            tier = coherence_tier_for_score(score)
        out.append(
            {
                "snapshot_id": str(r.snapshot_id),
                "user_id": str(r.user_id),
                "date": str(r.date.isoformat()),
                "timestamp": str(r.timestamp),
                "state_snapshot_id": str(r.state_snapshot_id) if r.state_snapshot_id else None,
                "projection_version": str(r.projection_version),
                "anchor_version": str(r.anchor_version),
                "z": json.loads(str(r.z_json)),
                "z_star": json.loads(str(r.z_star_json)),
                "r": float(r.r),
                "theta": float(r.theta),
                "velocity": float(r.velocity),
                "acceleration": float(r.acceleration),
                "coherence": {
                    "score": float(score),
                    "tier": tier,
                },
                "uncertainty": json.loads(str(r.uncertainty_json)),
                "source": str(r.source),
            }
        )
    return out


def get_ews_features(
    session: Session,
    *,
    user_id: str,
    day: date,
    window_days: int = 30,
) -> List[Dict[str, Any]]:
    start = day - timedelta(days=max(1, int(window_days)) - 1)
    rows = session.execute(
        select(EwsFeature)
        .where(and_(EwsFeature.user_id == user_id, EwsFeature.date >= start))
        .order_by(EwsFeature.date.asc(), EwsFeature.created_at.asc())
    ).scalars().all()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "feature_id": str(r.feature_id),
                "user_id": str(r.user_id),
                "date": str(r.date.isoformat()),
                "timestamp": str(r.timestamp),
                "window_size_days": int(r.window_size_days),
                "var_r": float(r.var_r),
                "ac1_r": float(r.ac1_r),
                "trend_speed": float(r.trend_speed),
                "trend_accel": float(r.trend_accel),
                "recovery_rate": float(r.recovery_rate),
                "ews_score": float(r.ews_score),
                "notes": _parse_json_or_none(str(r.notes_json) if r.notes_json else None) or {},
            }
        )
    return out


def get_drift_events(
    session: Session,
    *,
    user_id: str,
    day: date,
    window_days: int = 30,
) -> List[Dict[str, Any]]:
    start = day - timedelta(days=max(1, int(window_days)) - 1)
    rows = session.execute(
        select(DriftEvent)
        .where(and_(DriftEvent.user_id == user_id, DriftEvent.date >= start))
        .order_by(DriftEvent.date.asc(), DriftEvent.created_at.asc())
    ).scalars().all()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "event_id": str(r.event_id),
                "user_id": str(r.user_id),
                "date": str(r.date.isoformat()),
                "timestamp": str(r.timestamp),
                "coherence_score": float(r.coherence_score),
                "drift_domain": str(r.drift_domain),
                "triggered_instrument": str(r.triggered_instrument) if r.triggered_instrument else None,
                "ews_score": float(r.ews_score) if r.ews_score is not None else None,
                "reason_codes": _parse_json_or_none(str(r.reason_codes_json) if r.reason_codes_json else None) or [],
                "details": _parse_json_or_none(str(r.details_json) if r.details_json else None) or {},
            }
        )
    return out


def get_coherence_tier_contract() -> Dict[str, Any]:
    return coherence_tier_contract()


def get_drift_routing_contract(*, drift_routing_table_path: Optional[str] = None) -> Dict[str, Any]:
    version, routes = load_drift_routes(drift_routing_table_path)
    return route_contract(version, routes)


def suggest_emotion_deep_dive(
    session: Session,
    *,
    registry_root: str,
    user_id: str,
    day: date,
    lookback_safety_days: int = 30,
    deep_dive_item_count: int = 10,
) -> Dict[str, Any]:
    ensure_user(session, user_id)
    ensure_user_profile(session, user_id)
    reg_v = ensure_registry_active(session, registry_root)
    registry = load_registry(registry_root, reg_v)
    profile = get_user_profile(session, user_id)
    declined = {str(i) for i in profile.get("permanently_declined_item_ids", [])}

    latest_state = session.execute(
        select(StateSnapshot.x_uncertainty_json)
        .where(and_(StateSnapshot.user_id == user_id, StateSnapshot.date <= day))
        .order_by(StateSnapshot.date.desc(), StateSnapshot.created_at.desc())
        .limit(1)
    ).first()
    uncertainty_values: List[float] = []
    if latest_state and latest_state[0]:
        try:
            obj = json.loads(str(latest_state[0]))
            if isinstance(obj, dict):
                uncertainty_values = [float(v) for v in obj.values()]
        except Exception:
            uncertainty_values = []
    mean_unc = (sum(uncertainty_values) / len(uncertainty_values)) if uncertainty_values else 0.0
    max_unc = max(uncertainty_values) if uncertainty_values else 0.0

    open_safety = _list_open_safety_events_pg(
        session,
        user_id=user_id,
        day=day,
        lookback_days=max(1, int(lookback_safety_days)),
    )
    triggered = bool(max_unc >= 0.55 or mean_unc >= 0.45 or open_safety)
    trigger_reasons: List[str] = []
    if max_unc >= 0.55:
        trigger_reasons.append("high_uncertainty_peak")
    if mean_unc >= 0.45:
        trigger_reasons.append("high_uncertainty_mean")
    if open_safety:
        trigger_reasons.append("open_safety_event")

    last_asked = _get_last_asked_dates(session, user_id=user_id, day=day, lookback_days=30)
    candidate_ids: List[str] = []
    for item in registry.items.values():
        if item.id in declined:
            continue
        tags = {str(t).lower() for t in item.tags}
        if tags & EMOTION_TAGS:
            candidate_ids.append(item.id)

    if not candidate_ids:
        for item in registry.items.values():
            if item.id in declined:
                continue
            if str(item.intrusiveness) == "high":
                continue
            if str(item.sensitivity) == "high":
                continue
            candidate_ids.append(item.id)

    def _sort_key(item_id: str) -> Tuple[int, int, int, str]:
        item = registry.items[item_id]
        novelty_days = 3650 if item_id not in last_asked else max(0, (day - last_asked[item_id]).days)
        intr = {"low": 0, "medium": 1, "high": 2}.get(str(item.intrusiveness), 1)
        sens = {"low": 0, "medium": 1, "high": 2}.get(str(item.sensitivity), 1)
        return (-novelty_days, intr, sens, item_id)

    candidate_ids = sorted(set(candidate_ids), key=_sort_key)
    target = max(8, min(12, int(deep_dive_item_count)))
    selected = candidate_ids[:target]
    questions = [
        {
            "item_id": item_id,
            "text": registry.items[item_id].text,
            "response_type": registry.items[item_id].response_type,
            "timeframes_allowed": list(registry.items[item_id].timeframes_allowed),
            "tags": list(registry.items[item_id].tags),
        }
        for item_id in selected
    ]
    return {
        "user_id": user_id,
        "date": day.isoformat(),
        "triggered": bool(triggered),
        "trigger_reasons": trigger_reasons or ["manual_optional"],
        "uncertainty": {
            "mean": float(mean_unc),
            "max": float(max_unc),
            "dimensions_observed": int(len(uncertainty_values)),
        },
        "questions": questions if triggered else [],
        "candidate_count": len(candidate_ids),
        "open_safety_events": open_safety[:3],
    }


def upsert_experiment(
    session: Session,
    *,
    user_id: str,
    body: Dict[str, Any],
    now_day: Optional[date] = None,
) -> Dict[str, Any]:
    ensure_user(session, user_id)
    now_day = now_day or date.today()
    experiment_id = str(body.get("experiment_id") or uuid.uuid4())
    name = str(body.get("name") or "Untitled experiment").strip() or "Untitled experiment"
    status = str(body.get("status") or "active").strip().lower()
    if status not in {"active", "paused", "completed", "cancelled"}:
        raise ValueError("status must be one of: active, paused, completed, cancelled")
    start_day = parse_date(str(body.get("start_date") or now_day.isoformat()))
    end_day = parse_date(str(body.get("end_date"))) if body.get("end_date") else None
    if end_day is not None and end_day < start_day:
        raise ValueError("end_date must be >= start_date")
    target_metrics = body.get("target_metrics") if isinstance(body.get("target_metrics"), dict) else {}
    stopping_rules = body.get("stopping_rules") if isinstance(body.get("stopping_rules"), dict) else {}
    intervention = body.get("intervention") if isinstance(body.get("intervention"), dict) else {}
    baseline_window_days = max(3, min(120, int(body.get("baseline_window_days") or 14)))
    eval_window_days = max(3, min(120, int(body.get("eval_window_days") or 14)))

    row = session.execute(
        select(Experiment).where(and_(Experiment.experiment_id == experiment_id, Experiment.user_id == user_id))
    ).scalars().first()
    if row:
        row.name = name
        row.status = status
        row.start_date = start_day
        row.end_date = end_day
        row.target_metrics_json = json.dumps(target_metrics, ensure_ascii=False)
        row.baseline_window_days = baseline_window_days
        row.eval_window_days = eval_window_days
        row.stopping_rules_json = json.dumps(stopping_rules, ensure_ascii=False)
        row.intervention_json = json.dumps(intervention, ensure_ascii=False)
        row.updated_at = now_iso()
    else:
        row = Experiment(
            experiment_id=experiment_id,
            user_id=user_id,
            name=name,
            status=status,
            start_date=start_day,
            end_date=end_day,
            target_metrics_json=json.dumps(target_metrics, ensure_ascii=False),
            baseline_window_days=baseline_window_days,
            eval_window_days=eval_window_days,
            stopping_rules_json=json.dumps(stopping_rules, ensure_ascii=False),
            intervention_json=json.dumps(intervention, ensure_ascii=False),
            results_json=None,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        session.add(row)
    session.flush()
    return get_experiment(session, user_id=user_id, experiment_id=experiment_id)


def list_experiments(
    session: Session,
    *,
    user_id: str,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conditions = [Experiment.user_id == user_id]
    if status:
        conditions.append(Experiment.status == str(status).strip().lower())
    rows = (
        session.execute(
            select(Experiment)
            .where(and_(*conditions))
            .order_by(Experiment.start_date.desc(), Experiment.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [_row_to_experiment(row) for row in rows]


def get_experiment(
    session: Session,
    *,
    user_id: str,
    experiment_id: str,
) -> Dict[str, Any]:
    row = session.execute(
        select(Experiment).where(and_(Experiment.user_id == user_id, Experiment.experiment_id == experiment_id))
    ).scalars().first()
    if not row:
        raise ValueError("Unknown experiment")
    return _row_to_experiment(row)


def compute_experiment_results(
    session: Session,
    *,
    user_id: str,
    experiment_id: str,
    day: Optional[date] = None,
) -> Dict[str, Any]:
    row = session.execute(
        select(Experiment).where(and_(Experiment.user_id == user_id, Experiment.experiment_id == experiment_id))
    ).scalars().first()
    if not row:
        raise ValueError("Unknown experiment")
    current_day = day or date.today()
    start_day = row.start_date
    end_day = row.end_date if row.end_date else current_day
    if end_day > current_day:
        end_day = current_day

    baseline_days = max(3, int(row.baseline_window_days or 14))
    eval_days = max(3, int(row.eval_window_days or 14))
    baseline_start = start_day - timedelta(days=baseline_days)
    baseline_end = start_day - timedelta(days=1)
    eval_start = start_day
    eval_end = min(end_day, start_day + timedelta(days=eval_days - 1))

    baseline_rows = session.execute(
        select(CircleSnapshot.r).where(
            and_(
                CircleSnapshot.user_id == user_id,
                CircleSnapshot.date >= baseline_start,
                CircleSnapshot.date <= baseline_end,
            )
        ).order_by(CircleSnapshot.date.asc())
    ).all()
    eval_rows = session.execute(
        select(CircleSnapshot.r).where(
            and_(
                CircleSnapshot.user_id == user_id,
                CircleSnapshot.date >= eval_start,
                CircleSnapshot.date <= eval_end,
            )
        ).order_by(CircleSnapshot.date.asc())
    ).all()
    baseline_r = [float(r[0]) for r in baseline_rows]
    eval_r = [float(r[0]) for r in eval_rows]
    baseline_mean = (sum(baseline_r) / len(baseline_r)) if baseline_r else None
    eval_mean = (sum(eval_r) / len(eval_r)) if eval_r else None

    responsiveness_pct: Optional[float] = None
    adverse_signal = None
    confidence = 0.0
    if baseline_mean is not None and eval_mean is not None:
        denom = max(1e-6, abs(float(baseline_mean)))
        responsiveness_pct = max(-100.0, min(100.0, ((baseline_mean - eval_mean) / denom) * 100.0))
        adverse_signal = bool(eval_mean > baseline_mean * 1.15)
        confidence = min(0.99, (min(len(baseline_r), len(eval_r)) / 14.0))

    deescalation_candidate = {
        "flag": bool(
            responsiveness_pct is not None
            and responsiveness_pct < 5.0
            and adverse_signal is False
            and confidence >= 0.6
        ),
        "type": "review_only",
        "note": "Flag for clinician review, not a treatment instruction.",
    }

    results = {
        "experiment_id": experiment_id,
        "user_id": user_id,
        "baseline_window": {"start": baseline_start.isoformat(), "end": baseline_end.isoformat(), "n": len(baseline_r)},
        "evaluation_window": {"start": eval_start.isoformat(), "end": eval_end.isoformat(), "n": len(eval_r)},
        "metrics": {
            "coherence_r_baseline_mean": baseline_mean,
            "coherence_r_eval_mean": eval_mean,
            "responsiveness_pct": responsiveness_pct,
            "adverse_signal": adverse_signal,
            "confidence": float(confidence),
        },
        "deescalation_candidate": deescalation_candidate,
        "disclaimer": "Decision support only. Not diagnosis and not medical advice.",
    }
    row.results_json = json.dumps(results, ensure_ascii=False)
    row.updated_at = now_iso()
    session.flush()
    return results


def build_domain_promotion_readiness(
    session: Session,
    *,
    registry_root: str,
    min_users: int,
    min_cycles: int,
    min_weeks: int,
    min_paired_ratio: float,
) -> Dict[str, Any]:
    reg_v = ensure_registry_active(session, registry_root)
    registry = load_registry(registry_root, reg_v)
    scale_domains = registry_scale_domains(registry)
    item_domains = registry_item_domains(registry)
    domains = [d for d in registry_domains(registry) if d != CARDIOMETABOLIC_DOMAIN]
    report_rows: List[Dict[str, Any]] = []

    for domain in domains:
        domain_items = sorted([item_id for item_id, ds in item_domains.items() if domain in ds])
        if not domain_items:
            continue
        rows = session.execute(
            select(AnswerEvent.user_id, DailySession.date)
            .join(DailySession, DailySession.session_id == AnswerEvent.session_id)
            .where(AnswerEvent.item_id.in_(domain_items))
            .order_by(DailySession.date.asc())
        ).all()
        per_user_days: Dict[str, Set[str]] = {}
        all_days: Set[str] = set()
        for row in rows:
            uid = str(row[0])
            d_iso = str(row[1].isoformat())
            per_user_days.setdefault(uid, set()).add(d_iso)
            all_days.add(d_iso)

        users_count = len(per_user_days)
        temporal_span_weeks = 0
        if all_days:
            min_d = min(parse_date(d) for d in all_days)
            max_d = max(parse_date(d) for d in all_days)
            temporal_span_weeks = int((max_d - min_d).days / 7) + 1

        users_with_cycles = 0
        total_user_days = 0
        paired_days = 0
        for uid, day_set in per_user_days.items():
            week_keys = {parse_date(d).strftime("%Y-%W") for d in day_set}
            if len(week_keys) >= int(min_cycles):
                users_with_cycles += 1
            total_user_days += len(day_set)
            for d_iso in day_set:
                has_state = session.execute(
                    select(StateSnapshot.snapshot_id).where(
                        and_(StateSnapshot.user_id == uid, StateSnapshot.date == parse_date(d_iso))
                    ).limit(1)
                ).first()
                if has_state:
                    paired_days += 1
        paired_ratio = (float(paired_days) / float(total_user_days)) if total_user_days > 0 else 0.0

        domain_scales = [sid for sid, dset in scale_domains.items() if domain in dset]
        variance_ok = False
        if domain_scales:
            vals = [
                float(r[0])
                for r in session.execute(
                    select(ScaleScore.normalized_score).where(ScaleScore.scale_id.in_(domain_scales))
                ).all()
            ]
            if len(vals) >= 2:
                mean_val = sum(vals) / len(vals)
                variance = sum((v - mean_val) ** 2 for v in vals) / max(1, len(vals) - 1)
                variance_ok = bool(variance > 1e-6)

        ready = bool(
            users_count >= int(min_users)
            and users_with_cycles >= int(min_users)
            and temporal_span_weeks >= int(min_weeks)
            and paired_ratio >= float(min_paired_ratio)
            and variance_ok
        )
        report_rows.append(
            {
                "domain_id": domain,
                "users": users_count,
                "users_with_min_cycles": users_with_cycles,
                "temporal_span_weeks": temporal_span_weeks,
                "paired_ratio": float(paired_ratio),
                "variance_ok": bool(variance_ok),
                "ready_for_promotion": ready,
                "required": {
                    "min_users": int(min_users),
                    "min_cycles": int(min_cycles),
                    "min_weeks": int(min_weeks),
                    "min_paired_ratio": float(min_paired_ratio),
                },
            }
        )

    return {
        "registry_version": reg_v,
        "domains": report_rows,
        "ready_domains": [r["domain_id"] for r in report_rows if r.get("ready_for_promotion")],
    }


def build_tta_summary(
    session: Session,
    *,
    registry_root: str,
    user_id: str,
    day: date,
) -> Dict[str, Any]:
    reg_v = ensure_registry_active(session, registry_root)
    registry = load_registry(registry_root, reg_v)
    progress = compute_user_scale_progress(session, registry=registry, user_id=user_id, day=day)

    changed: List[Dict[str, Any]] = []
    for scale_id, p in progress.items():
        last = p.get("last_score") if isinstance(p.get("last_score"), dict) else None
        if not last:
            continue
        changed.append(
            {
                "scale_id": scale_id,
                "name": p.get("name"),
                "score": last.get("normalized_score"),
                "delta_vs_prev": last.get("delta_vs_prev"),
                "personal_z": last.get("personal_z"),
                "risk_tier": last.get("risk_tier"),
            }
        )
    changed.sort(
        key=lambda x: (
            abs(float(x.get("delta_vs_prev") or 0.0)),
            abs(float(x.get("personal_z") or 0.0)),
        ),
        reverse=True,
    )
    changed = changed[:5]

    circle = get_circle_snapshots(session, user_id=user_id, day=day, window_days=30)
    trend = None
    if len(circle) >= 2:
        latest = circle[-1]
        prev = circle[-2]
        trend = {
            "r": latest.get("r"),
            "velocity": latest.get("velocity"),
            "acceleration": latest.get("acceleration"),
            "delta_r": float(latest.get("r") or 0.0) - float(prev.get("r") or 0.0),
        }

    interpretation: List[str] = []
    for item in changed:
        name = str(item.get("name") or item.get("scale_id"))
        tier = item.get("risk_tier")
        delta = item.get("delta_vs_prev")
        if tier:
            interpretation.append(f"{name}: {tier} tier, delta {round(float(delta or 0.0), 2)} vs previous.")
        else:
            interpretation.append(f"{name}: delta {round(float(delta or 0.0), 2)} vs previous.")

    return {
        "user_id": user_id,
        "date": day.isoformat(),
        "top_scale_changes": changed,
        "trajectory": trend,
        "narrative": interpretation[:4],
        "guardrails": [
            "Decision support only, not diagnosis.",
            "Scale interpretations follow validated instrument scoring.",
            "Trajectory reflects state change over time with uncertainty.",
        ],
    }


def _row_to_experiment(row: Experiment) -> Dict[str, Any]:
    def _json_or_empty(value: Any) -> Dict[str, Any]:
        try:
            obj = json.loads(str(value or "{}"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {
        "experiment_id": str(row.experiment_id),
        "user_id": str(row.user_id),
        "name": str(row.name),
        "status": str(row.status),
        "start_date": row.start_date.isoformat(),
        "end_date": row.end_date.isoformat() if row.end_date else None,
        "target_metrics": _json_or_empty(row.target_metrics_json),
        "baseline_window_days": int(row.baseline_window_days),
        "eval_window_days": int(row.eval_window_days),
        "stopping_rules": _json_or_empty(row.stopping_rules_json),
        "intervention": _json_or_empty(row.intervention_json),
        "results": _json_or_empty(row.results_json),
        "created_at": str(row.created_at),
        "updated_at": str(row.updated_at),
    }


def record_operational_metric(
    session: Session,
    *,
    metric_type: str,
    status: str,
    metric_date: date,
    latency_ms: Optional[float] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    session.add(
        OperationalMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=str(metric_type),
            status=str(status),
            metric_date=metric_date,
            latency_ms=(float(latency_ms) if latency_ms is not None else None),
            user_id=(str(user_id) if user_id else None),
            session_id=(str(session_id) if session_id else None),
            details_json=json.dumps(details or {}, ensure_ascii=False),
            created_at=now_iso(),
        )
    )


def build_slo_report(
    session: Session,
    *,
    day: date,
    window_days: int,
    safe_fallback_rate_max: float,
    answer_ingestion_p95_ms_max: float,
    snapshot_write_failure_rate_max: float,
) -> Dict[str, Any]:
    lookback = max(1, int(window_days))
    cutoff = day - timedelta(days=lookback - 1)

    decision_rows = session.execute(
        select(PolicyDecisionRow.mode).where(PolicyDecisionRow.date >= cutoff)
    ).all()
    decision_total = len(decision_rows)
    safe_fallback_count = sum(1 for row in decision_rows if str(row[0]) == "safe_fallback")
    safe_fallback_rate = (
        float(safe_fallback_count) / float(decision_total)
        if decision_total > 0
        else None
    )

    lat_rows = session.execute(
        select(OperationalMetric.latency_ms)
        .where(
            and_(
                OperationalMetric.metric_type == "answer_ingestion",
                OperationalMetric.status == "ok",
                OperationalMetric.metric_date >= cutoff,
                OperationalMetric.latency_ms.is_not(None),
            )
        )
    ).all()
    latencies = sorted(float(r[0]) for r in lat_rows if r[0] is not None)
    answer_p95 = _percentile(latencies, 95.0) if latencies else None

    snapshot_rows = session.execute(
        select(OperationalMetric.status)
        .where(
            and_(
                OperationalMetric.metric_type == "snapshot_write",
                OperationalMetric.metric_date >= cutoff,
            )
        )
    ).all()
    snapshot_total = len(snapshot_rows)
    snapshot_failures = sum(1 for r in snapshot_rows if str(r[0]) == "error")
    snapshot_failure_rate = (
        float(snapshot_failures) / float(snapshot_total)
        if snapshot_total > 0
        else None
    )

    alerts: List[Dict[str, Any]] = []
    if safe_fallback_rate is not None and safe_fallback_rate > float(safe_fallback_rate_max):
        alerts.append(
            {
                "code": "safe_fallback_rate",
                "severity": "high",
                "message": "Safe fallback rate exceeded threshold.",
                "value": safe_fallback_rate,
                "threshold": float(safe_fallback_rate_max),
            }
        )
    if answer_p95 is not None and answer_p95 > float(answer_ingestion_p95_ms_max):
        alerts.append(
            {
                "code": "answer_ingestion_latency_p95_ms",
                "severity": "high",
                "message": "Answer ingestion latency p95 exceeded threshold.",
                "value": answer_p95,
                "threshold": float(answer_ingestion_p95_ms_max),
            }
        )
    if snapshot_failure_rate is not None and snapshot_failure_rate > float(snapshot_write_failure_rate_max):
        alerts.append(
            {
                "code": "snapshot_write_failure_rate",
                "severity": "high",
                "message": "Snapshot write failure rate exceeded threshold.",
                "value": snapshot_failure_rate,
                "threshold": float(snapshot_write_failure_rate_max),
            }
        )

    return {
        "date": day.isoformat(),
        "window_days": lookback,
        "cutoff_date": cutoff.isoformat(),
        "targets": {
            "safe_fallback_rate_max": float(safe_fallback_rate_max),
            "answer_ingestion_p95_ms_max": float(answer_ingestion_p95_ms_max),
            "snapshot_write_failure_rate_max": float(snapshot_write_failure_rate_max),
        },
        "metrics": {
            "safe_fallback_rate": safe_fallback_rate,
            "safe_fallback_count": safe_fallback_count,
            "policy_decisions": decision_total,
            "answer_ingestion_latency_p95_ms": answer_p95,
            "answer_ingestion_samples": len(latencies),
            "snapshot_write_failure_rate": snapshot_failure_rate,
            "snapshot_write_failures": snapshot_failures,
            "snapshot_write_events": snapshot_total,
        },
        "alerts": alerts,
        "status": "ok" if not alerts else "degraded",
    }


def run_phi_retention(
    session: Session,
    *,
    day: date,
    answer_raw_retention_days: int,
    observation_raw_retention_days: int,
    policy_context_retention_days: int,
    dry_run: bool = True,
) -> Dict[str, Any]:
    answer_cutoff = day - timedelta(days=max(1, int(answer_raw_retention_days)))
    observation_cutoff = day - timedelta(days=max(1, int(observation_raw_retention_days)))
    policy_cutoff = day - timedelta(days=max(1, int(policy_context_retention_days)))

    answer_rows = session.execute(
        select(AnswerEvent.event_id)
        .join(DailySession, DailySession.session_id == AnswerEvent.session_id)
        .where(
            and_(
                DailySession.date <= answer_cutoff,
                AnswerEvent.raw_json.is_not(None),
            )
        )
    ).all()
    answer_ids = [str(r[0]) for r in answer_rows]

    observation_rows = session.execute(
        select(ObservationEvent.event_id, ObservationEvent.observed_at)
        .where(
            or_(
                ObservationEvent.features_json != "{}",
                ObservationEvent.provenance_json != "{}",
            )
        )
    ).all()
    observation_ids: List[str] = []
    for row in observation_rows:
        observed_at = str(row[1] or "")
        try:
            observed_day = parse_date(observed_at[:10])
        except Exception:
            continue
        if observed_day <= observation_cutoff:
            observation_ids.append(str(row[0]))

    policy_rows = session.execute(
        select(PolicyDecisionRow.decision_id)
        .where(
            and_(
                PolicyDecisionRow.date <= policy_cutoff,
                or_(
                    PolicyDecisionRow.context_json.is_not(None),
                    PolicyDecisionRow.candidate_set_json.is_not(None),
                ),
            )
        )
    ).all()
    policy_ids = [str(r[0]) for r in policy_rows]

    if not bool(dry_run):
        if answer_ids:
            session.execute(
                update(AnswerEvent)
                .where(AnswerEvent.event_id.in_(answer_ids))
                .values(raw_json=None)
            )
        if observation_ids:
            session.execute(
                update(ObservationEvent)
                .where(ObservationEvent.event_id.in_(observation_ids))
                .values(features_json="{}", provenance_json="{}")
            )
        if policy_ids:
            session.execute(
                update(PolicyDecisionRow)
                .where(PolicyDecisionRow.decision_id.in_(policy_ids))
                .values(context_json=None, candidate_set_json=None)
            )

    return {
        "as_of_date": day.isoformat(),
        "dry_run": bool(dry_run),
        "cutoffs": {
            "answer_raw": answer_cutoff.isoformat(),
            "observation_raw": observation_cutoff.isoformat(),
            "policy_context": policy_cutoff.isoformat(),
        },
        "affected_rows": {
            "answer_events_raw_redacted": len(answer_ids),
            "observation_events_redacted": len(observation_ids),
            "policy_decisions_redacted": len(policy_ids),
        },
    }


def build_clinical_audit_export(
    session: Session,
    *,
    start_day: date,
    end_day: date,
    user_id: Optional[str] = None,
    redact_user_ids: bool = True,
    redact_raw_payloads: bool = True,
    user_hash_salt: str = "",
) -> Dict[str, Any]:
    user_filter = [DailySession.date >= start_day, DailySession.date <= end_day]
    if user_id:
        user_filter.append(DailySession.user_id == str(user_id))

    sessions = session.execute(
        select(DailySession).where(and_(*user_filter)).order_by(DailySession.date.asc(), DailySession.created_at.asc())
    ).scalars().all()
    session_ids = [str(s.session_id) for s in sessions]
    session_users = {str(s.user_id) for s in sessions}
    if user_id:
        session_users.add(str(user_id))

    answer_rows: List[AnswerEvent] = []
    if session_ids:
        answer_rows = session.execute(
            select(AnswerEvent)
            .where(AnswerEvent.session_id.in_(session_ids))
            .order_by(AnswerEvent.answered_at.asc())
        ).scalars().all()

    score_filters = [ScaleScore.window_end >= start_day, ScaleScore.window_end <= end_day]
    if session_users:
        score_filters.append(ScaleScore.user_id.in_(tuple(session_users)))
    score_rows = session.execute(
        select(ScaleScore).where(and_(*score_filters)).order_by(ScaleScore.window_end.asc(), ScaleScore.computed_at.asc())
    ).scalars().all()

    state_filters = [StateSnapshot.date >= start_day, StateSnapshot.date <= end_day]
    if session_users:
        state_filters.append(StateSnapshot.user_id.in_(tuple(session_users)))
    state_rows = session.execute(
        select(StateSnapshot).where(and_(*state_filters)).order_by(StateSnapshot.date.asc(), StateSnapshot.created_at.asc())
    ).scalars().all()

    circle_filters = [CircleSnapshot.date >= start_day, CircleSnapshot.date <= end_day]
    if session_users:
        circle_filters.append(CircleSnapshot.user_id.in_(tuple(session_users)))
    circle_rows = session.execute(
        select(CircleSnapshot).where(and_(*circle_filters)).order_by(CircleSnapshot.date.asc(), CircleSnapshot.created_at.asc())
    ).scalars().all()

    decision_filters = [PolicyDecisionRow.date >= start_day, PolicyDecisionRow.date <= end_day]
    if session_users:
        decision_filters.append(PolicyDecisionRow.user_id.in_(tuple(session_users)))
    decision_rows = session.execute(
        select(PolicyDecisionRow).where(and_(*decision_filters)).order_by(PolicyDecisionRow.date.asc(), PolicyDecisionRow.created_at.asc())
    ).scalars().all()
    decision_ids = [str(d.decision_id) for d in decision_rows]

    outcome_rows: List[PolicyOutcomeRow] = []
    if decision_ids:
        outcome_rows = session.execute(
            select(PolicyOutcomeRow).where(PolicyOutcomeRow.decision_id.in_(decision_ids))
        ).scalars().all()

    def uid(value: str) -> str:
        if not bool(redact_user_ids):
            return str(value)
        return _hash_user_id(str(value), salt=user_hash_salt)

    sessions_out = [
        {
            "session_id": str(s.session_id),
            "user_id": uid(str(s.user_id)),
            "date": s.date.isoformat(),
            "timeframe": str(s.timeframe),
            "status": str(s.status),
            "selection_mode": str(s.selection_mode),
            "core_question_ids": json.loads(str(s.core_questions_json)),
            "policy_decision_id": str(s.policy_decision_id) if s.policy_decision_id else None,
        }
        for s in sessions
    ]

    answers_out = [
        {
            "event_id": str(r.event_id),
            "session_id": str(r.session_id),
            "user_id": uid(str(r.user_id)),
            "item_id": str(r.item_id),
            "answered_at": str(r.answered_at),
            "value": float(r.value),
            "raw": None if bool(redact_raw_payloads) else _parse_json_or_none(str(r.raw_json) if r.raw_json else None),
        }
        for r in answer_rows
    ]

    scores_out = [
        {
            "score_id": str(r.score_id),
            "user_id": uid(str(r.user_id)),
            "scale_id": str(r.scale_id),
            "window_end": r.window_end.isoformat(),
            "normalized_score": float(r.normalized_score),
            "personal_z": float(r.personal_z) if r.personal_z is not None else None,
            "risk_tier": str(r.risk_tier) if r.risk_tier is not None else None,
        }
        for r in score_rows
    ]

    states_out = [
        {
            "snapshot_id": str(r.snapshot_id),
            "user_id": uid(str(r.user_id)),
            "date": r.date.isoformat(),
            "state_schema_version": str(r.state_schema_version),
            "model_version": str(r.model_version),
            "x_hat": _parse_json_or_none(str(r.x_hat_json)),
            "x_uncertainty": _parse_json_or_none(str(r.x_uncertainty_json)),
        }
        for r in state_rows
    ]

    circles_out = [
        {
            "snapshot_id": str(r.snapshot_id),
            "user_id": uid(str(r.user_id)),
            "date": r.date.isoformat(),
            "projection_version": str(r.projection_version),
            "anchor_version": str(r.anchor_version),
            "r": float(r.r),
            "theta": float(r.theta),
            "velocity": float(r.velocity),
            "acceleration": float(r.acceleration),
        }
        for r in circle_rows
    ]

    decisions_out = [
        {
            "decision_id": str(r.decision_id),
            "user_id": uid(str(r.user_id)),
            "date": r.date.isoformat(),
            "selection_mode": str(r.selection_mode),
            "mode": str(r.mode),
            "policy_version": str(r.policy_version),
            "selected_item_ids": _parse_json_or_none(str(r.selected_item_ids_json)),
            "propensities": _parse_json_or_none(str(r.propensities_json) if r.propensities_json else None),
            "context": None if bool(redact_raw_payloads) else _parse_json_or_none(str(r.context_json) if r.context_json else None),
            "candidate_set": None if bool(redact_raw_payloads) else _parse_json_or_none(str(r.candidate_set_json) if r.candidate_set_json else None),
        }
        for r in decision_rows
    ]

    outcomes_out = [
        {
            "decision_id": str(r.decision_id),
            "user_id": uid(str(r.user_id)),
            "date": r.date.isoformat(),
            "completion_rate": float(r.completion_rate) if r.completion_rate is not None else None,
            "response_time_ms_median": float(r.response_time_ms_median) if r.response_time_ms_median is not None else None,
            "uncertainty_before_mean": float(r.uncertainty_before_mean) if r.uncertainty_before_mean is not None else None,
            "uncertainty_after_mean": float(r.uncertainty_after_mean) if r.uncertainty_after_mean is not None else None,
            "z_delta_norm": float(r.z_delta_norm) if r.z_delta_norm is not None else None,
        }
        for r in outcome_rows
    ]

    return {
        "generated_at": now_iso(),
        "window": {"start_date": start_day.isoformat(), "end_date": end_day.isoformat()},
        "redaction": {
            "redact_user_ids": bool(redact_user_ids),
            "redact_raw_payloads": bool(redact_raw_payloads),
        },
        "counts": {
            "sessions": len(sessions_out),
            "answers": len(answers_out),
            "scores": len(scores_out),
            "state_snapshots": len(states_out),
            "circle_snapshots": len(circles_out),
            "policy_decisions": len(decisions_out),
            "policy_outcomes": len(outcomes_out),
        },
        "sessions": sessions_out,
        "answers": answers_out,
        "scores": scores_out,
        "state_snapshots": states_out,
        "circle_snapshots": circles_out,
        "policy_decisions": decisions_out,
        "policy_outcomes": outcomes_out,
    }


def _upsert_daily_state_and_circle_snapshots_pg(
    session: Session,
    *,
    registry: Registry,
    user_id: str,
    day: date,
    state_schema_version: str,
    drift_routing_table_path: Optional[str] = None,
    site_config: Optional[SiteConfig] = None,
) -> Dict[str, Any]:
    latest_scores = _get_latest_scale_scores_for_state_pg(session, user_id=user_id, day=day)
    prev_state = _get_latest_state_snapshot_before_pg(session, user_id=user_id, day=day)
    prev_circle = _get_latest_circle_snapshot_before_pg(session, user_id=user_id, day=day)
    timestamp = date_to_start_iso(day)

    computed_state = compute_state_snapshot(
        registry=registry,
        latest_scores=latest_scores,
        previous_x_hat=(prev_state.get("x_hat") if prev_state else None),
    )

    state_model_version = STATE_MODEL_VERSION
    state_row = session.execute(
        select(StateSnapshot).where(
            and_(
                StateSnapshot.user_id == user_id,
                StateSnapshot.date == day,
                StateSnapshot.state_schema_version == str(state_schema_version or STATE_SCHEMA_VERSION_DEFAULT),
                StateSnapshot.model_version == state_model_version,
            )
        )
    ).scalars().first()
    if state_row is None:
        state_row = StateSnapshot(
            snapshot_id=str(uuid.uuid4()),
            user_id=user_id,
            date=day,
            timestamp=timestamp,
            state_schema_version=str(state_schema_version or STATE_SCHEMA_VERSION_DEFAULT),
            model_version=state_model_version,
            x_hat_json=json.dumps(computed_state["x_hat"], ensure_ascii=False),
            x_uncertainty_json=json.dumps(computed_state["x_uncertainty"], ensure_ascii=False),
            coverage_json=json.dumps(computed_state["coverage"], ensure_ascii=False),
            source="questions_agent",
            created_at=now_iso(),
        )
        session.add(state_row)
    else:
        state_row.timestamp = timestamp
        state_row.x_hat_json = json.dumps(computed_state["x_hat"], ensure_ascii=False)
        state_row.x_uncertainty_json = json.dumps(computed_state["x_uncertainty"], ensure_ascii=False)
        state_row.coverage_json = json.dumps(computed_state["coverage"], ensure_ascii=False)
        state_row.source = "questions_agent"
        state_row.created_at = now_iso()

    computed_circle = compute_circle_snapshot(
        day=day,
        x_hat=dict(computed_state["x_hat"]),
        x_uncertainty=dict(computed_state["x_uncertainty"]),
        previous_circle=prev_circle,
    )
    coherence_score = coherence_score_from_radius(float(computed_circle.get("r") or 0.0))
    coherence_tier = coherence_tier_for_score(float(coherence_score))
    projection_version = CIRCLE_PROJECTION_VERSION
    circle_row = session.execute(
        select(CircleSnapshot).where(
            and_(
                CircleSnapshot.user_id == user_id,
                CircleSnapshot.date == day,
                CircleSnapshot.projection_version == projection_version,
            )
        )
    ).scalars().first()
    if circle_row is None:
        circle_row = CircleSnapshot(
            snapshot_id=str(uuid.uuid4()),
            user_id=user_id,
            date=day,
            timestamp=timestamp,
            state_snapshot_id=state_row.snapshot_id,
            projection_version=projection_version,
            anchor_version=CIRCLE_ANCHOR_VERSION,
            z_json=json.dumps(computed_circle["z"], ensure_ascii=False),
            z_star_json=json.dumps(computed_circle["z_star"], ensure_ascii=False),
            r=float(computed_circle["r"]),
            theta=float(computed_circle["theta"]),
            velocity=float(computed_circle["velocity"]),
            acceleration=float(computed_circle["acceleration"]),
            coherence_score=float(coherence_score),
            coherence_tier_json=json.dumps(coherence_tier, ensure_ascii=False),
            uncertainty_json=json.dumps(computed_circle["uncertainty"], ensure_ascii=False),
            source="questions_agent",
            created_at=now_iso(),
        )
        session.add(circle_row)
    else:
        circle_row.timestamp = timestamp
        circle_row.state_snapshot_id = state_row.snapshot_id
        circle_row.anchor_version = CIRCLE_ANCHOR_VERSION
        circle_row.z_json = json.dumps(computed_circle["z"], ensure_ascii=False)
        circle_row.z_star_json = json.dumps(computed_circle["z_star"], ensure_ascii=False)
        circle_row.r = float(computed_circle["r"])
        circle_row.theta = float(computed_circle["theta"])
        circle_row.velocity = float(computed_circle["velocity"])
        circle_row.acceleration = float(computed_circle["acceleration"])
        circle_row.coherence_score = float(coherence_score)
        circle_row.coherence_tier_json = json.dumps(coherence_tier, ensure_ascii=False)
        circle_row.uncertainty_json = json.dumps(computed_circle["uncertainty"], ensure_ascii=False)
        circle_row.source = "questions_agent"
        circle_row.created_at = now_iso()

    _upsert_ews_and_drift_pg(
        session,
        registry=registry,
        user_id=user_id,
        day=day,
        computed_circle=computed_circle,
        latest_scores=latest_scores,
        drift_routing_table_path=drift_routing_table_path,
        site_config=site_config,
    )

    session.flush()
    return {
        "state_snapshot_id": str(state_row.snapshot_id),
        "circle_snapshot_id": str(circle_row.snapshot_id),
        "projection_version": str(circle_row.projection_version),
        "state_schema_version": str(state_row.state_schema_version),
    }


def _upsert_ews_and_drift_pg(
    session: Session,
    *,
    registry: Registry,
    user_id: str,
    day: date,
    computed_circle: Dict[str, Any],
    latest_scores: Dict[str, Dict[str, Any]],
    drift_routing_table_path: Optional[str] = None,
    site_config: Optional[SiteConfig] = None,
) -> None:
    anamnesis_enabled = bool(site_config and site_config.anamnesis_enabled)
    lookback_days = 14
    start = day - timedelta(days=lookback_days - 1)
    rows = session.execute(
        select(CircleSnapshot.date, CircleSnapshot.r, CircleSnapshot.velocity, CircleSnapshot.acceleration)
        .where(and_(CircleSnapshot.user_id == user_id, CircleSnapshot.date >= start, CircleSnapshot.date < day))
        .order_by(CircleSnapshot.date.asc(), CircleSnapshot.created_at.asc())
    ).all()

    history: List[Dict[str, Any]] = [
        {
            "date": d.isoformat(),
            "r": float(r),
            "velocity": float(v),
            "acceleration": float(a),
        }
        for d, r, v, a in rows
    ]
    history.append(
        {
            "date": day.isoformat(),
            "r": float(computed_circle.get("r") or 0.0),
            "velocity": float(computed_circle.get("velocity") or 0.0),
            "acceleration": float(computed_circle.get("acceleration") or 0.0),
        }
    )

    ews = compute_ews_features(circle_history=history, window_size_days=lookback_days)
    ews_row = session.execute(
        select(EwsFeature).where(
            and_(
                EwsFeature.user_id == user_id,
                EwsFeature.date == day,
                EwsFeature.window_size_days == lookback_days,
            )
        )
    ).scalars().first()
    if ews_row is None:
        ews_row = EwsFeature(
            feature_id=str(uuid.uuid4()),
            user_id=user_id,
            date=day,
            timestamp=date_to_start_iso(day),
            window_size_days=lookback_days,
            var_r=float(ews["var_r"]),
            ac1_r=float(ews["ac1_r"]),
            trend_speed=float(ews["trend_speed"]),
            trend_accel=float(ews["trend_accel"]),
            recovery_rate=float(ews["recovery_rate"]),
            ews_score=float(ews["ews_score"]),
            notes_json=json.dumps({"version": "ews_v1", "coherence_radius_window_days": lookback_days}, ensure_ascii=False),
            created_at=now_iso(),
        )
        session.add(ews_row)
    else:
        ews_row.timestamp = date_to_start_iso(day)
        ews_row.var_r = float(ews["var_r"])
        ews_row.ac1_r = float(ews["ac1_r"])
        ews_row.trend_speed = float(ews["trend_speed"])
        ews_row.trend_accel = float(ews["trend_accel"])
        ews_row.recovery_rate = float(ews["recovery_rate"])
        ews_row.ews_score = float(ews["ews_score"])
        ews_row.notes_json = json.dumps({"version": "ews_v1", "coherence_radius_window_days": lookback_days}, ensure_ascii=False)
        ews_row.created_at = now_iso()

    radius = float(computed_circle.get("r") or 0.0)
    coherence_score = coherence_score_from_radius(radius)
    baseline_r_vals = [float(h["r"]) for h in history[:-1]]
    if baseline_r_vals:
        mu = sum(baseline_r_vals) / float(len(baseline_r_vals))
        if len(baseline_r_vals) >= 2:
            var = sum((v - mu) ** 2 for v in baseline_r_vals) / float(len(baseline_r_vals) - 1)
            sigma = var ** 0.5
        else:
            sigma = 0.0
    else:
        mu = 0.0
        sigma = 0.0
    velocity = float(computed_circle.get("velocity") or 0.0)
    is_drift = bool(
        float(ews["ews_score"]) >= 0.65
        or (sigma > 0 and radius > (mu + 2.0 * sigma))
        or abs(velocity) >= 0.20
    )

    existing_drift = session.execute(
        select(DriftEvent).where(and_(DriftEvent.user_id == user_id, DriftEvent.date == day))
    ).scalars().first()
    if not is_drift:
        if existing_drift is not None:
            session.delete(existing_drift)
        # Even without drift, evaluate existing active episodes for resolution
        if anamnesis_enabled:
            _evaluate_anamnesis_resolutions_pg(
                session, user_id=user_id, day=day, ews_score=float(ews["ews_score"]),
            )
        return

    drift_domain = "general"
    triggered_instrument: Optional[str] = None
    reason_codes: List[str] = []
    if float(ews["ews_score"]) >= 0.65:
        reason_codes.append("ews_high")
    if sigma > 0 and radius > (mu + 2.0 * sigma):
        reason_codes.append("radius_above_personal_baseline")
    if abs(velocity) >= 0.20:
        reason_codes.append("velocity_spike")

    route_version = "drift_routing_v1"
    route_info: Dict[str, Any] = {}
    try:
        route_version, routes = load_drift_routes(drift_routing_table_path)
    except Exception:
        route_version, routes = load_drift_routes(None)
        reason_codes.append("drift_route_config_fallback")
    if latest_scores:
        scale_domains = registry_scale_domains(registry)
        route_info = resolve_drift_route(
            latest_scores=latest_scores,
            scale_domains=scale_domains,
            routes=routes,
        )
        drift_domain = str(route_info.get("drift_domain") or "general")
        triggered_instrument = (
            str(route_info.get("triggered_instrument"))
            if route_info.get("triggered_instrument")
            else None
        )
        reason_codes.append("scale_signal_alignment")
        reason_codes.append(f"drift_route:{route_info.get('route_id')}")

    payload_details = json.dumps(
        {
            "radius": radius,
            "baseline_radius_mean": round(float(mu), 6),
            "baseline_radius_std": round(float(sigma), 6),
            "velocity": velocity,
            "routing": {
                "version": route_version,
                "route_id": route_info.get("route_id"),
                "action": route_info.get("action"),
                "matched_scale_id": route_info.get("matched_scale_id"),
            },
        },
        ensure_ascii=False,
    )
    if existing_drift is None:
        session.add(
            DriftEvent(
                event_id=str(uuid.uuid4()),
                user_id=user_id,
                date=day,
                timestamp=date_to_start_iso(day),
                coherence_score=float(coherence_score),
                drift_domain=str(drift_domain),
                triggered_instrument=str(triggered_instrument) if triggered_instrument else None,
                ews_score=float(ews["ews_score"]),
                reason_codes_json=json.dumps(reason_codes, ensure_ascii=False),
                details_json=payload_details,
                created_at=now_iso(),
            )
        )
    else:
        existing_drift.timestamp = date_to_start_iso(day)
        existing_drift.coherence_score = float(coherence_score)
        existing_drift.drift_domain = str(drift_domain)
        existing_drift.triggered_instrument = str(triggered_instrument) if triggered_instrument else None
        existing_drift.ews_score = float(ews["ews_score"])
        existing_drift.reason_codes_json = json.dumps(reason_codes, ensure_ascii=False)
        existing_drift.details_json = payload_details
        existing_drift.created_at = now_iso()

    # --- Anamnesis: drift-triggered clinical branching (Claim Family 4) ---
    if anamnesis_enabled:
        _process_anamnesis_on_drift_pg(
            session,
            registry=registry,
            user_id=user_id,
            day=day,
            ews_score=float(ews["ews_score"]),
            baseline_radius_mean=float(mu),
            baseline_radius_std=float(sigma),
            baseline_radius_n=len(baseline_r_vals),
            radius=radius,
            velocity=velocity,
            drift_domain=drift_domain,
            drift_routing_table_path=drift_routing_table_path,
        )


# ---------------------------------------------------------------------------
# Anamnesis helpers (Claim Family 4: Drift-Triggered Branching)
# ---------------------------------------------------------------------------

def _load_active_anamnesis_episodes_pg(
    session: Session,
    *,
    user_id: str,
) -> List[AnamnesisEpisode]:
    """Load all active anamnesis episodes for a user."""
    rows = session.execute(
        select(AnamnesisEpisodeRow).where(
            and_(AnamnesisEpisodeRow.user_id == user_id, AnamnesisEpisodeRow.status == "active")
        )
    ).scalars().all()
    episodes: List[AnamnesisEpisode] = []
    for r in rows:
        try:
            scale_ids = tuple(json.loads(str(r.triggered_scale_ids_json or "[]")))
        except Exception:
            scale_ids = ()
        try:
            follow_ids = tuple(json.loads(str(r.follow_up_item_ids_json or "[]")))
        except Exception:
            follow_ids = ()
        episodes.append(AnamnesisEpisode(
            episode_id=str(r.episode_id),
            user_id=str(r.user_id),
            trigger_date=str(r.trigger_date),
            drift_domain=str(r.drift_domain),
            trigger_type=str(r.trigger_type),
            trigger_value=float(r.trigger_value),
            triggered_scale_ids=scale_ids,
            status=str(r.status),
            follow_up_item_ids=follow_ids,
            priority=int(r.priority),
            max_days=int(r.max_days),
            observations_collected=int(r.observations_collected),
            resolution_date=str(r.resolution_date) if r.resolution_date else None,
            resolution_reason=str(r.resolution_reason) if r.resolution_reason else None,
        ))
    return episodes


def _insert_anamnesis_episode_pg(
    session: Session,
    *,
    episode: AnamnesisEpisode,
) -> None:
    """Insert a new anamnesis episode row."""
    session.add(AnamnesisEpisodeRow(
        episode_id=episode.episode_id,
        user_id=episode.user_id,
        trigger_date=episode.trigger_date,
        drift_domain=episode.drift_domain,
        trigger_type=episode.trigger_type,
        trigger_value=float(episode.trigger_value),
        triggered_scale_ids_json=json.dumps(list(episode.triggered_scale_ids), ensure_ascii=False),
        status=episode.status,
        follow_up_item_ids_json=json.dumps(list(episode.follow_up_item_ids), ensure_ascii=False),
        priority=episode.priority,
        max_days=episode.max_days,
        observations_collected=episode.observations_collected,
        resolution_date=episode.resolution_date,
        resolution_reason=episode.resolution_reason,
        created_at=now_iso(),
        updated_at=now_iso(),
    ))


def _update_anamnesis_episode_status_pg(
    session: Session,
    *,
    episode_id: str,
    new_status: str,
    resolution_date: Optional[str] = None,
    resolution_reason: Optional[str] = None,
    observations_collected: Optional[int] = None,
) -> None:
    """Update the status of an existing anamnesis episode."""
    row = session.get(AnamnesisEpisodeRow, episode_id)
    if row is None:
        return
    row.status = new_status
    if resolution_date is not None:
        row.resolution_date = resolution_date
    if resolution_reason is not None:
        row.resolution_reason = resolution_reason
    if observations_collected is not None:
        row.observations_collected = observations_collected
    row.updated_at = now_iso()


def _evaluate_anamnesis_resolutions_pg(
    session: Session,
    *,
    user_id: str,
    day: date,
    ews_score: float,
) -> None:
    """Evaluate all active episodes for resolution when no drift is detected."""
    active_episodes = _load_active_anamnesis_episodes_pg(session, user_id=user_id)
    for ep in active_episodes:
        resolved_ep = evaluate_episode_resolution(
            ep,
            current_date=day,
            current_ews_score=ews_score,
        )
        if resolved_ep.status != "active":
            _update_anamnesis_episode_status_pg(
                session,
                episode_id=ep.episode_id,
                new_status=resolved_ep.status,
                resolution_date=resolved_ep.resolution_date,
                resolution_reason=resolved_ep.resolution_reason,
                observations_collected=resolved_ep.observations_collected,
            )


def _process_anamnesis_on_drift_pg(
    session: Session,
    *,
    registry: Registry,
    user_id: str,
    day: date,
    ews_score: float,
    baseline_radius_mean: float,
    baseline_radius_std: float,
    baseline_radius_n: int,
    radius: float,
    velocity: float,
    drift_domain: str,
    drift_routing_table_path: Optional[str] = None,
) -> None:
    """Process anamnesis logic when drift is detected."""
    active_episodes = _load_active_anamnesis_episodes_pg(session, user_id=user_id)

    # Check if there's already an active episode for this domain
    domain_episodes = [e for e in active_episodes if e.drift_domain == drift_domain]
    if domain_episodes:
        # Continue existing episode — no new episode needed
        return

    # Evaluate drift triggers using the anamnesis module
    from questions_agent_platform.pipeline.baseline import BaselineState as _BS
    radius_baseline = _BS(mean=baseline_radius_mean, var=baseline_radius_std ** 2 if baseline_radius_std > 0 else 0.0, n=max(1, baseline_radius_n))

    triggers = evaluate_drift_triggers(
        ews_score=ews_score,
        radius_current=radius,
        radius_baseline=radius_baseline,
        velocity=velocity,
    )
    if not triggers:
        return

    # Use the strongest trigger
    trigger_type, trigger_value = max(triggers, key=lambda t: abs(t[1]))

    # Determine which scales to probe for this drift domain
    scale_domains = {sid: list(getattr(s, "domains", [])) for sid, s in registry.scales.items()}
    triggered_scale_ids: List[str] = []
    for sid, domains in scale_domains.items():
        if drift_domain in domains or drift_domain == "general":
            triggered_scale_ids.append(sid)

    if not triggered_scale_ids:
        return

    # Collect follow-up item IDs from triggered scales
    follow_up_item_ids: List[str] = []
    for sid in triggered_scale_ids:
        scale = registry.scales.get(sid)
        if scale:
            for si in scale.items:
                if si.item_id not in follow_up_item_ids:
                    follow_up_item_ids.append(si.item_id)

    # Create the episode
    episode = create_anamnesis_episode(
        user_id=user_id,
        trigger_date=day,
        trigger_type=trigger_type,
        trigger_value=trigger_value,
        drift_domain=drift_domain,
        triggered_scale_ids=triggered_scale_ids,
        follow_up_item_ids=follow_up_item_ids,
        priority=max(1, min(100, int(50 - abs(trigger_value) * 10))),
        max_days=7,
    )
    _insert_anamnesis_episode_pg(session, episode=episode)

    # Enqueue follow-up items in the follow-up queue
    for item_id in follow_up_item_ids:
        _enqueue_follow_up_items_for_scale_pg(
            session,
            registry=registry,
            user_id=user_id,
            day=day,
            scale_id=triggered_scale_ids[0] if triggered_scale_ids else "unknown",
            answered_item_ids=set(),
            reason_code="anamnesis_probe",
            priority=float(episode.priority),
            reason_detail={
                "episode_id": episode.episode_id,
                "drift_domain": drift_domain,
                "trigger_type": trigger_type,
                "trigger_value": trigger_value,
            },
        )
        break  # _enqueue_follow_up_items_for_scale_pg handles all items for the scale


def _percentile(vals: Sequence[float], q: float) -> Optional[float]:
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    q_clamped = max(0.0, min(100.0, float(q)))
    rank = (q_clamped / 100.0) * (len(vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _hash_user_id(user_id: str, *, salt: str) -> str:
    token = f"{salt}:{user_id}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()[:24]


def _parse_json_or_none(raw: Optional[str]) -> Optional[Any]:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _get_latest_scale_scores_for_state_pg(
    session: Session,
    *,
    user_id: str,
    day: date,
) -> Dict[str, Dict[str, Any]]:
    rows = session.execute(
        select(
            ScaleScore.scale_id,
            ScaleScore.window_end,
            ScaleScore.normalized_score,
            ScaleScore.personal_z,
            ScaleScore.delta_vs_prev,
        )
        .where(and_(ScaleScore.user_id == user_id, ScaleScore.window_end <= day))
        .order_by(ScaleScore.window_end.asc(), ScaleScore.computed_at.asc())
    ).all()
    latest: Dict[str, Dict[str, Any]] = {}
    for scale_id, window_end, normalized_score, personal_z, delta_vs_prev in rows:
        latest[str(scale_id)] = {
            "scale_id": str(scale_id),
            "window_end": str(window_end.isoformat()),
            "normalized_score": float(normalized_score),
            "personal_z": float(personal_z) if personal_z is not None else None,
            "delta_vs_prev": float(delta_vs_prev) if delta_vs_prev is not None else None,
        }
    return latest


def _get_latest_state_snapshot_before_pg(
    session: Session,
    *,
    user_id: str,
    day: date,
) -> Optional[Dict[str, Any]]:
    row = session.execute(
        select(StateSnapshot)
        .where(and_(StateSnapshot.user_id == user_id, StateSnapshot.date < day))
        .order_by(StateSnapshot.date.desc(), StateSnapshot.created_at.desc())
        .limit(1)
    ).scalars().first()
    if not row:
        return None
    try:
        x_hat = json.loads(str(row.x_hat_json))
        if not isinstance(x_hat, dict):
            return None
        return {
            "date": row.date,
            "x_hat": {str(k): float(v) for k, v in x_hat.items()},
        }
    except Exception:
        return None


def _get_latest_circle_snapshot_before_pg(
    session: Session,
    *,
    user_id: str,
    day: date,
) -> Optional[Dict[str, Any]]:
    row = session.execute(
        select(CircleSnapshot)
        .where(and_(CircleSnapshot.user_id == user_id, CircleSnapshot.date < day))
        .order_by(CircleSnapshot.date.desc(), CircleSnapshot.created_at.desc())
        .limit(1)
    ).scalars().first()
    if not row:
        return None
    try:
        z = json.loads(str(row.z_json))
        z_star = json.loads(str(row.z_star_json))
        if not isinstance(z, list) or len(z) != 2 or not isinstance(z_star, list) or len(z_star) != 2:
            return None
        return {
            "date": row.date,
            "z": [float(z[0]), float(z[1])],
            "z_star": [float(z_star[0]), float(z_star[1])],
            "velocity": float(row.velocity or 0.0),
        }
    except Exception:
        return None


def _upsert_policy_outcome_from_answers_pg(
    session: Session,
    *,
    user_id: str,
    day: date,
    session_id: str,
    decision_id: str,
    core_item_ids: Tuple[str, ...],
    unlock_count: int,
) -> None:
    rows = session.execute(
        select(AnswerEvent.item_id, AnswerEvent.answered_at).where(AnswerEvent.session_id == session_id).order_by(AnswerEvent.answered_at.asc())
    ).all()
    answered_ids = {str(r[0]) for r in rows}
    completed_core = sum(1 for i in core_item_ids if str(i) in answered_ids)
    completion_rate = float(completed_core) / max(1, len(core_item_ids))

    times: List[datetime] = []
    for _, answered_at in rows:
        ts = _parse_ts_pg(str(answered_at))
        if ts is not None:
            times.append(ts)
    diffs_ms: List[float] = []
    for i in range(1, len(times)):
        dt = (times[i] - times[i - 1]).total_seconds() * 1000.0
        if dt >= 0:
            diffs_ms.append(float(dt))
    response_time_ms_median = median(diffs_ms) if diffs_ms else None

    from questions_agent_platform.policy.rewards import compute_reward

    # If Z uncertainty deltas were already attached, include them for the updated reward.
    existing = session.get(PolicyOutcomeRow, decision_id)
    reward = compute_reward(
        completion_rate=completion_rate,
        response_time_ms_median=response_time_ms_median,
        unlock_count=int(unlock_count),
        uncertainty_before_mean=float(existing.uncertainty_before_mean) if existing and existing.uncertainty_before_mean is not None else None,
        uncertainty_after_mean=float(existing.uncertainty_after_mean) if existing and existing.uncertainty_after_mean is not None else None,
    )

    outcome = existing
    if not outcome:
        outcome = PolicyOutcomeRow(
            decision_id=decision_id,
            user_id=user_id,
            date=day,
            updated_at=now_iso(),
        )
        session.add(outcome)

    outcome.completion_rate = float(completion_rate)
    outcome.completed_core_count = int(completed_core)
    outcome.response_time_ms_median = float(response_time_ms_median) if response_time_ms_median is not None else None
    outcome.reward_components_json = json.dumps(
        {
            "components": reward.components,
            "weights": reward.weights,
            "inputs": {
                "unlock_count": int(unlock_count),
                "response_time_ms_median": float(response_time_ms_median) if response_time_ms_median is not None else None,
            },
        },
        ensure_ascii=False,
    )
    outcome.total_reward = float(reward.total)
    outcome.updated_at = now_iso()


def _parse_ts_pg(ts: str) -> Optional[datetime]:
    try:
        s = str(ts)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def upsert_policy_outcome_update(
    session: Session,
    *,
    user_id: str,
    body: Dict[str, Any],
    store_z_snapshot: bool = False,
    store_z_delta: bool = True,
    quantize_decimals: int = 3,
) -> Dict[str, Any]:
    """
    Update (or create) a policy_outcomes row with delayed Anifold evidence updates.

    This endpoint is designed for the fusion/Anifold pipeline to attach Z+uncertainty
    deltas back to the selection decision_id for training and evaluation.
    """
    decision_id = str(body.get("decision_id") or "")
    if not decision_id:
        raise ValueError("Missing decision_id")

    decision = session.get(PolicyDecisionRow, decision_id)
    if not decision:
        raise ValueError("Unknown decision_id")
    if str(decision.user_id) != str(user_id):
        raise ValueError("decision_id does not belong to user_id")

    from questions_agent_platform.policy.hashing import stable_hash_floats

    z_before = body.get("z_before")
    z_after = body.get("z_after")
    u_before = body.get("uncertainty_before_diag")
    u_after = body.get("uncertainty_after_diag")

    def mean(v: Any) -> Optional[float]:
        if not isinstance(v, list) or not v:
            return None
        return float(sum(float(x) for x in v) / max(1, len(v)))

    def vmax(v: Any) -> Optional[float]:
        if not isinstance(v, list) or not v:
            return None
        try:
            return float(max(float(x) for x in v))
        except Exception:
            return None

    def norm(v: Any) -> Optional[float]:
        if not isinstance(v, list) or not v:
            return None
        try:
            s = sum(float(x) * float(x) for x in v)
            return float((s ** 0.5) / max(1.0, (len(v) ** 0.5)))
        except Exception:
            return None

    q = int(max(0, quantize_decimals))

    def qvec(v: Any) -> Optional[List[float]]:
        if not isinstance(v, list):
            return None
        out = []
        for x in v:
            try:
                out.append(round(float(x), q))
            except Exception:
                out.append(0.0)
        return out

    outcome = session.get(PolicyOutcomeRow, decision_id)
    if not outcome:
        outcome = PolicyOutcomeRow(
            decision_id=decision_id,
            user_id=user_id,
            date=decision.date,
            updated_at=now_iso(),
        )
        session.add(outcome)

    outcome.z_before_hash = stable_hash_floats(z_before, quantize=3) if isinstance(z_before, list) else None
    outcome.z_after_hash = stable_hash_floats(z_after, quantize=3) if isinstance(z_after, list) else None
    outcome.z_before_norm = norm(z_before)
    outcome.z_after_norm = norm(z_after)

    if bool(store_z_snapshot):
        zb = qvec(z_before)
        za = qvec(z_after)
        outcome.z_before_json = json.dumps(zb, ensure_ascii=False) if zb is not None else None
        outcome.z_after_json = json.dumps(za, ensure_ascii=False) if za is not None else None

    if bool(store_z_delta) and isinstance(z_before, list) and isinstance(z_after, list) and len(z_before) == len(z_after):
        delta = [float(a) - float(b) for a, b in zip(z_after, z_before)]
        outcome.z_delta_norm = norm(delta)
        outcome.z_delta_json = json.dumps([round(float(x), q) for x in delta], ensure_ascii=False)

    outcome.uncertainty_before_mean = mean(u_before)
    outcome.uncertainty_after_mean = mean(u_after)
    outcome.uncertainty_before_max = vmax(u_before)
    outcome.uncertainty_after_max = vmax(u_after)

    reward_overrides = body.get("reward_overrides") if isinstance(body.get("reward_overrides"), dict) else None

    # Recompute reward if we have adherence/burden info (usually populated by submit_answers).
    from questions_agent_platform.policy.rewards import compute_reward

    unlock_count = 0
    if outcome.reward_components_json:
        try:
            payload = json.loads(outcome.reward_components_json)
            if isinstance(payload, dict):
                inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
                unlock_count = int(inputs.get("unlock_count") or 0)
        except Exception:
            unlock_count = 0

    rr = compute_reward(
        completion_rate=float(outcome.completion_rate or 0.0),
        response_time_ms_median=float(outcome.response_time_ms_median) if outcome.response_time_ms_median is not None else None,
        unlock_count=int(unlock_count),
        uncertainty_before_mean=float(outcome.uncertainty_before_mean) if outcome.uncertainty_before_mean is not None else None,
        uncertainty_after_mean=float(outcome.uncertainty_after_mean) if outcome.uncertainty_after_mean is not None else None,
        overrides=reward_overrides,
    )
    outcome.reward_components_json = json.dumps(
        {
            "components": rr.components,
            "weights": rr.weights,
            "inputs": {
                "unlock_count": int(unlock_count),
                "response_time_ms_median": float(outcome.response_time_ms_median) if outcome.response_time_ms_median is not None else None,
            },
            "overrides": reward_overrides,
        },
        ensure_ascii=False,
    )
    outcome.total_reward = float(rr.total)

    outcome.updated_at = now_iso()
    return {"ok": True, "decision_id": decision_id}


def submit_observations(session: Session, *, user_id: str, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    ensure_user(session, user_id)
    inserted = 0
    coupling_outputs: List[Dict[str, Any]] = []
    for obs in observations:
        obs_type = str(obs.get("type") or "")
        if not obs_type:
            raise ValueError("Observation missing type")
        observed_at = str(obs.get("observed_at") or now_iso())
        features = obs.get("features") if obs.get("features") is not None else {}
        if not isinstance(features, dict):
            raise ValueError("Observation features must be an object")
        confidence = str(obs.get("confidence") or "medium")
        provenance = obs.get("provenance") if obs.get("provenance") is not None else {}
        if not isinstance(provenance, dict):
            raise ValueError("Observation provenance must be an object")

        observed_day = date.today()
        try:
            observed_day = parse_date(observed_at[:10])
        except Exception:
            observed_day = date.today()
        state_row = session.execute(
            select(StateSnapshot.x_hat_json, StateSnapshot.x_uncertainty_json)
            .where(and_(StateSnapshot.user_id == user_id, StateSnapshot.date <= observed_day))
            .order_by(StateSnapshot.date.desc(), StateSnapshot.created_at.desc())
            .limit(1)
        ).first()
        x_hat: Dict[str, float] = {}
        x_uncertainty: Dict[str, float] = {}
        if state_row:
            try:
                x_raw = json.loads(str(state_row[0]))
                if isinstance(x_raw, dict):
                    x_hat = {str(k): float(v) for k, v in x_raw.items()}
            except Exception:
                x_hat = {}
            try:
                unc_raw = json.loads(str(state_row[1]))
                if isinstance(unc_raw, dict):
                    x_uncertainty = {str(k): float(v) for k, v in unc_raw.items()}
            except Exception:
                x_uncertainty = {}

        quality_score = confidence_to_quality(str(confidence), obs.get("quality_score"))
        coupling = build_uncertainty_coupling(
            modality=obs_type,
            features=features,
            quality_score=quality_score,
            x_hat=x_hat,
            x_uncertainty=x_uncertainty,
        )
        event_id = str(uuid.uuid4())

        session.add(
            ObservationEvent(
                event_id=event_id,
                user_id=user_id,
                observed_at=observed_at,
                type=obs_type,
                features_json=json.dumps(features, ensure_ascii=False),
                confidence=confidence,
                provenance_json=json.dumps(provenance, ensure_ascii=False),
                coupling_json=json.dumps(coupling, ensure_ascii=False),
            )
        )
        coupling_outputs.append({"event_id": event_id, **coupling})
        inserted += 1
    return {"inserted_observation_events": inserted, "coupling_outputs": coupling_outputs}


def compute_user_scale_progress(session: Session, *, registry: Registry, user_id: str, day: date) -> Dict[str, Any]:
    answered_by_scale, _ = _compute_scale_windows(session, registry=registry, user_id=user_id, day=day)
    out: Dict[str, Any] = {}
    for scale in registry.scales.values():
        progress = compute_scale_progress(scale, tuple(sorted(answered_by_scale.get(scale.id, set()))))
        last = _get_last_scale_score(session, user_id=user_id, scale_id=scale.id)
        if last and not last.get("risk_tier"):
            last["risk_tier"] = infer_risk_tier_from_scale(
                scale,
                raw_score=float(last.get("raw_score") or 0.0),
            )
        unlocked = last is not None
        next_retest_due = None
        if last is not None:
            try:
                last_end = parse_date(last["window_end"])
                next_retest_due = (last_end + timedelta(days=int(scale.retest_interval_days))).isoformat()
            except Exception:
                next_retest_due = None
        unlock_hint = _build_unlock_hint(scale_name=scale.name, missing_count=int(progress["missing_count"]))
        latest_message = _build_latest_scale_message(scale_name=scale.name, risk_tier=last.get("risk_tier") if last else None)
        out[scale.id] = {
            "scale_id": scale.id,
            "name": scale.name,
            "questionnaire_id": scale.questionnaire_id,
            "unlocked": unlocked,
            "next_retest_due": next_retest_due,
            "answered_count": progress["answered_count"],
            "missing_count": progress["missing_count"],
            "items_required": progress["items_required"],
            "min_items_required": progress["min_items_required"],
            "unlock_hint": unlock_hint,
            "latest_message": latest_message,
            "last_score": last,
        }
    return out


def get_scale_history(session: Session, *, user_id: str, scale_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    rows = session.execute(
        select(ScaleScore)
        .where(and_(ScaleScore.user_id == user_id, ScaleScore.scale_id == scale_id))
        .order_by(ScaleScore.computed_at.asc())
        .limit(int(limit))
    ).scalars().all()
    return [_row_to_scale_score(r) for r in rows]


def list_users(session: Session) -> List[str]:
    rows = session.execute(select(User.user_id).order_by(User.created_at.desc())).all()
    return [str(r[0]) for r in rows]


def _get_last_asked_dates(session: Session, *, user_id: str, day: date, lookback_days: int) -> Dict[str, date]:
    start = day - timedelta(days=int(lookback_days))
    rows = session.execute(
        select(DailySession).where(and_(DailySession.user_id == user_id, DailySession.date >= start))
    ).scalars().all()
    last: Dict[str, date] = {}
    for ds in rows:
        d = ds.date
        for item_id in json.loads(ds.core_questions_json):
            last.setdefault(str(item_id), d)
        for batch in json.loads(ds.extra_batches_json):
            for item_id in batch:
                last.setdefault(str(item_id), d)
    return last


def _get_all_answered_item_ids_pg(session: Session, *, user_id: str) -> Set[str]:
    rows = session.execute(
        select(AnswerEvent.item_id).where(AnswerEvent.user_id == user_id).distinct()
    ).all()
    return {str(r[0]) for r in rows}


def _get_last_score_dates(session: Session, *, user_id: str) -> Dict[str, date]:
    rows = session.execute(
        select(ScaleScore.scale_id, func.max(ScaleScore.window_end))
        .where(ScaleScore.user_id == user_id)
        .group_by(ScaleScore.scale_id)
    ).all()
    out: Dict[str, date] = {}
    for scale_id, last_end in rows:
        if last_end:
            out[str(scale_id)] = last_end
    return out


def _get_baselines(session: Session, *, user_id: str) -> Dict[str, BaselineState]:
    rows = session.execute(
        select(ScaleBaseline).where(ScaleBaseline.user_id == user_id)
    ).scalars().all()
    out: Dict[str, BaselineState] = {}
    for r in rows:
        out[str(r.scale_id)] = BaselineState(mean=float(r.mean), var=float(r.var), n=int(r.n))
    return out


def _compute_scale_windows(
    session: Session, *, registry: Registry, user_id: str, day: date
) -> Tuple[Dict[str, Set[str]], Dict[str, float]]:
    answered_item_ids_by_scale: Dict[str, Set[str]] = {}
    rolling_normalized_by_scale: Dict[str, float] = {}
    end_iso = date_to_start_iso(day + timedelta(days=1))

    for scale in registry.scales.values():
        start_day = day - timedelta(days=int(scale.unlock_window_days) - 1)
        start_iso = date_to_start_iso(start_day)
        item_ids = [si.item_id for si in scale.items]
        latest = _fetch_latest_answers_for_items(
            session, user_id=user_id, item_ids=item_ids, start_iso=start_iso, end_iso=end_iso
        )
        answered_item_ids_by_scale[scale.id] = set(latest.keys())
        score = compute_scale_score(scale, latest)
        if score is not None:
            rolling_normalized_by_scale[scale.id] = float(score.normalized_score)

    return answered_item_ids_by_scale, rolling_normalized_by_scale


def _fetch_latest_answers_for_items(
    session: Session, *, user_id: str, item_ids: Sequence[str], start_iso: str, end_iso: str
) -> Dict[str, float]:
    if not item_ids:
        return {}
    rows = session.execute(
        select(AnswerEvent)
        .where(
            and_(
                AnswerEvent.user_id == user_id,
                AnswerEvent.item_id.in_(list(item_ids)),
                AnswerEvent.answered_at >= start_iso,
                AnswerEvent.answered_at < end_iso,
            )
        )
        .order_by(AnswerEvent.answered_at.desc())
    ).scalars().all()
    latest: Dict[str, float] = {}
    for r in rows:
        if r.item_id in latest:
            continue
        latest[r.item_id] = float(r.value)
    return latest


def _get_last_scale_score(session: Session, *, user_id: str, scale_id: str) -> Optional[Dict[str, Any]]:
    row = session.execute(
        select(ScaleScore)
        .where(and_(ScaleScore.user_id == user_id, ScaleScore.scale_id == scale_id))
        .order_by(ScaleScore.computed_at.desc())
        .limit(1)
    ).scalars().first()
    return _row_to_scale_score(row) if row else None


def _row_to_scale_score(row: ScaleScore) -> Dict[str, Any]:
    return {
        "score_id": row.score_id,
        "user_id": row.user_id,
        "scale_id": row.scale_id,
        "computed_at": row.computed_at,
        "window_start": row.window_start.isoformat(),
        "window_end": row.window_end.isoformat(),
        "raw_score": float(row.raw_score),
        "normalized_score": float(row.normalized_score),
        "confidence_tier": row.confidence_tier,
        "items_answered_count": int(row.items_answered_count),
        "items_required": int(row.items_required),
        "baseline_mean": float(row.baseline_mean) if row.baseline_mean is not None else None,
        "baseline_std": float(row.baseline_std) if row.baseline_std is not None else None,
        "personal_z": float(row.personal_z) if row.personal_z is not None else None,
        "delta_vs_prev": float(row.delta_vs_prev) if row.delta_vs_prev is not None else None,
        "risk_tier": str(row.risk_tier) if getattr(row, "risk_tier", None) is not None else None,
    }


def _resolve_site_config_obj(
    session: Session,
    user_id: str,
    *,
    site_config_id: Optional[str] = None,
) -> Optional[SiteConfig]:
    """Resolve SiteConfig for a user — explicit id > user profile > None."""
    registry = default_site_registry()
    config_id = site_config_id
    if not config_id:
        row = session.get(UserProfile, user_id)
        if row:
            config_id = str(getattr(row, "site_config_id", None) or "")
    if config_id:
        cfg = registry.get(config_id)
        if cfg:
            return cfg
    return registry.get("consumer")


def _insert_response_metadata_pg(
    session: Session,
    *,
    user_id: str,
    session_id: str,
    answer_event_id: str,
    metadata: ResponseMetadata,
    modifier: UncertaintyModifier,
) -> None:
    """Insert a response_metadata row for a single answer event."""
    session.add(ResponseMetadataRow(
        metadata_id=str(uuid.uuid4()),
        user_id=user_id,
        session_id=session_id,
        answer_event_id=answer_event_id,
        item_id=metadata.item_id,
        response_latency_ms=metadata.response_latency_ms,
        edit_count=metadata.edit_count,
        was_skipped=metadata.was_skipped,
        was_declined=metadata.was_declined,
        channel=metadata.channel,
        voice_hesitation_ms=metadata.voice_hesitation_ms,
        time_of_day_hour=metadata.time_of_day_hour,
        uncertainty_multiplier=float(modifier.multiplier),
        confidence_label=modifier.confidence_label,
        contributing_factors_json=json.dumps(list(modifier.contributing_factors), ensure_ascii=False),
        raw_components_json=json.dumps(modifier.raw_components, ensure_ascii=False),
        created_at=now_iso(),
    ))


def _upsert_session_uncertainty_profile_pg(
    session: Session,
    *,
    session_id: str,
    profile: "SessionUncertaintyProfile",
) -> None:
    """Insert or update the session-level uncertainty profile."""
    from questions_agent_platform.pipeline.response_metadata import SessionUncertaintyProfile as _SUP
    existing = session.execute(
        select(SessionUncertaintyProfileRow).where(
            and_(
                SessionUncertaintyProfileRow.user_id == profile.user_id,
                SessionUncertaintyProfileRow.session_id == session_id,
            )
        )
    ).scalars().first()
    if existing is None:
        session.add(SessionUncertaintyProfileRow(
            profile_id=str(uuid.uuid4()),
            user_id=profile.user_id,
            session_id=session_id,
            session_date=profile.session_date,
            session_multiplier=float(profile.session_multiplier),
            engagement_quality=profile.engagement_quality,
            median_latency_ms=float(profile.median_latency_ms),
            total_edits=profile.total_edits,
            skip_count=profile.skip_count,
            decline_count=profile.decline_count,
            item_count=len(profile.item_modifiers),
            created_at=now_iso(),
        ))
    else:
        existing.session_multiplier = float(profile.session_multiplier)
        existing.engagement_quality = profile.engagement_quality
        existing.median_latency_ms = float(profile.median_latency_ms)
        existing.total_edits = profile.total_edits
        existing.skip_count = profile.skip_count
        existing.decline_count = profile.decline_count
        existing.item_count = len(profile.item_modifiers)
        existing.created_at = now_iso()


def _load_item_modifiers_for_scale_pg(
    session: Session,
    *,
    session_id: str,
    scale_item_ids: Sequence[str],
) -> List[UncertaintyModifier]:
    """Load uncertainty modifiers for items belonging to a specific scale from this session."""
    if not scale_item_ids:
        return []
    rows = session.execute(
        select(ResponseMetadataRow).where(
            and_(
                ResponseMetadataRow.session_id == session_id,
                ResponseMetadataRow.item_id.in_(list(scale_item_ids)),
            )
        )
    ).scalars().all()
    modifiers: List[UncertaintyModifier] = []
    for r in rows:
        try:
            factors = tuple(json.loads(str(r.contributing_factors_json or "[]")))
        except Exception:
            factors = ()
        try:
            components = json.loads(str(r.raw_components_json or "{}"))
        except Exception:
            components = {}
        modifiers.append(UncertaintyModifier(
            item_id=str(r.item_id),
            multiplier=float(r.uncertainty_multiplier),
            confidence_label=str(r.confidence_label),
            contributing_factors=factors,
            raw_components=components,
        ))
    return modifiers


def _compute_and_store_scores(
    session: Session,
    *,
    registry: Registry,
    user_id: str,
    day: date,
    session_id: Optional[str] = None,
    site_config: Optional[SiteConfig] = None,
) -> List[Dict[str, Any]]:
    new_scores: List[Dict[str, Any]] = []
    end_iso = date_to_start_iso(day + timedelta(days=1))
    baselines = _get_baselines(session, user_id=user_id)
    metadata_enabled = bool(site_config and site_config.behavioural_metadata_enabled)

    # Load session engagement quality for SE adjustment
    _session_engagement: Optional[str] = None
    if metadata_enabled and session_id:
        sup_row = session.execute(
            select(SessionUncertaintyProfileRow.engagement_quality).where(
                SessionUncertaintyProfileRow.session_id == session_id
            )
        ).first()
        if sup_row:
            _session_engagement = str(sup_row[0])

    for scale in registry.scales.values():
        last = _get_last_scale_score(session, user_id=user_id, scale_id=scale.id)
        if last is not None:
            last_end = parse_date(last["window_end"])
            if (day - last_end).days < int(scale.retest_interval_days):
                continue

        start_day = day - timedelta(days=int(scale.unlock_window_days) - 1)
        start_iso = date_to_start_iso(start_day)
        item_ids = [si.item_id for si in scale.items]
        latest = _fetch_latest_answers_for_items(
            session, user_id=user_id, item_ids=item_ids, start_iso=start_iso, end_iso=end_iso
        )
        score_res = compute_scale_score(scale, latest)
        if score_res is None:
            continue

        exists = session.execute(
            select(ScaleScore.score_id).where(
                and_(ScaleScore.user_id == user_id, ScaleScore.scale_id == scale.id, ScaleScore.window_end == day)
            )
        ).first()
        if exists:
            continue

        baseline_before = baselines.get(scale.id)
        baseline_mean = baseline_before.mean if baseline_before else None
        baseline_std_val = baseline_std(baseline_before) if baseline_before else None
        personal_z = None
        if baseline_before and baseline_std_val and baseline_std_val >= 1e-6:
            personal_z = (score_res.normalized_score - baseline_before.mean) / baseline_std_val

        prev_score = last["normalized_score"] if last is not None else None
        delta_vs_prev = score_res.normalized_score - prev_score if prev_score is not None else None

        tier = confidence_tier(score_res.answered_count, score_res.items_required, scale.min_items_required)

        # --- Behavioural metadata SE adjustment (Claim Family 5) ---
        se_theta_val = getattr(score_res, "se_theta", None)
        se_theta_adj_val: Optional[float] = None
        uncertainty_mult_val: Optional[float] = None
        engagement_quality_val: Optional[str] = _session_engagement

        if metadata_enabled and se_theta_val is not None and session_id is not None:
            scale_item_ids = [si.item_id for si in scale.items]
            item_modifiers = _load_item_modifiers_for_scale_pg(
                session, session_id=session_id, scale_item_ids=scale_item_ids,
            )
            if item_modifiers:
                se_theta_adj_val = adjust_se_with_metadata(se_theta_val, item_modifiers)
                if se_theta_val > 0:
                    uncertainty_mult_val = round(se_theta_adj_val / se_theta_val, 4)

        score_id = str(uuid.uuid4())
        session.add(
            ScaleScore(
                score_id=score_id,
                user_id=user_id,
                scale_id=scale.id,
                computed_at=now_iso(),
                window_start=start_day,
                window_end=day,
                raw_score=float(score_res.raw_score),
                normalized_score=float(score_res.normalized_score),
                confidence_tier=tier,
                items_answered_count=int(score_res.answered_count),
                items_required=int(score_res.items_required),
                baseline_mean=float(baseline_mean) if baseline_mean is not None else None,
                baseline_std=float(baseline_std_val) if baseline_std_val is not None else None,
                personal_z=float(personal_z) if personal_z is not None else None,
                delta_vs_prev=float(delta_vs_prev) if delta_vs_prev is not None else None,
                risk_tier=str(score_res.risk_tier) if score_res.risk_tier else None,
                se_theta=float(se_theta_val) if se_theta_val is not None else None,
                se_theta_adjusted=float(se_theta_adj_val) if se_theta_adj_val is not None else None,
                uncertainty_multiplier=float(uncertainty_mult_val) if uncertainty_mult_val is not None else None,
                engagement_quality=engagement_quality_val,
            )
        )

        if baseline_before is None:
            baselines[scale.id] = BaselineState(mean=score_res.normalized_score, var=0.0, n=1)
        else:
            baselines[scale.id] = update_ewma_baseline(baseline_before, score_res.normalized_score, alpha=0.2)

        session.merge(
            ScaleBaseline(
                user_id=user_id,
                scale_id=scale.id,
                mean=float(baselines[scale.id].mean),
                var=float(baselines[scale.id].var),
                n=int(baselines[scale.id].n),
                updated_at=now_iso(),
            )
        )

        if personal_z is not None and abs(float(personal_z)) >= 1.5:
            _enqueue_follow_up_items_for_scale_pg(
                session,
                registry=registry,
                user_id=user_id,
                day=day,
                scale_id=scale.id,
                answered_item_ids=set(latest.keys()),
                reason_code="drift_followup",
                priority=min(100.0, 10.0 + 5.0 * abs(float(personal_z))),
                reason_detail={
                    "scale_id": scale.id,
                    "personal_z": float(personal_z),
                    "delta_vs_prev": float(delta_vs_prev) if delta_vs_prev is not None else None,
                },
            )
        elif personal_z is not None and abs(float(personal_z)) <= 0.75:
            _resolve_follow_up_queue_for_scale_pg(
                session,
                user_id=user_id,
                scale_id=scale.id,
                resolution_day=day,
                resolved_reason="stabilized",
            )

        score_payload = _get_last_scale_score(session, user_id=user_id, scale_id=scale.id) or {}
        if score_payload and not score_payload.get("risk_tier"):
            score_payload["risk_tier"] = infer_risk_tier_from_scale(
                scale,
                raw_score=float(score_payload.get("raw_score") or 0.0),
            )
        if score_payload:
            score_payload["unlock_message"] = _build_latest_scale_message(
                scale_name=scale.name,
                risk_tier=score_payload.get("risk_tier"),
            )
        new_scores.append(score_payload)

    return new_scores


def _cfg_stub(
    *,
    core_questions_per_day: int,
    extra_batch_size: int,
    extra_batches_max_per_day: int,
    item_repeat_cooldown_days: int,
    safety_min_questions_override: int,
):
    class _Cfg:
        def __init__(self):
            self.core_questions_per_day = core_questions_per_day
            self.extra_batch_size = extra_batch_size
            self.extra_batches_max_per_day = extra_batches_max_per_day
            self.item_repeat_cooldown_days = item_repeat_cooldown_days
            self.safety_min_questions_override = safety_min_questions_override

    return _Cfg()


def _build_unlock_hint(*, scale_name: str, missing_count: int) -> Optional[str]:
    if int(missing_count) <= 0:
        return None
    remaining = int(missing_count)
    suffix = "answer" if remaining == 1 else "answers"
    return f"{remaining} more {suffix} to unlock {scale_name}"


def _build_latest_scale_message(*, scale_name: str, risk_tier: Optional[str]) -> str:
    if not risk_tier:
        return f"{scale_name} unlocked"
    label = str(risk_tier).replace("_", " ")
    return f"{scale_name} unlocked ({label})"


def _list_versions(registry_root: str) -> List[str]:
    from questions_agent_platform.pipeline.registry import list_versions

    return list_versions(registry_root)


# ===========================================================================
# N-of-1 Personal Calibration — Production Service Functions
# ===========================================================================

def get_user_calibrations(
    session: Session,
    *,
    user_id: str,
) -> List[Dict[str, Any]]:
    """Return all personal calibration states for a user."""
    from questions_agent_platform.pipeline.n_of_1 import (
        calibration_from_dict,
        assess_reliable_change,
        assess_measurement_sufficiency,
    )
    from questions_agent_platform.prod.models import PersonalCalibrationRow

    rows = (
        session.execute(
            select(PersonalCalibrationRow).where(PersonalCalibrationRow.user_id == user_id)
        )
        .scalars()
        .all()
    )
    out: List[Dict[str, Any]] = []
    for row in rows:
        cal_dict = json.loads(row.calibration_json)
        cal = calibration_from_dict(cal_dict)
        entry: Dict[str, Any] = {
            "user_id": user_id,
            "scale_id": row.scale_id,
            "phase": row.phase,
            "n_observations": row.n_observations,
            "theta_personal": row.theta_personal,
            "se_personal": row.se_personal,
            "theta_baseline": row.theta_baseline,
            "se_baseline": row.se_baseline,
            "within_person_sd": row.within_person_sd,
            "shrinkage": row.shrinkage,
        }
        # Compute reliable change if we have a baseline
        if cal.theta_baseline is not None and cal.se_baseline is not None:
            rci = assess_reliable_change(cal)
            entry["reliable_change"] = {
                "rci": round(rci.rci, 4),
                "p_value": round(rci.p_value, 6),
                "significant": rci.significant,
                "direction": rci.direction,
                "magnitude": rci.magnitude,
                "delta_theta": round(rci.delta_theta, 4),
                "mid_exceeded": rci.mid_exceeded,
            }
        else:
            entry["reliable_change"] = None
        # Measurement sufficiency
        suff = assess_measurement_sufficiency(cal)
        entry["sufficiency"] = {
            "se_current": round(suff.se_current, 4),
            "se_target": round(suff.se_target, 4),
            "precision_ratio": round(suff.precision_ratio, 4),
            "sufficient": suff.sufficient,
            "reliability": round(suff.reliability, 4),
        }
        out.append(entry)
    return out


def upsert_calibration(
    session: Session,
    *,
    user_id: str,
    scale_id: str,
    theta_obs: float,
    se_obs: float,
    delta_t: float = 1.0,
) -> Dict[str, Any]:
    """Update the personal calibration for one (user, scale) with a new observation.

    Creates the calibration if it doesn't exist. Returns the updated calibration state.
    """
    from questions_agent_platform.pipeline.n_of_1 import (
        initial_calibration,
        update_calibration,
        calibration_to_dict,
        calibration_from_dict,
    )
    from questions_agent_platform.prod.models import PersonalCalibrationRow

    ensure_user(session, user_id)
    row = session.get(PersonalCalibrationRow, (user_id, scale_id))
    if row is None:
        cal = initial_calibration(scale_id)
    else:
        cal = calibration_from_dict(json.loads(row.calibration_json))

    cal = update_calibration(cal, theta_obs, se_obs, days_since_last=delta_t)
    cal_dict = calibration_to_dict(cal)

    if row is None:
        row = PersonalCalibrationRow(
            user_id=user_id,
            scale_id=scale_id,
            n_observations=cal.n_observations,
            phase=cal.phase,
            theta_personal=cal.theta_personal,
            se_personal=cal.se_personal,
            theta_baseline=cal.theta_baseline,
            se_baseline=cal.se_baseline,
            within_person_sd=cal.within_person_sd,
            shrinkage=cal.shrinkage,
            n_effective=cal.n_effective,
            calibration_json=json.dumps(cal_dict, ensure_ascii=False),
            updated_at=now_iso(),
        )
        session.add(row)
    else:
        row.n_observations = cal.n_observations
        row.phase = cal.phase
        row.theta_personal = cal.theta_personal
        row.se_personal = cal.se_personal
        row.theta_baseline = cal.theta_baseline
        row.se_baseline = cal.se_baseline
        row.within_person_sd = cal.within_person_sd
        row.shrinkage = cal.shrinkage
        row.n_effective = cal.n_effective
        row.calibration_json = json.dumps(cal_dict, ensure_ascii=False)
        row.updated_at = now_iso()

    session.flush()
    return {
        "user_id": user_id,
        "scale_id": scale_id,
        "phase": cal.phase,
        "n_observations": cal.n_observations,
        "theta_personal": round(cal.theta_personal, 6),
        "se_personal": round(cal.se_personal, 6),
    }


# ===========================================================================
# Coherence Detection — Production Service Functions
# ===========================================================================

def store_session_coherence(
    session: Session,
    *,
    user_id: str,
    session_id: str,
    day: date,
    assessment_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Store a coherence assessment for a session."""
    from questions_agent_platform.prod.models import SessionCoherence

    coherence_id = str(uuid.uuid4())
    row = SessionCoherence(
        coherence_id=coherence_id,
        user_id=user_id,
        session_id=session_id,
        date=day,
        coherence_score=float(assessment_dict["coherence_score"]),
        tier=str(assessment_dict["tier"]),
        n_flagged=int(assessment_dict["n_flagged"]),
        n_items=int(assessment_dict["n_items"]),
        signals_json=json.dumps(assessment_dict, ensure_ascii=False),
        created_at=now_iso(),
    )
    session.add(row)
    session.flush()
    return {
        "coherence_id": coherence_id,
        "coherence_score": float(assessment_dict["coherence_score"]),
        "tier": str(assessment_dict["tier"]),
        "n_flagged": int(assessment_dict["n_flagged"]),
    }


def get_session_coherence(
    session: Session,
    *,
    user_id: str,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve coherence assessment for a specific session."""
    from questions_agent_platform.prod.models import SessionCoherence

    row = (
        session.execute(
            select(SessionCoherence).where(
                SessionCoherence.user_id == user_id,
                SessionCoherence.session_id == session_id,
            )
        )
        .scalars()
        .first()
    )
    if not row:
        return None
    signals = json.loads(row.signals_json)
    return {
        "session_id": str(row.session_id),
        "user_id": str(row.user_id),
        "date": row.date.isoformat() if hasattr(row.date, "isoformat") else str(row.date),
        "coherence_score": float(row.coherence_score),
        "tier": str(row.tier),
        "n_flagged": int(row.n_flagged),
        "n_items": int(row.n_items),
        "signals": signals.get("signals", []),
    }


def get_coherence_history(
    session: Session,
    *,
    user_id: str,
    window_days: int = 30,
) -> List[Dict[str, Any]]:
    """Return coherence assessments for a user over a time window."""
    from questions_agent_platform.prod.models import SessionCoherence

    cutoff = date.today() - timedelta(days=max(1, window_days) - 1)
    rows = (
        session.execute(
            select(SessionCoherence)
            .where(SessionCoherence.user_id == user_id, SessionCoherence.date >= cutoff)
            .order_by(SessionCoherence.date.asc())
        )
        .scalars()
        .all()
    )
    out: List[Dict[str, Any]] = []
    for row in rows:
        signals = json.loads(row.signals_json)
        out.append({
            "session_id": str(row.session_id),
            "user_id": str(row.user_id),
            "date": row.date.isoformat() if hasattr(row.date, "isoformat") else str(row.date),
            "coherence_score": float(row.coherence_score),
            "tier": str(row.tier),
            "n_flagged": int(row.n_flagged),
            "n_items": int(row.n_items),
            "signals": signals.get("signals", []),
        })
    return out


# ===========================================================================
# Cardiometabolic Risk Index — Production Service Functions
# ===========================================================================

def compute_and_store_cardio_risk(
    session: Session,
    *,
    registry_root: str,
    user_id: str,
    day: date,
) -> Optional[Dict[str, Any]]:
    """Compute the composite cardiometabolic risk index from latest scale scores.

    Looks up the latest scores for all 6 cardio instruments and computes
    the weighted composite. Stores the result as a snapshot.
    """
    from questions_agent_platform.pipeline.cardio_risk_index import (
        compute_cardio_risk_index,
        cardio_risk_to_dict,
    )
    from questions_agent_platform.pipeline.scoring import ScaleScoreResult
    from questions_agent_platform.prod.models import CardioRiskSnapshot

    # Build instrument_scores from latest scale_scores
    instrument_scores: Dict[str, Any] = {}
    for scale_row in (
        session.execute(
            select(ScaleScore)
            .where(ScaleScore.user_id == user_id)
            .order_by(ScaleScore.computed_at.desc())
        )
        .scalars()
        .all()
    ):
        sid = str(scale_row.scale_id)
        # Map scale_id to scoring method if it's a cardio instrument
        method = _scale_id_to_cardio_method(sid)
        if method and method not in instrument_scores:
            instrument_scores[method] = ScaleScoreResult(
                raw_score=float(scale_row.raw_score),
                normalized_score=float(scale_row.normalized_score),
                answered_count=int(scale_row.items_answered_count),
                items_required=int(scale_row.items_required),
                risk_tier=scale_row.risk_tier,
            )

    index = compute_cardio_risk_index(instrument_scores)
    if index is None:
        return None

    index_dict = cardio_risk_to_dict(index)

    # Upsert snapshot
    existing = (
        session.execute(
            select(CardioRiskSnapshot).where(
                CardioRiskSnapshot.user_id == user_id,
                CardioRiskSnapshot.date == day,
            )
        )
        .scalars()
        .first()
    )
    if existing:
        existing.composite_risk = index.composite_risk
        existing.composite_risk_pct = index.composite_risk_pct
        existing.risk_tier = index.risk_tier
        existing.instruments_available = index.instruments_available
        existing.instruments_total = index.instruments_total
        existing.coverage = index.coverage
        existing.confidence = index.confidence
        existing.components_json = json.dumps(index_dict, ensure_ascii=False)
        existing.created_at = now_iso()
    else:
        snapshot = CardioRiskSnapshot(
            snapshot_id=str(uuid.uuid4()),
            user_id=user_id,
            date=day,
            composite_risk=index.composite_risk,
            composite_risk_pct=index.composite_risk_pct,
            risk_tier=index.risk_tier,
            instruments_available=index.instruments_available,
            instruments_total=index.instruments_total,
            coverage=index.coverage,
            confidence=index.confidence,
            components_json=json.dumps(index_dict, ensure_ascii=False),
            created_at=now_iso(),
        )
        session.add(snapshot)
    session.flush()
    return index_dict


def get_cardio_risk_history(
    session: Session,
    *,
    user_id: str,
    window_days: int = 90,
) -> List[Dict[str, Any]]:
    """Return cardio risk index snapshots over a time window."""
    from questions_agent_platform.prod.models import CardioRiskSnapshot

    cutoff = date.today() - timedelta(days=max(1, window_days) - 1)
    rows = (
        session.execute(
            select(CardioRiskSnapshot)
            .where(CardioRiskSnapshot.user_id == user_id, CardioRiskSnapshot.date >= cutoff)
            .order_by(CardioRiskSnapshot.date.asc())
        )
        .scalars()
        .all()
    )
    out: List[Dict[str, Any]] = []
    for row in rows:
        components = json.loads(row.components_json)
        out.append({
            "user_id": str(row.user_id),
            "date": row.date.isoformat() if hasattr(row.date, "isoformat") else str(row.date),
            "composite_risk": float(row.composite_risk),
            "composite_risk_pct": float(row.composite_risk_pct),
            "risk_tier": str(row.risk_tier),
            "instruments_available": int(row.instruments_available),
            "instruments_total": int(row.instruments_total),
            "coverage": float(row.coverage),
            "confidence": str(row.confidence),
            "components": components.get("components", []),
        })
    return out


def _scale_id_to_cardio_method(scale_id: str) -> Optional[str]:
    """Map a scale_id to its cardiometabolic scoring method name."""
    _MAP = {
        "scale_cm_findrisc": "instrument_findrisc",
        "scale_cm_ez_cvd": "instrument_ez_cvd",
        "scale_cm_ipaq_sf": "instrument_ipaq_sf",
        "scale_cm_lee_nafld": "instrument_lee_nafld",
        "scale_cm_scored": "instrument_scored",
        "scale_cm_audit_c": "instrument_audit_c",
    }
    return _MAP.get(scale_id)


# ---------------------------------------------------------------------------
# Site Configuration
# ---------------------------------------------------------------------------

def resolve_site_config(
    session: Session,
    *,
    user_id: str,
) -> Dict[str, Any]:
    """Resolve the active SiteConfig for a user.

    Returns a dict with the full config fields ready for API output.
    Falls back to 'consumer' if the user has no profile or an unknown config_id.
    """
    from questions_agent_platform.pipeline.site_config import (
        SiteConfig,
        default_site_registry,
    )

    registry = default_site_registry()

    # Get user's config_id from profile
    ensure_user_profile(session, user_id)
    row = session.get(UserProfile, user_id)
    config_id = str(getattr(row, "site_config_id", None) or "consumer")

    config = registry.get(config_id)
    if config is None:
        config = registry.get("consumer")

    return _site_config_to_dict(config)


def list_site_configs() -> List[Dict[str, Any]]:
    """Return all registered site configurations."""
    from questions_agent_platform.pipeline.site_config import default_site_registry

    registry = default_site_registry()
    return [_site_config_to_dict(c) for c in registry.list_configs()]


def _site_config_to_dict(config) -> Dict[str, Any]:
    """Convert a SiteConfig frozen dataclass to a serialisable dict."""
    return {
        "config_id": config.config_id,
        "display_name": config.display_name,
        "config_type": config.config_type,
        "registry_version": config.registry_version,
        "locale": config.locale,
        "show_clinical_thresholds": config.show_clinical_thresholds,
        "show_disease_names": config.show_disease_names,
        "show_risk_tiers": config.show_risk_tiers,
        "score_display_mode": config.score_display_mode,
        "wellness_language": config.wellness_language,
        "anifold_enabled": config.anifold_enabled,
        "concordance_mode": config.concordance_mode,
        "behavioural_metadata_enabled": config.behavioural_metadata_enabled,
        "anamnesis_enabled": config.anamnesis_enabled,
        "policy_mode": config.policy_mode,
        "daily_question_budget": config.daily_question_budget,
        "max_extra_batches": config.max_extra_batches,
        "item_repeat_cooldown_days": config.item_repeat_cooldown_days,
    }


def format_score_for_user(
    session: Session,
    *,
    user_id: str,
    scale_name: str,
    raw_score: float,
    normalized_score: float,
    risk_tier: Optional[str],
) -> Dict[str, Any]:
    """Format a scale score using the user's SiteConfig display rules."""
    from questions_agent_platform.pipeline.site_config import (
        default_site_registry,
        format_score_display,
    )

    registry = default_site_registry()
    ensure_user_profile(session, user_id)
    row = session.get(UserProfile, user_id)
    config_id = str(getattr(row, "site_config_id", None) or "consumer")
    config = registry.get(config_id) or registry.get("consumer")

    return format_score_display(
        config=config,
        scale_name=scale_name,
        raw_score=raw_score,
        normalized_score=normalized_score,
        risk_tier=risk_tier,
    )


# ---------------------------------------------------------------------------
# Anamnesis API service functions
# ---------------------------------------------------------------------------

def get_anamnesis_episodes(
    session: Session,
    *,
    user_id: str,
    status_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List anamnesis episodes for a user, optionally filtered by status."""
    ensure_user(session, user_id)
    q = select(AnamnesisEpisodeRow).where(AnamnesisEpisodeRow.user_id == user_id)
    if status_filter:
        q = q.where(AnamnesisEpisodeRow.status == status_filter)
    q = q.order_by(AnamnesisEpisodeRow.trigger_date.desc())
    rows = session.execute(q).scalars().all()
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            scale_ids = json.loads(str(r.triggered_scale_ids_json or "[]"))
        except Exception:
            scale_ids = []
        try:
            follow_ids = json.loads(str(r.follow_up_item_ids_json or "[]"))
        except Exception:
            follow_ids = []
        out.append({
            "episode_id": str(r.episode_id),
            "user_id": str(r.user_id),
            "trigger_date": str(r.trigger_date),
            "drift_domain": str(r.drift_domain),
            "trigger_type": str(r.trigger_type),
            "trigger_value": float(r.trigger_value),
            "triggered_scale_ids": scale_ids,
            "status": str(r.status),
            "follow_up_item_ids": follow_ids,
            "priority": int(r.priority),
            "max_days": int(r.max_days),
            "observations_collected": int(r.observations_collected),
            "resolution_date": str(r.resolution_date) if r.resolution_date else None,
            "resolution_reason": str(r.resolution_reason) if r.resolution_reason else None,
        })
    return out


def get_anamnesis_episode(
    session: Session,
    *,
    user_id: str,
    episode_id: str,
) -> Optional[Dict[str, Any]]:
    """Get a single anamnesis episode."""
    ensure_user(session, user_id)
    r = session.get(AnamnesisEpisodeRow, episode_id)
    if r is None or r.user_id != user_id:
        return None
    try:
        scale_ids = json.loads(str(r.triggered_scale_ids_json or "[]"))
    except Exception:
        scale_ids = []
    try:
        follow_ids = json.loads(str(r.follow_up_item_ids_json or "[]"))
    except Exception:
        follow_ids = []
    return {
        "episode_id": str(r.episode_id),
        "user_id": str(r.user_id),
        "trigger_date": str(r.trigger_date),
        "drift_domain": str(r.drift_domain),
        "trigger_type": str(r.trigger_type),
        "trigger_value": float(r.trigger_value),
        "triggered_scale_ids": scale_ids,
        "status": str(r.status),
        "follow_up_item_ids": follow_ids,
        "priority": int(r.priority),
        "max_days": int(r.max_days),
        "observations_collected": int(r.observations_collected),
        "resolution_date": str(r.resolution_date) if r.resolution_date else None,
        "resolution_reason": str(r.resolution_reason) if r.resolution_reason else None,
    }


def resolve_anamnesis_episode(
    session: Session,
    *,
    user_id: str,
    episode_id: str,
    resolved_by: str,
    resolved_reason: str,
) -> Dict[str, Any]:
    """Manually resolve an anamnesis episode."""
    ensure_user(session, user_id)
    r = session.get(AnamnesisEpisodeRow, episode_id)
    if r is None or r.user_id != user_id:
        raise ValueError("Episode not found")
    if r.status != "active":
        raise ValueError(f"Episode already resolved with status: {r.status}")
    r.status = "resolved"
    r.resolution_date = now_iso()[:10]
    r.resolution_reason = f"manual:{resolved_by}:{resolved_reason}"
    r.updated_at = now_iso()
    session.flush()
    return {"episode_id": episode_id, "status": "resolved", "resolution_reason": r.resolution_reason}


def get_session_engagement_profile(
    session: Session,
    *,
    user_id: str,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve engagement profile for a specific session."""
    ensure_user(session, user_id)
    row = session.execute(
        select(SessionUncertaintyProfileRow).where(
            and_(
                SessionUncertaintyProfileRow.user_id == user_id,
                SessionUncertaintyProfileRow.session_id == session_id,
            )
        )
    ).scalars().first()
    if row is None:
        return None
    return {
        "session_id": str(row.session_id),
        "user_id": str(row.user_id),
        "session_date": str(row.session_date),
        "session_multiplier": float(row.session_multiplier),
        "engagement_quality": str(row.engagement_quality),
        "median_latency_ms": float(row.median_latency_ms),
        "total_edits": int(row.total_edits),
        "skip_count": int(row.skip_count),
        "decline_count": int(row.decline_count),
        "item_count": int(row.item_count),
    }


def get_follow_up_queue(
    session: Session,
    *,
    user_id: str,
) -> List[Dict[str, Any]]:
    """Get all pending follow-up queue items for a user."""
    ensure_user(session, user_id)
    from questions_agent_platform.prod.models import FollowUpQueue
    rows = session.execute(
        select(FollowUpQueue).where(
            and_(FollowUpQueue.user_id == user_id, FollowUpQueue.status == "pending")
        ).order_by(FollowUpQueue.priority.asc())
    ).scalars().all()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "queue_id": str(r.queue_id),
            "user_id": str(r.user_id),
            "item_id": str(r.item_id),
            "scale_id": str(r.scale_id),
            "reason_code": str(r.reason_code),
            "priority": float(r.priority),
            "status": str(r.status),
            "created_at": str(r.created_at),
        })
    return out
