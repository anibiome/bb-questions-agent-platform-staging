from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from questions_agent_platform.pipeline.anamnesis import (
    AnamnesisEpisode,
    create_anamnesis_episode,
    evaluate_episode_resolution,
    run_anamnesis_check,
)
from questions_agent_platform.pipeline.baseline import BaselineState, baseline_std, update_ewma_baseline
from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import (
    ensure_user,
    get_active_registry_version,
    set_registry_version_active,
    upsert_registry_version,
)
from questions_agent_platform.pipeline.drift_routing import load_drift_routes, resolve_drift_route, route_contract
from questions_agent_platform.pipeline.follow_up_queue import (
    enqueue_follow_up_queue_item as _enqueue_follow_up_queue_item,
    get_due_follow_up_item_ids as _get_due_follow_up_item_ids,
)
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
from questions_agent_platform.pipeline.scoring import (
    compute_scale_progress,
    compute_scale_score,
    confidence_tier,
    infer_risk_tier_from_scale,
)
from questions_agent_platform.pipeline.item_efficiency import ItemEfficiencyAnalyzer
from questions_agent_platform.pipeline.selection import SelectionPlan, build_candidate_set, build_selection_plan
from questions_agent_platform.pipeline.site_config import SiteConfig, SiteConfigRegistry, config_consumer_wellness
from questions_agent_platform.pipeline.measurement_packages import (
    DecoherenceMode,
    evaluate_package_completion,
    get_package_scale_ids,
    get_package_scale_priorities,
    get_progressive_plan,
    resolve_current_package,
)
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
    MINIFOLD_VERSION,
    STATE_DIMENSIONS,
    STATE_MODEL_VERSION,
    STATE_SCHEMA_VERSION_DEFAULT,
    compute_circle_snapshot,
    compute_minifold_circles,
    compute_state_snapshot,
    dominant_decoherence_mode,
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

logger = logging.getLogger("questions_agent.service")


# ── Shadow Efficiency Cache ──────────────────────────────────────────────
# Computed once per registry version, cached in memory. The analyzer runs
# registry-only analysis (semantic + tag overlap) which requires no response
# data and completes in ~2 seconds for 941 items. The cache is keyed by
# registry version string, so it invalidates when the registry is updated.
_efficiency_cache: Dict[str, Dict[str, float]] = {}


def _boolish(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"false", "0", "no", "n", "off", "none", "null"}:
        return False
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    return default


def _normalize_string_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    elif value is None:
        raw_items = ()
    else:
        raw_items = (value,)
    out: List[str] = []
    for item in raw_items:
        token = str(item or "").strip()
        if token:
            out.append(token)
    return out


def _get_efficiency_scores(registry: "Registry", version: str) -> Optional[Dict[str, float]]:
    """Return item_id → efficiency_score map, cached per registry version."""
    if version in _efficiency_cache:
        return _efficiency_cache[version]
    try:
        items_raw = [
            {
                "id": item.id,
                "text": item.text,
                "response_type": item.response_type,
                "tags": list(item.tags),
            }
            for item in registry.items.values()
        ]
        scales_raw = [
            {
                "id": scale.id,
                "items": [{"item_id": si.item_id} for si in scale.items],
                "tags": list(scale.tags),
            }
            for scale in registry.scales.values()
        ]
        analyzer = ItemEfficiencyAnalyzer(
            {"items": items_raw, "scales": scales_raw, "multiplex_map": {}}
        )
        report = analyzer.analyze(
            semantic=True,
            tag_overlap=True,
            correlation=False,  # no response data in shadow mode yet
            irt_discrimination=False,
        )
        scores = {e.item_id: e.efficiency_score for e in report.items}
        _efficiency_cache[version] = scores
        logger.info("efficiency_shadow: computed scores for %d items (registry %s)",
                     len(scores), version)
        return scores
    except Exception as exc:
        logger.warning("efficiency_shadow: failed to compute scores: %s", exc)
        return None


@dataclass(frozen=True)
class DailySession:
    session_id: str
    user_id: str
    date: str
    registry_version: str
    timeframe: str
    status: str
    core_item_ids: Tuple[str, ...]
    extra_batches: Tuple[Tuple[str, ...], ...]
    extra_batches_used: int
    selection_explain: Dict[str, Any]
    selection_mode: str = "deterministic"
    policy_decision_id: Optional[str] = None
    pool_exhaustion: Optional[Dict[str, Any]] = None


def ensure_registry_active(conn: sqlite3.Connection, registry_root: str) -> str:
    active = get_active_registry_version(conn)
    if active:
        return active

    versions = _list_versions(registry_root)
    if not versions:
        raise ValueError("No registry versions found. Seed or upload a registry first.")
    active = versions[-1]
    upsert_registry_version(conn, active, status="active")
    set_registry_version_active(conn, active)
    conn.commit()
    return active


def upload_registry_bundle(conn: sqlite3.Connection, registry_root: str, bundle: Dict[str, Any]) -> str:
    registry = validate_registry_bundle(bundle)
    save_registry(registry_root, registry)
    upsert_registry_version(conn, registry.version, status="inactive")
    conn.commit()
    return registry.version


def activate_registry_version(conn: sqlite3.Connection, registry_root: str, version: str) -> None:
    versions = set(_list_versions(registry_root))
    if version not in versions:
        raise ValueError(f"Unknown registry version: {version}")
    set_registry_version_active(conn, version)
    conn.commit()


def _resolve_site_config(
    conn: sqlite3.Connection,
    user_id: str,
    site_config: Optional[SiteConfig] = None,
    default_config_id: str = "consumer",
) -> SiteConfig:
    """
    Resolve the active SiteConfig for a user.

    Priority:
    1. Explicit site_config parameter (caller override)
    2. user_profiles.site_config_id from DB
    3. default_config_id fallback
    """
    if site_config is not None:
        return site_config

    # Try to read from user profile
    row = conn.execute(
        "SELECT site_config_id FROM user_profiles WHERE user_id=? LIMIT 1;",
        (user_id,),
    ).fetchone()
    config_id = default_config_id
    if row and row["site_config_id"]:
        config_id = str(row["site_config_id"]).strip() or default_config_id

    # Resolve from registry
    registry = SiteConfigRegistry()
    resolved = registry.get(config_id)
    if resolved is not None:
        return resolved

    # Fallback to consumer
    return config_consumer_wellness()


def ensure_user_profile(conn: sqlite3.Connection, user_id: str) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO user_profiles(
          user_id, mode, permanently_declined_items_json,
          active_domains_json, queued_domains_json, promoted_domains_json,
          onboarding_complete, state_schema_version, updated_at
        ) VALUES (?, 'consumer', '[]', '["cardiometabolic"]', '[]', '["cardiometabolic"]', 0, 'v1_state_schema', ?);
        """,
        (user_id, now_iso()),
    )


def get_user_profile(conn: sqlite3.Connection, user_id: str) -> Dict[str, Any]:
    ensure_user_profile(conn, user_id)
    row = conn.execute(
        """
        SELECT mode, permanently_declined_items_json, active_domains_json, queued_domains_json,
               promoted_domains_json, onboarding_complete, state_schema_version
        FROM user_profiles
        WHERE user_id=?;
        """,
        (user_id,),
    ).fetchone()
    if not row:
        return {
            "mode": "consumer",
            "permanently_declined_item_ids": [],
            "active_domains": [CARDIOMETABOLIC_DOMAIN],
            "queued_domains": [],
            "promoted_domains": [CARDIOMETABOLIC_DOMAIN],
            "onboarding_complete": False,
            "state_schema_version": "v1_state_schema",
        }
    declined: List[str] = []
    try:
        raw_declined = json.loads(str(row["permanently_declined_items_json"] or "[]"))
        if isinstance(raw_declined, list):
            declined = [str(x) for x in raw_declined if str(x or "").strip()]
    except Exception:
        declined = []
    active_domains = parse_json_str_list(
        row["active_domains_json"],
        fallback=[CARDIOMETABOLIC_DOMAIN],
    )
    queued_domains = parse_json_str_list(row["queued_domains_json"], fallback=[])
    promoted_domains = parse_json_str_list(
        row["promoted_domains_json"],
        fallback=[CARDIOMETABOLIC_DOMAIN],
    )
    return {
        "mode": str(row["mode"] or "consumer"),
        "permanently_declined_item_ids": declined,
        "active_domains": active_domains or [CARDIOMETABOLIC_DOMAIN],
        "queued_domains": queued_domains,
        "promoted_domains": promoted_domains or [CARDIOMETABOLIC_DOMAIN],
        "onboarding_complete": bool(int(row["onboarding_complete"] or 0)),
        "state_schema_version": str(row["state_schema_version"] or "v1_state_schema"),
    }


def upsert_user_profile(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    patch: Dict[str, Any],
    registry: Registry,
    cfg: QuestionsAgentConfig,
) -> Dict[str, Any]:
    ensure_user(conn, user_id)
    ensure_user_profile(conn, user_id)
    profile = get_user_profile(conn, user_id)

    mode = str(patch.get("mode") or profile.get("mode") or "consumer").strip().lower()
    if mode not in {"trial", "consumer"}:
        raise ValueError("mode must be one of: trial, consumer")

    active_domains = profile.get("active_domains") or [CARDIOMETABOLIC_DOMAIN]
    queued_domains = profile.get("queued_domains") or []
    promoted_domains = profile.get("promoted_domains") or [CARDIOMETABOLIC_DOMAIN]
    onboarding_complete = _boolish(profile.get("onboarding_complete", False), default=False)
    if "onboarding_complete" in patch:
        onboarding_complete = _boolish(patch.get("onboarding_complete"), default=False)

    if "active_domains" in patch:
        active_domains = [canonical_domain_id(str(x)) for x in _normalize_string_list(patch.get("active_domains"))]
    if "queued_domains" in patch:
        queued_domains = [canonical_domain_id(str(x)) for x in _normalize_string_list(patch.get("queued_domains"))]
    if "promoted_domains" in patch:
        promoted_domains = [canonical_domain_id(str(x)) for x in _normalize_string_list(patch.get("promoted_domains"))]

    conn.execute(
        """
        UPDATE user_profiles
        SET mode=?,
            onboarding_complete=?,
            active_domains_json=?,
            queued_domains_json=?,
            promoted_domains_json=?,
            updated_at=?
        WHERE user_id=?;
        """,
        (
            mode,
            1 if onboarding_complete else 0,
            encode_json_str_list(active_domains),
            encode_json_str_list(queued_domains),
            encode_json_str_list(promoted_domains),
            now_iso(),
            user_id,
        ),
    )
    conn.commit()
    refreshed = get_user_profile(conn, user_id)
    synced = _sync_user_domain_profile(
        conn,
        user_id=user_id,
        profile=refreshed,
        registry=registry,
        cfg=cfg,
    )
    return synced


def list_safety_events(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
    lookback_days: int,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    start = (day - timedelta(days=max(1, int(lookback_days)) - 1)).isoformat()
    clauses = ["user_id=?", "date>=?", "date<=?"]
    params: List[Any] = [user_id, start, day.isoformat()]
    if status:
        clauses.append("status=?")
        params.append(str(status))
    rows = conn.execute(
        f"""
        SELECT *
        FROM safety_events
        WHERE {" AND ".join(clauses)}
        ORDER BY date DESC, created_at DESC;
        """,
        tuple(params),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            details = json.loads(str(row["details_json"] or "{}"))
        except Exception:
            details = {}
        out.append(
            {
                "event_id": str(row["event_id"]),
                "user_id": str(row["user_id"]),
                "session_id": str(row["session_id"]) if row["session_id"] else None,
                "date": str(row["date"]),
                "trigger_source": str(row["trigger_source"]),
                "severity": str(row["severity"]),
                "status": str(row["status"]),
                "item_id": str(row["item_id"]) if row["item_id"] else None,
                "reason_code": str(row["reason_code"]),
                "details": details,
                "created_at": str(row["created_at"]),
                "resolved_at": str(row["resolved_at"]) if row["resolved_at"] else None,
                "resolved_by": str(row["resolved_by"]) if row["resolved_by"] else None,
                "resolved_reason": str(row["resolved_reason"]) if row["resolved_reason"] else None,
            }
        )
    return out


def resolve_safety_event(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    event_id: str,
    resolved_by: str,
    resolved_reason: str,
) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT event_id, user_id, status FROM safety_events WHERE event_id=?;",
        (event_id,),
    ).fetchone()
    if not row:
        raise ValueError("Unknown safety event")
    if str(row["user_id"]) != str(user_id):
        raise ValueError("safety event does not belong to user")
    if str(row["status"]) != "open":
        return {"ok": True, "event_id": event_id, "status": str(row["status"])}
    conn.execute(
        """
        UPDATE safety_events
        SET status='resolved',
            resolved_at=?,
            resolved_by=?,
            resolved_reason=?
        WHERE event_id=?;
        """,
        (now_iso(), str(resolved_by), str(resolved_reason), event_id),
    )
    conn.commit()
    return {"ok": True, "event_id": event_id, "status": "resolved"}


def _mark_previous_day_incomplete_sessions_abandoned(conn: sqlite3.Connection, *, user_id: str, day: date) -> None:
    conn.execute(
        """
        UPDATE daily_sessions
        SET status='abandoned'
        WHERE user_id=?
          AND date<?
          AND status IN ('created','started');
        """,
        (user_id, day.isoformat()),
    )


def _sync_user_domain_profile(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    profile: Dict[str, Any],
    registry: Registry,
    cfg: QuestionsAgentConfig,
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
        max_active_new_domains=int(cfg.max_active_new_domains),
    )
    conn.execute(
        """
        UPDATE user_profiles
        SET active_domains_json=?,
            queued_domains_json=?,
            promoted_domains_json=?,
            updated_at=?
        WHERE user_id=?;
        """,
        (
            encode_json_str_list(active),
            encode_json_str_list(queued),
            encode_json_str_list(promoted),
            now_iso(),
            user_id,
        ),
    )
    conn.commit()
    out = dict(profile)
    out["active_domains"] = list(active)
    out["queued_domains"] = list(queued)
    out["promoted_domains"] = list(promoted)
    out["allowed_domains"] = sorted(set(allowed_domains))
    return out


def _create_safety_event(
    conn: sqlite3.Connection,
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
    conn.execute(
        """
        INSERT INTO safety_events(
          event_id, user_id, session_id, date,
          trigger_source, severity, status, item_id, reason_code,
          details_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?);
        """,
        (
            event_id,
            user_id,
            session_id,
            day.isoformat(),
            str(trigger_source or "runtime"),
            str(severity or "medium"),
            item_id,
            str(reason_code or "safety_trigger"),
            json.dumps(details or {}, ensure_ascii=False),
            now_iso(),
        ),
    )
    conn.commit()
    return event_id


def _list_open_safety_events(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
    lookback_days: int,
) -> List[Dict[str, Any]]:
    start = (day - timedelta(days=max(1, int(lookback_days)) - 1)).isoformat()
    rows = conn.execute(
        """
        SELECT *
        FROM safety_events
        WHERE user_id=?
          AND status='open'
          AND date>=?
          AND date<=?
        ORDER BY date DESC, created_at DESC;
        """,
        (user_id, start, day.isoformat()),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            details = json.loads(str(row["details_json"] or "{}"))
        except Exception:
            details = {}
        out.append(
            {
                "event_id": str(row["event_id"]),
                "user_id": str(row["user_id"]),
                "session_id": str(row["session_id"]) if row["session_id"] else None,
                "date": str(row["date"]),
                "trigger_source": str(row["trigger_source"]),
                "severity": str(row["severity"]),
                "status": str(row["status"]),
                "item_id": str(row["item_id"]) if row["item_id"] else None,
                "reason_code": str(row["reason_code"]),
                "details": details,
                "created_at": str(row["created_at"]),
            }
        )
    return out


def _apply_safety_mode_selection(
    *,
    registry: Registry,
    cfg: QuestionsAgentConfig,
    user_id: str,
    day: date,
    base_selection: SelectionPlan,
    answered_item_ids_by_scale: Dict[str, Set[str]],
    last_asked_date_by_item_id: Dict[str, date],
    last_score_date_by_scale_id: Dict[str, date],
    baselines_by_scale_id: Dict[str, BaselineState],
    rolling_normalized_by_scale_id: Dict[str, float],
    allowed_item_ids: Optional[Set[str]],
    declined_item_ids: Set[str],
    priority_item_ids: Optional[Set[str]],
    follow_up_item_ids: Set[str],
) -> SelectionPlan:
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
        # Last resort: keep the first deterministic item to avoid empty payloads.
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

    return SelectionPlan(
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


def get_or_create_daily_session(
    conn: sqlite3.Connection,
    *,
    cfg: QuestionsAgentConfig,
    registry_root: str,
    user_id: str,
    day: date,
    selection_mode: Optional[str] = None,
    policy_context: Optional[Dict[str, Any]] = None,
    identity_mask_id: Optional[str] = None,
) -> DailySession:
    logger.info("session.create user=%s day=%s", user_id, day.isoformat())
    ensure_user(conn, user_id)
    ensure_user_profile(conn, user_id)
    registry_version = ensure_registry_active(conn, registry_root)
    registry = load_registry(registry_root, registry_version)

    _mark_previous_day_incomplete_sessions_abandoned(conn, user_id=user_id, day=day)
    conn.commit()

    existing = conn.execute(
        "SELECT * FROM daily_sessions WHERE user_id=? AND date=?;",
        (user_id, day.isoformat()),
    ).fetchone()
    if existing:
        logger.debug("session.existing user=%s day=%s session_id=%s", user_id, day.isoformat(), existing["session_id"])
        return _row_to_session(existing)

    requested_selection_mode = normalize_selection_mode(
        selection_mode or cfg.policy_default_mode or "deterministic"
    )
    selection_mode_final = requested_selection_mode
    rollback_guard: Optional[Dict[str, Any]] = None
    if requested_selection_mode == "policy_live" and _boolish(cfg.policy_auto_rollback_enabled, default=True):
        rollback_guard = _evaluate_policy_live_rollback_guard(conn, cfg=cfg, day=day)
        if _boolish(rollback_guard.get("rollback", False), default=False):
            selection_mode_final = "deterministic"

    last_asked = _get_last_asked_dates(conn, user_id=user_id, day=day, lookback_days=max(30, cfg.item_repeat_cooldown_days + 2))
    last_scores = _get_last_score_dates(conn, user_id=user_id)
    baselines = _get_baselines(conn, user_id=user_id)
    answered_by_scale, rolling_by_scale = _compute_scale_windows(
        conn, registry=registry, user_id=user_id, day=day
    )

    ctx_obj = policy_context or {}
    if _boolish(ctx_obj.get("safety_trigger_active", False), default=False):
        _create_safety_event(
            conn,
            user_id=user_id,
            day=day,
            session_id=None,
            trigger_source="fusion_context",
            severity=str(ctx_obj.get("safety_severity") or "high"),
            reason_code=str(ctx_obj.get("safety_reason_code") or "safety_trigger_active"),
            details={"source": "context", "payload_keys": sorted(list(ctx_obj.keys()))},
        )

    profile = get_user_profile(conn, user_id)
    onboarding_complete = bool(profile.get("onboarding_complete", False))
    declined_item_ids = {str(i) for i in profile.get("permanently_declined_item_ids", [])}
    allowed_item_ids: Optional[Set[str]] = None
    priority_item_ids: Optional[Set[str]] = None
    follow_up_item_ids = _get_due_follow_up_item_ids(conn, user_id=user_id, day=day)

    # Load previous session engagement quality for selection feedback (Claim Family 5)
    _prev_engagement: Optional[str] = None
    prev_profile = _get_latest_session_uncertainty_profile(conn, user_id=user_id, before_date=day)
    if prev_profile is not None:
        _prev_engagement = prev_profile.engagement_quality

    answered_item_ids: Set[str] = set()
    if not onboarding_complete:
        answered_item_ids = _get_all_answered_item_ids(conn, user_id=user_id)
    onboarding_gate = evaluate_onboarding_cardiometabolic_gate(
        onboarding_complete=onboarding_complete,
        registry=registry,
        declined_item_ids=declined_item_ids,
        answered_item_ids=answered_item_ids,
    )
    allowed_item_ids = onboarding_gate.allowed_item_ids
    priority_item_ids = onboarding_gate.priority_item_ids
    if onboarding_gate.mark_onboarding_complete:
        conn.execute(
            "UPDATE user_profiles SET onboarding_complete=1, updated_at=? WHERE user_id=?;",
            (now_iso(), user_id),
        )
        conn.commit()

    # Refresh profile after onboarding transitions and apply domain governance.
    profile = _sync_user_domain_profile(
        conn,
        user_id=user_id,
        profile=get_user_profile(conn, user_id),
        registry=registry,
        cfg=cfg,
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

    open_safety_events = _list_open_safety_events(
        conn,
        user_id=user_id,
        day=day,
        lookback_days=int(cfg.safety_event_lookback_days),
    )
    safety_mode_active = bool(open_safety_events)
    if safety_mode_active:
        selection_mode_final = "deterministic"

    # --- Progressive measurement plan: resolve active package BEFORE selection ---
    # The active package constrains which items the selection engine
    # prioritises, so we compute it early and pass the package's scale
    # IDs into build_selection_plan().
    completed_pkg_ids = _get_completed_package_ids(conn, user_id=user_id)
    session_count = _count_completed_sessions(conn, user_id=user_id)
    latest_minifolds = _get_latest_minifold_snapshots(conn, user_id=user_id, day=day)
    drift_mode_str = dominant_decoherence_mode(latest_minifolds) if latest_minifolds else None
    drift_mode_enum = DecoherenceMode(drift_mode_str) if drift_mode_str else None
    progressive_plan = get_progressive_plan(
        user_id=user_id,
        completed_sessions=session_count,
        drift_mode=drift_mode_enum,
        completed_packages=list(completed_pkg_ids),
    )
    active_pkg_id = resolve_current_package(progressive_plan)
    active_pkg_scale_ids: Optional[Set[str]] = None
    active_pkg_scale_priorities: Optional[Dict[str, float]] = None
    if active_pkg_id is not None:
        active_pkg_scale_ids = set(get_package_scale_ids(active_pkg_id))
        active_pkg_scale_priorities = get_package_scale_priorities(active_pkg_id)

    # Shadow efficiency analysis — compute once per registry version, cached.
    # Provides soft-constraint scores that prefer efficient (non-redundant) items.
    _eff_scores = _get_efficiency_scores(registry, registry_version)

    # Deterministic baseline selection (always computed for auditability + fallbacks).
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
        active_package_scale_ids=active_pkg_scale_ids,
        active_package_scale_priorities=active_pkg_scale_priorities,
        efficiency_scores=_eff_scores,
    )
    if safety_mode_active:
        selection = _apply_safety_mode_selection(
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
            policy_root=cfg.policy_root,
            policy_default_version=cfg.policy_default_version,
            epsilon_explore=float(cfg.policy_epsilon_explore),
            include_context_snapshot=bool(cfg.policy_log_context_snapshot),
            include_candidate_set_snapshot=bool(cfg.policy_log_candidate_set_snapshot),
        )
        row = policy_exec.decision_row_payload
        conn.execute(
            """
            INSERT INTO policy_decisions(
              decision_id, user_id, date, identity_mask_id,
              selection_mode, mode, policy_version, feature_version,
              context_hash, candidate_set_hash,
              context_json, candidate_set_json,
              selected_item_ids_json, deterministic_baseline_selected_json,
              propensities_json, explanations_json, counterfactuals_json,
              created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                row["decision_id"],
                row["user_id"],
                row["date"],
                row["identity_mask_id"],
                row["selection_mode"],
                row["mode"],
                row["policy_version"],
                row["feature_version"],
                row["context_hash"],
                row["candidate_set_hash"],
                row["context_json"],
                row["candidate_set_json"],
                row["selected_item_ids_json"],
                row["deterministic_baseline_selected_json"],
                row["propensities_json"],
                row["explanations_json"],
                row["counterfactuals_json"],
                now_iso(),
            ),
        )
        policy_decision_id = policy_exec.decision.decision_id
        served_core_item_ids = policy_exec.served_core_item_ids
        policy_summary = policy_exec.summary

    session_id = str(uuid.uuid4())
    core_json = json.dumps(list(served_core_item_ids), ensure_ascii=False)
    extra_batches_filtered = filter_extra_batches(
        extra_batches=selection.extra_batches,
        core_item_ids=served_core_item_ids,
        declined_item_ids=declined_item_ids,
        allowed_item_ids=allowed_item_ids,
    )
    extra_json = json.dumps([list(batch) for batch in extra_batches_filtered], ensure_ascii=False)
    explain_obj = build_selection_explain_payload(
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
    # Concordance gating: inform clients if paired measurement mode is active
    session_site_config = _resolve_site_config(conn, user_id, default_config_id=cfg.site_config_id)
    if session_site_config.concordance_mode:
        explain_obj["concordance_mode"] = True
    if _prev_engagement:
        explain_obj["previous_engagement_quality"] = _prev_engagement

    # Progressive measurement plan: attach pre-computed plan to explain payload.
    # The plan was already computed before selection (above) so that the
    # active package's scale IDs could be fed into build_selection_plan().
    explain_obj["progressive_plan"] = {
        "recommended_sequence": list(progressive_plan.recommended_sequence),
        "current_package": active_pkg_id,
        "completed_packages": sorted(completed_pkg_ids),
        "session_count": session_count,
        "dominant_drift_mode": drift_mode_str,
        "active_package_scale_ids": sorted(active_pkg_scale_ids) if active_pkg_scale_ids else [],
    }
    # Track package lifecycle: mark active package as in_progress
    if active_pkg_id is not None:
        _upsert_package_status(conn, user_id=user_id, package_id=active_pkg_id, status="in_progress")

    explain_json = json.dumps(explain_obj, ensure_ascii=False)

    conn.execute(
        """
        INSERT INTO daily_sessions(
          session_id, user_id, date, registry_version,
          timeframe, status,
          selection_mode, policy_decision_id,
          core_questions_json, extra_batches_json, extra_batches_used,
          selection_explain_json, created_at
        ) VALUES (?, ?, ?, ?, ?, 'created', ?, ?, ?, ?, 0, ?, ?);
        """,
        (
            session_id,
            user_id,
            day.isoformat(),
            registry_version,
            str(selection.timeframe),
            selection_mode_final,
            policy_decision_id,
            core_json,
            extra_json,
            explain_json,
            now_iso(),
        ),
    )
    conn.commit()

    # --- Pool exhaustion detection ---
    pool_exhaustion: Optional[Dict[str, Any]] = None
    if len(served_core_item_ids) < cfg.core_questions_per_day:
        total_registry_items = len(registry.items)
        available_after_declines = total_registry_items - len(declined_item_ids)
        # Compute impossible scales (not enough remaining items to meet min_items_required)
        impossible_scales = []
        for scale in registry.scales.values():
            required_item_ids = {si.item_id for si in scale.items}
            remaining = required_item_ids - declined_item_ids
            if len(remaining) < scale.min_items_required:
                impossible_scales.append(scale.id)
        pool_exhaustion = {
            "items_served": len(served_core_item_ids),
            "items_requested": cfg.core_questions_per_day,
            "total_registry_items": total_registry_items,
            "declined_count": len(declined_item_ids),
            "available_after_declines": available_after_declines,
            "impossible_scales": impossible_scales,
            "warning": (
                f"Only {len(served_core_item_ids)}/{cfg.core_questions_per_day} items available. "
                f"{len(declined_item_ids)} items permanently declined, "
                f"{len(impossible_scales)} scales now impossible to complete."
            ),
        }
        logger.warning(
            "session.pool_exhaustion user=%s day=%s served=%d/%d declined=%d impossible_scales=%d",
            user_id, day.isoformat(), len(served_core_item_ids),
            cfg.core_questions_per_day, len(declined_item_ids), len(impossible_scales),
        )

    logger.info(
        "session.new user=%s day=%s session_id=%s items=%d mode=%s registry=%s",
        user_id, day.isoformat(), session_id, len(served_core_item_ids),
        selection_mode_final, registry_version,
    )

    return DailySession(
        session_id=session_id,
        user_id=user_id,
        date=day.isoformat(),
        registry_version=registry_version,
        timeframe=str(selection.timeframe),
        status="created",
        core_item_ids=tuple(served_core_item_ids),
        extra_batches=extra_batches_filtered,
        extra_batches_used=0,
        selection_explain=explain_obj,
        selection_mode=selection_mode_final,
        policy_decision_id=policy_decision_id,
        pool_exhaustion=pool_exhaustion,
    )


def get_daily_session(conn: sqlite3.Connection, user_id: str, day: date) -> Optional[DailySession]:
    row = conn.execute(
        "SELECT * FROM daily_sessions WHERE user_id=? AND date=?;",
        (user_id, day.isoformat()),
    ).fetchone()
    return _row_to_session(row) if row else None


def _evaluate_policy_live_rollback_guard(
    conn: sqlite3.Connection,
    *,
    cfg: QuestionsAgentConfig,
    day: date,
) -> Dict[str, Any]:
    window_days = int(max(1, cfg.policy_rollback_window_days))
    min_outcomes = int(max(1, cfg.policy_rollback_min_outcomes))
    max_safety = int(max(0, cfg.policy_rollback_max_safety_violations))
    completion_drop = float(max(0.0, cfg.policy_rollback_min_completion_drop))
    burden_ratio = float(max(1.0, cfg.policy_rollback_burden_increase_ratio))

    recent_start = day - timedelta(days=window_days - 1)
    recent_end = day
    prior_start = day - timedelta(days=(2 * window_days) - 1)
    prior_end = day - timedelta(days=window_days)

    recent = conn.execute(
        """
        SELECT
          COUNT(*) AS n,
          AVG(o.completion_rate) AS completion_rate,
          AVG(o.response_time_ms_median) AS burden_ms
        FROM policy_decisions d
        JOIN policy_outcomes o ON o.decision_id = d.decision_id
        WHERE d.selection_mode='policy_live' AND d.mode='live'
          AND d.date>=? AND d.date<=?;
        """,
        (recent_start.isoformat(), recent_end.isoformat()),
    ).fetchone()
    prior = conn.execute(
        """
        SELECT
          COUNT(*) AS n,
          AVG(o.completion_rate) AS completion_rate,
          AVG(o.response_time_ms_median) AS burden_ms
        FROM policy_decisions d
        JOIN policy_outcomes o ON o.decision_id = d.decision_id
        WHERE d.selection_mode='policy_live' AND d.mode='live'
          AND d.date>=? AND d.date<=?;
        """,
        (prior_start.isoformat(), prior_end.isoformat()),
    ).fetchone()

    recent_count = int(recent["n"] or 0) if recent else 0
    prior_count = int(prior["n"] or 0) if prior else 0
    recent_completion = float(recent["completion_rate"]) if recent and recent["completion_rate"] is not None else None
    prior_completion = float(prior["completion_rate"]) if prior and prior["completion_rate"] is not None else None
    recent_burden = float(recent["burden_ms"]) if recent and recent["burden_ms"] is not None else None
    prior_burden = float(prior["burden_ms"]) if prior and prior["burden_ms"] is not None else None

    safety_violations = _count_policy_live_safety_violations(
        conn,
        start_date=recent_start.isoformat(),
        end_date=recent_end.isoformat(),
    )

    degradation = False
    if (
        recent_count >= min_outcomes
        and prior_count >= min_outcomes
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
        "window_days": window_days,
        "min_outcomes": min_outcomes,
        "max_safety_violations": max_safety,
        "completion_drop_threshold": completion_drop,
        "burden_increase_ratio_threshold": burden_ratio,
        "recent": {
            "start_date": recent_start.isoformat(),
            "end_date": recent_end.isoformat(),
            "outcomes": recent_count,
            "completion_rate": recent_completion,
            "burden_ms_median": recent_burden,
            "safety_violations": int(safety_violations),
        },
        "prior": {
            "start_date": prior_start.isoformat(),
            "end_date": prior_end.isoformat(),
            "outcomes": prior_count,
            "completion_rate": prior_completion,
            "burden_ms_median": prior_burden,
        },
    }


def _count_policy_live_safety_violations(conn: sqlite3.Connection, *, start_date: str, end_date: str) -> int:
    rows = conn.execute(
        """
        SELECT selected_item_ids_json, candidate_set_json
        FROM policy_decisions
        WHERE selection_mode='policy_live' AND mode='live'
          AND date>=? AND date<=?;
        """,
        (start_date, end_date),
    ).fetchall()

    violations = 0
    for row in rows:
        try:
            selected = json.loads(str(row["selected_item_ids_json"] or "[]"))
            cand_obj = json.loads(str(row["candidate_set_json"] or "{}"))
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


def take_next_extra_batch(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    user_id: Optional[str] = None,
    extra_batches_max_per_day: int = 3,
) -> Optional[Tuple[str, ...]]:
    row = conn.execute("SELECT * FROM daily_sessions WHERE session_id=?;", (session_id,)).fetchone()
    if not row:
        return None
    if user_id is not None and str(row["user_id"]) != str(user_id):
        raise PermissionError("session_id does not belong to user_id")
    extra_batches = json.loads(row["extra_batches_json"])
    pool: List[str] = []
    for batch in extra_batches:
        if isinstance(batch, list):
            pool.extend(str(item_id) for item_id in batch)
    max_extras = min(max(0, int(extra_batches_max_per_day)), len(pool))
    used = int(row["extra_batches_used"] or 0)
    if used >= max_extras:
        return None
    batch = [pool[used]]
    conn.execute(
        "UPDATE daily_sessions SET extra_batches_used=? WHERE session_id=?;",
        (used + 1, session_id),
    )
    conn.commit()
    return tuple(batch)


def _enqueue_follow_up_items_for_scale(
    conn: sqlite3.Connection,
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
    profile = get_user_profile(conn, user_id)
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
        _enqueue_follow_up_queue_item(
            conn,
            user_id=user_id,
            item_id=item_id,
            scale_id=scale_id,
            reason_code=reason_code,
            reason_detail=reason_detail,
            priority=float(priority),
            earliest_date=day,
            timeframe=timeframe,
        )


def _resolve_follow_up_queue_for_item(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    item_id: str,
    resolution_day: date,
    resolved_reason: str,
    status: str = "resolved",
) -> None:
    conn.execute(
        """
        UPDATE follow_up_queue
        SET status=?,
            resolved_at=?,
            resolved_reason=?
        WHERE user_id=?
          AND item_id=?
          AND status='pending'
          AND earliest_date<=?;
        """,
        (str(status), date_to_start_iso(resolution_day), str(resolved_reason), user_id, item_id, resolution_day.isoformat()),
    )


def _resolve_follow_up_queue_for_scale(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    scale_id: str,
    resolution_day: date,
    resolved_reason: str,
) -> None:
    conn.execute(
        """
        UPDATE follow_up_queue
        SET status='resolved',
            resolved_at=?,
            resolved_reason=?
        WHERE user_id=?
          AND scale_id=?
          AND status='pending'
          AND earliest_date<=?;
        """,
        (date_to_start_iso(resolution_day), str(resolved_reason), user_id, scale_id, resolution_day.isoformat()),
    )


def _mark_item_permanently_declined(conn: sqlite3.Connection, *, user_id: str, item_id: str) -> None:
    ensure_user_profile(conn, user_id)
    row = conn.execute(
        "SELECT permanently_declined_items_json FROM user_profiles WHERE user_id=?;",
        (user_id,),
    ).fetchone()
    current: List[str] = []
    if row and row["permanently_declined_items_json"]:
        try:
            parsed = json.loads(str(row["permanently_declined_items_json"]))
            if isinstance(parsed, list):
                current = [str(x) for x in parsed]
        except Exception:
            current = []
    if item_id in current:
        return
    current.append(str(item_id))
    conn.execute(
        """
        UPDATE user_profiles
        SET permanently_declined_items_json=?, updated_at=?
        WHERE user_id=?;
        """,
        (json.dumps(sorted(set(current)), ensure_ascii=False), now_iso(), user_id),
    )
    _resolve_follow_up_queue_for_item(
        conn,
        user_id=user_id,
        item_id=item_id,
        resolution_day=date.today(),
        resolved_reason="permanently_declined",
        status="cancelled",
    )
    conn.commit()


def _insert_scale_evidence_rows(
    conn: sqlite3.Connection,
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
            conn.execute(
                """
                INSERT INTO scale_evidence(
                  evidence_id, user_id, answer_event_id, session_id, item_id, scale_id,
                  weight, contribution_value, timeframe, window_date, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(uuid.uuid4()),
                    user_id,
                    answer_event_id,
                    session_id,
                    item_id,
                    scale.id,
                    float(scale_item.weight),
                    float(contribution),
                    str(timeframe),
                    window_date.isoformat(),
                    now_iso(),
                ),
            )


def submit_answers(
    conn: sqlite3.Connection,
    *,
    registry_root: str,
    user_id: str,
    session_id: str,
    answers: List[Dict[str, Any]],
    drift_routing_table_path: Optional[str] = None,
    site_config: Optional[SiteConfig] = None,
) -> Dict[str, Any]:
    """
    Store answer events (idempotent), capture behavioural metadata,
    compute newly completed scale scores with uncertainty adjustment,
    and run anamnesis checks.
    """
    logger.info("answers.submit user=%s session=%s count=%d", user_id, session_id, len(answers))
    ensure_user(conn, user_id)
    session_row = conn.execute("SELECT * FROM daily_sessions WHERE session_id=?;", (session_id,)).fetchone()
    if not session_row:
        raise ValueError("Unknown session_id")
    if str(session_row["user_id"]) != user_id:
        raise ValueError("session_id does not belong to user_id")
    session_date = parse_date(str(session_row["date"]))

    registry_version = str(session_row["registry_version"])
    registry = load_registry(registry_root, registry_version)
    session_timeframe = (
        str(session_row["timeframe"])
        if "timeframe" in session_row.keys() and session_row["timeframe"]
        else "last_7_days"
    )

    # Resolve site config for feature gating
    resolved_config = _resolve_site_config(
        conn, user_id, site_config=site_config, default_config_id="consumer"
    )
    metadata_enabled = bool(resolved_config.behavioural_metadata_enabled)

    # Accumulators for response metadata (Claim Family 5)
    session_modifiers: List[UncertaintyModifier] = []
    session_metadata_list: List[ResponseMetadata] = []

    inserted = 0
    for a in answers:
        item_id = str(a.get("item_id") or "")
        if item_id not in registry.items:
            raise ValueError(f"Unknown item_id: {item_id}")
        raw_obj = a.get("raw")
        if isinstance(raw_obj, dict) and _boolish(raw_obj.get("permanently_declined"), default=False):
            _mark_item_permanently_declined(conn, user_id=user_id, item_id=item_id)
            continue
        if isinstance(raw_obj, dict) and _boolish(raw_obj.get("safety_trigger"), default=False):
            _create_safety_event(
                conn,
                user_id=user_id,
                day=session_date,
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

        try:
            event_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO answer_events(
                  event_id, client_event_id, user_id, session_id, item_id,
                  answered_at, value, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    event_id,
                    client_event_id,
                    user_id,
                    session_id,
                    item_id,
                    answered_at,
                    value,
                    json.dumps(a.get("raw"), ensure_ascii=False) if a.get("raw") is not None else None,
                ),
            )
            _insert_scale_evidence_rows(
                conn,
                registry=registry,
                user_id=user_id,
                session_id=session_id,
                answer_event_id=event_id,
                item_id=item_id,
                answer_value=float(value),
                timeframe=session_timeframe,
                window_date=session_date,
            )
            _resolve_follow_up_queue_for_item(
                conn,
                user_id=user_id,
                item_id=item_id,
                resolution_day=session_date,
                resolved_reason="answered",
            )

            # --- Behavioural metadata capture (Claim Family 5) ---
            meta_raw = a.get("metadata") if isinstance(a.get("metadata"), dict) else None
            if metadata_enabled and meta_raw is not None:
                resp_meta = ResponseMetadata(
                    item_id=item_id,
                    response_latency_ms=float(meta_raw["response_latency_ms"]) if meta_raw.get("response_latency_ms") is not None else None,
                    edit_count=int(meta_raw.get("edit_count", 0)),
                    was_skipped=_boolish(meta_raw.get("was_skipped", False), default=False),
                    was_declined=_boolish(meta_raw.get("was_declined", False), default=False),
                    channel=str(meta_raw.get("channel", "tap")),
                    voice_hesitation_ms=float(meta_raw["voice_hesitation_ms"]) if meta_raw.get("voice_hesitation_ms") is not None else None,
                    time_of_day_hour=int(meta_raw["time_of_day_hour"]) if meta_raw.get("time_of_day_hour") is not None else None,
                )
                item_spec = registry.items.get(item_id)
                resp_type_str = item_spec.response_type if item_spec else "likert_0_4"
                is_sensitive = bool(getattr(item_spec, "is_sensitive", False)) if item_spec else False
                modifier = compute_uncertainty_modifier(
                    resp_meta,
                    response_type=resp_type_str,
                    is_sensitive=is_sensitive,
                )
                _insert_response_metadata(
                    conn,
                    user_id=user_id,
                    session_id=session_id,
                    answer_event_id=event_id,
                    metadata=resp_meta,
                    modifier=modifier,
                )
                session_modifiers.append(modifier)
                session_metadata_list.append(resp_meta)

            inserted += 1
        except sqlite3.IntegrityError:
            # idempotent duplicate
            continue

    # --- Session uncertainty profile ---
    if metadata_enabled and session_modifiers:
        session_profile = compute_session_uncertainty_profile(
            session_modifiers,
            session_metadata_list,
            session_date=session_date.isoformat(),
            user_id=user_id,
        )
        _upsert_session_uncertainty_profile(
            conn,
            session_id=session_id,
            profile=session_profile,
        )

    conn.commit()

    core_item_ids = tuple(json.loads(session_row["core_questions_json"]))
    answered_in_session_rows = conn.execute(
        "SELECT DISTINCT item_id FROM answer_events WHERE session_id=?;",
        (session_id,),
    ).fetchall()
    answered_in_session = {str(r["item_id"]) for r in answered_in_session_rows}
    core_answered_count = sum(1 for item_id in core_item_ids if item_id in answered_in_session)
    new_status = "created"
    if answered_in_session:
        new_status = "started"
    if core_item_ids and core_answered_count >= len(core_item_ids):
        new_status = "completed"
    conn.execute(
        "UPDATE daily_sessions SET status=? WHERE session_id=?;",
        (new_status, session_id),
    )
    conn.commit()

    new_scores = _compute_and_store_scores(
        conn, registry=registry, user_id=user_id, day=session_date,
        session_id=session_id, site_config=resolved_config,
    )
    profile = get_user_profile(conn, user_id)
    _upsert_daily_state_and_circle_snapshots(
        conn,
        registry=registry,
        user_id=user_id,
        day=session_date,
        state_schema_version=str(profile.get("state_schema_version") or STATE_SCHEMA_VERSION_DEFAULT),
        drift_routing_table_path=drift_routing_table_path,
        site_config=resolved_config,
    )
    progress = compute_user_scale_progress(conn, registry=registry, user_id=user_id, day=session_date)
    decision_id = str(session_row["policy_decision_id"]) if ("policy_decision_id" in session_row.keys() and session_row["policy_decision_id"]) else None
    if decision_id:
        _upsert_policy_outcome_from_answers(
            conn,
            user_id=user_id,
            day=session_date,
            session_id=session_id,
            decision_id=decision_id,
            core_item_ids=tuple(json.loads(session_row["core_questions_json"])),
            unlock_count=len(new_scores or []),
        )
    # --- Package lifecycle: evaluate completion after scoring ---
    # After new scale scores are computed, check if any in-progress
    # measurement packages are now complete (all their scales scored).
    newly_completed_pkg = _evaluate_and_advance_packages(conn, user_id=user_id)

    logger.info(
        "answers.complete user=%s session=%s inserted=%d unlocks=%d pkg_completed=%s",
        user_id, session_id, inserted, len(new_scores or []),
        newly_completed_pkg or "none",
    )
    result: Dict[str, Any] = {
        "inserted_answer_events": inserted,
        "new_scale_scores": new_scores,
        "scale_progress": progress,
    }
    if newly_completed_pkg:
        result["completed_package"] = newly_completed_pkg
    return result


def _insert_response_metadata(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    session_id: str,
    answer_event_id: str,
    metadata: ResponseMetadata,
    modifier: UncertaintyModifier,
) -> None:
    """Insert a response_metadata row for a single answer event."""
    metadata_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT OR IGNORE INTO response_metadata(
          metadata_id, user_id, session_id, answer_event_id, item_id,
          response_latency_ms, edit_count, was_skipped, was_declined,
          channel, voice_hesitation_ms, time_of_day_hour,
          uncertainty_multiplier, confidence_label,
          contributing_factors_json, raw_components_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            metadata_id,
            user_id,
            session_id,
            answer_event_id,
            metadata.item_id,
            metadata.response_latency_ms,
            metadata.edit_count,
            1 if metadata.was_skipped else 0,
            1 if metadata.was_declined else 0,
            metadata.channel,
            metadata.voice_hesitation_ms,
            metadata.time_of_day_hour,
            float(modifier.multiplier),
            modifier.confidence_label,
            json.dumps(list(modifier.contributing_factors), ensure_ascii=False),
            json.dumps(modifier.raw_components, ensure_ascii=False),
            now_iso(),
        ),
    )


def _upsert_session_uncertainty_profile(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    profile: SessionUncertaintyProfile,
) -> None:
    """Insert or replace the session-level uncertainty profile."""
    profile_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT OR REPLACE INTO session_uncertainty_profiles(
          profile_id, user_id, session_id, session_date,
          session_multiplier, engagement_quality,
          median_latency_ms, total_edits, skip_count, decline_count,
          item_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            profile_id,
            profile.user_id,
            session_id,
            profile.session_date,
            float(profile.session_multiplier),
            profile.engagement_quality,
            float(profile.median_latency_ms),
            profile.total_edits,
            profile.skip_count,
            profile.decline_count,
            len(profile.item_modifiers),
            now_iso(),
        ),
    )


def _get_latest_session_uncertainty_profile(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    before_date: date,
) -> Optional[SessionUncertaintyProfile]:
    """Fetch the most recent session uncertainty profile before a given date."""
    row = conn.execute(
        """
        SELECT *
        FROM session_uncertainty_profiles
        WHERE user_id=? AND session_date<?
        ORDER BY session_date DESC, created_at DESC
        LIMIT 1;
        """,
        (user_id, before_date.isoformat()),
    ).fetchone()
    if not row:
        return None
    return SessionUncertaintyProfile(
        session_date=str(row["session_date"]),
        user_id=str(row["user_id"]),
        item_modifiers=(),  # not reconstructing per-item for session lookup
        session_multiplier=float(row["session_multiplier"]),
        engagement_quality=str(row["engagement_quality"]),
        median_latency_ms=float(row["median_latency_ms"]),
        total_edits=int(row["total_edits"]),
        skip_count=int(row["skip_count"]),
        decline_count=int(row["decline_count"]),
    )


def _get_completed_package_ids(
    conn: sqlite3.Connection,
    *,
    user_id: str,
) -> Set[str]:
    """Return set of measurement package IDs already completed by this user."""
    rows = conn.execute(
        "SELECT measurement_package_id FROM user_measurement_packages WHERE user_id=? AND status='completed';",
        (user_id,),
    ).fetchall()
    return {str(r["measurement_package_id"]) for r in rows}


def _count_completed_sessions(
    conn: sqlite3.Connection,
    *,
    user_id: str,
) -> int:
    """Count total completed sessions for progressive plan computation."""
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM daily_sessions WHERE user_id=? AND status='completed';",
        (user_id,),
    ).fetchone()
    return int(row["cnt"]) if row else 0


def _upsert_package_status(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    package_id: str,
    status: str,
) -> None:
    """Create or update a user_measurement_packages row.

    Lifecycle:  pending → in_progress → completed.
    Never downgrade status (completed stays completed).
    """
    existing = conn.execute(
        "SELECT status FROM user_measurement_packages WHERE user_id=? AND measurement_package_id=?;",
        (user_id, package_id),
    ).fetchone()

    if existing is not None:
        current = str(existing["status"])
        # Never downgrade: completed > in_progress > pending
        _rank = {"pending": 0, "in_progress": 1, "completed": 2}
        if _rank.get(status, 0) <= _rank.get(current, 0):
            return
        updates = ["status=?"]
        params: list = [status]
        if status == "completed":
            updates.append("completed_at=?")
            params.append(now_iso())
        params.extend([user_id, package_id])
        conn.execute(
            f"UPDATE user_measurement_packages SET {', '.join(updates)} WHERE user_id=? AND measurement_package_id=?;",
            tuple(params),
        )
    else:
        conn.execute(
            """
            INSERT INTO user_measurement_packages(
              package_id, user_id, measurement_package_id, status,
              session_count, started_at, created_at
            ) VALUES (?, ?, ?, ?, 0, ?, ?);
            """,
            (
                str(uuid.uuid4()),
                user_id,
                package_id,
                status,
                now_iso() if status == "in_progress" else None,
                now_iso(),
            ),
        )
    conn.commit()


def _get_scored_scale_ids(
    conn: sqlite3.Connection,
    *,
    user_id: str,
) -> frozenset:
    """Return the set of scale IDs that have at least one score for this user."""
    rows = conn.execute(
        "SELECT DISTINCT scale_id FROM scale_scores WHERE user_id=?;",
        (user_id,),
    ).fetchall()
    return frozenset(str(r["scale_id"]) for r in rows)


def _evaluate_and_advance_packages(
    conn: sqlite3.Connection,
    *,
    user_id: str,
) -> Optional[str]:
    """Check if any in-progress packages are now complete.

    Evaluates all in-progress packages against the user's scored
    scales.  If a package is complete, marks it as such.

    Returns the first newly-completed package ID, or None.
    """
    scored = _get_scored_scale_ids(conn, user_id=user_id)
    rows = conn.execute(
        "SELECT measurement_package_id FROM user_measurement_packages WHERE user_id=? AND status='in_progress';",
        (user_id,),
    ).fetchall()

    newly_completed: Optional[str] = None
    for row in rows:
        pkg_id = str(row["measurement_package_id"])
        if evaluate_package_completion(pkg_id, scored):
            _upsert_package_status(conn, user_id=user_id, package_id=pkg_id, status="completed")
            if newly_completed is None:
                newly_completed = pkg_id
            logger.info("package.completed user=%s package=%s", user_id, pkg_id)

    return newly_completed


def _load_item_modifiers_for_scale(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    scale_item_ids: Sequence[str],
) -> List[UncertaintyModifier]:
    """Load uncertainty modifiers for items belonging to a specific scale from this session."""
    if not scale_item_ids:
        return []
    placeholders = ",".join("?" for _ in scale_item_ids)
    rows = conn.execute(
        f"""
        SELECT item_id, uncertainty_multiplier, confidence_label,
               contributing_factors_json, raw_components_json
        FROM response_metadata
        WHERE session_id=? AND item_id IN ({placeholders});
        """,
        (session_id, *scale_item_ids),
    ).fetchall()
    modifiers: List[UncertaintyModifier] = []
    for r in rows:
        try:
            factors = tuple(json.loads(str(r["contributing_factors_json"] or "[]")))
        except Exception:
            factors = ()
        try:
            components = json.loads(str(r["raw_components_json"] or "{}"))
        except Exception:
            components = {}
        modifiers.append(UncertaintyModifier(
            item_id=str(r["item_id"]),
            multiplier=float(r["uncertainty_multiplier"]),
            confidence_label=str(r["confidence_label"]),
            contributing_factors=factors,
            raw_components=components,
        ))
    return modifiers


def submit_observations(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    observations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Store non-question signals (wearables, voice, imaging, omics) as observation events.

    v1 stores these events for future selection logic; it does not yet alter scoring.
    """
    ensure_user(conn, user_id)
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
        state_row = conn.execute(
            """
            SELECT x_hat_json, x_uncertainty_json
            FROM state_snapshots
            WHERE user_id=? AND date<=?
            ORDER BY date DESC, created_at DESC
            LIMIT 1;
            """,
            (user_id, observed_day.isoformat()),
        ).fetchone()
        x_hat: Dict[str, float] = {}
        x_uncertainty: Dict[str, float] = {}
        if state_row:
            try:
                x_hat_raw = json.loads(str(state_row["x_hat_json"]))
                if isinstance(x_hat_raw, dict):
                    x_hat = {str(k): float(v) for k, v in x_hat_raw.items()}
            except Exception:
                x_hat = {}
            try:
                unc_raw = json.loads(str(state_row["x_uncertainty_json"]))
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

        conn.execute(
            """
            INSERT INTO observation_events(
              event_id, user_id, observed_at, type, features_json, confidence, provenance_json, coupling_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                event_id,
                user_id,
                observed_at,
                obs_type,
                json.dumps(features, ensure_ascii=False),
                confidence,
                json.dumps(provenance, ensure_ascii=False),
                json.dumps(coupling, ensure_ascii=False),
            ),
        )
        coupling_outputs.append({"event_id": event_id, **coupling})
        inserted += 1

    conn.commit()
    return {"inserted_observation_events": inserted, "coupling_outputs": coupling_outputs}


def _upsert_policy_outcome_from_answers(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
    session_id: str,
    decision_id: str,
    core_item_ids: Tuple[str, ...],
    unlock_count: int,
) -> None:
    rows = conn.execute(
        "SELECT item_id, answered_at FROM answer_events WHERE session_id=? ORDER BY answered_at ASC;",
        (session_id,),
    ).fetchall()
    answered_ids = {str(r["item_id"]) for r in rows}
    completed_core = sum(1 for i in core_item_ids if str(i) in answered_ids)
    completion_rate = float(completed_core) / max(1, len(core_item_ids))

    times: List[datetime] = []
    for r in rows:
        ts = _parse_ts(str(r["answered_at"]))
        if ts is not None:
            times.append(ts)

    diffs_ms: List[float] = []
    for i in range(1, len(times)):
        dt = (times[i] - times[i - 1]).total_seconds() * 1000.0
        if dt >= 0:
            diffs_ms.append(float(dt))
    response_time_ms_median = median(diffs_ms) if diffs_ms else None

    from questions_agent_platform.policy.rewards import compute_reward

    existing = conn.execute(
        "SELECT uncertainty_before_mean, uncertainty_after_mean FROM policy_outcomes WHERE decision_id=?;",
        (decision_id,),
    ).fetchone()
    ub = float(existing["uncertainty_before_mean"]) if existing and existing["uncertainty_before_mean"] is not None else None
    ua = float(existing["uncertainty_after_mean"]) if existing and existing["uncertainty_after_mean"] is not None else None

    reward = compute_reward(
        completion_rate=completion_rate,
        response_time_ms_median=response_time_ms_median,
        unlock_count=int(unlock_count),
        uncertainty_before_mean=ub,
        uncertainty_after_mean=ua,
    )

    conn.execute(
        "INSERT OR IGNORE INTO policy_outcomes(decision_id, user_id, date, updated_at) VALUES (?, ?, ?, ?);",
        (decision_id, user_id, day.isoformat(), now_iso()),
    )
    conn.execute(
        """
        UPDATE policy_outcomes
        SET completion_rate=?,
            completed_core_count=?,
            response_time_ms_median=?,
            reward_components_json=?,
            total_reward=?,
            updated_at=?
        WHERE decision_id=?;
        """,
        (
            float(completion_rate),
            int(completed_core),
            float(response_time_ms_median) if response_time_ms_median is not None else None,
            json.dumps(
                {
                    "components": reward.components,
                    "weights": reward.weights,
                    "inputs": {
                        "unlock_count": int(unlock_count),
                        "response_time_ms_median": float(response_time_ms_median) if response_time_ms_median is not None else None,
                    },
                },
                ensure_ascii=False,
            ),
            float(reward.total),
            now_iso(),
            decision_id,
        ),
    )
    conn.commit()


def _parse_ts(ts: str) -> Optional[datetime]:
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
    conn: sqlite3.Connection,
    *,
    cfg: QuestionsAgentConfig,
    user_id: str,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Reference-engine outcome update: attach Anifold-provided Z/uncertainty snapshots to a decision_id.

    This enables delayed reward components (e.g., uncertainty reduction) and offline evaluation.
    """
    ensure_user(conn, user_id)
    decision_id = str(body.get("decision_id") or "")
    if not decision_id:
        raise ValueError("Missing decision_id")

    dec = conn.execute(
        "SELECT decision_id, user_id, date FROM policy_decisions WHERE decision_id=?;",
        (decision_id,),
    ).fetchone()
    if not dec:
        raise ValueError("Unknown decision_id")
    if str(dec["user_id"]) != str(user_id):
        raise ValueError("decision_id does not belong to user_id")

    z_before = body.get("z_before")
    z_after = body.get("z_after")
    u_before = body.get("uncertainty_before_diag")
    u_after = body.get("uncertainty_after_diag")

    from questions_agent_platform.policy.hashing import stable_hash_floats
    from questions_agent_platform.policy.rewards import compute_reward

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

    q = int(max(0, cfg.policy_quantize_decimals))

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

    conn.execute(
        "INSERT OR IGNORE INTO policy_outcomes(decision_id, user_id, date, updated_at) VALUES (?, ?, ?, ?);",
        (decision_id, user_id, str(dec["date"]), now_iso()),
    )

    # Pull existing adherence/burden reward inputs if already computed.
    existing = conn.execute(
        "SELECT completion_rate, response_time_ms_median, reward_components_json FROM policy_outcomes WHERE decision_id=?;",
        (decision_id,),
    ).fetchone()
    completion_rate = float(existing["completion_rate"]) if existing and existing["completion_rate"] is not None else 0.0
    response_time_ms_median = (
        float(existing["response_time_ms_median"]) if existing and existing["response_time_ms_median"] is not None else None
    )
    unlock_count = 0
    if existing and existing["reward_components_json"]:
        try:
            payload = json.loads(str(existing["reward_components_json"]))
            if isinstance(payload, dict):
                inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
                unlock_count = int(inputs.get("unlock_count") or 0)
        except Exception:
            unlock_count = 0

    reward_overrides = body.get("reward_overrides") if isinstance(body.get("reward_overrides"), dict) else None
    rr = compute_reward(
        completion_rate=completion_rate,
        response_time_ms_median=response_time_ms_median,
        unlock_count=int(unlock_count),
        uncertainty_before_mean=mean(u_before),
        uncertainty_after_mean=mean(u_after),
        overrides=reward_overrides,
    )

    z_before_hash = stable_hash_floats(z_before, quantize=3) if isinstance(z_before, list) else None
    z_after_hash = stable_hash_floats(z_after, quantize=3) if isinstance(z_after, list) else None
    z_before_norm = norm(z_before)
    z_after_norm = norm(z_after)

    z_before_json = None
    z_after_json = None
    if bool(cfg.policy_store_z_snapshot):
        zb = qvec(z_before)
        za = qvec(z_after)
        z_before_json = json.dumps(zb, ensure_ascii=False) if zb is not None else None
        z_after_json = json.dumps(za, ensure_ascii=False) if za is not None else None

    z_delta_json = None
    z_delta_norm = None
    if bool(cfg.policy_store_z_delta) and isinstance(z_before, list) and isinstance(z_after, list) and len(z_before) == len(z_after):
        delta = [float(a) - float(b) for a, b in zip(z_after, z_before)]
        z_delta_norm = norm(delta)
        z_delta_json = json.dumps([round(float(x), q) for x in delta], ensure_ascii=False)

    conn.execute(
        """
        UPDATE policy_outcomes
        SET z_before_hash=?,
            z_after_hash=?,
            z_before_json=?,
            z_after_json=?,
            z_delta_json=?,
            z_before_norm=?,
            z_after_norm=?,
            z_delta_norm=?,
            uncertainty_before_mean=?,
            uncertainty_after_mean=?,
            uncertainty_before_max=?,
            uncertainty_after_max=?,
            reward_components_json=?,
            total_reward=?,
            updated_at=?
        WHERE decision_id=?;
        """,
        (
            z_before_hash,
            z_after_hash,
            z_before_json,
            z_after_json,
            z_delta_json,
            float(z_before_norm) if z_before_norm is not None else None,
            float(z_after_norm) if z_after_norm is not None else None,
            float(z_delta_norm) if z_delta_norm is not None else None,
            mean(u_before),
            mean(u_after),
            vmax(u_before),
            vmax(u_after),
            json.dumps(
                {
                    "components": rr.components,
                    "weights": rr.weights,
                    "inputs": {"unlock_count": int(unlock_count), "response_time_ms_median": response_time_ms_median},
                    "overrides": reward_overrides,
                },
                ensure_ascii=False,
            ),
            float(rr.total),
            now_iso(),
            decision_id,
        ),
    )
    conn.commit()
    return {"ok": True, "decision_id": decision_id}


def get_state_snapshots(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
    window_days: int = 30,
) -> List[Dict[str, Any]]:
    start = (day - timedelta(days=max(1, int(window_days)) - 1)).isoformat()
    rows = conn.execute(
        """
        SELECT *
        FROM state_snapshots
        WHERE user_id=?
          AND date>=?
        ORDER BY date ASC, created_at ASC;
        """,
        (user_id, start),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "snapshot_id": str(r["snapshot_id"]),
                "user_id": str(r["user_id"]),
                "date": str(r["date"]),
                "timestamp": str(r["timestamp"]),
                "state_schema_version": str(r["state_schema_version"]),
                "model_version": str(r["model_version"]),
                "x_hat": json.loads(str(r["x_hat_json"])),
                "x_uncertainty": json.loads(str(r["x_uncertainty_json"])),
                "coverage": json.loads(str(r["coverage_json"])),
                "source": str(r["source"]),
            }
        )
    return out


def get_circle_snapshots(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
    window_days: int = 30,
) -> List[Dict[str, Any]]:
    start = (day - timedelta(days=max(1, int(window_days)) - 1)).isoformat()
    rows = conn.execute(
        """
        SELECT *
        FROM circle_snapshots
        WHERE user_id=?
          AND date>=?
          AND projection_version=?
        ORDER BY date ASC, created_at ASC;
        """,
        (user_id, start, CIRCLE_PROJECTION_VERSION),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            """
            SELECT *
            FROM circle_snapshots
            WHERE user_id=?
              AND date>=?
            ORDER BY date ASC, created_at ASC;
            """,
            (user_id, start),
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        uncertainty = json.loads(str(r["uncertainty_json"]))
        score = (
            float(r["coherence_score"])
            if "coherence_score" in r.keys() and r["coherence_score"] is not None
            else coherence_score_from_radius(float(r["r"]))
        )
        tier = (
            json.loads(str(r["coherence_tier_json"]))
            if "coherence_tier_json" in r.keys() and r["coherence_tier_json"]
            else coherence_tier_for_score(score)
        )
        if not isinstance(tier, dict):
            tier = coherence_tier_for_score(score)
        out.append(
            {
                "snapshot_id": str(r["snapshot_id"]),
                "user_id": str(r["user_id"]),
                "date": str(r["date"]),
                "timestamp": str(r["timestamp"]),
                "state_snapshot_id": str(r["state_snapshot_id"]) if r["state_snapshot_id"] else None,
                "projection_version": str(r["projection_version"]),
                "anchor_version": str(r["anchor_version"]),
                "z": json.loads(str(r["z_json"])),
                "z_star": json.loads(str(r["z_star_json"])),
                "r": float(r["r"]),
                "theta": float(r["theta"]),
                "velocity": float(r["velocity"]),
                "acceleration": float(r["acceleration"]),
                "coherence": {
                    "score": float(score),
                    "tier": tier,
                },
                "theta_defined": bool(uncertainty.get("theta_defined", True)),
                "semantic_axis_scores": uncertainty.get("semantic_axis_scores", {}),
                "semantic_concentration": uncertainty.get("semantic_concentration"),
                "projection_kind": uncertainty.get("projection_kind"),
                "uncertainty": uncertainty,
                "source": str(r["source"]),
            }
        )
    return out


def compute_user_fold(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
    window_days: int = 30,
) -> Dict[str, Any]:
    """Compute the QuestionsFold for a user on a given day.

    This is the public service-layer entry point for the ``/v1/fold``
    endpoint. It loads all required data from the database and
    delegates to ``compute_questions_fold()``.

    Returns the Identity Mask fragment as a JSON-serialisable dict.
    """
    from questions_agent_platform.pipeline.questions_fold import (
        compute_questions_fold,
    )

    # 1. Load the latest state snapshot (x_hat, x_uncertainty, coverage)
    state_snapshots = get_state_snapshots(conn, user_id=user_id, day=day, window_days=1)
    if not state_snapshots:
        # No state data yet — return empty fold with population priors
        x_hat = {d: 0.5 for d in STATE_DIMENSIONS}
        x_uncertainty = {d: 1.0 for d in STATE_DIMENSIONS}
        coverage = None
    else:
        latest_state = state_snapshots[-1]
        x_hat = latest_state["x_hat"]
        x_uncertainty = latest_state["x_uncertainty"]
        coverage = latest_state.get("coverage")

    # 2. Load previous fold (circle + minifold data from yesterday)
    circle_snaps = get_circle_snapshots(
        conn, user_id=user_id, day=day, window_days=max(2, window_days),
    )
    previous_fold: Optional[Dict[str, Any]] = None
    if len(circle_snaps) >= 2:
        prev = circle_snaps[-2]  # second-to-last is previous day
        previous_fold = {
            "circle": {
                "z": prev["z"],
                "z_star": prev["z_star"],
                "velocity": float(prev.get("velocity", 0.0)),
            },
            "day": prev["date"],
        }
        # Also load previous minifold snapshots
        prev_minifolds = _get_latest_minifold_snapshots(
            conn, user_id=user_id, day=day,
        )
        if prev_minifolds:
            previous_fold["minifolds"] = prev_minifolds

    # 3. Build circle history for EWS features
    circle_history = [
        {
            "r": float(cs["r"]),
            "date": cs["date"],
            "velocity": float(cs.get("velocity", 0.0)),
        }
        for cs in circle_snaps
    ]

    # 4. Compute the fold
    fold = compute_questions_fold(
        x_hat=x_hat,
        x_uncertainty=x_uncertainty,
        coverage=coverage,
        day=day,
        previous_fold=previous_fold,
        circle_history=circle_history,
    )

    return fold.to_identity_mask()


def get_ews_features(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
    window_days: int = 30,
) -> List[Dict[str, Any]]:
    start = (day - timedelta(days=max(1, int(window_days)) - 1)).isoformat()
    rows = conn.execute(
        """
        SELECT *
        FROM ews_features
        WHERE user_id=?
          AND date>=?
        ORDER BY date ASC, created_at ASC;
        """,
        (user_id, start),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "feature_id": str(r["feature_id"]),
                "user_id": str(r["user_id"]),
                "date": str(r["date"]),
                "timestamp": str(r["timestamp"]),
                "window_size_days": int(r["window_size_days"]),
                "var_r": float(r["var_r"]),
                "ac1_r": float(r["ac1_r"]),
                "trend_speed": float(r["trend_speed"]),
                "trend_accel": float(r["trend_accel"]),
                "recovery_rate": float(r["recovery_rate"]),
                "ews_score": float(r["ews_score"]),
                "notes": json.loads(str(r["notes_json"])) if r["notes_json"] else {},
            }
        )
    return out


def get_drift_events(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
    window_days: int = 30,
) -> List[Dict[str, Any]]:
    start = (day - timedelta(days=max(1, int(window_days)) - 1)).isoformat()
    rows = conn.execute(
        """
        SELECT *
        FROM drift_events
        WHERE user_id=?
          AND date>=?
        ORDER BY date ASC, created_at ASC;
        """,
        (user_id, start),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "event_id": str(r["event_id"]),
                "user_id": str(r["user_id"]),
                "date": str(r["date"]),
                "timestamp": str(r["timestamp"]),
                "coherence_score": float(r["coherence_score"]),
                "drift_domain": str(r["drift_domain"]),
                "triggered_instrument": str(r["triggered_instrument"]) if r["triggered_instrument"] else None,
                "ews_score": float(r["ews_score"]) if r["ews_score"] is not None else None,
                "reason_codes": json.loads(str(r["reason_codes_json"])) if r["reason_codes_json"] else [],
                "details": json.loads(str(r["details_json"])) if r["details_json"] else {},
            }
        )
    return out


def get_coherence_tier_contract() -> Dict[str, Any]:
    return coherence_tier_contract()


def get_drift_routing_contract(*, drift_routing_table_path: Optional[str] = None) -> Dict[str, Any]:
    version, routes = load_drift_routes(drift_routing_table_path)
    return route_contract(version, routes)


def suggest_emotion_deep_dive(
    conn: sqlite3.Connection,
    *,
    cfg: QuestionsAgentConfig,
    registry_root: str,
    user_id: str,
    day: date,
) -> Dict[str, Any]:
    ensure_user(conn, user_id)
    ensure_user_profile(conn, user_id)
    reg_v = ensure_registry_active(conn, registry_root)
    registry = load_registry(registry_root, reg_v)
    profile = get_user_profile(conn, user_id)
    declined = {str(i) for i in profile.get("permanently_declined_item_ids", [])}

    latest_state = conn.execute(
        """
        SELECT x_uncertainty_json
        FROM state_snapshots
        WHERE user_id=? AND date<=?
        ORDER BY date DESC, created_at DESC
        LIMIT 1;
        """,
        (user_id, day.isoformat()),
    ).fetchone()
    uncertainty_values: List[float] = []
    if latest_state and latest_state["x_uncertainty_json"]:
        try:
            obj = json.loads(str(latest_state["x_uncertainty_json"]))
            if isinstance(obj, dict):
                uncertainty_values = [float(v) for v in obj.values()]
        except Exception:
            uncertainty_values = []

    mean_unc = (sum(uncertainty_values) / len(uncertainty_values)) if uncertainty_values else 0.0
    max_unc = max(uncertainty_values) if uncertainty_values else 0.0

    open_safety = _list_open_safety_events(
        conn,
        user_id=user_id,
        day=day,
        lookback_days=int(cfg.safety_event_lookback_days),
    )
    triggered = bool(max_unc >= 0.55 or mean_unc >= 0.45 or open_safety)
    trigger_reasons: List[str] = []
    if max_unc >= 0.55:
        trigger_reasons.append("high_uncertainty_peak")
    if mean_unc >= 0.45:
        trigger_reasons.append("high_uncertainty_mean")
    if open_safety:
        trigger_reasons.append("open_safety_event")

    last_asked = _get_last_asked_dates(conn, user_id=user_id, day=day, lookback_days=30)
    candidate_ids: List[str] = []
    for item in registry.items.values():
        if item.id in declined:
            continue
        tags = {str(t).lower() for t in item.tags}
        if not (tags & EMOTION_TAGS):
            continue
        candidate_ids.append(item.id)

    if not candidate_ids:
        # fallback to low-burden generic probes when emotional tags are unavailable
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
    target = max(8, min(12, int(cfg.emotion_deep_dive_item_count)))
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
    conn: sqlite3.Connection,
    *,
    user_id: str,
    body: Dict[str, Any],
    now_day: Optional[date] = None,
) -> Dict[str, Any]:
    ensure_user(conn, user_id)
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
    baseline_window_days = int(body.get("baseline_window_days") or 14)
    eval_window_days = int(body.get("eval_window_days") or 14)
    baseline_window_days = max(3, min(120, baseline_window_days))
    eval_window_days = max(3, min(120, eval_window_days))

    exists = conn.execute(
        "SELECT experiment_id FROM experiments WHERE experiment_id=? AND user_id=?;",
        (experiment_id, user_id),
    ).fetchone()
    if exists:
        conn.execute(
            """
            UPDATE experiments
            SET name=?, status=?, start_date=?, end_date=?,
                target_metrics_json=?, baseline_window_days=?, eval_window_days=?,
                stopping_rules_json=?, intervention_json=?, updated_at=?
            WHERE experiment_id=? AND user_id=?;
            """,
            (
                name,
                status,
                start_day.isoformat(),
                end_day.isoformat() if end_day else None,
                json.dumps(target_metrics, ensure_ascii=False),
                baseline_window_days,
                eval_window_days,
                json.dumps(stopping_rules, ensure_ascii=False),
                json.dumps(intervention, ensure_ascii=False),
                now_iso(),
                experiment_id,
                user_id,
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO experiments(
              experiment_id, user_id, name, status, start_date, end_date,
              target_metrics_json, baseline_window_days, eval_window_days,
              stopping_rules_json, intervention_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                experiment_id,
                user_id,
                name,
                status,
                start_day.isoformat(),
                end_day.isoformat() if end_day else None,
                json.dumps(target_metrics, ensure_ascii=False),
                baseline_window_days,
                eval_window_days,
                json.dumps(stopping_rules, ensure_ascii=False),
                json.dumps(intervention, ensure_ascii=False),
                now_iso(),
                now_iso(),
            ),
        )
    conn.commit()
    return get_experiment(conn, user_id=user_id, experiment_id=experiment_id)


def list_experiments(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    clauses = ["user_id=?"]
    params: List[Any] = [user_id]
    if status:
        clauses.append("status=?")
        params.append(str(status).strip().lower())
    rows = conn.execute(
        f"""
        SELECT *
        FROM experiments
        WHERE {" AND ".join(clauses)}
        ORDER BY start_date DESC, created_at DESC;
        """,
        tuple(params),
    ).fetchall()
    return [_row_to_experiment(r) for r in rows]


def get_experiment(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    experiment_id: str,
) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM experiments WHERE user_id=? AND experiment_id=?;",
        (user_id, experiment_id),
    ).fetchone()
    if not row:
        raise ValueError("Unknown experiment")
    return _row_to_experiment(row)


def compute_experiment_results(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    experiment_id: str,
    day: Optional[date] = None,
) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM experiments WHERE user_id=? AND experiment_id=?;",
        (user_id, experiment_id),
    ).fetchone()
    if not row:
        raise ValueError("Unknown experiment")
    current_day = day or date.today()
    start_day = parse_date(str(row["start_date"]))
    end_day = parse_date(str(row["end_date"])) if row["end_date"] else current_day
    if end_day > current_day:
        end_day = current_day

    baseline_days = max(3, int(row["baseline_window_days"] or 14))
    eval_days = max(3, int(row["eval_window_days"] or 14))

    baseline_start = start_day - timedelta(days=baseline_days)
    baseline_end = start_day - timedelta(days=1)
    eval_start = start_day
    eval_end = min(end_day, start_day + timedelta(days=eval_days - 1))

    baseline_rows = conn.execute(
        """
        SELECT r
        FROM circle_snapshots
        WHERE user_id=? AND date>=? AND date<=?
        ORDER BY date ASC;
        """,
        (user_id, baseline_start.isoformat(), baseline_end.isoformat()),
    ).fetchall()
    eval_rows = conn.execute(
        """
        SELECT r
        FROM circle_snapshots
        WHERE user_id=? AND date>=? AND date<=?
        ORDER BY date ASC;
        """,
        (user_id, eval_start.isoformat(), eval_end.isoformat()),
    ).fetchall()

    baseline_r = [float(r["r"]) for r in baseline_rows]
    eval_r = [float(r["r"]) for r in eval_rows]
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
    conn.execute(
        "UPDATE experiments SET results_json=?, updated_at=? WHERE experiment_id=? AND user_id=?;",
        (json.dumps(results, ensure_ascii=False), now_iso(), experiment_id, user_id),
    )
    conn.commit()
    return results


def build_domain_promotion_readiness(
    conn: sqlite3.Connection,
    *,
    registry_root: str,
    min_users: int,
    min_cycles: int,
    min_weeks: int,
    min_paired_ratio: float,
) -> Dict[str, Any]:
    reg_v = ensure_registry_active(conn, registry_root)
    registry = load_registry(registry_root, reg_v)
    scale_domains = registry_scale_domains(registry)
    item_domains = registry_item_domains(registry)

    domains = [d for d in registry_domains(registry) if d != CARDIOMETABOLIC_DOMAIN]
    report_rows: List[Dict[str, Any]] = []

    for domain in domains:
        domain_items = sorted([item_id for item_id, ds in item_domains.items() if domain in ds])
        if not domain_items:
            continue
        placeholders = ",".join(["?"] * len(domain_items))
        rows = conn.execute(
            f"""
            SELECT ae.user_id, ds.date
            FROM answer_events ae
            JOIN daily_sessions ds ON ds.session_id = ae.session_id
            WHERE ae.item_id IN ({placeholders})
            ORDER BY ds.date ASC;
            """,
            tuple(domain_items),
        ).fetchall()
        per_user_days: Dict[str, Set[str]] = {}
        all_days: Set[str] = set()
        for row in rows:
            uid = str(row["user_id"])
            d_iso = str(row["date"])
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
                state = conn.execute(
                    """
                    SELECT 1
                    FROM state_snapshots
                    WHERE user_id=? AND date=?
                    LIMIT 1;
                    """,
                    (uid, d_iso),
                ).fetchone()
                if state:
                    paired_days += 1

        paired_ratio = (float(paired_days) / float(total_user_days)) if total_user_days > 0 else 0.0

        domain_scales = [sid for sid, dset in scale_domains.items() if domain in dset]
        variance_ok = False
        if domain_scales:
            placeholders_scales = ",".join(["?"] * len(domain_scales))
            score_rows = conn.execute(
                f"""
                SELECT normalized_score
                FROM scale_scores
                WHERE scale_id IN ({placeholders_scales});
                """,
                tuple(domain_scales),
            ).fetchall()
            vals = [float(r["normalized_score"]) for r in score_rows]
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
    conn: sqlite3.Connection,
    *,
    registry_root: str,
    user_id: str,
    day: date,
) -> Dict[str, Any]:
    reg_v = ensure_registry_active(conn, registry_root)
    registry = load_registry(registry_root, reg_v)
    progress = compute_user_scale_progress(conn, registry=registry, user_id=user_id, day=day)

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

    circle = get_circle_snapshots(conn, user_id=user_id, day=day, window_days=30)
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


def _row_to_experiment(row: sqlite3.Row) -> Dict[str, Any]:
    def _json_or_empty(value: Any) -> Dict[str, Any]:
        try:
            obj = json.loads(str(value or "{}"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {
        "experiment_id": str(row["experiment_id"]),
        "user_id": str(row["user_id"]),
        "name": str(row["name"]),
        "status": str(row["status"]),
        "start_date": str(row["start_date"]),
        "end_date": str(row["end_date"]) if row["end_date"] else None,
        "target_metrics": _json_or_empty(row["target_metrics_json"]),
        "baseline_window_days": int(row["baseline_window_days"]),
        "eval_window_days": int(row["eval_window_days"]),
        "stopping_rules": _json_or_empty(row["stopping_rules_json"]),
        "intervention": _json_or_empty(row["intervention_json"]),
        "results": _json_or_empty(row["results_json"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def _upsert_daily_state_and_circle_snapshots(
    conn: sqlite3.Connection,
    *,
    registry: Registry,
    user_id: str,
    day: date,
    state_schema_version: str,
    drift_routing_table_path: Optional[str] = None,
    site_config: Optional[SiteConfig] = None,
) -> None:
    latest_scores = _get_latest_scale_scores_for_state(conn, user_id=user_id, day=day)
    prev_state = _get_latest_state_snapshot_before(conn, user_id=user_id, day=day)
    prev_circle = _get_latest_circle_snapshot_before(conn, user_id=user_id, day=day)
    timestamp = date_to_start_iso(day)

    computed_state = compute_state_snapshot(
        registry=registry,
        latest_scores=latest_scores,
        previous_x_hat=(prev_state.get("x_hat") if prev_state else None),
    )

    state_model_version = STATE_MODEL_VERSION
    existing_state = conn.execute(
        """
        SELECT snapshot_id
        FROM state_snapshots
        WHERE user_id=? AND date=? AND state_schema_version=? AND model_version=?
        LIMIT 1;
        """,
        (user_id, day.isoformat(), state_schema_version, state_model_version),
    ).fetchone()
    state_snapshot_id = str(existing_state["snapshot_id"]) if existing_state else str(uuid.uuid4())
    conn.execute(
        """
        INSERT OR REPLACE INTO state_snapshots(
          snapshot_id, user_id, date, timestamp,
          state_schema_version, model_version,
          x_hat_json, x_uncertainty_json, coverage_json,
          source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'questions_agent', ?);
        """,
        (
            state_snapshot_id,
            user_id,
            day.isoformat(),
            timestamp,
            str(state_schema_version or STATE_SCHEMA_VERSION_DEFAULT),
            state_model_version,
            json.dumps(computed_state["x_hat"], ensure_ascii=False),
            json.dumps(computed_state["x_uncertainty"], ensure_ascii=False),
            json.dumps(computed_state["coverage"], ensure_ascii=False),
            now_iso(),
        ),
    )

    computed_circle = compute_circle_snapshot(
        day=day,
        x_hat=dict(computed_state["x_hat"]),
        x_uncertainty=dict(computed_state["x_uncertainty"]),
        previous_circle=prev_circle,
    )
    coherence_score = float(
        computed_circle.get("local_coherence")
        if computed_circle.get("local_coherence") is not None
        else computed_circle.get("coherence")
        if computed_circle.get("coherence") is not None
        else coherence_score_from_radius(float(computed_circle.get("r") or 0.0))
    )
    coherence_tier = coherence_tier_for_score(float(coherence_score))
    projection_version = CIRCLE_PROJECTION_VERSION
    existing_circle = conn.execute(
        """
        SELECT snapshot_id
        FROM circle_snapshots
        WHERE user_id=? AND date=? AND projection_version=?
        LIMIT 1;
        """,
        (user_id, day.isoformat(), projection_version),
    ).fetchone()
    circle_snapshot_id = str(existing_circle["snapshot_id"]) if existing_circle else str(uuid.uuid4())
    conn.execute(
        """
        INSERT OR REPLACE INTO circle_snapshots(
          snapshot_id, user_id, date, timestamp, state_snapshot_id,
          projection_version, anchor_version,
          z_json, z_star_json,
          r, theta, velocity, acceleration, coherence_score, coherence_tier_json,
          uncertainty_json, source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'questions_agent', ?);
        """,
        (
            circle_snapshot_id,
            user_id,
            day.isoformat(),
            timestamp,
            state_snapshot_id,
            projection_version,
            CIRCLE_ANCHOR_VERSION,
            json.dumps(computed_circle["z"], ensure_ascii=False),
            json.dumps(computed_circle["z_star"], ensure_ascii=False),
            float(computed_circle["r"]),
            float(computed_circle["theta"]),
            float(computed_circle["velocity"]),
            float(computed_circle["acceleration"]),
            float(coherence_score),
            json.dumps(coherence_tier, ensure_ascii=False),
            json.dumps(computed_circle["uncertainty"], ensure_ascii=False),
            now_iso(),
        ),
    )
    # --- MiniFold sub-circles (per-decoherence-mode projections) ---
    prev_minifolds = _get_latest_minifold_snapshots(conn, user_id=user_id, day=day)
    minifold_circles = compute_minifold_circles(
        x_hat=dict(computed_state["x_hat"]),
        x_uncertainty=dict(computed_state["x_uncertainty"]),
        previous_minifolds=prev_minifolds,
        day=day,
    )
    _upsert_minifold_snapshots(
        conn,
        user_id=user_id,
        day=day,
        circle_snapshot_id=circle_snapshot_id,
        minifold_circles=minifold_circles,
    )

    _upsert_ews_and_drift(
        conn,
        registry=registry,
        user_id=user_id,
        day=day,
        computed_circle=computed_circle,
        latest_scores=latest_scores,
        drift_routing_table_path=drift_routing_table_path,
        site_config=site_config,
    )
    conn.commit()


def _get_latest_minifold_snapshots(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
) -> Dict[str, Dict[str, Any]]:
    """Load previous MiniFold sub-circle snapshots for EMA attractor update."""
    result: Dict[str, Dict[str, Any]] = {}
    rows = conn.execute(
        """
        SELECT mode, z_json, z_star_json, r, theta, velocity, acceleration, date
        FROM minifold_snapshots
        WHERE user_id=? AND date < ?
        ORDER BY date DESC
        LIMIT 4;
        """,
        (user_id, day.isoformat()),
    ).fetchall()
    for row in rows:
        mode = str(row["mode"])
        if mode not in result:  # keep latest per mode
            result[mode] = {
                "z": json.loads(str(row["z_json"])),
                "z_star": json.loads(str(row["z_star_json"])),
                "r": float(row["r"]),
                "theta": float(row["theta"]),
                "velocity": float(row["velocity"]),
                "acceleration": float(row["acceleration"]),
                "date": str(row["date"]),
            }
    return result


def _upsert_minifold_snapshots(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
    circle_snapshot_id: str,
    minifold_circles: Dict[str, Dict[str, Any]],
) -> None:
    """Store MiniFold sub-circle snapshots for each decoherence mode."""
    for mode, mf in minifold_circles.items():
        snapshot_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT OR REPLACE INTO minifold_snapshots(
              snapshot_id, user_id, date, circle_snapshot_id, mode, version,
              z_json, z_star_json, r, theta, velocity, acceleration,
              coherence, uncertainty, coverage_ratio,
              dimensions_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                snapshot_id,
                user_id,
                day.isoformat(),
                circle_snapshot_id,
                mode,
                MINIFOLD_VERSION,
                json.dumps(mf["z"], ensure_ascii=False),
                json.dumps(mf["z_star"], ensure_ascii=False),
                float(mf["r"]),
                float(mf["theta"]),
                float(mf["velocity"]),
                float(mf["acceleration"]),
                float(mf.get("local_coherence", mf.get("coherence", 0.0))),
                float(mf["uncertainty"]),
                float(mf["coverage_ratio"]),
                json.dumps(mf["dimensions"], ensure_ascii=False),
                now_iso(),
            ),
        )


def _upsert_ews_and_drift(
    conn: sqlite3.Connection,
    *,
    registry: Registry,
    user_id: str,
    day: date,
    computed_circle: Dict[str, Any],
    latest_scores: Dict[str, Dict[str, Any]],
    drift_routing_table_path: Optional[str] = None,
    site_config: Optional[SiteConfig] = None,
) -> None:
    lookback_days = 14
    start = (day - timedelta(days=lookback_days - 1)).isoformat()
    rows = conn.execute(
        """
        SELECT date, r, velocity, acceleration
        FROM circle_snapshots
        WHERE user_id=?
          AND date>=?
          AND date<?
          AND projection_version=?
        ORDER BY date ASC, created_at ASC;
        """,
        (user_id, start, day.isoformat(), CIRCLE_PROJECTION_VERSION),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            """
            SELECT date, r, velocity, acceleration
            FROM circle_snapshots
            WHERE user_id=?
              AND date>=?
              AND date<?
            ORDER BY date ASC, created_at ASC;
            """,
            (user_id, start, day.isoformat()),
        ).fetchall()

    history: List[Dict[str, Any]] = [
        {
            "date": str(r["date"]),
            "r": float(r["r"]),
            "velocity": float(r["velocity"]),
            "acceleration": float(r["acceleration"]),
        }
        for r in rows
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
    existing = conn.execute(
        """
        SELECT feature_id
        FROM ews_features
        WHERE user_id=? AND date=? AND window_size_days=?
        LIMIT 1;
        """,
        (user_id, day.isoformat(), lookback_days),
    ).fetchone()
    feature_id = str(existing["feature_id"]) if existing else str(uuid.uuid4())
    conn.execute(
        """
        INSERT OR REPLACE INTO ews_features(
          feature_id, user_id, date, timestamp, window_size_days,
          var_r, ac1_r, trend_speed, trend_accel, recovery_rate, ews_score,
          notes_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            feature_id,
            user_id,
            day.isoformat(),
            date_to_start_iso(day),
            lookback_days,
            float(ews["var_r"]),
            float(ews["ac1_r"]),
            float(ews["trend_speed"]),
            float(ews["trend_accel"]),
            float(ews["recovery_rate"]),
            float(ews["ews_score"]),
            json.dumps(
                {
                    "version": "ews_v1",
                    "coherence_radius_window_days": lookback_days,
                },
                ensure_ascii=False,
            ),
            now_iso(),
        ),
    )

    radius = float(computed_circle.get("r") or 0.0)
    coherence_score = float(
        computed_circle.get("coherence")
        if computed_circle.get("coherence") is not None
        else coherence_score_from_radius(radius)
    )
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

    anamnesis_enabled = bool(site_config and site_config.anamnesis_enabled)

    if not is_drift:
        conn.execute("DELETE FROM drift_events WHERE user_id=? AND date=?;", (user_id, day.isoformat()))
        # Even without drift, evaluate existing active episodes for resolution
        if anamnesis_enabled:
            _evaluate_anamnesis_resolutions(
                conn,
                user_id=user_id,
                day=day,
                ews_score=float(ews["ews_score"]),
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

    existing_event = conn.execute(
        "SELECT event_id FROM drift_events WHERE user_id=? AND date=? LIMIT 1;",
        (user_id, day.isoformat()),
    ).fetchone()
    event_id = str(existing_event["event_id"]) if existing_event else str(uuid.uuid4())
    conn.execute(
        """
        INSERT OR REPLACE INTO drift_events(
          event_id, user_id, date, timestamp, coherence_score,
          drift_domain, triggered_instrument, ews_score,
          reason_codes_json, details_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            event_id,
            user_id,
            day.isoformat(),
            date_to_start_iso(day),
            float(coherence_score),
            str(drift_domain),
            str(triggered_instrument) if triggered_instrument else None,
            float(ews["ews_score"]),
            json.dumps(reason_codes, ensure_ascii=False),
            json.dumps(
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
            ),
            now_iso(),
        ),
    )

    # --- Anamnesis: drift-triggered clinical branching (Claim Family 4) ---
    if anamnesis_enabled:
        _process_anamnesis_on_drift(
            conn,
            registry=registry,
            user_id=user_id,
            day=day,
            ews_score=float(ews["ews_score"]),
            radius=radius,
            baseline_radius_mean=mu,
            baseline_radius_std=sigma,
            baseline_radius_n=len(baseline_r_vals),
            velocity=velocity,
            drift_domain=drift_domain,
            drift_routing_table_path=drift_routing_table_path,
        )


def _load_active_anamnesis_episodes(
    conn: sqlite3.Connection,
    *,
    user_id: str,
) -> List[AnamnesisEpisode]:
    """Load all active anamnesis episodes for a user."""
    rows = conn.execute(
        """
        SELECT *
        FROM anamnesis_episodes
        WHERE user_id=? AND status='active'
        ORDER BY trigger_date ASC;
        """,
        (user_id,),
    ).fetchall()
    episodes: List[AnamnesisEpisode] = []
    for r in rows:
        try:
            scale_ids = tuple(json.loads(str(r["triggered_scale_ids_json"] or "[]")))
        except Exception:
            scale_ids = ()
        try:
            follow_up_ids = tuple(json.loads(str(r["follow_up_item_ids_json"] or "[]")))
        except Exception:
            follow_up_ids = ()
        episodes.append(AnamnesisEpisode(
            episode_id=str(r["episode_id"]),
            user_id=str(r["user_id"]),
            trigger_date=str(r["trigger_date"]),
            drift_domain=str(r["drift_domain"]),
            trigger_type=str(r["trigger_type"]),
            trigger_value=float(r["trigger_value"]),
            triggered_scale_ids=scale_ids,
            status=str(r["status"]),
            follow_up_item_ids=follow_up_ids,
            priority=int(r["priority"]),
            max_days=int(r["max_days"]),
            observations_collected=int(r["observations_collected"]),
            resolution_date=str(r["resolution_date"]) if r["resolution_date"] else None,
            resolution_reason=str(r["resolution_reason"]) if r["resolution_reason"] else None,
        ))
    return episodes


def _insert_anamnesis_episode(
    conn: sqlite3.Connection,
    episode: AnamnesisEpisode,
) -> None:
    """Insert a new anamnesis episode."""
    conn.execute(
        """
        INSERT OR IGNORE INTO anamnesis_episodes(
          episode_id, user_id, trigger_date, drift_domain,
          trigger_type, trigger_value, triggered_scale_ids_json,
          status, follow_up_item_ids_json, priority, max_days,
          observations_collected, resolution_date, resolution_reason,
          created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            episode.episode_id,
            episode.user_id,
            episode.trigger_date,
            episode.drift_domain,
            episode.trigger_type,
            float(episode.trigger_value),
            json.dumps(list(episode.triggered_scale_ids), ensure_ascii=False),
            episode.status,
            json.dumps(list(episode.follow_up_item_ids), ensure_ascii=False),
            episode.priority,
            episode.max_days,
            episode.observations_collected,
            episode.resolution_date,
            episode.resolution_reason,
            now_iso(),
            now_iso(),
        ),
    )


def _update_anamnesis_episode_status(
    conn: sqlite3.Connection,
    *,
    episode_id: str,
    new_status: str,
    resolution_date: Optional[str] = None,
    resolution_reason: Optional[str] = None,
    observations_collected: Optional[int] = None,
) -> None:
    """Update an anamnesis episode's status and resolution info."""
    updates = ["status=?", "updated_at=?"]
    params: List[Any] = [new_status, now_iso()]
    if resolution_date is not None:
        updates.append("resolution_date=?")
        params.append(resolution_date)
    if resolution_reason is not None:
        updates.append("resolution_reason=?")
        params.append(resolution_reason)
    if observations_collected is not None:
        updates.append("observations_collected=?")
        params.append(observations_collected)
    params.append(episode_id)
    conn.execute(
        f"UPDATE anamnesis_episodes SET {', '.join(updates)} WHERE episode_id=?;",
        tuple(params),
    )


def _process_anamnesis_on_drift(
    conn: sqlite3.Connection,
    *,
    registry: Registry,
    user_id: str,
    day: date,
    ews_score: float,
    radius: float,
    baseline_radius_mean: float,
    baseline_radius_std: float,
    baseline_radius_n: int,
    velocity: float,
    drift_domain: str,
    drift_routing_table_path: Optional[str] = None,
) -> None:
    """
    Run the anamnesis loop after drift is detected.
    Creates new episodes, resolves old ones, and enqueues follow-up items.
    """
    from questions_agent_platform.pipeline.anamnesis import (
        build_domain_to_items_map,
        build_domain_to_scales_map,
    )
    from questions_agent_platform.pipeline.drift_routing import load_drift_routes as _load_dr

    active_episodes = _load_active_anamnesis_episodes(conn, user_id=user_id)

    # Build radius baseline for the anamnesis check
    radius_baseline: Optional[BaselineState] = None
    if baseline_radius_n >= 2 and baseline_radius_std > 0:
        radius_baseline = BaselineState(
            mean=baseline_radius_mean,
            var=baseline_radius_std ** 2,
            n=baseline_radius_n,
        )

    # Build domain → items/scales maps from drift routing table
    try:
        _, routes = _load_dr(drift_routing_table_path)
    except Exception:
        _, routes = _load_dr(None)

    scale_items: Dict[str, List[str]] = {}
    for scale in registry.scales.values():
        scale_items[scale.id] = [si.item_id for si in scale.items]

    domain_to_items = build_domain_to_items_map(routes, scale_items)
    domain_to_scales = build_domain_to_scales_map(routes)

    actions = run_anamnesis_check(
        user_id=user_id,
        current_date=day,
        active_episodes=active_episodes,
        ews_score=ews_score,
        radius_current=radius,
        radius_baseline=radius_baseline,
        velocity=velocity,
        drift_domain_to_scale_ids=domain_to_scales,
        drift_domain_to_item_ids=domain_to_items,
    )

    for action in actions:
        if action.action_type == "start_episode":
            episode = create_anamnesis_episode(
                user_id=user_id,
                trigger_date=day,
                trigger_type="drift",
                trigger_value=ews_score,
                drift_domain=drift_domain,
                triggered_scale_ids=domain_to_scales.get(drift_domain, []),
                follow_up_item_ids=list(action.follow_up_item_ids),
            )
            _insert_anamnesis_episode(conn, episode)
            # Enqueue follow-up items into the follow_up_queue
            for item_id in action.follow_up_item_ids:
                _enqueue_follow_up_queue_item(
                    conn,
                    user_id=user_id,
                    item_id=item_id,
                    scale_id=None,
                    reason_code="anamnesis_drift_probe",
                    priority=float(action.priority_boost),
                    earliest_date=day,
                    reason_detail={
                        "episode_id": episode.episode_id,
                        "drift_domain": drift_domain,
                    },
                )
            logger.info(
                "anamnesis.start_episode user=%s domain=%s items=%d",
                user_id, drift_domain, len(action.follow_up_item_ids),
            )

        elif action.action_type == "resolve_episode":
            _update_anamnesis_episode_status(
                conn,
                episode_id=action.episode_id or "",
                new_status="resolved",
                resolution_date=day.isoformat(),
                resolution_reason=action.reason,
            )
            # Resolve any pending follow-up queue items tied to this episode
            if action.episode_id:
                _resolve_follow_up_for_anamnesis_episode(
                    conn, user_id=user_id, episode_id=action.episode_id, day=day,
                )
            logger.info(
                "anamnesis.resolve_episode user=%s episode=%s reason=%s",
                user_id, action.episode_id, action.reason,
            )

        elif action.action_type == "continue_episode":
            if action.episode_id:
                _update_anamnesis_episode_status(
                    conn,
                    episode_id=action.episode_id,
                    new_status="active",
                    observations_collected=len(action.follow_up_item_ids),
                )


def _evaluate_anamnesis_resolutions(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
    ews_score: float,
) -> None:
    """When no drift is detected, evaluate active episodes for resolution."""
    active_episodes = _load_active_anamnesis_episodes(conn, user_id=user_id)
    for ep in active_episodes:
        resolved = evaluate_episode_resolution(
            ep,
            current_date=day,
            current_ews_score=ews_score,
        )
        if resolved.status != "active":
            _update_anamnesis_episode_status(
                conn,
                episode_id=resolved.episode_id,
                new_status=resolved.status,
                resolution_date=day.isoformat(),
                resolution_reason=resolved.resolution_reason or "drift_subsided",
            )
            _resolve_follow_up_for_anamnesis_episode(
                conn, user_id=user_id, episode_id=resolved.episode_id, day=day,
            )


def _resolve_follow_up_for_anamnesis_episode(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    episode_id: str,
    day: date,
) -> None:
    """Resolve pending follow-up queue items linked to an anamnesis episode."""
    conn.execute(
        """
        UPDATE follow_up_queue
        SET status='resolved', resolved_at=?, resolved_reason=?
        WHERE user_id=? AND status='pending'
          AND reason_code='anamnesis_drift_probe'
          AND reason_detail_json LIKE ?;
        """,
        (
            now_iso(),
            f"episode_resolved:{episode_id}",
            user_id,
            f'%"{episode_id}"%',
        ),
    )


def _get_latest_scale_scores_for_state(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT scale_id, window_end, normalized_score, personal_z, delta_vs_prev
        FROM scale_scores
        WHERE user_id=? AND window_end<=?
        ORDER BY window_end ASC, computed_at ASC;
        """,
        (user_id, day.isoformat()),
    ).fetchall()
    latest: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        latest[str(r["scale_id"])] = {
            "scale_id": str(r["scale_id"]),
            "window_end": str(r["window_end"]),
            "normalized_score": float(r["normalized_score"]),
            "personal_z": float(r["personal_z"]) if r["personal_z"] is not None else None,
            "delta_vs_prev": float(r["delta_vs_prev"]) if r["delta_vs_prev"] is not None else None,
        }
    return latest


def _get_latest_state_snapshot_before(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT date, x_hat_json
        FROM state_snapshots
        WHERE user_id=? AND date<?
        ORDER BY date DESC, created_at DESC
        LIMIT 1;
        """,
        (user_id, day.isoformat()),
    ).fetchone()
    if not row:
        return None
    try:
        x_hat = json.loads(str(row["x_hat_json"]))
        if not isinstance(x_hat, dict):
            return None
        return {"date": parse_date(str(row["date"])), "x_hat": {str(k): float(v) for k, v in x_hat.items()}}
    except Exception:
        return None


def _get_latest_circle_snapshot_before(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    day: date,
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT date, z_json, z_star_json, r, velocity, uncertainty_json
        FROM circle_snapshots
        WHERE user_id=? AND date<?
          AND projection_version=?
        ORDER BY date DESC, created_at DESC
        LIMIT 1;
        """,
        (user_id, day.isoformat(), CIRCLE_PROJECTION_VERSION),
    ).fetchone()
    if not row:
        row = conn.execute(
            """
            SELECT date, z_json, z_star_json, r, velocity, uncertainty_json
            FROM circle_snapshots
            WHERE user_id=? AND date<?
            ORDER BY date DESC, created_at DESC
            LIMIT 1;
            """,
            (user_id, day.isoformat()),
        ).fetchone()
    if not row:
        return None
    try:
        z = json.loads(str(row["z_json"]))
        z_star = json.loads(str(row["z_star_json"]))
        if not isinstance(z, list) or len(z) != 2 or not isinstance(z_star, list) or len(z_star) != 2:
            return None
        return {
            "date": parse_date(str(row["date"])),
            "z": [float(z[0]), float(z[1])],
            "z_star": [float(z_star[0]), float(z_star[1])],
            "r": float(row["r"] or 0.0),
            "velocity": float(row["velocity"] or 0.0),
            "uncertainty": json.loads(str(row["uncertainty_json"])) if row["uncertainty_json"] else {},
        }
    except Exception:
        return None


def compute_user_scale_progress(conn: sqlite3.Connection, *, registry: Registry, user_id: str, day: date) -> Dict[str, Any]:
    answered_by_scale, _ = _compute_scale_windows(conn, registry=registry, user_id=user_id, day=day)
    out: Dict[str, Any] = {}
    for scale in registry.scales.values():
        progress = compute_scale_progress(scale, tuple(sorted(answered_by_scale.get(scale.id, set()))))
        last = _get_last_scale_score(conn, user_id=user_id, scale_id=scale.id)
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


def list_users(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT user_id FROM users ORDER BY created_at DESC;").fetchall()
    return [str(r["user_id"]) for r in rows]


def get_scale_history(conn: sqlite3.Connection, *, user_id: str, scale_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT * FROM scale_scores
        WHERE user_id=? AND scale_id=?
        ORDER BY computed_at ASC
        LIMIT ?;
        """,
        (user_id, scale_id, int(limit)),
    ).fetchall()
    return [_row_to_scale_score(r) for r in rows]


def _row_to_session(row: sqlite3.Row) -> DailySession:
    core = tuple(json.loads(row["core_questions_json"]))
    extra = tuple(tuple(b) for b in json.loads(row["extra_batches_json"]))
    explain = json.loads(row["selection_explain_json"])
    return DailySession(
        session_id=str(row["session_id"]),
        user_id=str(row["user_id"]),
        date=str(row["date"]),
        registry_version=str(row["registry_version"]),
        timeframe=str(row["timeframe"]) if "timeframe" in row.keys() and row["timeframe"] else "last_7_days",
        status=str(row["status"]) if "status" in row.keys() and row["status"] else "created",
        core_item_ids=core,
        extra_batches=extra,
        extra_batches_used=int(row["extra_batches_used"]),
        selection_explain=explain,
        selection_mode=str(row["selection_mode"]) if "selection_mode" in row.keys() else "deterministic",
        policy_decision_id=str(row["policy_decision_id"]) if ("policy_decision_id" in row.keys() and row["policy_decision_id"]) else None,
    )


def _get_last_asked_dates(conn: sqlite3.Connection, *, user_id: str, day: date, lookback_days: int) -> Dict[str, date]:
    start = (day - timedelta(days=int(lookback_days))).isoformat()
    rows = conn.execute(
        """
        SELECT date, core_questions_json, extra_batches_json
        FROM daily_sessions
        WHERE user_id=? AND date>=?
        ORDER BY date DESC;
        """,
        (user_id, start),
    ).fetchall()
    last: Dict[str, date] = {}
    for r in rows:
        d = parse_date(str(r["date"]))
        for item_id in json.loads(r["core_questions_json"]):
            last.setdefault(str(item_id), d)
        for batch in json.loads(r["extra_batches_json"]):
            for item_id in batch:
                last.setdefault(str(item_id), d)
    return last


def _get_all_answered_item_ids(conn: sqlite3.Connection, *, user_id: str) -> Set[str]:
    rows = conn.execute(
        "SELECT DISTINCT item_id FROM answer_events WHERE user_id=?;",
        (user_id,),
    ).fetchall()
    return {str(r["item_id"]) for r in rows}


def _get_last_score_dates(conn: sqlite3.Connection, *, user_id: str) -> Dict[str, date]:
    rows = conn.execute(
        """
        SELECT scale_id, MAX(window_end) AS last_end
        FROM scale_scores
        WHERE user_id=?
        GROUP BY scale_id;
        """,
        (user_id,),
    ).fetchall()
    out: Dict[str, date] = {}
    for r in rows:
        if r["last_end"]:
            out[str(r["scale_id"])] = parse_date(str(r["last_end"]))
    return out


def _get_baselines(conn: sqlite3.Connection, *, user_id: str) -> Dict[str, BaselineState]:
    rows = conn.execute(
        "SELECT scale_id, mean, var, n FROM scale_baselines WHERE user_id=?;",
        (user_id,),
    ).fetchall()
    out: Dict[str, BaselineState] = {}
    for r in rows:
        out[str(r["scale_id"])] = BaselineState(mean=float(r["mean"]), var=float(r["var"]), n=int(r["n"]))
    return out


def _compute_scale_windows(
    conn: sqlite3.Connection, *, registry: Registry, user_id: str, day: date
) -> Tuple[Dict[str, Set[str]], Dict[str, float]]:
    answered_item_ids_by_scale: Dict[str, Set[str]] = {}
    rolling_normalized_by_scale: Dict[str, float] = {}
    end_iso = date_to_start_iso(day + timedelta(days=1))

    for scale in registry.scales.values():
        start_day = day - timedelta(days=int(scale.unlock_window_days) - 1)
        start_iso = date_to_start_iso(start_day)
        item_ids = [si.item_id for si in scale.items]
        latest = _fetch_latest_answers_for_items(
            conn, user_id=user_id, item_ids=item_ids, start_iso=start_iso, end_iso=end_iso
        )
        answered_item_ids_by_scale[scale.id] = set(latest.keys())
        score = compute_scale_score(scale, latest)
        if score is not None:
            rolling_normalized_by_scale[scale.id] = float(score.normalized_score)

    return answered_item_ids_by_scale, rolling_normalized_by_scale


def _fetch_latest_answers_for_items(
    conn: sqlite3.Connection, *, user_id: str, item_ids: Sequence[str], start_iso: str, end_iso: str
) -> Dict[str, float]:
    if not item_ids:
        return {}
    placeholders = ",".join(["?"] * len(item_ids))
    rows = conn.execute(
        f"""
        SELECT item_id, value, answered_at
        FROM answer_events
        WHERE user_id=?
          AND item_id IN ({placeholders})
          AND answered_at>=?
          AND answered_at<?
        ORDER BY answered_at DESC;
        """,
        (user_id, *list(item_ids), start_iso, end_iso),
    ).fetchall()
    latest: Dict[str, float] = {}
    for r in rows:
        item_id = str(r["item_id"])
        if item_id in latest:
            continue
        latest[item_id] = float(r["value"])
    return latest


def _get_last_scale_score(conn: sqlite3.Connection, *, user_id: str, scale_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT * FROM scale_scores
        WHERE user_id=? AND scale_id=?
        ORDER BY computed_at DESC
        LIMIT 1;
        """,
        (user_id, scale_id),
    ).fetchone()
    return _row_to_scale_score(row) if row else None


def _row_to_scale_score(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "score_id": str(row["score_id"]),
        "user_id": str(row["user_id"]),
        "scale_id": str(row["scale_id"]),
        "computed_at": str(row["computed_at"]),
        "window_start": str(row["window_start"]),
        "window_end": str(row["window_end"]),
        "raw_score": float(row["raw_score"]),
        "normalized_score": float(row["normalized_score"]),
        "confidence_tier": str(row["confidence_tier"]),
        "items_answered_count": int(row["items_answered_count"]),
        "items_required": int(row["items_required"]),
        "baseline_mean": float(row["baseline_mean"]) if row["baseline_mean"] is not None else None,
        "baseline_std": float(row["baseline_std"]) if row["baseline_std"] is not None else None,
        "personal_z": float(row["personal_z"]) if row["personal_z"] is not None else None,
        "delta_vs_prev": float(row["delta_vs_prev"]) if row["delta_vs_prev"] is not None else None,
        "risk_tier": str(row["risk_tier"]) if "risk_tier" in row.keys() and row["risk_tier"] is not None else None,
    }


def _compute_and_store_scores(
    conn: sqlite3.Connection,
    *,
    registry: Registry,
    user_id: str,
    day: date,
    session_id: Optional[str] = None,
    site_config: Optional[SiteConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Computes scale scores that are eligible to be recorded on this day, updates baselines,
    and returns newly inserted score records.  When site_config is provided and metadata is
    enabled, SE(theta) is adjusted using behavioural uncertainty modifiers (Claim Family 5).
    """
    new_scores: List[Dict[str, Any]] = []
    end_iso = date_to_start_iso(day + timedelta(days=1))
    metadata_enabled = bool(site_config and site_config.behavioural_metadata_enabled) and session_id is not None

    # Pull baseline state once
    baselines = _get_baselines(conn, user_id=user_id)

    # Load session engagement quality if metadata is enabled
    _session_engagement: Optional[str] = None
    if metadata_enabled:
        sup_row = conn.execute(
            "SELECT engagement_quality FROM session_uncertainty_profiles WHERE session_id=? LIMIT 1;",
            (session_id,),
        ).fetchone()
        if sup_row:
            _session_engagement = str(sup_row["engagement_quality"])

    for scale in registry.scales.values():
        # Gating: only record a new score if never scored, or retest interval has passed.
        last = _get_last_scale_score(conn, user_id=user_id, scale_id=scale.id)
        if last is not None:
            last_end = parse_date(last["window_end"])
            if (day - last_end).days < int(scale.retest_interval_days):
                continue

        start_day = day - timedelta(days=int(scale.unlock_window_days) - 1)
        start_iso = date_to_start_iso(start_day)
        item_ids = [si.item_id for si in scale.items]
        latest = _fetch_latest_answers_for_items(
            conn, user_id=user_id, item_ids=item_ids, start_iso=start_iso, end_iso=end_iso
        )
        score_res = compute_scale_score(scale, latest)
        if score_res is None:
            continue

        # Idempotency: at most one score per (user, scale, window_end)
        exists = conn.execute(
            "SELECT 1 FROM scale_scores WHERE user_id=? AND scale_id=? AND window_end=? LIMIT 1;",
            (user_id, scale.id, day.isoformat()),
        ).fetchone()
        if exists:
            continue

        baseline_before = baselines.get(scale.id)
        baseline_mean = baseline_before.mean if baseline_before else None
        baseline_std_value = baseline_std(baseline_before) if baseline_before else None
        personal_z = None
        if baseline_before and baseline_std_value and baseline_std_value >= 1e-6:
            personal_z = (score_res.normalized_score - baseline_before.mean) / baseline_std_value

        prev_score = last["normalized_score"] if last is not None else None
        delta_vs_prev = score_res.normalized_score - prev_score if prev_score is not None else None

        tier = confidence_tier(score_res.answered_count, score_res.items_required, scale.min_items_required)

        # --- Behavioural metadata SE adjustment (Claim Family 5) ---
        se_theta_val = score_res.se_theta
        se_theta_adj_val: Optional[float] = None
        uncertainty_mult_val: Optional[float] = None
        engagement_quality_val: Optional[str] = _session_engagement

        if metadata_enabled and se_theta_val is not None and session_id is not None:
            scale_item_ids = [si.item_id for si in scale.items]
            item_modifiers = _load_item_modifiers_for_scale(
                conn, session_id=session_id, scale_item_ids=scale_item_ids,
            )
            if item_modifiers:
                se_theta_adj_val = adjust_se_with_metadata(se_theta_val, item_modifiers)
                if se_theta_val > 0:
                    uncertainty_mult_val = round(se_theta_adj_val / se_theta_val, 4)

        score_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO scale_scores(
              score_id, user_id, scale_id, computed_at, window_start, window_end,
              raw_score, normalized_score, confidence_tier, items_answered_count, items_required,
              baseline_mean, baseline_std, personal_z, delta_vs_prev, risk_tier,
              se_theta, se_theta_adjusted, uncertainty_multiplier, engagement_quality
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                score_id,
                user_id,
                scale.id,
                now_iso(),
                start_day.isoformat(),
                day.isoformat(),
                float(score_res.raw_score),
                float(score_res.normalized_score),
                tier,
                int(score_res.answered_count),
                int(score_res.items_required),
                float(baseline_mean) if baseline_mean is not None else None,
                float(baseline_std_value) if baseline_std_value is not None else None,
                float(personal_z) if personal_z is not None else None,
                float(delta_vs_prev) if delta_vs_prev is not None else None,
                str(score_res.risk_tier) if score_res.risk_tier else None,
                float(se_theta_val) if se_theta_val is not None else None,
                float(se_theta_adj_val) if se_theta_adj_val is not None else None,
                float(uncertainty_mult_val) if uncertainty_mult_val is not None else None,
                engagement_quality_val,
            ),
        )

        # Update baseline using normalized score with per-scale alpha
        scale_alpha = float(getattr(scale, "ewma_alpha", 0.2) or 0.2)
        if baseline_before is None:
            baselines[scale.id] = BaselineState(mean=score_res.normalized_score, var=0.0, n=1)
        else:
            baselines[scale.id] = update_ewma_baseline(baseline_before, score_res.normalized_score, alpha=scale_alpha)

        conn.execute(
            """
            INSERT OR REPLACE INTO scale_baselines(user_id, scale_id, mean, var, n, updated_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                user_id,
                scale.id,
                float(baselines[scale.id].mean),
                float(baselines[scale.id].var),
                int(baselines[scale.id].n),
                now_iso(),
            ),
        )

        if personal_z is not None and abs(float(personal_z)) >= 1.5:
            _enqueue_follow_up_items_for_scale(
                conn,
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
            _resolve_follow_up_queue_for_scale(
                conn,
                user_id=user_id,
                scale_id=scale.id,
                resolution_day=day,
                resolved_reason="stabilized",
            )

        score_payload = _get_last_scale_score(conn, user_id=user_id, scale_id=scale.id) or {}
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
            # Concordance gating: flag scores that are eligible for paired measurement validation
            if site_config and site_config.concordance_mode:
                score_payload["concordance_eligible"] = True
        new_scores.append(score_payload)

    conn.commit()
    return new_scores


def _list_versions(registry_root: str) -> List[str]:
    from questions_agent_platform.pipeline.registry import list_versions

    return list_versions(registry_root)


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
