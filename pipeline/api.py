from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import date, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import parse_qs, urlparse

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.dashboard import render_index, render_user
from questions_agent_platform.pipeline.db import connect, ensure_user
from questions_agent_platform.pipeline.projection import build_questions_projection_payload
from questions_agent_platform.pipeline.registry import Registry, load_registry
from questions_agent_platform.pipeline.response_types import get_response_type
from questions_agent_platform.pipeline.service import (
    activate_registry_version,
    build_domain_promotion_readiness,
    build_tta_summary,
    compute_experiment_results,
    compute_user_fold,
    compute_user_scale_progress,
    ensure_registry_active,
    get_experiment,
    get_circle_snapshots,
    get_coherence_tier_contract,
    get_drift_routing_contract,
    get_drift_events,
    get_ews_features,
    get_or_create_daily_session,
    get_scale_history,
    get_state_snapshots,
    get_user_profile,
    list_experiments,
    list_safety_events,
    list_users,
    resolve_safety_event,
    suggest_emotion_deep_dive,
    submit_answers,
    submit_observations,
    take_next_extra_batch,
    upsert_experiment,
    upsert_user_profile,
    upsert_policy_outcome_update,
    upload_registry_bundle,
)
from questions_agent_platform.pipeline.time_utils import now_iso, parse_date


def run_server(cfg: QuestionsAgentConfig) -> None:
    server = ThreadingHTTPServer((cfg.host, int(cfg.port)), _make_handler(cfg))
    print(f"Questions Agent API listening on http://{cfg.host}:{cfg.port}")
    server.serve_forever()


def _make_handler(cfg: QuestionsAgentConfig):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            try:
                self._handle_get()
            except ValueError as e:
                self._json_error(HTTPStatus.BAD_REQUEST, str(e))
            except Exception as e:  # pragma: no cover
                self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._handle_post()
            except ValueError as e:
                self._json_error(HTTPStatus.BAD_REQUEST, str(e))
            except Exception as e:  # pragma: no cover
                self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

        def do_PATCH(self) -> None:  # noqa: N802
            try:
                self._handle_patch()
            except ValueError as e:
                self._json_error(HTTPStatus.BAD_REQUEST, str(e))
            except Exception as e:  # pragma: no cover
                self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

        def _handle_get(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path == "/health":
                self._json(HTTPStatus.OK, {"ok": True, "ts": now_iso()})
                return

            if path == "/dashboard":
                if not cfg.dashboard_enabled:
                    self._text(HTTPStatus.NOT_FOUND, "dashboard disabled")
                    return
                with connect(cfg.database_path) as conn:
                    users = list_users(conn)
                self._html(HTTPStatus.OK, render_index(users))
                return

            if path.startswith("/dashboard/users/"):
                if not cfg.dashboard_enabled:
                    self._text(HTTPStatus.NOT_FOUND, "dashboard disabled")
                    return
                user_id = path.split("/", 3)[3]
                with connect(cfg.database_path) as conn:
                    ensure_user(conn, user_id)
                    reg_v = ensure_registry_active(conn, cfg.registry_root)
                    registry = load_registry(cfg.registry_root, reg_v)
                    progress = compute_user_scale_progress(conn, registry=registry, user_id=user_id, day=date.today())
                    histories = {sid: get_scale_history(conn, user_id=user_id, scale_id=sid, limit=60) for sid in registry.scales.keys()}
                    latest_session = _get_latest_session(conn, user_id=user_id)
                self._html(
                    HTTPStatus.OK,
                    render_user(user_id=user_id, scale_progress=progress, scale_histories=histories, latest_session=latest_session),
                )
                return

            if path == "/v1/admin/registry/versions":
                with connect(cfg.database_path) as conn:
                    active = ensure_registry_active(conn, cfg.registry_root)
                    versions = _list_registry_versions(cfg.registry_root)
                self._json(HTTPStatus.OK, {"active_version": active, "versions": versions})
                return

            if path == "/v1/admin/domains/readiness":
                min_users = _get_query_int(qs.get("min_users"), default=100, min_value=1, max_value=100000)
                min_cycles = _get_query_int(qs.get("min_cycles"), default=4, min_value=1, max_value=1000)
                min_weeks = _get_query_int(qs.get("min_weeks"), default=8, min_value=1, max_value=10000)
                min_paired_ratio = _get_query_float(qs.get("min_paired_ratio"), default=0.80, min_value=0.0, max_value=1.0)
                with connect(cfg.database_path) as conn:
                    report = build_domain_promotion_readiness(
                        conn,
                        registry_root=cfg.registry_root,
                        min_users=min_users,
                        min_cycles=min_cycles,
                        min_weeks=min_weeks,
                        min_paired_ratio=min_paired_ratio,
                    )
                self._json(HTTPStatus.OK, report)
                return

            if path == "/v1/coherence/tier-contract":
                self._json(HTTPStatus.OK, get_coherence_tier_contract())
                return

            if path == "/v1/drift-routing-contract":
                self._json(
                    HTTPStatus.OK,
                    get_drift_routing_contract(drift_routing_table_path=cfg.drift_routing_table_path),
                )
                return

            if path.startswith("/v1/users/"):
                segments = path.strip("/").split("/")
                # v1/users/{user_id}/...
                if len(segments) < 3:
                    self._json_error(HTTPStatus.NOT_FOUND, "not found")
                    return
                user_id = segments[2]

                if len(segments) == 4 and segments[3] == "daily-questions":
                    day = _get_query_date(qs.get("date"))
                    with connect(cfg.database_path) as conn:
                        session = get_or_create_daily_session(
                            conn,
                            cfg=cfg,
                            registry_root=cfg.registry_root,
                            user_id=user_id,
                            day=day,
                        )
                        registry = load_registry(cfg.registry_root, session.registry_version)
                        progress = compute_user_scale_progress(conn, registry=registry, user_id=user_id, day=day)
                        questions = _build_questions_payload(
                            registry=registry,
                            item_ids=session.core_item_ids,
                            selection_explain=session.selection_explain,
                            scale_progress=progress,
                        )
                    extra_total = min(3, sum(len(b) for b in session.extra_batches))
                    used_extra = min(int(session.extra_batches_used), int(extra_total))
                    self._json(
                        HTTPStatus.OK,
                        {
                            "session_id": session.session_id,
                            "user_id": user_id,
                            "date": session.date,
                            "registry_version": session.registry_version,
                            "timeframe": session.timeframe,
                            "status": session.status,
                            "questions": questions,
                            "extra": {
                                "batch_size": 1,
                                "max_batches": 3,
                                "available_batches": max(0, int(extra_total - used_extra)),
                                "used_batches": used_extra,
                            },
                            "selection_explain": session.selection_explain,
                            "scale_progress": progress,
                        },
                    )
                    return

                if len(segments) == 4 and segments[3] == "profile":
                    with connect(cfg.database_path) as conn:
                        profile = get_user_profile(conn, user_id)
                    self._json(HTTPStatus.OK, {"user_id": user_id, "profile": profile})
                    return

                if len(segments) == 4 and segments[3] == "progress":
                    day = _get_query_date(qs.get("date"))
                    window_days = _get_query_window_days(qs.get("window"), default=30, min_days=1, max_days=365)
                    start = (day - timedelta(days=window_days - 1)).isoformat()
                    with connect(cfg.database_path) as conn:
                        reg_v = ensure_registry_active(conn, cfg.registry_root)
                        registry = load_registry(cfg.registry_root, reg_v)
                        progress = compute_user_scale_progress(conn, registry=registry, user_id=user_id, day=day)
                        session_rows = conn.execute(
                            """
                            SELECT date, status
                            FROM daily_sessions
                            WHERE user_id=? AND date>=? AND date<=?;
                            """,
                            (user_id, start, day.isoformat()),
                        ).fetchall()
                        answer_count = int(
                            conn.execute(
                                """
                                SELECT COUNT(*)
                                FROM answer_events ae
                                JOIN daily_sessions ds ON ds.session_id = ae.session_id
                                WHERE ae.user_id=? AND ds.date>=? AND ds.date<=?;
                                """,
                                (user_id, start, day.isoformat()),
                            ).fetchone()[0]
                            or 0
                        )
                        answered_days = int(
                            conn.execute(
                                """
                                SELECT COUNT(DISTINCT ds.date)
                                FROM answer_events ae
                                JOIN daily_sessions ds ON ds.session_id = ae.session_id
                                WHERE ae.user_id=? AND ds.date>=? AND ds.date<=?;
                                """,
                                (user_id, start, day.isoformat()),
                            ).fetchone()[0]
                            or 0
                        )
                    active_days = len(session_rows)
                    completed_days = sum(1 for row in session_rows if str(row["status"] or "") == "completed")
                    started_days = sum(
                        1
                        for row in session_rows
                        if str(row["status"] or "") in {"started", "completed"}
                    )
                    completion_rate = float(completed_days / active_days) if active_days > 0 else 0.0
                    self._json(
                        HTTPStatus.OK,
                        {
                            "user_id": user_id,
                            "date": day.isoformat(),
                            "window_days": window_days,
                            "coverage": {
                                "active_days": active_days,
                                "answered_days": answered_days,
                                "started_days": started_days,
                                "completed_days": completed_days,
                            },
                            "completion_rate": completion_rate,
                            "answers_count": answer_count,
                            "scales": progress,
                        },
                    )
                    return

                if len(segments) == 4 and segments[3] == "scales":
                    day = _get_query_date(qs.get("date"))
                    with connect(cfg.database_path) as conn:
                        reg_v = ensure_registry_active(conn, cfg.registry_root)
                        registry = load_registry(cfg.registry_root, reg_v)
                        progress = compute_user_scale_progress(conn, registry=registry, user_id=user_id, day=day)
                    self._json(HTTPStatus.OK, {"user_id": user_id, "date": day.isoformat(), "scales": progress})
                    return

                if len(segments) == 6 and segments[3] == "scales" and segments[5] == "history":
                    scale_id = segments[4]
                    limit = int((qs.get("limit") or ["200"])[0])
                    with connect(cfg.database_path) as conn:
                        history = get_scale_history(conn, user_id=user_id, scale_id=scale_id, limit=limit)
                    self._json(HTTPStatus.OK, {"user_id": user_id, "scale_id": scale_id, "history": history})
                    return

                if len(segments) == 5 and segments[3] == "projection" and segments[4] == "questions":
                    day = _get_query_date(qs.get("date"))
                    with connect(cfg.database_path) as conn:
                        reg_v = ensure_registry_active(conn, cfg.registry_root)
                        payload = build_questions_projection_payload(
                            conn, user_id=user_id, registry_version=reg_v, day=day
                        )
                    self._json(HTTPStatus.OK, payload)
                    return

                if len(segments) == 5 and segments[3] == "ani" and segments[4] == "summary":
                    day = _get_query_date(qs.get("date"))
                    with connect(cfg.database_path) as conn:
                        summary = build_tta_summary(
                            conn,
                            registry_root=cfg.registry_root,
                            user_id=user_id,
                            day=day,
                        )
                    self._json(HTTPStatus.OK, summary)
                    return

                if len(segments) == 4 and segments[3] == "safety-events":
                    day = _get_query_date(qs.get("date"))
                    lookback_days = _get_query_int(qs.get("days"), default=30, min_value=1, max_value=3650)
                    status = (qs.get("status") or [None])[0]
                    with connect(cfg.database_path) as conn:
                        events = list_safety_events(
                            conn,
                            user_id=user_id,
                            day=day,
                            lookback_days=lookback_days,
                            status=str(status) if status else None,
                        )
                    self._json(
                        HTTPStatus.OK,
                        {
                            "user_id": user_id,
                            "date": day.isoformat(),
                            "lookback_days": lookback_days,
                            "status": str(status) if status else None,
                            "events": events,
                        },
                    )
                    return

                if len(segments) == 4 and segments[3] == "emotion" and (qs.get("deep_dive") or [None])[0] == "1":
                    # Backward-compatible alias: /v1/users/{id}/emotion?deep_dive=1
                    day = _get_query_date(qs.get("date"))
                    with connect(cfg.database_path) as conn:
                        payload = suggest_emotion_deep_dive(
                            conn,
                            cfg=cfg,
                            registry_root=cfg.registry_root,
                            user_id=user_id,
                            day=day,
                        )
                    self._json(HTTPStatus.OK, payload)
                    return

                if len(segments) == 5 and segments[3] == "emotion" and segments[4] == "deep_dive":
                    day = _get_query_date(qs.get("date"))
                    with connect(cfg.database_path) as conn:
                        payload = suggest_emotion_deep_dive(
                            conn,
                            cfg=cfg,
                            registry_root=cfg.registry_root,
                            user_id=user_id,
                            day=day,
                        )
                    self._json(HTTPStatus.OK, payload)
                    return

                if len(segments) == 4 and segments[3] == "experiments":
                    status = (qs.get("status") or [None])[0]
                    with connect(cfg.database_path) as conn:
                        rows = list_experiments(conn, user_id=user_id, status=str(status) if status else None)
                    self._json(
                        HTTPStatus.OK,
                        {
                            "user_id": user_id,
                            "status": str(status) if status else None,
                            "experiments": rows,
                        },
                    )
                    return

                if len(segments) == 5 and segments[3] == "experiments":
                    experiment_id = segments[4]
                    with connect(cfg.database_path) as conn:
                        try:
                            row = get_experiment(conn, user_id=user_id, experiment_id=experiment_id)
                        except ValueError as exc:
                            self._json_error(HTTPStatus.NOT_FOUND, str(exc))
                            return
                    self._json(HTTPStatus.OK, row)
                    return

                if len(segments) == 6 and segments[3] == "experiments" and segments[5] == "results":
                    experiment_id = segments[4]
                    day = _get_query_date(qs.get("date"))
                    with connect(cfg.database_path) as conn:
                        try:
                            row = compute_experiment_results(
                                conn,
                                user_id=user_id,
                                experiment_id=experiment_id,
                                day=day,
                            )
                        except ValueError as exc:
                            self._json_error(HTTPStatus.NOT_FOUND, str(exc))
                            return
                    self._json(HTTPStatus.OK, row)
                    return

                if len(segments) == 4 and segments[3] == "state":
                    day = _get_query_date(qs.get("date"))
                    try:
                        window_days = _get_query_window_days(qs.get("window"), default=30, min_days=1, max_days=365)
                    except ValueError as exc:
                        self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
                        return
                    with connect(cfg.database_path) as conn:
                        snapshots = get_state_snapshots(
                            conn,
                            user_id=user_id,
                            day=day,
                            window_days=window_days,
                        )
                    self._json(
                        HTTPStatus.OK,
                        {
                            "user_id": user_id,
                            "date": day.isoformat(),
                            "window_days": window_days,
                            "snapshots": snapshots,
                        },
                    )
                    return

                if len(segments) == 4 and segments[3] == "circle":
                    day = _get_query_date(qs.get("date"))
                    try:
                        window_days = _get_query_window_days(qs.get("window"), default=30, min_days=1, max_days=365)
                    except ValueError as exc:
                        self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
                        return
                    with connect(cfg.database_path) as conn:
                        snapshots = get_circle_snapshots(
                            conn,
                            user_id=user_id,
                            day=day,
                            window_days=window_days,
                        )
                    self._json(
                        HTTPStatus.OK,
                        {
                            "user_id": user_id,
                            "date": day.isoformat(),
                            "window_days": window_days,
                            "snapshots": snapshots,
                        },
                    )
                    return

                if len(segments) == 4 and segments[3] == "fold":
                    day = _get_query_date(qs.get("date"))
                    try:
                        window_days = _get_query_window_days(qs.get("window"), default=30, min_days=1, max_days=365)
                    except ValueError as exc:
                        self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
                        return
                    include_poe = (qs.get("poe") or [None])[0] == "1"
                    with connect(cfg.database_path) as conn:
                        ensure_user(conn, user_id)
                        fold_payload = compute_user_fold(
                            conn,
                            user_id=user_id,
                            day=day,
                            window_days=window_days,
                        )
                    result: Dict[str, Any] = {
                        "user_id": user_id,
                        "fold": fold_payload,
                    }
                    if include_poe:
                        # Extract PoE-compatible minimal payload for AniFold fusion
                        result["poe"] = _extract_poe_from_fold(fold_payload)
                    self._json(HTTPStatus.OK, result)
                    return

                if len(segments) == 4 and segments[3] == "ews":
                    day = _get_query_date(qs.get("date"))
                    try:
                        window_days = _get_query_window_days(qs.get("window"), default=30, min_days=1, max_days=365)
                    except ValueError as exc:
                        self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
                        return
                    with connect(cfg.database_path) as conn:
                        rows = get_ews_features(
                            conn,
                            user_id=user_id,
                            day=day,
                            window_days=window_days,
                        )
                    self._json(
                        HTTPStatus.OK,
                        {
                            "user_id": user_id,
                            "date": day.isoformat(),
                            "window_days": window_days,
                            "features": rows,
                        },
                    )
                    return

                if len(segments) == 4 and segments[3] == "drift-events":
                    day = _get_query_date(qs.get("date"))
                    try:
                        window_days = _get_query_window_days(qs.get("window"), default=30, min_days=1, max_days=365)
                    except ValueError as exc:
                        self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
                        return
                    with connect(cfg.database_path) as conn:
                        rows = get_drift_events(
                            conn,
                            user_id=user_id,
                            day=day,
                            window_days=window_days,
                        )
                    self._json(
                        HTTPStatus.OK,
                        {
                            "user_id": user_id,
                            "date": day.isoformat(),
                            "window_days": window_days,
                            "events": rows,
                        },
                    )
                    return

            self._json_error(HTTPStatus.NOT_FOUND, "not found")

        def _handle_post(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/v1/admin/registry/upload":
                body = self._read_json_body()
                with connect(cfg.database_path) as conn:
                    version = upload_registry_bundle(conn, cfg.registry_root, body)
                self._json(HTTPStatus.OK, {"uploaded_version": version})
                return

            if path.startswith("/v1/admin/registry/activate/"):
                version = path.split("/", 5)[5]
                with connect(cfg.database_path) as conn:
                    activate_registry_version(conn, cfg.registry_root, version)
                self._json(HTTPStatus.OK, {"active_version": version})
                return

            if path.startswith("/v1/users/"):
                segments = path.strip("/").split("/")
                if len(segments) < 3:
                    self._json_error(HTTPStatus.NOT_FOUND, "not found")
                    return
                user_id = segments[2]

                if len(segments) == 5 and segments[3] == "daily-questions" and segments[4] == "select":
                    body = self._read_json_body()
                    try:
                        day = parse_date(str(body.get("date") or "")) if body.get("date") else date.today()
                    except Exception:
                        self._json_error(HTTPStatus.BAD_REQUEST, "invalid date; expected YYYY-MM-DD")
                        return
                    selection_mode = str(body.get("selection_mode") or cfg.policy_default_mode or "deterministic")
                    identity_mask_id = body.get("identity_mask_id")
                    include_explanations = bool(body.get("include_explanations", False))
                    context = body.get("context") if isinstance(body.get("context"), dict) else None
                    allow_context_batches = body.get("allow_context_batches")
                    if allow_context_batches is not None:
                        context = dict(context or {})
                        context["allow_context_batches"] = bool(allow_context_batches)
                    k_core_raw = body.get("k_core")
                    k_core = None
                    if k_core_raw is not None:
                        try:
                            k_core = int(k_core_raw)
                        except Exception:
                            self._json_error(HTTPStatus.BAD_REQUEST, "k_core must be an integer")
                            return
                        if k_core <= 0:
                            self._json_error(HTTPStatus.BAD_REQUEST, "k_core must be > 0")
                            return
                    cfg_req = replace(cfg, core_questions_per_day=k_core) if k_core is not None else cfg

                    with connect(cfg.database_path) as conn:
                        session = get_or_create_daily_session(
                            conn,
                            cfg=cfg_req,
                            registry_root=cfg.registry_root,
                            user_id=user_id,
                            day=day,
                            selection_mode=selection_mode,
                            policy_context=context,
                            identity_mask_id=str(identity_mask_id) if identity_mask_id else None,
                        )
                        registry = load_registry(cfg.registry_root, session.registry_version)
                        progress = compute_user_scale_progress(conn, registry=registry, user_id=user_id, day=day)
                        questions = _build_questions_payload(
                            registry=registry,
                            item_ids=session.core_item_ids,
                            selection_explain=session.selection_explain,
                            scale_progress=progress,
                        )

                        selection_explain = session.selection_explain or {}
                        if include_explanations and session.policy_decision_id:
                            row = conn.execute(
                                "SELECT explanations_json FROM policy_decisions WHERE decision_id=?;",
                                (session.policy_decision_id,),
                            ).fetchone()
                            if row and row["explanations_json"]:
                                try:
                                    selection_explain.setdefault("policy", {})["explanations"] = json.loads(
                                        str(row["explanations_json"])
                                    )
                                except Exception:
                                    selection_explain.setdefault("policy", {})["explanations"] = []
                    extra_total = min(3, sum(len(b) for b in session.extra_batches))
                    used_extra = min(int(session.extra_batches_used), int(extra_total))

                    self._json(
                        HTTPStatus.OK,
                        {
                            "session_id": session.session_id,
                            "user_id": user_id,
                            "date": session.date,
                            "registry_version": session.registry_version,
                            "timeframe": session.timeframe,
                            "status": session.status,
                            "questions": questions,
                            "extra": {
                                "batch_size": 1,
                                "max_batches": 3,
                                "available_batches": max(0, int(extra_total - used_extra)),
                                "used_batches": used_extra,
                            },
                            "selection_explain": selection_explain,
                            "scale_progress": progress,
                        },
                    )
                    return

                if len(segments) == 5 and segments[3] == "policy" and segments[4] == "outcomes":
                    body = self._read_json_body()
                    with connect(cfg.database_path) as conn:
                        result = upsert_policy_outcome_update(conn, cfg=cfg, user_id=user_id, body=body)
                    self._json(HTTPStatus.OK, result)
                    return

                if len(segments) == 6 and segments[3] == "safety-events" and segments[5] == "resolve":
                    body = self._read_json_body()
                    resolved_by = str(body.get("resolved_by") or "").strip()
                    resolved_reason = str(body.get("resolved_reason") or "").strip()
                    if not resolved_by:
                        self._json_error(HTTPStatus.BAD_REQUEST, "resolved_by is required")
                        return
                    if not resolved_reason:
                        self._json_error(HTTPStatus.BAD_REQUEST, "resolved_reason is required")
                        return
                    event_id = segments[4]
                    with connect(cfg.database_path) as conn:
                        try:
                            result = resolve_safety_event(
                                conn,
                                user_id=user_id,
                                event_id=event_id,
                                resolved_by=resolved_by,
                                resolved_reason=resolved_reason,
                            )
                        except ValueError as exc:
                            code = HTTPStatus.NOT_FOUND if "unknown safety event" in str(exc).lower() else HTTPStatus.BAD_REQUEST
                            self._json_error(code, str(exc))
                            return
                    self._json(HTTPStatus.OK, result)
                    return

                if len(segments) == 4 and segments[3] == "experiments":
                    body = self._read_json_body()
                    with connect(cfg.database_path) as conn:
                        result = upsert_experiment(conn, user_id=user_id, body=body)
                    self._json(HTTPStatus.OK, result)
                    return

                if len(segments) == 4 and segments[3] == "answers":
                    body = self._read_json_body()
                    session_id = str(body.get("session_id") or "")
                    answers = body.get("answers") or []
                    if not session_id:
                        self._json_error(HTTPStatus.BAD_REQUEST, "missing session_id")
                        return
                    if not isinstance(answers, list):
                        self._json_error(HTTPStatus.BAD_REQUEST, "answers must be a list")
                        return
                    with connect(cfg.database_path) as conn:
                        result = submit_answers(
                            conn,
                            registry_root=cfg.registry_root,
                            user_id=user_id,
                            session_id=session_id,
                            answers=answers,
                            drift_routing_table_path=cfg.drift_routing_table_path,
                        )
                    self._json(HTTPStatus.OK, result)
                    return

                if len(segments) == 4 and segments[3] == "request-more-context":
                    body = self._read_json_body()
                    session_id = str(body.get("session_id") or "")
                    if not session_id:
                        self._json_error(HTTPStatus.BAD_REQUEST, "missing session_id")
                        return
                    with connect(cfg.database_path) as conn:
                        batch = take_next_extra_batch(conn, session_id)
                        if batch is None:
                            self._json(HTTPStatus.OK, {"batch": None, "message": "no more batches"})
                            return
                        # Use registry from that session version
                        row = conn.execute("SELECT registry_version FROM daily_sessions WHERE session_id=?;", (session_id,)).fetchone()
                        reg_v = str(row["registry_version"]) if row else ensure_registry_active(conn, cfg.registry_root)
                        registry = load_registry(cfg.registry_root, reg_v)
                        questions = _build_questions_payload(
                            registry=registry,
                            item_ids=batch,
                            selection_explain={},
                            scale_progress={},
                        )
                    self._json(HTTPStatus.OK, {"batch": questions})
                    return

                if len(segments) == 4 and segments[3] == "observations":
                    body = self._read_json_body()
                    observations = body.get("observations") or []
                    if not isinstance(observations, list):
                        self._json_error(HTTPStatus.BAD_REQUEST, "observations must be a list")
                        return
                    with connect(cfg.database_path) as conn:
                        result = submit_observations(conn, user_id=user_id, observations=observations)
                    self._json(HTTPStatus.OK, result)
                    return

            self._json_error(HTTPStatus.NOT_FOUND, "not found")

        def _handle_patch(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path.startswith("/v1/users/"):
                segments = path.strip("/").split("/")
                if len(segments) < 3:
                    self._json_error(HTTPStatus.NOT_FOUND, "not found")
                    return
                user_id = segments[2]

                if len(segments) == 4 and segments[3] == "profile":
                    body = self._read_json_body()
                    with connect(cfg.database_path) as conn:
                        reg_v = ensure_registry_active(conn, cfg.registry_root)
                        registry = load_registry(cfg.registry_root, reg_v)
                        profile = upsert_user_profile(
                            conn,
                            user_id=user_id,
                            patch=body,
                            registry=registry,
                            cfg=cfg,
                        )
                    self._json(HTTPStatus.OK, {"user_id": user_id, "profile": profile})
                    return

            self._json_error(HTTPStatus.NOT_FOUND, "not found")

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                raise ValueError("Invalid JSON body")
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def _json(self, status: int, obj: Any) -> None:
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _json_error(self, status: int, message: str) -> None:
            self._json(int(status), {"error": message})

        def _text(self, status: int, text: str) -> None:
            payload = text.encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _html(self, status: int, html_text: str) -> None:
            payload = html_text.encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # Quiet by default; uncomment for debugging.
            return

    return Handler


def _item_to_question(
    registry: Registry,
    item_id: str,
    *,
    progress_hint: Optional[str] = None,
    feeds_scales: Optional[List[str]] = None,
) -> Dict[str, Any]:
    item = registry.items[item_id]
    rt = get_response_type(item.response_type)
    out = {
        "item_id": item.id,
        "text": item.text,
        "response_type": item.response_type,
        "options": [{"value": v, "label": label} for v, label in rt.options],
        "tags": list(item.tags),
        "sensitivity": item.sensitivity,
    }
    if progress_hint:
        out["progress_hint"] = str(progress_hint)
    if feeds_scales:
        out["feeds_scales"] = [str(x) for x in feeds_scales]
    return out


def _get_query_date(values: Optional[List[str]]) -> date:
    if not values:
        return date.today()
    return parse_date(values[0])


def _get_query_window_days(
    values: Optional[List[str]],
    *,
    default: int,
    min_days: int,
    max_days: int,
) -> int:
    if not values:
        return int(default)
    raw = str(values[0]).strip()
    if raw.endswith("d"):
        raw = raw[:-1]
    try:
        days = int(raw)
    except Exception as exc:
        raise ValueError("window must be an integer number of days") from exc
    return max(int(min_days), min(int(max_days), days))


def _get_query_int(
    values: Optional[List[str]],
    *,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    if not values:
        return int(default)
    raw = str(values[0]).strip()
    try:
        val = int(raw)
    except Exception as exc:
        raise ValueError("query value must be an integer") from exc
    return max(int(min_value), min(int(max_value), val))


def _get_query_float(
    values: Optional[List[str]],
    *,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    if not values:
        return float(default)
    raw = str(values[0]).strip()
    try:
        val = float(raw)
    except Exception as exc:
        raise ValueError("query value must be a float") from exc
    return max(float(min_value), min(float(max_value), val))


def _list_registry_versions(registry_root: str) -> List[str]:
    from questions_agent_platform.pipeline.registry import list_versions

    return list_versions(registry_root)


def _get_latest_session(conn: sqlite3.Connection, *, user_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM daily_sessions WHERE user_id=? ORDER BY date DESC LIMIT 1;",
        (user_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "session_id": str(row["session_id"]),
        "date": str(row["date"]),
        "registry_version": str(row["registry_version"]),
        "selection_explain": json.loads(row["selection_explain_json"]),
    }


def _build_questions_payload(
    *,
    registry: Registry,
    item_ids: Sequence[str],
    selection_explain: Optional[Dict[str, Any]],
    scale_progress: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    scale_names_by_item = _scale_names_by_item(registry)
    hints = _progress_hints_by_item(
        registry=registry,
        item_ids=item_ids,
        selection_explain=selection_explain or {},
        scale_progress=scale_progress or {},
        scale_names_by_item=scale_names_by_item,
    )
    out = []
    for item_id in item_ids:
        out.append(
            _item_to_question(
                registry,
                item_id,
                progress_hint=hints.get(item_id),
                feeds_scales=scale_names_by_item.get(item_id, []),
            )
        )
    return out


def _scale_names_by_item(registry: Registry) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for scale in registry.scales.values():
        for si in scale.items:
            out.setdefault(si.item_id, [])
            if scale.name not in out[si.item_id]:
                out[si.item_id].append(scale.name)
    return out


def _progress_hints_by_item(
    *,
    registry: Registry,
    item_ids: Sequence[str],
    selection_explain: Dict[str, Any],
    scale_progress: Dict[str, Any],
    scale_names_by_item: Dict[str, List[str]],
) -> Dict[str, str]:
    primary = selection_explain.get("primary_scale_by_item_id")
    primary = primary if isinstance(primary, dict) else {}
    out: Dict[str, str] = {}
    for item_id in item_ids:
        primary_scale_id = str(primary.get(item_id)) if item_id in primary else None
        if primary_scale_id and primary_scale_id in scale_progress:
            p = scale_progress[primary_scale_id] or {}
            hint = p.get("unlock_hint")
            if hint:
                out[str(item_id)] = str(hint)
                continue
            name = str(p.get("name") or primary_scale_id)
            out[str(item_id)] = f"Feeds {name}"
            continue
        names = list(scale_names_by_item.get(item_id, []))
        if names:
            out[str(item_id)] = f"Feeds {', '.join(names[:2])}"
    return out


def _extract_poe_from_fold(fold: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the minimal PoE-compatible payload from an Identity Mask fold.

    Product-of-Experts fusion requires μ (mean) and precision = 1/σ².
    This avoids re-computing the fold just to get the PoE view.
    """
    mu = fold.get("mu", {})
    sigma_diag = fold.get("sigma_diag", {})
    precision = {}
    for dim, sigma in sigma_diag.items():
        s = max(0.001, float(sigma))
        precision[dim] = round(1.0 / (s * s), 6)

    circle = fold.get("circle", {})
    coverage = fold.get("coverage", {})
    return {
        "modality": "questionnaire",
        "mu": mu,
        "precision": precision,
        "coherence": circle.get("coherence", 0.0),
        "coverage_ratio": coverage.get("coverage_ratio", 0.0),
        "day": fold.get("day", ""),
    }
