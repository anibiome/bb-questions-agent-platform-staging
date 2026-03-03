from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import func, select

from questions_agent_platform.pipeline.dashboard import render_index, render_user
from questions_agent_platform.pipeline.registry import load_registry
from questions_agent_platform.pipeline.response_types import get_response_type
from questions_agent_platform.pipeline.time_utils import now_iso, parse_date
from questions_agent_platform.prod.auth import require_api_key
from questions_agent_platform.prod.db import make_engine, make_session_factory, session_scope
from questions_agent_platform.prod.models import AnswerEvent, DailySession
from questions_agent_platform.prod.projection_pg import build_projection_payload_pg
from questions_agent_platform.prod.schemas import (
    AnamnesisEpisodeListOut,
    AnamnesisEpisodeOut,
    CalibrationListOut,
    CardioRiskHistoryOut,
    CoherenceHistoryOut,
    DailyQuestionsOut,
    DailyQuestionsSelectIn,
    ExperimentUpsertIn,
    PolicyOutcomeUpdateIn,
    RequestMoreContextIn,
    SafetyResolveIn,
    SessionUncertaintyProfileOut,
    SiteConfigListOut,
    SiteConfigOut,
    SubmitAnswersIn,
    SubmitAnswersOut,
    SubmitObservationsIn,
    SubmitObservationsOut,
    RegistryUploadIn,
    UserProfilePatchIn,
)
from questions_agent_platform.prod.service_pg import (
    SnapshotWriteError,
    activate_registry_version,
    build_clinical_audit_export,
    build_domain_promotion_readiness,
    build_slo_report,
    build_tta_summary,
    compute_and_store_cardio_risk,
    compute_experiment_results,
    compute_user_scale_progress,
    evaluate_policy_live_rollback_guard_pg,
    ensure_registry_active,
    format_score_for_user,
    get_cardio_risk_history,
    get_coherence_history,
    get_experiment,
    get_circle_snapshots,
    get_coherence_tier_contract,
    get_drift_routing_contract,
    get_drift_events,
    get_ews_features,
    get_or_create_daily_session,
    get_session_coherence,
    get_user_calibrations,
    get_user_profile,
    get_scale_history,
    get_state_snapshots,
    list_experiments,
    list_safety_events,
    list_site_configs,
    list_users,
    record_operational_metric,
    resolve_safety_event,
    resolve_site_config,
    run_phi_retention,
    suggest_emotion_deep_dive,
    upsert_calibration,
    upsert_experiment,
    upsert_user_profile,
    upsert_policy_outcome_update,
    get_anamnesis_episodes,
    get_anamnesis_episode,
    get_follow_up_queue,
    get_session_engagement_profile,
    resolve_anamnesis_episode,
    submit_answers,
    submit_observations,
    take_next_extra_batch,
    upload_registry_bundle,
)
from questions_agent_platform.prod.settings import load_settings

import logging
import traceback

logger = logging.getLogger("questions_agent.api")


def _configure_logging(level: str = "INFO") -> None:
    """Configure structured logging for production."""
    fmt = (
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt, force=True)
    # Silence noisy libraries
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


settings = load_settings()
_configure_logging(settings.log_level)
engine = make_engine(settings.database_url)
SessionLocal = make_session_factory(engine)


class ErrorMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions → structured JSON error + log."""

    async def dispatch(self, request: Request, call_next):
        req_id = str(request.headers.get("x-request-id") or "").strip() or f"req_{uuid4().hex[:20]}"
        request.state.request_id = req_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
            elapsed = (time.perf_counter() - started) * 1000.0
            if response.status_code >= 400:
                logger.warning(
                    "request_id=%s req=%s %s status=%d elapsed_ms=%.1f",
                    req_id, request.method, request.url.path, response.status_code, elapsed,
                )
            else:
                logger.debug(
                    "request_id=%s req=%s %s status=%d elapsed_ms=%.1f",
                    req_id, request.method, request.url.path, response.status_code, elapsed,
                )
            response.headers["x-request-id"] = req_id
            return response
        except HTTPException:
            raise  # let FastAPI handle these
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            logger.error(
                "request_id=%s req=%s %s UNHANDLED error=%s elapsed_ms=%.1f\n%s",
                req_id, request.method, request.url.path, str(exc)[:300], elapsed,
                traceback.format_exc(),
            )
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )
            response.headers["x-request-id"] = req_id
            return response


@asynccontextmanager
async def _lifespan(_: FastAPI):
    from questions_agent_platform.prod.migrations import ensure_schema_ready

    logger.info("Starting Questions Agent API — migration mode=%s", settings.db_migration_mode)
    ensure_schema_ready(
        database_url=settings.database_url,
        mode=settings.db_migration_mode,
        baseline_existing=settings.db_migration_baseline_existing,
    )
    logger.info("Questions Agent API ready — schema migration complete")
    yield
    logger.info("Questions Agent API shutting down")


app = FastAPI(title="Questions Agent (Prod)", version="1.0", lifespan=_lifespan)
app.add_middleware(ErrorMiddleware)

# --- CORS ---
import os as _os
_cors_origins = [
    o.strip()
    for o in _os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
if _cors_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["X-API-Key"],
        allow_credentials=False,
        max_age=600,
    )


def _auth(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    require_api_key(x_api_key, settings.api_key, authorization=authorization)


def _model_dump(model: Any, **kwargs: Any) -> Dict[str, Any]:
    """Serialize a Pydantic model to dict. Requires Pydantic v2."""
    return model.model_dump(**kwargs)

def _parse_date_query_or_400(value: Optional[str], *, field_name: str = "date") -> date:
    if value is None or str(value).strip() == "":
        return date.today()
    try:
        return parse_date(str(value))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid {field_name}; expected YYYY-MM-DD") from exc



@app.get("/health")
def health():
    from sqlalchemy import text

    try:
        with session_scope(SessionLocal) as session:
            session.execute(text("SELECT 1"))
        return {"ok": True, "ts": now_iso(), "db": "ok"}
    except Exception:
        logger.exception("Health check DB probe failed")
        return JSONResponse(status_code=503, content={"ok": False, "ts": now_iso(), "db": "degraded"})


@app.get("/v1/coherence/tier-contract")
def coherence_tier_contract_endpoint(_: None = Depends(_auth)):
    return get_coherence_tier_contract()


@app.get("/v1/drift-routing-contract")
def drift_routing_contract_endpoint(_: None = Depends(_auth)):
    return get_drift_routing_contract(drift_routing_table_path=settings.drift_routing_table_path or None)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(_: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        users = list_users(session)
    return render_index(users)


@app.get("/dashboard/users/{user_id}", response_class=HTMLResponse)
def dashboard_user(user_id: str, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        reg_v = ensure_registry_active(session, settings.registry_root)
        registry = load_registry(settings.registry_root, reg_v)
        progress = compute_user_scale_progress(session, registry=registry, user_id=user_id, day=date.today())
        histories = {sid: get_scale_history(session, user_id=user_id, scale_id=sid, limit=60) for sid in registry.scales.keys()}
        latest_session = _get_latest_session(session, user_id=user_id)
    return render_user(user_id=user_id, scale_progress=progress, scale_histories=histories, latest_session=latest_session)


@app.get("/v1/admin/registry/versions")
def list_registry_versions(_: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        active = ensure_registry_active(session, settings.registry_root)
        versions = _list_versions(settings.registry_root)
    return {"active_version": active, "versions": versions}


@app.post("/v1/admin/registry/upload")
def upload_registry(body: RegistryUploadIn, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        version = upload_registry_bundle(session, settings.registry_root, _model_dump(body))
    return {"uploaded_version": version}


@app.post("/v1/admin/registry/activate/{version}")
def activate_registry(version: str, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        activate_registry_version(session, settings.registry_root, version)
    return {"active_version": version}


@app.get("/v1/admin/policy/versions")
def list_policy_versions(_: None = Depends(_auth)):
    from questions_agent_platform.policy.registry import get_active_policy_version, list_policy_versions

    active = get_active_policy_version(settings.policy_root) or settings.policy_default_version
    versions = list_policy_versions(settings.policy_root)
    return {"active_version": active, "versions": versions}


@app.post("/v1/admin/policy/activate/{version}")
def activate_policy(version: str, _: None = Depends(_auth)):
    from questions_agent_platform.policy.registry import list_policy_versions, set_active_policy_version

    versions = set(list_policy_versions(settings.policy_root))
    if version not in versions:
        raise HTTPException(status_code=400, detail="Unknown policy version")
    try:
        set_active_policy_version(settings.policy_root, version)
    except Exception as e:
        logger.error("Failed to activate policy version %s: %s", version, e)
        raise HTTPException(status_code=500, detail="Failed to activate policy version")
    return {"active_version": version}


@app.get("/v1/admin/policy/metrics")
def policy_metrics(days: int = 30, _: None = Depends(_auth)):
    from questions_agent_platform.prod.models import PolicyDecision as PolicyDecisionRow
    from questions_agent_platform.prod.models import PolicyOutcome as PolicyOutcomeRow

    lookback_days = max(1, min(3650, int(days)))
    cutoff = date.today() - timedelta(days=lookback_days - 1)

    with session_scope(SessionLocal) as session:
        decisions = (
            session.execute(select(PolicyDecisionRow).where(PolicyDecisionRow.date >= cutoff)).scalars().all()
        )
        outcomes = (
            session.execute(select(PolicyOutcomeRow).where(PolicyOutcomeRow.date >= cutoff)).scalars().all()
        )
        rollback_guard = evaluate_policy_live_rollback_guard_pg(
            session,
            day=date.today(),
            window_days=settings.policy_rollback_window_days,
            min_outcomes=settings.policy_rollback_min_outcomes,
            max_safety_violations=settings.policy_rollback_max_safety_violations,
            min_completion_drop=settings.policy_rollback_min_completion_drop,
            burden_increase_ratio=settings.policy_rollback_burden_increase_ratio,
        )

    def mean(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return float(sum(float(v) for v in vals) / max(1, len(vals)))

    def median(vals):
        vals = sorted([float(v) for v in vals if v is not None])
        if not vals:
            return None
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            return float(vals[mid])
        return float((vals[mid - 1] + vals[mid]) / 2.0)

    by_policy = {}
    for d in decisions:
        key = str(d.policy_version)
        by_policy.setdefault(key, 0)
        by_policy[key] += 1

    fallback = sum(1 for d in decisions if str(d.mode) == "safe_fallback")
    completion_rates = [o.completion_rate for o in outcomes]
    burdens = [o.response_time_ms_median for o in outcomes]
    unc_red = []
    for o in outcomes:
        if o.uncertainty_before_mean is None or o.uncertainty_after_mean is None:
            continue
        unc_red.append(float(o.uncertainty_before_mean) - float(o.uncertainty_after_mean))

    return {
        "window": {"days": lookback_days, "cutoff_date": cutoff.isoformat()},
        "decisions": {
            "count": len(decisions),
            "by_policy_version": by_policy,
            "safe_fallback_count": fallback,
            "safe_fallback_rate": (float(fallback) / len(decisions)) if decisions else None,
        },
        "outcomes": {
            "count": len(outcomes),
            "completion_rate_mean": mean(completion_rates),
            "burden_ms_median": median(burdens),
            "uncertainty_reduction_mean": mean(unc_red),
        },
        "rollback_guard": rollback_guard,
        "notes": [
            "Monitor constraint violations (should be 0), safe_fallback rate, adherence, burden, and uncertainty reduction.",
            "Runtime auto-rollback guard is evaluated on each policy_live selection.",
            "Manual rollback remains available by switching active policy version (or pinning via POLICY_DEFAULT_VERSION).",
        ],
    }


@app.get("/v1/admin/slo")
def slo_status(days: Optional[int] = None, _: None = Depends(_auth)):
    lookback_days = settings.slo_window_days if days is None else max(1, min(3650, int(days)))
    with session_scope(SessionLocal) as session:
        report = build_slo_report(
            session,
            day=date.today(),
            window_days=lookback_days,
            safe_fallback_rate_max=settings.slo_safe_fallback_rate_max,
            answer_ingestion_p95_ms_max=settings.slo_answer_ingestion_p95_ms_max,
            snapshot_write_failure_rate_max=settings.slo_snapshot_write_failure_rate_max,
        )
    return report


@app.post("/v1/admin/privacy/retention/run")
def run_privacy_retention(dry_run: bool = True, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        result = run_phi_retention(
            session,
            day=date.today(),
            answer_raw_retention_days=settings.phi_retention_answer_raw_days,
            observation_raw_retention_days=settings.phi_retention_observation_raw_days,
            policy_context_retention_days=settings.phi_retention_policy_context_days,
            dry_run=bool(dry_run),
        )
        record_operational_metric(
            session,
            metric_type="privacy_retention_job",
            status="ok",
            metric_date=date.today(),
            details={"dry_run": bool(dry_run), "affected_rows": result.get("affected_rows", {})},
        )
    return result


@app.get("/v1/admin/audit/export")
def audit_export(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: Optional[str] = None,
    redact_user_ids: Optional[bool] = None,
    redact_raw_payloads: Optional[bool] = None,
    _: None = Depends(_auth),
):
    try:
        end_day = parse_date(end_date) if end_date else date.today()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid end_date; expected YYYY-MM-DD") from exc
    try:
        start_day = parse_date(start_date) if start_date else (end_day - timedelta(days=29))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid start_date; expected YYYY-MM-DD") from exc
    if start_day > end_day:
        raise HTTPException(status_code=400, detail="start_date must be <= end_date")

    use_redact_user_ids = settings.phi_export_redact_user_ids if redact_user_ids is None else bool(redact_user_ids)
    use_redact_raw_payloads = settings.phi_export_redact_raw_payloads if redact_raw_payloads is None else bool(redact_raw_payloads)

    with session_scope(SessionLocal) as session:
        payload = build_clinical_audit_export(
            session,
            start_day=start_day,
            end_day=end_day,
            user_id=user_id,
            redact_user_ids=use_redact_user_ids,
            redact_raw_payloads=use_redact_raw_payloads,
            user_hash_salt=settings.phi_export_user_hash_salt,
        )
    return payload


@app.get("/v1/users/{user_id}/daily-questions", response_model=DailyQuestionsOut)
def daily_questions(user_id: str, date_param: Optional[str] = None, _: None = Depends(_auth)):
    day = _parse_date_query_or_400(date_param, field_name="date")
    with session_scope(SessionLocal) as session:
        ds = get_or_create_daily_session(
            session,
            registry_root=settings.registry_root,
            user_id=user_id,
            day=day,
            core_questions_per_day=settings.core_questions_per_day,
            extra_batch_size=settings.extra_batch_size,
            extra_batches_max_per_day=settings.extra_batches_max_per_day,
            item_repeat_cooldown_days=settings.item_repeat_cooldown_days,
            max_active_new_domains=settings.max_active_new_domains,
            safety_event_lookback_days=settings.safety_event_lookback_days,
            safety_min_questions_override=settings.safety_min_questions_override,
            policy_auto_rollback_enabled=settings.policy_auto_rollback_enabled,
            policy_rollback_window_days=settings.policy_rollback_window_days,
            policy_rollback_min_outcomes=settings.policy_rollback_min_outcomes,
            policy_rollback_max_safety_violations=settings.policy_rollback_max_safety_violations,
            policy_rollback_min_completion_drop=settings.policy_rollback_min_completion_drop,
            policy_rollback_burden_increase_ratio=settings.policy_rollback_burden_increase_ratio,
        )
        registry = load_registry(settings.registry_root, ds.registry_version)
        progress = compute_user_scale_progress(session, registry=registry, user_id=user_id, day=day)
        selection_explain = json.loads(ds.selection_explain_json)
        questions = _build_questions_payload(
            registry=registry,
            item_ids=json.loads(ds.core_questions_json),
            selection_explain=selection_explain,
            scale_progress=progress,
        )
        session_id = str(ds.session_id)
        registry_version = str(ds.registry_version)
        timeframe = str(ds.timeframe or "last_7_days")
        status = str(ds.status or "created")
        extra_pool_size = min(int(settings.extra_batches_max_per_day), sum(len(b) for b in json.loads(ds.extra_batches_json)))
        used_extras = min(int(ds.extra_batches_used or 0), int(extra_pool_size))
    return DailyQuestionsOut(
        session_id=session_id,
        user_id=user_id,
        date=day.isoformat(),
        registry_version=registry_version,
        timeframe=timeframe,
        status=status,
        questions=questions,
        extra={
            "batch_size": 1,
            "max_batches": int(settings.extra_batches_max_per_day),
            "available_batches": max(0, int(extra_pool_size - used_extras)),
            "used_batches": used_extras,
        },
        selection_explain=selection_explain,
        scale_progress=progress,
    )


@app.post("/v1/users/{user_id}/daily-questions/select", response_model=DailyQuestionsOut)
def daily_questions_select(user_id: str, body: DailyQuestionsSelectIn, _: None = Depends(_auth)):
    day = _parse_date_query_or_400(body.date, field_name="date")
    selection_mode = str(body.selection_mode or "deterministic")
    policy_context = _model_dump(body.context) if body.context is not None else None
    if body.allow_context_batches is not None:
        policy_context = dict(policy_context or {})
        policy_context["allow_context_batches"] = bool(body.allow_context_batches)
    include_explanations = bool(body.include_explanations)
    k_core = int(body.k_core) if body.k_core is not None else int(settings.core_questions_per_day)

    with session_scope(SessionLocal) as session:
        ds = get_or_create_daily_session(
            session,
            registry_root=settings.registry_root,
            user_id=user_id,
            day=day,
            core_questions_per_day=k_core,
            extra_batch_size=settings.extra_batch_size,
            extra_batches_max_per_day=settings.extra_batches_max_per_day,
            item_repeat_cooldown_days=settings.item_repeat_cooldown_days,
            max_active_new_domains=settings.max_active_new_domains,
            safety_event_lookback_days=settings.safety_event_lookback_days,
            safety_min_questions_override=settings.safety_min_questions_override,
            selection_mode=selection_mode,
            policy_context=policy_context,
            identity_mask_id=body.identity_mask_id,
            policy_root=settings.policy_root,
            policy_default_version=settings.policy_default_version,
            policy_epsilon_explore=settings.policy_epsilon_explore,
            policy_log_context_snapshot=settings.policy_log_context_snapshot,
            policy_log_candidate_set_snapshot=settings.policy_log_candidate_set_snapshot,
            policy_auto_rollback_enabled=settings.policy_auto_rollback_enabled,
            policy_rollback_window_days=settings.policy_rollback_window_days,
            policy_rollback_min_outcomes=settings.policy_rollback_min_outcomes,
            policy_rollback_max_safety_violations=settings.policy_rollback_max_safety_violations,
            policy_rollback_min_completion_drop=settings.policy_rollback_min_completion_drop,
            policy_rollback_burden_increase_ratio=settings.policy_rollback_burden_increase_ratio,
        )
        registry = load_registry(settings.registry_root, ds.registry_version)
        progress = compute_user_scale_progress(session, registry=registry, user_id=user_id, day=day)

        selection_explain = json.loads(ds.selection_explain_json)
        questions = _build_questions_payload(
            registry=registry,
            item_ids=json.loads(ds.core_questions_json),
            selection_explain=selection_explain,
            scale_progress=progress,
        )
        if include_explanations and ds.policy_decision_id:
            from questions_agent_platform.prod.models import PolicyDecision as PolicyDecisionRow

            pd = session.get(PolicyDecisionRow, ds.policy_decision_id)
            if pd and pd.explanations_json:
                try:
                    selection_explain.setdefault("policy", {})["explanations"] = json.loads(pd.explanations_json)
                except Exception:
                    selection_explain.setdefault("policy", {})["explanations"] = []
        session_id = str(ds.session_id)
        registry_version = str(ds.registry_version)
        timeframe = str(ds.timeframe or "last_7_days")
        status = str(ds.status or "created")
        extra_pool_size = min(int(settings.extra_batches_max_per_day), sum(len(b) for b in json.loads(ds.extra_batches_json)))
        used_extras = min(int(ds.extra_batches_used or 0), int(extra_pool_size))

    return DailyQuestionsOut(
        session_id=session_id,
        user_id=user_id,
        date=day.isoformat(),
        registry_version=registry_version,
        timeframe=timeframe,
        status=status,
        questions=questions,
        extra={
            "batch_size": 1,
            "max_batches": int(settings.extra_batches_max_per_day),
            "available_batches": max(0, int(extra_pool_size - used_extras)),
            "used_batches": used_extras,
        },
        selection_explain=selection_explain,
        scale_progress=progress,
    )


@app.post("/v1/users/{user_id}/answers", response_model=SubmitAnswersOut)
def post_answers(user_id: str, body: SubmitAnswersIn, _: None = Depends(_auth)):
    started = time.perf_counter()
    try:
        with session_scope(SessionLocal) as session:
            result = submit_answers(
                session,
                registry_root=settings.registry_root,
                user_id=user_id,
                session_id=body.session_id,
                answers=[_model_dump(a) for a in body.answers],
                drift_routing_table_path=settings.drift_routing_table_path or None,
            )
            ds = session.get(DailySession, body.session_id)
            metric_day = ds.date if ds is not None else date.today()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            record_operational_metric(
                session,
                metric_type="answer_ingestion",
                status="ok",
                metric_date=metric_day,
                latency_ms=elapsed_ms,
                user_id=user_id,
                session_id=body.session_id,
                details={"inserted_answer_events": int(result.get("inserted_answer_events") or 0)},
            )
            snapshot_write = result.get("snapshot_write") if isinstance(result, dict) else None
            record_operational_metric(
                session,
                metric_type="snapshot_write",
                status="ok",
                metric_date=metric_day,
                user_id=user_id,
                session_id=body.session_id,
                details={
                    "state_snapshot_id": snapshot_write.get("state_snapshot_id") if isinstance(snapshot_write, dict) else None,
                    "circle_snapshot_id": snapshot_write.get("circle_snapshot_id") if isinstance(snapshot_write, dict) else None,
                },
            )
        return SubmitAnswersOut(**result)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            with session_scope(SessionLocal) as log_session:
                ds = log_session.get(DailySession, body.session_id)
                metric_day = ds.date if ds is not None else date.today()
                record_operational_metric(
                    log_session,
                    metric_type="answer_ingestion",
                    status="error",
                    metric_date=metric_day,
                    latency_ms=elapsed_ms,
                    user_id=user_id,
                    session_id=body.session_id,
                    details={"error": str(exc)[:500]},
                )
                if isinstance(exc, SnapshotWriteError):
                    record_operational_metric(
                        log_session,
                        metric_type="snapshot_write",
                        status="error",
                        metric_date=metric_day,
                        user_id=user_id,
                        session_id=body.session_id,
                        details={"error": str(exc)[:500]},
                    )
        except Exception:
            logger.exception(
                "failed to persist operational metrics for submit_answers failure "
                "(user_id=%s session_id=%s)",
                user_id,
                body.session_id,
            )
        raise


@app.post("/v1/users/{user_id}/policy/outcomes")
def policy_outcomes(user_id: str, body: PolicyOutcomeUpdateIn, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        res = upsert_policy_outcome_update(
            session,
            user_id=user_id,
            body=_model_dump(body),
            store_z_snapshot=settings.policy_store_z_snapshot,
            store_z_delta=settings.policy_store_z_delta,
            quantize_decimals=settings.policy_quantize_decimals,
        )
    return res


@app.get("/v1/users/{user_id}/profile")
def get_profile(user_id: str, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        profile = get_user_profile(session, user_id)
    return {"user_id": user_id, "profile": profile}


@app.patch("/v1/users/{user_id}/profile")
def patch_profile(user_id: str, body: UserProfilePatchIn, _: None = Depends(_auth)):
    try:
        with session_scope(SessionLocal) as session:
            reg_v = ensure_registry_active(session, settings.registry_root)
            registry = load_registry(settings.registry_root, reg_v)
            profile = upsert_user_profile(
                session,
                user_id=user_id,
                patch=_model_dump(body, exclude_none=True),
                registry=registry,
                max_active_new_domains=settings.max_active_new_domains,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"user_id": user_id, "profile": profile}


@app.get("/v1/users/{user_id}/progress")
def get_progress(user_id: str, date_param: Optional[str] = None, window: Optional[str] = None, _: None = Depends(_auth)):
    day = _parse_date_query_or_400(date_param, field_name="date")
    window_days = _parse_window_days(window, default=30, min_days=1, max_days=365)
    start = day - timedelta(days=window_days - 1)

    with session_scope(SessionLocal) as session:
        reg_v = ensure_registry_active(session, settings.registry_root)
        registry = load_registry(settings.registry_root, reg_v)
        progress = compute_user_scale_progress(session, registry=registry, user_id=user_id, day=day)

        session_rows = (
            session.execute(
                select(DailySession.date, DailySession.status).where(
                    DailySession.user_id == user_id,
                    DailySession.date >= start,
                    DailySession.date <= day,
                )
            )
            .all()
        )
        answer_count = int(
            session.execute(
                select(func.count())
                .select_from(AnswerEvent)
                .join(DailySession, DailySession.session_id == AnswerEvent.session_id)
                .where(
                    AnswerEvent.user_id == user_id,
                    DailySession.date >= start,
                    DailySession.date <= day,
                )
            ).scalar()
            or 0
        )
        answered_days = int(
            session.execute(
                select(func.count(func.distinct(DailySession.date)))
                .select_from(AnswerEvent)
                .join(DailySession, DailySession.session_id == AnswerEvent.session_id)
                .where(
                    AnswerEvent.user_id == user_id,
                    DailySession.date >= start,
                    DailySession.date <= day,
                )
            ).scalar()
            or 0
        )

    active_days = len(session_rows)
    completed_days = sum(1 for row in session_rows if str(row.status or "") == "completed")
    started_days = sum(1 for row in session_rows if str(row.status or "") in {"started", "completed"})
    completion_rate = float(completed_days / active_days) if active_days > 0 else 0.0

    return {
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
    }


@app.get("/v1/users/{user_id}/safety-events")
def get_safety_events(
    user_id: str,
    date_param: Optional[str] = None,
    days: int = 30,
    status: Optional[str] = None,
    _: None = Depends(_auth),
):
    day = _parse_date_query_or_400(date_param, field_name="date")
    lookback_days = max(1, min(3650, int(days)))
    with session_scope(SessionLocal) as session:
        events = list_safety_events(
            session,
            user_id=user_id,
            day=day,
            lookback_days=lookback_days,
            status=status,
        )
    return {
        "user_id": user_id,
        "date": day.isoformat(),
        "lookback_days": lookback_days,
        "status": status,
        "events": events,
    }


@app.post("/v1/users/{user_id}/safety-events/{event_id}/resolve")
def resolve_safety(
    user_id: str,
    event_id: str,
    body: SafetyResolveIn,
    _: None = Depends(_auth),
):
    try:
        with session_scope(SessionLocal) as session:
            result = resolve_safety_event(
                session,
                user_id=user_id,
                event_id=event_id,
                resolved_by=body.resolved_by,
                resolved_reason=body.resolved_reason,
            )
    except ValueError as exc:
        code = 404 if "Unknown safety event" in str(exc) else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    return result


@app.get("/v1/users/{user_id}/emotion/deep_dive")
def emotion_deep_dive(user_id: str, date_param: Optional[str] = None, _: None = Depends(_auth)):
    day = _parse_date_query_or_400(date_param, field_name="date")
    with session_scope(SessionLocal) as session:
        payload = suggest_emotion_deep_dive(
            session,
            registry_root=settings.registry_root,
            user_id=user_id,
            day=day,
            lookback_safety_days=settings.safety_event_lookback_days,
            deep_dive_item_count=settings.emotion_deep_dive_item_count,
        )
    return payload


@app.post("/v1/users/{user_id}/experiments")
def upsert_experiment_endpoint(user_id: str, body: ExperimentUpsertIn, _: None = Depends(_auth)):
    try:
        with session_scope(SessionLocal) as session:
            payload = upsert_experiment(
                session,
                user_id=user_id,
                body=_model_dump(body, exclude_none=True),
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return payload


@app.get("/v1/users/{user_id}/experiments")
def list_experiments_endpoint(user_id: str, status: Optional[str] = None, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        rows = list_experiments(session, user_id=user_id, status=status)
    return {"user_id": user_id, "status": status, "experiments": rows}


@app.get("/v1/users/{user_id}/experiments/{experiment_id}")
def get_experiment_endpoint(user_id: str, experiment_id: str, _: None = Depends(_auth)):
    try:
        with session_scope(SessionLocal) as session:
            row = get_experiment(session, user_id=user_id, experiment_id=experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return row


@app.get("/v1/users/{user_id}/experiments/{experiment_id}/results")
def experiment_results_endpoint(
    user_id: str,
    experiment_id: str,
    date_param: Optional[str] = None,
    _: None = Depends(_auth),
):
    day = _parse_date_query_or_400(date_param, field_name="date")
    try:
        with session_scope(SessionLocal) as session:
            results = compute_experiment_results(
                session,
                user_id=user_id,
                experiment_id=experiment_id,
                day=day,
            )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return results


@app.get("/v1/users/{user_id}/ani/summary")
def ani_summary(user_id: str, date_param: Optional[str] = None, _: None = Depends(_auth)):
    day = _parse_date_query_or_400(date_param, field_name="date")
    with session_scope(SessionLocal) as session:
        payload = build_tta_summary(
            session,
            registry_root=settings.registry_root,
            user_id=user_id,
            day=day,
        )
    return payload


@app.get("/v1/admin/domains/readiness")
def domain_readiness(
    min_users: int = 100,
    min_cycles: int = 4,
    min_weeks: int = 8,
    min_paired_ratio: float = 0.80,
    _: None = Depends(_auth),
):
    with session_scope(SessionLocal) as session:
        report = build_domain_promotion_readiness(
            session,
            registry_root=settings.registry_root,
            min_users=int(min_users),
            min_cycles=int(min_cycles),
            min_weeks=int(min_weeks),
            min_paired_ratio=float(min_paired_ratio),
        )
    return report


# ---------------------------------------------------------------------------
# Site Configuration
# ---------------------------------------------------------------------------


@app.get("/v1/admin/site-configs", response_model=SiteConfigListOut)
def list_site_configs_endpoint(_: None = Depends(_auth)):
    configs = list_site_configs()
    return SiteConfigListOut(configs=configs)


@app.get("/v1/users/{user_id}/site-config", response_model=SiteConfigOut)
def user_site_config(user_id: str, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        config = resolve_site_config(session, user_id=user_id)
    return SiteConfigOut(**config)


@app.post("/v1/users/{user_id}/request-more-context")
def request_more_context(user_id: str, body: RequestMoreContextIn, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        try:
            batch = take_next_extra_batch(
                session,
                body.session_id,
                user_id=user_id,
                extra_batches_max_per_day=settings.extra_batches_max_per_day,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not batch:
            return {"batch": None, "message": "no more batches"}
        ds = session.get(DailySession, body.session_id)
        reg_v = ds.registry_version if ds else ensure_registry_active(session, settings.registry_root)
        registry = load_registry(settings.registry_root, reg_v)
        questions = _build_questions_payload(
            registry=registry,
            item_ids=batch,
            selection_explain={},
            scale_progress={},
        )
    return {"batch": questions}


@app.get("/v1/users/{user_id}/scales")
def user_scales(user_id: str, date_param: Optional[str] = None, _: None = Depends(_auth)):
    day = _parse_date_query_or_400(date_param, field_name="date")
    with session_scope(SessionLocal) as session:
        reg_v = ensure_registry_active(session, settings.registry_root)
        registry = load_registry(settings.registry_root, reg_v)
        progress = compute_user_scale_progress(session, registry=registry, user_id=user_id, day=day)
    return {"user_id": user_id, "date": day.isoformat(), "scales": progress}


@app.get("/v1/users/{user_id}/scales/{scale_id}/history")
def scale_history(user_id: str, scale_id: str, limit: int = 200, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        history = get_scale_history(session, user_id=user_id, scale_id=scale_id, limit=limit)
    return {"user_id": user_id, "scale_id": scale_id, "history": history}


@app.get("/v1/users/{user_id}/projection/questions")
def projection_questions(user_id: str, date_param: Optional[str] = None, _: None = Depends(_auth)):
    day = _parse_date_query_or_400(date_param, field_name="date")
    with session_scope(SessionLocal) as session:
        reg_v = ensure_registry_active(session, settings.registry_root)
        payload = build_projection_payload_pg(session, user_id=user_id, registry_version=reg_v, day=day)
    return payload


@app.get("/v1/users/{user_id}/state")
def state_snapshots(user_id: str, date_param: Optional[str] = None, window: Optional[str] = None, _: None = Depends(_auth)):
    day = _parse_date_query_or_400(date_param, field_name="date")
    window_days = _parse_window_days(window, default=30, min_days=1, max_days=365)
    with session_scope(SessionLocal) as session:
        snapshots = get_state_snapshots(
            session,
            user_id=user_id,
            day=day,
            window_days=window_days,
        )
    return {
        "user_id": user_id,
        "date": day.isoformat(),
        "window_days": window_days,
        "snapshots": snapshots,
    }


@app.get("/v1/users/{user_id}/circle")
def circle_snapshots(user_id: str, date_param: Optional[str] = None, window: Optional[str] = None, _: None = Depends(_auth)):
    day = _parse_date_query_or_400(date_param, field_name="date")
    window_days = _parse_window_days(window, default=30, min_days=1, max_days=365)
    with session_scope(SessionLocal) as session:
        snapshots = get_circle_snapshots(
            session,
            user_id=user_id,
            day=day,
            window_days=window_days,
        )
    return {
        "user_id": user_id,
        "date": day.isoformat(),
        "window_days": window_days,
        "snapshots": snapshots,
    }


@app.get("/v1/users/{user_id}/ews")
def ews_features(user_id: str, date_param: Optional[str] = None, window: Optional[str] = None, _: None = Depends(_auth)):
    day = _parse_date_query_or_400(date_param, field_name="date")
    window_days = _parse_window_days(window, default=30, min_days=1, max_days=365)
    with session_scope(SessionLocal) as session:
        rows = get_ews_features(
            session,
            user_id=user_id,
            day=day,
            window_days=window_days,
        )
    return {
        "user_id": user_id,
        "date": day.isoformat(),
        "window_days": window_days,
        "features": rows,
    }


@app.get("/v1/users/{user_id}/drift-events")
def drift_events(user_id: str, date_param: Optional[str] = None, window: Optional[str] = None, _: None = Depends(_auth)):
    day = _parse_date_query_or_400(date_param, field_name="date")
    window_days = _parse_window_days(window, default=30, min_days=1, max_days=365)
    with session_scope(SessionLocal) as session:
        rows = get_drift_events(
            session,
            user_id=user_id,
            day=day,
            window_days=window_days,
        )
    return {
        "user_id": user_id,
        "date": day.isoformat(),
        "window_days": window_days,
        "events": rows,
    }


@app.post("/v1/users/{user_id}/observations", response_model=SubmitObservationsOut)
def post_observations(user_id: str, body: SubmitObservationsIn, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        result = submit_observations(session, user_id=user_id, observations=[_model_dump(o) for o in body.observations])
    return SubmitObservationsOut(**result)


# ---------------------------------------------------------------------------
# N-of-1 Personal Calibration
# ---------------------------------------------------------------------------


@app.get("/v1/users/{user_id}/calibrations", response_model=CalibrationListOut)
def list_calibrations(user_id: str, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        calibrations = get_user_calibrations(session, user_id=user_id)
    return CalibrationListOut(user_id=user_id, calibrations=calibrations)


@app.post("/v1/users/{user_id}/calibrations/{scale_id}")
def update_calibration(
    user_id: str,
    scale_id: str,
    theta_obs: float,
    se_obs: float,
    delta_t: float = 1.0,
    _: None = Depends(_auth),
):
    try:
        with session_scope(SessionLocal) as session:
            result = upsert_calibration(
                session,
                user_id=user_id,
                scale_id=scale_id,
                theta_obs=theta_obs,
                se_obs=se_obs,
                delta_t=delta_t,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


# ---------------------------------------------------------------------------
# Coherence Detection
# ---------------------------------------------------------------------------


@app.get("/v1/users/{user_id}/sessions/{session_id}/coherence")
def session_coherence_endpoint(user_id: str, session_id: str, _: None = Depends(_auth)):
    with session_scope(SessionLocal) as session:
        result = get_session_coherence(session, user_id=user_id, session_id=session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No coherence assessment for this session")
    return result


@app.get("/v1/users/{user_id}/coherence/history", response_model=CoherenceHistoryOut)
def coherence_history_endpoint(
    user_id: str,
    window: Optional[str] = None,
    _: None = Depends(_auth),
):
    window_days = _parse_window_days(window, default=30, min_days=1, max_days=365)
    with session_scope(SessionLocal) as session:
        assessments = get_coherence_history(session, user_id=user_id, window_days=window_days)
    return CoherenceHistoryOut(user_id=user_id, assessments=assessments)


# ---------------------------------------------------------------------------
# Cardiometabolic Risk Index
# ---------------------------------------------------------------------------


@app.post("/v1/users/{user_id}/cardio-risk/compute")
def compute_cardio_risk_endpoint(
    user_id: str,
    date_param: Optional[str] = None,
    _: None = Depends(_auth),
):
    day = _parse_date_query_or_400(date_param, field_name="date")
    with session_scope(SessionLocal) as session:
        result = compute_and_store_cardio_risk(
            session,
            registry_root=settings.registry_root,
            user_id=user_id,
            day=day,
        )
    if result is None:
        return {"user_id": user_id, "date": day.isoformat(), "message": "insufficient instrument data"}
    return result


@app.get("/v1/users/{user_id}/cardio-risk/history", response_model=CardioRiskHistoryOut)
def cardio_risk_history_endpoint(
    user_id: str,
    window: Optional[str] = None,
    _: None = Depends(_auth),
):
    window_days = _parse_window_days(window, default=90, min_days=1, max_days=365)
    with session_scope(SessionLocal) as session:
        snapshots = get_cardio_risk_history(session, user_id=user_id, window_days=window_days)
    return CardioRiskHistoryOut(user_id=user_id, snapshots=snapshots)


# ---------------------------------------------------------------------------
# Anamnesis Episodes (Claim Family 4)
# ---------------------------------------------------------------------------

@app.get("/v1/users/{user_id}/anamnesis/episodes", response_model=AnamnesisEpisodeListOut)
def anamnesis_episodes_list_endpoint(
    user_id: str,
    status: Optional[str] = None,
    _: None = Depends(_auth),
):
    with session_scope(SessionLocal) as session:
        episodes = get_anamnesis_episodes(session, user_id=user_id, status_filter=status)
    return AnamnesisEpisodeListOut(user_id=user_id, episodes=episodes)


@app.get("/v1/users/{user_id}/anamnesis/episodes/{episode_id}", response_model=AnamnesisEpisodeOut)
def anamnesis_episode_get_endpoint(
    user_id: str,
    episode_id: str,
    _: None = Depends(_auth),
):
    with session_scope(SessionLocal) as session:
        ep = get_anamnesis_episode(session, user_id=user_id, episode_id=episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return AnamnesisEpisodeOut(**ep)


@app.post("/v1/users/{user_id}/anamnesis/episodes/{episode_id}/resolve")
def anamnesis_episode_resolve_endpoint(
    user_id: str,
    episode_id: str,
    body: Dict[str, Any],
    _: None = Depends(_auth),
):
    resolved_by = str(body.get("resolved_by") or "api")
    resolved_reason = str(body.get("resolved_reason") or "manual_resolve")
    with session_scope(SessionLocal) as session:
        try:
            result = resolve_anamnesis_episode(
                session, user_id=user_id, episode_id=episode_id,
                resolved_by=resolved_by, resolved_reason=resolved_reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


# ---------------------------------------------------------------------------
# Session Engagement (Claim Family 5)
# ---------------------------------------------------------------------------

@app.get("/v1/users/{user_id}/sessions/{session_id}/engagement", response_model=SessionUncertaintyProfileOut)
def session_engagement_endpoint(
    user_id: str,
    session_id: str,
    _: None = Depends(_auth),
):
    with session_scope(SessionLocal) as session:
        profile = get_session_engagement_profile(session, user_id=user_id, session_id=session_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="No engagement profile for this session")
    return SessionUncertaintyProfileOut(**profile)


@app.get("/v1/users/{user_id}/follow-up-queue")
def follow_up_queue_endpoint(
    user_id: str,
    _: None = Depends(_auth),
):
    with session_scope(SessionLocal) as session:
        items = get_follow_up_queue(session, user_id=user_id)
    return {"user_id": user_id, "items": items}


def _item_to_question(
    registry,
    item_id: str,
    *,
    progress_hint: Optional[str] = None,
    feeds_scales: Optional[List[str]] = None,
):
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


def _get_latest_session(session, *, user_id: str):
    row = session.execute(
        select(DailySession).where(DailySession.user_id == user_id).order_by(DailySession.date.desc()).limit(1)
    ).scalars().first()
    if not row:
        return None
    return {
        "session_id": row.session_id,
        "date": row.date.isoformat(),
        "registry_version": row.registry_version,
        "selection_explain": json.loads(row.selection_explain_json),
    }


def _list_versions(registry_root: str):
    from questions_agent_platform.pipeline.registry import list_versions

    return list_versions(registry_root)


def _parse_window_days(value: Optional[str], *, default: int, min_days: int, max_days: int) -> int:
    if value is None or str(value).strip() == "":
        return int(default)
    raw = str(value).strip()
    if raw.endswith("d"):
        raw = raw[:-1]
    try:
        days = int(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="window must be an integer number of days") from exc
    return max(int(min_days), min(int(max_days), days))


def _build_questions_payload(
    *,
    registry,
    item_ids: Sequence[str],
    selection_explain: Optional[Dict[str, Any]],
    scale_progress: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    scale_names = _scale_names_by_item(registry)
    hints = _progress_hints_by_item(
        item_ids=item_ids,
        selection_explain=selection_explain or {},
        scale_progress=scale_progress or {},
        scale_names_by_item=scale_names,
    )
    out: List[Dict[str, Any]] = []
    for item_id in item_ids:
        out.append(
            _item_to_question(
                registry,
                str(item_id),
                progress_hint=hints.get(str(item_id)),
                feeds_scales=scale_names.get(str(item_id), []),
            )
        )
    return out


def _scale_names_by_item(registry) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for scale in registry.scales.values():
        for si in scale.items:
            out.setdefault(si.item_id, [])
            if scale.name not in out[si.item_id]:
                out[si.item_id].append(scale.name)
    return out


def _progress_hints_by_item(
    *,
    item_ids: Sequence[str],
    selection_explain: Dict[str, Any],
    scale_progress: Dict[str, Any],
    scale_names_by_item: Dict[str, List[str]],
) -> Dict[str, str]:
    primary = selection_explain.get("primary_scale_by_item_id")
    primary = primary if isinstance(primary, dict) else {}
    out: Dict[str, str] = {}
    for item_id in item_ids:
        iid = str(item_id)
        primary_scale_id = str(primary.get(iid)) if iid in primary else None
        if primary_scale_id and primary_scale_id in scale_progress:
            p = scale_progress[primary_scale_id] or {}
            hint = p.get("unlock_hint")
            if hint:
                out[iid] = str(hint)
                continue
            name = str(p.get("name") or primary_scale_id)
            out[iid] = f"Feeds {name}"
            continue
        names = list(scale_names_by_item.get(iid, []))
        if names:
            out[iid] = f"Feeds {', '.join(names[:2])}"
    return out
