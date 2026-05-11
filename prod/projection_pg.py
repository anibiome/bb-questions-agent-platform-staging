from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, select
from sqlalchemy.orm import Session

from questions_agent_platform.pipeline.baseline import BaselineState
from questions_agent_platform.pipeline.projection import (
    VECTOR_DIM,
)
from questions_agent_platform.pipeline.time_utils import date_to_start_iso
from questions_agent_platform.contracts import (
    CONTRACT_QUESTIONS_TO_FUSION_EVIDENCE,
    QUESTION_EVIDENCE_CLASS,
    build_behavioral_evidence_boundary,
    build_contract_header,
    build_integration_anchor,
    validate_projection_payload_contract,
)

from questions_agent_platform.prod.models import (
    DailySession,
    QuestionsProjection,
    ScaleBaseline,
    ScaleScore,
)


def build_projection_payload_pg(
    session: Session,
    *,
    user_id: str,
    registry_version: str,
    day: date,
) -> Dict[str, Any]:
    """
    Use the same hashing + evidence logic, but load/store via Postgres tables.
    """
    # Load baselines and scores into the shapes expected by the hashing logic.
    baselines = _get_baselines(session, user_id=user_id)
    latest_scores = _get_latest_scores(session, user_id=user_id, day=day)

    # Reuse the in-memory builder by temporarily writing into a dict
    anchor = _get_integration_anchor(session, user_id=user_id, day=day)
    payload = _build_payload_from_state(
        user_id=user_id,
        registry_version=registry_version,
        day=day,
        baselines=baselines,
        latest_scores=latest_scores,
        anchor=anchor,
    )
    current_projection = payload["modality_projections"]["questionnaires"]["projection"]
    previous_projection = _get_previous_projection_vector(
        session,
        user_id=user_id,
        before_timestamp=date_to_start_iso(day),
    )
    if previous_projection is not None and len(previous_projection) == len(
        current_projection
    ):
        payload["modality_projections"]["questionnaires"]["velocity"] = [
            float(c) - float(p) for c, p in zip(current_projection, previous_projection)
        ]

    ts = payload["timestamp"]
    validate_projection_payload_contract(payload, vector_dim=VECTOR_DIM)
    session.merge(
        QuestionsProjection(
            user_id=user_id,
            timestamp=ts,
            payload_json=json.dumps(payload, ensure_ascii=False),
        )
    )
    return payload


def _build_payload_from_state(
    *,
    user_id: str,
    registry_version: str,
    day: date,
    baselines: Dict[str, BaselineState],
    latest_scores: Dict[str, Dict[str, Any]],
    anchor: Dict[str, Any],
) -> Dict[str, Any]:
    # We call into the existing helper by temporarily emulating the sqlite layer:
    # The builder is self-contained and uses only the latest scores/baselines.
    from questions_agent_platform.pipeline.projection import (
        _hash_vector_from_scores,
        _hash_vector_from_baselines,
        _diag_uncertainty,
        _confidence_tier_from_counts,
        _l2_distance,
        _score_summary,
        SCHEMA_VERSION,
        PIPELINE_VERSION,
    )

    current_vec, current_counts = _hash_vector_from_scores(
        latest_scores, dim=VECTOR_DIM
    )
    attractor_vec, attractor_counts = _hash_vector_from_baselines(
        baselines, dim=VECTOR_DIM
    )
    uncertainty = _diag_uncertainty(current_counts, baselines)
    velocity = (
        [0.0] * VECTOR_DIM
    )  # default; replaced in build_projection_payload_pg when previous projection exists
    distance = _l2_distance(current_vec, attractor_vec)

    timestamp = date_to_start_iso(day)
    payload = {
        "contract": {
            **build_contract_header(
                name=CONTRACT_QUESTIONS_TO_FUSION_EVIDENCE,
                schema_version="1.0",
                producer="questions_agent_platform.prod",
            ),
            "join_keys": anchor,
        },
        "schema_version": SCHEMA_VERSION,
        "subject_id": user_id,
        "timestamp": timestamp,
        "evidence_class": QUESTION_EVIDENCE_CLASS,
        "claim_boundary": build_behavioral_evidence_boundary(),
        "modality_projections": {
            "questionnaires": {
                "projection": current_vec,
                "uncertainty": {"diag": uncertainty},
                "attractor_candidate": attractor_vec,
                "attractor_source": "subject_baseline",
                "velocity": velocity,
                "distance_from_attractor": distance,
                "qc_status": "pass",
                "confidence_tier": _confidence_tier_from_counts(current_counts),
            }
        },
        "derived_features": {
            "scale_scores": {
                sid: _score_summary(s) for sid, s in latest_scores.items()
            },
        },
        "event_log": [],
        "provenance": {
            "pipeline_version": PIPELINE_VERSION,
            "registry_version": registry_version,
            "processing_date": date_to_start_iso(day),
        },
    }
    return payload


def _get_integration_anchor(
    session: Session, *, user_id: str, day: date
) -> Dict[str, Any]:
    row = (
        session.execute(
            select(DailySession)
            .where(and_(DailySession.user_id == user_id, DailySession.date == day))
            .order_by(desc(DailySession.created_at))
            .limit(1)
        )
        .scalars()
        .first()
    )
    return build_integration_anchor(
        user_id=user_id,
        day=day,
        session_id=(str(row.session_id) if row is not None else None),
        decision_id=(
            str(row.policy_decision_id)
            if row is not None and row.policy_decision_id
            else None
        ),
    )


def _get_baselines(session: Session, *, user_id: str) -> Dict[str, BaselineState]:
    rows = (
        session.execute(select(ScaleBaseline).where(ScaleBaseline.user_id == user_id))
        .scalars()
        .all()
    )
    out: Dict[str, BaselineState] = {}
    for r in rows:
        out[str(r.scale_id)] = BaselineState(
            mean=float(r.mean), var=float(r.var), n=int(r.n)
        )
    return out


def _get_latest_scores(
    session: Session, *, user_id: str, day: date
) -> Dict[str, Dict[str, Any]]:
    rows = (
        session.execute(
            select(ScaleScore)
            .where(and_(ScaleScore.user_id == user_id, ScaleScore.window_end <= day))
            .order_by(ScaleScore.window_end.asc(), ScaleScore.computed_at.asc())
        )
        .scalars()
        .all()
    )
    latest: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        latest[str(r.scale_id)] = {
            "scale_id": str(r.scale_id),
            "window_end": r.window_end.isoformat(),
            "normalized_score": float(r.normalized_score),
            "confidence_tier": str(r.confidence_tier),
            "personal_z": float(r.personal_z) if r.personal_z is not None else None,
            "delta_vs_prev": float(r.delta_vs_prev)
            if r.delta_vs_prev is not None
            else None,
        }
    return latest


def _get_previous_projection_vector(
    session: Session,
    *,
    user_id: str,
    before_timestamp: str,
) -> Optional[List[float]]:
    row = (
        session.execute(
            select(QuestionsProjection)
            .where(
                and_(
                    QuestionsProjection.user_id == user_id,
                    QuestionsProjection.timestamp < before_timestamp,
                )
            )
            .order_by(desc(QuestionsProjection.timestamp))
            .limit(1)
        )
        .scalars()
        .first()
    )
    if row is None:
        return None
    try:
        payload = json.loads(str(row.payload_json))
        vec = payload["modality_projections"]["questionnaires"]["projection"]
        if isinstance(vec, list):
            return [float(x) for x in vec]
    except Exception:
        return None
    return None
