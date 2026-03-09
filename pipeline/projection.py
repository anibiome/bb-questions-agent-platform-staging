from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from questions_agent_platform.pipeline.baseline import BaselineState
from questions_agent_platform.pipeline.time_utils import date_to_start_iso, now_iso
from questions_agent_platform.contracts import (
    CONTRACT_QUESTIONS_TO_FUSION_EVIDENCE,
    build_contract_header,
    build_integration_anchor,
    validate_projection_payload_contract,
)


VECTOR_DIM = 128
SCHEMA_VERSION = "1.0"
PIPELINE_VERSION = "questions-agent-v1"


def build_questions_projection_payload(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    registry_version: str,
    day: date,
) -> Dict[str, Any]:
    """
    Build a structured evidence payload for the "questionnaires" modality (for Anifold fusion).

    This follows the "projection + uncertainty + attractor + velocity" template
    referenced in Multi_Omics_Projection_Specification_FINAL (2).docx:
      - This module produces geometry-ready evidence (not clinical interpretation).
      - Anifold is responsible for recomputing Z from evidence (this module does not produce Z).
    """
    # Current features from latest scale scores up to this day.
    latest_scores = _get_latest_scale_scores(conn, user_id=user_id, day=day)
    baselines = _get_baselines(conn, user_id=user_id)

    current_vec, current_counts = _hash_vector_from_scores(latest_scores, dim=VECTOR_DIM)
    attractor_vec, attractor_counts = _hash_vector_from_baselines(baselines, dim=VECTOR_DIM)

    uncertainty = _diag_uncertainty(current_counts, baselines)

    # Velocity: diff from previous stored projection (if any)
    timestamp = date_to_start_iso(day)
    anchor = _get_integration_anchor(conn, user_id=user_id, day=day)
    prev = _get_latest_projection_before(conn, user_id=user_id, timestamp=timestamp)
    velocity = [0.0] * VECTOR_DIM
    if prev is not None:
        prev_vec = prev.get("projection") or []
        if isinstance(prev_vec, list) and len(prev_vec) == VECTOR_DIM:
            velocity = [float(c) - float(p) for c, p in zip(current_vec, prev_vec)]

    distance = _l2_distance(current_vec, attractor_vec)

    payload = {
        "contract": {
            **build_contract_header(
                name=CONTRACT_QUESTIONS_TO_FUSION_EVIDENCE,
                schema_version="1.0",
                producer="questions_agent_platform.pipeline",
            ),
            "join_keys": anchor,
        },
        "schema_version": SCHEMA_VERSION,
        "subject_id": user_id,
        "timestamp": timestamp,
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
            "scale_scores": {sid: _score_summary(s) for sid, s in latest_scores.items()},
        },
        "event_log": [],
        "provenance": {
            "pipeline_version": PIPELINE_VERSION,
            "registry_version": registry_version,
            "processing_date": now_iso(),
        },
    }

    validate_projection_payload_contract(payload, vector_dim=VECTOR_DIM)

    conn.execute(
        """
        INSERT OR REPLACE INTO projections_questions(user_id, timestamp, payload_json)
        VALUES (?, ?, ?);
        """,
        (user_id, timestamp, json.dumps(payload, ensure_ascii=False)),
    )
    conn.commit()
    return payload


def _stable_bucket(feature: str, dim: int) -> Tuple[int, float]:
    h = hashlib.sha256(feature.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "big") % int(dim)
    sign = 1.0 if (h[4] & 1) == 1 else -1.0
    return idx, sign


def _hash_vector_from_scores(scores: Dict[str, Dict[str, Any]], *, dim: int) -> Tuple[List[float], List[int]]:
    vec = [0.0] * int(dim)
    counts = [0] * int(dim)

    for scale_id, score in scores.items():
        normalized = float(score.get("normalized_score", 0.0))
        z = score.get("personal_z")
        z_val = float(z) if z is not None else 0.0

        # Scale score feature (0..1)
        idx, sign = _stable_bucket(f"score:{scale_id}", dim)
        vec[idx] += sign * (normalized / 100.0)
        counts[idx] += 1

        # Drift feature (clipped)
        idx2, sign2 = _stable_bucket(f"z:{scale_id}", dim)
        vec[idx2] += sign2 * max(-3.0, min(3.0, z_val)) / 3.0
        counts[idx2] += 1

    _l2_normalize_inplace(vec)
    return vec, counts


def _hash_vector_from_baselines(baselines: Dict[str, BaselineState], *, dim: int) -> Tuple[List[float], List[int]]:
    vec = [0.0] * int(dim)
    counts = [0] * int(dim)

    for scale_id, b in baselines.items():
        idx, sign = _stable_bucket(f"baseline:{scale_id}", dim)
        vec[idx] += sign * (float(b.mean) / 100.0)
        counts[idx] += 1

    _l2_normalize_inplace(vec)
    return vec, counts


def _diag_uncertainty(counts: List[int], baselines: Dict[str, BaselineState]) -> List[float]:
    # v1 heuristic: fewer contributions => higher uncertainty. Baseline variance increases uncertainty slightly.
    baseline_var = 0.0
    if baselines:
        baseline_var = sum(float(b.var) for b in baselines.values()) / max(1, len(baselines))

    out = []
    for c in counts:
        base = 1.0 / (1.0 + float(c))
        out.append(float(min(1.0, base + 0.01 * baseline_var)))
    return out


def _confidence_tier_from_counts(counts: List[int]) -> str:
    # If the vector has many populated dims, we treat confidence as higher.
    populated = sum(1 for c in counts if c > 0)
    if populated >= 32:
        return "high"
    if populated >= 12:
        return "medium"
    return "low"


def _l2_distance(a: List[float], b: List[float]) -> float:
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def _l2_normalize_inplace(vec: List[float]) -> None:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 1e-12:
        return
    inv = 1.0 / float(norm)
    for i in range(len(vec)):
        vec[i] = float(vec[i]) * inv


def _score_summary(score: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "normalized_score": float(score.get("normalized_score", 0.0)),
        "confidence_tier": score.get("confidence_tier"),
        "window_end": score.get("window_end"),
        "personal_z": score.get("personal_z"),
        "delta_vs_prev": score.get("delta_vs_prev"),
    }


def _get_latest_scale_scores(conn: sqlite3.Connection, *, user_id: str, day: date) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT * FROM scale_scores
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
            "confidence_tier": str(r["confidence_tier"]),
            "personal_z": float(r["personal_z"]) if r["personal_z"] is not None else None,
            "delta_vs_prev": float(r["delta_vs_prev"]) if r["delta_vs_prev"] is not None else None,
        }
    return latest


def _get_baselines(conn: sqlite3.Connection, *, user_id: str) -> Dict[str, BaselineState]:
    rows = conn.execute(
        "SELECT scale_id, mean, var, n FROM scale_baselines WHERE user_id=?;",
        (user_id,),
    ).fetchall()
    out: Dict[str, BaselineState] = {}
    for r in rows:
        out[str(r["scale_id"])] = BaselineState(mean=float(r["mean"]), var=float(r["var"]), n=int(r["n"]))
    return out


def _get_latest_projection_before(conn: sqlite3.Connection, *, user_id: str, timestamp: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT timestamp, payload_json
        FROM projections_questions
        WHERE user_id=? AND timestamp<?
        ORDER BY timestamp DESC
        LIMIT 1;
        """,
        (user_id, timestamp),
    ).fetchone()
    if not row:
        return None
    payload = json.loads(row["payload_json"])
    proj = payload.get("modality_projections", {}).get("questionnaires", {})
    return {"timestamp": str(row["timestamp"]), "projection": proj.get("projection")}


def _get_integration_anchor(conn: sqlite3.Connection, *, user_id: str, day: date) -> Dict[str, Any]:
    row = conn.execute(
        """
        SELECT session_id, policy_decision_id
        FROM daily_sessions
        WHERE user_id=? AND date=?
        ORDER BY created_at DESC
        LIMIT 1;
        """,
        (user_id, day.isoformat()),
    ).fetchone()
    return build_integration_anchor(
        user_id=user_id,
        day=day,
        session_id=(str(row["session_id"]) if row and row["session_id"] is not None else None),
        decision_id=(str(row["policy_decision_id"]) if row and row["policy_decision_id"] is not None else None),
    )
