from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import connect, init_db
from questions_agent_platform.pipeline.demo import seed_demo_registry
from questions_agent_platform.pipeline.demo_signals import demo_value_for_item
from questions_agent_platform.pipeline.projection import build_questions_projection_payload
from questions_agent_platform.pipeline.registry import load_registry
from questions_agent_platform.pipeline.service import (
    ensure_registry_active,
    get_or_create_daily_session,
    submit_answers,
    take_next_extra_batch,
    upsert_policy_outcome_update,
)
from questions_agent_platform.tools.policy_audit import evaluate_policy_decision_rows, parse_json_obj


@dataclass(frozen=True)
class ClosedLoopTruthDemoResult:
    json_path: Path
    html_path: Path


def generate_closed_loop_truth_demo(
    *,
    cfg: QuestionsAgentConfig,
    out_dir: str,
    days: int,
    user_id: str,
    seed: int,
    epsilon: float,
    reset_db: bool,
) -> ClosedLoopTruthDemoResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    days = max(1, int(days))
    run_user_id = str(user_id).strip() or "closed-loop-user-1"
    start_day = date.today() - timedelta(days=days - 1)
    end_day = start_day + timedelta(days=days - 1)

    if bool(reset_db):
        init_db(cfg.database_path)

    cfg_effective = replace(
        cfg,
        policy_default_mode="policy_live",
        policy_epsilon_explore=float(epsilon),
        policy_log_context_snapshot=True,
        policy_log_candidate_set_snapshot=True,
        policy_store_z_delta=True,
        policy_auto_rollback_enabled=False,
    )
    rng = Random(int(seed))

    with connect(cfg.database_path) as conn:
        _ensure_seeded_registry(conn, cfg=cfg_effective)
        _run_replay_and_updates(
            conn=conn,
            cfg=cfg_effective,
            user_id=run_user_id,
            start_day=start_day,
            days=days,
            rng=rng,
        )
        payload = _build_truth_payload(
            conn=conn,
            db_path=cfg.database_path,
            user_id=run_user_id,
            start_day=start_day,
            end_day=end_day,
            days=days,
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    stem = f"closed_loop_truth_demo_{ts}"
    json_path = out / f"{stem}.json"
    html_path = out / f"{stem}.html"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(payload), encoding="utf-8")
    return ClosedLoopTruthDemoResult(json_path=json_path, html_path=html_path)


def _ensure_seeded_registry(conn: sqlite3.Connection, *, cfg: QuestionsAgentConfig) -> None:
    versions_dir = Path(cfg.registry_root) / "versions"
    if not versions_dir.exists() or not any(versions_dir.iterdir()):
        seed_demo_registry(cfg.registry_root)
    ensure_registry_active(conn, cfg.registry_root)


def _run_replay_and_updates(
    *,
    conn: sqlite3.Connection,
    cfg: QuestionsAgentConfig,
    user_id: str,
    start_day: date,
    days: int,
    rng: Random,
) -> None:
    for day_index in range(days):
        day = start_day + timedelta(days=day_index)
        session = get_or_create_daily_session(
            conn,
            cfg=cfg,
            registry_root=cfg.registry_root,
            user_id=user_id,
            day=day,
            selection_mode="policy_live",
        )
        registry = load_registry(cfg.registry_root, session.registry_version)

        core_answers = []
        for item_id in session.core_item_ids:
            tags = tuple(registry.items[item_id].tags)
            core_answers.append(
                {
                    "client_event_id": f"{user_id}::{day.isoformat()}::core::{item_id}",
                    "item_id": item_id,
                    "value": demo_value_for_item(tags=tags, day_index=day_index),
                    "answered_at": f"{day.isoformat()}T12:00:00Z",
                }
            )
        submit_answers(
            conn,
            registry_root=cfg.registry_root,
            user_id=user_id,
            session_id=session.session_id,
            answers=core_answers,
        )

        explain = session.selection_explain if isinstance(session.selection_explain, dict) else {}
        near_unlock = bool(explain.get("near_unlock_scales"))
        want_extra = rng.random() < (0.55 if near_unlock else 0.20)
        batches_used = 0
        while want_extra and batches_used < int(cfg.extra_batches_max_per_day):
            batch = take_next_extra_batch(
                conn,
                session.session_id,
                user_id=user_id,
                extra_batches_max_per_day=cfg.extra_batches_max_per_day,
            )
            if not batch:
                break
            extra_answers = []
            for item_id in batch:
                tags = tuple(registry.items[item_id].tags)
                extra_answers.append(
                    {
                        "client_event_id": f"{user_id}::{day.isoformat()}::extra{batches_used}::{item_id}",
                        "item_id": item_id,
                        "value": demo_value_for_item(tags=tags, day_index=day_index),
                        "answered_at": f"{day.isoformat()}T12:{5 + batches_used:02d}:00Z",
                    }
                )
            submit_answers(
                conn,
                registry_root=cfg.registry_root,
                user_id=user_id,
                session_id=session.session_id,
                answers=extra_answers,
            )
            batches_used += 1
            want_extra = rng.random() < 0.25

        projection = build_questions_projection_payload(
            conn,
            user_id=user_id,
            registry_version=session.registry_version,
            day=day,
        )
        decision_id = session.policy_decision_id
        if decision_id:
            q = projection["modality_projections"]["questionnaires"]
            z_after = [float(v) for v in q.get("projection", [])]
            uncertainty_after = [float(v) for v in q.get("uncertainty", {}).get("diag", [])]
            z_before, uncertainty_before = _make_pre_snapshots(
                z_after=z_after,
                uncertainty_after=uncertainty_after,
                day_index=day_index,
            )
            upsert_policy_outcome_update(
                conn,
                cfg=cfg,
                user_id=user_id,
                body={
                    "decision_id": decision_id,
                    "z_before": z_before,
                    "z_after": z_after,
                    "uncertainty_before_diag": uncertainty_before,
                    "uncertainty_after_diag": uncertainty_after,
                },
            )

def _make_pre_snapshots(*, z_after: Sequence[float], uncertainty_after: Sequence[float], day_index: int) -> Tuple[List[float], List[float]]:
    z_before: List[float] = []
    for idx, val in enumerate(z_after):
        jitter = 0.02 * math.sin((day_index + 1) * (idx + 1) * 0.037)
        z_before.append(float(val) + float(jitter))

    uncertainty_before: List[float] = []
    for idx, val in enumerate(uncertainty_after):
        bump = 0.05 + 0.02 * abs(math.sin((day_index + 1) * (idx + 1) * 0.019))
        uncertainty_before.append(float(min(1.0, max(0.0, float(val) + bump))))
    return z_before, uncertainty_before


def _build_truth_payload(
    *,
    conn: sqlite3.Connection,
    db_path: str,
    user_id: str,
    start_day: date,
    end_day: date,
    days: int,
) -> Dict[str, Any]:
    sessions = conn.execute(
        """
        SELECT session_id, date, selection_mode, policy_decision_id
        FROM daily_sessions
        WHERE user_id=? AND date>=? AND date<=?
        ORDER BY date ASC;
        """,
        (user_id, start_day.isoformat(), end_day.isoformat()),
    ).fetchall()
    decisions = conn.execute(
        """
        SELECT decision_id, date, selected_item_ids_json, candidate_set_json, explanations_json, context_json
        FROM policy_decisions
        WHERE user_id=? AND date>=? AND date<=?
        ORDER BY date ASC;
        """,
        (user_id, start_day.isoformat(), end_day.isoformat()),
    ).fetchall()
    projections = conn.execute(
        """
        SELECT timestamp, payload_json
        FROM projections_questions
        WHERE user_id=? AND timestamp>=? AND timestamp<?
        ORDER BY timestamp ASC;
        """,
        (
            user_id,
            f"{start_day.isoformat()}T00:00:00Z",
            f"{(end_day + timedelta(days=1)).isoformat()}T00:00:00Z",
        ),
    ).fetchall()
    outcomes = conn.execute(
        """
        SELECT decision_id, completion_rate, total_reward, uncertainty_before_mean, uncertainty_after_mean
        FROM policy_outcomes
        WHERE user_id=? AND date>=? AND date<=?;
        """,
        (user_id, start_day.isoformat(), end_day.isoformat()),
    ).fetchall()

    sessions_by_date = {str(r["date"]): str(r["session_id"]) for r in sessions}
    outcomes_by_decision = {str(r["decision_id"]): r for r in outcomes}
    anchors_by_decision = _projection_anchor_index(projections)

    checks = evaluate_policy_decision_rows(decisions)
    decision_count = len(decisions)
    decision_with_projection = 0
    decision_with_outcome = 0
    decision_with_full_chain = 0
    missing_projection_decisions: List[str] = []
    missing_outcome_decisions: List[str] = []
    per_day: List[Dict[str, Any]] = []

    for row in decisions:
        decision_id = str(row["decision_id"])
        decision_day = str(row["date"])
        session_id = sessions_by_date.get(decision_day)
        anchor = anchors_by_decision.get(decision_id)
        has_projection = (
            bool(anchor)
            and str(anchor.get("subject_id") or "") == user_id
            and str(anchor.get("day") or "") == decision_day
            and (session_id is None or str(anchor.get("session_id") or "") == session_id)
        )
        has_outcome = decision_id in outcomes_by_decision
        if has_projection:
            decision_with_projection += 1
        else:
            missing_projection_decisions.append(decision_id)
        if has_outcome:
            decision_with_outcome += 1
        else:
            missing_outcome_decisions.append(decision_id)
        if has_projection and has_outcome:
            decision_with_full_chain += 1

        out_row = outcomes_by_decision.get(decision_id)
        uncertainty_reduction = None
        if out_row and out_row["uncertainty_before_mean"] is not None and out_row["uncertainty_after_mean"] is not None:
            uncertainty_reduction = float(out_row["uncertainty_before_mean"]) - float(out_row["uncertainty_after_mean"])

        per_day.append(
            {
                "date": decision_day,
                "decision_id": decision_id,
                "session_id": session_id,
                "projection_anchor_ok": bool(has_projection),
                "outcome_ok": bool(has_outcome),
                "completion_rate": _f_or_none(out_row["completion_rate"]) if out_row else None,
                "total_reward": _f_or_none(out_row["total_reward"]) if out_row else None,
                "uncertainty_reduction": _f_or_none(uncertainty_reduction),
            }
        )

    full_chain_coverage = (float(decision_with_full_chain) / float(decision_count)) if decision_count else 0.0
    explainability_coverage = (
        float(checks["explainability_complete_count"]) / float(decision_count)
        if decision_count
        else 0.0
    )
    projection_coverage = (float(decision_with_projection) / float(decision_count)) if decision_count else 0.0
    outcome_coverage = (float(decision_with_outcome) / float(decision_count)) if decision_count else 0.0

    gates = {
        "has_policy_decisions": decision_count > 0,
        "safety_no_violations": int(checks["safety_violation_total"]) == 0,
        "candidate_snapshot_complete": int(checks.get("candidate_set_snapshot_count") or 0) == decision_count if decision_count else False,
        "context_snapshot_complete": int(checks.get("context_snapshot_count") or 0) == decision_count if decision_count else False,
        "projection_anchor_complete": decision_with_projection == decision_count if decision_count else False,
        "outcome_complete": decision_with_outcome == decision_count if decision_count else False,
        "explainability_complete": int(checks["explainability_complete_count"]) == decision_count if decision_count else False,
        "full_chain_complete": decision_with_full_chain == decision_count if decision_count else False,
    }
    verdict = "PASS" if all(bool(v) for v in gates.values()) else "FAIL"

    return {
        "meta": {
            "label": "Questions Agent Closed-Loop Truth Demo",
            "generated_at_utc": _now_iso(),
            "db_path": db_path,
            "user_id": user_id,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "days_requested": int(days),
            "verdict": verdict,
        },
        "coverage": {
            "sessions": len(sessions),
            "policy_decisions": decision_count,
            "policy_outcomes": len(outcomes),
            "projections_questions": len(projections),
            "decision_with_projection_anchor_count": decision_with_projection,
            "decision_with_outcome_count": decision_with_outcome,
            "decision_with_full_chain_count": decision_with_full_chain,
            "projection_coverage": _f_or_none(projection_coverage),
            "outcome_coverage": _f_or_none(outcome_coverage),
            "full_chain_coverage": _f_or_none(full_chain_coverage),
            "explainability_coverage": _f_or_none(explainability_coverage),
            "candidate_set_snapshot_coverage": _f_or_none(checks.get("candidate_set_snapshot_coverage")),
            "context_snapshot_coverage": _f_or_none(checks.get("context_snapshot_coverage")),
        },
        "checks": checks,
        "quality_gates": gates,
        "missing_links": {
            "missing_projection_decision_ids": missing_projection_decisions,
            "missing_outcome_decision_ids": missing_outcome_decisions,
        },
        "daily": per_day,
        "notes": [
            "Questions Agent emits structured evidence for fusion; it does not compute Anifold Z.",
            "Outcome updates attach Anifold-provided Z/uncertainty snapshots to decision_id for delayed reward accounting.",
            "Constraint checks enforce selected items are inside candidate set C and include mandatory anchors.",
        ],
    }


def _projection_anchor_index(rows: Sequence[sqlite3.Row]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        payload = parse_json_obj(row["payload_json"])
        if not payload:
            continue
        contract = payload.get("contract") if isinstance(payload.get("contract"), dict) else {}
        join_keys = contract.get("join_keys") if isinstance(contract.get("join_keys"), dict) else {}
        decision_id = str(join_keys.get("decision_id") or "")
        if not decision_id:
            continue
        out[decision_id] = {
            "decision_id": decision_id,
            "session_id": str(join_keys.get("session_id") or ""),
            "subject_id": str(join_keys.get("subject_id") or ""),
            "day": str(join_keys.get("date") or join_keys.get("day") or ""),
        }
    return out


def _f_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _render_html(payload: Dict[str, Any]) -> str:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    gates = payload.get("quality_gates") if isinstance(payload.get("quality_gates"), dict) else {}
    missing = payload.get("missing_links") if isinstance(payload.get("missing_links"), dict) else {}
    daily = payload.get("daily") if isinstance(payload.get("daily"), list) else []
    notes = payload.get("notes") if isinstance(payload.get("notes"), list) else []

    def gate_li(name: str, ok: bool) -> str:
        state = "PASS" if bool(ok) else "FAIL"
        cls = "ok" if bool(ok) else "bad"
        return f"<li><span class='pill {cls}'>{state}</span>{name}</li>"

    daily_rows = []
    for row in daily:
        if not isinstance(row, dict):
            continue
        daily_rows.append(
            "<tr>"
            f"<td>{row.get('date', '')}</td>"
            f"<td>{row.get('decision_id', '')}</td>"
            f"<td>{'yes' if row.get('projection_anchor_ok') else 'no'}</td>"
            f"<td>{'yes' if row.get('outcome_ok') else 'no'}</td>"
            f"<td>{_fmt_num(row.get('completion_rate'))}</td>"
            f"<td>{_fmt_num(row.get('uncertainty_reduction'))}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Closed-Loop Truth Demo</title>
  <style>
    :root {{
      --bg: #f3f3f1;
      --fg: #111;
      --muted: #555;
      --card: #fff;
      --line: #dfdfdc;
      --ok: #0f7a4c;
      --bad: #a8382b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--fg);
      line-height: 1.4;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    .meta {{ color: var(--muted); margin-bottom: 20px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin: 6px 0; }}
    .pill {{
      display: inline-block;
      min-width: 52px;
      text-align: center;
      border-radius: 999px;
      padding: 2px 9px;
      margin-right: 8px;
      font-size: 12px;
      font-weight: 700;
      color: white;
    }}
    .pill.ok {{ background: var(--ok); }}
    .pill.bad {{ background: var(--bad); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      font-size: 13px;
      vertical-align: top;
    }}
    th {{ background: #fafaf8; }}
    tr:last-child td {{ border-bottom: none; }}
    .muted {{ color: var(--muted); }}
    code {{
      background: #f8f8f6;
      border: 1px solid var(--line);
      padding: 1px 6px;
      border-radius: 6px;
      font-family: ui-monospace, Menlo, monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Closed-Loop Truth Demo</h1>
    <div class="meta">
      User: <code>{meta.get("user_id", "")}</code> |
      Window: <code>{meta.get("start_date", "")}</code> → <code>{meta.get("end_date", "")}</code> |
      Verdict: <strong>{meta.get("verdict", "UNKNOWN")}</strong>
    </div>

    <div class="grid">
      <section class="card">
        <h2>Coverage</h2>
        <ul>
          <li>Sessions: <strong>{coverage.get("sessions", 0)}</strong></li>
          <li>Policy decisions: <strong>{coverage.get("policy_decisions", 0)}</strong></li>
          <li>Policy outcomes: <strong>{coverage.get("policy_outcomes", 0)}</strong></li>
          <li>Projections: <strong>{coverage.get("projections_questions", 0)}</strong></li>
          <li>Full chain coverage: <strong>{_fmt_num(coverage.get("full_chain_coverage"))}</strong></li>
          <li>Candidate snapshot coverage: <strong>{_fmt_num(coverage.get("candidate_set_snapshot_coverage"))}</strong></li>
          <li>Context snapshot coverage: <strong>{_fmt_num(coverage.get("context_snapshot_coverage"))}</strong></li>
        </ul>
      </section>
      <section class="card">
        <h2>Constraint Checks</h2>
        <ul>
          <li>Safety violations: <strong>{checks.get("safety_violation_total", 0)}</strong></li>
          <li>Explainability complete: <strong>{checks.get("explainability_complete_count", 0)}</strong></li>
          <li>Candidate snapshots: <strong>{checks.get("candidate_set_snapshot_count", 0)}</strong></li>
          <li>Context snapshots: <strong>{checks.get("context_snapshot_count", 0)}</strong></li>
          <li>Outside C: <strong>{(checks.get("violations") or {}).get("selected_outside_candidate_set", 0)}</strong></li>
          <li>Blocked selected: <strong>{(checks.get("violations") or {}).get("blocked_item_selected", 0)}</strong></li>
        </ul>
      </section>
      <section class="card">
        <h2>Quality Gates</h2>
        <ul>
          {''.join(gate_li(name, bool(ok)) for name, ok in gates.items())}
        </ul>
      </section>
    </div>

    <section class="card" style="margin-bottom:16px;">
      <h2>Missing Links</h2>
      <div class="muted">Missing projection anchors: {len(missing.get("missing_projection_decision_ids", []) if isinstance(missing.get("missing_projection_decision_ids"), list) else [])}</div>
      <div class="muted">Missing outcomes: {len(missing.get("missing_outcome_decision_ids", []) if isinstance(missing.get("missing_outcome_decision_ids"), list) else [])}</div>
    </section>

    <h2 style="margin-bottom:8px;">Daily Traceability</h2>
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Decision</th>
          <th>Projection Anchor</th>
          <th>Outcome</th>
          <th>Completion</th>
          <th>Uncertainty Δ</th>
        </tr>
      </thead>
      <tbody>
        {''.join(daily_rows)}
      </tbody>
    </table>

    <section class="card" style="margin-top:16px;">
      <h2>Notes</h2>
      <ul>
        {''.join(f"<li>{str(n)}</li>" for n in notes)}
      </ul>
    </section>
  </div>
</body>
</html>
"""


def _fmt_num(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "-"
