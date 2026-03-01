from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from questions_agent_platform.tools.policy_audit import evaluate_policy_decision_rows


@dataclass(frozen=True)
class BriefResult:
    json_path: Path
    html_path: Path


def generate_policy_brief(
    *,
    db_path: str,
    out_dir: str,
    label: str = "Questions Agent Policy Brief",
    policy_root: Optional[str] = None,
    min_eval_rows: int = 100,
    min_segment_rows: int = 30,
) -> BriefResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        report = _build_report(
            conn=conn,
            db_path=db_path,
            label=label,
            policy_root=policy_root,
            min_eval_rows=min_eval_rows,
            min_segment_rows=min_segment_rows,
        )
    finally:
        conn.close()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    stem = f"policy_brief_{ts}"
    json_path = out / f"{stem}.json"
    html_path = out / f"{stem}.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(report), encoding="utf-8")
    return BriefResult(json_path=json_path, html_path=html_path)


def _build_report(
    *,
    conn: sqlite3.Connection,
    db_path: str,
    label: str,
    policy_root: Optional[str],
    min_eval_rows: int,
    min_segment_rows: int,
) -> Dict[str, Any]:
    total_sessions = _q1(conn, "SELECT COUNT(*) AS n FROM daily_sessions;")
    total_decisions = _q1(conn, "SELECT COUNT(*) AS n FROM policy_decisions;")
    total_outcomes = _q1(conn, "SELECT COUNT(*) AS n FROM policy_outcomes;")
    raw_eval_rows = _q1(
        conn,
        """
        SELECT COUNT(*) AS n
        FROM policy_decisions d
        JOIN policy_outcomes o ON o.decision_id = d.decision_id
        WHERE d.propensities_json IS NOT NULL AND o.total_reward IS NOT NULL;
        """,
    )
    offline_eval: Optional[Dict[str, Any]] = None
    if policy_root:
        try:
            from questions_agent_platform.policy.offline_eval import evaluate_sqlite

            offline_eval = evaluate_sqlite(
                db_path=db_path,
                policy_root=policy_root,
                policy_version=None,
                min_date=None,
                max_date=None,
                include_shadow=False,
            )
        except Exception as exc:
            offline_eval = {"error": str(exc)}
    eval_rows = int(offline_eval.get("n_eval_rows", 0)) if isinstance(offline_eval, dict) and "n_eval_rows" in offline_eval else int(raw_eval_rows)

    mode_rows = _qa(
        conn,
        """
        SELECT selection_mode, mode, COUNT(*) AS n
        FROM policy_decisions
        GROUP BY selection_mode, mode
        ORDER BY n DESC;
        """,
    )
    mode_counts = [
        {"selection_mode": str(r["selection_mode"]), "mode": str(r["mode"]), "count": int(r["n"])}
        for r in mode_rows
    ]

    outcome_rows = _qa(
        conn,
        """
        SELECT
          d.selection_mode AS selection_mode,
          d.mode AS mode,
          COUNT(*) AS n,
          AVG(o.completion_rate) AS completion_rate_avg,
          AVG(o.response_time_ms_median) AS response_time_ms_median_avg,
          AVG(o.total_reward) AS total_reward_avg,
          AVG(o.uncertainty_before_mean - o.uncertainty_after_mean) AS uncertainty_reduction_avg,
          AVG(o.z_delta_norm) AS z_delta_norm_avg
        FROM policy_decisions d
        JOIN policy_outcomes o ON o.decision_id = d.decision_id
        GROUP BY d.selection_mode, d.mode
        ORDER BY n DESC;
        """,
    )
    outcome_summary = [
        {
            "selection_mode": str(r["selection_mode"]),
            "mode": str(r["mode"]),
            "n": int(r["n"]),
            "completion_rate_avg": _f(r["completion_rate_avg"]),
            "response_time_ms_median_avg": _f(r["response_time_ms_median_avg"]),
            "total_reward_avg": _f(r["total_reward_avg"]),
            "uncertainty_reduction_avg": _f(r["uncertainty_reduction_avg"]),
            "z_delta_norm_avg": _f(r["z_delta_norm_avg"]),
        }
        for r in outcome_rows
    ]

    checks = _run_invariant_and_explainability_checks(conn)
    decision_count = int(total_decisions)
    outcome_coverage = (float(total_outcomes) / float(decision_count)) if decision_count > 0 else 0.0
    explainability_coverage = (
        float(checks["explainability_complete_count"]) / float(decision_count)
        if decision_count > 0
        else 0.0
    )
    candidate_snapshot_coverage = _f(checks.get("candidate_set_snapshot_coverage"))
    context_snapshot_coverage = _f(checks.get("context_snapshot_coverage"))

    gates = {
        "safety_no_violations": checks["safety_violation_total"] == 0,
        "outcome_coverage_ge_80pct": outcome_coverage >= 0.8 if decision_count > 0 else False,
        "explainability_ge_95pct": explainability_coverage >= 0.95 if decision_count > 0 else False,
        "candidate_snapshot_coverage_ge_95pct": candidate_snapshot_coverage >= 0.95 if decision_count > 0 else False,
        "context_snapshot_coverage_ge_80pct": context_snapshot_coverage >= 0.8 if decision_count > 0 else False,
        "offline_eval_sample_ge_threshold": int(eval_rows) >= int(min_eval_rows),
        "offline_eval_dr_not_worse_than_baseline": _dr_not_worse(offline_eval),
        "offline_eval_segments_min_rows_ge_threshold": _segments_min_rows_ok(offline_eval, min_segment_rows=int(min_segment_rows)),
        "offline_eval_segments_dr_not_worse_than_baseline": _segments_dr_not_worse(offline_eval),
    }
    pass_count = sum(1 for ok in gates.values() if ok)

    if decision_count == 0:
        verdict = "NO_POLICY_DATA_YET"
    elif checks["safety_violation_total"] > 0:
        verdict = "BLOCKED_BY_SAFETY_ISSUES"
    elif pass_count == len(gates):
        verdict = "READY_FOR_POLICY_LIVE_ROLLOUT"
    else:
        verdict = "SHADOW_DATA_COLLECTION_IN_PROGRESS"

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "meta": {
            "label": label,
            "generated_at_utc": now_iso,
            "db_path": db_path,
            "verdict": verdict,
            "min_eval_rows": int(min_eval_rows),
            "min_segment_rows": int(min_segment_rows),
        },
        "coverage": {
            "daily_sessions": int(total_sessions),
            "policy_decisions": int(total_decisions),
            "policy_outcomes": int(total_outcomes),
            "offline_eval_rows": int(eval_rows),
            "offline_eval_rows_raw": int(raw_eval_rows),
            "mode_counts": mode_counts,
            "outcome_coverage_ratio": _f(outcome_coverage),
            "candidate_set_snapshot_coverage_ratio": _f(candidate_snapshot_coverage),
            "context_snapshot_coverage_ratio": _f(context_snapshot_coverage),
        },
        "offline_eval": offline_eval,
        "quality_gates": gates,
        "checks": checks,
        "outcome_summary": outcome_summary,
        "next_actions": _recommend_next_actions(
            verdict=verdict,
            gates=gates,
            decision_count=decision_count,
            eval_rows=int(eval_rows),
            min_eval_rows=int(min_eval_rows),
            min_segment_rows=int(min_segment_rows),
            offline_eval=offline_eval,
        ),
    }


def _run_invariant_and_explainability_checks(conn: sqlite3.Connection) -> Dict[str, Any]:
    rows = _qa(
        conn,
        """
        SELECT
          decision_id,
          selected_item_ids_json,
          candidate_set_json,
          explanations_json,
          context_json
        FROM policy_decisions;
        """,
    )
    return evaluate_policy_decision_rows(rows)


def _recommend_next_actions(
    *,
    verdict: str,
    gates: Dict[str, bool],
    decision_count: int,
    eval_rows: int,
    min_eval_rows: int,
    min_segment_rows: int,
    offline_eval: Optional[Dict[str, Any]],
) -> List[str]:
    if verdict == "NO_POLICY_DATA_YET":
        return [
            "Run selection in policy_shadow mode to start logging decisions and outcomes.",
            "Keep policy_log_context_snapshot and policy_log_candidate_set_snapshot enabled.",
            f"After >={int(min_eval_rows)} valid eval rows, run offline IPS/DR and compare against deterministic baseline.",
        ]
    actions: List[str] = []
    if not gates.get("safety_no_violations", False):
        actions.append("Block rollout and fix constraint enforcement violations before any live traffic.")
    if not gates.get("outcome_coverage_ge_80pct", False):
        actions.append("Increase policy outcome attachment coverage to at least 80% of decisions.")
    if not gates.get("explainability_ge_95pct", False):
        actions.append("Fix explanation payload completeness for selected items.")
    if not gates.get("candidate_snapshot_coverage_ge_95pct", False):
        actions.append("Enable candidate_set snapshot logging and backfill collection until coverage reaches 95%.")
    if not gates.get("context_snapshot_coverage_ge_80pct", False):
        actions.append("Increase context snapshot logging coverage to at least 80% for reproducible offline evaluation.")
    if not gates.get("offline_eval_sample_ge_threshold", False):
        actions.append(f"Collect more exploration/eval rows (current={eval_rows}, target>={int(min_eval_rows)}).")
    if not gates.get("offline_eval_dr_not_worse_than_baseline", False):
        actions.append("Policy DR estimate is below baseline. Tune reward/features or hold policy rollout.")
    if not gates.get("offline_eval_segments_min_rows_ge_threshold", False):
        actions.append(
            f"Collect more segment data until each required segment has >= {int(min_segment_rows)} eval rows."
        )
    if not gates.get("offline_eval_segments_dr_not_worse_than_baseline", False):
        actions.append("At least one segment has DR below baseline. Tune per-segment features/rewards before rollout.")
    if isinstance(offline_eval, dict) and offline_eval.get("error"):
        actions.append(f"Offline eval failed: {offline_eval.get('error')}")
    if not actions:
        actions.append("Proceed with gradual live rollout: 1% -> 5% -> 20% with automatic rollback guards.")
    if decision_count < 300:
        actions.append("Continue shadow accumulation to improve statistical confidence by segment.")
    return actions


def _dr_not_worse(offline_eval: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(offline_eval, dict):
        return False
    if offline_eval.get("error"):
        return False
    dr_policy = offline_eval.get("dr_policy")
    dr_baseline = offline_eval.get("dr_baseline")
    if dr_policy is None or dr_baseline is None:
        return False
    try:
        return float(dr_policy) >= float(dr_baseline) - 1e-9
    except Exception:
        return False


def _segments_min_rows_ok(offline_eval: Optional[Dict[str, Any]], *, min_segment_rows: int) -> bool:
    if not isinstance(offline_eval, dict):
        return False
    segments = offline_eval.get("segments")
    if not isinstance(segments, dict) or not segments:
        return False
    threshold = int(max(1, min_segment_rows))
    for payload in segments.values():
        if not isinstance(payload, dict):
            return False
        if int(payload.get("n_eval_rows", 0)) < threshold:
            return False
    return True


def _segments_dr_not_worse(offline_eval: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(offline_eval, dict):
        return False
    segments = offline_eval.get("segments")
    if not isinstance(segments, dict) or not segments:
        return False
    for payload in segments.values():
        if not isinstance(payload, dict):
            return False
        dr_policy = payload.get("dr_policy")
        dr_baseline = payload.get("dr_baseline")
        if dr_policy is None or dr_baseline is None:
            return False
        try:
            if float(dr_policy) < float(dr_baseline) - 1e-9:
                return False
        except Exception:
            return False
    return True


def _render_html(report: Dict[str, Any]) -> str:
    meta = report["meta"]
    coverage = report["coverage"]
    checks = report["checks"]
    gates = report["quality_gates"]
    outcome_rows = report["outcome_summary"]
    actions = report["next_actions"]
    offline_eval = report.get("offline_eval") if isinstance(report.get("offline_eval"), dict) else {}

    gate_items = "".join(
        f"<li><strong>{name}</strong>: {'PASS' if ok else 'FAIL'}</li>"
        for name, ok in gates.items()
    )
    mode_items = "".join(
        f"<li>{r['selection_mode']} / {r['mode']}: {r['count']}</li>"
        for r in coverage["mode_counts"]
    )
    outcome_table_rows = "".join(
        "<tr>"
        f"<td>{r['selection_mode']}</td>"
        f"<td>{r['mode']}</td>"
        f"<td>{r['n']}</td>"
        f"<td>{_fmt(r['completion_rate_avg'])}</td>"
        f"<td>{_fmt(r['response_time_ms_median_avg'])}</td>"
        f"<td>{_fmt(r['total_reward_avg'])}</td>"
        f"<td>{_fmt(r['uncertainty_reduction_avg'])}</td>"
        f"<td>{_fmt(r['z_delta_norm_avg'])}</td>"
        "</tr>"
        for r in outcome_rows
    )
    action_items = "".join(f"<li>{a}</li>" for a in actions)
    offline_eval_rows = ""
    offline_segment_rows = ""
    if offline_eval:
        ordered = [
            "policy_version",
            "n_eval_rows",
            "n_match_policy",
            "n_match_baseline",
            "ips_policy",
            "ips_baseline",
            "ips_delta_policy_minus_baseline",
            "dr_policy",
            "dr_baseline",
            "dr_delta_policy_minus_baseline",
            "notes",
            "error",
        ]
        keys = [k for k in ordered if k in offline_eval] + [k for k in offline_eval.keys() if k not in ordered]
        offline_eval_rows = "".join(
            f"<tr><th>{k}</th><td>{offline_eval.get(k)}</td></tr>"
            for k in keys
        )
        segments = offline_eval.get("segments")
        if isinstance(segments, dict):
            offline_segment_rows = "".join(
                "<tr>"
                f"<td>{name}</td>"
                f"<td>{payload.get('n_eval_rows')}</td>"
                f"<td>{_fmt(payload.get('dr_policy'))}</td>"
                f"<td>{_fmt(payload.get('dr_baseline'))}</td>"
                f"<td>{_fmt(payload.get('dr_delta_policy_minus_baseline'))}</td>"
                "</tr>"
                for name, payload in segments.items()
                if isinstance(payload, dict)
            )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{meta['label']}</title>
  <style>
    :root {{
      --bg: #0f1216;
      --panel: #171c22;
      --text: #e9eef5;
      --muted: #a4b2c3;
      --ok: #22c55e;
      --bad: #ef4444;
      --line: #2a3340;
      --accent: #5aa8ff;
    }}
    body {{
      margin: 0; padding: 24px;
      background: linear-gradient(180deg, #0f1216, #131922 32%, #10151d);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
    }}
    h1, h2 {{ margin: 0 0 10px 0; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 20px; color: var(--accent); }}
    .meta {{ color: var(--muted); margin-bottom: 20px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
    }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 10px;
      border: 1px solid var(--line);
      font-size: 12px;
      margin-bottom: 10px;
    }}
    .ok {{ color: var(--ok); }}
    .bad {{ color: var(--bad); }}
    ul {{ margin: 8px 0 0 18px; }}
    li {{ margin: 6px 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 6px 4px;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .wide {{ grid-column: 1 / -1; }}
  </style>
</head>
<body>
  <h1>{meta['label']}</h1>
  <div class="meta">
    Verdict: <strong>{meta['verdict']}</strong> |
    Generated: {meta['generated_at_utc']} |
    DB: {meta['db_path']}
  </div>

  <div class="grid">
    <section class="card">
      <h2>Coverage</h2>
      <ul>
        <li>daily_sessions: {coverage['daily_sessions']}</li>
        <li>policy_decisions: {coverage['policy_decisions']}</li>
        <li>policy_outcomes: {coverage['policy_outcomes']}</li>
        <li>offline_eval_rows: {coverage['offline_eval_rows']}</li>
        <li>offline_eval_rows_raw: {coverage['offline_eval_rows_raw']}</li>
        <li>outcome_coverage_ratio: {_fmt(coverage['outcome_coverage_ratio'])}</li>
        <li>candidate_snapshot_coverage_ratio: {_fmt(coverage['candidate_set_snapshot_coverage_ratio'])}</li>
        <li>context_snapshot_coverage_ratio: {_fmt(coverage['context_snapshot_coverage_ratio'])}</li>
      </ul>
      <div class="pill">Mode mix</div>
      <ul>{mode_items}</ul>
    </section>

    <section class="card">
      <h2>Quality Gates</h2>
      <ul>{gate_items}</ul>
    </section>

    <section class="card">
      <h2>Safety Checks</h2>
      <ul>
        <li>Total violations: {checks['safety_violation_total']}</li>
        <li>outside candidate set: {checks['violations']['selected_outside_candidate_set']}</li>
        <li>blocked selected: {checks['violations']['blocked_item_selected']}</li>
        <li>mandatory missing: {checks['violations']['mandatory_missing']}</li>
        <li>mandatory+blocked conflicts: {checks['violations']['mandatory_blocked_conflict']}</li>
        <li>wrong selected count: {checks['violations']['wrong_selected_count']}</li>
      </ul>
    </section>

    <section class="card">
      <h2>Explainability</h2>
      <ul>
        <li>decision_count: {checks['decision_count']}</li>
        <li>complete_payload_count: {checks['explainability_complete_count']}</li>
        <li>candidate_set_snapshot_count: {checks['candidate_set_snapshot_count']}</li>
        <li>context_snapshot_count: {checks['context_snapshot_count']}</li>
      </ul>
    </section>

    <section class="card wide">
      <h2>Outcome Summary</h2>
      <table>
        <thead>
          <tr>
            <th>selection_mode</th>
            <th>mode</th>
            <th>n</th>
            <th>completion</th>
            <th>resp_ms</th>
            <th>reward</th>
            <th>uncertainty_reduction</th>
            <th>z_delta_norm</th>
          </tr>
        </thead>
        <tbody>{outcome_table_rows}</tbody>
      </table>
    </section>

    <section class="card wide">
      <h2>Offline Eval (IPS/DR)</h2>
      <table>
        <tbody>{offline_eval_rows}</tbody>
      </table>
    </section>

    <section class="card wide">
      <h2>Offline Eval by Segment</h2>
      <table>
        <thead>
          <tr>
            <th>segment</th>
            <th>n_eval_rows</th>
            <th>dr_policy</th>
            <th>dr_baseline</th>
            <th>dr_delta</th>
          </tr>
        </thead>
        <tbody>{offline_segment_rows}</tbody>
      </table>
    </section>

    <section class="card wide">
      <h2>Recommended Next Actions</h2>
      <ul>{action_items}</ul>
    </section>
  </div>
</body>
</html>
"""


def _q1(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    if row is None:
        return 0
    return int(row["n"] or 0)


def _qa(conn: sqlite3.Connection, sql: str) -> List[sqlite3.Row]:
    return conn.execute(sql).fetchall()


def _f(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _fmt(v: Any) -> str:
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "0.0000"
