from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from questions_agent_platform.tools.policy_audit import evaluate_policy_decision_rows, parse_json_list


@dataclass(frozen=True)
class WeeklyReportResult:
    json_path: Path
    html_path: Path


def generate_weekly_policy_report(
    *,
    db_path: str,
    out_dir: str,
    label: str = "Questions Agent Weekly Policy Report",
    policy_root: Optional[str] = None,
    days: int = 7,
    min_eval_rows: int = 100,
    min_segment_rows: int = 30,
) -> WeeklyReportResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    end_day = date.today()
    window_days = int(max(1, days))
    start_day = end_day - timedelta(days=window_days - 1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        payload = _build_report(
            conn=conn,
            db_path=db_path,
            label=label,
            policy_root=policy_root,
            start_day=start_day,
            end_day=end_day,
            min_eval_rows=min_eval_rows,
            min_segment_rows=min_segment_rows,
        )
    finally:
        conn.close()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    stem = f"weekly_policy_report_{ts}"
    json_path = out / f"{stem}.json"
    html_path = out / f"{stem}.html"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(payload), encoding="utf-8")
    return WeeklyReportResult(json_path=json_path, html_path=html_path)


def _build_report(
    *,
    conn: sqlite3.Connection,
    db_path: str,
    label: str,
    policy_root: Optional[str],
    start_day: date,
    end_day: date,
    min_eval_rows: int,
    min_segment_rows: int,
) -> Dict[str, Any]:
    window = {"start_date": start_day.isoformat(), "end_date": end_day.isoformat(), "days": int((end_day - start_day).days + 1)}

    decisions_total = _q1(
        conn,
        """
        SELECT COUNT(*) AS n
        FROM policy_decisions
        WHERE date >= ? AND date <= ?;
        """,
        [start_day.isoformat(), end_day.isoformat()],
    )
    outcomes_total = _q1(
        conn,
        """
        SELECT COUNT(*) AS n
        FROM policy_outcomes o
        JOIN policy_decisions d ON d.decision_id = o.decision_id
        WHERE d.date >= ? AND d.date <= ?;
        """,
        [start_day.isoformat(), end_day.isoformat()],
    )
    safe_fallback_total = _q1(
        conn,
        """
        SELECT COUNT(*) AS n
        FROM policy_decisions
        WHERE date >= ? AND date <= ? AND mode = 'safe_fallback';
        """,
        [start_day.isoformat(), end_day.isoformat()],
    )

    mode_mix_rows = _qa(
        conn,
        """
        SELECT selection_mode, mode, COUNT(*) AS n
        FROM policy_decisions
        WHERE date >= ? AND date <= ?
        GROUP BY selection_mode, mode
        ORDER BY n DESC;
        """,
        [start_day.isoformat(), end_day.isoformat()],
    )
    mode_mix = [
        {"selection_mode": str(r["selection_mode"]), "mode": str(r["mode"]), "count": int(r["n"])}
        for r in mode_mix_rows
    ]

    daily_rows = _qa(
        conn,
        """
        SELECT
          d.date AS date,
          COUNT(DISTINCT d.decision_id) AS decision_count,
          COUNT(o.decision_id) AS outcome_count,
          AVG(o.completion_rate) AS completion_rate_avg,
          AVG(o.response_time_ms_median) AS burden_ms_avg,
          AVG(o.total_reward) AS reward_avg,
          AVG(o.uncertainty_before_mean - o.uncertainty_after_mean) AS uncertainty_reduction_avg
        FROM policy_decisions d
        LEFT JOIN policy_outcomes o ON o.decision_id = d.decision_id
        WHERE d.date >= ? AND d.date <= ?
        GROUP BY d.date
        ORDER BY d.date ASC;
        """,
        [start_day.isoformat(), end_day.isoformat()],
    )
    daily = [
        {
            "date": str(r["date"]),
            "decisions": int(r["decision_count"] or 0),
            "outcomes": int(r["outcome_count"] or 0),
            "completion_rate_avg": _f_or_none(r["completion_rate_avg"]),
            "burden_ms_avg": _f_or_none(r["burden_ms_avg"]),
            "reward_avg": _f_or_none(r["reward_avg"]),
            "uncertainty_reduction_avg": _f_or_none(r["uncertainty_reduction_avg"]),
        }
        for r in daily_rows
    ]

    reason_counts = _count_reason_codes(
        conn,
        start_day=start_day,
        end_day=end_day,
    )

    offline_eval: Optional[Dict[str, Any]] = None
    if policy_root:
        try:
            from questions_agent_platform.policy.offline_eval import evaluate_sqlite

            offline_eval = evaluate_sqlite(
                db_path=db_path,
                policy_root=policy_root,
                policy_version=None,
                min_date=start_day.isoformat(),
                max_date=end_day.isoformat(),
                include_shadow=False,
            )
        except Exception as exc:
            offline_eval = {"error": str(exc)}

    checks = _window_invariant_and_explainability_checks(conn, start_day=start_day, end_day=end_day)
    eval_rows = int(offline_eval.get("n_eval_rows", 0)) if isinstance(offline_eval, dict) and "n_eval_rows" in offline_eval else 0
    outcome_coverage = (float(outcomes_total) / float(decisions_total)) if decisions_total > 0 else 0.0
    explainability_coverage = (
        float(checks["explainability_complete_count"]) / float(decisions_total)
        if decisions_total > 0
        else 0.0
    )
    candidate_snapshot_coverage = _f_or_none(checks.get("candidate_set_snapshot_coverage")) or 0.0
    context_snapshot_coverage = _f_or_none(checks.get("context_snapshot_coverage")) or 0.0

    gates = {
        "safety_no_violations": checks["safety_violation_total"] == 0,
        "outcome_coverage_ge_80pct": outcome_coverage >= 0.8 if decisions_total > 0 else False,
        "explainability_ge_95pct": explainability_coverage >= 0.95 if decisions_total > 0 else False,
        "candidate_snapshot_coverage_ge_95pct": candidate_snapshot_coverage >= 0.95 if decisions_total > 0 else False,
        "context_snapshot_coverage_ge_80pct": context_snapshot_coverage >= 0.8 if decisions_total > 0 else False,
        "offline_eval_sample_ge_threshold": int(eval_rows) >= int(max(1, min_eval_rows)),
        "offline_eval_dr_not_worse_than_baseline": _dr_not_worse(offline_eval),
        "offline_eval_segments_min_rows_ge_threshold": _segments_min_rows_ok(offline_eval, min_segment_rows=max(1, int(min_segment_rows))),
        "offline_eval_segments_dr_not_worse_than_baseline": _segments_dr_not_worse(offline_eval),
    }
    pass_count = sum(1 for ok in gates.values() if ok)

    if decisions_total == 0:
        verdict = "NO_WEEKLY_POLICY_DATA"
    elif pass_count == len(gates):
        verdict = "WEEKLY_HEALTHY"
    else:
        verdict = "WEEKLY_ACTION_REQUIRED"

    return {
        "meta": {
            "label": label,
            "generated_at_utc": _now_iso(),
            "db_path": db_path,
            "verdict": verdict,
            "min_eval_rows": int(min_eval_rows),
            "min_segment_rows": int(min_segment_rows),
        },
        "window": window,
        "totals": {
            "decisions": int(decisions_total),
            "outcomes": int(outcomes_total),
            "safe_fallbacks": int(safe_fallback_total),
            "safe_fallback_rate": _f_or_none((float(safe_fallback_total) / float(decisions_total)) if decisions_total else None),
            "outcome_coverage_ratio": _f_or_none(outcome_coverage),
            "candidate_set_snapshot_coverage_ratio": _f_or_none(candidate_snapshot_coverage),
            "context_snapshot_coverage_ratio": _f_or_none(context_snapshot_coverage),
        },
        "daily": daily,
        "mode_mix": mode_mix,
        "top_reason_codes": reason_counts,
        "offline_eval": offline_eval,
        "quality_gates": gates,
        "checks": checks,
        "next_actions": _recommend_next_actions(verdict=verdict, gates=gates, eval_rows=eval_rows, min_eval_rows=int(min_eval_rows)),
    }


def _count_reason_codes(conn: sqlite3.Connection, *, start_day: date, end_day: date) -> List[Dict[str, Any]]:
    rows = _qa(
        conn,
        """
        SELECT explanations_json
        FROM policy_decisions
        WHERE date >= ? AND date <= ?;
        """,
        [start_day.isoformat(), end_day.isoformat()],
    )
    counts: Dict[str, int] = {}
    for row in rows:
        exps = parse_json_list(row["explanations_json"])
        for exp in exps:
            if not isinstance(exp, dict):
                continue
            reason_codes = exp.get("reason_codes")
            if not isinstance(reason_codes, list):
                continue
            for code in reason_codes:
                key = str(code)
                counts[key] = counts.get(key, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    return [{"reason_code": k, "count": int(v)} for k, v in top]


def _window_invariant_and_explainability_checks(conn: sqlite3.Connection, *, start_day: date, end_day: date) -> Dict[str, Any]:
    rows = _qa(
        conn,
        """
        SELECT decision_id, selected_item_ids_json, candidate_set_json, explanations_json, context_json
        FROM policy_decisions
        WHERE date >= ? AND date <= ?;
        """,
        [start_day.isoformat(), end_day.isoformat()],
    )
    return evaluate_policy_decision_rows(rows)


def _recommend_next_actions(*, verdict: str, gates: Dict[str, bool], eval_rows: int, min_eval_rows: int) -> List[str]:
    if verdict == "NO_WEEKLY_POLICY_DATA":
        return [
            "Run policy_shadow or policy_live this week to generate decision/outcome evidence.",
            "Keep context and candidate snapshots enabled for reproducible offline evaluation.",
        ]
    out: List[str] = []
    if not gates.get("safety_no_violations", False):
        out.append("Fix safety violations before increasing live traffic.")
    if not gates.get("outcome_coverage_ge_80pct", False):
        out.append("Increase outcome attachment coverage above 80%.")
    if not gates.get("explainability_ge_95pct", False):
        out.append("Backfill missing explanation payload fields.")
    if not gates.get("candidate_snapshot_coverage_ge_95pct", False):
        out.append("Increase candidate snapshot logging coverage to at least 95%.")
    if not gates.get("context_snapshot_coverage_ge_80pct", False):
        out.append("Increase context snapshot logging coverage to at least 80%.")
    if not gates.get("offline_eval_sample_ge_threshold", False):
        out.append(f"Collect more eval rows this week (current={int(eval_rows)}, target>={int(min_eval_rows)}).")
    if not gates.get("offline_eval_dr_not_worse_than_baseline", False):
        out.append("Hold rollout: weekly DR estimate underperforms deterministic baseline.")
    if not gates.get("offline_eval_segments_min_rows_ge_threshold", False):
        out.append("Add traffic/sample for sparse segments to satisfy segment-level power.")
    if not gates.get("offline_eval_segments_dr_not_worse_than_baseline", False):
        out.append("Tune features/reward where segment-level DR underperforms baseline.")
    if not out:
        out.append("Keep rollout schedule and continue daily monitoring of rollback guard.")
    return out


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


def _render_html(payload: Dict[str, Any]) -> str:
    meta = payload["meta"]
    window = payload["window"]
    totals = payload["totals"]
    checks = payload["checks"]
    gates = payload["quality_gates"]
    daily = payload["daily"]
    mode_mix = payload["mode_mix"]
    reasons = payload["top_reason_codes"]
    offline_eval = payload.get("offline_eval") if isinstance(payload.get("offline_eval"), dict) else {}
    actions = payload["next_actions"]

    kpi_cards = [
        ("Decisions", totals["decisions"]),
        ("Outcomes", totals["outcomes"]),
        ("Outcome Coverage", _fmt(totals["outcome_coverage_ratio"])),
        ("Candidate Snapshot Coverage", _fmt(totals["candidate_set_snapshot_coverage_ratio"])),
        ("Context Snapshot Coverage", _fmt(totals["context_snapshot_coverage_ratio"])),
        ("Safe Fallback Rate", _fmt(totals["safe_fallback_rate"])),
    ]
    kpi_html = "".join(
        f"<div class='kpi'><div class='k'>{name}</div><div class='v'>{value}</div></div>"
        for name, value in kpi_cards
    )
    gate_items = "".join(
        f"<li><strong>{name}</strong>: {'PASS' if ok else 'FAIL'}</li>"
        for name, ok in gates.items()
    )
    daily_rows = "".join(
        "<tr>"
        f"<td>{r['date']}</td>"
        f"<td>{r['decisions']}</td>"
        f"<td>{r['outcomes']}</td>"
        f"<td>{_fmt(r['completion_rate_avg'])}</td>"
        f"<td>{_fmt(r['burden_ms_avg'])}</td>"
        f"<td>{_fmt(r['reward_avg'])}</td>"
        f"<td>{_fmt(r['uncertainty_reduction_avg'])}</td>"
        "</tr>"
        for r in daily
    )
    mode_rows = "".join(
        "<tr>"
        f"<td>{r['selection_mode']}</td>"
        f"<td>{r['mode']}</td>"
        f"<td>{r['count']}</td>"
        "</tr>"
        for r in mode_mix
    )
    reason_rows = "".join(
        "<tr>"
        f"<td>{r['reason_code']}</td>"
        f"<td>{r['count']}</td>"
        "</tr>"
        for r in reasons
    )
    action_items = "".join(f"<li>{a}</li>" for a in actions)
    offline_eval_rows = ""
    if offline_eval:
        ordered = [
            "policy_version",
            "n_eval_rows",
            "dr_policy",
            "dr_baseline",
            "dr_delta_policy_minus_baseline",
            "error",
        ]
        keys = [k for k in ordered if k in offline_eval] + [k for k in offline_eval.keys() if k not in ordered]
        offline_eval_rows = "".join(
            f"<tr><th>{k}</th><td>{offline_eval.get(k)}</td></tr>"
            for k in keys
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
      --line: #2a3340;
      --accent: #5aa8ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
      background: linear-gradient(180deg, #0f1216, #131922 32%, #10151d);
    }}
    h1, h2 {{ margin: 0 0 10px 0; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 20px; color: var(--accent); }}
    .meta {{ color: var(--muted); margin-bottom: 18px; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .kpi {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
    }}
    .k {{ color: var(--muted); font-size: 12px; }}
    .v {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
    }}
    .wide {{ grid-column: 1 / -1; }}
    ul {{ margin: 8px 0 0 18px; }}
    li {{ margin: 5px 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 6px 4px;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    @media print {{
      body {{
        background: #ffffff;
        color: #111111;
        padding: 12mm;
      }}
      .card, .kpi {{
        background: #ffffff;
        border-color: #d1d5db;
      }}
      h2 {{ color: #1f2937; }}
    }}
  </style>
</head>
<body>
  <h1>{meta['label']}</h1>
  <div class="meta">
    Verdict: <strong>{meta['verdict']}</strong> |
    Window: {window['start_date']} to {window['end_date']} ({window['days']} days) |
    Generated: {meta['generated_at_utc']}
  </div>

  <div class="kpi-grid">{kpi_html}</div>

  <div class="grid">
    <section class="card">
      <h2>Quality Gates</h2>
      <ul>{gate_items}</ul>
    </section>

    <section class="card">
      <h2>Safety + Explainability</h2>
      <ul>
        <li>Safety violations: {checks['safety_violation_total']}</li>
        <li>Explainability complete: {checks['explainability_complete_count']} / {checks['decision_count']}</li>
        <li>Candidate snapshots: {checks['candidate_set_snapshot_count']} / {checks['decision_count']}</li>
        <li>Context snapshots: {checks['context_snapshot_count']} / {checks['decision_count']}</li>
      </ul>
    </section>

    <section class="card wide">
      <h2>Daily Trend</h2>
      <table>
        <thead>
          <tr>
            <th>date</th>
            <th>decisions</th>
            <th>outcomes</th>
            <th>completion</th>
            <th>burden_ms</th>
            <th>reward</th>
            <th>uncertainty_reduction</th>
          </tr>
        </thead>
        <tbody>{daily_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Mode Mix</h2>
      <table>
        <thead>
          <tr><th>selection_mode</th><th>mode</th><th>count</th></tr>
        </thead>
        <tbody>{mode_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Top Reason Codes</h2>
      <table>
        <thead>
          <tr><th>reason_code</th><th>count</th></tr>
        </thead>
        <tbody>{reason_rows}</tbody>
      </table>
    </section>

    <section class="card wide">
      <h2>Offline Eval (Window)</h2>
      <table><tbody>{offline_eval_rows}</tbody></table>
    </section>

    <section class="card wide">
      <h2>Recommended Actions</h2>
      <ul>{action_items}</ul>
    </section>
  </div>
</body>
</html>
"""


def _q1(conn: sqlite3.Connection, sql: str, params: Optional[List[Any]] = None) -> int:
    row = conn.execute(sql, params or []).fetchone()
    if row is None:
        return 0
    return int(row["n"] or 0)


def _qa(conn: sqlite3.Connection, sql: str, params: Optional[List[Any]] = None) -> List[sqlite3.Row]:
    return conn.execute(sql, params or []).fetchall()


def _f_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _fmt(v: Any) -> str:
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "-"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
