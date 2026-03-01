from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from questions_agent_platform.tools.generate_policy_brief import generate_policy_brief


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _http_json(method: str, url: str, *, api_key: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    headers = {"X-API-Key": str(api_key)}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, method=method.upper(), headers=headers, data=body)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"HTTP {exc.code} for {method} {url}: {detail}") from exc


def build_policy_release_report(
    *,
    candidate_version: str,
    gate_report: Dict[str, Any],
    policy_versions: Dict[str, Any],
    policy_metrics: Dict[str, Any],
    slo_report: Dict[str, Any],
) -> Dict[str, Any]:
    quality_gates = gate_report.get("quality_gates") if isinstance(gate_report.get("quality_gates"), dict) else {}
    quality_gate_failures = [name for name, ok in quality_gates.items() if not bool(ok)]
    candidate = str(candidate_version).strip()
    active = str(policy_versions.get("active_version") or "")
    versions_raw = policy_versions.get("versions")
    versions = [str(v) for v in versions_raw] if isinstance(versions_raw, list) else []
    rollback_guard = policy_metrics.get("rollback_guard") if isinstance(policy_metrics.get("rollback_guard"), dict) else {}
    rollback_required = bool(rollback_guard.get("rollback", False))

    checks: List[Dict[str, Any]] = [
        {
            "id": "candidate_exists",
            "label": "Candidate version exists in registry",
            "pass": candidate in versions,
            "detail": {"candidate_version": candidate, "known_versions": versions},
        },
        {
            "id": "candidate_not_already_active",
            "label": "Candidate version is not already active",
            "pass": bool(candidate) and candidate != active,
            "detail": {"candidate_version": candidate, "active_version": active},
        },
        {
            "id": "quality_gates_pass",
            "label": "Offline quality gates pass",
            "pass": len(quality_gate_failures) == 0 and bool(quality_gates),
            "detail": {"failed_gates": quality_gate_failures, "quality_gates": quality_gates},
        },
        {
            "id": "slo_status_ok",
            "label": "Operational SLO status is healthy",
            "pass": str(slo_report.get("status")) == "ok",
            "detail": {"slo_status": slo_report.get("status"), "alerts": slo_report.get("alerts")},
        },
        {
            "id": "rollback_guard_clear",
            "label": "Rollback guard is clear for live policy mode",
            "pass": not rollback_required,
            "detail": {"rollback_guard": rollback_guard},
        },
    ]
    blocked_reasons = [check["id"] for check in checks if not bool(check["pass"])]
    ready = len(blocked_reasons) == 0
    return {
        "generated_at_utc": _now_iso(),
        "candidate_version": candidate,
        "active_version": active,
        "ready_for_promotion": ready,
        "blocked_reasons": blocked_reasons,
        "checks": checks,
        "evidence": {
            "quality_gates": quality_gates,
            "slo_status": slo_report.get("status"),
            "slo_alerts": slo_report.get("alerts"),
            "rollback_guard": rollback_guard,
            "policy_metrics_window": policy_metrics.get("window"),
            "policy_decisions": policy_metrics.get("decisions"),
            "policy_outcomes": policy_metrics.get("outcomes"),
        },
        "next_actions": _next_actions(ready=ready, blocked_reasons=blocked_reasons, candidate=candidate),
    }


def _next_actions(*, ready: bool, blocked_reasons: List[str], candidate: str) -> List[str]:
    if ready:
        return [
            f"Promote policy version `{candidate}` with gradual rollout gates (1% -> 5% -> 20% -> 50%).",
            "Monitor /v1/admin/policy/metrics and /v1/admin/slo during each ramp phase.",
            "Keep rollback drill artifacts attached to the release record.",
        ]
    actions: List[str] = []
    if "candidate_exists" in blocked_reasons:
        actions.append("Publish candidate artifact into policy registry before promotion.")
    if "candidate_not_already_active" in blocked_reasons:
        actions.append("Select a non-active candidate version for promotion.")
    if "quality_gates_pass" in blocked_reasons:
        actions.append("Fix policy quality gates before rollout (use policy brief report failures).")
    if "slo_status_ok" in blocked_reasons:
        actions.append("Resolve active SLO alerts before policy promotion.")
    if "rollback_guard_clear" in blocked_reasons:
        actions.append("Hold rollout; rollback guard currently indicates degraded live performance.")
    if not actions:
        actions.append("Investigate unknown blocker in release checklist.")
    return actions


def render_markdown(report: Dict[str, Any]) -> str:
    rows = []
    rows.append("# Questions Agent Policy Promotion Checklist")
    rows.append("")
    rows.append(f"- Generated at: `{report.get('generated_at_utc')}`")
    rows.append(f"- Candidate version: `{report.get('candidate_version')}`")
    rows.append(f"- Active version: `{report.get('active_version')}`")
    rows.append(f"- Ready for promotion: **{'YES' if report.get('ready_for_promotion') else 'NO'}**")
    rows.append("")
    rows.append("## Checklist")
    for check in report.get("checks", []):
        mark = "✅" if bool(check.get("pass")) else "❌"
        rows.append(f"- {mark} `{check.get('id')}` — {check.get('label')}")
    rows.append("")
    rows.append("## Next Actions")
    for action in report.get("next_actions", []):
        rows.append(f"- {action}")
    rows.append("")
    return "\n".join(rows).strip() + "\n"


def _load_gate_report(
    *,
    gate_report_json: Optional[str],
    db_path: Optional[str],
    policy_root: Optional[str],
    out_dir: str,
    min_eval_rows: int,
    min_segment_rows: int,
) -> Dict[str, Any]:
    if gate_report_json:
        return json.loads(Path(gate_report_json).expanduser().read_text(encoding="utf-8"))
    if not db_path:
        raise ValueError("Provide --gate-report-json or --db to build gate report")
    result = generate_policy_brief(
        db_path=str(db_path),
        out_dir=str(out_dir),
        label="Questions Agent Policy Gate",
        policy_root=policy_root,
        min_eval_rows=int(min_eval_rows),
        min_segment_rows=int(min_segment_rows),
    )
    return json.loads(Path(result.json_path).read_text(encoding="utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Policy release promotion checklist + optional activation.")
    parser.add_argument("--candidate-version", type=str, required=True)
    parser.add_argument("--base-url", type=str, required=True, help="Prod API base URL")
    parser.add_argument("--api-key", type=str, required=True, help="X-API-Key")
    parser.add_argument("--gate-report-json", type=str, default=None, help="Existing policy gate JSON report")
    parser.add_argument("--db", type=str, default=None, help="SQLite DB path for generating gate report if JSON not provided")
    parser.add_argument("--policy-root", type=str, default=None, help="Policy root for offline evaluation during gate report generation")
    parser.add_argument("--out-dir", type=str, default="questions_agent_platform/output/policy_release")
    parser.add_argument("--min-eval-rows", type=int, default=100)
    parser.add_argument("--min-segment-rows", type=int, default=30)
    parser.add_argument("--activate-if-ready", action="store_true")
    parser.add_argument("--fail-on-block", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = str(args.base_url).rstrip("/")
    api_key = str(args.api_key)

    gate_report = _load_gate_report(
        gate_report_json=args.gate_report_json,
        db_path=args.db,
        policy_root=args.policy_root,
        out_dir=str(out_dir),
        min_eval_rows=int(args.min_eval_rows),
        min_segment_rows=int(args.min_segment_rows),
    )
    policy_versions = _http_json("GET", f"{base_url}/v1/admin/policy/versions", api_key=api_key)
    policy_metrics = _http_json(
        "GET",
        f"{base_url}/v1/admin/policy/metrics?{urllib.parse.urlencode({'days': 30})}",
        api_key=api_key,
    )
    slo_report = _http_json(
        "GET",
        f"{base_url}/v1/admin/slo?{urllib.parse.urlencode({'days': 14})}",
        api_key=api_key,
    )
    report = build_policy_release_report(
        candidate_version=args.candidate_version,
        gate_report=gate_report,
        policy_versions=policy_versions,
        policy_metrics=policy_metrics,
        slo_report=slo_report,
    )

    activation: Dict[str, Any] = {"attempted": False}
    if args.activate_if_ready and bool(report.get("ready_for_promotion")):
        activation["attempted"] = True
        activation["response"] = _http_json(
            "POST",
            f"{base_url}/v1/admin/policy/activate/{urllib.parse.quote(str(args.candidate_version))}",
            api_key=api_key,
        )
    report["activation"] = activation

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = out_dir / f"policy_release_checklist_{stamp}.json"
    md_path = out_dir / f"policy_release_checklist_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(str(json_path))
    print(str(md_path))
    if args.fail_on_block and not bool(report.get("ready_for_promotion")):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
