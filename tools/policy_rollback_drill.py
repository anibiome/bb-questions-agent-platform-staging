from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def run_rollback_drill(
    *,
    base_url: str,
    api_key: str,
    fallback_version: Optional[str] = None,
    restore_original: bool = True,
) -> Dict[str, Any]:
    root = str(base_url).rstrip("/")
    versions_before = _http_json("GET", f"{root}/v1/admin/policy/versions", api_key=api_key)
    active_before = str(versions_before.get("active_version") or "")
    versions_raw = versions_before.get("versions")
    versions = [str(v) for v in versions_raw] if isinstance(versions_raw, list) else []
    fallback = str(fallback_version or "").strip()
    if not fallback:
        fallback = next((v for v in versions if v != active_before), "")
    if not active_before:
        raise RuntimeError("Active policy version is empty")
    if not fallback:
        raise RuntimeError("No fallback policy version available for rollback drill")
    if fallback == active_before:
        raise RuntimeError("Fallback version must differ from active version")

    steps: List[Dict[str, Any]] = []
    activate_fallback_resp = _http_json(
        "POST",
        f"{root}/v1/admin/policy/activate/{urllib.parse.quote(fallback)}",
        api_key=api_key,
    )
    steps.append(
        {
            "step": "activate_fallback",
            "target_version": fallback,
            "response": activate_fallback_resp,
        }
    )
    verify_after_fallback = _http_json("GET", f"{root}/v1/admin/policy/versions", api_key=api_key)
    active_after_fallback = str(verify_after_fallback.get("active_version") or "")
    steps.append(
        {
            "step": "verify_fallback_active",
            "expected": fallback,
            "actual": active_after_fallback,
            "pass": active_after_fallback == fallback,
        }
    )
    if active_after_fallback != fallback:
        return {
            "generated_at_utc": _now_iso(),
            "status": "fail",
            "active_before": active_before,
            "fallback_version": fallback,
            "steps": steps,
        }

    active_final = active_after_fallback
    if restore_original:
        restore_resp = _http_json(
            "POST",
            f"{root}/v1/admin/policy/activate/{urllib.parse.quote(active_before)}",
            api_key=api_key,
        )
        steps.append(
            {
                "step": "restore_original",
                "target_version": active_before,
                "response": restore_resp,
            }
        )
        verify_restore = _http_json("GET", f"{root}/v1/admin/policy/versions", api_key=api_key)
        active_final = str(verify_restore.get("active_version") or "")
        steps.append(
            {
                "step": "verify_original_restored",
                "expected": active_before,
                "actual": active_final,
                "pass": active_final == active_before,
            }
        )

    success = bool(active_after_fallback == fallback and (not restore_original or active_final == active_before))
    return {
        "generated_at_utc": _now_iso(),
        "status": "pass" if success else "fail",
        "active_before": active_before,
        "fallback_version": fallback,
        "restore_original": bool(restore_original),
        "active_final": active_final,
        "steps": steps,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Questions Agent Rollback Drill")
    lines.append("")
    lines.append(f"- Generated at: `{report.get('generated_at_utc')}`")
    lines.append(f"- Status: **{str(report.get('status', '')).upper()}**")
    lines.append(f"- Active before drill: `{report.get('active_before')}`")
    lines.append(f"- Fallback tested: `{report.get('fallback_version')}`")
    lines.append(f"- Active after drill: `{report.get('active_final')}`")
    lines.append("")
    lines.append("## Steps")
    for step in report.get("steps", []):
        name = str(step.get("step") or "")
        if "pass" in step:
            mark = "✅" if bool(step.get("pass")) else "❌"
            lines.append(f"- {mark} `{name}` expected `{step.get('expected')}` got `{step.get('actual')}`")
        else:
            lines.append(f"- ✅ `{name}`")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run policy rollback activation drill via admin API.")
    parser.add_argument("--base-url", type=str, required=True, help="Prod API base URL")
    parser.add_argument("--api-key", type=str, required=True, help="X-API-Key")
    parser.add_argument("--fallback-version", type=str, default=None, help="Optional explicit rollback target version")
    parser.add_argument("--skip-restore", action="store_true", help="Do not restore original active version at the end")
    parser.add_argument("--out-dir", type=str, default="questions_agent_platform/output/policy_release")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args(argv)

    report = run_rollback_drill(
        base_url=str(args.base_url),
        api_key=str(args.api_key),
        fallback_version=args.fallback_version,
        restore_original=not bool(args.skip_restore),
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = out_dir / f"policy_rollback_drill_{stamp}.json"
    md_path = out_dir / f"policy_rollback_drill_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(str(json_path))
    print(str(md_path))
    if args.fail_on_error and str(report.get("status")) != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
