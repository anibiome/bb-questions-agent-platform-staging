from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from questions_agent_platform.pipeline.anifold_adapter import AnifoldAdapterClient, AnifoldAdapterConfig


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run end-to-end smoke flow: context -> select -> answers -> projection -> evidence submit -> outcome callback."
    )
    parser.add_argument("--questions-base-url", type=str, required=True)
    parser.add_argument("--anifold-base-url", type=str, required=True)
    parser.add_argument("--user-id", type=str, default="smoke-user-1")
    parser.add_argument("--date", type=str, default=date.today().isoformat())
    parser.add_argument("--selection-mode", type=str, default="policy_shadow")
    parser.add_argument("--identity-mask-id", type=str, default="identity_mask_default")
    parser.add_argument("--questions-api-key", type=str, default="")
    parser.add_argument("--anifold-api-key", type=str, default="")
    parser.add_argument(
        "--questions-date-param-name",
        type=str,
        default="date_param",
        help="Use date_param for prod stack, date for reference stack.",
    )
    parser.add_argument("--skip-outcome-callback", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    qa_base = str(args.questions_base_url).rstrip("/")
    anifold_base = str(args.anifold_base_url).rstrip("/")
    user_id = str(args.user_id).strip()
    day = str(args.date).strip()
    selection_mode = str(args.selection_mode).strip() or "policy_shadow"
    identity_mask_id = str(args.identity_mask_id).strip() or None
    date_param_name = str(args.questions_date_param_name).strip() or "date_param"

    if not user_id:
        raise ValueError("--user-id is required")
    try:
        date.fromisoformat(day)
    except Exception as exc:
        raise ValueError("--date must be YYYY-MM-DD") from exc

    questions_headers = _headers(args.questions_api_key)
    adapter = AnifoldAdapterClient(
        AnifoldAdapterConfig(
            base_url=anifold_base,
            api_key=(str(args.anifold_api_key).strip() or None),
        )
    )

    timing_ms: Dict[str, float] = {}

    t0 = time.perf_counter()
    context = adapter.fetch_questions_context(user_id=user_id, day=day, identity_mask_id=identity_mask_id)
    timing_ms["fetch_context"] = (time.perf_counter() - t0) * 1000.0

    select_body = adapter.build_daily_select_request(
        day=day,
        context_payload=context,
        selection_mode=selection_mode,
        include_explanations=True,
        allow_context_batches=True,
        k_core=5,
        identity_mask_id=identity_mask_id,
    )

    t1 = time.perf_counter()
    select_payload = _http_json(
        "POST",
        f"{qa_base}/v1/users/{urllib.parse.quote(user_id)}/daily-questions/select",
        headers={**questions_headers, "Content-Type": "application/json"},
        payload=select_body,
    )
    timing_ms["daily_select"] = (time.perf_counter() - t1) * 1000.0
    session_id = str(select_payload.get("session_id") or "")
    if not session_id:
        raise RuntimeError("daily-questions/select did not return session_id")
    questions = select_payload.get("questions") or []
    if not isinstance(questions, list) or not questions:
        raise RuntimeError("daily-questions/select returned no questions")
    decision_id = _extract_decision_id(select_payload)

    answer_batch = _build_answer_batch(user_id=user_id, day=day, session_id=session_id, questions=questions)
    t2 = time.perf_counter()
    answers_payload = _http_json(
        "POST",
        f"{qa_base}/v1/users/{urllib.parse.quote(user_id)}/answers",
        headers={**questions_headers, "Content-Type": "application/json"},
        payload={"session_id": session_id, "answers": answer_batch},
    )
    timing_ms["submit_answers"] = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    projection_payload = _http_json(
        "GET",
        f"{qa_base}/v1/users/{urllib.parse.quote(user_id)}/projection/questions"
        f"?{urllib.parse.quote(date_param_name)}={urllib.parse.quote(day)}",
        headers=questions_headers,
    )
    timing_ms["fetch_projection"] = (time.perf_counter() - t3) * 1000.0

    t4 = time.perf_counter()
    evidence_result = adapter.submit_questions_evidence(
        user_id=user_id,
        day=day,
        evidence_payload=projection_payload,
        session_id=session_id,
        decision_id=decision_id,
    )
    timing_ms["submit_evidence_to_anifold"] = (time.perf_counter() - t4) * 1000.0

    outcome_callback_result: Optional[Dict[str, Any]] = None
    if not args.skip_outcome_callback and decision_id:
        outcome_payload = _build_outcome_payload(decision_id=decision_id, context=context)
        t5 = time.perf_counter()
        outcome_callback_result = _http_json(
            "POST",
            f"{qa_base}/v1/users/{urllib.parse.quote(user_id)}/policy/outcomes",
            headers={**questions_headers, "Content-Type": "application/json"},
            payload=outcome_payload,
        )
        timing_ms["outcome_callback_to_questions"] = (time.perf_counter() - t5) * 1000.0

    summary: Dict[str, Any] = {
        "ok": True,
        "user_id": user_id,
        "date": day,
        "selection_mode": selection_mode,
        "session_id": session_id,
        "decision_id": decision_id,
        "question_count": len(questions),
        "inserted_answer_events": int(answers_payload.get("inserted_answer_events") or 0),
        "new_scale_scores_count": len(answers_payload.get("new_scale_scores") or []),
        "projection_contract": ((projection_payload.get("contract") or {}).get("name")),
        "evidence_submit_status": evidence_result.get("status"),
        "outcome_callback_result": outcome_callback_result,
        "timing_ms": {k: round(float(v), 2) for k, v in timing_ms.items()},
    }

    encoded = json.dumps(summary, ensure_ascii=False, indent=2)
    print(encoded)
    if args.out:
        out_path = Path(str(args.out)).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
        print(str(out_path))
    return 0


def _headers(api_key: str) -> Dict[str, str]:
    key = str(api_key or "").strip()
    if not key:
        return {}
    return {"X-API-Key": key}


def _extract_decision_id(select_payload: Dict[str, Any]) -> Optional[str]:
    session_decision_id = select_payload.get("policy_decision_id")
    if isinstance(session_decision_id, str) and session_decision_id.strip():
        return session_decision_id.strip()
    explain = select_payload.get("selection_explain")
    if isinstance(explain, dict):
        policy = explain.get("policy")
        if isinstance(policy, dict):
            decision_id = policy.get("decision_id")
            if isinstance(decision_id, str) and decision_id.strip():
                return decision_id.strip()
    return None


def _build_answer_batch(*, user_id: str, day: str, session_id: str, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    answers: List[Dict[str, Any]] = []
    for idx, q in enumerate(questions):
        item_id = str(q.get("item_id") or "").strip()
        if not item_id:
            raise RuntimeError("question item missing item_id")
        value = _pick_answer_value(q)
        answers.append(
            {
                "client_event_id": f"{user_id}::{day}::{session_id}::{idx}::{item_id}",
                "item_id": item_id,
                "value": value,
                "answered_at": f"{day}T12:00:{idx:02d}Z",
            }
        )
    return answers


def _pick_answer_value(question: Dict[str, Any]) -> float:
    options = question.get("options")
    if isinstance(options, list) and options:
        numeric: List[float] = []
        for opt in options:
            if isinstance(opt, dict) and "value" in opt:
                try:
                    numeric.append(float(opt["value"]))
                except Exception:
                    pass
        if numeric:
            numeric_sorted = sorted(numeric)
            return float(numeric_sorted[len(numeric_sorted) // 2])
    return 1.0


def _build_outcome_payload(*, decision_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    z_before = list(context.get("anifold_z") or [])
    uncertainty_before = list(context.get("z_uncertainty_diag") or [])
    if z_before:
        z_after = [round(float(v) + 0.01, 6) for v in z_before]
    else:
        z_after = []
    if uncertainty_before:
        uncertainty_after = [max(0.0, round(float(v) - 0.01, 6)) for v in uncertainty_before]
    else:
        uncertainty_after = []
    body: Dict[str, Any] = {
        "contract_name": "fusion_to_questions_outcome",
        "schema_version": "1.0",
        "decision_id": str(decision_id),
    }
    if z_before:
        body["z_before"] = z_before
    if z_after:
        body["z_after"] = z_after
    if uncertainty_before:
        body["uncertainty_before_diag"] = uncertainty_before
    if uncertainty_after:
        body["uncertainty_after_diag"] = uncertainty_after
    return body


def _http_json(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url=url, method=method, headers=headers or {}, data=body)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"HTTP {exc.code} for {method} {url}: {detail}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
