from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence


REQUIRED_EXPLANATION_FIELDS = frozenset(
    {"item_id", "policy_score", "deterministic_score", "reason_codes", "top_feature_contributions"}
)


def parse_json_obj(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    try:
        obj = json.loads(str(raw))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def parse_json_list(raw: Any) -> List[Any]:
    if raw is None:
        return []
    try:
        obj = json.loads(str(raw))
    except Exception:
        return []
    return obj if isinstance(obj, list) else []


def is_explainability_complete(*, selected: Sequence[str], explanations: Sequence[Any]) -> bool:
    if not isinstance(explanations, list):
        return False
    if len(explanations) != len(selected):
        return False

    by_item = {str(e.get("item_id")): e for e in explanations if isinstance(e, dict)}
    for item_id in selected:
        entry = by_item.get(str(item_id))
        if not isinstance(entry, dict):
            return False
        if not REQUIRED_EXPLANATION_FIELDS.issubset(set(entry.keys())):
            return False
        if not isinstance(entry.get("reason_codes"), list):
            return False
        if not isinstance(entry.get("top_feature_contributions"), list):
            return False
    return True


def evaluate_policy_decision_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    violations_outside_c = 0
    violations_blocked = 0
    violations_mandatory = 0
    conflicts_mandatory_blocked = 0
    violations_k = 0
    explainability_complete_count = 0

    candidate_set_snapshot_count = 0
    context_snapshot_count = 0

    for row in rows:
        selected = [str(x) for x in parse_json_list(_row_value(row, "selected_item_ids_json"))]
        candidate_obj = parse_json_obj(_row_value(row, "candidate_set_json"))
        explanations = parse_json_list(_row_value(row, "explanations_json"))
        context_obj = parse_json_obj(_row_value(row, "context_json"))

        if candidate_obj:
            candidate_set_snapshot_count += 1

            candidates = candidate_obj.get("candidates") if isinstance(candidate_obj.get("candidates"), list) else []
            candidate_ids = {str(c.get("item_id")) for c in candidates}
            blocked_ids = {
                str(c.get("item_id"))
                for c in candidates
                if isinstance(c.get("constraint_tags"), list) and "blocked" in c.get("constraint_tags")
            }
            mandatory_ids = {
                str(x)
                for x in (
                    candidate_obj.get("mandatory_item_ids")
                    if isinstance(candidate_obj.get("mandatory_item_ids"), list)
                    else []
                )
            }
            required_mandatory = mandatory_ids - blocked_ids
            expected_k = int(candidate_obj.get("k_core") or len(selected))
            selected_set = set(selected)

            if any(item_id not in candidate_ids for item_id in selected):
                violations_outside_c += 1
            if any(item_id in blocked_ids for item_id in selected):
                violations_blocked += 1
            if mandatory_ids & blocked_ids:
                conflicts_mandatory_blocked += 1
            if not required_mandatory.issubset(selected_set):
                violations_mandatory += 1
            if len(selected) != expected_k:
                violations_k += 1

        if context_obj:
            context_snapshot_count += 1

        if is_explainability_complete(selected=selected, explanations=explanations):
            explainability_complete_count += 1

    decision_count = int(len(rows))
    safety_violation_total = int(
        violations_outside_c + violations_blocked + violations_mandatory + conflicts_mandatory_blocked + violations_k
    )
    candidate_set_snapshot_missing_count = int(max(0, decision_count - candidate_set_snapshot_count))
    context_snapshot_missing_count = int(max(0, decision_count - context_snapshot_count))
    candidate_set_snapshot_coverage = (
        float(candidate_set_snapshot_count) / float(decision_count) if decision_count else 0.0
    )
    context_snapshot_coverage = float(context_snapshot_count) / float(decision_count) if decision_count else 0.0

    return {
        "decision_count": decision_count,
        "safety_violation_total": safety_violation_total,
        "violations": {
            "selected_outside_candidate_set": int(violations_outside_c),
            "blocked_item_selected": int(violations_blocked),
            "mandatory_missing": int(violations_mandatory),
            "mandatory_blocked_conflict": int(conflicts_mandatory_blocked),
            "wrong_selected_count": int(violations_k),
        },
        "explainability_complete_count": int(explainability_complete_count),
        "candidate_set_snapshot_count": int(candidate_set_snapshot_count),
        "candidate_set_snapshot_missing_count": int(candidate_set_snapshot_missing_count),
        "candidate_set_snapshot_coverage": float(candidate_set_snapshot_coverage),
        "context_snapshot_count": int(context_snapshot_count),
        "context_snapshot_missing_count": int(context_snapshot_missing_count),
        "context_snapshot_coverage": float(context_snapshot_coverage),
    }


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except Exception:
        return None
