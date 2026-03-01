"""
Registry validation tool — catches data issues before production.

Run via CLI:
    python -m questions_agent_platform.pipeline.cli --config config.json validate-registry

Checks performed:
  1. Parse integrity (valid JSON, required fields)
  2. Referential integrity (scale→item, scale→questionnaire)
  3. Scoring method validity
  4. Duplicate detection (item IDs, scale IDs)
  5. Multiplexing analysis (items shared across scales)
  6. Onboarding order continuity
  7. Timeframe coverage
  8. Intrusiveness distribution (ensures <50% are "high")
  9. Tag coverage (every item has at least one tag)
  10. Scale feasibility (min_items_required ≤ total items in scale)
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ValidationResult:
    errors: List[str]
    warnings: List[str]
    info: List[str]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_registry_dir(registry_root: str, version: str) -> ValidationResult:
    """Validate a registry version on disk."""
    version_dir = Path(registry_root) / "versions" / version
    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []

    # --- 1. File existence ---
    items_path = version_dir / "items.json"
    scales_path = version_dir / "scales.json"
    questionnaires_path = version_dir / "questionnaires.json"

    for p, name in [
        (items_path, "items.json"),
        (scales_path, "scales.json"),
        (questionnaires_path, "questionnaires.json"),
    ]:
        if not p.exists():
            errors.append(f"MISSING_FILE: {name} not found at {p}")

    if errors:
        return ValidationResult(errors=errors, warnings=warnings, info=info)

    # --- 2. Parse JSON ---
    try:
        items_raw = json.loads(items_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"PARSE_ERROR: items.json: {e}")
        items_raw = []

    try:
        scales_raw = json.loads(scales_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"PARSE_ERROR: scales.json: {e}")
        scales_raw = []

    try:
        questionnaires_raw = json.loads(questionnaires_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"PARSE_ERROR: questionnaires.json: {e}")
        questionnaires_raw = []

    if errors:
        return ValidationResult(errors=errors, warnings=warnings, info=info)

    # --- 3. Type checks ---
    if not isinstance(items_raw, list):
        errors.append("TYPE_ERROR: items.json must be a JSON array")
    if not isinstance(scales_raw, list):
        errors.append("TYPE_ERROR: scales.json must be a JSON array")
    if not isinstance(questionnaires_raw, list):
        errors.append("TYPE_ERROR: questionnaires.json must be a JSON array")
    if errors:
        return ValidationResult(errors=errors, warnings=warnings, info=info)

    # --- 4. Required fields ---
    item_ids: Set[str] = set()
    for idx, item in enumerate(items_raw):
        if not isinstance(item, dict):
            errors.append(f"ITEM[{idx}]: not a JSON object")
            continue
        for field in ("id", "text", "response_type"):
            if not item.get(field) or not isinstance(item.get(field), str):
                errors.append(f"ITEM[{idx}]: missing or empty required field '{field}'")
        iid = item.get("id", "")
        if iid in item_ids:
            errors.append(f"DUPLICATE_ITEM: '{iid}' appears more than once")
        item_ids.add(iid)

    questionnaire_ids: Set[str] = set()
    for idx, q in enumerate(questionnaires_raw):
        if not isinstance(q, dict):
            errors.append(f"QUESTIONNAIRE[{idx}]: not a JSON object")
            continue
        for field in ("id", "version", "name"):
            if not q.get(field) or not isinstance(q.get(field), str):
                errors.append(f"QUESTIONNAIRE[{idx}]: missing or empty required field '{field}'")
        qid = q.get("id", "")
        if qid in questionnaire_ids:
            errors.append(f"DUPLICATE_QUESTIONNAIRE: '{qid}' appears more than once")
        questionnaire_ids.add(qid)

    scale_ids: Set[str] = set()
    for idx, s in enumerate(scales_raw):
        if not isinstance(s, dict):
            errors.append(f"SCALE[{idx}]: not a JSON object")
            continue
        for field in ("id", "questionnaire_id", "version", "name", "response_type"):
            if not s.get(field) or not isinstance(s.get(field), str):
                errors.append(f"SCALE[{idx}]: missing or empty required field '{field}'")
        sid = s.get("id", "")
        if sid in scale_ids:
            errors.append(f"DUPLICATE_SCALE: '{sid}' appears more than once")
        scale_ids.add(sid)

    # --- 5. Referential integrity ---
    ALLOWED_SCORING_METHODS = {
        "sum", "mean", "instrument_findrisc", "instrument_ez_cvd",
        "instrument_scored", "instrument_lee_nafld", "instrument_ipaq_sf",
        "instrument_audit_c",
    }
    items_in_any_scale: Set[str] = set()

    for s in scales_raw:
        if not isinstance(s, dict):
            continue
        sid = s.get("id", "?")
        qid = s.get("questionnaire_id", "")
        if qid and qid not in questionnaire_ids:
            errors.append(f"SCALE '{sid}': references unknown questionnaire '{qid}'")

        scoring = s.get("scoring", {})
        method = scoring.get("method", "sum") if isinstance(scoring, dict) else "sum"
        if method not in ALLOWED_SCORING_METHODS:
            errors.append(f"SCALE '{sid}': invalid scoring method '{method}'")

        scale_items = s.get("items", [])
        if not isinstance(scale_items, list) or not scale_items:
            errors.append(f"SCALE '{sid}': items must be a non-empty list")
            continue

        min_req = s.get("min_items_required")
        if min_req is not None:
            try:
                min_req = int(min_req)
                if min_req < 1:
                    errors.append(f"SCALE '{sid}': min_items_required must be >= 1")
                if min_req > len(scale_items):
                    errors.append(
                        f"SCALE '{sid}': min_items_required ({min_req}) > total items ({len(scale_items)})"
                    )
            except (ValueError, TypeError):
                errors.append(f"SCALE '{sid}': min_items_required must be an integer")

        for si in scale_items:
            if not isinstance(si, dict):
                errors.append(f"SCALE '{sid}': item entry is not an object")
                continue
            si_id = si.get("item_id", "")
            if si_id not in item_ids:
                errors.append(f"SCALE '{sid}': references unknown item '{si_id}'")
            items_in_any_scale.add(si_id)

    # --- 6. Orphan items (not in any scale) ---
    orphans = item_ids - items_in_any_scale
    if orphans:
        warnings.append(
            f"ORPHAN_ITEMS: {len(orphans)} items not in any scale: "
            + ", ".join(sorted(orphans)[:10])
            + ("..." if len(orphans) > 10 else "")
        )

    # --- 7. Tag coverage ---
    items_without_tags = []
    all_tags: Set[str] = set()
    for item in items_raw:
        if not isinstance(item, dict):
            continue
        tags = item.get("tags", [])
        if not tags or not isinstance(tags, list) or all(not str(t).strip() for t in tags):
            items_without_tags.append(item.get("id", "?"))
        else:
            all_tags.update(str(t).strip() for t in tags if str(t).strip())
    if items_without_tags:
        warnings.append(
            f"MISSING_TAGS: {len(items_without_tags)} items have no tags: "
            + ", ".join(items_without_tags[:10])
            + ("..." if len(items_without_tags) > 10 else "")
        )

    # --- 8. Intrusiveness distribution ---
    intrusiveness_counts: Dict[str, int] = {}
    for item in items_raw:
        if not isinstance(item, dict):
            continue
        lvl = str(item.get("intrusiveness", "low"))
        intrusiveness_counts[lvl] = intrusiveness_counts.get(lvl, 0) + 1
    high_count = intrusiveness_counts.get("high", 0)
    total_items = len(items_raw)
    if total_items > 0 and high_count / total_items > 0.5:
        warnings.append(
            f"HIGH_INTRUSIVENESS: {high_count}/{total_items} items ({100*high_count/total_items:.0f}%) "
            f"are high-intrusiveness. Selection constraint limits to 1/session."
        )

    # --- 9. Onboarding order continuity ---
    onboarding_orders = []
    for item in items_raw:
        if not isinstance(item, dict):
            continue
        order = item.get("onboarding_order")
        if order is not None:
            try:
                onboarding_orders.append((int(order), item.get("id", "?")))
            except (ValueError, TypeError):
                warnings.append(
                    f"ONBOARDING_ORDER: item '{item.get('id', '?')}' has non-integer order: {order}"
                )
    if onboarding_orders:
        onboarding_orders.sort()
        orders = [o for o, _ in onboarding_orders]
        expected = list(range(orders[0], orders[0] + len(orders)))
        if orders != expected:
            warnings.append(
                f"ONBOARDING_ORDER: gaps or duplicates in sequence. "
                f"Found: {orders}, expected: {expected}"
            )

    # --- 10. Multiplexing ratio ---
    multiplex_count = {}
    for s in scales_raw:
        if not isinstance(s, dict):
            continue
        for si in s.get("items", []):
            if isinstance(si, dict):
                si_id = si.get("item_id", "")
                multiplex_count[si_id] = multiplex_count.get(si_id, 0) + 1
    if multiplex_count:
        total_scale_refs = sum(multiplex_count.values())
        unique_items_in_scales = len(multiplex_count)
        ratio = total_scale_refs / max(1, unique_items_in_scales)
        info.append(
            f"MULTIPLEXING: {unique_items_in_scales} unique items feed {total_scale_refs} "
            f"scale slots (ratio: {ratio:.2f}x)"
        )
        if ratio < 1.2:
            warnings.append(
                f"LOW_MULTIPLEXING: ratio {ratio:.2f}x — consider sharing items across scales "
                f"for better information density"
            )

    # --- 11. Timeframe analysis ---
    tf_counts: Dict[str, int] = {}
    for item in items_raw:
        if not isinstance(item, dict):
            continue
        for tf in item.get("timeframes_allowed", ["last_7_days"]):
            tf_counts[str(tf)] = tf_counts.get(str(tf), 0) + 1
    if tf_counts:
        info.append(f"TIMEFRAMES: {dict(sorted(tf_counts.items()))}")

    # --- Summary ---
    info.insert(0,
        f"REGISTRY v{version}: {len(items_raw)} items, {len(scales_raw)} scales, "
        f"{len(questionnaires_raw)} questionnaires, {len(all_tags)} unique tags"
    )

    return ValidationResult(errors=errors, warnings=warnings, info=info)


def print_validation_result(result: ValidationResult) -> None:
    """Print validation result in a structured format."""
    print(f"\n{'='*60}")
    print(f"REGISTRY VALIDATION REPORT")
    print(f"{'='*60}")

    if result.info:
        for msg in result.info:
            print(f"  [INFO] {msg}")

    if result.warnings:
        print(f"\n  WARNINGS ({len(result.warnings)}):")
        for msg in result.warnings:
            print(f"  [WARN] {msg}")

    if result.errors:
        print(f"\n  ERRORS ({len(result.errors)}):")
        for msg in result.errors:
            print(f"  [ERR]  {msg}")

    print(f"\n{'='*60}")
    if result.ok:
        print("  RESULT: PASS")
    else:
        print(f"  RESULT: FAIL ({len(result.errors)} errors)")
    print(f"{'='*60}\n")


def run_validate_registry(registry_root: str, version: Optional[str] = None) -> ValidationResult:
    """Validate a registry, optionally specifying version."""
    from questions_agent_platform.pipeline.registry import list_versions

    if version:
        versions = [version]
    else:
        versions = list_versions(registry_root)
        if not versions:
            return ValidationResult(
                errors=[f"NO_VERSIONS: no registry versions found in {registry_root}"],
                warnings=[],
                info=[],
            )

    combined = ValidationResult(errors=[], warnings=[], info=[])
    for v in versions:
        result = validate_registry_dir(registry_root, v)
        combined.errors.extend(result.errors)
        combined.warnings.extend(result.warnings)
        combined.info.extend(result.info)

    return combined
