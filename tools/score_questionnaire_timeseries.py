"""Score questionnaire replay CSV into per-session longitudinal scale rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from questions_agent_platform.pipeline.registry import Registry, validate_registry_bundle
from questions_agent_platform.pipeline.scoring import compute_scale_score

_REGISTRY_FILES: Tuple[str, ...] = ("items.json", "questionnaires.json", "scales.json")
_FIELDNAMES: Tuple[str, ...] = (
    "access_code",
    "user_id",
    "session_date",
    "scale_id",
    "scale_name",
    "questionnaire_id",
    "scale_method",
    "citation",
    "raw_score",
    "normalized_score",
    "answered_count",
    "items_required",
    "confidence_tier",
    "risk_tier",
    "delta_vs_prev",
    "first_answer_timestamp",
    "last_answer_timestamp",
    "source_row_count",
)


@dataclass
class _SessionScaleState:
    answers: MutableMapping[str, Tuple[str, float]]
    access_code: str = ""
    row_count: int = 0


def score_questionnaire_timeseries(
    *,
    input_csv: str,
    registry_source: str,
    out_dir: str,
    date_column: str = "scan_date_local",
) -> Dict[str, Any]:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if date_column not in {"scan_date_local", "scan_date_utc"}:
        raise ValueError(f"Unsupported date_column: {date_column}")

    registry = _load_registry_bundle(Path(registry_source))
    scales_by_questionnaire: Dict[str, List[Any]] = defaultdict(list)
    for scale in registry.scales.values():
        scales_by_questionnaire[str(scale.questionnaire_id)].append(scale)

    session_scale_rows: Dict[Tuple[str, str, str], _SessionScaleState] = {}
    row_counter: Counter[str] = Counter()
    rows_read = 0

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_read += 1
            user_id = str(row.get("user_id") or "").strip()
            session_date = str(row.get(date_column) or "").strip()
            questionnaire_id = str(row.get("target_questionnaire_id") or "").strip()
            item_id = str(row.get("question_id") or "").strip()
            answer_value_raw = str(row.get("answer_value") or "").strip()
            answer_timestamp = str(row.get("answer_timestamp") or row.get("scan_time") or "").strip()
            if not user_id or not session_date or not questionnaire_id or not item_id or not answer_value_raw:
                row_counter["skipped_incomplete_rows"] += 1
                continue
            try:
                answer_value = float(answer_value_raw)
            except ValueError:
                row_counter["skipped_non_numeric_rows"] += 1
                continue

            candidate_scales = scales_by_questionnaire.get(questionnaire_id) or []
            if not candidate_scales:
                row_counter["skipped_unknown_questionnaire_rows"] += 1
                continue

            matched = False
            for scale in candidate_scales:
                scale_item_ids = {item.item_id for item in scale.items}
                if item_id not in scale_item_ids:
                    continue
                key = (user_id, session_date, scale.id)
                state = session_scale_rows.setdefault(
                    key,
                    _SessionScaleState(
                        answers={},
                        access_code=str(row.get("access_code") or "").strip(),
                    ),
                )
                prior = state.answers.get(item_id)
                if prior is None or answer_timestamp >= prior[0]:
                    state.answers[item_id] = (answer_timestamp, answer_value)
                state.row_count += 1
                matched = True
            if not matched:
                row_counter["skipped_unmapped_item_rows"] += 1

    output_rows: List[Dict[str, Any]] = []
    per_scale_prev: Dict[Tuple[str, str], float] = {}

    for user_id, session_date, scale_id, state in _iter_ordered_states(session_scale_rows):
        scale = registry.scales[scale_id]
        answers = {item_id: value for item_id, (_, value) in state.answers.items()}
        score = compute_scale_score(scale, answers)
        if score is None:
            row_counter["skipped_insufficient_items"] += 1
            continue

        prev_key = (user_id, scale_id)
        prev_normalized = per_scale_prev.get(prev_key)
        timestamps = [timestamp for timestamp, _ in state.answers.values() if timestamp]
        output_rows.append(
            {
                "access_code": state.access_code,
                "user_id": user_id,
                "session_date": session_date,
                "scale_id": scale.id,
                "scale_name": scale.name,
                "questionnaire_id": scale.questionnaire_id,
                "scale_method": scale.method,
                "citation": scale.citation,
                "raw_score": float(score.raw_score),
                "normalized_score": float(score.normalized_score),
                "answered_count": int(score.answered_count),
                "items_required": int(score.items_required),
                "confidence_tier": score.confidence or "",
                "risk_tier": score.risk_tier or "",
                "delta_vs_prev": (
                    float(score.normalized_score - prev_normalized)
                    if prev_normalized is not None
                    else ""
                ),
                "first_answer_timestamp": min(timestamps) if timestamps else "",
                "last_answer_timestamp": max(timestamps) if timestamps else "",
                "source_row_count": int(state.row_count),
            }
        )
        per_scale_prev[prev_key] = float(score.normalized_score)
        row_counter["score_rows_written"] += 1

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    output_csv = _write_csv(out_root / "questionnaire_scale_timeseries.csv", output_rows, _FIELDNAMES)
    summary = {
        "input_csv": str(input_path),
        "date_column": date_column,
        "rows_read": rows_read,
        "sessions_seen": len({(user_id, session_date) for user_id, session_date, _ in session_scale_rows}),
        "score_rows_written": len(output_rows),
        "per_counter": dict(sorted(row_counter.items())),
        "output_csv": str(output_csv),
    }
    summary_path = out_root / "questionnaire_scale_timeseries_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path)
    return summary


def _iter_ordered_states(
    session_scale_rows: Mapping[Tuple[str, str, str], _SessionScaleState]
) -> Iterable[Tuple[str, str, str, _SessionScaleState]]:
    for user_id, session_date, scale_id in sorted(session_scale_rows):
        yield user_id, session_date, scale_id, session_scale_rows[(user_id, session_date, scale_id)]


def _load_registry_bundle(source_dir: Path) -> Registry:
    if not source_dir.exists():
        raise FileNotFoundError(f"Registry source not found: {source_dir}")
    bundle: Dict[str, Any] = {"version": source_dir.name}
    for filename in _REGISTRY_FILES:
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Registry source is missing required file: {path}")
        bundle[filename.replace(".json", "")] = json.loads(path.read_text(encoding="utf-8"))
    return validate_registry_bundle(bundle)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score questionnaire replay CSV into longitudinal per-session scale rows.",
    )
    parser.add_argument("--input-csv", required=True, help="Replay CSV from build_questionnaire_replay_csv.")
    parser.add_argument("--registry-source", required=True, help="Registry bundle directory.")
    parser.add_argument("--out-dir", required=True, help="Directory for timeseries outputs.")
    parser.add_argument(
        "--date-column",
        default="scan_date_local",
        choices=("scan_date_local", "scan_date_utc"),
        help="Replay CSV date column used as the session day.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = score_questionnaire_timeseries(
        input_csv=str(args.input_csv),
        registry_source=str(args.registry_source),
        out_dir=str(args.out_dir),
        date_column=str(args.date_column),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
