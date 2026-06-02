"""Replay Nika's daily-scan answers export into Questions Agent state."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import connect, init_db, set_registry_version_active, upsert_registry_version
from questions_agent_platform.pipeline.projection import build_questions_projection_payload
from questions_agent_platform.pipeline.registry import load_registry
from questions_agent_platform.pipeline.service import get_or_create_daily_session, submit_answers

_REGISTRY_FILES: Tuple[str, ...] = ("items.json", "questionnaires.json", "scales.json")
_DATE_COLUMNS: Tuple[str, ...] = ("scan_date_local", "scan_date_utc")


@dataclass(frozen=True)
class ReplayOutputs:
    summary_path: str
    sessions_path: str
    scale_scores_path: str
    projections_path: str
    skipped_rows_path: Optional[str]


def run_daily_scan_replay(
    *,
    cfg: QuestionsAgentConfig,
    input_csv: str,
    out_dir: str,
    registry_source: str,
    registry_version: str,
    date_column: str = "scan_date_local",
    selection_mode: str = "deterministic",
    build_projections: bool = True,
) -> Dict[str, Any]:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if date_column not in _DATE_COLUMNS:
        raise ValueError(f"Unsupported date_column: {date_column}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    init_db(cfg.database_path)
    _stage_registry_bundle(
        registry_root=cfg.registry_root,
        registry_source=registry_source,
        registry_version=registry_version,
    )

    with connect(cfg.database_path) as conn:
        upsert_registry_version(conn, registry_version, status="inactive")
        set_registry_version_active(conn, registry_version)
        conn.commit()

    registry = load_registry(cfg.registry_root, registry_version)
    normalized_rows, skipped_rows, summary = _normalize_rows(
        input_path=input_path,
        registry=registry,
        date_column=date_column,
    )

    session_rows: List[Dict[str, Any]] = []
    projection_rows = 0

    with connect(cfg.database_path) as conn:
        for user_id, session_day, answers in normalized_rows:
            session = get_or_create_daily_session(
                conn,
                cfg=cfg,
                registry_root=cfg.registry_root,
                user_id=user_id,
                day=session_day,
                selection_mode=selection_mode,
            )
            result = submit_answers(
                conn,
                registry_root=cfg.registry_root,
                user_id=user_id,
                session_id=session.session_id,
                answers=answers,
            )
            if build_projections:
                build_questions_projection_payload(
                    conn,
                    user_id=user_id,
                    registry_version=registry_version,
                    day=session_day,
                )
                projection_rows += 1
            session_rows.append(
                {
                    "user_id": user_id,
                    "session_date": session_day.isoformat(),
                    "session_id": session.session_id,
                    "selection_mode": session.selection_mode,
                    "submitted_answers": len(answers),
                    "inserted_answer_events": int(result.get("inserted_answer_events") or 0),
                    "new_scale_scores_count": len(result.get("new_scale_scores") or []),
                    "completed_package": result.get("completed_package"),
                }
            )

        scale_score_path = _export_scale_scores(
            conn=conn,
            out_dir=out_root,
            registry=registry,
        )
        projection_path = _export_projection_payloads(conn=conn, out_dir=out_root)
        answer_event_count = int(
            conn.execute("SELECT COUNT(*) AS cnt FROM answer_events;").fetchone()["cnt"]
        )
        scale_score_count = int(
            conn.execute("SELECT COUNT(*) AS cnt FROM scale_scores;").fetchone()["cnt"]
        )
        distinct_scale_count = int(
            conn.execute("SELECT COUNT(DISTINCT scale_id) AS cnt FROM scale_scores;").fetchone()["cnt"]
        )

    sessions_path = _write_csv(
        out_root / "daily_scan_replay_sessions.csv",
        session_rows,
        fieldnames=(
            "user_id",
            "session_date",
            "session_id",
            "selection_mode",
            "submitted_answers",
            "inserted_answer_events",
            "new_scale_scores_count",
            "completed_package",
        ),
    )
    skipped_rows_path = None
    if skipped_rows:
        skipped_rows_path = _write_csv(
            out_root / "daily_scan_replay_skipped_rows.csv",
            skipped_rows,
            fieldnames=(
                "reason",
                "user_id",
                "scan_doc_id",
                "question_id",
                "answer_timestamp",
                "answer_value",
            ),
        )

    outputs = ReplayOutputs(
        summary_path=str(out_root / "daily_scan_replay_summary.json"),
        sessions_path=str(sessions_path),
        scale_scores_path=str(scale_score_path),
        projections_path=str(projection_path),
        skipped_rows_path=str(skipped_rows_path) if skipped_rows_path else None,
    )
    summary.update(
        {
            "database_path": cfg.database_path,
            "registry_root": cfg.registry_root,
            "registry_version": registry_version,
            "selection_mode": selection_mode,
            "date_column": date_column,
            "session_count": len(session_rows),
            "users_count": len({row["user_id"] for row in session_rows}),
            "answer_event_count": answer_event_count,
            "scale_score_count": scale_score_count,
            "distinct_scale_count": distinct_scale_count,
            "projection_count": projection_rows,
            "outputs": {
                "summary_json": outputs.summary_path,
                "sessions_csv": outputs.sessions_path,
                "scale_scores_csv": outputs.scale_scores_path,
                "projection_jsonl": outputs.projections_path,
                "skipped_rows_csv": outputs.skipped_rows_path,
            },
        }
    )
    Path(outputs.summary_path).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _normalize_rows(
    *,
    input_path: Path,
    registry: Any,
    date_column: str,
) -> Tuple[List[Tuple[str, date, List[Dict[str, Any]]]], List[Dict[str, Any]], Dict[str, Any]]:
    session_map: Dict[Tuple[str, date], List[Dict[str, Any]]] = {}
    skipped_rows: List[Dict[str, Any]] = []
    unknown_items: Counter[str] = Counter()
    summary = {
        "input_csv": str(input_path),
        "rows_read": 0,
        "rows_replayed": 0,
        "rows_skipped": 0,
    }

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            summary["rows_read"] += 1
            normalized = _normalize_answer_row(
                row=row,
                registry=registry,
                date_column=date_column,
            )
            if normalized is None:
                skipped_rows.append(
                    {
                        "reason": "invalid_or_incomplete_row",
                        "user_id": str(row.get("user_id") or ""),
                        "scan_doc_id": str(row.get("scan_doc_id") or ""),
                        "question_id": str(row.get("question_id") or ""),
                        "answer_timestamp": str(row.get("answer_timestamp") or row.get("scan_time") or ""),
                        "answer_value": str(row.get("answer_value") or ""),
                    }
                )
                summary["rows_skipped"] += 1
                continue

            user_id, session_day, answer = normalized
            item_id = str(answer["item_id"])
            if item_id not in registry.items:
                unknown_items[item_id] += 1
                skipped_rows.append(
                    {
                        "reason": "unknown_registry_item",
                        "user_id": user_id,
                        "scan_doc_id": str(row.get("scan_doc_id") or ""),
                        "question_id": item_id,
                        "answer_timestamp": str(answer["answered_at"]),
                        "answer_value": str(row.get("answer_value") or ""),
                    }
                )
                summary["rows_skipped"] += 1
                continue

            key = (user_id, session_day)
            session_map.setdefault(key, []).append(answer)
            summary["rows_replayed"] += 1

    ordered_sessions: List[Tuple[str, date, List[Dict[str, Any]]]] = []
    for (user_id, session_day), answers in sorted(session_map.items(), key=lambda item: (item[0][0], item[0][1].isoformat())):
        answers.sort(key=lambda answer: str(answer["answered_at"]))
        ordered_sessions.append((user_id, session_day, answers))

    summary["unknown_item_counts"] = dict(sorted(unknown_items.items()))
    return ordered_sessions, skipped_rows, summary


def _normalize_answer_row(
    *,
    row: Dict[str, Any],
    registry: Any,
    date_column: str,
) -> Optional[Tuple[str, date, Dict[str, Any]]]:
    user_id = str(row.get("user_id") or "").strip()
    item_id = str(row.get("question_id") or "").strip()
    day_text = str(row.get(date_column) or row.get("scan_date_utc") or "").strip()
    answered_at = str(row.get("answer_timestamp") or row.get("scan_time") or "").strip()

    if not user_id or not item_id or not day_text or not answered_at:
        return None

    session_day = _parse_day(day_text)
    value = _parse_float(row.get("answer_value"))
    if session_day is None or value is None:
        return None

    answer: Dict[str, Any] = {
        "client_event_id": _build_client_event_id(
            user_id=user_id,
            scan_doc_id=str(row.get("scan_doc_id") or ""),
            item_id=item_id,
            question_index=str(row.get("question_index") or ""),
            answered_at=answered_at,
        ),
        "item_id": item_id,
        "value": value,
        "answered_at": answered_at,
        "raw": _build_raw_payload(row=row),
    }
    return user_id, session_day, answer


def _build_raw_payload(*, row: Dict[str, Any]) -> Dict[str, Any]:
    raw_payload: Dict[str, Any] = {
        "source": "daily_scan_answers.csv",
        "access_code": str(row.get("access_code") or ""),
        "scan_doc_id": str(row.get("scan_doc_id") or ""),
        "question_string": str(row.get("question_string") or ""),
        "sample_barcodes": str(row.get("sample_barcodes") or ""),
        "sample_doc_ids": str(row.get("sample_doc_ids") or ""),
        "sample_dates": str(row.get("sample_dates") or ""),
        "sample_types": str(row.get("sample_types") or ""),
        "reversed": str(row.get("reversed") or ""),
    }
    answer_json = str(row.get("answer_json") or "").strip()
    if answer_json:
        try:
            raw_payload["source_answer"] = json.loads(answer_json)
        except Exception:
            raw_payload["source_answer_json"] = answer_json
    return raw_payload


def _build_client_event_id(
    *,
    user_id: str,
    scan_doc_id: str,
    item_id: str,
    question_index: str,
    answered_at: str,
) -> str:
    digest = hashlib.sha256(
        "::".join((user_id, scan_doc_id, item_id, question_index, answered_at)).encode("utf-8")
    ).hexdigest()
    return f"daily-scan::{digest}"


def _parse_day(value: str) -> Optional[date]:
    text = value.strip()
    if not text:
        return None
    candidate = text[:10]
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        return None


def _parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stage_registry_bundle(
    *,
    registry_root: str,
    registry_source: str,
    registry_version: str,
) -> None:
    source_dir = Path(registry_source)
    if not source_dir.exists():
        raise FileNotFoundError(f"Registry source not found: {source_dir}")
    missing = [name for name in _REGISTRY_FILES if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Registry source is missing required files: {', '.join(sorted(missing))}"
        )

    target_dir = Path(registry_root) / "versions" / registry_version
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in _REGISTRY_FILES:
        shutil.copyfile(source_dir / name, target_dir / name)


def _export_scale_scores(*, conn: Any, out_dir: Path, registry: Any) -> Path:
    rows = conn.execute(
        """
        SELECT *
        FROM scale_scores
        ORDER BY user_id ASC, window_end ASC, scale_id ASC, computed_at ASC;
        """
    ).fetchall()
    export_rows: List[Dict[str, Any]] = []
    for row in rows:
        export_row = dict(row)
        scale = registry.scales.get(str(row["scale_id"]))
        export_row["scale_name"] = scale.name if scale else ""
        export_row["questionnaire_id"] = scale.questionnaire_id if scale else ""
        export_row["scale_method"] = scale.method if scale else ""
        export_row["citation"] = scale.citation if scale else ""
        export_rows.append(export_row)
    fieldnames: Sequence[str]
    if export_rows:
        fieldnames = tuple(export_rows[0].keys())
    else:
        fieldnames = (
            "score_id",
            "user_id",
            "scale_id",
            "computed_at",
            "window_start",
            "window_end",
            "raw_score",
            "normalized_score",
            "confidence_tier",
            "items_answered_count",
            "items_required",
            "baseline_mean",
            "baseline_std",
            "personal_z",
            "delta_vs_prev",
            "scale_name",
            "questionnaire_id",
            "scale_method",
            "citation",
        )
    return _write_csv(out_dir / "daily_scan_replay_scale_scores.csv", export_rows, fieldnames=fieldnames)


def _export_projection_payloads(*, conn: Any, out_dir: Path) -> Path:
    rows = conn.execute(
        """
        SELECT user_id, timestamp, payload_json
        FROM projections_questions
        ORDER BY user_id ASC, timestamp ASC;
        """
    ).fetchall()
    output_path = out_dir / "daily_scan_replay_projection_payloads.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_path


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], *, fieldnames: Sequence[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
