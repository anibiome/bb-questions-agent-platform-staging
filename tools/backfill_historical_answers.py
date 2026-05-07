#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
for path in (PROJECT_PARENT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _install_repo_package_alias() -> None:
    """Make the raw checkout importable as `questions_agent_platform`.

    This repo mixes `pipeline.*` imports with package-style
    `questions_agent_platform.pipeline.*` imports. When the script runs from a
    plain checkout, we need to install a package alias so both styles resolve.
    """

    package_name = "questions_agent_platform"
    if package_name in sys.modules:
        return

    init_file = PROJECT_ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(PROJECT_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to create package alias for {package_name!r} from {PROJECT_ROOT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


_install_repo_package_alias()

from pipeline.config import ensure_paths, load_config  # noqa: E402
from pipeline.db import connect, init_db  # noqa: E402
from pipeline.registry import list_versions, load_registry  # noqa: E402
from pipeline.service import get_or_create_daily_session, submit_answers  # noqa: E402


@dataclass(frozen=True)
class ResolvedAnswerRow:
    row_index: int
    user_id: str
    day: date
    answered_at: str
    item_id: str
    value: float
    raw_question_id: str
    raw_question_text: str
    scan_doc_id: str
    question_index: int
    raw_row: Dict[str, Any]
    resolution_mode: str


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill historical questionnaire answers into the reference questions-agent "
            "SQLite engine so validated scale scores and baselines can be computed "
            "retroactively."
        )
    )
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--input-csv", required=True, help="Historical answers CSV to import")
    parser.add_argument("--out-json", default=None, help="Optional summary JSON path")
    parser.add_argument("--unmatched-csv", default=None, help="Optional unresolved rows CSV path")
    parser.add_argument("--selection-mode", default="deterministic", help="Selection mode for session creation")
    parser.add_argument("--user-id-column", default="user_id")
    parser.add_argument("--date-column", default="scan_date_local")
    parser.add_argument("--answered-at-column", default="answer_timestamp")
    parser.add_argument("--scan-time-column", default="scan_time")
    parser.add_argument("--scan-doc-id-column", default="scan_doc_id")
    parser.add_argument("--question-index-column", default="question_index")
    parser.add_argument("--question-id-column", default="question_id")
    parser.add_argument("--question-text-column", default="question_string")
    parser.add_argument("--item-id-column", default="")
    parser.add_argument("--value-column", default="answer_value")
    parser.add_argument(
        "--mapping-json",
        default=None,
        help="Optional JSON mapping for raw question ids/text to registry item ids",
    )
    parser.add_argument(
        "--value-offset",
        type=float,
        default=0.0,
        help="Optional numeric offset applied to every imported answer value",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to inspect/import")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and summarize without writing to SQLite")
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Skip unresolved rows instead of failing before import",
    )
    parser.add_argument(
        "--bootstrap-production-registry",
        action="store_true",
        help="If registry_root is empty, copy data/registry_production/v3 into versions/v3",
    )
    parser.add_argument(
        "--bootstrap-demo-registry",
        action="store_true",
        help="If registry_root is empty, copy data/registry_demo/v1 into versions/v1",
    )
    args = parser.parse_args(argv)

    cfg = load_config(str(args.config))
    ensure_paths(cfg)
    _maybe_bootstrap_registry(
        cfg.registry_root,
        bootstrap_production=bool(args.bootstrap_production_registry),
        bootstrap_demo=bool(args.bootstrap_demo_registry),
    )
    init_db(cfg.database_path)

    registry_version = _pick_registry_version(cfg.registry_root)
    registry = load_registry(cfg.registry_root, registry_version)
    resolver = _ItemResolver(registry=registry, mapping_path=args.mapping_json)

    raw_rows = _read_csv_rows(str(args.input_csv), limit=int(args.limit))
    resolved_rows: List[ResolvedAnswerRow] = []
    unresolved_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(raw_rows, start=1):
        try:
            resolved_rows.append(
                _resolve_row(
                    row,
                    row_index=idx,
                    resolver=resolver,
                    value_offset=float(args.value_offset),
                    user_id_column=str(args.user_id_column),
                    date_column=str(args.date_column),
                    answered_at_column=str(args.answered_at_column),
                    scan_time_column=str(args.scan_time_column),
                    scan_doc_id_column=str(args.scan_doc_id_column),
                    question_index_column=str(args.question_index_column),
                    question_id_column=str(args.question_id_column),
                    question_text_column=str(args.question_text_column),
                    item_id_column=str(args.item_id_column),
                    value_column=str(args.value_column),
                )
            )
        except Exception as exc:
            unresolved_rows.append(
                {
                    "row_index": idx,
                    "error": str(exc),
                    "user_id": row.get(str(args.user_id_column), ""),
                    "question_id": row.get(str(args.question_id_column), ""),
                    "question_string": row.get(str(args.question_text_column), ""),
                    "scan_doc_id": row.get(str(args.scan_doc_id_column), ""),
                }
            )

    grouped = _group_rows(resolved_rows)
    summary: Dict[str, Any] = {
        "ok": True,
        "dry_run": bool(args.dry_run),
        "input_csv": str(Path(str(args.input_csv)).expanduser().resolve()),
        "database_path": cfg.database_path,
        "registry_root": cfg.registry_root,
        "registry_version": registry_version,
        "rows_read": len(raw_rows),
        "rows_resolved": len(resolved_rows),
        "rows_unresolved": len(unresolved_rows),
        "group_count": len(grouped),
        "selection_mode": str(args.selection_mode),
        "value_offset": float(args.value_offset),
        "resolution_modes": _count_resolution_modes(resolved_rows),
        "sample_groups": [
            {
                "user_id": user_id,
                "day": group_day.isoformat(),
                "answer_count": len(rows),
            }
            for (user_id, group_day), rows in list(grouped.items())[:10]
        ],
    }

    if unresolved_rows:
        summary["unresolved_examples"] = unresolved_rows[:25]
        if args.unmatched_csv:
            _write_csv(
                str(args.unmatched_csv),
                unresolved_rows,
                ["row_index", "error", "user_id", "question_id", "question_string", "scan_doc_id"],
            )
        if not args.allow_unmatched:
            summary["ok"] = False
            summary["status"] = "blocked_unresolved_rows"
            _emit_summary(summary, args.out_json)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2

    if args.dry_run:
        summary["status"] = "dry_run_complete"
        _emit_summary(summary, args.out_json)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    inserted_events = 0
    new_scale_scores = 0
    completed_packages: List[Dict[str, Any]] = []

    with connect(cfg.database_path) as conn:
        for (user_id, group_day), rows in grouped.items():
            session = get_or_create_daily_session(
                conn,
                cfg=cfg,
                registry_root=cfg.registry_root,
                user_id=user_id,
                day=group_day,
                selection_mode=str(args.selection_mode),
            )
            answers = [
                {
                    "client_event_id": _stable_client_event_id(row),
                    "item_id": row.item_id,
                    "value": row.value,
                    "answered_at": row.answered_at,
                    "raw": {
                        "import_source": "historical_backfill",
                        "raw_question_id": row.raw_question_id,
                        "raw_question_text": row.raw_question_text,
                        "scan_doc_id": row.scan_doc_id,
                        "question_index": row.question_index,
                        "resolution_mode": row.resolution_mode,
                        "source_row_index": row.row_index,
                        "source_row": row.raw_row,
                    },
                }
                for row in rows
            ]
            result = submit_answers(
                conn,
                registry_root=cfg.registry_root,
                user_id=user_id,
                session_id=session.session_id,
                answers=answers,
            )
            inserted_events += int(result.get("inserted_answer_events") or 0)
            scores = result.get("new_scale_scores") or []
            new_scale_scores += len(scores)
            if result.get("completed_package"):
                completed_packages.append(
                    {
                        "user_id": user_id,
                        "day": group_day.isoformat(),
                        "completed_package": result["completed_package"],
                    }
                )

    summary["status"] = "import_complete"
    summary["inserted_answer_events"] = inserted_events
    summary["new_scale_scores_count"] = new_scale_scores
    summary["completed_packages"] = completed_packages
    _emit_summary(summary, args.out_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _read_csv_rows(path: str, *, limit: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            rows.append({str(k): ("" if v is None else str(v)) for k, v in row.items()})
            if limit > 0 and idx >= limit:
                break
    return rows


def _resolve_row(
    row: Dict[str, str],
    *,
    row_index: int,
    resolver: "_ItemResolver",
    value_offset: float,
    user_id_column: str,
    date_column: str,
    answered_at_column: str,
    scan_time_column: str,
    scan_doc_id_column: str,
    question_index_column: str,
    question_id_column: str,
    question_text_column: str,
    item_id_column: str,
    value_column: str,
) -> ResolvedAnswerRow:
    user_id = _required(row.get(user_id_column), f"missing {user_id_column}")
    raw_question_id = str(row.get(question_id_column) or "").strip()
    raw_question_text = str(row.get(question_text_column) or "").strip()
    item_id, resolution_mode = resolver.resolve(
        explicit_item_id=str(row.get(item_id_column) or "").strip() if item_id_column else "",
        raw_question_id=raw_question_id,
        raw_question_text=raw_question_text,
    )
    raw_value = _required(row.get(value_column), f"missing {value_column}")
    value = float(raw_value) + value_offset
    group_day = _parse_day(
        raw_day=str(row.get(date_column) or "").strip(),
        answered_at=str(row.get(answered_at_column) or "").strip(),
        scan_time=str(row.get(scan_time_column) or "").strip(),
    )
    question_index = _safe_int(row.get(question_index_column), default=row_index)
    answered_at = _normalize_answered_at(
        answered_at=str(row.get(answered_at_column) or "").strip(),
        scan_time=str(row.get(scan_time_column) or "").strip(),
        fallback_day=group_day,
        fallback_index=question_index,
    )
    return ResolvedAnswerRow(
        row_index=row_index,
        user_id=user_id,
        day=group_day,
        answered_at=answered_at,
        item_id=item_id,
        value=value,
        raw_question_id=raw_question_id,
        raw_question_text=raw_question_text,
        scan_doc_id=str(row.get(scan_doc_id_column) or "").strip(),
        question_index=question_index,
        raw_row=dict(row),
        resolution_mode=resolution_mode,
    )


def _stable_client_event_id(row: ResolvedAnswerRow) -> str:
    scan_part = row.scan_doc_id or "scanless"
    raw_key = row.raw_question_id or _normalize_text(row.raw_question_text) or row.item_id
    return (
        f"historical::{row.user_id}::{row.day.isoformat()}::{scan_part}"
        f"::{row.question_index:04d}::{raw_key}::{row.item_id}"
    )


def _group_rows(rows: Iterable[ResolvedAnswerRow]) -> Dict[Tuple[str, date], List[ResolvedAnswerRow]]:
    grouped: Dict[Tuple[str, date], List[ResolvedAnswerRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.user_id, row.day)].append(row)
    for key in grouped:
        grouped[key].sort(
            key=lambda item: (
                item.answered_at,
                item.scan_doc_id,
                item.question_index,
                item.row_index,
            )
        )
    return dict(sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1].isoformat())))


def _count_resolution_modes(rows: Sequence[ResolvedAnswerRow]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.resolution_mode] += 1
    return dict(sorted(counts.items()))


class _ItemResolver:
    def __init__(self, *, registry: Any, mapping_path: Optional[str]) -> None:
        self.registry = registry
        self._mapping_by_id, self._mapping_by_text = _load_mapping(mapping_path)
        self._items_by_text: Dict[str, List[str]] = defaultdict(list)
        self._items_by_normalized_text: Dict[str, List[str]] = defaultdict(list)
        for item in registry.items.values():
            self._items_by_text[item.text].append(item.id)
            self._items_by_normalized_text[_normalize_text(item.text)].append(item.id)

    def resolve(
        self,
        *,
        explicit_item_id: str,
        raw_question_id: str,
        raw_question_text: str,
    ) -> Tuple[str, str]:
        if explicit_item_id:
            if explicit_item_id not in self.registry.items:
                raise ValueError(f"unknown explicit item_id {explicit_item_id!r}")
            return explicit_item_id, "explicit_item_id"

        if raw_question_id:
            mapped = self._mapping_by_id.get(raw_question_id)
            if mapped:
                self._assert_item_exists(mapped, f"mapping for question_id {raw_question_id!r}")
                return mapped, "mapping_question_id"
            if raw_question_id in self.registry.items:
                return raw_question_id, "question_id_equals_item_id"

        if raw_question_text:
            mapped = self._mapping_by_text.get(_normalize_text(raw_question_text))
            if mapped:
                self._assert_item_exists(mapped, f"mapping for question_string {raw_question_text!r}")
                return mapped, "mapping_question_text"

            exact = self._items_by_text.get(raw_question_text, [])
            if len(exact) == 1:
                return exact[0], "exact_text_match"
            if len(exact) > 1:
                raise ValueError(f"ambiguous exact text match for {raw_question_text!r}: {exact}")

            normalized = self._items_by_normalized_text.get(_normalize_text(raw_question_text), [])
            if len(normalized) == 1:
                return normalized[0], "normalized_text_match"
            if len(normalized) > 1:
                raise ValueError(f"ambiguous normalized text match for {raw_question_text!r}: {normalized}")

        raise ValueError("unable to resolve registry item_id from row")

    def _assert_item_exists(self, item_id: str, source: str) -> None:
        if item_id not in self.registry.items:
            raise ValueError(f"{source} points to unknown item_id {item_id!r}")


def _load_mapping(path: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not path:
        return {}, {}
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    by_id: Dict[str, str] = {}
    by_text: Dict[str, str] = {}

    if isinstance(payload, dict):
        raw_by_id = payload.get("by_question_id")
        raw_by_text = payload.get("by_question_string")
        if isinstance(raw_by_id, dict) or isinstance(raw_by_text, dict):
            for key, value in (raw_by_id or {}).items():
                by_id[str(key)] = str(value)
            for key, value in (raw_by_text or {}).items():
                by_text[_normalize_text(str(key))] = str(value)
            return by_id, by_text
        for key, value in payload.items():
            target = str(value)
            by_id[str(key)] = target
            by_text[_normalize_text(str(key))] = target
        return by_id, by_text

    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            item_id = str(entry.get("item_id") or "").strip()
            if not item_id:
                continue
            question_id = str(entry.get("question_id") or "").strip()
            question_text = str(entry.get("question_string") or "").strip()
            if question_id:
                by_id[question_id] = item_id
            if question_text:
                by_text[_normalize_text(question_text)] = item_id
        return by_id, by_text

    raise ValueError("mapping JSON must be an object or list")


def _maybe_bootstrap_registry(
    registry_root: str,
    *,
    bootstrap_production: bool,
    bootstrap_demo: bool,
) -> None:
    if list_versions(registry_root):
        return
    if bootstrap_production and bootstrap_demo:
        raise ValueError("choose only one bootstrap mode")
    if not bootstrap_production and not bootstrap_demo:
        return

    if bootstrap_production:
        version = "v3"
        source_dir = PROJECT_ROOT / "data" / "registry_production" / version
    else:
        version = "v1"
        source_dir = PROJECT_ROOT / "data" / "registry_demo" / version

    if not source_dir.exists():
        raise ValueError(f"registry bootstrap source not found: {source_dir}")

    dest_dir = Path(registry_root).expanduser() / "versions" / version
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)


def _pick_registry_version(registry_root: str) -> str:
    versions = list_versions(registry_root)
    if not versions:
        raise ValueError("No registry versions found. Seed or bootstrap a registry first.")
    return versions[-1]


def _parse_day(*, raw_day: str, answered_at: str, scan_time: str) -> date:
    if raw_day:
        try:
            return date.fromisoformat(raw_day[:10])
        except ValueError:
            pass
    for candidate in (answered_at, scan_time):
        if not candidate:
            continue
        parsed = _parse_datetime(candidate)
        if parsed is not None:
            return parsed.date()
    raise ValueError("unable to determine canonical local day")


def _normalize_answered_at(*, answered_at: str, scan_time: str, fallback_day: date, fallback_index: int) -> str:
    for candidate in (answered_at, scan_time):
        if not candidate:
            continue
        parsed = _parse_datetime(candidate)
        if parsed is not None:
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    second = max(0, min(59, int(fallback_index) % 60))
    return f"{fallback_day.isoformat()}T12:00:{second:02d}Z"


def _parse_datetime(value: str) -> Optional[datetime]:
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def _safe_int(value: Any, *, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _required(value: Any, message: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(message)
    return text


def _emit_summary(summary: Dict[str, Any], out_json: Optional[str]) -> None:
    if not out_json:
        return
    out_path = Path(str(out_json)).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
