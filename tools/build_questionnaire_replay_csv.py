"""Flatten questionnaireAnswers export JSON into replay-ready answer CSV rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from questions_agent_platform.pipeline.registry import Registry, validate_registry_bundle

_REGISTRY_FILES: Tuple[str, ...] = ("items.json", "questionnaires.json", "scales.json")
_SOURCE_TO_TARGET_QUESTIONNAIRES: Mapping[str, Tuple[str, ...]] = {
    "DASS-21": ("q_dass_21_depression", "q_dass_21_stress"),
    "PANAS-SF": ("q_panas_sf_positive", "q_panas_sf_negative"),
    "QOLS": ("q_qols_flanagan",),
    "CASP-19": (),
}
_FIELDNAMES: Tuple[str, ...] = (
    "access_code",
    "user_id",
    "scan_doc_id",
    "scan_time",
    "answer_timestamp",
    "scan_date_utc",
    "scan_date_local",
    "question_index",
    "question_id",
    "question_string",
    "answer_value",
    "answer_json",
    "sample_barcodes",
    "sample_doc_ids",
    "sample_dates",
    "sample_types",
    "reversed",
    "source_questionnaire_id",
    "source_question_id",
    "target_questionnaire_id",
    "match_strategy",
    "match_score",
)
_UNMATCHED_FIELDNAMES: Tuple[str, ...] = (
    "source_questionnaire_id",
    "source_question_id",
    "question_string",
    "reason",
    "question_index",
    "scan_doc_id",
    "answer_timestamp",
    "answer_value",
    "best_candidate_item_id",
    "best_candidate_questionnaire_id",
    "best_candidate_text",
    "best_candidate_score",
)
_LEADING_PHRASES: Tuple[str, ...] = (
    "indicate the extent you have felt ",
    "how often have you felt ",
    "how satisfied are you with ",
)
_TIMEFRAME_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bover the past month\b"),
    re.compile(r"\bin the past 30 days\b"),
    re.compile(r"\bduring the past 30 days\b"),
    re.compile(r"\bthese days\b"),
)
_TOKEN_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "your",
        "you",
        "have",
        "felt",
        "how",
        "often",
        "are",
        "with",
        "indicate",
        "extent",
        "over",
        "past",
        "month",
        "in",
        "during",
        "days",
    }
)


@dataclass(frozen=True)
class RegistryCandidate:
    item_id: str
    questionnaire_id: str
    questionnaire_name: str
    item_text: str
    normalized_text: str
    semantic_tokens: Tuple[str, ...]


@dataclass(frozen=True)
class MatchOutcome:
    candidate: Optional[RegistryCandidate]
    strategy: str
    score: float
    reason: Optional[str] = None


@dataclass(frozen=True)
class PendingReplayRow:
    row: Dict[str, Any]
    target_item_id: str
    similarity_score: float


def build_questionnaire_replay_csv(
    *,
    input_json: str,
    registry_source: str,
    out_dir: str,
    access_code: Optional[str] = None,
) -> Dict[str, Any]:
    input_path = Path(input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    registry = _load_registry_bundle(Path(registry_source))
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    export = json.loads(input_path.read_text(encoding="utf-8"))
    resolved_access_code = (access_code or input_path.stem).strip()
    rows: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []
    per_questionnaire = _init_questionnaire_summary()
    total_docs = 0
    processed_docs = 0
    seen_doc_signatures: set[str] = set()

    for block in export.get("sample_blocks", []):
        sample = block.get("sample") or {}
        digital_data = block.get("digital_data") or {}
        for questionnaire_doc in digital_data.get("questionnaireAnswers", []):
            total_docs += 1
            doc_signature = _questionnaire_doc_signature(questionnaire_doc)
            source_qid = str(questionnaire_doc.get("questionnaireId") or "").strip()
            if doc_signature in seen_doc_signatures:
                per_questionnaire[source_qid]["duplicate_source_documents"] += 1
                continue
            seen_doc_signatures.add(doc_signature)
            processed_docs += 1
            pending_rows = _build_pending_rows(
                questionnaire_doc=questionnaire_doc,
                sample=sample,
                registry=registry,
                access_code=resolved_access_code,
                export_user=export.get("user") or {},
                per_questionnaire=per_questionnaire,
                unmatched_rows=unmatched_rows,
            )
            for selected_row, duplicate_rows in _dedupe_target_items(pending_rows):
                rows.append(selected_row.row)
                source_qid = str(selected_row.row["source_questionnaire_id"])
                per_questionnaire[source_qid]["matched_rows"] += 1
                for duplicate in duplicate_rows:
                    per_questionnaire[source_qid]["duplicate_target_item_drops"] += 1
                    unmatched_rows.append(
                        {
                            "source_questionnaire_id": duplicate.row["source_questionnaire_id"],
                            "source_question_id": duplicate.row["source_question_id"],
                            "question_string": duplicate.row["question_string"],
                            "reason": "duplicate_target_item",
                            "question_index": duplicate.row["question_index"],
                            "scan_doc_id": duplicate.row["scan_doc_id"],
                            "answer_timestamp": duplicate.row["answer_timestamp"],
                            "answer_value": duplicate.row["answer_value"],
                            "best_candidate_item_id": duplicate.target_item_id,
                            "best_candidate_questionnaire_id": duplicate.row["target_questionnaire_id"],
                            "best_candidate_text": str(
                                json.loads(str(duplicate.row["answer_json"])).get("mapped_item_text") or ""
                            ),
                            "best_candidate_score": f"{duplicate.similarity_score:.4f}",
                        }
                    )

    output_csv = _write_csv(out_root / "questionnaire_answers_replay.csv", rows, _FIELDNAMES)
    unmatched_csv = _write_csv(
        out_root / "questionnaire_answers_unmatched.csv",
        unmatched_rows,
        _UNMATCHED_FIELDNAMES,
    )
    summary = _build_summary(
        input_json=str(input_path),
        output_csv=str(output_csv),
        unmatched_csv=str(unmatched_csv),
        rows_written=len(rows),
        unmatched_rows=len(unmatched_rows),
        total_docs=total_docs,
        processed_docs=processed_docs,
        per_questionnaire=per_questionnaire,
    )
    coverage_json = out_root / "questionnaire_answers_coverage.json"
    coverage_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["coverage_json"] = str(coverage_json)
    return summary


def _build_pending_rows(
    *,
    questionnaire_doc: Mapping[str, Any],
    sample: Mapping[str, Any],
    registry: Registry,
    access_code: str,
    export_user: Mapping[str, Any],
    per_questionnaire: Dict[str, Counter[str]],
    unmatched_rows: List[Dict[str, Any]],
) -> List[PendingReplayRow]:
    source_qid = str(questionnaire_doc.get("questionnaireId") or "").strip()
    user_id = str(questionnaire_doc.get("userId") or export_user.get("user_id") or export_user.get("id") or "").strip()
    answered_at = str(questionnaire_doc.get("date") or "").strip()
    day = answered_at[:10]
    doc_id = str(questionnaire_doc.get("_docId") or "").strip()
    answers = questionnaire_doc.get("answers") or []

    bucket = per_questionnaire[source_qid]
    bucket["documents"] += 1
    bucket["answers_seen"] += len(answers)
    bucket["supported_documents"] += 1 if _SOURCE_TO_TARGET_QUESTIONNAIRES.get(source_qid) else 0

    catalog = build_source_questionnaire_catalog(registry).get(source_qid, ())
    pending_rows: List[PendingReplayRow] = []
    for idx, answer in enumerate(answers):
        raw_value = answer.get("intensity")
        if raw_value in (None, ""):
            bucket["unmatched_rows"] += 1
            unmatched_rows.append(
                _build_unmatched_row(
                    source_questionnaire_id=source_qid,
                    source_question_id=str(answer.get("questionId") or ""),
                    question_string=str(answer.get("questionString") or ""),
                    reason="missing_numeric_value",
                    question_index=str(idx),
                    scan_doc_id=doc_id,
                    answer_timestamp=answered_at,
                    answer_value=str(raw_value or ""),
                    match=None,
                )
            )
            continue

        match = match_source_question(
            registry=registry,
            source_questionnaire_id=source_qid,
            question_text=str(answer.get("questionString") or ""),
        )
        if match.candidate is None:
            bucket["unmatched_rows"] += 1
            unmatched_rows.append(
                _build_unmatched_row(
                    source_questionnaire_id=source_qid,
                    source_question_id=str(answer.get("questionId") or ""),
                    question_string=str(answer.get("questionString") or ""),
                    reason=str(match.reason or "unmatched_question"),
                    question_index=str(idx),
                    scan_doc_id=doc_id,
                    answer_timestamp=answered_at,
                    answer_value=str(raw_value),
                    match=match,
                )
            )
            continue

        row = {
            "access_code": access_code,
            "user_id": user_id,
            "scan_doc_id": doc_id,
            "scan_time": answered_at,
            "answer_timestamp": answered_at,
            "scan_date_utc": day,
            "scan_date_local": day,
            "question_index": str(idx),
            "question_id": match.candidate.item_id,
            "question_string": str(answer.get("questionString") or ""),
            "answer_value": str(raw_value),
            "answer_json": json.dumps(
                {
                    "source_questionnaire_id": source_qid,
                    "source_question_id": answer.get("questionId"),
                    "source_answer": answer,
                    "mapped_item_id": match.candidate.item_id,
                    "mapped_item_text": match.candidate.item_text,
                    "mapped_questionnaire_id": match.candidate.questionnaire_id,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "sample_barcodes": str(sample.get("sample_barcode") or ""),
            "sample_doc_ids": str(sample.get("doc_id") or ""),
            "sample_dates": str(sample.get("collection_date") or ""),
            "sample_types": str(sample.get("sample_type") or sample.get("sampleType") or ""),
            "reversed": str(answer.get("scaleInverted") or ""),
            "source_questionnaire_id": source_qid,
            "source_question_id": str(answer.get("questionId") or ""),
            "target_questionnaire_id": match.candidate.questionnaire_id,
            "match_strategy": match.strategy,
            "match_score": f"{match.score:.4f}",
        }
        pending_rows.append(
            PendingReplayRow(
                row=row,
                target_item_id=match.candidate.item_id,
                similarity_score=match.score,
            )
        )
        bucket[f"matched_via_{match.strategy}"] += 1

    if catalog:
        bucket["supported_rows_seen"] += len(answers)
    return pending_rows


def _dedupe_target_items(
    pending_rows: Sequence[PendingReplayRow],
) -> Iterable[Tuple[PendingReplayRow, List[PendingReplayRow]]]:
    grouped: Dict[str, List[PendingReplayRow]] = defaultdict(list)
    for pending in pending_rows:
        grouped[pending.target_item_id].append(pending)
    for grouped_rows in grouped.values():
        ordered = sorted(
            grouped_rows,
            key=lambda row: (-row.similarity_score, int(str(row.row["question_index"]) or "0")),
        )
        yield ordered[0], ordered[1:]


def build_source_questionnaire_catalog(registry: Registry) -> Dict[str, Tuple[RegistryCandidate, ...]]:
    catalog: Dict[str, List[RegistryCandidate]] = defaultdict(list)
    items_by_id = registry.items
    questionnaire_names = {qid: q.name for qid, q in registry.questionnaires.items()}

    for source_qid, target_qids in _SOURCE_TO_TARGET_QUESTIONNAIRES.items():
        seen_item_ids: set[str] = set()
        for scale in registry.scales.values():
            if scale.questionnaire_id not in target_qids:
                continue
            for scale_item in scale.items:
                if scale_item.item_id in seen_item_ids:
                    continue
                item = items_by_id.get(scale_item.item_id)
                if item is None:
                    continue
                normalized = normalize_prompt_text(item.text)
                catalog[source_qid].append(
                    RegistryCandidate(
                        item_id=item.id,
                        questionnaire_id=scale.questionnaire_id,
                        questionnaire_name=questionnaire_names.get(scale.questionnaire_id, ""),
                        item_text=item.text,
                        normalized_text=normalized,
                        semantic_tokens=tuple(_semantic_tokens(normalized)),
                    )
                )
                seen_item_ids.add(scale_item.item_id)

    return {key: tuple(value) for key, value in catalog.items()}


def match_source_question(
    *,
    registry: Registry,
    source_questionnaire_id: str,
    question_text: str,
) -> MatchOutcome:
    catalog = build_source_questionnaire_catalog(registry).get(source_questionnaire_id, ())
    if not catalog:
        reason = (
            "unsupported_source_questionnaire"
            if source_questionnaire_id in _SOURCE_TO_TARGET_QUESTIONNAIRES
            else "unknown_source_questionnaire"
        )
        return MatchOutcome(candidate=None, strategy="none", score=0.0, reason=reason)

    normalized = normalize_prompt_text(question_text)
    exact = [candidate for candidate in catalog if candidate.normalized_text == normalized]
    if len(exact) == 1:
        return MatchOutcome(candidate=exact[0], strategy="exact_normalized", score=1.0)
    if len(exact) > 1:
        return MatchOutcome(candidate=None, strategy="none", score=0.0, reason="ambiguous_exact_match")

    source_tokens = tuple(_semantic_tokens(normalized))
    ranked = sorted(
        (
            (
                _candidate_similarity(
                    normalized_text=normalized,
                    semantic_tokens=source_tokens,
                    candidate=candidate,
                ),
                candidate,
            )
            for candidate in catalog
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best_candidate = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_score >= 0.86 and (best_score - second_score) >= 0.02:
        return MatchOutcome(candidate=best_candidate, strategy="fuzzy_text", score=best_score)

    return MatchOutcome(candidate=None, strategy="none", score=best_score, reason="no_confident_match")


def normalize_prompt_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("couldn't", "could not")
    normalized = normalized.replace("n't", " not")
    normalized = normalized.replace("eg,", "eg")
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    for phrase in _LEADING_PHRASES:
        if normalized.startswith(phrase):
            normalized = normalized[len(phrase):]
            break
    for pattern in _TIMEFRAME_PATTERNS:
        normalized = pattern.sub(" ", normalized)
    normalized = normalized.replace("this nervous", "nervous")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _semantic_tokens(normalized_text: str) -> Iterable[str]:
    for token in normalized_text.split():
        if token not in _TOKEN_STOPWORDS:
            yield token


def _candidate_similarity(
    *,
    normalized_text: str,
    semantic_tokens: Sequence[str],
    candidate: RegistryCandidate,
) -> float:
    char_score = SequenceMatcher(None, normalized_text, candidate.normalized_text).ratio()
    source_set = set(semantic_tokens)
    target_set = set(candidate.semantic_tokens)
    token_score = 0.0
    if source_set and target_set:
        token_score = len(source_set & target_set) / max(len(source_set), len(target_set))
    return max(char_score, token_score)


def _build_unmatched_row(
    *,
    source_questionnaire_id: str,
    source_question_id: str,
    question_string: str,
    reason: str,
    question_index: str,
    scan_doc_id: str,
    answer_timestamp: str,
    answer_value: str,
    match: Optional[MatchOutcome],
) -> Dict[str, Any]:
    candidate = match.candidate if match is not None else None
    return {
        "source_questionnaire_id": source_questionnaire_id,
        "source_question_id": source_question_id,
        "question_string": question_string,
        "reason": reason,
        "question_index": question_index,
        "scan_doc_id": scan_doc_id,
        "answer_timestamp": answer_timestamp,
        "answer_value": answer_value,
        "best_candidate_item_id": candidate.item_id if candidate else "",
        "best_candidate_questionnaire_id": candidate.questionnaire_id if candidate else "",
        "best_candidate_text": candidate.item_text if candidate else "",
        "best_candidate_score": f"{match.score:.4f}" if match is not None else "",
    }


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


def _questionnaire_doc_signature(questionnaire_doc: Mapping[str, Any]) -> str:
    doc_id = str(questionnaire_doc.get("_docId") or "").strip()
    if doc_id:
        return f"doc::{questionnaire_doc.get('questionnaireId') or ''}::{doc_id}"
    answers = questionnaire_doc.get("answers") or []
    fingerprint = [
        {
            "questionId": answer.get("questionId"),
            "questionString": answer.get("questionString"),
            "intensity": answer.get("intensity"),
        }
        for answer in answers
    ]
    return json.dumps(
        {
            "questionnaireId": questionnaire_doc.get("questionnaireId"),
            "userId": questionnaire_doc.get("userId"),
            "date": questionnaire_doc.get("date"),
            "answers": fingerprint,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _init_questionnaire_summary() -> Dict[str, Counter[str]]:
    summary: Dict[str, Counter[str]] = {}
    for source_qid in _SOURCE_TO_TARGET_QUESTIONNAIRES:
        summary[source_qid] = Counter(
            {
                "documents": 0,
                "supported_documents": 0,
                "duplicate_source_documents": 0,
                "answers_seen": 0,
                "supported_rows_seen": 0,
                "matched_rows": 0,
                "unmatched_rows": 0,
                "duplicate_target_item_drops": 0,
                "matched_via_exact_normalized": 0,
                "matched_via_fuzzy_text": 0,
            }
        )
    return summary


def _build_summary(
    *,
    input_json: str,
    output_csv: str,
    unmatched_csv: str,
    rows_written: int,
    unmatched_rows: int,
    total_docs: int,
    processed_docs: int,
    per_questionnaire: Dict[str, Counter[str]],
) -> Dict[str, Any]:
    return {
        "input_json": input_json,
        "output_csv": output_csv,
        "unmatched_csv": unmatched_csv,
        "documents_seen": total_docs,
        "documents_processed": processed_docs,
        "rows_written": rows_written,
        "unmatched_rows": unmatched_rows,
        "source_questionnaires": {
            key: {
                **dict(counter),
                "target_questionnaire_ids": list(_SOURCE_TO_TARGET_QUESTIONNAIRES.get(key, ())),
            }
            for key, counter in per_questionnaire.items()
        },
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten questionnaireAnswers export JSON into replay-ready CSV rows.",
    )
    parser.add_argument("--input-json", required=True, help="Path to exported access-code JSON.")
    parser.add_argument(
        "--registry-source",
        required=True,
        help="Directory containing items.json/questionnaires.json/scales.json.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for CSV and coverage outputs.")
    parser.add_argument(
        "--access-code",
        help="Access code override. Defaults to the input JSON filename stem.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = build_questionnaire_replay_csv(
        input_json=args.input_json,
        registry_source=args.registry_source,
        out_dir=args.out_dir,
        access_code=args.access_code,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
