from __future__ import annotations

import csv
import json
from pathlib import Path

from questions_agent_platform.tools.score_questionnaire_timeseries import (
    score_questionnaire_timeseries,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_score_questionnaire_timeseries_writes_all_sessions(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    _write_json(
        registry_dir / "items.json",
        [
            {"id": "item_1", "text": "Item 1", "response_type": "likert_0_3"},
            {"id": "item_2", "text": "Item 2", "response_type": "likert_0_3"},
        ],
    )
    _write_json(
        registry_dir / "questionnaires.json",
        [
            {
                "id": "q_test",
                "version": "1",
                "name": "Test questionnaire",
                "domains": ["test"],
                "license": "open_access",
            }
        ],
    )
    _write_json(
        registry_dir / "scales.json",
        [
            {
                "id": "scale_test",
                "questionnaire_id": "q_test",
                "version": "1",
                "name": "Test Scale",
                "response_type": "likert_0_3",
                "min_items_required": 2,
                "unlock_window_days": 14,
                "retest_interval_days": 90,
                "scoring": {"method": "sum", "normalize_min": 0, "normalize_max": 6},
                "items": [
                    {"item_id": "item_1"},
                    {"item_id": "item_2"},
                ],
            }
        ],
    )

    input_csv = tmp_path / "replay.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "access_code",
                "user_id",
                "scan_date_local",
                "question_id",
                "answer_value",
                "answer_timestamp",
                "target_questionnaire_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "access_code": "ABC123",
                    "user_id": "user-1",
                    "scan_date_local": "2025-01-01",
                    "question_id": "item_1",
                    "answer_value": "1",
                    "answer_timestamp": "2025-01-01T09:00:00Z",
                    "target_questionnaire_id": "q_test",
                },
                {
                    "access_code": "ABC123",
                    "user_id": "user-1",
                    "scan_date_local": "2025-01-01",
                    "question_id": "item_2",
                    "answer_value": "2",
                    "answer_timestamp": "2025-01-01T09:05:00Z",
                    "target_questionnaire_id": "q_test",
                },
                {
                    "access_code": "ABC123",
                    "user_id": "user-1",
                    "scan_date_local": "2025-01-10",
                    "question_id": "item_1",
                    "answer_value": "3",
                    "answer_timestamp": "2025-01-10T09:00:00Z",
                    "target_questionnaire_id": "q_test",
                },
                {
                    "access_code": "ABC123",
                    "user_id": "user-1",
                    "scan_date_local": "2025-01-10",
                    "question_id": "item_2",
                    "answer_value": "3",
                    "answer_timestamp": "2025-01-10T09:05:00Z",
                    "target_questionnaire_id": "q_test",
                },
            ]
        )

    summary = score_questionnaire_timeseries(
        input_csv=str(input_csv),
        registry_source=str(registry_dir),
        out_dir=str(tmp_path / "out"),
    )

    assert summary["rows_read"] == 4
    assert summary["score_rows_written"] == 2

    with Path(summary["output_csv"]).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["session_date"] for row in rows] == ["2025-01-01", "2025-01-10"]
    assert [float(row["raw_score"]) for row in rows] == [3.0, 6.0]
    assert [float(row["normalized_score"]) for row in rows] == [50.0, 100.0]
    assert rows[0]["delta_vs_prev"] == ""
    assert float(rows[1]["delta_vs_prev"]) == 50.0


def test_score_questionnaire_timeseries_prefers_latest_answer_per_item(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    _write_json(
        registry_dir / "items.json",
        [
            {"id": "item_1", "text": "Item 1", "response_type": "likert_0_3"},
            {"id": "item_2", "text": "Item 2", "response_type": "likert_0_3"},
        ],
    )
    _write_json(
        registry_dir / "questionnaires.json",
        [
            {
                "id": "q_test",
                "version": "1",
                "name": "Test questionnaire",
                "domains": ["test"],
                "license": "open_access",
            }
        ],
    )
    _write_json(
        registry_dir / "scales.json",
        [
            {
                "id": "scale_test",
                "questionnaire_id": "q_test",
                "version": "1",
                "name": "Test Scale",
                "response_type": "likert_0_3",
                "min_items_required": 2,
                "unlock_window_days": 14,
                "retest_interval_days": 90,
                "scoring": {"method": "sum", "normalize_min": 0, "normalize_max": 6},
                "items": [
                    {"item_id": "item_1"},
                    {"item_id": "item_2"},
                ],
            }
        ],
    )

    input_csv = tmp_path / "replay.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "access_code",
                "user_id",
                "scan_date_local",
                "question_id",
                "answer_value",
                "answer_timestamp",
                "target_questionnaire_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "access_code": "ABC123",
                    "user_id": "user-1",
                    "scan_date_local": "2025-01-01",
                    "question_id": "item_1",
                    "answer_value": "1",
                    "answer_timestamp": "2025-01-01T09:00:00Z",
                    "target_questionnaire_id": "q_test",
                },
                {
                    "access_code": "ABC123",
                    "user_id": "user-1",
                    "scan_date_local": "2025-01-01",
                    "question_id": "item_1",
                    "answer_value": "3",
                    "answer_timestamp": "2025-01-01T09:10:00Z",
                    "target_questionnaire_id": "q_test",
                },
                {
                    "access_code": "ABC123",
                    "user_id": "user-1",
                    "scan_date_local": "2025-01-01",
                    "question_id": "item_2",
                    "answer_value": "2",
                    "answer_timestamp": "2025-01-01T09:05:00Z",
                    "target_questionnaire_id": "q_test",
                },
            ]
        )

    summary = score_questionnaire_timeseries(
        input_csv=str(input_csv),
        registry_source=str(registry_dir),
        out_dir=str(tmp_path / "out"),
    )

    assert summary["score_rows_written"] == 1
    with Path(summary["output_csv"]).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert float(rows[0]["raw_score"]) == 5.0
    assert rows[0]["first_answer_timestamp"] == "2025-01-01T09:05:00Z"
    assert rows[0]["last_answer_timestamp"] == "2025-01-01T09:10:00Z"
