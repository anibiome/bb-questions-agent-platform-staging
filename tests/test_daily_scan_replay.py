from __future__ import annotations

import csv
import json
from pathlib import Path

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.registry import (
    Item,
    Questionnaire,
    Registry,
    Scale,
    ScaleItem,
    save_registry,
    validate_registry_bundle,
)
from questions_agent_platform.tools.replay_daily_scan_answers import run_daily_scan_replay


def test_validate_registry_bundle_preserves_zero_day_windows() -> None:
    registry = validate_registry_bundle(
        {
            "version": "v1",
            "items": [
                {
                    "id": "proto_q_1",
                    "text": "How steady do you feel today?",
                    "response_type": "likert_0_4",
                }
            ],
            "questionnaires": [
                {"id": "q1", "version": "1", "name": "Daily Scan"}
            ],
            "scales": [
                {
                    "id": "scale_daily_scan",
                    "questionnaire_id": "q1",
                    "version": "1",
                    "name": "Daily Scan Scale",
                    "response_type": "likert_0_4",
                    "min_items_required": 1,
                    "unlock_window_days": 0,
                    "retest_interval_days": 0,
                    "scoring": {
                        "method": "sum",
                        "normalize_min": 0.0,
                        "normalize_max": 4.0,
                    },
                    "items": [{"item_id": "proto_q_1"}],
                }
            ],
        }
    )
    scale = registry.scales["scale_daily_scan"]
    assert scale.unlock_window_days == 0
    assert scale.retest_interval_days == 0


def test_run_daily_scan_replay_exports_scale_scores_and_projections(tmp_path: Path) -> None:
    db_path = tmp_path / "qa.sqlite"
    registry_root = tmp_path / "registry"
    source_root = tmp_path / "registry_source"
    source_dir = source_root / "versions" / "v1"
    out_dir = tmp_path / "out"
    input_csv = tmp_path / "daily_scan_answers.csv"

    registry = Registry(
        version="v1",
        items={
            "proto_q_1": Item(
                id="proto_q_1",
                text="How steady do you feel today?",
                response_type="likert_0_4",
            )
        },
        questionnaires={
            "q1": Questionnaire(id="q1", version="1", name="Daily Scan")
        },
        scales={
            "scale_daily_scan": Scale(
                id="scale_daily_scan",
                questionnaire_id="q1",
                version="1",
                name="Daily Scan Scale",
                method="sum",
                min_items_required=1,
                unlock_window_days=0,
                retest_interval_days=0,
                response_type="likert_0_4",
                normalize_min=0.0,
                normalize_max=4.0,
                items=(ScaleItem(item_id="proto_q_1"),),
                citation="internal test",
            )
        },
    )
    save_registry(str(source_root), registry)

    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "access_code": "AC1",
                "user_id": "user-1",
                "scan_doc_id": "scan-1",
                "scan_time": "2026-04-05T10:00:00Z",
                "answer_timestamp": "2026-04-05T10:00:00Z",
                "scan_date_utc": "2026-04-05",
                "scan_date_local": "2026-04-05",
                "question_index": "0",
                "question_id": "proto_q_1",
                "question_string": "How steady do you feel today?",
                "answer_value": "3",
                "answer_json": json.dumps({"questionId": "proto_q_1", "intensity": 3}),
            }
        )
        writer.writerow(
            {
                "access_code": "AC1",
                "user_id": "user-1",
                "scan_doc_id": "scan-2",
                "scan_time": "2026-04-06T10:00:00Z",
                "answer_timestamp": "2026-04-06T10:00:00Z",
                "scan_date_utc": "2026-04-06",
                "scan_date_local": "2026-04-06",
                "question_index": "0",
                "question_id": "proto_q_1",
                "question_string": "How steady do you feel today?",
                "answer_value": "4",
                "answer_json": json.dumps({"questionId": "proto_q_1", "intensity": 4}),
            }
        )

    cfg = QuestionsAgentConfig(
        database_path=str(db_path),
        registry_root=str(registry_root),
        host="127.0.0.1",
        port=0,
    )
    summary = run_daily_scan_replay(
        cfg=cfg,
        input_csv=str(input_csv),
        out_dir=str(out_dir),
        registry_source=str(source_dir),
        registry_version="v1",
    )

    assert summary["rows_read"] == 2
    assert summary["rows_replayed"] == 2
    assert summary["session_count"] == 2
    assert summary["scale_score_count"] >= 2
    assert (out_dir / "daily_scan_replay_scale_scores.csv").exists()
    assert (out_dir / "daily_scan_replay_projection_payloads.jsonl").exists()

    scale_rows = (out_dir / "daily_scan_replay_scale_scores.csv").read_text(encoding="utf-8")
    assert "scale_daily_scan" in scale_rows
    assert "internal test" in scale_rows

    projection_lines = [
        json.loads(line)
        for line in (out_dir / "daily_scan_replay_projection_payloads.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(projection_lines) == 2
    assert projection_lines[-1]["derived_features"]["scale_scores"]["scale_daily_scan"]["normalized_score"] == 100.0
