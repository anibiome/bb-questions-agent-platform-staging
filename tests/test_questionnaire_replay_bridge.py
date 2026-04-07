from __future__ import annotations

import csv
import json
from pathlib import Path

from questions_agent_platform.tools.build_questionnaire_replay_csv import (
    build_questionnaire_replay_csv,
    match_source_question,
)
from questions_agent_platform.tools.replay_daily_scan_answers import _stage_registry_bundle
from questions_agent_platform.pipeline.registry import Registry, Item, Questionnaire, Scale, ScaleItem, save_registry, validate_registry_bundle


def _build_registry() -> Registry:
    return validate_registry_bundle(
        {
            "version": "v1",
            "items": [
                {
                    "id": "dass_wind_down",
                    "text": "I found it hard to wind down in the past 30 days.",
                    "response_type": "dass_0_3",
                },
                {
                    "id": "panas_interested",
                    "text": "How often have you felt interested in things that you do over the past month?",
                    "response_type": "affect_0_4",
                },
                {
                    "id": "qols_health",
                    "text": "How satisfied are you with your health - being physically fit and vigorous?",
                    "response_type": "satisfaction_0_6",
                },
            ],
            "questionnaires": [
                {"id": "q_dass_21_stress", "version": "1", "name": "DASS-21-stress"},
                {"id": "q_panas_sf_positive", "version": "1", "name": "PANAS-SF-positive"},
                {"id": "q_qols_flanagan", "version": "1", "name": "QOLS Flanagan"},
            ],
            "scales": [
                {
                    "id": "scale_dass_21_stress",
                    "questionnaire_id": "q_dass_21_stress",
                    "version": "1",
                    "name": "DASS-21-stress",
                    "response_type": "dass_0_3",
                    "min_items_required": 1,
                    "unlock_window_days": 0,
                    "retest_interval_days": 90,
                    "scoring": {
                        "method": "sum",
                        "normalize_min": 0.0,
                        "normalize_max": 42.0,
                    },
                    "items": [{"item_id": "dass_wind_down"}],
                },
                {
                    "id": "scale_panas_sf_positive",
                    "questionnaire_id": "q_panas_sf_positive",
                    "version": "1",
                    "name": "PANAS-SF-positive",
                    "response_type": "affect_0_4",
                    "min_items_required": 1,
                    "unlock_window_days": 0,
                    "retest_interval_days": 90,
                    "scoring": {
                        "method": "sum",
                        "normalize_min": 9.0,
                        "normalize_max": 45.0,
                    },
                    "items": [{"item_id": "panas_interested"}],
                },
                {
                    "id": "scale_qols_flanagan",
                    "questionnaire_id": "q_qols_flanagan",
                    "version": "1",
                    "name": "QOLS Flanagan",
                    "response_type": "satisfaction_0_6",
                    "min_items_required": 1,
                    "unlock_window_days": 0,
                    "retest_interval_days": 90,
                    "scoring": {
                        "method": "sum",
                        "normalize_min": 15.0,
                        "normalize_max": 105.0,
                    },
                    "items": [{"item_id": "qols_health"}],
                },
            ],
        }
    )


def test_match_source_question_handles_questionnaire_text_variants() -> None:
    registry = _build_registry()

    dass = match_source_question(
        registry=registry,
        source_questionnaire_id="DASS-21",
        question_text="I found it hard to wind down.",
    )
    assert dass.candidate is not None
    assert dass.candidate.item_id == "dass_wind_down"

    panas = match_source_question(
        registry=registry,
        source_questionnaire_id="PANAS-SF",
        question_text="Indicate the extent you have felt interested in things that you do over the past month.",
    )
    assert panas.candidate is not None
    assert panas.candidate.item_id == "panas_interested"

    qols = match_source_question(
        registry=registry,
        source_questionnaire_id="QOLS",
        question_text="How satisfied are you with health - being physically fit and vigorous?",
    )
    assert qols.candidate is not None
    assert qols.candidate.item_id == "qols_health"

    casp = match_source_question(
        registry=registry,
        source_questionnaire_id="CASP-19",
        question_text="I feel free to plan for the future.",
    )
    assert casp.candidate is None
    assert casp.reason == "unsupported_source_questionnaire"


def test_build_questionnaire_replay_csv_writes_supported_rows_and_reports_missing_questionnaire(
    tmp_path: Path,
) -> None:
    registry_root = tmp_path / "registry"
    save_registry(str(registry_root), _build_registry())
    registry_source = registry_root / "versions" / "v1"

    input_json = tmp_path / "access.json"
    input_json.write_text(
        json.dumps(
            {
                "user": {"id": "user-1"},
                "sample_blocks": [
                    {
                        "sample": {"sample_barcode": "S1", "collection_date": "2026-04-05"},
                        "digital_data": {
                            "questionnaireAnswers": [
                                {
                                    "_docId": "doc-1",
                                    "userId": "user-1",
                                    "questionnaireId": "DASS-21",
                                    "date": "2026-04-05T10:00:00Z",
                                    "answers": [
                                        {
                                            "questionId": "src-1",
                                            "questionString": "I found it hard to wind down.",
                                            "intensity": 2.0,
                                            "scaleInverted": False,
                                            "textAnswer": "Applied to me a good part of the time",
                                        }
                                    ],
                                },
                                {
                                    "_docId": "doc-2",
                                    "userId": "user-1",
                                    "questionnaireId": "CASP-19",
                                    "date": "2026-04-05T10:00:00Z",
                                    "answers": [
                                        {
                                            "questionId": "src-2",
                                            "questionString": "I feel free to plan for the future.",
                                            "intensity": 3.0,
                                            "scaleInverted": False,
                                            "textAnswer": "Always",
                                        }
                                    ],
                                },
                            ]
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    summary = build_questionnaire_replay_csv(
        input_json=str(input_json),
        registry_source=str(registry_source),
        out_dir=str(out_dir),
        access_code="AC1",
    )

    with (out_dir / "questionnaire_answers_replay.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["question_id"] == "dass_wind_down"
    assert rows[0]["source_questionnaire_id"] == "DASS-21"

    with (out_dir / "questionnaire_answers_unmatched.csv").open("r", encoding="utf-8", newline="") as handle:
        unmatched = list(csv.DictReader(handle))
    assert len(unmatched) == 1
    assert unmatched[0]["source_questionnaire_id"] == "CASP-19"
    assert unmatched[0]["reason"] == "unsupported_source_questionnaire"

    assert summary["rows_written"] == 1
    assert summary["source_questionnaires"]["DASS-21"]["matched_rows"] == 1
    assert summary["source_questionnaires"]["CASP-19"]["unmatched_rows"] == 1
