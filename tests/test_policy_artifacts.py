from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from questions_agent_platform.pipeline.db import init_db
from questions_agent_platform.pipeline.selection import Candidate as PipelineCandidate
from questions_agent_platform.pipeline.selection import CandidateSet as PipelineCandidateSet
from questions_agent_platform.policy.cli import _cmd_train_sqlite
from questions_agent_platform.policy.features import build_feature_mapping_v1
from questions_agent_platform.policy.offline_eval import evaluate_sqlite
from questions_agent_platform.policy.registry import PolicyArtifactError, resolve_policy_params
from questions_agent_platform.policy.runtime import execute_policy_selection


def _candidate_set() -> PipelineCandidateSet:
    return PipelineCandidateSet(
        date="2026-02-16",
        k_core=2,
        extra_batch_size=1,
        max_extra_batches=1,
        mandatory_item_ids=("anchor_1",),
        candidates=(
            PipelineCandidate(
                item_id="anchor_1",
                item_type="anchor",
                scale_ids=("s_anchor",),
                deterministic_score=10.0,
                deterministic_rank=1,
                constraint_tags=("mandatory",),
                reason_codes=("anchor_continuity",),
                features={"multiplex_count": 1, "novelty_days": 10, "expected_burden": 1.0, "sensitivity": "low"},
            ),
            PipelineCandidate(
                item_id="opt_1",
                item_type="scale_item",
                scale_ids=("s_opt",),
                deterministic_score=8.0,
                deterministic_rank=2,
                constraint_tags=("optional",),
                reason_codes=("near_unlock:s_opt",),
                features={"multiplex_count": 1, "novelty_days": 6, "expected_burden": 1.0, "sensitivity": "low"},
            ),
        ),
        ordered_candidate_item_ids=("anchor_1", "opt_1"),
        excluded_item_ids=(),
        near_unlock_scales=(),
        due_retest_scales=(),
        drift_scales=(),
        timeframe="last_7_days",
    )


def _write_broken_policy(root: Path, *, version: str = "v1") -> None:
    (root / "versions" / version).mkdir(parents=True, exist_ok=True)
    (root / "active_policy_version.json").write_text(
        json.dumps({"active_version": version}),
        encoding="utf-8",
    )
    (root / "versions" / version / "policy_params.json").write_text(
        "{not valid json",
        encoding="utf-8",
    )


def test_resolve_policy_params_bootstraps_default_only_for_empty_registry(tmp_path: Path) -> None:
    params = resolve_policy_params(
        str(tmp_path / "policy"),
        default_version="v1",
        mapping=build_feature_mapping_v1(),
        allow_bootstrap_default=True,
    )
    assert params.policy_version == "v1"


def test_resolve_policy_params_rejects_corrupt_artifact(tmp_path: Path) -> None:
    root = tmp_path / "policy"
    _write_broken_policy(root)
    with pytest.raises(PolicyArtifactError, match="unreadable"):
        resolve_policy_params(
            str(root),
            default_version="v1",
            mapping=build_feature_mapping_v1(),
            allow_bootstrap_default=True,
        )


def test_resolve_policy_params_rejects_missing_active_artifact(tmp_path: Path) -> None:
    root = tmp_path / "policy"
    root.mkdir(parents=True, exist_ok=True)
    (root / "active_policy_version.json").write_text(
        json.dumps({"active_version": "v9"}),
        encoding="utf-8",
    )
    with pytest.raises(PolicyArtifactError, match="missing"):
        resolve_policy_params(
            str(root),
            default_version="v1",
            mapping=build_feature_mapping_v1(),
            allow_bootstrap_default=True,
        )


def test_execute_policy_selection_rejects_corrupt_policy_artifact(tmp_path: Path) -> None:
    root = tmp_path / "policy"
    _write_broken_policy(root)

    with pytest.raises(PolicyArtifactError, match="unreadable"):
        execute_policy_selection(
            pipeline_candidate_set=_candidate_set(),
            user_id="u1",
            day=date(2026, 2, 16),
            deterministic_baseline_selected=("anchor_1", "opt_1"),
            selection_mode="policy_live",
            runtime_mode="live",
            identity_mask_id=None,
            context_obj={},
            policy_root=str(root),
            policy_default_version="v1",
            epsilon_explore=0.0,
            include_context_snapshot=False,
            include_candidate_set_snapshot=False,
        )


def test_offline_eval_rejects_corrupt_policy_artifact(tmp_path: Path) -> None:
    db_path = tmp_path / "qa.sqlite"
    init_db(str(db_path))
    root = tmp_path / "policy"
    _write_broken_policy(root)

    with pytest.raises(PolicyArtifactError, match="unreadable"):
        evaluate_sqlite(
            db_path=str(db_path),
            policy_root=str(root),
            include_shadow=False,
        )


def test_train_sqlite_rejects_corrupt_policy_artifact(tmp_path: Path) -> None:
    db_path = tmp_path / "qa.sqlite"
    init_db(str(db_path))
    root = tmp_path / "policy"
    _write_broken_policy(root)

    with pytest.raises(PolicyArtifactError, match="unreadable"):
        _cmd_train_sqlite(
            db_path=str(db_path),
            policy_root=str(root),
            in_version=None,
            out_version="v2",
            min_date=None,
            max_date=None,
        )
