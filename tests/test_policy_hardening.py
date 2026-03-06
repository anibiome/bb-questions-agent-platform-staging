from __future__ import annotations

from datetime import date
import json

from questions_agent_platform.policy.cli import (
    _candidate_set_from_json as cli_candidate_set_from_json,
    _context_from_json as cli_context_from_json,
)
from questions_agent_platform.policy.offline_eval import (
    _candidate_set_from_json as offline_candidate_set_from_json,
    _context_from_json as offline_context_from_json,
)
from questions_agent_platform.policy.hashing import stable_hash_floats
from questions_agent_platform.policy.rewards import compute_reward
from questions_agent_platform.policy.runtime import build_policy_context
from questions_agent_platform.policy.types import (
    CandidateItem,
    CandidateSet,
    CounterfactualExplanation,
    PolicyContext,
    PolicyDecision,
    SelectedItemExplanation,
)
from questions_agent_platform.policy.world_model import build_measurement_actions, response_validity_snapshot
from questions_agent_platform.pipeline.config import load_config
from questions_agent_platform.prod.settings import load_settings


def test_world_model_hardens_invalid_numeric_and_scalar_fields() -> None:
    candidate_set = CandidateSet(
        user_id="u1",
        day=date(2026, 3, 6),
        k_core=1,
        mandatory_item_ids=(),
        deterministic_baseline_selected=("item_1",),
        candidates=(
            CandidateItem(
                item_id="item_1",
                item_type="scale_item",
                scale_ids=(1, "sleep"),  # type: ignore[arg-type]
                deterministic_score="8.0",  # type: ignore[arg-type]
                constraint_tags=("optional",),
                reason_codes=("near_unlock:sleep",),
                features={
                    "expected_burden": "2.5",
                    "multiplex_count": "2.0",
                    "novelty_days": "30.0",
                    "sensitivity": "medium",
                },
            ),
        ),
    )
    decision = PolicyDecision(
        decision_id="dec_001",
        user_id="u1",
        day=date(2026, 3, 6),
        policy_version="v1",
        feature_version="f1",
        mode="shadow",
        selected_item_ids=("item_1",),
        propensities=None,
        explanations=(
            SelectedItemExplanation(
                item_id="item_1",
                selection_rank=1,
                policy_score="bad",  # type: ignore[arg-type]
                deterministic_score="8.0",  # type: ignore[arg-type]
                reason_codes=("near_unlock:sleep",),
                constraint_justification=(),
                top_feature_contributions=(),
            ),
        ),
        counterfactuals=(CounterfactualExplanation(item_id="item_2", policy_score=0.1, reason_codes=()),),
        deterministic_baseline_selected=("item_1",),
        context_hash="ctx",
        candidate_set_hash="cand",
    )
    context = PolicyContext(
        response_validity_score="bad",  # type: ignore[arg-type]
        response_validity_tier="medium",
        biological_coherence_score="0.61",  # type: ignore[arg-type]
        biological_decoherence_radius="0.39",  # type: ignore[arg-type]
    )

    snapshot = response_validity_snapshot(context)
    actions = build_measurement_actions(
        decision=decision,
        candidate_set=candidate_set,
        context=context,
    )

    assert snapshot == {"score": None, "tier": "medium"}
    assert actions[0]["expected_information_gain"] == 0.63
    assert actions[0]["parameters"]["scale_ids"] == ["1", "sleep"]
    assert actions[0]["provenance"]["response_validity"]["tier"] == "medium"


def test_reward_hashing_and_context_accept_string_inputs() -> None:
    reward = compute_reward(
        completion_rate="0.8",  # type: ignore[arg-type]
        response_time_ms_median="1200.0",  # type: ignore[arg-type]
        unlock_count="2.0",  # type: ignore[arg-type]
        uncertainty_before_mean="0.9",  # type: ignore[arg-type]
        uncertainty_after_mean="0.4",  # type: ignore[arg-type]
        weights={"completion": "2.0", "burden_penalty": "bad"},
        overrides={"weights": {"unlock_value": "0.5", "uncertainty_reduction": None}, "components": {"bonus": "1.25"}},
    )
    ctx = build_policy_context(
        day=date(2026, 3, 6),
        context_obj={
            "allow_context_batches": "false",
            "safety_trigger_active": "true",
            "completion_rate_14d": "0.7",
            "response_validity_score": "0.9",
            "biological_coherence_score": "0.61",
        },
        identity_mask_id="",
    )

    assert reward.weights["completion"] == 2.0
    assert reward.weights["burden_penalty"] == 0.2
    assert reward.weights["unlock_value"] == 0.5
    assert reward.components["unlock_value"] == 2.0 / 3.0
    assert reward.total > 1.0
    assert stable_hash_floats(["1.2345", 2], quantize="3.0") == stable_hash_floats([1.2345, 2.0], quantize=3)
    assert ctx.allow_context_batches is False
    assert ctx.safety_trigger_active is True
    assert ctx.completion_rate_14d == 0.7
    assert ctx.response_validity_score == 0.9
    assert ctx.identity_mask_id is None


def test_load_settings_accepts_float_like_numeric_env(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite:///tmp.db")
    monkeypatch.setenv("REGISTRY_ROOT", "/tmp/registry")
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("PORT", "8080.0")
    monkeypatch.setenv("CORE_QUESTIONS_PER_DAY", "6.0")
    monkeypatch.setenv("POLICY_EPSILON_EXPLORE", "0.15")
    monkeypatch.setenv("POLICY_QUANTIZE_DECIMALS", "4.0")
    monkeypatch.setenv("SLO_WINDOW_DAYS", "21.0")
    monkeypatch.setenv("PHI_RETENTION_POLICY_CONTEXT_DAYS", "120.0")

    settings = load_settings()

    assert settings.port == 8080
    assert settings.core_questions_per_day == 6
    assert settings.policy_epsilon_explore == 0.15
    assert settings.policy_quantize_decimals == 4
    assert settings.slo_window_days == 21
    assert settings.phi_retention_policy_context_days == 120


def test_policy_json_loaders_harden_stringy_booleans_and_scalar_sequences() -> None:
    context_payload = {
        "safety_trigger_active": "false",
        "allow_context_batches": "false",
        "identity_mask_id": "mask_1",
    }
    candidate_payload = {
        "user_id": "u1",
        "day": "2026-03-06",
        "k_core": "5.0",
        "candidates": {
            "item_id": "item_1",
            "item_type": "scale_item",
            "scale_ids": "sleep",
            "deterministic_score": "bad",
            "constraint_tags": "optional",
            "reason_codes": "near_unlock:sleep",
            "features": {"expected_burden": 2.0},
        },
        "mandatory_item_ids": "item_1",
        "deterministic_baseline_selected": "item_1",
    }

    for context_loader, candidate_loader in (
        (cli_context_from_json, cli_candidate_set_from_json),
        (offline_context_from_json, offline_candidate_set_from_json),
    ):
        ctx = context_loader(context_payload)
        cand = candidate_loader(candidate_payload)

        assert ctx.safety_trigger_active is False
        assert ctx.allow_context_batches is False
        assert cand.k_core == 5
        assert len(cand.candidates) == 1
        assert cand.candidates[0].scale_ids == ("sleep",)
        assert cand.candidates[0].constraint_tags == ("optional",)
        assert cand.candidates[0].reason_codes == ("near_unlock:sleep",)
        assert cand.candidates[0].deterministic_score == 0.0
        assert cand.mandatory_item_ids == ("item_1",)
        assert cand.deterministic_baseline_selected == ("item_1",)


def test_pipeline_load_config_accepts_string_booleans_and_float_like_numbers(tmp_path) -> None:
    config_path = tmp_path / "qa_config.json"
    config_path.write_text(
        json.dumps(
            {
                "database_path": str(tmp_path / "qa.sqlite"),
                "registry_root": str(tmp_path / "registry"),
                "port": "8088.0",
                "dashboard_enabled": "false",
                "core_questions_per_day": "6.0",
                "extra_batch_size": "4.0",
                "policy_log_context_snapshot": "false",
                "policy_log_candidate_set_snapshot": "false",
                "policy_store_z_snapshot": "false",
                "policy_store_z_delta": "false",
                "policy_quantize_decimals": "4.0",
                "policy_auto_rollback_enabled": "false",
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(config_path))

    assert cfg.port == 8088
    assert cfg.dashboard_enabled is False
    assert cfg.core_questions_per_day == 6
    assert cfg.extra_batch_size == 4
    assert cfg.policy_log_context_snapshot is False
    assert cfg.policy_log_candidate_set_snapshot is False
    assert cfg.policy_store_z_snapshot is False
    assert cfg.policy_store_z_delta is False
    assert cfg.policy_quantize_decimals == 4
    assert cfg.policy_auto_rollback_enabled is False
