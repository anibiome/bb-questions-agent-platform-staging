import tempfile
import unittest
from datetime import date
from pathlib import Path

from questions_agent_platform.contracts import (
    CANONICAL_CONTRACT_VERSION,
    QUESTION_EVIDENCE_CLASS,
    canonical_contract_version,
    validate_projection_payload_contract,
)
from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import connect, init_db
from questions_agent_platform.pipeline.demo import seed_demo_registry
from questions_agent_platform.pipeline.projection import (
    VECTOR_DIM,
    build_questions_projection_payload,
)
from questions_agent_platform.pipeline.service import get_or_create_daily_session
from questions_agent_platform.prod.schemas import (
    AnswerIn,
    DailyQuestionsSelectIn,
    PolicyContextIn,
    PolicyOutcomeUpdateIn,
)


class TestContracts(unittest.TestCase):
    def test_canonical_contract_version_aliases(self) -> None:
        self.assertEqual(canonical_contract_version(None), CANONICAL_CONTRACT_VERSION)
        self.assertEqual(canonical_contract_version("1"), CANONICAL_CONTRACT_VERSION)
        self.assertEqual(canonical_contract_version("1.0"), CANONICAL_CONTRACT_VERSION)
        self.assertEqual(canonical_contract_version("v1"), CANONICAL_CONTRACT_VERSION)
        with self.assertRaises(ValueError):
            canonical_contract_version("2.0")

    def test_policy_context_legacy_and_alias_versions(self) -> None:
        legacy = PolicyContextIn(
            anifold_z=[0.1, 0.2], z_uncertainty_diag=[0.3, 0.4], z_velocity=[0.0, 0.1]
        )
        self.assertEqual(legacy.schema_version, CANONICAL_CONTRACT_VERSION)

        alias = PolicyContextIn(
            schema_version="1",
            anifold_z=[0.1],
            z_uncertainty_diag=[0.2],
            z_velocity=[0.3],
        )
        self.assertEqual(alias.schema_version, CANONICAL_CONTRACT_VERSION)

        with self.assertRaises(Exception):
            PolicyContextIn(anifold_z=[0.1, 0.2], z_uncertainty_diag=[0.3])

    def test_select_request_strict_and_mode_validation(self) -> None:
        req = DailyQuestionsSelectIn(selection_mode="POLICY_LIVE", schema_version="v1")
        self.assertEqual(req.selection_mode, "policy_live")
        self.assertEqual(req.schema_version, CANONICAL_CONTRACT_VERSION)
        with self.assertRaises(Exception):
            DailyQuestionsSelectIn(selection_mode="invalid_mode")

    def test_policy_outcome_legacy_and_vector_guards(self) -> None:
        legacy = PolicyOutcomeUpdateIn(decision_id="d1")
        self.assertEqual(legacy.schema_version, CANONICAL_CONTRACT_VERSION)

        alias = PolicyOutcomeUpdateIn(
            schema_version="1",
            decision_id="d2",
            z_before=[0.1, 0.2],
            z_after=[0.3, 0.4],
            uncertainty_before_diag=[0.5, 0.6],
            uncertainty_after_diag=[0.7, 0.8],
        )
        self.assertEqual(alias.schema_version, CANONICAL_CONTRACT_VERSION)

        with self.assertRaises(Exception):
            PolicyOutcomeUpdateIn(
                decision_id="d3",
                z_before=[0.1, 0.2],
                z_after=[0.3],
            )

    def test_answer_schema_allows_decline_without_numeric_value(self) -> None:
        declined = AnswerIn(
            client_event_id="evt_decline_1",
            item_id="item_x",
            raw={"permanently_declined": True},
        )
        self.assertIsNone(declined.value)

        with self.assertRaises(Exception):
            AnswerIn(
                client_event_id="evt_missing_value",
                item_id="item_x",
            )

    def test_projection_contract_contains_join_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            cfg = QuestionsAgentConfig(
                database_path=db_path, registry_root=registry_root
            )

            init_db(db_path)
            seed_demo_registry(registry_root)

            with connect(db_path) as conn:
                session = get_or_create_daily_session(
                    conn,
                    cfg=cfg,
                    registry_root=registry_root,
                    user_id="u_contract",
                    day=date(2026, 2, 6),
                )
                payload = build_questions_projection_payload(
                    conn,
                    user_id="u_contract",
                    registry_version=session.registry_version,
                    day=date(2026, 2, 6),
                )

            contract = payload.get("contract") or {}
            join_keys = contract.get("join_keys") or {}
            self.assertEqual(
                str(contract.get("schema_version")), CANONICAL_CONTRACT_VERSION
            )
            self.assertEqual(str(join_keys.get("subject_id")), "u_contract")
            self.assertEqual(str(join_keys.get("session_id")), session.session_id)
            self.assertEqual(payload["evidence_class"], QUESTION_EVIDENCE_CLASS)
            self.assertIn(
                "behavioral_question_runtime", payload["claim_boundary"]["allowed_use"]
            )
            self.assertFalse(
                payload["claim_boundary"]["blocked_claims"]["raw_biological_truth"]
            )
            self.assertFalse(
                payload["claim_boundary"]["blocked_claims"]["clinical_diagnosis"]
            )
            self.assertFalse(
                payload["claim_boundary"]["blocked_claims"]["treatment_recommendation"]
            )
            self.assertIn(
                "explicit_promotion_approval",
                payload["claim_boundary"]["promotion_required_gates"],
            )
            q = payload["modality_projections"]["questionnaires"]
            self.assertEqual(len(q["projection"]), VECTOR_DIM)
            self.assertEqual(len(q["uncertainty"]["diag"]), VECTOR_DIM)
            self.assertEqual(len(q["velocity"]), VECTOR_DIM)
            self.assertEqual(len(q["attractor_candidate"]), VECTOR_DIM)

            missing_boundary = dict(payload)
            missing_boundary.pop("claim_boundary")
            with self.assertRaises(ValueError):
                validate_projection_payload_contract(
                    missing_boundary, vector_dim=VECTOR_DIM
                )


if __name__ == "__main__":
    unittest.main()
