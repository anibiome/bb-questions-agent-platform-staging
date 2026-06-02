from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


class ManifestControlsTest(unittest.TestCase):
    def test_manifest_bounds_questions_agent_source_truth(self) -> None:
        manifest = read("ani-manifest.yaml")
        anifesto = read("anifesto.yaml")

        self.assertIn("source_of_truth: false", manifest)
        self.assertIn("behavioral_question_runtime_and_runtime_bridge_contracts_only", manifest)
        self.assertIn("behavioral_or_derived_evidence_not_raw_biological_or_clinical_truth", manifest)
        for token in (
            "participant_source_truth: false",
            "raw_biological_truth: false",
            "raw_omics_truth: false",
            "clinical_diagnosis: false",
            "treatment_recommendation: false",
            "treatment_efficacy: false",
            "patent_ready_proof: false",
            "regulated_software: false",
            "explicit_promotion_approval",
        ):
            self.assertIn(token, manifest)

        self.assertIn("behavioral_question_runtime_and_bridge_contract_surface", anifesto)
        self.assertIn("behavioral_or_derived_evidence_not_raw_biological_or_clinical_truth", anifesto)
        self.assertIn("participant_source_truth", anifesto)
        self.assertIn("raw_biological_truth", anifesto)
        self.assertIn("patent_ready_proof", anifesto)

    def test_front_door_bounds_behavioral_runtime_claims(self) -> None:
        readme = read("README.md")
        code_room = read("CODE_ROOM_README.md")

        self.assertIn("## Evidence Boundary", readme)
        self.assertIn("behavioral-question runtime", readme)
        self.assertIn("not raw biological source truth", readme)
        self.assertIn("treatment efficacy", readme)
        self.assertIn("patent-ready proof", readme)

        self.assertIn("participant source truth", code_room)
        self.assertIn("raw biological truth", code_room)
        self.assertIn("treatment efficacy", code_room)
        self.assertIn("patent-ready proof", code_room)
        self.assertIn("explicit_promotion_approval", code_room)

    def test_compliance_and_ci_run_manifest_controls(self) -> None:
        workflow_text = read(".github/workflows/code-room-readiness.yml") + read(
            ".github/workflows/regulated-review.yml"
        )
        result = subprocess.run(
            ["python3", "tools/repo_compliance_check.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("tests.test_manifest_controls", workflow_text)
        self.assertIn("workflow_dispatch", workflow_text)
        self.assertIn("permissions:", workflow_text)
        self.assertIn("contents: read", workflow_text)
        self.assertIn("actions/checkout@v6", workflow_text)
        self.assertIn("actions/setup-python@v6", workflow_text)

    def test_stale_readiness_language_is_not_active(self) -> None:
        active_docs = {
            "ARCHITECTURE_MAP_V2.md": ["production-ready baseline"],
            "PRODUCTION_READINESS_CHECKLIST_V2.md": ["production-ready v2 baseline"],
            "CHANGE_CONTROL_PLAN.md": ["production-ready runtime surfaces"],
            "DEV_HANDOFF_QUICKSTART.md": [
                "53 API endpoints",
                "production FastAPI/Postgres stack",
                "781 tests",
                "deployable reference implementation",
                "You are production-ready for this scope",
            ],
            "DEV_IMPLEMENTATION_GUIDE.md": [
                "Production deployment (FastAPI",
                "all production endpoints",
                "721 tests",
                "for production)",
            ],
            "BRUNO_TECHNICAL_OVERVIEW.md": [
                "721 tests",
                "16,454 lines",
                "5,642 lines",
            ],
            "README.md": [
                "questions_agent_platform/SCHEMA_CONTRACTS_V1.md",
                "questions_agent_platform/DEV_EXECUTION_ORDER_INTEGRATION.md",
                "questions_agent_platform/DEV_HANDOFF_QUICKSTART.md",
                "questions_agent_platform/ARCHITECTURE_MAP_V2.md",
            ],
            "CURRENT_STATE_AND_DEEPTECH_ROADMAP_2026-02-07.md": [
                "questions_agent_platform/SCHEMA_CONTRACTS_V1.md",
            ],
            "AGENTS.md": [
                "questions_agent_platform/SCHEMA_CONTRACTS_V1.md",
                "questions_agent_platform/DEV_EXECUTION_ORDER_INTEGRATION.md",
                "/Users/brunobalen/Documents/ANI_DEEP_LATENT_RECONSTRUCTION_PROTOCOL.md",
                "/Users/brunobalen/Documents/ANI_DATA_TRUTH_PROTOCOL.md",
            ],
            "tools/generate_collateral.py": [
                "deliver a production-ready Questions Agent",
                "Production deployment stack",
                "We shipped a production service",
                "This is deployable as a standalone service",
                "This v1 is deployable",
            ],
        }

        for rel_path, phrases in active_docs.items():
            text = read(rel_path)
            for phrase in phrases:
                self.assertNotIn(phrase, text, rel_path)


if __name__ == "__main__":
    unittest.main()
