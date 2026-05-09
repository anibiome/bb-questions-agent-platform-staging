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


if __name__ == "__main__":
    unittest.main()
