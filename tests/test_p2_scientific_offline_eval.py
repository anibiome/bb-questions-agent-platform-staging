import json
import tempfile
import unittest
import uuid
from datetime import date, timedelta
from pathlib import Path

from questions_agent_platform.pipeline.db import connect, ensure_user, init_db
from questions_agent_platform.pipeline.time_utils import now_iso
from questions_agent_platform.tools.run_scientific_offline_eval import build_scientific_offline_eval_report


def _seed_policy_eval_logs(db_path: str, *, user_id: str, n_days: int = 8) -> None:
    start = date(2026, 2, 1)
    with connect(db_path) as conn:
        ensure_user(conn, user_id)
        for i in range(n_days):
            day = start + timedelta(days=i)
            decision_id = str(uuid.uuid4())
            context = {
                "anifold_z": [0.1 + 0.01 * i, 0.2 + 0.02 * i, 0.3 + 0.01 * i],
                "z_uncertainty_diag": [0.4 - 0.01 * i, 0.5 - 0.01 * i, 0.6 - 0.01 * i],
                "z_velocity": [0.02, 0.01, 0.00],
                "z_distance_to_attractor": 0.25 + 0.01 * i,
                "completion_rate_7d": 0.80,
                "completion_rate_14d": 0.82,
                "completion_rate_30d": 0.84,
                "burden_ms_median_14d": 900.0,
                "day_of_week": i % 7,
                "safety_trigger_active": False,
                "allow_context_batches": True,
            }
            candidates = [
                {
                    "item_id": "item_a",
                    "item_type": "anchor",
                    "scale_ids": ["scale_a"],
                    "deterministic_score": 0.9,
                    "constraint_tags": ["mandatory"],
                    "reason_codes": ["anchor_continuity"],
                    "features": {
                        "multiplex_count": 3,
                        "novelty_days": 5,
                        "expected_burden": 1.0,
                        "missing_to_unlock": 1,
                        "retest_due": 0,
                        "drift_relevance": 0.7,
                        "sensitivity": "low",
                    },
                },
                {
                    "item_id": "item_b",
                    "item_type": "unlock_item",
                    "scale_ids": ["scale_b"],
                    "deterministic_score": 0.8,
                    "constraint_tags": ["optional"],
                    "reason_codes": ["near_unlock:scale_b"],
                    "features": {
                        "multiplex_count": 2,
                        "novelty_days": 8,
                        "expected_burden": 1.2,
                        "missing_to_unlock": 1,
                        "retest_due": 0,
                        "drift_relevance": 0.6,
                        "sensitivity": "low",
                    },
                },
                {
                    "item_id": "item_c",
                    "item_type": "drift_probe",
                    "scale_ids": ["scale_c"],
                    "deterministic_score": 0.7,
                    "constraint_tags": ["optional"],
                    "reason_codes": ["rapid_drift_probe"],
                    "features": {
                        "multiplex_count": 1,
                        "novelty_days": 20,
                        "expected_burden": 1.0,
                        "missing_to_unlock": 2,
                        "retest_due": 0,
                        "drift_relevance": 0.8,
                        "sensitivity": "low",
                    },
                },
            ]
            candidate_set = {
                "user_id": user_id,
                "day": day.isoformat(),
                "k_core": 2,
                "candidates": candidates,
                "mandatory_item_ids": ["item_a"],
                "deterministic_baseline_selected": ["item_a", "item_b"],
            }
            selected = ["item_a", "item_b"]
            conn.execute(
                """
                INSERT INTO policy_decisions(
                  decision_id, user_id, date, identity_mask_id, selection_mode, mode,
                  policy_version, feature_version, context_hash, candidate_set_hash,
                  context_json, candidate_set_json, selected_item_ids_json,
                  deterministic_baseline_selected_json, propensities_json,
                  explanations_json, counterfactuals_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    decision_id,
                    user_id,
                    day.isoformat(),
                    None,
                    "policy_live",
                    "live",
                    "v1",
                    "policy_features_v1",
                    f"ctx_hash_{i}",
                    f"cand_hash_{i}",
                    json.dumps(context, ensure_ascii=False),
                    json.dumps(candidate_set, ensure_ascii=False),
                    json.dumps(selected, ensure_ascii=False),
                    json.dumps(["item_a", "item_b"], ensure_ascii=False),
                    json.dumps({"__slate__": 0.5, "item_b": 0.5}, ensure_ascii=False),
                    json.dumps({"item_a": {"policy_score": 0.9}}, ensure_ascii=False),
                    json.dumps([{"item_id": "item_c", "policy_score": 0.3}], ensure_ascii=False),
                    now_iso(),
                ),
            )
            conn.execute(
                """
                INSERT INTO policy_outcomes(
                  decision_id, user_id, date, completion_rate, completed_core_count,
                  response_time_ms_median, reward_components_json, total_reward, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    decision_id,
                    user_id,
                    day.isoformat(),
                    1.0,
                    5,
                    950.0,
                    json.dumps({"components": {"completion": 1.0}}, ensure_ascii=False),
                    0.45 + 0.03 * i,
                    now_iso(),
                ),
            )
        conn.commit()


class TestP2ScientificOfflineEval(unittest.TestCase):
    def test_scientific_offline_eval_report_contains_ablations_and_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            policy_root = str(Path(td) / "policy")
            Path(policy_root).mkdir(parents=True, exist_ok=True)
            init_db(db_path)
            _seed_policy_eval_logs(db_path, user_id="u_eval", n_days=10)

            report = build_scientific_offline_eval_report(
                db_path=db_path,
                policy_root=policy_root,
                policy_version=None,
                min_date=None,
                max_date=None,
                include_shadow=False,
            )

            self.assertEqual(report["version"], "scientific_offline_eval_v1")
            self.assertIn("full_model", report)
            self.assertIn("ablations", report)
            self.assertIn("calibration", report)
            self.assertGreaterEqual(int(report["full_model"]["n_eval_rows"]), 1)
            self.assertGreaterEqual(int(report["calibration"]["n_eval_rows"]), 1)
            self.assertIn("without_z_features", report["ablations"])
            self.assertIn("without_uncertainty_features", report["ablations"])
            self.assertIn("without_z_and_uncertainty", report["ablations"])
            self.assertEqual(
                report["ablations"]["without_z_features"]["ablation"]["dropped_context_keys"],
                ["anifold_z", "z_velocity", "z_distance_to_attractor"],
            )
            self.assertIsInstance(report["calibration"]["bins"], list)


if __name__ == "__main__":
    unittest.main()
