import unittest
from datetime import date

from questions_agent_platform.pipeline.selection import Candidate as PipelineCandidate
from questions_agent_platform.pipeline.selection import CandidateSet as PipelineCandidateSet
from questions_agent_platform.policy.runtime import execute_policy_selection


class TestPolicyRuntime(unittest.TestCase):
    def _candidate_set(self) -> PipelineCandidateSet:
        return PipelineCandidateSet(
            date="2026-02-16",
            k_core=2,
            extra_batch_size=1,
            max_extra_batches=3,
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
                PipelineCandidate(
                    item_id="opt_2",
                    item_type="scale_item",
                    scale_ids=("s_opt",),
                    deterministic_score=2.0,
                    deterministic_rank=3,
                    constraint_tags=("optional",),
                    reason_codes=(),
                    features={"multiplex_count": 1, "novelty_days": 2, "expected_burden": 1.0, "sensitivity": "low"},
                ),
            ),
            ordered_candidate_item_ids=("anchor_1", "opt_1", "opt_2"),
            excluded_item_ids=(),
            near_unlock_scales=(),
            due_retest_scales=(),
            drift_scales=(),
            timeframe="last_7_days",
        )

    def test_execute_policy_shadow_keeps_deterministic_served_set(self) -> None:
        result = execute_policy_selection(
            pipeline_candidate_set=self._candidate_set(),
            user_id="u1",
            day=date(2026, 2, 16),
            deterministic_baseline_selected=("anchor_1", "opt_1"),
            selection_mode="policy_shadow",
            runtime_mode="shadow",
            identity_mask_id=None,
            context_obj={},
            policy_root="",
            policy_default_version="v1",
            epsilon_explore=0.0,
            include_context_snapshot=False,
            include_candidate_set_snapshot=False,
        )
        self.assertEqual(result.served_core_item_ids, ("anchor_1", "opt_1"))
        self.assertEqual(result.summary["mode"], "shadow")
        self.assertEqual(result.decision_row_payload["selection_mode"], "policy_shadow")

    def test_execute_policy_live_serves_policy_selected_set(self) -> None:
        result = execute_policy_selection(
            pipeline_candidate_set=self._candidate_set(),
            user_id="u1",
            day=date(2026, 2, 16),
            deterministic_baseline_selected=("anchor_1", "opt_2"),
            selection_mode="policy_live",
            runtime_mode="live",
            identity_mask_id="mask-default",
            context_obj={"allow_context_batches": True},
            policy_root="",
            policy_default_version="v1",
            epsilon_explore=0.0,
            include_context_snapshot=True,
            include_candidate_set_snapshot=True,
        )
        self.assertEqual(len(result.served_core_item_ids), 2)
        self.assertIn("anchor_1", result.served_core_item_ids)
        self.assertEqual(result.decision_row_payload["selection_mode"], "policy_live")
        self.assertIsNotNone(result.decision_row_payload["context_json"])
        self.assertIsNotNone(result.decision_row_payload["candidate_set_json"])


if __name__ == "__main__":
    unittest.main()
