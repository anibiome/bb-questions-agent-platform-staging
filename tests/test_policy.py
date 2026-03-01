import unittest
from datetime import date

from questions_agent_platform.policy.bandit import PolicyParams, PolicyRuntimeConfig, make_policy_decision
from questions_agent_platform.policy.features import build_feature_mapping_v1
from questions_agent_platform.policy.types import CandidateItem, CandidateSet, PolicyContext


class TestPolicy(unittest.TestCase):
    def test_policy_never_selects_blocked_items(self) -> None:
        mapping = build_feature_mapping_v1()
        params = PolicyParams(policy_version="v1", feature_version=mapping.feature_version, lambda_reg=1.0, A=(), b=())

        c1 = CandidateItem(
            item_id="anchor_1",
            item_type="anchor",
            scale_ids=(),
            deterministic_score=10.0,
            constraint_tags=("mandatory",),
            reason_codes=("anchor_continuity",),
            features={"multiplex_count": 1, "novelty_days": 30, "expected_burden": 1.0, "sensitivity": "low"},
        )
        c2 = CandidateItem(
            item_id="opt_1",
            item_type="unlock_item",
            scale_ids=("s1",),
            deterministic_score=5.0,
            constraint_tags=("optional",),
            reason_codes=("near_unlock:s1",),
            features={"multiplex_count": 1, "novelty_days": 10, "missing_to_unlock": 1, "expected_burden": 1.0, "sensitivity": "low"},
        )
        c3 = CandidateItem(
            item_id="blocked_1",
            item_type="retest_item",
            scale_ids=("s2",),
            deterministic_score=100.0,
            constraint_tags=("optional", "blocked"),
            reason_codes=("retest_due:s2",),
            features={"multiplex_count": 1, "novelty_days": 1, "expected_burden": 1.0, "sensitivity": "low"},
        )

        cs = CandidateSet(
            user_id="u1",
            day=date(2026, 2, 2),
            k_core=2,
            candidates=(c1, c2, c3),
            mandatory_item_ids=("anchor_1",),
            deterministic_baseline_selected=("anchor_1", "opt_1"),
        )
        ctx = PolicyContext(day_of_week=0)
        decision = make_policy_decision(
            candidate_set=cs,
            context=ctx,
            params=params,
            mapping=mapping,
            mode="live",
            runtime=PolicyRuntimeConfig(epsilon_explore=0.0, rng_seed=123, max_counterfactuals=5),
        )
        self.assertIn("anchor_1", decision.selected_item_ids)
        self.assertNotIn("blocked_1", decision.selected_item_ids)
        self.assertEqual(len(decision.selected_item_ids), 2)

    def test_fallback_when_mandatory_exceeds_budget(self) -> None:
        mapping = build_feature_mapping_v1()
        params = PolicyParams(policy_version="v1", feature_version=mapping.feature_version, lambda_reg=1.0, A=(), b=())
        candidates = []
        for i in range(3):
            candidates.append(
                CandidateItem(
                    item_id=f"a{i}",
                    item_type="anchor",
                    scale_ids=(),
                    deterministic_score=10.0 - i,
                    constraint_tags=("mandatory",),
                    reason_codes=("anchor_continuity",),
                    features={"multiplex_count": 1, "novelty_days": 100, "expected_burden": 1.0, "sensitivity": "low"},
                )
            )

        cs = CandidateSet(
            user_id="u1",
            day=date(2026, 2, 2),
            k_core=2,
            candidates=tuple(candidates),
            mandatory_item_ids=("a0", "a1", "a2"),
            deterministic_baseline_selected=("a0", "a1"),
        )
        ctx = PolicyContext(day_of_week=0)
        decision = make_policy_decision(
            candidate_set=cs,
            context=ctx,
            params=params,
            mapping=mapping,
            mode="live",
            runtime=PolicyRuntimeConfig(epsilon_explore=0.0, rng_seed=1, max_counterfactuals=3),
        )
        self.assertEqual(decision.mode, "safe_fallback")
        self.assertEqual(tuple(decision.selected_item_ids), ("a0", "a1"))

    def test_epsilon_exploration_logs_propensities(self) -> None:
        mapping = build_feature_mapping_v1()
        # Use any params; exploration doesn't need them.
        params = PolicyParams(policy_version="v1", feature_version=mapping.feature_version, lambda_reg=1.0, A=(), b=())

        c1 = CandidateItem(
            item_id="anchor_1",
            item_type="anchor",
            scale_ids=(),
            deterministic_score=10.0,
            constraint_tags=("mandatory",),
            reason_codes=("anchor_continuity",),
            features={"multiplex_count": 1, "novelty_days": 30, "expected_burden": 1.0, "sensitivity": "low"},
        )
        optionals = []
        for i in range(10):
            optionals.append(
                CandidateItem(
                    item_id=f"opt_{i}",
                    item_type="other",
                    scale_ids=(),
                    deterministic_score=float(i),
                    constraint_tags=("optional",),
                    reason_codes=(),
                    features={"multiplex_count": 1, "novelty_days": 30, "expected_burden": 1.0, "sensitivity": "low"},
                )
            )

        cs = CandidateSet(
            user_id="u1",
            day=date(2026, 2, 2),
            k_core=3,
            candidates=tuple([c1] + optionals),
            mandatory_item_ids=("anchor_1",),
            deterministic_baseline_selected=("anchor_1", "opt_9", "opt_8"),
        )
        ctx = PolicyContext(day_of_week=0)
        decision = make_policy_decision(
            candidate_set=cs,
            context=ctx,
            params=params,
            mapping=mapping,
            mode="live",
            runtime=PolicyRuntimeConfig(epsilon_explore=1.0, rng_seed=42, max_counterfactuals=5),
        )
        self.assertIsNotNone(decision.propensities)
        assert decision.propensities is not None
        self.assertIn("__slate__", decision.propensities)
        # Propensities should exist for chosen optional items (mandatory isn't sampled).
        chosen = set(decision.selected_item_ids)
        chosen.remove("anchor_1")
        self.assertTrue(all(i in decision.propensities for i in chosen))

    def test_fallback_sanitizes_baseline_and_keeps_mandatory(self) -> None:
        mapping = build_feature_mapping_v1()
        params = PolicyParams(policy_version="v1", feature_version=mapping.feature_version, lambda_reg=1.0, A=(), b=())

        mandatory = CandidateItem(
            item_id="anchor_1",
            item_type="anchor",
            scale_ids=(),
            deterministic_score=10.0,
            constraint_tags=("mandatory",),
            reason_codes=("anchor_continuity",),
            features={"multiplex_count": 1, "novelty_days": 10, "expected_burden": 1.0, "sensitivity": "low"},
        )
        optional = CandidateItem(
            item_id="opt_1",
            item_type="other",
            scale_ids=(),
            deterministic_score=1.0,
            constraint_tags=("optional",),
            reason_codes=(),
            features={"multiplex_count": 1, "novelty_days": 10, "expected_burden": 1.0, "sensitivity": "low"},
        )

        # Baseline intentionally points to missing/non-candidate IDs to force fallback sanitization.
        cs = CandidateSet(
            user_id="u1",
            day=date(2026, 2, 2),
            k_core=2,
            candidates=(mandatory, optional),
            mandatory_item_ids=("anchor_1",),
            deterministic_baseline_selected=("missing_a", "missing_b"),
        )
        ctx = PolicyContext(day_of_week=0)
        decision = make_policy_decision(
            candidate_set=cs,
            context=ctx,
            params=params,
            mapping=mapping,
            mode="live",
            runtime=PolicyRuntimeConfig(epsilon_explore=0.0, rng_seed=7, max_counterfactuals=3),
        )
        self.assertIn(decision.mode, ("live", "safe_fallback"))
        self.assertEqual(len(decision.selected_item_ids), 2)
        self.assertIn("anchor_1", decision.selected_item_ids)
        self.assertIn("opt_1", decision.selected_item_ids)

    def test_safety_protocol_skips_blocked_items(self) -> None:
        mapping = build_feature_mapping_v1()
        params = PolicyParams(policy_version="v1", feature_version=mapping.feature_version, lambda_reg=1.0, A=(), b=())
        blocked_anchor = CandidateItem(
            item_id="anchor_blocked",
            item_type="anchor",
            scale_ids=(),
            deterministic_score=10.0,
            constraint_tags=("mandatory", "blocked"),
            reason_codes=("anchor_continuity",),
            features={"multiplex_count": 1, "novelty_days": 1, "expected_burden": 1.0, "sensitivity": "low"},
        )
        optional_a = CandidateItem(
            item_id="opt_a",
            item_type="other",
            scale_ids=(),
            deterministic_score=8.0,
            constraint_tags=("optional",),
            reason_codes=(),
            features={"multiplex_count": 1, "novelty_days": 20, "expected_burden": 1.0, "sensitivity": "low"},
        )
        optional_b = CandidateItem(
            item_id="opt_b",
            item_type="other",
            scale_ids=(),
            deterministic_score=7.0,
            constraint_tags=("optional",),
            reason_codes=(),
            features={"multiplex_count": 1, "novelty_days": 30, "expected_burden": 1.0, "sensitivity": "low"},
        )
        cs = CandidateSet(
            user_id="u1",
            day=date(2026, 2, 2),
            k_core=2,
            candidates=(blocked_anchor, optional_a, optional_b),
            mandatory_item_ids=("anchor_blocked",),
            deterministic_baseline_selected=("anchor_blocked", "opt_a"),
        )
        ctx = PolicyContext(day_of_week=0, safety_trigger_active=True)
        decision = make_policy_decision(
            candidate_set=cs,
            context=ctx,
            params=params,
            mapping=mapping,
            mode="live",
            runtime=PolicyRuntimeConfig(epsilon_explore=0.0, rng_seed=1, max_counterfactuals=2),
        )
        self.assertEqual(decision.mode, "safe_fallback")
        self.assertEqual(len(decision.selected_item_ids), 2)
        self.assertNotIn("anchor_blocked", decision.selected_item_ids)


if __name__ == "__main__":
    unittest.main()
