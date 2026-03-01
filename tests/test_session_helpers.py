import unittest

from questions_agent_platform.pipeline.registry import Item, Registry
from questions_agent_platform.pipeline.session_helpers import (
    SELECTION_MODE_POLICY_LIVE,
    SELECTION_MODE_POLICY_OFFLINE_REPLAY,
    SELECTION_MODE_POLICY_SHADOW,
    allowed_item_ids_for_domains,
    build_selection_explain_payload,
    evaluate_onboarding_cardiometabolic_gate,
    filter_extra_batches,
    merge_allowed_item_ids,
    normalize_selection_mode,
    policy_runtime_mode,
)


class TestSessionHelpers(unittest.TestCase):
    def _registry(self) -> Registry:
        return Registry(
            version="v1",
            items={
                "cardio_q1": Item(id="cardio_q1", text="q", response_type="likert5", tags=("cardiometabolic",)),
                "cardio_q2": Item(id="cardio_q2", text="q", response_type="likert5", tags=("cardiometabolic",)),
                "sleep_q1": Item(id="sleep_q1", text="q", response_type="likert5", tags=("sleep",)),
            },
            scales={},
            questionnaires={},
        )

    def test_normalize_selection_mode_accepts_known_values(self) -> None:
        self.assertEqual(normalize_selection_mode("Policy_Live"), SELECTION_MODE_POLICY_LIVE)
        self.assertEqual(normalize_selection_mode("policy_shadow"), SELECTION_MODE_POLICY_SHADOW)
        self.assertEqual(normalize_selection_mode("policy_offline_replay"), SELECTION_MODE_POLICY_OFFLINE_REPLAY)

    def test_normalize_selection_mode_rejects_unknown_value(self) -> None:
        with self.assertRaises(ValueError):
            normalize_selection_mode("policy_magic")

    def test_policy_runtime_mode_mapping(self) -> None:
        self.assertEqual(policy_runtime_mode(SELECTION_MODE_POLICY_LIVE), "live")
        self.assertEqual(policy_runtime_mode(SELECTION_MODE_POLICY_SHADOW), "shadow")
        self.assertEqual(policy_runtime_mode(SELECTION_MODE_POLICY_OFFLINE_REPLAY), "replay")

    def test_filter_extra_batches_removes_core_declined_and_disallowed(self) -> None:
        filtered = filter_extra_batches(
            extra_batches=(("core_1", "allowed_a", "declined_x"), ("allowed_b", "outside_domain")),
            core_item_ids=("core_1",),
            declined_item_ids={"declined_x"},
            allowed_item_ids={"allowed_a", "allowed_b"},
        )
        self.assertEqual(filtered, (("allowed_a",), ("allowed_b",)))

    def test_build_selection_explain_payload_shape(self) -> None:
        payload = build_selection_explain_payload(
            reasons_by_item_id={"q1": ("near_unlock",)},
            primary_scale_by_item_id={"q1": "scale_1"},
            near_unlock_scales=("scale_1",),
            due_retest_scales=(),
            drift_scales=(),
            follow_up_item_ids={"q1"},
            profile={"mode": "consumer", "active_domains": ["cardiometabolic"], "queued_domains": []},
            allowed_domains={"cardiometabolic"},
            requested_selection_mode="policy_shadow",
            selection_mode="policy_shadow",
            policy_summary={"decision_id": "d1"},
            deterministic_baseline_core_item_ids=("q1",),
            served_core_item_ids=("q1",),
            rollback_guard=None,
            safety_mode_active=False,
            open_safety_events=[],
            timeframe="last_7_days",
        )
        self.assertEqual(payload["selection_mode"], "policy_shadow")
        self.assertEqual(payload["timeframe"], "last_7_days")
        self.assertEqual(payload["served_core_item_ids"], ["q1"])
        self.assertEqual(payload["allowed_domains"], ["cardiometabolic"])

    def test_evaluate_onboarding_gate_prioritizes_pending_cardiometabolic(self) -> None:
        gate = evaluate_onboarding_cardiometabolic_gate(
            onboarding_complete=False,
            registry=self._registry(),
            declined_item_ids={"cardio_q2"},
            answered_item_ids=set(),
        )
        self.assertFalse(gate.mark_onboarding_complete)
        self.assertEqual(gate.allowed_item_ids, {"cardio_q1"})
        self.assertEqual(gate.priority_item_ids, {"cardio_q1"})

    def test_evaluate_onboarding_gate_marks_complete_when_pending_empty(self) -> None:
        gate = evaluate_onboarding_cardiometabolic_gate(
            onboarding_complete=False,
            registry=self._registry(),
            declined_item_ids={"cardio_q2"},
            answered_item_ids={"cardio_q1"},
        )
        self.assertTrue(gate.mark_onboarding_complete)
        self.assertIsNone(gate.allowed_item_ids)
        self.assertIsNone(gate.priority_item_ids)

    def test_evaluate_onboarding_gate_marks_complete_without_cardio_items(self) -> None:
        registry = Registry(
            version="v1",
            items={"sleep_q1": Item(id="sleep_q1", text="q", response_type="likert5", tags=("sleep",))},
            scales={},
            questionnaires={},
        )
        gate = evaluate_onboarding_cardiometabolic_gate(
            onboarding_complete=False,
            registry=registry,
            declined_item_ids=set(),
            answered_item_ids=set(),
        )
        self.assertTrue(gate.mark_onboarding_complete)

    def test_allowed_item_ids_for_domains_filters_domain_scope(self) -> None:
        allowed = allowed_item_ids_for_domains(
            registry=self._registry(),
            allowed_domains={"sleep"},
        )
        self.assertEqual(allowed, {"sleep_q1"})
        all_items = allowed_item_ids_for_domains(
            registry=self._registry(),
            allowed_domains=set(),
        )
        self.assertEqual(all_items, {"cardio_q1", "cardio_q2", "sleep_q1"})

    def test_merge_allowed_item_ids(self) -> None:
        merged = merge_allowed_item_ids(
            current_allowed_item_ids={"a", "b"},
            domain_allowed_item_ids={"b", "c"},
        )
        self.assertEqual(merged, {"b"})
        merged_from_none = merge_allowed_item_ids(
            current_allowed_item_ids=None,
            domain_allowed_item_ids={"x"},
        )
        self.assertEqual(merged_from_none, {"x"})


if __name__ == "__main__":
    unittest.main()
