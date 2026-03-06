from __future__ import annotations

import unittest
from unittest.mock import patch

from questions_agent_platform.pipeline import rl_item_selection as rl


class TestRLOptionalRuntime(unittest.TestCase):
    def test_adaptive_selector_falls_back_to_irt_when_torch_missing(self) -> None:
        state = rl.CATState(n_items_answered=20)
        irt_scores = {0: 0.9, 1: 0.2, 2: 0.1}
        item_features = [[0.0] * rl.ItemFeatureEncoder.ITEM_FEATURE_DIM for _ in range(3)]
        available_mask = [1.0, 1.0, 0.0]

        with patch.object(rl, "torch", None):
            selector = rl.AdaptiveItemSelector(n_items=3, config=rl.DeepCATConfig())
            result = selector.select(
                state=state,
                irt_scores=irt_scores,
                item_features=item_features,
                available_mask=available_mask,
            )

        self.assertEqual(result.selection_source, "irt")
        self.assertEqual(result.item_index, 0)
        self.assertEqual(result.metadata["rl_status"], "unavailable")

    def test_deepcat_selector_raises_clear_error_when_torch_missing(self) -> None:
        with patch.object(rl, "torch", None):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                rl.DeepCATSelector(n_items=4)

        self.assertIn("torch is required for DeepCATSelector", str(ctx.exception))
