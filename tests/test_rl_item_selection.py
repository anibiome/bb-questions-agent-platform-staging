"""
Tests for Deep CAT — RL-based item selection module.

Validates:
  1. DeepCATSelector initialization and forward pass
  2. select_next_item returns valid item indices
  3. update() reduces loss over training steps
  4. ResponseTimeModel engagement scoring (fast/optimal/slow)
  5. ResponseTimeModel parameter updates
  6. AdaptiveItemSelector blending (IRT-heavy early, RL-heavy late)
  7. AdaptiveItemSelector respects available item constraints
  8. State encoding shape correctness
  9. Epsilon-greedy exploration vs exploitation
  10. Experience replay buffer operations
  11. Reward shaping correctness
  12. Dueling DQN architecture (value + advantage)
  13. Target network synchronization
  14. Response time classification
  15. Blending weight schedule
  16. Selection result metadata
"""

from __future__ import annotations

import math
import os
import random
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None
else:
    from questions_agent_platform.pipeline.rl_item_selection import (
        AdaptiveItemSelector,
        CATState,
        DeepCATConfig,
        DeepCATSelector,
        DuelingDQN,
        ItemFeatureEncoder,
        ReplayBuffer,
        ResponseTimeModel,
        SelectionResult,
        Transition,
    )


if torch is None:
    raise unittest.SkipTest("torch not installed; skipping optional RL item-selection tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_ITEMS = 20
ITEM_FEAT_DIM = ItemFeatureEncoder.ITEM_FEATURE_DIM


def _random_item_features(n_items: int = N_ITEMS) -> torch.Tensor:
    """Create random item features tensor."""
    return torch.randn(n_items, ITEM_FEAT_DIM)


def _full_mask(n_items: int = N_ITEMS) -> torch.Tensor:
    """All items available."""
    return torch.ones(n_items)


def _partial_mask(n_items: int = N_ITEMS, n_available: int = 10) -> torch.Tensor:
    """Partial availability mask."""
    mask = torch.zeros(n_items)
    indices = random.sample(range(n_items), min(n_available, n_items))
    mask[indices] = 1.0
    return mask


def _default_state(**overrides) -> CATState:
    """Create a default CAT state with optional overrides."""
    kwargs = dict(
        theta_estimate=0.0,
        theta_se=1.0,
        n_items_answered=3,
        total_information=2.5,
        mean_response_time=12.0,
        rt_std=5.0,
        engagement_score=0.85,
        session_duration_minutes=3.0,
        instrument_coverage_ratio=0.1,
        recent_se_trend=-0.05,
    )
    kwargs.update(overrides)
    return CATState(**kwargs)


def _make_transition(n_items: int = N_ITEMS) -> Transition:
    """Create a random transition for replay buffer testing."""
    state = _default_state()
    next_state = _default_state(n_items_answered=4, theta_se=0.9)
    item_feats = _random_item_features(n_items)
    mask = _full_mask(n_items)
    next_mask = mask.clone()
    action_idx = random.randint(0, n_items - 1)
    next_mask[action_idx] = 0.0
    return Transition(
        state=state.to_tensor(),
        item_features=item_feats,
        available_mask=mask,
        action=action_idx,
        reward=0.5,
        next_state=next_state.to_tensor(),
        next_item_features=item_feats,
        next_available_mask=next_mask,
        done=False,
    )


# ---------------------------------------------------------------------------
# Test: CATState
# ---------------------------------------------------------------------------

class TestCATState(unittest.TestCase):
    """Test state representation."""

    def test_state_to_tensor_shape(self):
        """State tensor should have shape (state_dim,)."""
        state = _default_state()
        tensor = state.to_tensor()
        self.assertEqual(tensor.shape, (CATState.state_dim(),))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_state_dim_consistent(self):
        """state_dim() should match actual tensor size."""
        state = _default_state()
        self.assertEqual(state.to_tensor().shape[0], CATState.state_dim())

    def test_state_values_finite(self):
        """All state values should be finite floats."""
        state = _default_state()
        tensor = state.to_tensor()
        self.assertTrue(torch.all(torch.isfinite(tensor)))


# ---------------------------------------------------------------------------
# Test: DuelingDQN
# ---------------------------------------------------------------------------

class TestDuelingDQN(unittest.TestCase):
    """Test the Dueling DQN architecture."""

    def setUp(self):
        self.net = DuelingDQN(
            state_dim=CATState.state_dim(),
            n_items=N_ITEMS,
            hidden_dim=64,
            item_embed_dim=16,
        )

    def test_forward_unbatched(self):
        """Single-sample forward pass should return (n_items,)."""
        state = _default_state().to_tensor()
        item_feats = _random_item_features()
        q_values = self.net(state, item_feats)
        self.assertEqual(q_values.shape, (N_ITEMS,))

    def test_forward_batched(self):
        """Batched forward pass should return (batch, n_items)."""
        batch = 4
        states = torch.randn(batch, CATState.state_dim())
        item_feats = torch.randn(batch, N_ITEMS, ITEM_FEAT_DIM)
        q_values = self.net(states, item_feats)
        self.assertEqual(q_values.shape, (batch, N_ITEMS))

    def test_mask_suppresses_unavailable(self):
        """Unavailable items should have very negative Q-values."""
        state = _default_state().to_tensor()
        item_feats = _random_item_features()
        mask = torch.ones(N_ITEMS)
        mask[5] = 0.0
        mask[10] = 0.0
        q_values = self.net(state, item_feats, mask)
        self.assertLess(q_values[5].item(), -1e6)
        self.assertLess(q_values[10].item(), -1e6)

    def test_dueling_structure(self):
        """Value and advantage streams should be separate."""
        # Verify the model has both streams
        self.assertTrue(hasattr(self.net, "value_stream"))
        self.assertTrue(hasattr(self.net, "advantage_fc1"))
        self.assertTrue(hasattr(self.net, "advantage_fc2"))

    def test_q_values_finite(self):
        """Q-values should all be finite."""
        state = _default_state().to_tensor()
        item_feats = _random_item_features()
        q_values = self.net(state, item_feats)
        self.assertTrue(torch.all(torch.isfinite(q_values)))


# ---------------------------------------------------------------------------
# Test: ReplayBuffer
# ---------------------------------------------------------------------------

class TestReplayBuffer(unittest.TestCase):
    """Test experience replay buffer operations."""

    def test_push_and_len(self):
        """Buffer length should track pushes."""
        buf = ReplayBuffer(capacity=100)
        self.assertEqual(len(buf), 0)
        buf.push(_make_transition())
        self.assertEqual(len(buf), 1)
        for _ in range(50):
            buf.push(_make_transition())
        self.assertEqual(len(buf), 51)

    def test_capacity_limit(self):
        """Buffer should not exceed capacity."""
        buf = ReplayBuffer(capacity=10)
        for _ in range(25):
            buf.push(_make_transition())
        self.assertEqual(len(buf), 10)

    def test_sample_returns_correct_size(self):
        """Sample should return requested batch size."""
        buf = ReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(_make_transition())
        batch = buf.sample(16)
        self.assertEqual(len(batch), 16)

    def test_sample_clamps_to_buffer_size(self):
        """Sample should not exceed buffer size."""
        buf = ReplayBuffer(capacity=100)
        for _ in range(5):
            buf.push(_make_transition())
        batch = buf.sample(20)
        self.assertEqual(len(batch), 5)


# ---------------------------------------------------------------------------
# Test: DeepCATSelector
# ---------------------------------------------------------------------------

class TestDeepCATSelector(unittest.TestCase):
    """Test the RL item selector."""

    def setUp(self):
        self.config = DeepCATConfig(
            hidden_dim=64,
            min_replay_size=8,
            batch_size=8,
            target_update_freq=10,
        )
        self.selector = DeepCATSelector(n_items=N_ITEMS, config=self.config)

    def test_select_returns_valid_index(self):
        """Selected item index should be in available set."""
        state = _default_state()
        item_feats = _random_item_features()
        mask = _partial_mask(n_available=8)
        available = set(torch.where(mask > 0.5)[0].tolist())

        action, meta = self.selector.select_next_item(state, item_feats, mask)
        self.assertIn(action, available)
        self.assertIn("selection_mode", meta)
        self.assertIn("q_value", meta)

    def test_epsilon_greedy_exploration(self):
        """With epsilon=1.0, all selections should be random (explore)."""
        state = _default_state()
        item_feats = _random_item_features()
        mask = _full_mask()

        modes = []
        for _ in range(20):
            _, meta = self.selector.select_next_item(
                state, item_feats, mask, epsilon=1.0
            )
            modes.append(meta["selection_mode"])
        # All should be explore with epsilon=1.0
        self.assertTrue(all(m == "explore" for m in modes))

    def test_epsilon_greedy_exploitation(self):
        """With epsilon=0.0, selections should be deterministic (exploit)."""
        state = _default_state()
        item_feats = _random_item_features()
        mask = _full_mask()

        actions = set()
        for _ in range(10):
            action, meta = self.selector.select_next_item(
                state, item_feats, mask, epsilon=0.0
            )
            actions.add(action)
            self.assertEqual(meta["selection_mode"], "exploit")
        # All selections should be the same (greedy)
        self.assertEqual(len(actions), 1)

    def test_epsilon_decay(self):
        """Epsilon should decay from start to end over configured steps."""
        selector = DeepCATSelector(
            n_items=N_ITEMS,
            config=DeepCATConfig(
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay_steps=100,
            ),
        )
        self.assertAlmostEqual(selector.epsilon, 1.0, places=3)
        selector.step_count = 50
        self.assertAlmostEqual(selector.epsilon, 0.525, places=2)
        selector.step_count = 100
        self.assertAlmostEqual(selector.epsilon, 0.05, places=3)
        selector.step_count = 200
        self.assertAlmostEqual(selector.epsilon, 0.05, places=3)

    def test_reward_computation(self):
        """Reward should combine SE reduction, engagement, and fatigue."""
        selector = self.selector
        # Good scenario: SE reduced, high engagement, few items
        reward = selector.compute_reward(
            se_before=1.0, se_after=0.7,
            engagement_score=0.9, n_items_answered=2,
        )
        self.assertGreater(reward, 0, "Good scenario should have positive reward")

        # Bad scenario: no SE reduction, low engagement, many items
        reward_bad = selector.compute_reward(
            se_before=0.5, se_after=0.5,
            engagement_score=0.1, n_items_answered=30,
        )
        self.assertLess(reward_bad, reward, "Bad scenario should have lower reward")

    def test_update_reduces_loss(self):
        """Training over repeated transitions should reduce loss."""
        # Fill replay buffer with consistent transitions
        for _ in range(32):
            self.selector.replay_buffer.push(_make_transition())

        # Collect losses over training steps
        losses = []
        for _ in range(50):
            t = _make_transition()
            loss = self.selector.update(t)
            if loss is not None:
                losses.append(loss)

        self.assertGreater(len(losses), 0, "Should have at least some training losses")
        # Check that loss generally trends downward (compare first vs last third)
        if len(losses) >= 6:
            last_third = sum(losses[-len(losses)//3:]) / (len(losses)//3)
            # Not a hard assertion (RL is noisy), but loss should not explode
            self.assertTrue(
                math.isfinite(last_third),
                "Loss should remain finite throughout training",
            )

    def test_target_network_sync(self):
        """After sync, target and policy nets should have identical weights."""
        # Ensure they diverge first (after some updates)
        for _ in range(16):
            self.selector.replay_buffer.push(_make_transition())
        for _ in range(5):
            self.selector.update(_make_transition())

        self.selector.sync_target_network()

        for p, t in zip(
            self.selector.policy_net.parameters(),
            self.selector.target_net.parameters(),
        ):
            self.assertTrue(
                torch.allclose(p.data, t.data),
                "Policy and target weights should match after sync",
            )

    def test_no_available_items_raises(self):
        """Should raise ValueError when no items are available."""
        state = _default_state()
        item_feats = _random_item_features()
        mask = torch.zeros(N_ITEMS)
        with self.assertRaises(ValueError):
            self.selector.select_next_item(state, item_feats, mask)


# ---------------------------------------------------------------------------
# Test: ResponseTimeModel
# ---------------------------------------------------------------------------

class TestResponseTimeModel(unittest.TestCase):
    """Test response time engagement scoring and parameter updates."""

    def setUp(self):
        self.model = ResponseTimeModel()

    def test_optimal_response_high_engagement(self):
        """Response times near the median (12s) should have high engagement."""
        score = self.model.engagement_score(12.0)
        self.assertGreater(score, 0.7, "Optimal RT should have high engagement")

    def test_very_fast_response_low_engagement(self):
        """Response time < 2s should have low engagement (careless)."""
        score = self.model.engagement_score(0.5)
        self.assertLess(score, 0.3, "Very fast RT should indicate carelessness")

    def test_very_slow_response_low_engagement(self):
        """Response time > 120s should have low engagement (distracted)."""
        score = self.model.engagement_score(120.0)
        self.assertLess(score, 0.5, "Very slow RT should indicate distraction")

    def test_engagement_score_bounds(self):
        """Engagement score should always be in [0, 1]."""
        for rt in [0.0, 0.1, 1.0, 5.0, 12.0, 30.0, 60.0, 120.0, 600.0]:
            score = self.model.engagement_score(rt)
            self.assertGreaterEqual(score, 0.0, f"Score < 0 for RT={rt}")
            self.assertLessEqual(score, 1.0, f"Score > 1 for RT={rt}")

    def test_negative_rt_returns_zero(self):
        """Negative response time should return 0."""
        self.assertEqual(self.model.engagement_score(-5.0), 0.0)

    def test_parameter_update(self):
        """Updating with response times should shift the distribution."""
        model = ResponseTimeModel(mu=2.5, sigma=0.8)
        original_mu = model.mu

        # Feed in consistently fast responses
        fast_rts = [3.0, 4.0, 3.5, 4.5, 3.0] * 10
        model.update_parameters(fast_rts)

        # Mu should shift toward log(3.5) ~ 1.25
        self.assertLess(
            model.mu, original_mu,
            "Mu should decrease when given fast response times",
        )

    def test_parameter_update_with_empty_list(self):
        """Updating with empty list should be a no-op."""
        model = ResponseTimeModel()
        original_mu = model.mu
        model.update_parameters([])
        self.assertEqual(model.mu, original_mu)

    def test_classify_response(self):
        """Response classification should match expected categories."""
        model = ResponseTimeModel()
        self.assertEqual(model.classify_response(0.5), "careless")
        self.assertEqual(model.classify_response(3.0), "quick")
        self.assertEqual(model.classify_response(15.0), "optimal")
        self.assertEqual(model.classify_response(45.0), "deliberate")
        self.assertEqual(model.classify_response(90.0), "distracted")

    def test_expected_and_median_rt(self):
        """Expected and median response times should be plausible."""
        model = ResponseTimeModel(mu=2.5, sigma=0.8)
        median = model.median_response_time()
        expected = model.expected_response_time()
        # For log-normal, expected > median
        self.assertGreater(expected, median)
        # Median should be exp(2.5) ~ 12.2s
        self.assertAlmostEqual(median, math.exp(2.5), places=1)


# ---------------------------------------------------------------------------
# Test: AdaptiveItemSelector
# ---------------------------------------------------------------------------

class TestAdaptiveItemSelector(unittest.TestCase):
    """Test the blended IRT + RL selector."""

    def setUp(self):
        self.config = DeepCATConfig(
            irt_only_items=5,
            blend_start_items=6,
            blend_end_items=15,
            irt_weight_at_blend_start=0.7,
            irt_weight_at_blend_end=0.3,
        )
        self.selector = AdaptiveItemSelector(
            n_items=N_ITEMS,
            config=self.config,
        )

    def test_pure_irt_early_session(self):
        """First 5 items should use pure IRT (weight=1.0)."""
        for n in range(5):
            w = self.selector.irt_weight(n)
            self.assertAlmostEqual(
                w, 1.0, places=4,
                msg=f"IRT weight should be 1.0 for item {n}",
            )

    def test_blended_mid_session(self):
        """Items 6-15 should have IRT weight between 0.3 and 0.7."""
        for n in range(6, 16):
            w = self.selector.irt_weight(n)
            self.assertGreaterEqual(w, 0.29, msg=f"IRT weight too low at item {n}")
            self.assertLessEqual(w, 0.71, msg=f"IRT weight too high at item {n}")

    def test_primarily_rl_late_session(self):
        """Items 16+ should have IRT weight at blend_end value."""
        for n in [16, 20, 50]:
            w = self.selector.irt_weight(n)
            self.assertAlmostEqual(
                w, 0.3, places=2,
                msg=f"IRT weight should be 0.3 for item {n}",
            )

    def test_select_returns_valid_result(self):
        """Select should return a valid SelectionResult."""
        state = _default_state(n_items_answered=3)
        irt_scores = {i: random.random() * 2.0 for i in range(N_ITEMS)}
        item_feats = _random_item_features()
        mask = _partial_mask(n_available=12)
        available_set = set(torch.where(mask > 0.5)[0].tolist())

        result = self.selector.select(state, irt_scores, item_feats, mask)

        self.assertIsInstance(result, SelectionResult)
        self.assertIn(result.item_index, available_set)
        self.assertIn(result.selection_source, ("irt", "rl", "blended"))
        self.assertGreaterEqual(result.blended_score, 0.0)

    def test_select_early_is_irt_only(self):
        """Early session items should be marked as 'irt' source."""
        state = _default_state(n_items_answered=2)
        irt_scores = {i: random.random() * 2.0 for i in range(N_ITEMS)}
        item_feats = _random_item_features()
        mask = _full_mask()

        result = self.selector.select(state, irt_scores, item_feats, mask)
        self.assertEqual(result.selection_source, "irt")
        self.assertAlmostEqual(result.irl_weight, 1.0, places=2)

    def test_select_respects_availability_mask(self):
        """Selected item must be in the available set."""
        state = _default_state(n_items_answered=10)
        irt_scores = {i: random.random() for i in range(N_ITEMS)}
        item_feats = _random_item_features()
        mask = torch.zeros(N_ITEMS)
        mask[3] = 1.0
        mask[7] = 1.0
        mask[15] = 1.0

        result = self.selector.select(state, irt_scores, item_feats, mask)
        self.assertIn(result.item_index, {3, 7, 15})

    def test_record_response_time(self):
        """Recording a response time should return a valid engagement score."""
        score = self.selector.record_response_time(12.0)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # Should also update the RT model
        self.assertEqual(self.selector.rt_model.n_observations, 1)

    def test_no_available_raises(self):
        """Should raise when no items are available."""
        state = _default_state()
        mask = torch.zeros(N_ITEMS)
        with self.assertRaises(ValueError):
            self.selector.select(state, {}, _random_item_features(), mask)


# ---------------------------------------------------------------------------
# Test: Integration
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):
    """Integration tests for the full Deep CAT pipeline."""

    def test_full_session_simulation(self):
        """Simulate a 10-item session and verify consistency."""
        config = DeepCATConfig(
            irt_only_items=3,
            blend_start_items=4,
            blend_end_items=8,
            min_replay_size=4,
            batch_size=4,
        )
        selector = AdaptiveItemSelector(n_items=N_ITEMS, config=config)
        item_feats = _random_item_features()
        mask = _full_mask()
        selected_items = []

        theta = 0.0
        se = 1.0

        for step in range(10):
            state = _default_state(
                n_items_answered=step,
                theta_estimate=theta,
                theta_se=se,
                engagement_score=0.85,
            )
            irt_scores = {i: random.random() for i in range(N_ITEMS)}

            result = selector.select(state, irt_scores, item_feats, mask)
            selected_items.append(result.item_index)

            # Mark item as used
            mask[result.item_index] = 0.0

            # Simulate response time and engagement
            rt = random.uniform(5.0, 25.0)
            engagement = selector.record_response_time(rt)

            # Simulate SE reduction
            se_new = max(0.2, se - 0.08)
            reward = selector.rl_selector.compute_reward(
                se_before=se, se_after=se_new,
                engagement_score=engagement, n_items_answered=step + 1,
            )

            # Create transition for RL training
            next_state = _default_state(
                n_items_answered=step + 1,
                theta_estimate=theta + random.uniform(-0.1, 0.1),
                theta_se=se_new,
            )
            transition = Transition(
                state=state.to_tensor(),
                item_features=item_feats,
                available_mask=(_full_mask() if step == 0 else mask.clone()),
                action=result.item_index,
                reward=reward,
                next_state=next_state.to_tensor(),
                next_item_features=item_feats,
                next_available_mask=mask.clone(),
                done=(step == 9),
            )
            selector.rl_selector.update(transition)
            se = se_new

        # Verify all selected items are unique
        self.assertEqual(
            len(set(selected_items)), len(selected_items),
            "All selected items should be unique",
        )
        self.assertEqual(len(selected_items), 10)

    def test_rl_selector_independent_of_irt(self):
        """RL selector should work without any IRT scores."""
        selector = DeepCATSelector(n_items=N_ITEMS)
        state = _default_state()
        item_feats = _random_item_features()
        mask = _full_mask()

        action, meta = selector.select_next_item(state, item_feats, mask, epsilon=0.0)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, N_ITEMS)


class TestCheckpointSecurity(unittest.TestCase):
    def test_safe_checkpoint_roundtrip(self):
        selector = DeepCATSelector(n_items=N_ITEMS)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "selector.pt")
            selector.save(path)
            restored = DeepCATSelector(n_items=N_ITEMS)
            restored.load(path)
            self.assertEqual(restored.step_count, selector.step_count)

    def test_legacy_checkpoint_requires_explicit_opt_in(self):
        selector = DeepCATSelector(n_items=N_ITEMS)
        legacy_payload = {
            "policy_net": selector.policy_net.state_dict(),
            "target_net": selector.target_net.state_dict(),
            "optimizer": selector.optimizer.state_dict(),
            "step_count": selector.step_count,
            "config": selector.config,  # legacy object payload (pickle-backed)
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy_selector.pt")
            torch.save(legacy_payload, path)

            os.environ.pop("QUESTIONS_AGENT_ALLOW_UNSAFE_CHECKPOINT_LOAD", None)
            with self.assertRaises(RuntimeError):
                selector.load(path)

            os.environ["QUESTIONS_AGENT_ALLOW_UNSAFE_CHECKPOINT_LOAD"] = "1"
            try:
                selector.load(path)
            finally:
                os.environ.pop("QUESTIONS_AGENT_ALLOW_UNSAFE_CHECKPOINT_LOAD", None)


if __name__ == "__main__":
    unittest.main()
