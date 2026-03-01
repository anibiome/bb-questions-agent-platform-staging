"""
Tests for IRT (Graded Response Model) scoring layer.

Validates:
  1. GRM probability functions sum to 1 and are monotone
  2. Fisher information is positive and peaks near thresholds
  3. EAP estimation recovers known theta values
  4. MLE estimation converges for non-degenerate patterns
  5. SE(theta) decreases with more items (precision gain)
  6. Default parameter generation produces valid GRM params
  7. Full scale scoring integrates correctly with classical scoring
  8. Edge cases: extreme responses, single item, all same response
"""

import math
import unittest
from typing import Dict, List

from questions_agent_platform.pipeline.irt import (
    GRMItemParams,
    IRTScoreResult,
    categories_for_response_type,
    default_item_params,
    default_scale_params,
    estimate_theta_eap,
    estimate_theta_mle,
    grm_category_probs,
    grm_cumulative_prob,
    grm_item_information,
    grm_log_likelihood,
    marginal_reliability,
    score_scale_irt,
    se_at_theta,
    total_information,
)


def _make_5cat_item(item_id: str = "test_item", a: float = 1.0) -> GRMItemParams:
    """Standard 5-category item with evenly spaced thresholds."""
    return GRMItemParams(
        item_id=item_id,
        discrimination=a,
        thresholds=(-1.5, -0.5, 0.5, 1.5),
    )


def _make_scale_items(n: int = 5, a: float = 1.0) -> List[GRMItemParams]:
    """Create n 5-category items with given discrimination."""
    return [_make_5cat_item(f"item_{i}", a) for i in range(n)]


class TestGRMProbabilities(unittest.TestCase):
    """Test GRM probability computations."""

    def test_cumulative_prob_bounds(self):
        """P*(θ) should be in [0, 1]."""
        item = _make_5cat_item()
        for theta in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            for b in item.thresholds:
                p = grm_cumulative_prob(theta, item.discrimination, b)
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_cumulative_prob_monotone(self):
        """P*(θ) should increase with θ for fixed b."""
        item = _make_5cat_item()
        b = item.thresholds[1]
        prev_p = 0.0
        for theta in [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]:
            p = grm_cumulative_prob(theta, item.discrimination, b)
            self.assertGreaterEqual(p, prev_p)
            prev_p = p

    def test_category_probs_sum_to_one(self):
        """Category probabilities must sum to 1.0."""
        item = _make_5cat_item()
        for theta in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            probs = grm_category_probs(theta, item)
            self.assertAlmostEqual(sum(probs), 1.0, places=10)

    def test_category_probs_positive(self):
        """All category probabilities must be positive."""
        item = _make_5cat_item()
        for theta in [-5.0, -2.0, 0.0, 2.0, 5.0]:
            probs = grm_category_probs(theta, item)
            for p in probs:
                self.assertGreater(p, 0.0)

    def test_category_probs_correct_count(self):
        """5-category item should produce 5 probabilities."""
        item = _make_5cat_item()
        probs = grm_category_probs(0.0, item)
        self.assertEqual(len(probs), 5)

    def test_extreme_theta_concentrates_probability(self):
        """Very high θ should concentrate probability in highest category."""
        item = _make_5cat_item(a=2.0)
        probs = grm_category_probs(5.0, item)
        self.assertGreater(probs[-1], 0.95, "High θ should favor top category")

        probs_low = grm_category_probs(-5.0, item)
        self.assertGreater(probs_low[0], 0.95, "Low θ should favor bottom category")

    def test_binary_item_reduces_to_2pl(self):
        """Binary GRM should behave like 2PL."""
        item = GRMItemParams(item_id="binary", discrimination=1.5, thresholds=(0.0,))
        probs = grm_category_probs(0.0, item)
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(probs[0], 0.5, places=5)
        self.assertAlmostEqual(probs[1], 0.5, places=5)


class TestFisherInformation(unittest.TestCase):
    """Test Fisher information computation."""

    def test_information_positive(self):
        """Item information should always be positive."""
        item = _make_5cat_item()
        for theta in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            info = grm_item_information(theta, item)
            self.assertGreater(info, 0.0)

    def test_information_peaks_near_thresholds(self):
        """Information should be higher near thresholds than at extremes."""
        item = _make_5cat_item(a=1.5)
        info_center = grm_item_information(0.0, item)
        info_extreme = grm_item_information(4.0, item)
        self.assertGreater(info_center, info_extreme)

    def test_higher_discrimination_more_information(self):
        """Higher a → more information."""
        item_low = _make_5cat_item(a=0.5)
        item_high = _make_5cat_item(a=2.0)
        info_low = grm_item_information(0.0, item_low)
        info_high = grm_item_information(0.0, item_high)
        self.assertGreater(info_high, info_low)

    def test_total_information_additive(self):
        """Total information = sum of item informations."""
        items = _make_scale_items(3)
        theta = 0.5
        expected = sum(grm_item_information(theta, it) for it in items)
        actual = total_information(theta, items)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_se_decreases_with_more_items(self):
        """SE(θ) should decrease as we add more items."""
        items_3 = _make_scale_items(3)
        items_10 = _make_scale_items(10)
        se_3 = se_at_theta(0.0, items_3)
        se_10 = se_at_theta(0.0, items_10)
        self.assertLess(se_10, se_3)


class TestThetaEstimation(unittest.TestCase):
    """Test EAP and MLE theta estimation."""

    def test_eap_mid_response_near_zero(self):
        """Mid-range responses (category 2 of 0-4) should give θ ≈ 0."""
        items = _make_scale_items(5)
        responses = [2, 2, 2, 2, 2]
        theta, post_sd = estimate_theta_eap(items, responses)
        self.assertAlmostEqual(theta, 0.0, delta=0.3)

    def test_eap_high_response_positive_theta(self):
        """All-high responses should give θ > 0."""
        items = _make_scale_items(5)
        responses = [4, 4, 4, 4, 4]
        theta, _ = estimate_theta_eap(items, responses)
        self.assertGreater(theta, 1.0)

    def test_eap_low_response_negative_theta(self):
        """All-low responses should give θ < 0."""
        items = _make_scale_items(5)
        responses = [0, 0, 0, 0, 0]
        theta, _ = estimate_theta_eap(items, responses)
        self.assertLess(theta, -1.0)

    def test_eap_posterior_sd_shrinks_with_items(self):
        """Posterior SD should shrink with more items."""
        items_3 = _make_scale_items(3)
        items_10 = _make_scale_items(10)
        _, sd_3 = estimate_theta_eap(items_3, [2] * 3)
        _, sd_10 = estimate_theta_eap(items_10, [2] * 10)
        self.assertLess(sd_10, sd_3)

    def test_mle_converges_for_mixed_pattern(self):
        """MLE should converge for a non-degenerate response pattern."""
        items = _make_scale_items(5)
        responses = [1, 2, 3, 2, 1]
        theta, converged = estimate_theta_mle(items, responses)
        self.assertTrue(converged)
        self.assertGreater(theta, -3.0)
        self.assertLess(theta, 3.0)

    def test_mle_eap_agree_approximately(self):
        """MLE and EAP should give similar estimates for reasonable patterns."""
        items = _make_scale_items(8)
        responses = [1, 2, 3, 2, 1, 3, 2, 2]
        theta_eap, _ = estimate_theta_eap(items, responses)
        theta_mle, _ = estimate_theta_mle(items, responses)
        self.assertAlmostEqual(theta_eap, theta_mle, delta=0.5)

    def test_eap_with_informative_prior(self):
        """Strong prior should pull estimate toward prior mean."""
        items = _make_scale_items(2)  # few items
        responses = [4, 4]  # high responses
        theta_flat, _ = estimate_theta_eap(items, responses, prior_sd=3.0)
        theta_tight, _ = estimate_theta_eap(items, responses, prior_mean=0.0, prior_sd=0.3)
        # Tight prior should pull theta closer to 0
        self.assertLess(abs(theta_tight), abs(theta_flat))


class TestLogLikelihood(unittest.TestCase):
    """Test log-likelihood computation."""

    def test_log_likelihood_negative(self):
        """Log-likelihood should always be negative."""
        items = _make_scale_items(3)
        responses = [2, 2, 2]
        for theta in [-2.0, 0.0, 2.0]:
            ll = grm_log_likelihood(theta, items, responses)
            self.assertLess(ll, 0.0)

    def test_log_likelihood_maximized_near_true_theta(self):
        """LL should be highest near the generating θ for consistent responses."""
        items = _make_scale_items(10, a=2.0)
        # All category 4 → high θ
        responses = [4] * 10
        ll_high = grm_log_likelihood(2.5, items, responses)
        ll_mid = grm_log_likelihood(0.0, items, responses)
        ll_low = grm_log_likelihood(-2.5, items, responses)
        self.assertGreater(ll_high, ll_mid)
        self.assertGreater(ll_mid, ll_low)


class TestScoreScaleIRT(unittest.TestCase):
    """Test the full IRT scoring interface."""

    def test_basic_scoring(self):
        """Full scoring should return valid IRTScoreResult."""
        items = _make_scale_items(5)
        responses = {"item_0": 2, "item_1": 2, "item_2": 3, "item_3": 1, "item_4": 2}
        result = score_scale_irt(items, responses)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, IRTScoreResult)
        self.assertEqual(result.n_items_used, 5)
        self.assertEqual(result.estimation_method, "eap")
        self.assertGreater(result.information, 0)
        self.assertGreater(result.se_theta, 0)
        self.assertLessEqual(result.reliability, 1.0)
        self.assertGreaterEqual(result.reliability, 0.0)
        # CI should bracket theta
        self.assertLess(result.theta_ci_lower, result.theta)
        self.assertGreater(result.theta_ci_upper, result.theta)

    def test_returns_none_for_too_few_items(self):
        """Should return None if fewer than 2 items answered."""
        items = _make_scale_items(5)
        responses = {"item_0": 2}
        result = score_scale_irt(items, responses)
        self.assertIsNone(result)

    def test_partial_responses_ok(self):
        """Should work with partial response patterns (some items missing)."""
        items = _make_scale_items(5)
        responses = {"item_0": 2, "item_2": 3}  # only 2 of 5
        result = score_scale_irt(items, responses)
        self.assertIsNotNone(result)
        self.assertEqual(result.n_items_used, 2)

    def test_mle_method(self):
        """MLE method should work for reasonable patterns."""
        items = _make_scale_items(5)
        responses = {"item_0": 1, "item_1": 2, "item_2": 3, "item_3": 2, "item_4": 1}
        result = score_scale_irt(items, responses, method="mle")
        self.assertIsNotNone(result)
        # Method might fall back to EAP if MLE doesn't converge
        self.assertIn(result.estimation_method, ("mle", "eap"))


class TestMarginalReliability(unittest.TestCase):
    """Test reliability computation."""

    def test_perfect_measurement(self):
        """SE near 0 → reliability near 1."""
        r = marginal_reliability(0.01)
        self.assertGreater(r, 0.99)

    def test_uninformative_measurement(self):
        """SE = prior_sd → reliability = 0."""
        r = marginal_reliability(1.0, prior_var=1.0)
        self.assertAlmostEqual(r, 0.0, places=5)

    def test_moderate_se(self):
        """SE = 0.5 with prior_var = 1.0 → reliability = 0.75."""
        r = marginal_reliability(0.5, prior_var=1.0)
        self.assertAlmostEqual(r, 0.75, places=5)


class TestDefaultParameters(unittest.TestCase):
    """Test default GRM parameter generation."""

    def test_default_5cat_item(self):
        """5-category item should get 4 thresholds."""
        params = default_item_params("test", n_categories=5)
        self.assertEqual(len(params.thresholds), 4)
        self.assertEqual(params.item_id, "test")
        self.assertGreater(params.discrimination, 0)

    def test_thresholds_ascending(self):
        """Thresholds must be in ascending order."""
        for n_cat in [2, 3, 4, 5, 6, 7, 8]:
            params = default_item_params("test", n_categories=n_cat)
            for i in range(len(params.thresholds) - 1):
                self.assertLess(
                    params.thresholds[i], params.thresholds[i + 1],
                    f"Thresholds not ascending for {n_cat}-category item",
                )

    def test_weight_affects_discrimination(self):
        """Higher weight → higher discrimination."""
        low = default_item_params("test", n_categories=5, weight=0.5)
        high = default_item_params("test", n_categories=5, weight=2.0)
        self.assertGreater(high.discrimination, low.discrimination)

    def test_default_scale_params(self):
        """Should generate params for all items in a scale."""
        scale_items = [("item_a", 1.0), ("item_b", 0.7), ("item_c", 1.5)]
        params = default_scale_params(scale_items, n_categories=5)
        self.assertEqual(len(params), 3)
        self.assertEqual(params[0].item_id, "item_a")
        self.assertEqual(params[1].item_id, "item_b")
        self.assertEqual(params[2].item_id, "item_c")

    def test_categories_for_response_type(self):
        """Known response types should return correct category counts."""
        self.assertEqual(categories_for_response_type("likert_0_4"), 5)
        self.assertEqual(categories_for_response_type("likert_0_3"), 4)
        self.assertEqual(categories_for_response_type("bool_0_1"), 2)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and numerical stability."""

    def test_all_same_response(self):
        """All same category should not crash."""
        items = _make_scale_items(5, a=1.5)
        responses = [0, 0, 0, 0, 0]
        theta, sd = estimate_theta_eap(items, responses)
        self.assertTrue(math.isfinite(theta))
        self.assertTrue(math.isfinite(sd))

    def test_extreme_discrimination(self):
        """Very high discrimination should not cause overflow."""
        item = GRMItemParams(
            item_id="extreme",
            discrimination=5.0,
            thresholds=(-1.0, 0.0, 1.0),
        )
        probs = grm_category_probs(0.0, item)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)
        info = grm_item_information(0.0, item)
        self.assertTrue(math.isfinite(info))

    def test_two_items_minimum(self):
        """Score should work with exactly 2 items."""
        items = _make_scale_items(2)
        responses = {"item_0": 1, "item_1": 3}
        result = score_scale_irt(items, responses)
        self.assertIsNotNone(result)
        self.assertEqual(result.n_items_used, 2)

    def test_information_gain_from_adding_item(self):
        """Adding an item should always increase total information."""
        items_4 = _make_scale_items(4)
        items_5 = _make_scale_items(5)
        for theta in [-2.0, 0.0, 2.0]:
            info_4 = total_information(theta, items_4)
            info_5 = total_information(theta, items_5)
            self.assertGreater(info_5, info_4)


class TestIRTClassicalIntegration(unittest.TestCase):
    """Test that IRT scoring integrates with classical ScaleScoreResult."""

    def test_generic_scale_has_irt_fields(self):
        """compute_scale_score for generic scales should populate IRT fields."""
        from questions_agent_platform.pipeline.scoring import compute_scale_score
        from questions_agent_platform.pipeline.registry import Scale, ScaleItem

        scale = Scale(
            id="test_scale",
            questionnaire_id="q_test",
            version="1",
            name="Test Scale",
            method="mean",
            min_items_required=3,
            unlock_window_days=14,
            retest_interval_days=90,
            response_type="likert_0_4",
            normalize_min=0.0,
            normalize_max=4.0,
            items=tuple(
                ScaleItem(item_id=f"item_{i}", reverse=False, weight=1.0)
                for i in range(5)
            ),
            tags=("test",),
        )

        answers = {f"item_{i}": 2.0 for i in range(5)}
        result = compute_scale_score(scale, answers)
        self.assertIsNotNone(result)
        # IRT fields should be populated
        self.assertIsNotNone(result.theta, "theta should be set")
        self.assertIsNotNone(result.se_theta, "se_theta should be set")
        self.assertIsNotNone(result.irt_reliability, "irt_reliability should be set")
        self.assertIsNotNone(result.information, "information should be set")
        # Classical fields still work
        self.assertIsNotNone(result.raw_score)
        self.assertIsNotNone(result.normalized_score)
        self.assertEqual(result.answered_count, 5)

    def test_irt_theta_correlates_with_raw_score(self):
        """Higher raw scores should correspond to higher theta."""
        from questions_agent_platform.pipeline.scoring import compute_scale_score
        from questions_agent_platform.pipeline.registry import Scale, ScaleItem

        scale = Scale(
            id="test_scale",
            questionnaire_id="q_test",
            version="1",
            name="Test",
            method="mean",
            min_items_required=3,
            unlock_window_days=14,
            retest_interval_days=90,
            response_type="likert_0_4",
            normalize_min=0.0,
            normalize_max=4.0,
            items=tuple(
                ScaleItem(item_id=f"item_{i}", reverse=False, weight=1.0)
                for i in range(5)
            ),
        )

        low_answers = {f"item_{i}": 0.0 for i in range(5)}
        high_answers = {f"item_{i}": 4.0 for i in range(5)}

        low_result = compute_scale_score(scale, low_answers)
        high_result = compute_scale_score(scale, high_answers)

        self.assertIsNotNone(low_result.theta)
        self.assertIsNotNone(high_result.theta)
        self.assertGreater(
            high_result.theta, low_result.theta,
            "Higher raw score should give higher theta",
        )


if __name__ == "__main__":
    unittest.main()
