"""
Tests for Concordance Study Framework.

Validates:
  1. ICC computation (perfect agreement, zero agreement, realistic)
  2. Bland-Altman analysis (no bias, systematic bias, limits of agreement)
  3. Cohen's kappa (perfect, chance, moderate agreement)
  4. Full concordance report generation
  5. Multi-instrument summary
  6. Edge cases
"""

import unittest
import random
from typing import List

from questions_agent_platform.pipeline.concordance import (
    PairedScore,
    compute_icc,
    compute_bland_altman,
    compute_kappa,
    generate_concordance_report,
    generate_multi_instrument_summary,
)


def _make_pairs(
    adaptive: List[float],
    fullform: List[float],
    instrument: str = "test_instrument",
    tier_func=None,
) -> List[PairedScore]:
    """Helper to create paired scores."""
    if tier_func is None:
        def tier_func(x: float) -> str:
            return "low" if x < 5 else ("moderate" if x < 10 else "high")

    return [
        PairedScore(
            participant_id=f"p{i}",
            instrument_id=instrument,
            timepoint=0,
            adaptive_score=a,
            fullform_score=f,
            adaptive_tier=tier_func(a),
            fullform_tier=tier_func(f),
        )
        for i, (a, f) in enumerate(zip(adaptive, fullform))
    ]


class TestICC(unittest.TestCase):
    """Test Intraclass Correlation Coefficient computation."""

    def test_perfect_agreement(self):
        """Identical scores should give ICC = 1.0."""
        scores = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
        pairs = _make_pairs(scores, scores)
        result = compute_icc(pairs)
        self.assertAlmostEqual(result.icc, 1.0, delta=0.01)
        self.assertTrue(result.passes_threshold)

    def test_high_agreement(self):
        """Scores with small noise should give high ICC."""
        random.seed(42)
        base = [float(i) for i in range(20)]
        adaptive = base
        fullform = [x + random.gauss(0, 0.3) for x in base]
        pairs = _make_pairs(adaptive, fullform)
        result = compute_icc(pairs)
        self.assertGreater(result.icc, 0.95)

    def test_no_agreement(self):
        """Random uncorrelated scores should give low ICC."""
        random.seed(42)
        adaptive = [random.uniform(0, 20) for _ in range(30)]
        fullform = [random.uniform(0, 20) for _ in range(30)]
        pairs = _make_pairs(adaptive, fullform)
        result = compute_icc(pairs)
        self.assertLess(result.icc, 0.3)
        self.assertFalse(result.passes_threshold)

    def test_insufficient_data(self):
        """Fewer than 3 pairs should return 0."""
        pairs = _make_pairs([1.0, 2.0], [1.1, 2.1])
        result = compute_icc(pairs)
        self.assertEqual(result.icc, 0.0)
        self.assertEqual(result.n_pairs, 2)

    def test_constant_scores(self):
        """All identical scores (no variance) should be handled."""
        pairs = _make_pairs([5.0] * 10, [5.0] * 10)
        result = compute_icc(pairs)
        # When all scores are identical, ICC should be very high or 1.0
        self.assertGreaterEqual(result.icc, 0.0)

    def test_ci_bounds(self):
        """Confidence interval should contain the point estimate."""
        base = list(range(20))
        adaptive = [float(x) for x in base]
        fullform = [float(x) + 0.5 for x in base]
        pairs = _make_pairs(adaptive, fullform)
        result = compute_icc(pairs)
        self.assertLessEqual(result.ci_lower, result.icc)
        self.assertGreaterEqual(result.ci_upper, result.icc)


class TestBlandAltman(unittest.TestCase):
    """Test Bland-Altman analysis."""

    def test_no_bias(self):
        """Identical scores should show zero bias."""
        scores = [float(i) for i in range(10)]
        pairs = _make_pairs(scores, scores)
        result = compute_bland_altman(pairs)
        self.assertAlmostEqual(result.mean_difference, 0.0, delta=0.01)
        self.assertAlmostEqual(result.sd_difference, 0.0, delta=0.01)

    def test_systematic_bias(self):
        """Constant offset should show as mean difference."""
        base = [float(i) for i in range(20)]
        adaptive = [x + 2.0 for x in base]  # systematic +2 bias
        pairs = _make_pairs(adaptive, base)
        result = compute_bland_altman(pairs)
        self.assertAlmostEqual(result.mean_difference, 2.0, delta=0.1)

    def test_limits_contain_most_points(self):
        """~95% of differences should fall within limits of agreement."""
        random.seed(42)
        base = [random.uniform(0, 20) for _ in range(100)]
        fullform = [x + random.gauss(0, 1.0) for x in base]
        pairs = _make_pairs(base, fullform)
        result = compute_bland_altman(pairs)
        self.assertLess(result.proportion_outside, 0.10)

    def test_clinical_tolerance(self):
        """Clinical tolerance check should work."""
        pairs = _make_pairs([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1])
        result = compute_bland_altman(pairs, clinical_tolerance=1.0)
        self.assertTrue(result.passes_clinical_range)

    def test_clinical_tolerance_fail(self):
        """Wide limits should fail clinical tolerance."""
        random.seed(42)
        base = [float(i) for i in range(20)]
        fullform = [x + random.gauss(0, 5.0) for x in base]
        pairs = _make_pairs(base, fullform)
        result = compute_bland_altman(pairs, clinical_tolerance=2.0)
        self.assertFalse(result.passes_clinical_range)


class TestKappa(unittest.TestCase):
    """Test Cohen's kappa."""

    def test_perfect_agreement(self):
        """Identical tiers should give kappa = 1.0."""
        scores = [2.0, 7.0, 12.0, 3.0, 8.0, 15.0, 1.0, 4.0, 18.0, 6.0]
        pairs = _make_pairs(scores, scores)
        result = compute_kappa(pairs)
        self.assertAlmostEqual(result.kappa, 1.0, delta=0.01)
        self.assertTrue(result.passes_threshold)

    def test_moderate_agreement(self):
        """Some disagreement should give kappa < 1."""
        # Most agree, a few disagree
        adaptive = [2.0, 7.0, 12.0, 3.0, 11.0, 15.0, 1.0, 4.0, 18.0, 6.0]
        fullform = [2.0, 7.0, 12.0, 3.0, 8.0, 15.0, 1.0, 4.0, 18.0, 6.0]
        # item 4: adaptive=11 (high), fullform=8 (moderate) — disagreement
        pairs = _make_pairs(adaptive, fullform)
        result = compute_kappa(pairs)
        self.assertGreater(result.kappa, 0.5)
        self.assertLess(result.kappa, 1.0)

    def test_categories_detected(self):
        """Should detect all unique categories."""
        scores = [2.0, 7.0, 12.0]
        pairs = _make_pairs(scores, scores)
        result = compute_kappa(pairs)
        self.assertIn("low", result.categories)
        self.assertIn("moderate", result.categories)
        self.assertIn("high", result.categories)

    def test_insufficient_data(self):
        """Single pair should return 0."""
        pairs = _make_pairs([5.0], [5.0])
        result = compute_kappa(pairs)
        self.assertEqual(result.kappa, 0.0)


class TestConcordanceReport(unittest.TestCase):
    """Test full concordance report generation."""

    def test_passing_report(self):
        """Perfect agreement should produce passing report."""
        random.seed(42)
        base = [float(i) for i in range(30)]
        adaptive = base
        fullform = [x + random.gauss(0, 0.2) for x in base]
        pairs = _make_pairs(adaptive, fullform, instrument="findrisc")
        report = generate_concordance_report(pairs, instrument_id="findrisc")
        self.assertEqual(report.instrument_id, "findrisc")
        self.assertEqual(report.n_participants, 30)
        self.assertGreater(report.icc.icc, 0.9)

    def test_failing_report_lists_reasons(self):
        """Poor agreement should list failure reasons."""
        random.seed(42)
        adaptive = [random.uniform(0, 20) for _ in range(30)]
        fullform = [random.uniform(0, 20) for _ in range(30)]
        pairs = _make_pairs(adaptive, fullform, instrument="bad_inst")
        report = generate_concordance_report(pairs, instrument_id="bad_inst")
        self.assertFalse(report.overall_pass)
        self.assertGreater(len(report.failure_reasons), 0)


class TestMultiInstrumentSummary(unittest.TestCase):
    """Test multi-instrument summary."""

    def test_summary_counts(self):
        """Summary should correctly count passing/failing instruments."""
        random.seed(42)
        # Good instrument
        base = [float(i) for i in range(30)]
        good_pairs = _make_pairs(base, [x + random.gauss(0, 0.2) for x in base],
                                 instrument="good")
        good_report = generate_concordance_report(good_pairs, instrument_id="good")

        # Bad instrument
        bad_pairs = _make_pairs(
            [random.uniform(0, 20) for _ in range(30)],
            [random.uniform(0, 20) for _ in range(30)],
            instrument="bad",
        )
        bad_report = generate_concordance_report(bad_pairs, instrument_id="bad")

        summary = generate_multi_instrument_summary([good_report, bad_report])
        self.assertEqual(summary["n_instruments"], 2)
        self.assertFalse(summary["all_pass"])
        self.assertIn("good", summary["instruments"])
        self.assertIn("bad", summary["instruments"])

    def test_all_pass_summary(self):
        """Summary should report all_pass=True when all pass."""
        random.seed(42)
        base = [float(i) for i in range(30)]
        reports = []
        for inst in ["findrisc", "ez_cvd", "scored"]:
            pairs = _make_pairs(base, [x + random.gauss(0, 0.2) for x in base],
                                instrument=inst)
            reports.append(generate_concordance_report(pairs, instrument_id=inst))

        summary = generate_multi_instrument_summary(reports)
        self.assertEqual(summary["n_pass"], 3)
        self.assertTrue(summary["all_pass"])
        self.assertGreater(summary["mean_icc"], 0.9)


if __name__ == "__main__":
    unittest.main()
