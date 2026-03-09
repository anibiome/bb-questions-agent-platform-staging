"""Tests for the N-of-1 Adaptive Calibration Engine.

Validates the Kalman-filter-based personal theta tracking, Reliable Change
Index, within-person variability estimation, measurement sufficiency, and
calibration phase management.

Reference values are derived from:
  - Jacobson & Truax, 1991 (RCI formula)
  - Standard Kalman filter equations
  - IRT measurement theory (SE, reliability)
"""

import math
import unittest

from questions_agent_platform.pipeline.n_of_1 import (
    PersonalCalibration,
    initial_calibration,
    update_calibration,
    assess_reliable_change,
    assess_measurement_sufficiency,
    estimate_within_person_sd,
    calibration_to_dict,
    calibration_from_dict,
    _determine_phase,
    _normal_sf,
    _z_critical,
    N_OF_1_VERSION,
    WARMING_MIN_OBS,
    CALIBRATED_MIN_OBS,
    MAX_TRAJECTORY_LENGTH,
    MIN_OBS_FOR_WITHIN_SD,
)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialCalibration(unittest.TestCase):

    def test_defaults(self):
        """Initial state matches population prior."""
        cal = initial_calibration("phq9")
        self.assertEqual(cal.scale_id, "phq9")
        self.assertEqual(cal.n_observations, 0)
        self.assertAlmostEqual(cal.theta_personal, 0.0, places=6)
        self.assertAlmostEqual(cal.se_personal, 1.0, places=6)
        self.assertEqual(cal.phase, "warming")
        self.assertAlmostEqual(cal.shrinkage, 0.0, places=6)
        self.assertIsNone(cal.theta_baseline)
        self.assertIsNone(cal.within_person_sd)
        self.assertEqual(len(cal.trajectory_thetas), 0)

    def test_custom_population_prior(self):
        """Custom population mu and sigma are stored."""
        cal = initial_calibration("gad7", population_mu=0.5, population_sigma=1.5)
        self.assertAlmostEqual(cal.theta_personal, 0.5, places=6)
        self.assertAlmostEqual(cal.se_personal, 1.5, places=6)
        self.assertAlmostEqual(cal.theta_population_mu, 0.5, places=6)
        self.assertAlmostEqual(cal.theta_population_sigma, 1.5, places=6)

    def test_version(self):
        cal = initial_calibration("test")
        self.assertEqual(cal.version, N_OF_1_VERSION)


# ---------------------------------------------------------------------------
# Kalman filter update
# ---------------------------------------------------------------------------

class TestKalmanUpdate(unittest.TestCase):
    """Core Kalman filter: predict + update step."""

    def test_first_observation_shifts_toward_data(self):
        """First observation should pull posterior toward observed theta."""
        cal = initial_calibration("test")
        updated = update_calibration(cal, theta_observed=1.0, se_observed=0.5)

        self.assertEqual(updated.n_observations, 1)
        # Posterior should be between prior (0.0) and observation (1.0)
        self.assertGreater(updated.theta_personal, 0.0)
        self.assertLess(updated.theta_personal, 1.0)
        # SE should decrease (gained information)
        self.assertLess(updated.se_personal, cal.se_personal)

    def test_precise_observation_dominates(self):
        """Very precise observation (small SE) → posterior near observation."""
        cal = initial_calibration("test")
        updated = update_calibration(cal, theta_observed=2.0, se_observed=0.01)

        # With SE=0.01, the observation is extremely precise
        self.assertAlmostEqual(updated.theta_personal, 2.0, places=1)

    def test_imprecise_observation_moves_little(self):
        """Very imprecise observation (large SE) → posterior stays near prior."""
        cal = initial_calibration("test")
        updated = update_calibration(cal, theta_observed=2.0, se_observed=10.0)

        # With SE=10, the observation is very noisy
        self.assertAlmostEqual(updated.theta_personal, 0.0, delta=0.5)

    def test_se_decreases_with_observations(self):
        """SE should generally decrease as we accumulate observations."""
        cal = initial_calibration("test")
        ses = [cal.se_personal]
        for i in range(20):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
            ses.append(cal.se_personal)

        # SE should be monotonically non-increasing (mostly decreasing)
        # Process noise adds some back, but with consistent observations
        # the trend is downward
        self.assertLess(ses[-1], ses[0])

    def test_convergence_with_consistent_data(self):
        """Repeated identical observations → theta converges to observed value."""
        cal = initial_calibration("test")
        for _ in range(50):
            cal = update_calibration(cal, theta_observed=1.5, se_observed=0.3)

        self.assertAlmostEqual(cal.theta_personal, 1.5, delta=0.15)

    def test_process_noise_prevents_overcertainty(self):
        """Process noise prevents SE from collapsing to zero."""
        cal = initial_calibration("test", process_noise_sd=0.1)
        for _ in range(100):
            cal = update_calibration(cal, theta_observed=0.0, se_observed=0.3)

        # SE should floor above 0 due to process noise
        self.assertGreater(cal.se_personal, 0.01)

    def test_time_gap_increases_uncertainty(self):
        """Large time gap → more process noise → larger predicted variance."""
        cal = initial_calibration("test")
        cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)

        # Short gap
        updated_short = update_calibration(cal, theta_observed=0.5, se_observed=0.3, days_since_last=1.0)
        # Long gap
        updated_long = update_calibration(cal, theta_observed=0.5, se_observed=0.3, days_since_last=30.0)

        # Longer gap → more uncertainty before update → slightly less precise after
        self.assertGreaterEqual(updated_long.se_personal, updated_short.se_personal)

    def test_trajectory_accumulates(self):
        """Trajectory stores observed theta and SE values."""
        cal = initial_calibration("test")
        for i in range(5):
            cal = update_calibration(cal, theta_observed=float(i), se_observed=0.5)

        self.assertEqual(len(cal.trajectory_thetas), 5)
        self.assertEqual(len(cal.trajectory_ses), 5)
        self.assertAlmostEqual(cal.trajectory_thetas[-1], 4.0, places=4)

    def test_se_floored_at_minimum(self):
        """SE of 0 should not crash — floored at 0.01."""
        cal = initial_calibration("test")
        updated = update_calibration(cal, theta_observed=1.0, se_observed=0.0)
        self.assertIsNotNone(updated)
        self.assertGreater(updated.se_personal, 0)


class TestTrajectoryManagement(unittest.TestCase):
    """Rolling trajectory window."""

    def test_trajectory_capped_at_max_length(self):
        """Trajectory should not exceed MAX_TRAJECTORY_LENGTH."""
        cal = initial_calibration("test")
        for i in range(MAX_TRAJECTORY_LENGTH + 20):
            cal = update_calibration(cal, theta_observed=float(i % 10), se_observed=0.5)

        self.assertEqual(len(cal.trajectory_thetas), MAX_TRAJECTORY_LENGTH)
        self.assertEqual(len(cal.trajectory_ses), MAX_TRAJECTORY_LENGTH)

    def test_oldest_values_dropped(self):
        """When trajectory is full, oldest values are dropped."""
        cal = initial_calibration("test")
        for i in range(MAX_TRAJECTORY_LENGTH + 5):
            cal = update_calibration(cal, theta_observed=float(i), se_observed=0.5)

        # The first value in trajectory should be 5.0 (0-4 dropped)
        self.assertAlmostEqual(cal.trajectory_thetas[0], 5.0, places=4)


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------

class TestPhaseTransitions(unittest.TestCase):

    def test_warming_phase(self):
        for n in range(0, WARMING_MIN_OBS):
            self.assertEqual(_determine_phase(n), "warming")

    def test_calibrating_phase(self):
        for n in range(WARMING_MIN_OBS, CALIBRATED_MIN_OBS):
            self.assertEqual(_determine_phase(n), "calibrating")

    def test_calibrated_phase(self):
        for n in (CALIBRATED_MIN_OBS, CALIBRATED_MIN_OBS + 10, 100):
            self.assertEqual(_determine_phase(n), "calibrated")

    def test_phase_transition_in_update(self):
        """Phase transitions happen automatically during update."""
        cal = initial_calibration("test")
        self.assertEqual(cal.phase, "warming")

        # Warming phase
        for _ in range(WARMING_MIN_OBS - 1):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
        self.assertEqual(cal.phase, "warming")

        # Transition to calibrating
        cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
        self.assertEqual(cal.phase, "calibrating")
        self.assertEqual(cal.n_observations, WARMING_MIN_OBS)

        # Continue to calibrated
        for _ in range(CALIBRATED_MIN_OBS - WARMING_MIN_OBS):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
        self.assertEqual(cal.phase, "calibrated")


# ---------------------------------------------------------------------------
# Baseline establishment
# ---------------------------------------------------------------------------

class TestBaseline(unittest.TestCase):

    def test_no_baseline_during_warming(self):
        """Baseline is None during warming phase."""
        cal = initial_calibration("test")
        for _ in range(WARMING_MIN_OBS - 1):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
        self.assertIsNone(cal.theta_baseline)

    def test_baseline_set_at_warming_end(self):
        """Baseline established when n_observations reaches WARMING_MIN_OBS."""
        cal = initial_calibration("test")
        for _ in range(WARMING_MIN_OBS):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)

        self.assertIsNotNone(cal.theta_baseline)
        self.assertIsNotNone(cal.se_baseline)
        # Baseline should be close to the observed theta
        self.assertAlmostEqual(cal.theta_baseline, 0.5, delta=0.3)

    def test_baseline_does_not_change_after_set(self):
        """Once established, baseline remains fixed."""
        cal = initial_calibration("test")
        for _ in range(WARMING_MIN_OBS):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
        baseline = cal.theta_baseline

        # Additional observations should not change baseline
        for _ in range(20):
            cal = update_calibration(cal, theta_observed=2.0, se_observed=0.3)
        self.assertAlmostEqual(cal.theta_baseline, baseline, places=6)


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------

class TestShrinkage(unittest.TestCase):

    def test_initial_shrinkage_zero(self):
        """No observations → shrinkage = 0 (pure population prior)."""
        cal = initial_calibration("test")
        self.assertAlmostEqual(cal.shrinkage, 0.0, places=6)

    def test_shrinkage_increases_with_observations(self):
        """Shrinkage should increase as we accumulate data."""
        cal = initial_calibration("test")
        shrinkages = []
        for _ in range(30):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
            shrinkages.append(cal.shrinkage)

        # First shrinkage should be positive
        self.assertGreater(shrinkages[0], 0.0)
        # Last shrinkage should be higher than first
        self.assertGreater(shrinkages[-1], shrinkages[0])

    def test_shrinkage_bounded_0_1(self):
        """Shrinkage always in [0, 1]."""
        cal = initial_calibration("test")
        for _ in range(50):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
            self.assertGreaterEqual(cal.shrinkage, 0.0)
            self.assertLessEqual(cal.shrinkage, 1.0)


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------

class TestEffectiveSampleSize(unittest.TestCase):

    def test_initial_n_effective_zero(self):
        cal = initial_calibration("test")
        self.assertAlmostEqual(cal.n_effective, 0.0, places=4)

    def test_n_effective_increases(self):
        """n_effective should increase with observations."""
        cal = initial_calibration("test")
        for _ in range(20):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)

        self.assertGreater(cal.n_effective, 1.0)

    def test_n_effective_interpretation(self):
        """n_effective = pop_sigma^2 / se_personal^2 ≈ precision gain."""
        cal = initial_calibration("test", population_sigma=1.0)
        for _ in range(20):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)

        expected = 1.0 / (cal.se_personal ** 2)
        self.assertAlmostEqual(cal.n_effective, expected, places=2)


# ---------------------------------------------------------------------------
# Reliable Change Index
# ---------------------------------------------------------------------------

class TestReliableChange(unittest.TestCase):

    def _build_calibrated(
        self,
        baseline_theta: float = 0.5,
        current_theta: float = 0.5,
        se: float = 0.2,
    ) -> PersonalCalibration:
        """Build a calibration state with specific baseline and current theta."""
        cal = initial_calibration("test")
        # Warm up with baseline theta
        for _ in range(WARMING_MIN_OBS):
            cal = update_calibration(cal, theta_observed=baseline_theta, se_observed=se)
        # Push current theta
        for _ in range(5):
            cal = update_calibration(cal, theta_observed=current_theta, se_observed=se)
        return cal

    def test_no_rci_during_warming(self):
        """RCI returns None when no baseline exists."""
        cal = initial_calibration("test")
        for _ in range(WARMING_MIN_OBS - 1):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
        self.assertIsNone(assess_reliable_change(cal))

    def test_stable_no_significant_change(self):
        """Stable theta → RCI not significant."""
        cal = self._build_calibrated(baseline_theta=0.5, current_theta=0.5, se=0.3)
        rci = assess_reliable_change(cal)
        self.assertIsNotNone(rci)
        assert rci is not None
        self.assertFalse(rci.significant)
        self.assertEqual(rci.direction, "stable")
        self.assertEqual(rci.magnitude, "minimal")

    def test_large_improvement_significant(self):
        """Large positive shift → significant improvement."""
        cal = self._build_calibrated(baseline_theta=0.0, current_theta=2.0, se=0.2)
        rci = assess_reliable_change(cal)
        assert rci is not None
        self.assertTrue(rci.significant)
        self.assertEqual(rci.direction, "improved")
        self.assertGreater(rci.rci, 1.96)

    def test_large_decline_significant(self):
        """Large negative shift → significant decline."""
        cal = self._build_calibrated(baseline_theta=1.0, current_theta=-1.0, se=0.2)
        rci = assess_reliable_change(cal)
        assert rci is not None
        self.assertTrue(rci.significant)
        self.assertEqual(rci.direction, "declined")
        self.assertLess(rci.rci, -1.96)

    def test_rci_formula_hand_check(self):
        """Verify RCI = delta_theta / se_diff against hand calculation."""
        cal = self._build_calibrated(baseline_theta=0.5, current_theta=1.5, se=0.15)
        rci = assess_reliable_change(cal)
        assert rci is not None
        # Hand check: se_diff = sqrt(se_personal^2 + se_baseline^2)
        expected_se_diff = math.sqrt(cal.se_personal ** 2 + cal.se_baseline ** 2)
        expected_delta = cal.theta_personal - cal.theta_baseline
        expected_rci = expected_delta / expected_se_diff
        self.assertAlmostEqual(rci.rci, round(expected_rci, 4), places=3)

    def test_mid_exceeded(self):
        """MID flag should be True when |delta| >= 0.5."""
        cal = self._build_calibrated(baseline_theta=0.0, current_theta=1.0, se=0.15)
        rci = assess_reliable_change(cal, mid=0.5)
        assert rci is not None
        self.assertTrue(rci.mid_exceeded)

    def test_mid_not_exceeded(self):
        """MID flag False for small changes."""
        cal = self._build_calibrated(baseline_theta=0.5, current_theta=0.55, se=0.15)
        rci = assess_reliable_change(cal, mid=0.5)
        assert rci is not None
        self.assertFalse(rci.mid_exceeded)

    def test_magnitude_categories(self):
        """Change magnitude: minimal (<0.2), moderate (0.2-0.5), large (>=0.5)."""
        # Minimal
        cal = self._build_calibrated(baseline_theta=0.0, current_theta=0.1, se=0.15)
        rci = assess_reliable_change(cal)
        assert rci is not None
        self.assertEqual(rci.magnitude, "minimal")

    def test_p_value_range(self):
        """p_value should be in (0, 1]."""
        cal = self._build_calibrated(baseline_theta=0.5, current_theta=1.5, se=0.2)
        rci = assess_reliable_change(cal)
        assert rci is not None
        self.assertGreater(rci.p_value, 0.0)
        self.assertLessEqual(rci.p_value, 1.0)


# ---------------------------------------------------------------------------
# Measurement sufficiency
# ---------------------------------------------------------------------------

class TestMeasurementSufficiency(unittest.TestCase):

    def test_initial_not_sufficient(self):
        """Population prior (SE=1.0) is not sufficient for MID=0.5."""
        cal = initial_calibration("test")
        suff = assess_measurement_sufficiency(cal)
        self.assertFalse(suff.sufficient)
        self.assertLess(suff.precision_ratio, 1.0)

    def test_precise_is_sufficient(self):
        """After many observations, SE should be small enough."""
        cal = initial_calibration("test")
        for _ in range(50):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.2)
        suff = assess_measurement_sufficiency(cal, mid=0.5)
        self.assertTrue(suff.sufficient)
        self.assertGreater(suff.precision_ratio, 1.0)

    def test_se_target_from_mid(self):
        """se_target = MID / z_critical."""
        cal = initial_calibration("test")
        suff = assess_measurement_sufficiency(cal, mid=0.5, alpha=0.05)
        z_crit = _z_critical(0.05)
        expected_target = 0.5 / z_crit
        self.assertAlmostEqual(suff.se_target, expected_target, places=4)

    def test_reliability_formula(self):
        """reliability = 1 - SE^2 / pop_sigma^2."""
        cal = initial_calibration("test", population_sigma=1.0)
        for _ in range(30):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
        suff = assess_measurement_sufficiency(cal)
        expected_rel = 1.0 - cal.se_personal ** 2
        self.assertAlmostEqual(suff.reliability, max(0, expected_rel), places=4)

    def test_reliability_bounded_0_1(self):
        """Reliability always in [0, 1]."""
        # Initial state: SE=1, reliability=0
        cal = initial_calibration("test")
        suff = assess_measurement_sufficiency(cal)
        self.assertGreaterEqual(suff.reliability, 0.0)
        self.assertLessEqual(suff.reliability, 1.0)


# ---------------------------------------------------------------------------
# Within-person variability
# ---------------------------------------------------------------------------

class TestWithinPersonSD(unittest.TestCase):

    def test_insufficient_observations(self):
        """Fewer than MIN_OBS_FOR_WITHIN_SD → None."""
        thetas = [0.5] * (MIN_OBS_FOR_WITHIN_SD - 1)
        ses = [0.3] * (MIN_OBS_FOR_WITHIN_SD - 1)
        self.assertIsNone(estimate_within_person_sd(thetas, ses))

    def test_constant_theta_zero_variability(self):
        """Constant theta with some measurement error → within_sd ≈ 0."""
        thetas = [0.5] * 20
        ses = [0.3] * 20
        sd = estimate_within_person_sd(thetas, ses)
        self.assertIsNotNone(sd)
        assert sd is not None
        self.assertAlmostEqual(sd, 0.0, places=4)

    def test_variable_theta_positive_sd(self):
        """Varying theta → positive within-person SD."""
        # Theta varies with SD~0.5, SE=0.1 (measurement error is small)
        import random
        random.seed(42)
        true_sd = 0.5
        se = 0.1
        thetas = [random.gauss(0.0, true_sd) for _ in range(100)]
        ses = [se] * 100
        sd = estimate_within_person_sd(thetas, ses)
        assert sd is not None
        # Should be close to true_sd (within tolerance)
        self.assertAlmostEqual(sd, true_sd, delta=0.15)

    def test_measurement_error_correction(self):
        """When obs variance < mean(SE^2), within_sd floors at 0."""
        # Constant true theta but large SE → correction gives 0
        thetas = [0.5 + 0.01 * i for i in range(15)]  # small variation
        ses = [5.0] * 15  # very large SE
        sd = estimate_within_person_sd(thetas, ses)
        assert sd is not None
        self.assertAlmostEqual(sd, 0.0, places=4)

    def test_within_sd_in_calibration_update(self):
        """within_person_sd populated after MIN_OBS_FOR_WITHIN_SD updates."""
        cal = initial_calibration("test")
        for i in range(MIN_OBS_FOR_WITHIN_SD - 1):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
        self.assertIsNone(cal.within_person_sd)

        cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
        self.assertIsNotNone(cal.within_person_sd)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

class TestStatisticalHelpers(unittest.TestCase):

    def test_normal_sf_at_zero(self):
        """P(Z > 0) = 0.5."""
        self.assertAlmostEqual(_normal_sf(0.0), 0.5, places=6)

    def test_normal_sf_at_196(self):
        """P(Z > 1.96) ≈ 0.025."""
        self.assertAlmostEqual(_normal_sf(1.96), 0.025, places=3)

    def test_normal_sf_symmetry(self):
        """sf(-z) = 1 - sf(z)."""
        for z in (0.5, 1.0, 1.5, 2.0):
            self.assertAlmostEqual(_normal_sf(-z), 1.0 - _normal_sf(z), places=6)

    def test_z_critical_005(self):
        """z_critical for alpha=0.05 ≈ 1.96."""
        z = _z_critical(0.05)
        self.assertAlmostEqual(z, 1.96, delta=0.01)

    def test_z_critical_001(self):
        """z_critical for alpha=0.01 ≈ 2.576."""
        z = _z_critical(0.01)
        self.assertAlmostEqual(z, 2.576, delta=0.01)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):

    def test_round_trip(self):
        """to_dict → from_dict preserves all fields."""
        cal = initial_calibration("phq9", population_mu=0.5, population_sigma=1.5)
        for _ in range(15):
            cal = update_calibration(cal, theta_observed=0.8, se_observed=0.25)

        d = calibration_to_dict(cal)
        restored = calibration_from_dict(d)

        self.assertEqual(restored.scale_id, cal.scale_id)
        self.assertEqual(restored.n_observations, cal.n_observations)
        self.assertAlmostEqual(restored.theta_personal, cal.theta_personal, places=6)
        self.assertAlmostEqual(restored.se_personal, cal.se_personal, places=6)
        self.assertEqual(restored.phase, cal.phase)
        self.assertAlmostEqual(restored.shrinkage, cal.shrinkage, places=6)
        self.assertEqual(len(restored.trajectory_thetas), len(cal.trajectory_thetas))
        self.assertEqual(restored.version, cal.version)

    def test_dict_has_required_keys(self):
        """Serialized dict contains all expected keys."""
        cal = initial_calibration("test")
        d = calibration_to_dict(cal)
        required = {
            "scale_id", "n_observations", "theta_personal", "se_personal",
            "theta_population_mu", "theta_population_sigma",
            "theta_baseline", "se_baseline", "within_person_sd",
            "trajectory_thetas", "trajectory_ses", "phase",
            "shrinkage", "n_effective", "process_noise_sd", "version",
        }
        for key in required:
            self.assertIn(key, d, f"Missing key: {key}")

    def test_round_trip_with_none_fields(self):
        """Round trip works when baseline/within_sd are None."""
        cal = initial_calibration("test")
        d = calibration_to_dict(cal)
        restored = calibration_from_dict(d)
        self.assertIsNone(restored.theta_baseline)
        self.assertIsNone(restored.within_person_sd)

    def test_trajectory_serialized_as_list(self):
        """Trajectory is stored as a list (JSON-compatible)."""
        cal = initial_calibration("test")
        cal = update_calibration(cal, theta_observed=0.5, se_observed=0.3)
        d = calibration_to_dict(cal)
        self.assertIsInstance(d["trajectory_thetas"], list)
        self.assertIsInstance(d["trajectory_ses"], list)


# ---------------------------------------------------------------------------
# Integration: full lifecycle
# ---------------------------------------------------------------------------

class TestFullLifecycle(unittest.TestCase):
    """End-to-end simulation of a person being measured over time."""

    def test_30_day_stable_person(self):
        """Person with stable theta ≈ 0.5 over 30 days."""
        cal = initial_calibration("phq9")
        for day in range(30):
            # Simulate IRT scoring with some noise
            theta_obs = 0.5 + 0.1 * math.sin(day * 0.5)  # slight oscillation
            cal = update_calibration(
                cal, theta_observed=theta_obs, se_observed=0.25,
                days_since_last=1.0,
            )

        # Should be calibrated
        self.assertEqual(cal.phase, "calibrated")
        # Theta should be near 0.5
        self.assertAlmostEqual(cal.theta_personal, 0.5, delta=0.2)
        # High shrinkage
        self.assertGreater(cal.shrinkage, 0.5)
        # Baseline established
        self.assertIsNotNone(cal.theta_baseline)
        # Within-person SD should be small (oscillation ≈ 0.07 SD)
        self.assertIsNotNone(cal.within_person_sd)
        # No significant change (stable)
        rci = assess_reliable_change(cal)
        assert rci is not None
        self.assertFalse(rci.significant)

    def test_intervention_detected(self):
        """Person improves after an intervention → RCI detects change."""
        cal = initial_calibration("phq9")

        # Phase 1: baseline at theta ≈ -0.5 (below average)
        for _ in range(15):
            cal = update_calibration(cal, theta_observed=-0.5, se_observed=0.25)

        baseline = cal.theta_baseline
        assert baseline is not None
        self.assertAlmostEqual(baseline, -0.5, delta=0.3)

        # Phase 2: intervention → theta improves to +1.0
        for _ in range(20):
            cal = update_calibration(cal, theta_observed=1.0, se_observed=0.25)

        # Should detect significant improvement
        rci = assess_reliable_change(cal)
        assert rci is not None
        self.assertTrue(rci.significant)
        self.assertEqual(rci.direction, "improved")
        self.assertTrue(rci.mid_exceeded)

    def test_measurement_becomes_sufficient(self):
        """Sufficiency starts False and becomes True with data."""
        cal = initial_calibration("test")
        suff0 = assess_measurement_sufficiency(cal)
        self.assertFalse(suff0.sufficient)

        for _ in range(40):
            cal = update_calibration(cal, theta_observed=0.5, se_observed=0.2)

        suff40 = assess_measurement_sufficiency(cal, mid=0.5)
        self.assertTrue(suff40.sufficient)
        self.assertGreater(suff40.reliability, 0.8)

    def test_irregular_measurement_schedule(self):
        """System handles irregular gaps without errors."""
        cal = initial_calibration("test")
        gaps = [1, 1, 3, 1, 7, 1, 1, 14, 1, 1, 2, 1, 30, 1, 1, 1]
        for gap in gaps:
            cal = update_calibration(
                cal, theta_observed=0.5, se_observed=0.3,
                days_since_last=float(gap),
            )
        self.assertEqual(cal.n_observations, len(gaps))
        # Should still have valid estimates
        self.assertIsNotNone(cal.theta_personal)
        self.assertGreater(cal.se_personal, 0)


if __name__ == "__main__":
    unittest.main()
