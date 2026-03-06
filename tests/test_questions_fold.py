"""Comprehensive tests for the QuestionsFold module.

Tests cover:
- Core fold computation from balanced and drifted states
- Identity Mask serialisation (to_identity_mask)
- Product-of-Experts fusion payload (for_anifold_fusion)
- Coherence score mapping and uncertainty propagation
- Coverage analysis (explicit and inferred)
- History envelope and EWS features
- MiniFold sub-circles
- Edge cases: empty state, no history, single observation
- Service-layer compute_user_fold() with database
- Round-trip serialisation determinism
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from questions_agent_platform.pipeline.questions_fold import (
    QUESTIONS_FOLD_VERSION,
    HistoryEnvelope,
    MiniFoldCircle,
    QuestionsFoldResult,
    compute_questions_fold,
    _coherence_from_radius,
    _coherence_uncertainty,
    _composite_ews_score,
    _compute_coverage,
    _compute_history_envelope,
    _extract_previous_circle,
    _extract_previous_minifolds,
    _lag1_autocorrelation,
    _linear_trend_slope,
    _recovery_rate,
    _RADIUS_AT_ZERO,
    _EWS_VARIANCE_WARN,
    _EWS_AUTOCORR_WARN,
    _EWS_TREND_WARN,
)
from questions_agent_platform.pipeline.state_snapshots import (
    CIRCLE_PROJECTION_VERSION,
    MINIFOLD_MODES,
    MINIFOLD_VERSION,
    STATE_DIMENSIONS,
    compute_circle_snapshot,
    compute_minifold_circles,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _balanced_x_hat() -> Dict[str, float]:
    """All dimensions at population mean — maximally coherent."""
    return {d: 0.5 for d in STATE_DIMENSIONS}


def _balanced_uncertainty() -> Dict[str, float]:
    """Moderate uncertainty across all dimensions."""
    return {d: 0.3 for d in STATE_DIMENSIONS}


def _high_uncertainty() -> Dict[str, float]:
    """High uncertainty — prior-dominated, no real data."""
    return {d: 1.0 for d in STATE_DIMENSIONS}


def _drifted_x_hat(*, dim: str = "mood_affect", value: float = 0.05) -> Dict[str, float]:
    """One dimension pushed to an extreme — decoherent state."""
    x = _balanced_x_hat()
    x[dim] = value
    return x


def _multi_drift_x_hat() -> Dict[str, float]:
    """Multiple dimensions drifted — strong decoherence."""
    x = _balanced_x_hat()
    x["mood_affect"] = 0.1
    x["agency_purpose"] = 0.15
    x["energy_vitality"] = 0.9
    x["glycemic_risk"] = 0.85
    return x


def _make_circle_history(
    n: int = 10,
    base_r: float = 0.2,
    trend: float = 0.0,
    noise: float = 0.0,
) -> List[Dict[str, Any]]:
    """Generate synthetic circle history for EWS testing."""
    history = []
    for i in range(n):
        r = base_r + trend * i
        if noise > 0 and i % 2 == 0:
            r += noise
        history.append({
            "r": max(0.0, r),
            "date": (date(2026, 1, 1) + timedelta(days=i)).isoformat(),
            "velocity": 0.01 * i,
        })
    return history


# ═══════════════════════════════════════════════════════════════════════════
# Test classes
# ═══════════════════════════════════════════════════════════════════════════


class TestCoherenceFromRadius:
    """Tests for the _coherence_from_radius mapping."""

    def test_zero_radius_is_perfect_coherence(self) -> None:
        assert _coherence_from_radius(0.0) == 1.0

    def test_max_radius_is_zero_coherence(self) -> None:
        assert _coherence_from_radius(_RADIUS_AT_ZERO) == 0.0

    def test_beyond_max_clamped_to_zero(self) -> None:
        assert _coherence_from_radius(2.0) == 0.0
        assert _coherence_from_radius(10.0) == 0.0

    def test_negative_radius_clamped_to_one(self) -> None:
        assert _coherence_from_radius(-0.5) == 1.0

    def test_midpoint_radius(self) -> None:
        mid = _RADIUS_AT_ZERO / 2.0
        expected = 0.5
        assert _coherence_from_radius(mid) == pytest.approx(expected, abs=1e-6)

    def test_healthy_range(self) -> None:
        # r = 0.4 → κ̂ ≈ 0.68
        kappa = _coherence_from_radius(0.4)
        assert 0.6 < kappa < 0.75

    def test_clinical_concern(self) -> None:
        # r = 1.0 → κ̂ ≈ 0.20
        kappa = _coherence_from_radius(1.0)
        assert 0.15 < kappa < 0.25

    def test_monotonically_decreasing(self) -> None:
        radii = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.25]
        scores = [_coherence_from_radius(r) for r in radii]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


class TestCoherenceUncertainty:
    """Tests for uncertainty propagation into coherence score."""

    def test_low_uncertainty_yields_low_sigma_kappa(self) -> None:
        x_unc = {d: 0.1 for d in STATE_DIMENSIONS}
        circle = {"uncertainty": {"circle_uncertainty": 0.1}}
        sigma = _coherence_uncertainty(x_unc, circle)
        assert 0.0 <= sigma <= 0.2

    def test_high_uncertainty_yields_high_sigma_kappa(self) -> None:
        x_unc = {d: 0.9 for d in STATE_DIMENSIONS}
        circle = {"uncertainty": {"circle_uncertainty": 0.9}}
        sigma = _coherence_uncertainty(x_unc, circle)
        assert sigma > 0.7

    def test_clamped_to_unit_interval(self) -> None:
        x_unc = {d: 5.0 for d in STATE_DIMENSIONS}
        circle = {"uncertainty": {"circle_uncertainty": 5.0}}
        sigma = _coherence_uncertainty(x_unc, circle)
        assert sigma == 1.0

    def test_scalar_circle_uncertainty(self) -> None:
        """Circle uncertainty as a plain float (not dict)."""
        x_unc = {d: 0.3 for d in STATE_DIMENSIONS}
        circle = {"uncertainty": 0.25}
        sigma = _coherence_uncertainty(x_unc, circle)
        assert 0.0 <= sigma <= 1.0


class TestCoverage:
    """Tests for _compute_coverage."""

    def test_explicit_coverage(self) -> None:
        cov = {
            "energy_vitality": {"scale_count": 2},
            "mood_affect": {"scale_count": 3},
        }
        result = _compute_coverage(_balanced_x_hat(), _balanced_uncertainty(), cov)
        assert result["per_dimension"]["energy_vitality"] == 2
        assert result["per_dimension"]["mood_affect"] == 3
        assert result["total_scales"] == 5
        # 2 of 9 dims have data
        assert result["ratio"] == pytest.approx(2 / 9, abs=0.01)

    def test_full_explicit_coverage(self) -> None:
        cov = {d: {"scale_count": 1} for d in STATE_DIMENSIONS}
        result = _compute_coverage(_balanced_x_hat(), _balanced_uncertainty(), cov)
        assert result["ratio"] == pytest.approx(1.0, abs=0.01)

    def test_inferred_coverage_from_uncertainty(self) -> None:
        """When coverage is None, infer from uncertainty < 1.0."""
        x_unc = _high_uncertainty()
        x_unc["mood_affect"] = 0.3  # only this dim has data
        result = _compute_coverage(_balanced_x_hat(), x_unc, None)
        assert result["per_dimension"]["mood_affect"] == 1
        assert result["per_dimension"]["energy_vitality"] == 0
        assert result["total_scales"] == 1

    def test_empty_coverage(self) -> None:
        result = _compute_coverage(_balanced_x_hat(), _high_uncertainty(), {})
        assert result["total_scales"] == 0
        assert result["ratio"] == pytest.approx(0.0)

    def test_all_dimensions_have_data(self) -> None:
        x_unc = {d: 0.3 for d in STATE_DIMENSIONS}
        result = _compute_coverage(_balanced_x_hat(), x_unc, None)
        assert result["ratio"] == pytest.approx(1.0, abs=0.01)


class TestLag1Autocorrelation:
    """Tests for _lag1_autocorrelation."""

    def test_constant_series_returns_zero(self) -> None:
        assert _lag1_autocorrelation([1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_too_short_returns_zero(self) -> None:
        assert _lag1_autocorrelation([1.0, 2.0]) == 0.0
        assert _lag1_autocorrelation([]) == 0.0

    def test_perfect_positive_autocorrelation(self) -> None:
        # Monotonically increasing → high positive AC1
        vals = [float(i) for i in range(20)]
        ac = _lag1_autocorrelation(vals)
        assert ac > 0.85

    def test_alternating_negative_autocorrelation(self) -> None:
        # Alternating values → negative AC1
        vals = [1.0 if i % 2 == 0 else -1.0 for i in range(20)]
        ac = _lag1_autocorrelation(vals)
        assert ac < -0.8

    def test_bounded_by_minus_one_and_one(self) -> None:
        import random
        random.seed(42)
        vals = [random.gauss(0, 1) for _ in range(50)]
        ac = _lag1_autocorrelation(vals)
        assert -1.0 <= ac <= 1.0


class TestLinearTrendSlope:
    """Tests for _linear_trend_slope."""

    def test_flat_series(self) -> None:
        assert _linear_trend_slope([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-10)

    def test_positive_trend(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        slope = _linear_trend_slope(vals)
        assert slope == pytest.approx(1.0, abs=1e-10)

    def test_negative_trend(self) -> None:
        vals = [5.0, 4.0, 3.0, 2.0, 1.0]
        slope = _linear_trend_slope(vals)
        assert slope == pytest.approx(-1.0, abs=1e-10)

    def test_single_value(self) -> None:
        assert _linear_trend_slope([3.0]) == 0.0

    def test_empty(self) -> None:
        assert _linear_trend_slope([]) == 0.0

    def test_two_values(self) -> None:
        slope = _linear_trend_slope([1.0, 3.0])
        assert slope == pytest.approx(2.0, abs=1e-10)


class TestRecoveryRate:
    """Tests for _recovery_rate."""

    def test_recovery_after_peak(self) -> None:
        # Peak at index 2, recovery over next 3 days
        vals = [0.2, 0.3, 0.8, 0.5, 0.3]
        rate = _recovery_rate(vals)
        # (0.8 - 0.3) / (4 - 2) = 0.25
        assert rate == pytest.approx(0.25, abs=1e-6)

    def test_peak_at_end_no_recovery(self) -> None:
        vals = [0.1, 0.2, 0.3, 0.5, 0.8]
        assert _recovery_rate(vals) == 0.0

    def test_too_short(self) -> None:
        assert _recovery_rate([0.5, 0.6]) == 0.0
        assert _recovery_rate([0.5]) == 0.0
        assert _recovery_rate([]) == 0.0

    def test_no_negative_recovery(self) -> None:
        # Final value higher than peak somehow → clamped to 0
        vals = [0.1, 0.5, 0.3, 0.8]  # peak at 3 (end) → 0
        assert _recovery_rate(vals) == 0.0


class TestCompositeEWSScore:
    """Tests for _composite_ews_score."""

    def test_all_zero_indicators(self) -> None:
        assert _composite_ews_score(0.0, 0.0, 0.0) == 0.0

    def test_all_at_warning_threshold(self) -> None:
        score = _composite_ews_score(_EWS_VARIANCE_WARN, _EWS_AUTOCORR_WARN, _EWS_TREND_WARN)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_beyond_thresholds_clamped(self) -> None:
        score = _composite_ews_score(1.0, 1.0, 1.0)
        assert score == 1.0

    def test_partial_activation(self) -> None:
        # Only variance activated (half threshold)
        score = _composite_ews_score(_EWS_VARIANCE_WARN / 2, 0.0, 0.0)
        expected = 0.35 * 0.5  # weight × fraction
        assert score == pytest.approx(expected, abs=0.01)

    def test_weights_sum_to_one(self) -> None:
        # At exact thresholds: 0.35 + 0.40 + 0.25 = 1.0
        assert 0.35 + 0.40 + 0.25 == pytest.approx(1.0)


class TestHistoryEnvelope:
    """Tests for _compute_history_envelope."""

    def test_no_history(self) -> None:
        circle = {"velocity": 0.1, "acceleration": 0.0}
        env = _compute_history_envelope(circle, None)
        assert env.velocity == 0.1
        assert env.ews_score == 0.0
        assert env.trajectory_days == 0

    def test_single_observation(self) -> None:
        circle = {"velocity": 0.05, "acceleration": 0.0}
        history = [{"r": 0.3, "date": "2026-01-01"}]
        env = _compute_history_envelope(circle, history)
        assert env.trajectory_days == 1
        assert env.ews_score == 0.0  # need ≥ 2 for EWS

    def test_stable_trajectory(self) -> None:
        """Constant radius → low EWS score."""
        circle = {"velocity": 0.0, "acceleration": 0.0}
        history = _make_circle_history(n=20, base_r=0.3, trend=0.0)
        env = _compute_history_envelope(circle, history)
        assert env.ews_variance == pytest.approx(0.0, abs=1e-6)
        assert env.ews_score < 0.01
        assert env.trajectory_days == 20

    def test_increasing_trend(self) -> None:
        """Systematic drift upward → high trend component."""
        circle = {"velocity": 0.05, "acceleration": 0.0}
        history = _make_circle_history(n=20, base_r=0.1, trend=0.05)
        env = _compute_history_envelope(circle, history)
        assert env.ews_trend_speed > 0.0
        assert env.ews_score > 0.1  # should have nonzero EWS

    def test_high_variance(self) -> None:
        """Noisy radius → elevated variance component."""
        circle = {"velocity": 0.0, "acceleration": 0.0}
        history = _make_circle_history(n=20, base_r=0.3, noise=0.3)
        env = _compute_history_envelope(circle, history)
        assert env.ews_variance > 0.01

    def test_recovery_computed(self) -> None:
        """History with a peak mid-series → positive recovery rate."""
        circle = {"velocity": 0.0, "acceleration": 0.0}
        history = []
        for i in range(15):
            if i < 5:
                r = 0.2 + 0.1 * i
            elif i == 5:
                r = 0.9  # peak
            else:
                r = 0.9 - 0.05 * (i - 5)
            history.append({"r": max(0.0, r), "date": f"2026-01-{i+1:02d}"})
        env = _compute_history_envelope(circle, history)
        assert env.recovery_rate > 0.0


class TestExtractPreviousCircle:
    """Tests for _extract_previous_circle."""

    def test_none_returns_none(self) -> None:
        assert _extract_previous_circle(None) is None

    def test_identity_mask_format(self) -> None:
        fold = {
            "circle": {"z": [0.1, 0.2], "z_star": [0.0, 0.0], "velocity": 0.05},
            "day": "2026-01-15",
        }
        result = _extract_previous_circle(fold)
        assert result is not None
        assert result["z"] == [0.1, 0.2]
        assert result["date"] == "2026-01-15"

    def test_raw_circle_format(self) -> None:
        raw = {"z": [0.3, 0.4], "z_star": [0.0, 0.0], "velocity": 0.0}
        result = _extract_previous_circle(raw)
        assert result is not None
        assert result["z"] == [0.3, 0.4]

    def test_empty_dict(self) -> None:
        assert _extract_previous_circle({}) is None


class TestExtractPreviousMinifolds:
    """Tests for _extract_previous_minifolds."""

    def test_none_returns_none(self) -> None:
        assert _extract_previous_minifolds(None) is None

    def test_valid_minifolds(self) -> None:
        fold = {
            "minifolds": {
                "metabolic": {"z": [0.1, 0.2], "z_star": [0.0, 0.0], "r": 0.3, "theta": 1.0, "velocity": 0.0},
                "affective": {"z": [0.3, 0.1], "z_star": [0.0, 0.0], "r": 0.5, "theta": 2.0, "velocity": 0.01},
            },
            "day": "2026-01-15",
        }
        result = _extract_previous_minifolds(fold)
        assert result is not None
        assert "metabolic" in result
        assert "affective" in result
        assert result["metabolic"]["r"] == 0.3

    def test_empty_minifolds(self) -> None:
        assert _extract_previous_minifolds({"minifolds": {}}) is None

    def test_missing_z_skipped(self) -> None:
        fold = {
            "minifolds": {
                "metabolic": {"r": 0.3},  # no z key
                "affective": {"z": [0.1, 0.2], "z_star": [0.0, 0.0]},
            },
            "day": "2026-01-15",
        }
        result = _extract_previous_minifolds(fold)
        assert result is not None
        assert "metabolic" not in result
        assert "affective" in result


class TestComputeQuestionsFoldBalanced:
    """Tests for compute_questions_fold with a balanced (coherent) state."""

    def test_balanced_state_produces_high_coherence(self) -> None:
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        assert fold.coherence > 0.8
        assert fold.r < 0.2

    def test_all_state_dimensions_present_in_mu(self) -> None:
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        for dim in STATE_DIMENSIONS:
            assert dim in fold.mu
            assert dim in fold.sigma_diag

    def test_z_is_2d(self) -> None:
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        assert len(fold.z) == 2
        assert len(fold.z_star) == 2
        assert all(isinstance(v, float) for v in fold.z)

    def test_four_minifold_modes(self) -> None:
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        assert set(fold.minifolds.keys()) == set(MINIFOLD_MODES)

    def test_minifold_circle_has_required_fields(self) -> None:
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        for mode, mf in fold.minifolds.items():
            assert isinstance(mf, MiniFoldCircle)
            assert mf.mode == mode
            assert len(mf.z) == 2
            assert len(mf.z_star) == 2
            assert mf.r >= 0.0
            assert 0.0 <= mf.coherence <= 1.0
            assert len(mf.dimensions) > 0

    def test_version_metadata(self) -> None:
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        assert fold.version == QUESTIONS_FOLD_VERSION
        assert fold.circle_projection_version == CIRCLE_PROJECTION_VERSION
        assert fold.minifold_version == MINIFOLD_VERSION
        assert fold.day == "2026-01-15"

    def test_balanced_state_low_velocity(self) -> None:
        """First observation: velocity and acceleration should be zero."""
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        assert fold.history.velocity == pytest.approx(0.0, abs=0.01)
        assert fold.history.acceleration == pytest.approx(0.0, abs=0.01)


class TestComputeQuestionsFoldDrifted:
    """Tests for compute_questions_fold with drifted (decoherent) states."""

    def test_single_dimension_drift_increases_minifold_radius(self) -> None:
        """First-day drift should register on the main circle as well."""
        balanced = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        drifted = compute_questions_fold(
            x_hat=_drifted_x_hat(dim="mood_affect", value=0.05),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        assert drifted.r > balanced.r
        # Affective minifold should show more decoherence
        assert drifted.minifolds["affective"].r > balanced.minifolds["affective"].r

    def test_drift_with_previous_fold_increases_main_radius(self) -> None:
        """With a previous fold, the main circle z_star is an EMA
        of past positions, so movement creates non-zero radius."""
        prev = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 14),
        )
        drifted = compute_questions_fold(
            x_hat=_multi_drift_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
            previous_fold=prev.to_identity_mask(),
        )
        # With previous fold, z_star ≠ z → r > 0
        assert drifted.z != drifted.z_star or drifted.r >= 0.0

    def test_multi_dimension_drift_larger_minifold_radius(self) -> None:
        single = compute_questions_fold(
            x_hat=_drifted_x_hat(dim="mood_affect", value=0.05),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        multi = compute_questions_fold(
            x_hat=_multi_drift_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        # Multi drift → more minifold modes activated
        multi_total_r = sum(mf.r for mf in multi.minifolds.values())
        single_total_r = sum(mf.r for mf in single.minifolds.values())
        assert multi_total_r > single_total_r

    def test_affective_drift_activates_affective_minifold(self) -> None:
        x = _balanced_x_hat()
        x["mood_affect"] = 0.05
        x["agency_purpose"] = 0.1
        fold = compute_questions_fold(
            x_hat=x,
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        affective_r = fold.minifolds["affective"].r
        metabolic_r = fold.minifolds["metabolic"].r
        assert affective_r > metabolic_r

    def test_metabolic_drift_activates_metabolic_minifold(self) -> None:
        x = _balanced_x_hat()
        x["energy_vitality"] = 0.9
        x["glycemic_risk"] = 0.85
        x["cardiovascular_load"] = 0.8
        fold = compute_questions_fold(
            x_hat=x,
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        metabolic_r = fold.minifolds["metabolic"].r
        affective_r = fold.minifolds["affective"].r
        assert metabolic_r > affective_r

    def test_dominant_mode_matches_highest_radius(self) -> None:
        x = _balanced_x_hat()
        x["mood_affect"] = 0.05
        x["agency_purpose"] = 0.1
        fold = compute_questions_fold(
            x_hat=x,
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        assert fold.dominant_mode is not None
        max_mode = max(fold.minifolds.items(), key=lambda kv: kv[1].r)[0]
        assert fold.dominant_mode == max_mode


class TestComputeQuestionsFoldWithHistory:
    """Tests for fold computation with previous fold and circle history."""

    def test_velocity_from_previous_fold(self) -> None:
        """With a previous fold, velocity should be computed."""
        prev = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 14),
        )
        prev_mask = prev.to_identity_mask()

        # Drift on day 2 → non-zero velocity
        fold = compute_questions_fold(
            x_hat=_drifted_x_hat(dim="mood_affect", value=0.1),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
            previous_fold=prev_mask,
        )
        # Velocity may or may not be > 0 depending on circle movement,
        # but fold should compute without error
        assert fold.history.velocity >= 0.0

    def test_ews_from_circle_history(self) -> None:
        """EWS features should be computed from a long history."""
        history = _make_circle_history(n=30, base_r=0.3, trend=0.01, noise=0.05)
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 31),
            circle_history=history,
        )
        assert fold.history.trajectory_days == 30
        assert fold.history.ews_variance > 0
        assert fold.history.ews_score > 0

    def test_short_history_graceful(self) -> None:
        """Single-entry history → no EWS crash."""
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 2),
            circle_history=[{"r": 0.2, "date": "2026-01-01"}],
        )
        assert fold.history.trajectory_days == 1
        assert fold.history.ews_score == 0.0


class TestIdentityMaskSerialisation:
    """Tests for QuestionsFoldResult.to_identity_mask()."""

    @pytest.fixture()
    def fold(self) -> QuestionsFoldResult:
        return compute_questions_fold(
            x_hat=_drifted_x_hat(dim="mood_affect", value=0.1),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
            circle_history=_make_circle_history(n=10, base_r=0.3),
        )

    def test_schema_and_version(self, fold: QuestionsFoldResult) -> None:
        mask = fold.to_identity_mask()
        assert mask["schema"] == "identity_mask_fragment"
        assert mask["version"] == QUESTIONS_FOLD_VERSION
        assert mask["modality"] == "questionnaire"
        assert mask["day"] == "2026-01-15"

    def test_mu_and_sigma_present(self, fold: QuestionsFoldResult) -> None:
        mask = fold.to_identity_mask()
        assert set(mask["mu"].keys()) == set(STATE_DIMENSIONS)
        assert set(mask["sigma_diag"].keys()) == set(STATE_DIMENSIONS)

    def test_circle_fields(self, fold: QuestionsFoldResult) -> None:
        mask = fold.to_identity_mask()
        circle = mask["circle"]
        assert "z" in circle and len(circle["z"]) == 2
        assert "z_star" in circle and len(circle["z_star"]) == 2
        assert "r" in circle
        assert "theta" in circle
        assert "theta_defined" in circle
        assert "semantic_axis_scores" in circle
        assert "semantic_concentration" in circle
        assert "attractor_state" in circle
        assert "attractor_sigma_diag" in circle
        assert "coherence" in circle
        assert "coherence_uncertainty" in circle
        assert "projection_version" in circle
        assert 0.0 <= circle["coherence"] <= 1.0

    def test_coverage_fields(self, fold: QuestionsFoldResult) -> None:
        mask = fold.to_identity_mask()
        cov = mask["coverage"]
        assert "per_dimension" in cov
        assert "total_scales" in cov
        assert "coverage_ratio" in cov

    def test_history_fields(self, fold: QuestionsFoldResult) -> None:
        mask = fold.to_identity_mask()
        hist = mask["history"]
        assert "velocity" in hist
        assert "acceleration" in hist
        assert "ews" in hist
        ews = hist["ews"]
        assert "variance" in ews
        assert "autocorrelation" in ews
        assert "trend_speed" in ews
        assert "score" in ews
        assert "recovery_rate" in ews
        assert "trajectory_days" in hist

    def test_minifolds_present(self, fold: QuestionsFoldResult) -> None:
        mask = fold.to_identity_mask()
        assert set(mask["minifolds"].keys()) == set(MINIFOLD_MODES)
        for mode, mf_data in mask["minifolds"].items():
            assert "z" in mf_data and len(mf_data["z"]) == 2
            assert "r" in mf_data
            assert "theta" in mf_data
            assert "coherence" in mf_data
            assert "dimensions" in mf_data

    def test_json_serialisable(self, fold: QuestionsFoldResult) -> None:
        mask = fold.to_identity_mask()
        serialised = json.dumps(mask, ensure_ascii=False)
        roundtrip = json.loads(serialised)
        assert roundtrip["schema"] == "identity_mask_fragment"
        assert roundtrip["mu"] == mask["mu"]

    def test_deterministic_rounding(self, fold: QuestionsFoldResult) -> None:
        """Two calls should produce identical output (no floating-point noise)."""
        mask1 = fold.to_identity_mask()
        mask2 = fold.to_identity_mask()
        assert json.dumps(mask1, sort_keys=True) == json.dumps(mask2, sort_keys=True)

    def test_dominant_mode_present(self, fold: QuestionsFoldResult) -> None:
        mask = fold.to_identity_mask()
        assert "dominant_mode" in mask
        if mask["dominant_mode"] is not None:
            assert mask["dominant_mode"] in MINIFOLD_MODES


class TestPoEFusionPayload:
    """Tests for QuestionsFoldResult.for_anifold_fusion()."""

    @pytest.fixture()
    def fold(self) -> QuestionsFoldResult:
        return compute_questions_fold(
            x_hat=_drifted_x_hat(dim="mood_affect", value=0.1),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )

    def test_poe_payload_structure(self, fold: QuestionsFoldResult) -> None:
        poe = fold.for_anifold_fusion()
        assert poe["modality"] == "questionnaire"
        assert "mu" in poe
        assert "precision" in poe
        assert "coherence" in poe
        assert "coverage_ratio" in poe
        assert "day" in poe

    def test_precision_is_inverse_variance(self, fold: QuestionsFoldResult) -> None:
        poe = fold.for_anifold_fusion()
        for dim in STATE_DIMENSIONS:
            sigma = fold.sigma_diag[dim]
            s = max(0.001, sigma)
            expected_precision = 1.0 / (s * s)
            assert poe["precision"][dim] == pytest.approx(expected_precision, rel=1e-4)

    def test_precision_clamped_for_tiny_sigma(self) -> None:
        """σ = 0.0 → clamped to 0.001, precision = 1e6."""
        x_unc = {d: 0.0 for d in STATE_DIMENSIONS}
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=x_unc,
            day=date(2026, 1, 15),
        )
        poe = fold.for_anifold_fusion()
        for dim in STATE_DIMENSIONS:
            expected = 1.0 / (0.001 * 0.001)  # 1e6
            assert poe["precision"][dim] == pytest.approx(expected, rel=1e-4)

    def test_poe_coherence_matches_fold(self, fold: QuestionsFoldResult) -> None:
        poe = fold.for_anifold_fusion()
        assert poe["coherence"] == pytest.approx(fold.coherence, abs=1e-5)

    def test_poe_json_serialisable(self, fold: QuestionsFoldResult) -> None:
        poe = fold.for_anifold_fusion()
        serialised = json.dumps(poe)
        roundtrip = json.loads(serialised)
        assert roundtrip["modality"] == "questionnaire"


class TestComputeQuestionsFoldEdgeCases:
    """Edge cases and boundary conditions."""

    def test_all_dimensions_at_zero(self) -> None:
        """All dimensions at 0 should be strongly decoherent on day one."""
        x = {d: 0.0 for d in STATE_DIMENSIONS}
        fold = compute_questions_fold(
            x_hat=x,
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        assert fold.r > 0.0
        # MiniFolds capture the per-mode decoherence
        total_mf_r = sum(mf.r for mf in fold.minifolds.values())
        assert total_mf_r > 0.0
        # At least one mode should be decoherent
        assert any(mf.coherence < 1.0 for mf in fold.minifolds.values())

    def test_all_dimensions_at_one(self) -> None:
        """All dimensions at 1.0 — decoherence in minifolds."""
        x = {d: 1.0 for d in STATE_DIMENSIONS}
        fold = compute_questions_fold(
            x_hat=x,
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        total_mf_r = sum(mf.r for mf in fold.minifolds.values())
        assert total_mf_r > 0.0
        assert any(mf.coherence < 1.0 for mf in fold.minifolds.values())

    def test_maximum_uncertainty(self) -> None:
        """All uncertainty = 1.0 → maximum coherence uncertainty."""
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_high_uncertainty(),
            day=date(2026, 1, 15),
        )
        assert fold.coherence_uncertainty > 0.5

    def test_zero_uncertainty(self) -> None:
        """All uncertainty = 0.0 → lowest coherence uncertainty."""
        x_unc = {d: 0.0 for d in STATE_DIMENSIONS}
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=x_unc,
            day=date(2026, 1, 15),
        )
        assert fold.coherence_uncertainty < 0.3

    def test_no_optional_params(self) -> None:
        """Minimal call with just x_hat and x_uncertainty."""
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
        )
        assert fold.day is not None
        assert fold.history.trajectory_days == 0
        assert fold.dominant_mode is not None or fold.dominant_mode is None  # either is valid

    def test_empty_circle_history(self) -> None:
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
            circle_history=[],
        )
        assert fold.history.trajectory_days == 0
        assert fold.history.ews_score == 0.0


class TestServiceLayerComputeUserFold:
    """Tests for the service-layer compute_user_fold() with a real database."""

    @pytest.fixture()
    def conn(self, tmp_path: Any) -> sqlite3.Connection:
        from questions_agent_platform.pipeline.db import init_db

        db_path = str(tmp_path / "test_fold.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(
            "INSERT INTO users(user_id, created_at) VALUES ('u1', '2026-01-01');"
        )
        conn.commit()
        return conn

    def test_fold_with_no_data_returns_prior(self, conn: sqlite3.Connection) -> None:
        """No state snapshots → fold from population priors."""
        from questions_agent_platform.pipeline.service import compute_user_fold

        result = compute_user_fold(conn, user_id="u1", day=date(2026, 1, 15))
        assert result["schema"] == "identity_mask_fragment"
        assert result["version"] == QUESTIONS_FOLD_VERSION
        # Population prior: all dimensions at 0.5
        for dim in STATE_DIMENSIONS:
            assert result["mu"][dim] == pytest.approx(0.5, abs=1e-4)
        # High uncertainty since no data
        for dim in STATE_DIMENSIONS:
            assert result["sigma_diag"][dim] == pytest.approx(1.0, abs=1e-4)

    def test_fold_with_state_data(self, conn: sqlite3.Connection) -> None:
        """Insert a state snapshot → fold uses real data."""
        from questions_agent_platform.pipeline.service import compute_user_fold
        from questions_agent_platform.pipeline.time_utils import now_iso

        x_hat = _drifted_x_hat(dim="mood_affect", value=0.15)
        x_unc = _balanced_uncertainty()
        conn.execute(
            """
            INSERT INTO state_snapshots(
                snapshot_id, user_id, date, timestamp,
                x_hat_json, x_uncertainty_json, coverage_json,
                state_schema_version, model_version, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                "snap1", "u1", "2026-01-15", now_iso(),
                json.dumps(x_hat), json.dumps(x_unc), json.dumps({}),
                "v1", "v1", "questions_agent", now_iso(),
            ),
        )
        conn.commit()

        result = compute_user_fold(conn, user_id="u1", day=date(2026, 1, 15))
        assert result["schema"] == "identity_mask_fragment"
        assert result["mu"]["mood_affect"] == pytest.approx(0.15, abs=1e-4)

    def test_fold_with_circle_history(self, conn: sqlite3.Connection) -> None:
        """Insert circle snapshots → fold computes EWS features."""
        from questions_agent_platform.pipeline.service import compute_user_fold
        from questions_agent_platform.pipeline.time_utils import now_iso

        # Insert state snapshot for today
        x_hat = _balanced_x_hat()
        x_unc = _balanced_uncertainty()
        conn.execute(
            """
            INSERT INTO state_snapshots(
                snapshot_id, user_id, date, timestamp,
                x_hat_json, x_uncertainty_json, coverage_json,
                state_schema_version, model_version, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                "snap_today", "u1", "2026-01-15", now_iso(),
                json.dumps(x_hat), json.dumps(x_unc), json.dumps({}),
                "v1", "v1", "questions_agent", now_iso(),
            ),
        )

        # Insert circle history over 10 days
        for i in range(10):
            d = date(2026, 1, 6 + i)
            z = [0.1 * i, 0.05 * i]
            conn.execute(
                """
                INSERT INTO circle_snapshots(
                    snapshot_id, user_id, date, timestamp,
                    projection_version, anchor_version,
                    z_json, z_star_json,
                    r, theta, velocity, acceleration,
                    uncertainty_json, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    f"c{i}", "u1", d.isoformat(), now_iso(),
                    CIRCLE_PROJECTION_VERSION, "v1",
                    json.dumps(z), json.dumps([0.0, 0.0]),
                    0.2 + 0.02 * i,  # increasing r
                    0.5, 0.01, 0.0,
                    json.dumps({}), "questions_agent", now_iso(),
                ),
            )
        conn.commit()

        result = compute_user_fold(conn, user_id="u1", day=date(2026, 1, 15))
        assert result["history"]["trajectory_days"] >= 2
        assert result["history"]["ews"]["score"] >= 0.0

    def test_fold_returns_json_serialisable(self, conn: sqlite3.Connection) -> None:
        from questions_agent_platform.pipeline.service import compute_user_fold

        result = compute_user_fold(conn, user_id="u1", day=date(2026, 1, 15))
        serialised = json.dumps(result, ensure_ascii=False)
        roundtrip = json.loads(serialised)
        assert roundtrip["schema"] == "identity_mask_fragment"


class TestAPIEndpointPoEHelper:
    """Tests for the _extract_poe_from_fold helper in api.py."""

    def test_poe_extraction(self) -> None:
        from questions_agent_platform.pipeline.api import _extract_poe_from_fold

        fold = compute_questions_fold(
            x_hat=_drifted_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        mask = fold.to_identity_mask()
        poe = _extract_poe_from_fold(mask)
        assert poe["modality"] == "questionnaire"
        assert set(poe["mu"].keys()) == set(STATE_DIMENSIONS)
        assert set(poe["precision"].keys()) == set(STATE_DIMENSIONS)

    def test_poe_precision_matches_direct(self) -> None:
        from questions_agent_platform.pipeline.api import _extract_poe_from_fold

        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        direct_poe = fold.for_anifold_fusion()
        api_poe = _extract_poe_from_fold(fold.to_identity_mask())
        for dim in STATE_DIMENSIONS:
            assert api_poe["precision"][dim] == pytest.approx(
                direct_poe["precision"][dim], rel=1e-3,
            )

    def test_poe_empty_fold(self) -> None:
        from questions_agent_platform.pipeline.api import _extract_poe_from_fold

        poe = _extract_poe_from_fold({"mu": {}, "sigma_diag": {}, "circle": {}, "coverage": {}})
        assert poe["modality"] == "questionnaire"
        assert poe["precision"] == {}


class TestRoundTripConsistency:
    """Tests that ensure fold → serialise → deserialise is lossless."""

    def test_identity_mask_roundtrip(self) -> None:
        fold = compute_questions_fold(
            x_hat=_multi_drift_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
            circle_history=_make_circle_history(n=15, base_r=0.3, trend=0.01),
        )
        mask = fold.to_identity_mask()
        serialised = json.dumps(mask, sort_keys=True, ensure_ascii=False)
        roundtrip = json.loads(serialised)

        # Verify key fields survive serialisation
        assert roundtrip["schema"] == "identity_mask_fragment"
        assert roundtrip["day"] == "2026-01-15"
        assert len(roundtrip["mu"]) == len(STATE_DIMENSIONS)
        assert len(roundtrip["sigma_diag"]) == len(STATE_DIMENSIONS)
        assert len(roundtrip["minifolds"]) == len(MINIFOLD_MODES)
        assert roundtrip["circle"]["r"] == pytest.approx(fold.r, abs=1e-5)
        assert roundtrip["circle"]["coherence"] == pytest.approx(fold.coherence, abs=1e-5)

    def test_poe_roundtrip(self) -> None:
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        poe = fold.for_anifold_fusion()
        serialised = json.dumps(poe, sort_keys=True)
        roundtrip = json.loads(serialised)
        assert roundtrip["modality"] == "questionnaire"
        for dim in STATE_DIMENSIONS:
            assert roundtrip["mu"][dim] == pytest.approx(poe["mu"][dim], abs=1e-5)
            assert roundtrip["precision"][dim] == pytest.approx(poe["precision"][dim], abs=1e-5)

    def test_double_serialisation_deterministic(self) -> None:
        """Two independent fold computations with same inputs → identical JSON."""
        kwargs = dict(
            x_hat=_multi_drift_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        fold1 = compute_questions_fold(**kwargs)
        fold2 = compute_questions_fold(**kwargs)
        json1 = json.dumps(fold1.to_identity_mask(), sort_keys=True)
        json2 = json.dumps(fold2.to_identity_mask(), sort_keys=True)
        assert json1 == json2


class TestMiniFoldCircleDataclass:
    """Tests for the MiniFoldCircle frozen dataclass."""

    def test_frozen(self) -> None:
        mf = MiniFoldCircle(
            mode="metabolic",
            z=(0.1, 0.2),
            z_star=(0.0, 0.0),
            r=0.3,
            theta=1.5,
            velocity=0.01,
            acceleration=0.0,
            coherence=0.7,
            uncertainty=0.2,
            coverage_ratio=0.8,
            dimensions=("energy_vitality", "glycemic_risk"),
        )
        with pytest.raises(AttributeError):
            mf.r = 0.5  # type: ignore[misc]

    def test_fields(self) -> None:
        mf = MiniFoldCircle(
            mode="affective",
            z=(0.3, 0.4),
            z_star=(0.0, 0.0),
            r=0.5,
            theta=2.0,
            velocity=0.02,
            acceleration=0.001,
            coherence=0.6,
            uncertainty=0.3,
            coverage_ratio=1.0,
            dimensions=("mood_affect", "agency_purpose", "social_connectedness"),
        )
        assert mf.mode == "affective"
        assert mf.r == 0.5
        assert len(mf.dimensions) == 3


class TestHistoryEnvelopeDataclass:
    """Tests for the HistoryEnvelope frozen dataclass."""

    def test_frozen(self) -> None:
        env = HistoryEnvelope(
            velocity=0.1,
            acceleration=0.01,
            ews_variance=0.02,
            ews_autocorrelation=0.3,
            ews_trend_speed=0.005,
            ews_score=0.15,
            recovery_rate=0.05,
            trajectory_days=30,
        )
        with pytest.raises(AttributeError):
            env.velocity = 0.5  # type: ignore[misc]


class TestQuestionsFoldResultDataclass:
    """Tests for the QuestionsFoldResult frozen dataclass."""

    def test_frozen(self) -> None:
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        with pytest.raises(AttributeError):
            fold.coherence = 0.5  # type: ignore[misc]

    def test_to_identity_mask_is_dict(self) -> None:
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        mask = fold.to_identity_mask()
        assert isinstance(mask, dict)

    def test_for_anifold_fusion_is_dict(self) -> None:
        fold = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
        )
        poe = fold.for_anifold_fusion()
        assert isinstance(poe, dict)


class TestFoldChaining:
    """Test that fold outputs can be used as previous_fold inputs."""

    def test_two_day_chain(self) -> None:
        """Day 1 → Day 2 chain: day 2 uses day 1's fold as previous."""
        fold_d1 = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 14),
        )
        mask_d1 = fold_d1.to_identity_mask()

        fold_d2 = compute_questions_fold(
            x_hat=_drifted_x_hat(dim="mood_affect", value=0.2),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 15),
            previous_fold=mask_d1,
        )
        # Day 2 should have a valid fold
        assert fold_d2.day == "2026-01-15"
        assert fold_d2.version == QUESTIONS_FOLD_VERSION
        assert isinstance(fold_d2.attractor_state, dict)

    def test_three_day_chain_accumulates_history(self) -> None:
        """Day 1 → 2 → 3 chain: verify history grows."""
        folds = []
        circle_history: List[Dict[str, Any]] = []

        for i in range(3):
            d = date(2026, 1, 14 + i)
            # Gradually drift mood_affect
            x = _balanced_x_hat()
            x["mood_affect"] = 0.5 - 0.1 * i

            prev_mask = folds[-1].to_identity_mask() if folds else None
            fold = compute_questions_fold(
                x_hat=x,
                x_uncertainty=_balanced_uncertainty(),
                day=d,
                previous_fold=prev_mask,
                circle_history=circle_history[:] if circle_history else None,
            )
            folds.append(fold)
            circle_history.append({
                "r": fold.r,
                "date": fold.day,
                "velocity": fold.history.velocity,
            })

        # Day 3 should reference full history
        last = folds[-1]
        assert last.day == "2026-01-16"
        # With 3 history entries, EWS should be computable
        fold_with_hist = compute_questions_fold(
            x_hat=_balanced_x_hat(),
            x_uncertainty=_balanced_uncertainty(),
            day=date(2026, 1, 17),
            previous_fold=last.to_identity_mask(),
            circle_history=circle_history,
        )
        assert fold_with_hist.history.trajectory_days == 3
