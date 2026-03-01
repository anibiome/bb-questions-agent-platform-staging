"""Tests for MiniFold sub-circle computation and measurement package wiring.

v8 integration tests verifying:
- MiniFold per-mode projections produce valid circles
- Dominant decoherence mode detection works
- Progressive plan generation is correct
- DB migration v10 creates tables
- MiniFold snapshots survive insert/read cycle
- Measurement package tracking works
"""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import date, timedelta
from typing import Any, Dict

import pytest

from questions_agent_platform.pipeline.state_snapshots import (
    MINIFOLD_VERSION,
    compute_minifold_circles,
    dominant_decoherence_mode,
)
from questions_agent_platform.pipeline.measurement_packages import (
    ALL_PACKAGES,
    AXIS_PACKAGES,
    CORE_SCREENER,
    DEEP_PACKAGES,
    DecoherenceMode,
    MeasurementPackage,
    PackageTier,
    get_axis_package,
    get_core_screener,
    get_deep_package,
    get_progressive_plan,
    get_scale_modes,
    get_scales_for_mode,
    get_multiplex_value,
)


# ---------------------------------------------------------------------------
# MiniFold computation tests
# ---------------------------------------------------------------------------
class TestMiniFoldComputation:
    """Test per-mode sub-circle projection."""

    @staticmethod
    def _balanced_x_hat() -> Dict[str, float]:
        return {
            "energy_vitality": 0.5,
            "sleep_quality": 0.5,
            "gut_gi": 0.5,
            "glycemic_risk": 0.5,
            "cardiovascular_load": 0.5,
            "mood_affect": 0.5,
            "cognitive_control": 0.5,
            "agency_purpose": 0.5,
            "social_connectedness": 0.5,
        }

    @staticmethod
    def _balanced_uncertainty() -> Dict[str, float]:
        return {d: 0.3 for d in TestMiniFoldComputation._balanced_x_hat()}

    def test_balanced_state_produces_near_zero_radius(self) -> None:
        """All dimensions at 0.5 → z ≈ [0,0], z_star = [0,0], so r ≈ 0."""
        mf = compute_minifold_circles(
            x_hat=self._balanced_x_hat(),
            x_uncertainty=self._balanced_uncertainty(),
        )
        assert set(mf.keys()) == {"metabolic", "autonomic", "immune", "affective"}
        for mode, circle in mf.items():
            # z_star = [0,0] (population prior), z ≈ [0,0] for balanced state
            assert circle["r"] < 0.01, f"{mode} r should be near 0"

    def test_metabolic_drift_detected(self) -> None:
        """High glycemic risk → metabolic mode shows decoherence."""
        x_hat = self._balanced_x_hat()
        x_hat["glycemic_risk"] = 0.9  # high risk
        x_hat["cardiovascular_load"] = 0.8
        mf = compute_minifold_circles(
            x_hat=x_hat,
            x_uncertainty=self._balanced_uncertainty(),
        )
        assert mf["metabolic"]["r"] > 0.1
        # Metabolic should be dominant
        assert dominant_decoherence_mode(mf) == "metabolic"

    def test_affective_drift_detected(self) -> None:
        """Low mood → affective mode shows decoherence."""
        x_hat = self._balanced_x_hat()
        x_hat["mood_affect"] = 0.1  # very low
        mf = compute_minifold_circles(
            x_hat=x_hat,
            x_uncertainty=self._balanced_uncertainty(),
        )
        assert mf["affective"]["r"] > 0.1
        assert dominant_decoherence_mode(mf) == "affective"

    def test_immune_drift_detected(self) -> None:
        """GI issues → immune mode shows decoherence."""
        x_hat = self._balanced_x_hat()
        x_hat["gut_gi"] = 0.15  # bad GI
        mf = compute_minifold_circles(
            x_hat=x_hat,
            x_uncertainty=self._balanced_uncertainty(),
        )
        assert mf["immune"]["r"] > 0.1

    def test_each_mode_has_required_fields(self) -> None:
        mf = compute_minifold_circles(
            x_hat=self._balanced_x_hat(),
            x_uncertainty=self._balanced_uncertainty(),
            day=date(2026, 2, 27),
        )
        required_fields = {
            "mode", "z", "z_star", "r", "theta",
            "velocity", "acceleration", "coherence",
            "uncertainty", "coverage_ratio", "dimensions", "date", "version",
        }
        for mode, circle in mf.items():
            missing = required_fields - set(circle.keys())
            assert not missing, f"{mode} missing fields: {missing}"
            assert circle["version"] == MINIFOLD_VERSION

    def test_velocity_with_previous(self) -> None:
        """Velocity computed from previous state."""
        prev = {
            "metabolic": {
                "z": [0.0, 0.0],
                "z_star": [0.0, 0.0],
                "r": 0.0, "theta": 0.0, "velocity": 0.0,
                "acceleration": 0.0, "date": "2026-02-25",
            }
        }
        x_hat = self._balanced_x_hat()
        x_hat["glycemic_risk"] = 0.9  # drift
        mf = compute_minifold_circles(
            x_hat=x_hat,
            x_uncertainty=self._balanced_uncertainty(),
            previous_minifolds=prev,
            day=date(2026, 2, 27),
        )
        # Should have non-zero velocity for metabolic
        assert mf["metabolic"]["velocity"] > 0.0

    def test_coherence_inversely_related_to_radius(self) -> None:
        x_hat = self._balanced_x_hat()
        x_hat["mood_affect"] = 0.1
        mf = compute_minifold_circles(
            x_hat=x_hat,
            x_uncertainty=self._balanced_uncertainty(),
        )
        affective = mf["affective"]
        assert affective["coherence"] == pytest.approx(
            max(0.0, 1.0 - affective["r"]), abs=0.001
        )

    def test_empty_previous_minifolds(self) -> None:
        """No previous → z_star = [0,0] (population prior), velocity == 0."""
        mf = compute_minifold_circles(
            x_hat=self._balanced_x_hat(),
            x_uncertainty=self._balanced_uncertainty(),
        )
        for mode, circle in mf.items():
            assert circle["z_star"] == [0.0, 0.0]
            assert circle["velocity"] == 0.0


class TestDominantDecoherenceMode:
    def test_empty_returns_none(self) -> None:
        assert dominant_decoherence_mode({}) is None

    def test_returns_highest_radius(self) -> None:
        mf = {
            "metabolic": {"r": 0.1},
            "autonomic": {"r": 0.3},
            "immune": {"r": 0.05},
            "affective": {"r": 0.15},
        }
        assert dominant_decoherence_mode(mf) == "autonomic"


# ---------------------------------------------------------------------------
# Measurement package structure tests
# ---------------------------------------------------------------------------
class TestMeasurementPackageStructure:
    def test_core_screener_tier(self) -> None:
        cs = get_core_screener()
        assert cs.tier == PackageTier.CORE
        assert cs.mode is None
        assert cs.estimated_items == 27

    def test_all_four_axis_packages(self) -> None:
        for mode in DecoherenceMode:
            pkg = get_axis_package(mode)
            assert pkg.tier == PackageTier.AXIS
            assert pkg.mode == mode
            assert pkg.prerequisite_package_id == "pkg_core_screener"

    def test_all_four_deep_packages(self) -> None:
        for mode in DecoherenceMode:
            pkg = get_deep_package(mode)
            assert pkg.tier == PackageTier.DEEP_DIVE
            assert pkg.mode == mode

    def test_total_package_count(self) -> None:
        assert len(ALL_PACKAGES) == 10  # 1 core + 4 axis + 4 deep + 1 repeat

    def test_scale_mode_map_coverage(self) -> None:
        """Every mode has at least 3 scales."""
        for mode in DecoherenceMode:
            scales = get_scales_for_mode(mode)
            assert len(scales) >= 3, f"{mode} has only {len(scales)} scales"


# ---------------------------------------------------------------------------
# Progressive plan tests
# ---------------------------------------------------------------------------
class TestProgressivePlanIntegration:
    def test_fresh_user_sequence(self) -> None:
        plan = get_progressive_plan(user_id="u1")
        assert plan.recommended_sequence[0] == "pkg_core_screener"
        # Should include all 4 axis packages
        axis_ids = {f"pkg_axis_{m.value}" for m in DecoherenceMode}
        sequence_set = set(plan.recommended_sequence)
        assert axis_ids.issubset(sequence_set)

    def test_metabolic_drift_prioritizes_metabolic_axis(self) -> None:
        plan = get_progressive_plan(
            user_id="u1",
            drift_mode=DecoherenceMode.METABOLIC,
        )
        # First axis should be metabolic (after core screener)
        seq = plan.recommended_sequence
        assert seq[0] == "pkg_core_screener"
        assert seq[1] == "pkg_axis_metabolic"
        # Deep dive should follow
        assert "pkg_deep_metabolic" in seq

    def test_affective_drift_reorders_axes(self) -> None:
        plan = get_progressive_plan(
            user_id="u1",
            drift_mode=DecoherenceMode.AFFECTIVE,
        )
        seq = plan.recommended_sequence
        axis_start = next(i for i, s in enumerate(seq) if s.startswith("pkg_axis_"))
        assert seq[axis_start] == "pkg_axis_affective"


# ---------------------------------------------------------------------------
# DB migration v10 tests
# ---------------------------------------------------------------------------
class TestDBMigrationV10:
    @pytest.fixture()
    def conn(self, tmp_path: Any) -> sqlite3.Connection:
        from questions_agent_platform.pipeline.db import init_db
        db_path = str(tmp_path / "test_v10.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def test_minifold_snapshots_table_exists(self, conn: sqlite3.Connection) -> None:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()}
        assert "minifold_snapshots" in tables

    def test_user_measurement_packages_table_exists(self, conn: sqlite3.Connection) -> None:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()}
        assert "user_measurement_packages" in tables

    def test_minifold_insert_and_read(self, conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO users(user_id, created_at) VALUES ('u1', '2026-01-01');")
        conn.execute(
            """
            INSERT INTO minifold_snapshots(
              snapshot_id, user_id, date, circle_snapshot_id, mode, version,
              z_json, z_star_json, r, theta, velocity, acceleration,
              coherence, uncertainty, coverage_ratio, dimensions_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                str(uuid.uuid4()), "u1", "2026-02-27", str(uuid.uuid4()),
                "metabolic", MINIFOLD_VERSION,
                "[0.1, 0.2]", "[0.0, 0.0]",
                0.22, 1.1, 0.05, 0.01,
                0.78, 0.3, 0.67, '["energy_vitality","glycemic_risk"]',
                "2026-02-27T00:00:00Z",
            ),
        )
        row = conn.execute(
            "SELECT * FROM minifold_snapshots WHERE user_id='u1' AND mode='metabolic';"
        ).fetchone()
        assert row is not None
        assert float(row["r"]) == pytest.approx(0.22)
        assert row["mode"] == "metabolic"

    def test_measurement_package_tracking(self, conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO users(user_id, created_at) VALUES ('u1', '2026-01-01');")
        conn.execute(
            """
            INSERT INTO user_measurement_packages(
              package_id, user_id, measurement_package_id, status, created_at
            ) VALUES (?, ?, ?, ?, ?);
            """,
            (str(uuid.uuid4()), "u1", "pkg_core_screener", "completed", "2026-02-27T00:00:00Z"),
        )
        row = conn.execute(
            "SELECT * FROM user_measurement_packages WHERE user_id='u1';"
        ).fetchone()
        assert row is not None
        assert row["measurement_package_id"] == "pkg_core_screener"
        assert row["status"] == "completed"


# ---------------------------------------------------------------------------
# Multiplex value tests
# ---------------------------------------------------------------------------
class TestMultiplexValue:
    def test_cross_mode_item_scores_high(self) -> None:
        """An item appearing in scales that span 2+ modes has high value."""
        scale_mode_map = {
            "scale_pss": (DecoherenceMode.AFFECTIVE, DecoherenceMode.AUTONOMIC),
            "scale_gad_7": (DecoherenceMode.AFFECTIVE,),
        }
        item_to_scales = {
            "pss_item_1": ["scale_pss"],
            "gad_item_1": ["scale_gad_7"],
        }
        # PSS item spans 2 modes
        assert get_multiplex_value("pss_item_1", scale_mode_map, item_to_scales) == 2.0
        # GAD item spans 1 mode
        assert get_multiplex_value("gad_item_1", scale_mode_map, item_to_scales) == 1.0
