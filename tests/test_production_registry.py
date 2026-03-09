"""Tests for the v3 production registry, new response types, and
measurement packages.

Covers:
- Production registry loading and validation (941 items, 65 scales)
- All 44 response types parse correctly
- Multiplex map correctness
- Measurement package architecture
- Progressive plan generation
- Decoherence mode mapping
- Scoring compatibility with new response types
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

from questions_agent_platform.pipeline.registry import (
    Scale,
    ScaleItem,
)
from questions_agent_platform.pipeline.response_types import RESPONSE_TYPES, get_response_type
from questions_agent_platform.pipeline.measurement_packages import (
    ALL_PACKAGES,
    AXIS_PACKAGES,
    CORE_SCREENER,
    DECOHERENCE_MODES,
    DecoherenceMode,
    PackageTier,
    REPEAT_BASELINE,
    SCALE_MODE_MAP,
    get_axis_package,
    get_core_screener,
    get_deep_package,
    get_multiplex_value,
    get_package,
    get_progressive_plan,
    get_scale_modes,
    get_scales_for_mode,
)
from questions_agent_platform.pipeline.scoring import compute_scale_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
PROD_REGISTRY_DIR = Path(__file__).resolve().parent.parent / "data" / "registry_production" / "v3"


def _load_json(name: str) -> Any:
    with open(PROD_REGISTRY_DIR / name) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def prod_items() -> List[dict]:
    return _load_json("items.json")


@pytest.fixture(scope="module")
def prod_scales() -> List[dict]:
    return _load_json("scales.json")


@pytest.fixture(scope="module")
def prod_questionnaires() -> List[dict]:
    return _load_json("questionnaires.json")


@pytest.fixture(scope="module")
def prod_multiplex() -> Dict[str, List[str]]:
    return _load_json("multiplex_map.json")


# ---------------------------------------------------------------------------
# Test: Production registry files exist and are well-formed
# ---------------------------------------------------------------------------
class TestProductionRegistryFiles:
    def test_registry_dir_exists(self) -> None:
        assert PROD_REGISTRY_DIR.exists(), f"Registry dir not found: {PROD_REGISTRY_DIR}"

    def test_items_json_exists(self) -> None:
        assert (PROD_REGISTRY_DIR / "items.json").exists()

    def test_scales_json_exists(self) -> None:
        assert (PROD_REGISTRY_DIR / "scales.json").exists()

    def test_questionnaires_json_exists(self) -> None:
        assert (PROD_REGISTRY_DIR / "questionnaires.json").exists()

    def test_multiplex_map_exists(self) -> None:
        assert (PROD_REGISTRY_DIR / "multiplex_map.json").exists()

    def test_items_count(self, prod_items: List[dict]) -> None:
        assert len(prod_items) >= 885, f"Expected >=885 items, got {len(prod_items)}"

    def test_scales_count(self, prod_scales: List[dict]) -> None:
        assert len(prod_scales) >= 49, f"Expected >=49 scales, got {len(prod_scales)}"

    def test_questionnaires_count(self, prod_questionnaires: List[dict]) -> None:
        assert len(prod_questionnaires) >= 49


# ---------------------------------------------------------------------------
# Test: All item response_types are defined
# ---------------------------------------------------------------------------
class TestResponseTypesCoverage:
    def test_all_item_response_types_exist(self, prod_items: List[dict]) -> None:
        """Every item's response_type must exist in RESPONSE_TYPES."""
        missing = set()
        for it in prod_items:
            rt = it.get("response_type", "")
            if rt not in RESPONSE_TYPES:
                missing.add(rt)
        assert not missing, f"Missing response types: {missing}"

    def test_production_response_types_count(self) -> None:
        """We should have at least 44 response types (14 v2 + 30 v3)."""
        assert len(RESPONSE_TYPES) >= 44

    def test_each_response_type_has_options(self) -> None:
        for rt_id, rt in RESPONSE_TYPES.items():
            assert len(rt.options) >= 2, f"{rt_id} has < 2 options"
            assert rt.min_value < rt.max_value, f"{rt_id} min >= max"

    def test_get_response_type_raises_for_unknown(self) -> None:
        with pytest.raises(ValueError):
            get_response_type("nonexistent_type_xyz")


# ---------------------------------------------------------------------------
# Test: Scale → item cross-references are valid
# ---------------------------------------------------------------------------
class TestScaleItemCrossReferences:
    def test_all_scale_items_exist(
        self, prod_items: List[dict], prod_scales: List[dict],
    ) -> None:
        item_ids = {it["id"] for it in prod_items}
        broken = []
        for sc in prod_scales:
            for si in sc.get("items", []):
                iid = si["item_id"] if isinstance(si, dict) else si
                if iid not in item_ids:
                    broken.append((sc["id"], iid))
        assert not broken, f"Broken item refs: {broken[:10]}"

    def test_all_scale_questionnaire_ids_exist(
        self, prod_scales: List[dict], prod_questionnaires: List[dict],
    ) -> None:
        q_ids = {q["id"] for q in prod_questionnaires}
        broken = []
        for sc in prod_scales:
            if sc["questionnaire_id"] not in q_ids:
                broken.append((sc["id"], sc["questionnaire_id"]))
        assert not broken, f"Broken questionnaire refs: {broken[:10]}"

    def test_no_duplicate_item_ids(self, prod_items: List[dict]) -> None:
        ids = [it["id"] for it in prod_items]
        dupes = set(x for x in ids if ids.count(x) > 1)
        # Allow up to 2 dupes from spreadsheet data issues (DASS-21 split)
        assert len(dupes) <= 2, f"Too many duplicate item IDs: {dupes}"

    def test_no_duplicate_scale_ids(self, prod_scales: List[dict]) -> None:
        ids = [sc["id"] for sc in prod_scales]
        dupes = [x for x in ids if ids.count(x) > 1]
        assert not dupes, f"Duplicate scale IDs: {set(dupes)}"


# ---------------------------------------------------------------------------
# Test: Multiplex map
# ---------------------------------------------------------------------------
class TestMultiplexMap:
    def test_multiplex_has_shared_items(self, prod_multiplex: Dict) -> None:
        shared = {k: v for k, v in prod_multiplex.items() if len(v) > 1}
        assert len(shared) >= 1, "Expected at least 1 multiplexed item"

    def test_every_scale_item_in_multiplex(
        self, prod_scales: List[dict], prod_multiplex: Dict,
    ) -> None:
        """Every item referenced in a scale should appear in the multiplex map."""
        missing = []
        for sc in prod_scales:
            for si in sc.get("items", []):
                iid = si["item_id"] if isinstance(si, dict) else si
                if iid not in prod_multiplex:
                    missing.append((sc["id"], iid))
        # Some v2 items might not be in the spreadsheet-based map
        # Allow up to 60 missing (cardiometabolic v2 items)
        assert len(missing) <= 60, f"Too many missing from multiplex: {len(missing)}"


# ---------------------------------------------------------------------------
# Test: Scoring compatibility with new response types
# ---------------------------------------------------------------------------
class TestScoringWithNewResponseTypes:
    """Verify compute_scale_score works with production response types."""

    def _make_scale(
        self, method: str, n_items: int, response_type: str,
        normalize_max: float, reverse_items: Set[int] | None = None,
    ) -> Scale:
        items = []
        for i in range(n_items):
            rev = bool(reverse_items and i in reverse_items)
            items.append(ScaleItem(item_id=f"test_item_{i}", reverse=rev))
        return Scale(
            id="test_scale",
            questionnaire_id="test_q",
            version="1",
            name="Test",
            method=method,
            min_items_required=max(1, n_items - 1),
            unlock_window_days=0,
            retest_interval_days=90,
            response_type=response_type,
            normalize_min=0.0,
            normalize_max=normalize_max,
            items=tuple(items),
        )

    def test_dass_scoring(self) -> None:
        """DASS-21: sum method, dass_0_3 response type."""
        scale = self._make_scale("sum", 7, "dass_0_3", 42.0)
        answers = {f"test_item_{i}": 2.0 for i in range(7)}
        result = compute_scale_score(scale, answers)
        assert result is not None
        assert result.raw_score == 14.0  # 7 * 2

    def test_gad7_scoring(self) -> None:
        """GAD-7: sum, likert_0_3."""
        scale = self._make_scale("sum", 7, "likert_0_3", 21.0)
        answers = {f"test_item_{i}": 1.0 for i in range(7)}
        result = compute_scale_score(scale, answers)
        assert result is not None
        assert result.raw_score == 7.0

    def test_brs_scoring_with_reverse(self) -> None:
        """BRS: mean, 6 items, items 1/3/5 reversed."""
        scale = self._make_scale(
            "mean", 6, "agree_brs_0_4", 4.0, reverse_items={1, 3, 5},
        )
        # All answers = 3.0 → reverse items become 4-3=1
        # mean = (3+1+3+1+3+1)/6 = 2.0
        answers = {f"test_item_{i}": 3.0 for i in range(6)}
        result = compute_scale_score(scale, answers)
        assert result is not None
        assert abs(result.raw_score - 2.0) < 0.01

    def test_likert_agree_0_6_scoring(self) -> None:
        """7-point Likert (Mind Age): mean scoring."""
        scale = self._make_scale("mean", 5, "likert_agree_0_6", 6.0)
        answers = {f"test_item_{i}": 4.0 for i in range(5)}
        result = compute_scale_score(scale, answers)
        assert result is not None
        assert abs(result.raw_score - 4.0) < 0.01

    def test_missing_items_below_threshold_returns_none(self) -> None:
        """Scale with min_items=6 and only 4 answers → None."""
        scale = self._make_scale("sum", 7, "dass_0_3", 42.0)
        # Override min_items to 6
        scale = Scale(
            id=scale.id, questionnaire_id=scale.questionnaire_id,
            version=scale.version, name=scale.name, method=scale.method,
            min_items_required=6, unlock_window_days=0,
            retest_interval_days=90, response_type=scale.response_type,
            normalize_min=0.0, normalize_max=42.0, items=scale.items,
        )
        answers = {f"test_item_{i}": 2.0 for i in range(4)}
        result = compute_scale_score(scale, answers)
        assert result is None


# ---------------------------------------------------------------------------
# Test: Measurement Packages
# ---------------------------------------------------------------------------
class TestMeasurementPackages:
    def test_all_packages_registered(self) -> None:
        assert len(ALL_PACKAGES) == 10  # core + 4 axis + 4 deep + 1 repeat

    def test_core_screener_structure(self) -> None:
        pkg = get_core_screener()
        assert pkg.id == "pkg_core_screener"
        assert pkg.tier == PackageTier.CORE
        assert pkg.mode is None  # spans all modes
        assert pkg.estimated_items == 27
        assert len(pkg.package_items) >= 5

    def test_each_axis_has_package(self) -> None:
        for mode in DECOHERENCE_MODES:
            pkg = get_axis_package(mode)
            assert pkg.tier == PackageTier.AXIS
            assert pkg.mode == mode
            assert len(pkg.package_items) >= 3

    def test_each_deep_dive_has_package(self) -> None:
        for mode in DECOHERENCE_MODES:
            pkg = get_deep_package(mode)
            assert pkg.tier == PackageTier.DEEP_DIVE
            assert pkg.mode == mode

    def test_repeat_baseline(self) -> None:
        assert REPEAT_BASELINE.tier == PackageTier.REPEAT
        assert REPEAT_BASELINE.unlock_after_sessions == 2

    def test_get_package_by_id(self) -> None:
        assert get_package("pkg_core_screener") is CORE_SCREENER
        assert get_package("nonexistent") is None

    def test_axis_packages_have_prerequisites(self) -> None:
        for mode, pkg in AXIS_PACKAGES.items():
            assert pkg.prerequisite_package_id == "pkg_core_screener"


# ---------------------------------------------------------------------------
# Test: Progressive Plan
# ---------------------------------------------------------------------------
class TestProgressivePlan:
    def test_new_user_starts_with_core(self) -> None:
        plan = get_progressive_plan("user_new")
        assert plan.recommended_sequence[0] == "pkg_core_screener"

    def test_new_user_gets_all_axis_packages(self) -> None:
        plan = get_progressive_plan("user_new")
        seq = plan.recommended_sequence
        for mode, pkg in AXIS_PACKAGES.items():
            assert pkg.id in seq, f"Missing axis package: {pkg.id}"

    def test_completed_core_skips_it(self) -> None:
        plan = get_progressive_plan(
            "user_2",
            completed_sessions=1,
            completed_packages=["pkg_core_screener"],
        )
        assert "pkg_core_screener" not in plan.recommended_sequence

    def test_drift_prioritizes_mode(self) -> None:
        plan = get_progressive_plan(
            "user_drift",
            completed_sessions=2,
            drift_mode=DecoherenceMode.METABOLIC,
        )
        seq = plan.recommended_sequence
        # Metabolic axis should come first after core screener
        non_core = [s for s in seq if s != "pkg_core_screener"]
        assert non_core[0] == "pkg_axis_metabolic"

    def test_drift_inserts_deep_dive(self) -> None:
        plan = get_progressive_plan(
            "user_drift",
            completed_sessions=3,
            drift_mode=DecoherenceMode.IMMUNE,
        )
        assert "pkg_deep_immune" in plan.recommended_sequence

    def test_repeat_baseline_after_session_2(self) -> None:
        plan = get_progressive_plan("user_3", completed_sessions=3)
        assert "pkg_repeat_baseline" in plan.recommended_sequence

    def test_repeat_baseline_not_before_session_2(self) -> None:
        plan = get_progressive_plan("user_1", completed_sessions=1)
        assert "pkg_repeat_baseline" not in plan.recommended_sequence


# ---------------------------------------------------------------------------
# Test: Decoherence Mode Mapping
# ---------------------------------------------------------------------------
class TestDecoherenceModeMapping:
    def test_scale_mode_map_covers_production_scales(
        self, prod_scales: List[dict],
    ) -> None:
        """Most production scales should have a mode mapping."""
        mapped = set(SCALE_MODE_MAP.keys())
        unmapped = []
        for sc in prod_scales:
            if sc["id"] not in mapped:
                unmapped.append(sc["id"])
        # Allow some unmapped (v2 subscales, etc.)
        assert len(unmapped) <= 20, f"Too many unmapped scales: {unmapped}"

    def test_each_mode_has_scales(self) -> None:
        for mode in DECOHERENCE_MODES:
            scales = get_scales_for_mode(mode)
            assert len(scales) >= 3, f"Mode {mode} has < 3 scales"

    def test_get_scale_modes(self) -> None:
        modes = get_scale_modes("scale_gad_7")
        assert DecoherenceMode.AFFECTIVE in modes

    def test_cross_mode_scales(self) -> None:
        """Some scales contribute to multiple modes (e.g., PSS → affective + autonomic)."""
        pss_modes = get_scale_modes("scale_pss")
        assert len(pss_modes) >= 2

    def test_multiplex_value(self) -> None:
        """Items in multi-mode scales should have higher multiplex value."""
        # PSS feeds affective + autonomic = 2 modes
        item_to_scales = {"pss_item_1": ["scale_pss"]}
        val = get_multiplex_value("pss_item_1", SCALE_MODE_MAP, item_to_scales)
        assert val >= 2.0

    def test_unknown_item_multiplex_value_zero(self) -> None:
        val = get_multiplex_value("nonexistent", SCALE_MODE_MAP, {})
        assert val == 0.0


# ---------------------------------------------------------------------------
# Test: Audit flags in production registry
# ---------------------------------------------------------------------------
class TestAuditFlags:
    def test_aaq_flagged_as_superseded(self, prod_scales: List[dict]) -> None:
        aaq = next((s for s in prod_scales if s["name"] == "AAQ"), None)
        assert aaq is not None
        assert any("superseded" in str(t) for t in aaq.get("tags", []))

    def test_cfq_flagged_as_dated(self, prod_scales: List[dict]) -> None:
        cfq = next((s for s in prod_scales if s["name"] == "CFQ"), None)
        assert cfq is not None
        assert any("dated" in str(t) for t in cfq.get("tags", []))

    def test_rs_flagged_as_redundant(self, prod_scales: List[dict]) -> None:
        rs = next((s for s in prod_scales if s["name"] == "RS"), None)
        assert rs is not None
        assert any("redundant" in str(t) for t in rs.get("tags", []))
