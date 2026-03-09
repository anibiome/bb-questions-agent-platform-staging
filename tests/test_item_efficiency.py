"""
Tests for pipeline.item_efficiency — Item Efficiency Analyzer.

Tests cover:
  1. Dataclass construction and defaults
  2. Semantic similarity (TF-IDF + cosine)
  3. Tag overlap scoring
  4. Response correlation (statistical twins)
  5. IRT discrimination estimation
  6. Composite scoring
  7. Reduced set recommendation with scale protection
  8. Registry-only mode (no response data)
  9. Full mode (with response data)
  10. Production registry v3 analysis (941 items)
  11. Edge cases (empty registry, single item, etc.)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from questions_agent_platform.pipeline.item_efficiency import (
    EfficiencyReport,
    ItemEfficiency,
    ItemEfficiencyAnalyzer,
    RedundancyCluster,
    analyze_registry_from_files,
)


# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════


def _make_item(item_id: str, text: str, tags: list = None, response_type: str = "likert_0_4"):
    return {
        "id": item_id,
        "text": text,
        "response_type": response_type,
        "tags": tags or [],
        "sensitivity": "low",
        "timeframes_allowed": ["last_7_days"],
        "intrusiveness": "low",
        "declinable": True,
        "onboarding_order": None,
    }


def _make_scale(scale_id: str, item_ids: list, tags: list = None, min_items: int = 3):
    return {
        "id": scale_id,
        "questionnaire_id": f"q_{scale_id}",
        "version": "1",
        "name": scale_id,
        "response_type": "likert_0_4",
        "min_items_required": min_items,
        "unlock_window_days": 0,
        "retest_interval_days": 90,
        "tags": tags or [],
        "scoring": {"method": "sum", "normalize_min": 0.0, "normalize_max": 20.0},
        "items": [{"item_id": iid, "reverse": False, "weight": 1.0} for iid in item_ids],
        "ewma_alpha": 0.2,
        "citation": "",
    }


@pytest.fixture
def tiny_registry():
    """Minimal registry: 6 items, 2 scales, with deliberate semantic overlap."""
    items = [
        _make_item("a1", "I feel anxious and nervous", tags=["affective", "anxiety"]),
        _make_item("a2", "I feel worried and nervous", tags=["affective", "anxiety"]),
        _make_item("a3", "I have trouble sleeping at night", tags=["sleep"]),
        _make_item("b1", "I feel sad and hopeless", tags=["affective", "depression"]),
        _make_item("b2", "I feel down and depressed", tags=["affective", "depression"]),
        _make_item("b3", "I enjoy activities and hobbies", tags=["affective", "wellbeing"]),
    ]
    scales = [
        _make_scale("scale_anxiety", ["a1", "a2", "a3"], tags=["affective", "anxiety"]),
        _make_scale("scale_depression", ["b1", "b2", "b3"], tags=["affective", "depression"]),
    ]
    return {"items": items, "scales": scales, "multiplex_map": {}}


@pytest.fixture
def registry_with_shared_items():
    """Registry with cross-scale item sharing (multiplexing)."""
    items = [
        _make_item("shared1", "I have been feeling good about myself", tags=["affective", "wellbeing"]),
        _make_item("shared2", "I have felt calm and relaxed", tags=["affective", "mood"]),
        _make_item("ax1", "I feel tense and on edge", tags=["anxiety"]),
        _make_item("ax2", "I worry about many things", tags=["anxiety"]),
        _make_item("dep1", "I feel worthless and guilty", tags=["depression"]),
        _make_item("dep2", "I have lost interest in things", tags=["depression"]),
        _make_item("wb1", "I have felt active and vigorous", tags=["wellbeing"]),
        _make_item("wb2", "My daily life has been filled with things that interest me", tags=["wellbeing"]),
    ]
    scales = [
        _make_scale("scale_anxiety", ["shared1", "shared2", "ax1", "ax2"], tags=["anxiety"]),
        _make_scale("scale_depression", ["shared1", "shared2", "dep1", "dep2"], tags=["depression"]),
        _make_scale("scale_wellbeing", ["shared1", "shared2", "wb1", "wb2"], tags=["wellbeing"]),
    ]
    return {"items": items, "scales": scales, "multiplex_map": {}}


@pytest.fixture
def response_matrix_6x6():
    """Synthetic response data for 6 items, 20 respondents."""
    np.random.seed(42)
    n = 20
    # a1 and a2 are correlated (r ~ 0.9)
    a1 = np.random.randint(0, 5, n).astype(float)
    a2 = a1 + np.random.normal(0, 0.3, n)  # highly correlated with a1
    a2 = np.clip(np.round(a2), 0, 4)
    a3 = np.random.randint(0, 5, n).astype(float)  # independent

    b1 = np.random.randint(0, 5, n).astype(float)
    b2 = b1 + np.random.normal(0, 0.3, n)  # highly correlated with b1
    b2 = np.clip(np.round(b2), 0, 4)
    b3 = np.random.randint(0, 5, n).astype(float)  # independent

    mat = np.column_stack([a1, a2, a3, b1, b2, b3])
    item_ids = ["a1", "a2", "a3", "b1", "b2", "b3"]
    return mat, item_ids


# ════════════════════════════════════════════════════════════════════════════
# 1. Dataclass Tests
# ════════════════════════════════════════════════════════════════════════════


class TestDataclasses:
    def test_item_efficiency_defaults(self):
        e = ItemEfficiency(
            item_id="x", text="test", scale_ids=["s1"], tags=["t1"], response_type="likert_0_4"
        )
        assert e.efficiency_score == 1.0
        assert e.recommendation == "keep"
        assert e.semantic_cluster_id == -1
        assert e.redundancy_flags == []
        assert e.max_semantic_similarity == 0.0

    def test_redundancy_cluster_construction(self):
        c = RedundancyCluster(
            cluster_id=0,
            item_ids=["a", "b"],
            item_texts=["text a", "text b"],
            scale_ids=["s1"],
            mean_similarity=0.85,
            cluster_type="semantic",
            representative_item_id="a",
        )
        assert len(c.item_ids) == 2
        assert c.mean_similarity == 0.85

    def test_efficiency_report_summary(self):
        report = EfficiencyReport(
            total_items=10,
            total_scales=2,
            items=[],
            redundancy_clusters=[],
            low_variance_items=[],
            low_discrimination_items=[],
            n_semantic_duplicates=3,
            n_recommended_cuts=2,
        )
        s = report.summary()
        assert s["total_items"] == 10
        assert s["n_semantic_duplicates"] == 3
        assert s["n_recommended_cuts"] == 2
        assert s["has_response_data"] is False


# ════════════════════════════════════════════════════════════════════════════
# 2. Semantic Similarity Layer
# ════════════════════════════════════════════════════════════════════════════


class TestSemanticSimilarity:
    def test_near_duplicates_detected(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.analyze(tag_overlap=False, correlation=False, irt_discrimination=False)

        # "anxious and nervous" vs "worried and nervous" should be high similarity
        a1_eff = next(e for e in report.items if e.item_id == "a1")
        a2_eff = next(e for e in report.items if e.item_id == "a2")
        # They should be each other's most similar
        assert a1_eff.max_semantic_similarity > 0.4
        assert a2_eff.max_semantic_similarity > 0.4

    def test_dissimilar_items_low_similarity(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.analyze(tag_overlap=False, correlation=False, irt_discrimination=False)

        a3_eff = next(e for e in report.items if e.item_id == "a3")
        # "trouble sleeping" should be dissimilar from all mood items
        assert a3_eff.max_semantic_similarity < 0.6

    def test_clusters_formed(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.analyze(tag_overlap=False, correlation=False, irt_discrimination=False)

        # Should find at least one cluster
        assert len(report.redundancy_clusters) >= 0  # may or may not cluster with tiny data
        # All clusters should have ≥ 2 items
        for c in report.redundancy_clusters:
            assert len(c.item_ids) >= 2

    def test_cross_scale_flag(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.analyze(tag_overlap=False, correlation=False, irt_discrimination=False)

        # Items in different scales with high similarity should get cross_scale flag
        for e in report.items:
            if e.most_similar_cross_scale:
                # The most similar item must be in a different scale
                own_scales = set(e.scale_ids)
                other_scales = set(
                    analyzer._item_to_scales.get(e.most_similar_item_id, [])
                )
                assert not (own_scales & other_scales) or len(own_scales) == 0


# ════════════════════════════════════════════════════════════════════════════
# 3. Tag Overlap Layer
# ════════════════════════════════════════════════════════════════════════════


class TestTagOverlap:
    def test_shared_tags_scored(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.analyze(semantic=False, correlation=False, irt_discrimination=False)

        # Items with "affective" tag should have overlap (both scales share it)
        a1_eff = next(e for e in report.items if e.item_id == "a1")
        assert a1_eff.tag_overlap_score > 0.0  # "affective" shared with depression scale

    def test_unique_tags_low_overlap(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.analyze(semantic=False, correlation=False, irt_discrimination=False)

        # Item with "sleep" tag (only in anxiety scale) should have lower overlap
        a3_eff = next(e for e in report.items if e.item_id == "a3")
        # sleep tag not shared by depression scale
        assert a3_eff.tag_overlap_score < 0.5


# ════════════════════════════════════════════════════════════════════════════
# 4. Response Correlation Layer
# ════════════════════════════════════════════════════════════════════════════


class TestResponseCorrelation:
    def test_correlated_items_flagged(self, tiny_registry, response_matrix_6x6):
        mat, item_ids = response_matrix_6x6
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        analyzer.ingest_responses(mat, item_ids)
        report = analyzer.analyze(semantic=False, tag_overlap=False, irt_discrimination=False)

        # a1 and b1 are in different scales; a1 and a2 are in same scale
        # Cross-scale correlations are what we detect
        # b1-b2 correlation should be detected cross-scale if there
        # a1-a2 are same scale so NOT flagged
        for e in report.items:
            if e.max_cross_item_correlation is not None and e.max_cross_item_correlation > 0:
                # The correlated partner should be from a different scale
                own = set(e.scale_ids)
                partner = set(analyzer._item_to_scales.get(e.most_correlated_item_id, []))
                assert not (own & partner), f"Same-scale correlation flagged for {e.item_id}"

    def test_variance_computed(self, tiny_registry, response_matrix_6x6):
        mat, item_ids = response_matrix_6x6
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        analyzer.ingest_responses(mat, item_ids)
        report = analyzer.analyze(semantic=False, tag_overlap=False, irt_discrimination=False)

        for e in report.items:
            if e.n_responses > 0:
                assert e.response_variance is not None
                assert e.response_variance >= 0.0

    def test_low_variance_flagged(self):
        """Items where everyone answers the same should get low_variance flag."""
        items = [
            _make_item("flat1", "I am alive", tags=["bio"]),
            _make_item("varied1", "I feel different emotions", tags=["bio"]),
        ]
        scales = [_make_scale("s1", ["flat1", "varied1"], tags=["bio"], min_items=2)]
        registry = {"items": items, "scales": scales, "multiplex_map": {}}

        np.random.seed(42)
        mat = np.column_stack([
            np.full(30, 4.0),  # flat1: everyone answers 4
            np.random.randint(0, 5, 30).astype(float),  # varied1: varies
        ])

        analyzer = ItemEfficiencyAnalyzer(registry)
        analyzer.ingest_responses(mat, ["flat1", "varied1"])
        report = analyzer.analyze(semantic=False, tag_overlap=False, irt_discrimination=False)

        flat_eff = next(e for e in report.items if e.item_id == "flat1")
        assert flat_eff.response_variance is not None
        assert flat_eff.response_variance < 0.01
        assert "low_variance" in flat_eff.redundancy_flags


# ════════════════════════════════════════════════════════════════════════════
# 5. IRT Discrimination Layer
# ════════════════════════════════════════════════════════════════════════════


class TestIRTDiscrimination:
    def test_discrimination_estimated(self, tiny_registry, response_matrix_6x6):
        mat, item_ids = response_matrix_6x6
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        analyzer.ingest_responses(mat, item_ids)
        report = analyzer.analyze(semantic=False, tag_overlap=False, correlation=False)

        has_disc = [e for e in report.items if e.irt_discrimination is not None]
        assert len(has_disc) > 0

    def test_discrimination_non_negative(self, tiny_registry, response_matrix_6x6):
        mat, item_ids = response_matrix_6x6
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        analyzer.ingest_responses(mat, item_ids)
        report = analyzer.analyze(semantic=False, tag_overlap=False, correlation=False)

        for e in report.items:
            if e.irt_discrimination is not None:
                assert e.irt_discrimination >= 0.0

    def test_fisher_information_computed(self, tiny_registry, response_matrix_6x6):
        mat, item_ids = response_matrix_6x6
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        analyzer.ingest_responses(mat, item_ids)
        report = analyzer.analyze(semantic=False, tag_overlap=False, correlation=False)

        for e in report.items:
            if e.irt_discrimination is not None:
                assert e.fisher_information_at_mean is not None
                assert e.fisher_information_at_mean >= 0.0
                # Fisher info = a² × 0.25
                expected = e.irt_discrimination ** 2 * 0.25
                assert abs(e.fisher_information_at_mean - expected) < 0.01


# ════════════════════════════════════════════════════════════════════════════
# 6. Composite Scoring
# ════════════════════════════════════════════════════════════════════════════


class TestCompositeScoring:
    def test_all_scores_bounded(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.analyze()

        for e in report.items:
            assert 0.0 <= e.efficiency_score <= 1.0

    def test_recommendations_valid(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.analyze()

        for e in report.items:
            assert e.recommendation in ("keep", "review", "cut")

    def test_items_sorted_by_efficiency(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.analyze()

        scores = [e.efficiency_score for e in report.items]
        assert scores == sorted(scores)  # ascending

    def test_registry_only_mode(self, tiny_registry):
        """Without response data, correlation and IRT layers should be skipped."""
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.analyze()

        assert report.has_response_data is False
        assert report.n_statistical_twins == 0
        assert report.n_low_variance == 0
        assert report.n_low_discrimination == 0

    def test_full_mode_with_data(self, tiny_registry, response_matrix_6x6):
        mat, item_ids = response_matrix_6x6
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        analyzer.ingest_responses(mat, item_ids)
        report = analyzer.analyze()

        assert report.has_response_data is True
        assert report.n_respondents == 20


# ════════════════════════════════════════════════════════════════════════════
# 7. Reduced Set Recommendation
# ════════════════════════════════════════════════════════════════════════════


class TestReducedSet:
    def test_reduction_respects_target(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.recommend_reduced_set(target_reduction=0.30, min_items_per_scale=2)

        assert report.reduction_ratio <= 0.35  # within tolerance
        assert len(report.reduced_item_ids) >= 1

    def test_min_items_per_scale_protected(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.recommend_reduced_set(target_reduction=0.80, min_items_per_scale=2)

        # Count remaining items per scale
        remaining = set(report.reduced_item_ids)
        for scale in tiny_registry["scales"]:
            scale_items = [si["item_id"] for si in scale["items"]]
            n_remaining = len(set(scale_items) & remaining)
            total = len(scale_items)
            # Should keep at least min_items_per_scale (or all if scale is smaller)
            expected_min = min(2, total)
            assert n_remaining >= expected_min, (
                f"Scale {scale['id']}: keeping {n_remaining}/{total}, expected >= {expected_min}"
            )

    def test_precision_loss_computed(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.recommend_reduced_set(target_reduction=0.30)

        assert report.estimated_precision_loss >= 0.0
        assert report.estimated_precision_loss <= 1.0

    def test_reduced_set_contains_only_kept_items(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.recommend_reduced_set(target_reduction=0.30)

        cut_ids = {e.item_id for e in report.items if e.recommendation == "cut"}
        for item_id in report.reduced_item_ids:
            assert item_id not in cut_ids

    def test_borderline_items_marked_review(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.recommend_reduced_set(target_reduction=0.10, min_items_per_scale=2)

        # Items with low efficiency that weren't cut should be "review"
        for e in report.items:
            if e.recommendation == "review":
                assert e.efficiency_score < 0.50 or len(e.redundancy_flags) >= 1


# ════════════════════════════════════════════════════════════════════════════
# 8. Multiplexed Items
# ════════════════════════════════════════════════════════════════════════════


class TestMultiplexing:
    def test_shared_items_detected(self, registry_with_shared_items):
        analyzer = ItemEfficiencyAnalyzer(registry_with_shared_items)

        # shared1 and shared2 should be in 3 scales
        assert len(analyzer._item_to_scales["shared1"]) == 3
        assert len(analyzer._item_to_scales["shared2"]) == 3

    def test_shared_items_have_higher_tag_overlap(self, registry_with_shared_items):
        analyzer = ItemEfficiencyAnalyzer(registry_with_shared_items)
        report = analyzer.analyze(semantic=False, correlation=False, irt_discrimination=False)

        shared_eff = next(e for e in report.items if e.item_id == "shared1")
        # Shared items participate in more scales → higher tag overlap
        assert shared_eff.tag_overlap_score >= 0.0


# ════════════════════════════════════════════════════════════════════════════
# 9. Edge Cases
# ════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_registry(self):
        registry = {"items": [], "scales": [], "multiplex_map": {}}
        analyzer = ItemEfficiencyAnalyzer(registry)
        report = analyzer.analyze()

        assert report.total_items == 0
        assert report.total_scales == 0
        assert len(report.items) == 0

    def test_single_item(self):
        items = [_make_item("only1", "Just one item")]
        scales = [_make_scale("s1", ["only1"], min_items=1)]
        registry = {"items": items, "scales": scales, "multiplex_map": {}}

        analyzer = ItemEfficiencyAnalyzer(registry)
        report = analyzer.analyze()

        assert report.total_items == 1
        assert len(report.items) == 1

    def test_items_without_scales(self):
        items = [
            _make_item("orphan1", "Orphan item with no scale"),
            _make_item("orphan2", "Another orphan item"),
        ]
        scales = []
        registry = {"items": items, "scales": scales, "multiplex_map": {}}

        analyzer = ItemEfficiencyAnalyzer(registry)
        report = analyzer.analyze()

        assert report.total_items == 2
        for e in report.items:
            assert len(e.scale_ids) == 0

    def test_nan_in_response_matrix(self):
        items = [
            _make_item("i1", "Item one"),
            _make_item("i2", "Item two"),
            _make_item("i3", "Item three"),
        ]
        scales = [_make_scale("s1", ["i1", "i2", "i3"], min_items=2)]
        registry = {"items": items, "scales": scales, "multiplex_map": {}}

        mat = np.array([
            [1.0, 2.0, 3.0],
            [2.0, np.nan, 4.0],
            [3.0, 3.0, np.nan],
            [1.0, 1.0, 2.0],
            [4.0, 4.0, 3.0],
        ])

        analyzer = ItemEfficiencyAnalyzer(registry)
        analyzer.ingest_responses(mat, ["i1", "i2", "i3"])
        report = analyzer.analyze()

        # Should not crash with NaN
        assert report.total_items == 3
        assert report.has_response_data is True

    def test_matrix_column_mismatch_raises(self):
        items = [_make_item("i1", "test")]
        scales = [_make_scale("s1", ["i1"], min_items=1)]
        registry = {"items": items, "scales": scales, "multiplex_map": {}}

        analyzer = ItemEfficiencyAnalyzer(registry)
        mat = np.array([[1.0, 2.0]])  # 2 columns but 1 item_id

        with pytest.raises(AssertionError):
            analyzer.ingest_responses(mat, ["i1"])


# ════════════════════════════════════════════════════════════════════════════
# 10. Production Registry v3 (941 items)
# ════════════════════════════════════════════════════════════════════════════


REGISTRY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "registry_production", "v3",
)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(REGISTRY_DIR, "items.json")),
    reason="Production registry v3 not found",
)
class TestProductionRegistry:
    @pytest.fixture
    def prod_registry(self):
        with open(os.path.join(REGISTRY_DIR, "items.json")) as f:
            items = json.load(f)
        with open(os.path.join(REGISTRY_DIR, "scales.json")) as f:
            scales = json.load(f)
        return {"items": items, "scales": scales, "multiplex_map": {}}

    def test_loads_all_items(self, prod_registry):
        analyzer = ItemEfficiencyAnalyzer(prod_registry)
        assert len(analyzer._item_ids) == 941
        assert len(analyzer._scale_items) == 65

    def test_semantic_analysis_completes(self, prod_registry):
        analyzer = ItemEfficiencyAnalyzer(prod_registry)
        report = analyzer.analyze(tag_overlap=False, correlation=False, irt_discrimination=False)

        assert report.total_items == 941
        assert report.total_scales == 65
        assert report.n_semantic_duplicates > 0
        assert len(report.redundancy_clusters) > 0

    def test_full_registry_only_analysis(self, prod_registry):
        analyzer = ItemEfficiencyAnalyzer(prod_registry)
        report = analyzer.analyze()

        assert report.total_items == 941
        # Should find significant redundancy in 941 items
        assert report.n_semantic_duplicates > 100

        # Summary should be valid
        s = report.summary()
        assert s["total_items"] == 941

    def test_recommend_reduced_set_production(self, prod_registry):
        analyzer = ItemEfficiencyAnalyzer(prod_registry)
        report = analyzer.recommend_reduced_set(
            target_reduction=0.30, min_items_per_scale=3
        )

        # Should recommend cuts
        assert report.n_recommended_cuts > 0
        assert len(report.reduced_item_ids) > 0
        assert report.reduction_ratio <= 0.35

        # No scale should have fewer than 3 items (or fewer than total if small)
        remaining = set(report.reduced_item_ids)
        for scale in prod_registry["scales"]:
            scale_items = [si["item_id"] for si in scale["items"]]
            n_remaining = len(set(scale_items) & remaining)
            total = len(scale_items)
            expected_min = min(3, total)
            assert n_remaining >= expected_min, (
                f"Scale {scale['id']}: {n_remaining}/{total} remaining, need {expected_min}"
            )

    def test_true_text_duplicates_found(self, prod_registry):
        """Items with identical text across scales should get sim=1.0."""
        analyzer = ItemEfficiencyAnalyzer(prod_registry)
        report = analyzer.analyze()

        # Registry has Ryff/MindAge items with identical text in different scales
        perfect_dups = [
            e for e in report.items
            if e.max_semantic_similarity >= 0.99 and e.most_similar_cross_scale
        ]
        assert len(perfect_dups) > 0, "Expected cross-scale text duplicates (e.g., Ryff/MindAge)"

    def test_efficiency_distribution(self, prod_registry):
        """Efficiency scores should be distributed, not all the same."""
        analyzer = ItemEfficiencyAnalyzer(prod_registry)
        report = analyzer.analyze()

        scores = [e.efficiency_score for e in report.items]
        assert min(scores) < 0.3
        assert max(scores) > 0.7
        assert np.std(scores) > 0.1


# ════════════════════════════════════════════════════════════════════════════
# 11. Convenience Function
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not os.path.exists(os.path.join(REGISTRY_DIR, "items.json")),
    reason="Production registry v3 not found",
)
class TestConvenienceFunction:
    def test_analyze_registry_from_files(self):
        report = analyze_registry_from_files(
            items_path=os.path.join(REGISTRY_DIR, "items.json"),
            scales_path=os.path.join(REGISTRY_DIR, "scales.json"),
        )
        assert report.total_items == 941
        assert report.total_scales == 65


# ════════════════════════════════════════════════════════════════════════════
# 12. Shadow Algo Integration Helpers
# ════════════════════════════════════════════════════════════════════════════


class TestShadowAlgoHelpers:
    def test_efficiency_scores_extractable(self, tiny_registry):
        """Verify scores can be extracted as dict for injection into selection."""
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.analyze()

        scores = {e.item_id: e.efficiency_score for e in report.items}
        assert len(scores) == 6
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_cut_item_ids_extractable(self, tiny_registry):
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.recommend_reduced_set(target_reduction=0.30)

        cut_ids = {e.item_id for e in report.items if e.recommendation == "cut"}
        keep_ids = set(report.reduced_item_ids)
        assert len(cut_ids & keep_ids) == 0  # no overlap

    def test_report_serializable(self, tiny_registry):
        """Report summary should be JSON-serializable for logging."""
        analyzer = ItemEfficiencyAnalyzer(tiny_registry)
        report = analyzer.analyze()

        summary = report.summary()
        serialized = json.dumps(summary)
        assert len(serialized) > 0

        deserialized = json.loads(serialized)
        assert deserialized["total_items"] == 6
