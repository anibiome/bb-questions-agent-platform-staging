"""
Item Efficiency Analyzer — Cross-Instrument Redundancy Detection & Adaptive Item Reduction.

Patent claim: "Adaptive item reduction via cross-instrument concordance analysis."

This module identifies redundant, low-variance, and semantically duplicate items
across all 65 instruments in the registry, producing a ranked efficiency score
per item and recommending a reduced item set that maintains measurement precision.

Four analysis layers:
  1. Semantic similarity  — TF-IDF + cosine on item text (no data needed)
  2. Tag-based overlap    — shared construct tags across instruments (no data needed)
  3. Response correlation  — statistical twins from real response data (data needed)
  4. IRT discrimination   — Fisher information contribution per item (data needed)

Runs as a shadow algorithm: accumulates response data, periodically recomputes
efficiency rankings, and can feed back into item selection as a soft constraint.

Usage:
    from pipeline.item_efficiency import ItemEfficiencyAnalyzer

    # Registry-only analysis (no response data needed)
    analyzer = ItemEfficiencyAnalyzer(registry)
    report = analyzer.analyze()

    # With response data (richer analysis)
    analyzer = ItemEfficiencyAnalyzer(registry)
    analyzer.ingest_responses(response_matrix)
    report = analyzer.analyze()

    # Get reduced item set
    reduced = analyzer.recommend_reduced_set(target_reduction=0.30)
"""

from __future__ import annotations

import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy imports — degrade gracefully
# ---------------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    from sklearn.cluster import AgglomerativeClustering

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


# ════════════════════════════════════════════════════════════════════════════
# Data classes
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class ItemEfficiency:
    """Per-item efficiency assessment."""

    item_id: str
    text: str
    scale_ids: List[str]
    tags: List[str]
    response_type: str

    # Semantic layer
    semantic_cluster_id: int = -1
    semantic_cluster_size: int = 1
    max_semantic_similarity: float = 0.0
    most_similar_item_id: Optional[str] = None
    most_similar_item_text: Optional[str] = None
    most_similar_cross_scale: bool = False

    # Tag overlap layer
    tag_overlap_score: float = 0.0  # 0-1: how many other scales share this item's tags

    # Response data layer (populated when data available)
    response_variance: Optional[float] = None  # low = everyone answers same
    max_cross_item_correlation: Optional[float] = None  # high = redundant
    most_correlated_item_id: Optional[str] = None
    n_responses: int = 0

    # IRT layer (populated when data available)
    irt_discrimination: Optional[float] = None  # low = flat info curve
    fisher_information_at_mean: Optional[float] = None

    # Composite
    efficiency_score: float = 1.0  # 0-1: higher = more efficient / unique
    redundancy_flags: List[str] = field(default_factory=list)
    recommendation: str = "keep"  # keep | review | cut


@dataclass
class RedundancyCluster:
    """Group of semantically or statistically similar items."""

    cluster_id: int
    item_ids: List[str]
    item_texts: List[str]
    scale_ids: List[str]  # scales involved
    mean_similarity: float
    cluster_type: str  # "semantic" | "statistical" | "both"
    representative_item_id: str  # best item to keep


@dataclass
class EfficiencyReport:
    """Full analysis output."""

    total_items: int
    total_scales: int
    items: List[ItemEfficiency]
    redundancy_clusters: List[RedundancyCluster]
    low_variance_items: List[str]
    low_discrimination_items: List[str]

    # Summary stats
    n_semantic_duplicates: int = 0
    n_statistical_twins: int = 0
    n_low_variance: int = 0
    n_low_discrimination: int = 0
    n_recommended_cuts: int = 0
    estimated_precision_loss: float = 0.0

    # Reduced set
    reduced_item_ids: List[str] = field(default_factory=list)
    reduction_ratio: float = 0.0

    has_response_data: bool = False
    n_respondents: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "total_scales": self.total_scales,
            "semantic_duplicate_clusters": len(self.redundancy_clusters),
            "n_semantic_duplicates": self.n_semantic_duplicates,
            "n_statistical_twins": self.n_statistical_twins,
            "n_low_variance": self.n_low_variance,
            "n_low_discrimination": self.n_low_discrimination,
            "n_recommended_cuts": self.n_recommended_cuts,
            "reduced_set_size": len(self.reduced_item_ids),
            "reduction_ratio": round(self.reduction_ratio, 3),
            "estimated_precision_loss": round(self.estimated_precision_loss, 4),
            "has_response_data": self.has_response_data,
            "n_respondents": self.n_respondents,
        }


# ════════════════════════════════════════════════════════════════════════════
# Core analyzer
# ════════════════════════════════════════════════════════════════════════════


class ItemEfficiencyAnalyzer:
    """
    Cross-instrument item redundancy detection and adaptive reduction.

    Operates in two modes:
      - Registry-only: semantic + tag analysis (no response data needed)
      - Full: adds response correlation + IRT discrimination (needs data)
    """

    # Thresholds (tunable)
    SEMANTIC_SIMILARITY_THRESHOLD = 0.55  # cosine above this = near-duplicate
    SEMANTIC_CLUSTER_DISTANCE = 0.45  # 1 - cosine for agglomerative clustering
    CORRELATION_THRESHOLD = 0.70  # Pearson |r| above this = statistical twin
    LOW_VARIANCE_THRESHOLD = 0.10  # items below this variance are uninformative
    LOW_DISCRIMINATION_THRESHOLD = 0.30  # IRT 'a' below this = flat info curve
    TAG_OVERLAP_WEIGHT = 0.15
    SEMANTIC_WEIGHT = 0.35
    CORRELATION_WEIGHT = 0.30
    DISCRIMINATION_WEIGHT = 0.20

    def __init__(self, registry: Any):
        """
        Args:
            registry: Registry object with .items, .scales, .questionnaires attributes
                      OR a dict with keys 'items', 'scales', 'multiplex_map'
        """
        if isinstance(registry, dict):
            self._items_raw = registry.get("items", [])
            self._scales_raw = registry.get("scales", [])
            self._multiplex = registry.get("multiplex_map", {})
        else:
            self._items_raw = (
                registry.items
                if hasattr(registry, "items") and isinstance(registry.items, list)
                else []
            )
            self._scales_raw = (
                registry.scales
                if hasattr(registry, "scales") and isinstance(registry.scales, list)
                else []
            )
            self._multiplex = (
                registry.multiplex_map
                if hasattr(registry, "multiplex_map")
                else {}
            )

        # Build lookups
        self._item_map: Dict[str, Dict] = {i["id"]: i for i in self._items_raw}
        self._item_ids: List[str] = [i["id"] for i in self._items_raw]
        self._item_texts: List[str] = [i.get("text", "") for i in self._items_raw]

        # Item → scale mapping
        self._item_to_scales: Dict[str, List[str]] = defaultdict(list)
        self._scale_items: Dict[str, List[str]] = {}
        for scale in self._scales_raw:
            sid = scale.get("id") or scale.get("name", "unknown")
            sitems = [
                si.get("item_id", si) if isinstance(si, dict) else si
                for si in scale.get("items", [])
            ]
            self._scale_items[sid] = sitems
            for iid in sitems:
                self._item_to_scales[iid].append(sid)

        # Response data (populated via ingest_responses)
        self._response_matrix: Optional[np.ndarray] = None  # [n_respondents, n_items]
        self._response_item_ids: List[str] = []
        self._n_respondents: int = 0

    # ── Public API ──────────────────────────────────────────────────────

    def ingest_responses(
        self,
        response_matrix: np.ndarray,
        item_ids: List[str],
    ) -> None:
        """
        Ingest a response matrix for correlation + IRT analysis.

        Args:
            response_matrix: shape [n_respondents, n_items], NaN for missing.
            item_ids: item IDs corresponding to columns.
        """
        assert response_matrix.shape[1] == len(item_ids), (
            f"Matrix has {response_matrix.shape[1]} columns but {len(item_ids)} item_ids"
        )
        self._response_matrix = response_matrix.astype(float)
        self._response_item_ids = list(item_ids)
        self._n_respondents = response_matrix.shape[0]

    def analyze(
        self,
        *,
        semantic: bool = True,
        tag_overlap: bool = True,
        correlation: bool = True,
        irt_discrimination: bool = True,
    ) -> EfficiencyReport:
        """
        Run full efficiency analysis.

        Returns an EfficiencyReport with per-item scores and recommendations.
        """
        # Initialize per-item records
        efficiencies: Dict[str, ItemEfficiency] = {}
        for item in self._items_raw:
            iid = item["id"]
            efficiencies[iid] = ItemEfficiency(
                item_id=iid,
                text=item.get("text", ""),
                scale_ids=self._item_to_scales.get(iid, []),
                tags=item.get("tags", []),
                response_type=item.get("response_type", "unknown"),
            )

        # Layer 1: Semantic similarity
        clusters: List[RedundancyCluster] = []
        if semantic and _HAS_SKLEARN and len(self._item_texts) > 1:
            clusters = self._compute_semantic_similarity(efficiencies)

        # Layer 2: Tag overlap
        if tag_overlap:
            self._compute_tag_overlap(efficiencies)

        # Layer 3: Response correlation
        if correlation and self._response_matrix is not None:
            self._compute_response_correlation(efficiencies)

        # Layer 4: IRT discrimination
        if irt_discrimination and self._response_matrix is not None:
            self._compute_irt_discrimination(efficiencies)

        # Composite scoring
        self._compute_composite_scores(efficiencies)

        # Build report
        items_list = sorted(
            efficiencies.values(), key=lambda e: e.efficiency_score
        )

        low_var = [e.item_id for e in items_list if e.response_variance is not None and e.response_variance < self.LOW_VARIANCE_THRESHOLD]
        low_disc = [e.item_id for e in items_list if e.irt_discrimination is not None and e.irt_discrimination < self.LOW_DISCRIMINATION_THRESHOLD]

        report = EfficiencyReport(
            total_items=len(self._items_raw),
            total_scales=len(self._scales_raw),
            items=items_list,
            redundancy_clusters=clusters,
            low_variance_items=low_var,
            low_discrimination_items=low_disc,
            n_semantic_duplicates=sum(
                1 for e in items_list if e.max_semantic_similarity >= self.SEMANTIC_SIMILARITY_THRESHOLD
            ),
            n_statistical_twins=sum(
                1 for e in items_list
                if e.max_cross_item_correlation is not None
                and e.max_cross_item_correlation >= self.CORRELATION_THRESHOLD
            ),
            n_low_variance=len(low_var),
            n_low_discrimination=len(low_disc),
            n_recommended_cuts=sum(
                1 for e in items_list if e.recommendation == "cut"
            ),
            has_response_data=self._response_matrix is not None,
            n_respondents=self._n_respondents,
        )

        return report

    def recommend_reduced_set(
        self,
        target_reduction: float = 0.30,
        *,
        min_items_per_scale: int = 3,
    ) -> EfficiencyReport:
        """
        Produce a recommended reduced item set.

        Args:
            target_reduction: fraction of items to remove (0.30 = cut 30%)
            min_items_per_scale: never reduce a scale below this many items
        """
        report = self.analyze()
        n_target_cuts = int(len(report.items) * target_reduction)

        # Reset all recommendations — analyze() sets them based on composite
        # scores alone, but recommend_reduced_set applies min_items_per_scale
        # protection that must override analyze()'s recommendations.
        for item_eff in report.items:
            item_eff.recommendation = "keep"

        # Sort by efficiency (lowest first = best candidates for cutting)
        candidates = sorted(report.items, key=lambda e: e.efficiency_score)

        cuts: Set[str] = set()
        scale_remaining: Dict[str, int] = {
            sid: len(items) for sid, items in self._scale_items.items()
        }

        for item_eff in candidates:
            if len(cuts) >= n_target_cuts:
                break

            # Check: would cutting this item violate min_items_per_scale?
            can_cut = True
            for sid in item_eff.scale_ids:
                if scale_remaining.get(sid, 0) <= min_items_per_scale:
                    can_cut = False
                    break

            if can_cut:
                cuts.add(item_eff.item_id)
                item_eff.recommendation = "cut"
                for sid in item_eff.scale_ids:
                    scale_remaining[sid] = scale_remaining.get(sid, 0) - 1

        # Mark borderline items as "review" (low efficiency but protected)
        for item_eff in report.items:
            if item_eff.item_id not in cuts:
                if item_eff.efficiency_score < 0.50 or len(item_eff.redundancy_flags) >= 1:
                    item_eff.recommendation = "review"

        # Items not cut
        report.reduced_item_ids = [
            e.item_id for e in report.items if e.item_id not in cuts
        ]
        report.n_recommended_cuts = len(cuts)
        report.reduction_ratio = len(cuts) / max(len(report.items), 1)

        # Estimate precision loss (simplified: assume each cut item's info is
        # partially redundant based on its max similarity/correlation)
        info_losses = []
        for item_eff in report.items:
            if item_eff.item_id in cuts:
                redundancy = max(
                    item_eff.max_semantic_similarity,
                    item_eff.max_cross_item_correlation or 0.0,
                )
                # If highly redundant, cutting it loses very little info
                info_losses.append(1.0 - redundancy)

        report.estimated_precision_loss = (
            float(np.mean(info_losses)) if info_losses else 0.0
        )

        return report

    # ── Layer 1: Semantic Similarity ────────────────────────────────────

    def _compute_semantic_similarity(
        self, efficiencies: Dict[str, ItemEfficiency]
    ) -> List[RedundancyCluster]:
        """TF-IDF + cosine similarity on item text."""
        texts = []
        ids = []
        for item in self._items_raw:
            text = item.get("text", "")
            if text:
                texts.append(self._normalize_text(text))
                ids.append(item["id"])

        if len(texts) < 2:
            return []

        # TF-IDF with n-grams for better semantic capture
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            stop_words="english",
            min_df=2,
            sublinear_tf=True,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tfidf = vectorizer.fit_transform(texts)

        # Compute pairwise similarity
        sim_matrix = sklearn_cosine(tfidf)

        # For each item, find most similar item (from DIFFERENT scale)
        for i, iid in enumerate(ids):
            best_sim = 0.0
            best_j = -1
            best_cross = False

            i_scales = set(self._item_to_scales.get(iid, []))

            for j, jid in enumerate(ids):
                if i == j:
                    continue
                s = float(sim_matrix[i, j])
                j_scales = set(self._item_to_scales.get(jid, []))
                is_cross = not bool(i_scales & j_scales)  # different scales

                if s > best_sim:
                    best_sim = s
                    best_j = j
                    best_cross = is_cross

            if iid in efficiencies and best_j >= 0:
                efficiencies[iid].max_semantic_similarity = round(best_sim, 4)
                efficiencies[iid].most_similar_item_id = ids[best_j]
                efficiencies[iid].most_similar_item_text = texts[best_j]
                efficiencies[iid].most_similar_cross_scale = best_cross

                if best_sim >= self.SEMANTIC_SIMILARITY_THRESHOLD:
                    efficiencies[iid].redundancy_flags.append("semantic_duplicate")

        # Agglomerative clustering to find redundancy groups
        distance_matrix = 1.0 - sim_matrix
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.clip(distance_matrix, 0, 2)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.SEMANTIC_CLUSTER_DISTANCE,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)

        # Build clusters
        cluster_items: Dict[int, List[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            cluster_items[label].append(i)

        clusters: List[RedundancyCluster] = []
        for cid, indices in cluster_items.items():
            if len(indices) < 2:
                continue

            c_ids = [ids[i] for i in indices]
            c_texts = [texts[i] for i in indices]
            c_scales = list(
                set(
                    s
                    for iid in c_ids
                    for s in self._item_to_scales.get(iid, [])
                )
            )

            # Mean pairwise similarity within cluster
            if len(indices) > 1:
                pairs = [
                    float(sim_matrix[i, j])
                    for i in indices
                    for j in indices
                    if i < j
                ]
                mean_sim = float(np.mean(pairs)) if pairs else 0.0
            else:
                mean_sim = 1.0

            # Representative = item with highest mean similarity to others
            best_rep_idx = max(
                indices,
                key=lambda i: float(
                    np.mean([sim_matrix[i, j] for j in indices if j != i])
                )
                if len(indices) > 1
                else 0.0,
            )

            cluster = RedundancyCluster(
                cluster_id=cid,
                item_ids=c_ids,
                item_texts=c_texts,
                scale_ids=c_scales,
                mean_similarity=round(mean_sim, 4),
                cluster_type="semantic",
                representative_item_id=ids[best_rep_idx],
            )
            clusters.append(cluster)

            # Tag items with their cluster
            for iid in c_ids:
                if iid in efficiencies:
                    efficiencies[iid].semantic_cluster_id = cid
                    efficiencies[iid].semantic_cluster_size = len(c_ids)

        # Sort by size (largest clusters first)
        clusters.sort(key=lambda c: len(c.item_ids), reverse=True)
        return clusters

    # ── Layer 2: Tag Overlap ────────────────────────────────────────────

    def _compute_tag_overlap(
        self, efficiencies: Dict[str, ItemEfficiency]
    ) -> None:
        """Score items by how many other scales share their tags."""
        # For each tag, which scales use it?
        tag_to_scales: Dict[str, Set[str]] = defaultdict(set)
        for scale in self._scales_raw:
            sid = scale.get("id") or scale.get("name")
            for tag in scale.get("tags", []):
                tag_to_scales[tag].add(sid)

        total_scales = len(self._scales_raw)

        for iid, eff in efficiencies.items():
            if not eff.tags:
                continue

            # How many scales share at least one tag with this item?
            shared_scales: Set[str] = set()
            for tag in eff.tags:
                shared_scales.update(tag_to_scales.get(tag, set()))

            # Remove this item's own scales
            own_scales = set(eff.scale_ids)
            cross_scales = shared_scales - own_scales

            eff.tag_overlap_score = round(
                len(cross_scales) / max(total_scales - len(own_scales), 1), 4
            )

    # ── Layer 3: Response Correlation ───────────────────────────────────

    def _compute_response_correlation(
        self, efficiencies: Dict[str, ItemEfficiency]
    ) -> None:
        """Find statistical twins from response data."""
        if self._response_matrix is None:
            return

        mat = self._response_matrix
        ids = self._response_item_ids
        n_items = mat.shape[1]

        # Compute variance per item
        for j, iid in enumerate(ids):
            col = mat[:, j]
            valid = col[~np.isnan(col)]
            if iid in efficiencies and len(valid) > 1:
                var = float(np.var(valid))
                efficiencies[iid].response_variance = round(var, 4)
                efficiencies[iid].n_responses = len(valid)

                if var < self.LOW_VARIANCE_THRESHOLD:
                    efficiencies[iid].redundancy_flags.append("low_variance")

        # Pairwise correlation (only for items with enough responses)
        # Use masked correlation to handle NaN
        min_shared = max(5, self._n_respondents // 3)

        for i in range(n_items):
            iid = ids[i]
            if iid not in efficiencies:
                continue

            best_r = 0.0
            best_j_id = None

            col_i = mat[:, i]
            i_scales = set(self._item_to_scales.get(iid, []))

            for j in range(n_items):
                if i == j:
                    continue
                jid = ids[j]
                j_scales = set(self._item_to_scales.get(jid, []))

                # Only flag cross-scale correlations (within-scale is expected)
                if i_scales & j_scales:
                    continue

                col_j = mat[:, j]
                mask = ~np.isnan(col_i) & ~np.isnan(col_j)
                n_shared = int(mask.sum())

                if n_shared < min_shared:
                    continue

                r = float(np.corrcoef(col_i[mask], col_j[mask])[0, 1])
                if abs(r) > abs(best_r):
                    best_r = r
                    best_j_id = jid

            if best_j_id is not None:
                efficiencies[iid].max_cross_item_correlation = round(
                    abs(best_r), 4
                )
                efficiencies[iid].most_correlated_item_id = best_j_id

                if abs(best_r) >= self.CORRELATION_THRESHOLD:
                    efficiencies[iid].redundancy_flags.append("statistical_twin")

    # ── Layer 4: IRT Discrimination ─────────────────────────────────────

    def _compute_irt_discrimination(
        self, efficiencies: Dict[str, ItemEfficiency]
    ) -> None:
        """
        Estimate IRT discrimination from response data using point-biserial
        correlation between item response and total scale score.

        This is a simplified discrimination estimate. Full IRT calibration
        (Samejima GRM) is done elsewhere in the pipeline.
        """
        if self._response_matrix is None:
            return

        mat = self._response_matrix
        ids = self._response_item_ids

        # For each scale, compute item-total correlation
        for sid, scale_item_ids in self._scale_items.items():
            # Find columns for this scale
            col_indices = [
                i for i, iid in enumerate(ids) if iid in scale_item_ids
            ]
            if len(col_indices) < 3:
                continue

            # Compute total score per respondent (nanmean × n_items)
            scale_mat = mat[:, col_indices]
            total = np.nanmean(scale_mat, axis=1) * len(col_indices)

            for ci in col_indices:
                iid = ids[ci]
                if iid not in efficiencies:
                    continue

                col = mat[:, ci]
                mask = ~np.isnan(col) & ~np.isnan(total)
                n_valid = int(mask.sum())

                if n_valid < 5:
                    continue

                # Corrected item-total: correlate item with (total minus item)
                item_vals = col[mask]
                corrected_total = total[mask] - item_vals
                if np.std(item_vals) < 1e-9 or np.std(corrected_total) < 1e-9:
                    r_it = 0.0
                else:
                    r_it = float(
                        np.corrcoef(item_vals, corrected_total)[0, 1]
                    )

                # Map r_it to approximate IRT discrimination
                # a ≈ r_it / sqrt(1 - r_it^2) * 1.702 (Lord's approximation)
                r_clamped = max(min(abs(r_it), 0.99), 0.0)
                if r_clamped > 0:
                    a_est = (
                        r_clamped / math.sqrt(1.0 - r_clamped**2)
                    )
                else:
                    a_est = 0.0

                eff = efficiencies[iid]
                # Keep the highest discrimination across scales
                if eff.irt_discrimination is None or a_est > eff.irt_discrimination:
                    eff.irt_discrimination = round(a_est, 4)

                    # Fisher info at mean (simplified)
                    eff.fisher_information_at_mean = round(a_est**2 * 0.25, 4)

                if a_est < self.LOW_DISCRIMINATION_THRESHOLD:
                    if "low_discrimination" not in eff.redundancy_flags:
                        eff.redundancy_flags.append("low_discrimination")

    # ── Composite Scoring ───────────────────────────────────────────────

    def _compute_composite_scores(
        self, efficiencies: Dict[str, ItemEfficiency]
    ) -> None:
        """
        Compute final efficiency score per item.

        Score = 1.0 = maximally efficient (unique, discriminating, informative)
        Score = 0.0 = maximally redundant (duplicate, flat, low variance)
        """
        has_data = self._response_matrix is not None

        for iid, eff in efficiencies.items():
            scores = []
            weights = []

            # Semantic uniqueness: 1 - max_similarity
            sem_score = 1.0 - eff.max_semantic_similarity
            scores.append(sem_score)
            weights.append(self.SEMANTIC_WEIGHT)

            # Tag uniqueness: 1 - overlap
            tag_score = 1.0 - eff.tag_overlap_score
            scores.append(tag_score)
            weights.append(self.TAG_OVERLAP_WEIGHT)

            if has_data:
                # Correlation uniqueness
                corr = eff.max_cross_item_correlation or 0.0
                corr_score = 1.0 - corr
                scores.append(corr_score)
                weights.append(self.CORRELATION_WEIGHT)

                # Discrimination value
                disc = eff.irt_discrimination
                if disc is not None:
                    # Normalize: a=0 → 0, a=2.0+ → 1.0
                    disc_score = min(disc / 2.0, 1.0)
                else:
                    disc_score = 0.5  # no info → neutral
                scores.append(disc_score)
                weights.append(self.DISCRIMINATION_WEIGHT)

                # Variance bonus/penalty
                if eff.response_variance is not None:
                    if eff.response_variance < self.LOW_VARIANCE_THRESHOLD:
                        # Heavy penalty for low variance
                        scores.append(0.1)
                        weights.append(0.15)

            # Weighted average
            total_w = sum(weights)
            eff.efficiency_score = round(
                sum(s * w for s, w in zip(scores, weights)) / total_w
                if total_w > 0
                else 0.5,
                4,
            )

            # Recommendation
            n_flags = len(eff.redundancy_flags)
            if n_flags >= 2 or eff.efficiency_score < 0.35:
                eff.recommendation = "cut"
            elif n_flags >= 1 or eff.efficiency_score < 0.50:
                eff.recommendation = "review"
            else:
                eff.recommendation = "keep"

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Lowercase, strip punctuation, normalize whitespace."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text


# ════════════════════════════════════════════════════════════════════════════
# Convenience: run from CLI or as standalone script
# ════════════════════════════════════════════════════════════════════════════

def analyze_registry_from_files(
    items_path: str,
    scales_path: str,
    multiplex_path: str = "",
) -> EfficiencyReport:
    """Load registry from JSON files and run analysis."""
    import json as _json

    with open(items_path) as f:
        items = _json.load(f)
    with open(scales_path) as f:
        scales = _json.load(f)

    multiplex = {}
    if multiplex_path:
        with open(multiplex_path) as f:
            multiplex = _json.load(f)

    registry = {"items": items, "scales": scales, "multiplex_map": multiplex}
    analyzer = ItemEfficiencyAnalyzer(registry)
    return analyzer.analyze()
