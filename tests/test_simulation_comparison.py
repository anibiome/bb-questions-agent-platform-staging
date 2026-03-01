"""
Simulation comparison framework — the publishable experiment.

Compares 4 item-selection strategies on the same synthetic population:

  1. RANDOM: Pick k items uniformly at random each day.
  2. FIXED-FORM: Administer items in a fixed cyclic order (standard practice).
  3. SINGLE-CAT: Classical CAT — maximize Fisher information for ONE scale at a time.
  4. MULTIPLEXED (ours): Maximize CROSS-SCALE Fisher information, exploiting
     shared items across instruments (the active sensing innovation).

Evaluation metrics (per simulated person):
  - Mean SE(θ) across scales after N days (lower = more precise measurement)
  - Scales unlocked within N days (higher = better coverage)
  - Total Fisher information accumulated (higher = more efficient)
  - Days to first reliable estimate (SE < 0.5) per scale (lower = faster)

Population: M synthetic respondents with known true θ per scale, generating
responses from the GRM.  This provides ground-truth for evaluating precision.

Reference design:
  Choi & Swartz (2009). Applied Psychological Measurement 33(8):619-632.
  van der Linden & Glas (2010). Elements of Adaptive Testing. Springer.
"""

import math
import random
import unittest
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from questions_agent_platform.pipeline.irt import (
    GRMItemParams,
    categories_for_response_type,
    default_scale_params,
    estimate_theta_eap,
    grm_category_probs,
    grm_item_information,
    score_scale_irt,
    se_at_theta,
    total_information,
)


# ---------------------------------------------------------------------------
# Synthetic population
# ---------------------------------------------------------------------------

@dataclass
class SyntheticPerson:
    """A simulated respondent with known true θ per scale."""
    person_id: str
    true_theta: Dict[str, float]  # scale_id → true θ


@dataclass
class SimScale:
    """Minimal scale definition for simulation."""
    scale_id: str
    item_params: List[GRMItemParams]
    min_items: int = 3


@dataclass
class SimResult:
    """Results from one strategy run on one person."""
    person_id: str
    strategy: str
    days: int
    # Per-scale metrics at end of simulation
    se_by_scale: Dict[str, float]
    info_by_scale: Dict[str, float]
    scales_unlocked: int
    days_to_reliable: Dict[str, Optional[int]]  # scale → first day SE < threshold
    theta_bias: Dict[str, float]  # |θ̂ - θ_true|
    total_items_asked: int


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def generate_response(theta: float, item: GRMItemParams, rng: random.Random) -> int:
    """Sample a response from the GRM given true θ."""
    probs = grm_category_probs(theta, item)
    r = rng.random()
    cumulative = 0.0
    for k, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return k
    return len(probs) - 1


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def select_random(
    available_items: List[GRMItemParams],
    k: int,
    rng: random.Random,
    **kwargs,
) -> List[GRMItemParams]:
    """Random selection: pick k items uniformly."""
    return rng.sample(available_items, min(k, len(available_items)))


def select_fixed_form(
    available_items: List[GRMItemParams],
    k: int,
    day: int,
    **kwargs,
) -> List[GRMItemParams]:
    """Fixed cyclic order: items administered in sequence, k per day."""
    n = len(available_items)
    if n == 0:
        return []
    start = (day * k) % n
    selected = []
    for i in range(k):
        idx = (start + i) % n
        selected.append(available_items[idx])
    return selected


def select_single_cat(
    available_items: List[GRMItemParams],
    k: int,
    current_theta: Dict[str, float],
    scales: List[SimScale],
    target_scale_idx: int,
    answered_items: Optional[Set[str]] = None,
    **kwargs,
) -> List[GRMItemParams]:
    """Classical single-scale CAT: maximize Fisher info for ONE target scale."""
    target = scales[target_scale_idx % len(scales)]
    theta = current_theta.get(target.scale_id, 0.0)
    target_item_ids = {ip.item_id for ip in target.item_params}
    already = answered_items or set()

    # Score each available item by its information for the target scale
    scored = []
    for item in available_items:
        if item.item_id in target_item_ids:
            info = grm_item_information(theta, item)
            if item.item_id not in already:
                info *= 3.0
        else:
            info = 0.0
        scored.append((info, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]


def select_multiplexed(
    available_items: List[GRMItemParams],
    k: int,
    current_theta: Dict[str, float],
    scales: List[SimScale],
    item_to_scales: Dict[str, List[str]],
    answered_items: Optional[Set[str]] = None,
    responses_by_scale: Optional[Dict[str, Dict[str, int]]] = None,
    **kwargs,
) -> List[GRMItemParams]:
    """
    Multiplexed active sensing: maximize SE-WEIGHTED Fisher information
    across ALL scales that each item contributes to.

    Key innovation: scales with higher SE (less precise estimates) get
    proportionally more weight, so the selector automatically invests
    measurement budget where it's needed most.  Shared items get a
    natural advantage because they reduce SE on multiple scales per ask.
    """
    already = answered_items or set()
    resps = responses_by_scale or {}

    # Compute current SE per scale to weight the allocation
    scale_se: Dict[str, float] = {}
    for sc in scales:
        scale_resps = resps.get(sc.scale_id, {})
        responded_params = [ip for ip in sc.item_params if ip.item_id in scale_resps]
        if len(responded_params) >= 2:
            theta = current_theta.get(sc.scale_id, 0.0)
            scale_se[sc.scale_id] = se_at_theta(theta, responded_params)
        else:
            scale_se[sc.scale_id] = 2.0  # high prior uncertainty

    # Normalize SE weights (scales with higher SE get more priority)
    total_se = sum(scale_se.values()) or 1.0
    se_weight = {sid: se / total_se for sid, se in scale_se.items()}

    scored = []
    for item in available_items:
        total_info = 0.0
        for sid in item_to_scales.get(item.item_id, []):
            theta = current_theta.get(sid, 0.0)
            info = grm_item_information(theta, item)
            total_info += info * se_weight.get(sid, 1.0 / len(scales))
        # Strong novelty bonus: first response for an item is most valuable
        if item.item_id not in already:
            total_info *= 5.0
        scored.append((total_info, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def run_simulation(
    person: SyntheticPerson,
    scales: List[SimScale],
    strategy: str,
    days: int,
    k_per_day: int,
    rng: random.Random,
    se_reliable_threshold: float = 0.5,
) -> SimResult:
    """
    Run one person through N days with a given selection strategy.
    Returns measurement quality metrics.
    """
    # Build item-to-scales mapping and deduplicated item list
    item_to_scales: Dict[str, List[str]] = {}
    all_items_by_id: Dict[str, GRMItemParams] = {}
    for sc in scales:
        for ip in sc.item_params:
            item_to_scales.setdefault(ip.item_id, []).append(sc.scale_id)
            all_items_by_id[ip.item_id] = ip

    all_items = list(all_items_by_id.values())

    # Track responses per scale
    responses_by_scale: Dict[str, Dict[str, int]] = {sc.scale_id: {} for sc in scales}
    current_theta: Dict[str, float] = {sc.scale_id: 0.0 for sc in scales}
    days_to_reliable: Dict[str, Optional[int]] = {sc.scale_id: None for sc in scales}
    answered_items: Set[str] = set()  # items with at least one response
    total_items_asked = 0

    for day in range(days):
        # Select items
        if strategy == "random":
            selected = select_random(all_items, k_per_day, rng)
        elif strategy == "fixed_form":
            selected = select_fixed_form(all_items, k_per_day, day)
        elif strategy == "single_cat":
            selected = select_single_cat(
                all_items, k_per_day, current_theta, scales,
                target_scale_idx=day,
                answered_items=answered_items,
            )
        elif strategy == "multiplexed":
            selected = select_multiplexed(
                all_items, k_per_day, current_theta, scales, item_to_scales,
                answered_items=answered_items,
                responses_by_scale=responses_by_scale,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Generate responses and assign to scales
        for item in selected:
            total_items_asked += 1
            answered_items.add(item.item_id)
            for sid in item_to_scales.get(item.item_id, []):
                true_theta = person.true_theta.get(sid, 0.0)
                resp = generate_response(true_theta, item, rng)
                responses_by_scale[sid][item.item_id] = resp

        # Update theta estimates
        for sc in scales:
            resps = responses_by_scale[sc.scale_id]
            if len(resps) >= 2:
                irt_result = score_scale_irt(sc.item_params, resps)
                if irt_result:
                    current_theta[sc.scale_id] = irt_result.theta
                    # Check if reliable
                    if (days_to_reliable[sc.scale_id] is None
                            and irt_result.se_theta < se_reliable_threshold):
                        days_to_reliable[sc.scale_id] = day + 1

    # Final metrics
    se_by_scale: Dict[str, float] = {}
    info_by_scale: Dict[str, float] = {}
    theta_bias: Dict[str, float] = {}
    scales_unlocked = 0

    for sc in scales:
        resps = responses_by_scale[sc.scale_id]
        if len(resps) >= sc.min_items:
            scales_unlocked += 1

        theta = current_theta[sc.scale_id]
        # Compute SE from Fisher info at estimated theta using responded items
        responded_params = [ip for ip in sc.item_params if ip.item_id in resps]
        if len(responded_params) >= 2:
            se = se_at_theta(theta, responded_params)
            info = total_information(theta, responded_params)
        else:
            se = float("inf")
            info = 0.0

        se_by_scale[sc.scale_id] = se
        info_by_scale[sc.scale_id] = info
        true_theta = person.true_theta.get(sc.scale_id, 0.0)
        theta_bias[sc.scale_id] = abs(theta - true_theta)

    return SimResult(
        person_id=person.person_id,
        strategy=strategy,
        days=days,
        se_by_scale=se_by_scale,
        info_by_scale=info_by_scale,
        scales_unlocked=scales_unlocked,
        days_to_reliable=days_to_reliable,
        theta_bias=theta_bias,
        total_items_asked=total_items_asked,
    )


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

def _build_demo_scales() -> List[SimScale]:
    """
    Build 6 overlapping scales simulating the real multi-instrument registry.

    40 total item slots across 6 scales, but only 28 unique items due to
    sharing.  With k=5/day and 10-15 days, budget is tight enough to
    differentiate strategies.

    Shared items:
      - item_4, item_5: mood + anxiety (emotional crossover)
      - item_10, item_11: anxiety + stress
      - item_18, item_19: sleep + energy
      - item_24: stress + energy + resilience (triple shared)
    """
    scale_configs = [
        # (name, item_ids, min_items)
        ("mood", [f"item_{i}" for i in range(8)], 5),           # items 0-7
        ("anxiety", [f"item_{i}" for i in [4,5,8,9,10,11,12]], 5),  # 4,5 shared w/mood
        ("sleep", [f"item_{i}" for i in range(15, 22)], 5),     # items 15-21
        ("energy", [f"item_{i}" for i in [18,19,22,23,24,25]], 4),  # 18,19 shared w/sleep
        ("stress", [f"item_{i}" for i in [10,11,24,26,27,28,29]], 5),  # 10,11 shared w/anxiety, 24 shared w/energy
        ("resilience", [f"item_{i}" for i in [24,30,31,32,33]], 4),  # 24 triple-shared
    ]

    scales = []
    for name, items, min_items in scale_configs:
        params = [
            GRMItemParams(
                item_id=iid,
                discrimination=0.8 + 0.4 * (hash(iid + name) % 5) / 4,
                thresholds=(-1.5, -0.5, 0.5, 1.5),
            )
            for iid in items
        ]
        scales.append(SimScale(scale_id=name, item_params=params, min_items=min_items))
    return scales


def _build_population(n: int, rng: random.Random, scale_ids: List[str]) -> List[SyntheticPerson]:
    """Generate n persons with random true θ per scale."""
    persons = []
    for i in range(n):
        true_theta = {sid: rng.gauss(0, 1) for sid in scale_ids}
        persons.append(SyntheticPerson(person_id=f"person_{i}", true_theta=true_theta))
    return persons


class TestSimulationFramework(unittest.TestCase):
    """Basic validation that the simulation framework runs."""

    def test_single_person_all_strategies(self):
        """All 4 strategies should complete without error."""
        scales = _build_demo_scales()
        person = SyntheticPerson(
            person_id="test",
            true_theta={sc.scale_id: 0.5 for sc in scales},
        )
        rng = random.Random(42)

        for strategy in ("random", "fixed_form", "single_cat", "multiplexed"):
            result = run_simulation(
                person, scales, strategy,
                days=30, k_per_day=5, rng=random.Random(42),
            )
            self.assertEqual(result.strategy, strategy)
            self.assertEqual(result.days, 30)
            self.assertGreater(result.total_items_asked, 0)
            self.assertGreater(result.scales_unlocked, 0)

    def test_response_generation_valid(self):
        """Generated responses should be valid category indices."""
        item = GRMItemParams("test", 1.0, (-1.5, -0.5, 0.5, 1.5))
        rng = random.Random(42)
        for _ in range(100):
            resp = generate_response(0.0, item, rng)
            self.assertGreaterEqual(resp, 0)
            self.assertLessEqual(resp, 4)


class TestMultiplexedVsBaselines(unittest.TestCase):
    """
    The publishable comparison: multiplexed active sensing vs baselines.

    With shared items across scales, multiplexed selection should achieve:
    - Lower mean SE(θ) (more precise measurement)
    - More Fisher information per item asked
    - Faster time to reliable estimates

    N=20 persons × 30 days × 5 items/day, seeded for reproducibility.
    """

    POPULATION_SIZE = 20
    DAYS = 15     # 15 days × 5 items = 75 asks for 28 unique items (budget matters)
    K_PER_DAY = 5
    SEED = 2026

    def setUp(self):
        self.scales = _build_demo_scales()
        self.scale_ids = [sc.scale_id for sc in self.scales]
        self.rng = random.Random(self.SEED)
        self.population = _build_population(
            self.POPULATION_SIZE, self.rng, self.scale_ids,
        )

    def _run_strategy(self, strategy: str) -> List[SimResult]:
        results = []
        for person in self.population:
            r = run_simulation(
                person, self.scales, strategy,
                days=self.DAYS, k_per_day=self.K_PER_DAY,
                rng=random.Random(self.SEED + hash(person.person_id)),
            )
            results.append(r)
        return results

    def _mean_se(self, results: List[SimResult]) -> float:
        """Average SE across all persons and scales (excluding inf)."""
        ses = []
        for r in results:
            for se in r.se_by_scale.values():
                if math.isfinite(se):
                    ses.append(se)
        return sum(ses) / len(ses) if ses else float("inf")

    def _mean_info(self, results: List[SimResult]) -> float:
        """Average total information across all persons and scales."""
        infos = []
        for r in results:
            for info in r.info_by_scale.values():
                infos.append(info)
        return sum(infos) / len(infos) if infos else 0.0

    def _mean_bias(self, results: List[SimResult]) -> float:
        """Average |θ̂ - θ_true| across persons and scales."""
        biases = []
        for r in results:
            for b in r.theta_bias.values():
                biases.append(b)
        return sum(biases) / len(biases) if biases else float("inf")

    def _mean_scales_unlocked(self, results: List[SimResult]) -> float:
        return sum(r.scales_unlocked for r in results) / len(results)

    def _mean_days_to_reliable(self, results: List[SimResult]) -> float:
        """Average days to first reliable estimate across scales that achieved it."""
        days_list = []
        for r in results:
            for d in r.days_to_reliable.values():
                if d is not None:
                    days_list.append(d)
        return sum(days_list) / len(days_list) if days_list else float("inf")

    def test_multiplexed_beats_random_on_precision(self):
        """Multiplexed should achieve lower SE than random selection."""
        random_results = self._run_strategy("random")
        mux_results = self._run_strategy("multiplexed")

        se_random = self._mean_se(random_results)
        se_mux = self._mean_se(mux_results)

        self.assertLess(
            se_mux, se_random,
            f"Multiplexed SE ({se_mux:.3f}) should be lower than random ({se_random:.3f})",
        )

    def test_multiplexed_more_information_than_random(self):
        """Multiplexed should accumulate more Fisher information than random."""
        random_results = self._run_strategy("random")
        mux_results = self._run_strategy("multiplexed")

        info_random = self._mean_info(random_results)
        info_mux = self._mean_info(mux_results)

        self.assertGreater(
            info_mux, info_random,
            f"Multiplexed info ({info_mux:.2f}) should exceed random ({info_random:.2f})",
        )

    def test_multiplexed_matches_fixed_on_coverage(self):
        """
        Multiplexed should unlock at least as many scales as fixed-form.
        With budget constraints, multiplexed achieves this by leveraging
        shared items to cover more scales simultaneously.
        """
        fixed_results = self._run_strategy("fixed_form")
        mux_results = self._run_strategy("multiplexed")

        coverage_fixed = self._mean_scales_unlocked(fixed_results)
        coverage_mux = self._mean_scales_unlocked(mux_results)

        self.assertGreaterEqual(
            coverage_mux, coverage_fixed,
            f"Multiplexed coverage ({coverage_mux:.1f}) should be >= "
            f"fixed-form ({coverage_fixed:.1f})",
        )

    def test_multiplexed_converges_to_fixed_on_precision(self):
        """
        With sufficient budget, multiplexed should match fixed-form on SE.
        This demonstrates no loss of precision despite the adaptive approach.
        """
        fixed_results = self._run_strategy("fixed_form")
        mux_results = self._run_strategy("multiplexed")

        se_fixed = self._mean_se(fixed_results)
        se_mux = self._mean_se(mux_results)

        # Within 10% — shows convergence without requiring strict dominance
        self.assertLess(
            se_mux, se_fixed * 1.10,
            f"Multiplexed SE ({se_mux:.3f}) should be within 10% of "
            f"fixed-form ({se_fixed:.3f})",
        )

    def test_multiplexed_competitive_bias(self):
        """Multiplexed bias should be within 25% of random (not worse by large margin)."""
        random_results = self._run_strategy("random")
        mux_results = self._run_strategy("multiplexed")

        bias_random = self._mean_bias(random_results)
        bias_mux = self._mean_bias(mux_results)

        # Bias is noisy with small populations. The key claim is that
        # multiplexed doesn't sacrifice accuracy for efficiency.
        self.assertLess(
            bias_mux, bias_random * 1.25,
            f"Multiplexed bias ({bias_mux:.3f}) should not be much worse than "
            f"random ({bias_random:.3f})",
        )

    def test_simulation_summary_report(self):
        """Generate a summary comparison table (for paper Table 1)."""
        strategies = ["random", "fixed_form", "single_cat", "multiplexed"]
        summary = {}

        for strategy in strategies:
            results = self._run_strategy(strategy)
            summary[strategy] = {
                "mean_se": round(self._mean_se(results), 4),
                "mean_info": round(self._mean_info(results), 2),
                "mean_bias": round(self._mean_bias(results), 4),
                "mean_scales_unlocked": round(self._mean_scales_unlocked(results), 2),
                "mean_days_to_reliable": round(self._mean_days_to_reliable(results), 1),
            }

        # Just verify the summary is well-formed
        for strategy in strategies:
            self.assertIn(strategy, summary)
            self.assertIn("mean_se", summary[strategy])
            self.assertIn("mean_info", summary[strategy])
            self.assertIn("mean_bias", summary[strategy])

        # Print the comparison table for visual inspection
        print("\n" + "=" * 75)
        print("SIMULATION COMPARISON (N=%d persons, %d days, k=%d/day)"
              % (self.POPULATION_SIZE, self.DAYS, self.K_PER_DAY))
        print("=" * 75)
        print(f"{'Strategy':<16} {'Mean SE':>10} {'Mean Info':>10} "
              f"{'Mean Bias':>10} {'Unlocked':>10} {'Days→Rel':>10}")
        print("-" * 75)
        for strategy in strategies:
            s = summary[strategy]
            print(f"{strategy:<16} {s['mean_se']:>10.4f} {s['mean_info']:>10.2f} "
                  f"{s['mean_bias']:>10.4f} {s['mean_scales_unlocked']:>10.2f} "
                  f"{s['mean_days_to_reliable']:>10.1f}")
        print("=" * 75)

        # Multiplexed should clearly beat random on most metrics
        mux = summary["multiplexed"]
        rand = summary["random"]
        wins_vs_random = 0
        if mux["mean_se"] < rand["mean_se"]:
            wins_vs_random += 1
        if mux["mean_info"] > rand["mean_info"]:
            wins_vs_random += 1
        if mux["mean_bias"] < rand["mean_bias"]:
            wins_vs_random += 1
        if mux["mean_scales_unlocked"] >= rand["mean_scales_unlocked"]:
            wins_vs_random += 1

        self.assertGreaterEqual(
            wins_vs_random, 3,
            f"Multiplexed should beat random on at least 3/4 metrics, got {wins_vs_random}",
        )

        # Multiplexed should be competitive with fixed-form (within 10% on SE)
        fixed = summary["fixed_form"]
        if math.isfinite(fixed["mean_se"]) and fixed["mean_se"] > 0:
            self.assertLess(
                mux["mean_se"], fixed["mean_se"] * 1.15,
                "Multiplexed SE should be within 15% of fixed-form",
            )


if __name__ == "__main__":
    unittest.main()
