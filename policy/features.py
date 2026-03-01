from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from questions_agent_platform.policy.types import CandidateItem, PolicyContext


FEATURE_VERSION = "policy_features_v1"


def feature_names_v1() -> List[str]:
    # Keep stable ordering; the policy artifacts store A and b aligned to this list.
    return [
        "bias",
        "deterministic_score",
        "is_anchor",
        "is_unlock_item",
        "is_retest_item",
        "is_drift_probe",
        "multiplex_count",
        "novelty_days",
        "missing_to_unlock",
        "is_due_retest",
        "override_allowed",
        "expected_burden",
        "sensitivity_is_medium",
        # Context summary scalars (do not store full Z in features by default)
        "z_norm",
        "z_velocity_norm",
        "uncertainty_mean",
        "uncertainty_max",
        "distance_to_attractor",
        "completion_rate_7d",
        "completion_rate_14d",
        "completion_rate_30d",
        "burden_ms_median_14d",
        # Day-of-week one-hot (Mon..Sun)
        "dow_0",
        "dow_1",
        "dow_2",
        "dow_3",
        "dow_4",
        "dow_5",
        "dow_6",
    ]


@dataclass(frozen=True)
class FeatureMapping:
    feature_version: str
    names: Tuple[str, ...]
    index: Dict[str, int]


def build_feature_mapping_v1() -> FeatureMapping:
    names = tuple(feature_names_v1())
    return FeatureMapping(
        feature_version=FEATURE_VERSION,
        names=names,
        index={n: i for i, n in enumerate(names)},
    )


def featurize_raw_v1(context: PolicyContext, candidate: CandidateItem, mapping: FeatureMapping) -> List[float]:
    vec = [0.0] * len(mapping.names)

    def setf(name: str, value: float) -> None:
        vec[mapping.index[name]] = float(value)

    setf("bias", 1.0)
    setf("deterministic_score", _clip(candidate.deterministic_score, -10.0, 20.0) / 10.0)

    setf("is_anchor", 1.0 if candidate.item_type == "anchor" else 0.0)
    setf("is_unlock_item", 1.0 if candidate.item_type == "unlock_item" else 0.0)
    setf("is_retest_item", 1.0 if candidate.item_type == "retest_item" else 0.0)
    setf("is_drift_probe", 1.0 if candidate.item_type == "drift_probe" else 0.0)

    multiplex = float(candidate.features.get("multiplex_count") or 0.0)
    setf("multiplex_count", _clip(multiplex, 0.0, 20.0) / 10.0)

    novelty = float(candidate.features.get("novelty_days") or 0.0)
    setf("novelty_days", _clip(novelty, 0.0, 365.0) / 180.0)

    missing = candidate.features.get("missing_to_unlock")
    missing_val = 10.0 if missing is None else float(missing)
    setf("missing_to_unlock", _clip(missing_val, 0.0, 10.0) / 10.0)

    setf("is_due_retest", 1.0 if bool(candidate.features.get("is_due_retest")) else 0.0)
    setf("override_allowed", 1.0 if "override_allowed" in candidate.constraint_tags else 0.0)

    burden = float(candidate.features.get("expected_burden") or 1.0)
    setf("expected_burden", _clip(burden, 0.0, 5.0) / 5.0)

    sensitivity = str(candidate.features.get("sensitivity") or "")
    setf("sensitivity_is_medium", 1.0 if sensitivity == "medium" else 0.0)

    # Context summary
    setf("z_norm", _safe_norm(context.anifold_z))
    setf("z_velocity_norm", _safe_norm(context.z_velocity))
    setf("uncertainty_mean", _safe_mean(context.z_uncertainty_diag))
    setf("uncertainty_max", _safe_max(context.z_uncertainty_diag))
    setf("distance_to_attractor", float(context.z_distance_to_attractor or 0.0))

    setf("completion_rate_7d", float(context.completion_rate_7d or 0.0))
    setf("completion_rate_14d", float(context.completion_rate_14d or 0.0))
    setf("completion_rate_30d", float(context.completion_rate_30d or 0.0))
    setf("burden_ms_median_14d", _clip(float(context.burden_ms_median_14d or 0.0), 0.0, 120_000.0) / 120_000.0)

    dow = context.day_of_week
    if dow is not None and 0 <= int(dow) <= 6:
        setf(f"dow_{int(dow)}", 1.0)

    return vec


def featurize_v1(
    context: PolicyContext,
    candidate: CandidateItem,
    mapping: FeatureMapping,
    *,
    feature_means: Optional[Sequence[float]] = None,
    feature_stds: Optional[Sequence[float]] = None,
) -> List[float]:
    raw = featurize_raw_v1(context, candidate, mapping)
    if not _has_valid_stats(raw, feature_means, feature_stds):
        return raw
    out = [0.0] * len(raw)
    assert feature_means is not None
    assert feature_stds is not None
    for i in range(len(raw)):
        std = float(feature_stds[i])
        if std <= 1e-9:
            out[i] = 0.0
        else:
            out[i] = (float(raw[i]) - float(feature_means[i])) / std
    return out


def top_contributions(mapping: FeatureMapping, weights: Sequence[float], phi: Sequence[float], n: int = 8) -> List[Tuple[str, float]]:
    pairs = []
    for name, idx in mapping.index.items():
        contrib = float(weights[idx]) * float(phi[idx])
        if abs(contrib) < 1e-12:
            continue
        pairs.append((name, contrib))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return pairs[: max(0, int(n))]


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _safe_norm(v: Optional[Sequence[float]]) -> float:
    if not v:
        return 0.0
    return float(math.sqrt(sum(float(x) * float(x) for x in v))) / max(1.0, math.sqrt(len(v)))


def _safe_mean(v: Optional[Sequence[float]]) -> float:
    if not v:
        return 0.0
    return float(sum(float(x) for x in v) / max(1, len(v)))


def _safe_max(v: Optional[Sequence[float]]) -> float:
    if not v:
        return 0.0
    return float(max(float(x) for x in v))


def compute_running_stats(vectors: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if not vectors:
        return (), ()
    d = len(vectors[0])
    n = 0
    means = [0.0] * d
    m2 = [0.0] * d
    for vec in vectors:
        if len(vec) != d:
            continue
        n += 1
        for i in range(d):
            x = float(vec[i])
            delta = x - means[i]
            means[i] += delta / float(n)
            delta2 = x - means[i]
            m2[i] += delta * delta2
    if n <= 1:
        stds = [1.0] * d
    else:
        stds = [math.sqrt(max(1e-12, m2[i] / float(n - 1))) for i in range(d)]
    return tuple(float(x) for x in means), tuple(float(x) for x in stds)


def default_stats(dim: int) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    d = max(0, int(dim))
    return (tuple(0.0 for _ in range(d)), tuple(1.0 for _ in range(d)))


def _has_valid_stats(
    vec: Sequence[float],
    means: Optional[Sequence[float]],
    stds: Optional[Sequence[float]],
) -> bool:
    if means is None or stds is None:
        return False
    return len(means) == len(vec) and len(stds) == len(vec)
