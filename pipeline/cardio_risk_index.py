"""Composite Cardiometabolic Risk Index.

Aggregates 6 validated cardiometabolic instruments into a single benchmark
score (0-1 scale) for comparison with AniFold latent-space findings.

Instruments and evidence-based weights:
  FINDRISC  (0.25)  10-year T2D risk     Lindstrom & Tuomilehto 2003
  EZ-CVD    (0.20)  Non-lab CVD risk     Gaziano et al. 2008
  IPAQ-SF   (0.20)  Physical activity    Craig et al. 2003
  Lee NAFLD (0.15)  NAFLD risk           Lee et al. 2018
  SCORED    (0.10)  CKD screening        Bansal et al. 2007
  AUDIT-C   (0.10)  Alcohol use risk     Bush et al. 1998

The composite is the weighted mean of normalized risk values (0-1) for
all instruments with available data.  Weights are renormalized to sum to
1.0 over the subset of available instruments so that partial coverage
produces a valid index (not artificially deflated).

This score is a *classical risk benchmark* --- a single number that
clinical systems understand --- alongside the richer latent-space
representation from the Coherence Circle metabolic MiniFold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from questions_agent_platform.pipeline.scoring import (
    CARDIO_METHOD_FINDRISC,
    CARDIO_METHOD_EZ_CVD,
    CARDIO_METHOD_IPAQ_SF,
    CARDIO_METHOD_LEE_NAFLD,
    CARDIO_METHOD_SCORED,
    CARDIO_METHOD_AUDIT_C,
    ScaleScoreResult,
)

CARDIO_RISK_INDEX_VERSION = "cardio_risk_v1"

# ---------------------------------------------------------------------------
# Evidence-based weights.  Higher weight = stronger predictive evidence
# for all-cause cardiometabolic morbidity/mortality.
# ---------------------------------------------------------------------------
_INSTRUMENT_WEIGHTS: Dict[str, float] = {
    CARDIO_METHOD_FINDRISC: 0.25,
    CARDIO_METHOD_EZ_CVD:   0.20,
    CARDIO_METHOD_IPAQ_SF:  0.20,
    CARDIO_METHOD_LEE_NAFLD: 0.15,
    CARDIO_METHOD_SCORED:   0.10,
    CARDIO_METHOD_AUDIT_C:  0.10,
}

# For IPAQ-SF, higher normalized_score = MORE activity = LOWER risk.
# All others: higher normalized_score = HIGHER risk.
_INVERTED_INSTRUMENTS = frozenset({CARDIO_METHOD_IPAQ_SF})

ALL_CARDIO_METHODS = frozenset(_INSTRUMENT_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CardioRiskComponent:
    """One instrument's contribution to the composite."""
    method: str
    risk_value: float          # 0-1, higher = worse
    weight: float              # unnormalized instrument weight
    normalized_weight: float   # weight after renormalization
    risk_tier: Optional[str]
    normalized_score: float    # original 0-100 score from instrument
    inverted: bool             # True if risk direction was flipped


@dataclass(frozen=True)
class CardioRiskIndex:
    """Composite cardiometabolic risk index result."""
    composite_risk: float                     # 0-1 weighted mean, higher = worse
    composite_risk_pct: float                 # 0-100 for display
    risk_tier: str                            # low / moderate / elevated / high
    components: Tuple[CardioRiskComponent, ...]
    instruments_available: int
    instruments_total: int
    coverage: float                           # fraction available / total
    confidence: str                           # high / medium / low
    version: str = CARDIO_RISK_INDEX_VERSION


# ---------------------------------------------------------------------------
# Tier classification for the composite
# ---------------------------------------------------------------------------

def _composite_tier(risk: float) -> str:
    """Map composite risk (0-1) to clinical tier.

    Thresholds chosen to roughly align with population risk percentiles:
      <0.25  ≈ lowest quartile   → low
      0.25-0.45                  → moderate
      0.45-0.65                  → elevated
      >0.65  ≈ highest quartile  → high
    """
    if risk < 0.25:
        return "low"
    if risk < 0.45:
        return "moderate"
    if risk < 0.65:
        return "elevated"
    return "high"


def _coverage_confidence(n_available: int, n_total: int) -> str:
    """Confidence in composite based on instrument coverage."""
    ratio = n_available / max(1, n_total)
    if ratio >= 0.8:
        return "high"
    if ratio >= 0.5:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_cardio_risk_index(
    instrument_scores: Dict[str, ScaleScoreResult],
) -> Optional[CardioRiskIndex]:
    """Compute composite cardiometabolic risk from instrument score results.

    Parameters
    ----------
    instrument_scores : dict
        Mapping of scoring method name → ScaleScoreResult.
        Keys must be from CARDIO_METHODS (e.g. "instrument_findrisc").
        Only instruments present in this dict are included.

    Returns
    -------
    CardioRiskIndex or None if no cardiometabolic instruments have scores.
    """
    components: List[CardioRiskComponent] = []
    weight_sum = 0.0
    weighted_risk_sum = 0.0

    for method, base_weight in _INSTRUMENT_WEIGHTS.items():
        score = instrument_scores.get(method)
        if score is None:
            continue

        norm = float(score.normalized_score)
        inverted = method in _INVERTED_INSTRUMENTS

        # Convert normalized 0-100 to risk 0-1 (invert if needed)
        risk_value = (100.0 - norm) / 100.0 if inverted else norm / 100.0
        risk_value = max(0.0, min(1.0, risk_value))

        components.append(CardioRiskComponent(
            method=method,
            risk_value=risk_value,
            weight=base_weight,
            normalized_weight=0.0,  # filled below
            risk_tier=score.risk_tier,
            normalized_score=norm,
            inverted=inverted,
        ))
        weight_sum += base_weight
        weighted_risk_sum += base_weight * risk_value

    if not components or weight_sum == 0.0:
        return None

    composite = weighted_risk_sum / weight_sum

    # Rewrite components with normalized weights
    final_components = tuple(
        CardioRiskComponent(
            method=c.method,
            risk_value=c.risk_value,
            weight=c.weight,
            normalized_weight=round(c.weight / weight_sum, 4),
            risk_tier=c.risk_tier,
            normalized_score=c.normalized_score,
            inverted=c.inverted,
        )
        for c in components
    )

    n_available = len(components)
    n_total = len(_INSTRUMENT_WEIGHTS)

    return CardioRiskIndex(
        composite_risk=round(composite, 6),
        composite_risk_pct=round(composite * 100.0, 2),
        risk_tier=_composite_tier(composite),
        components=final_components,
        instruments_available=n_available,
        instruments_total=n_total,
        coverage=round(n_available / n_total, 4),
        confidence=_coverage_confidence(n_available, n_total),
    )


def cardio_risk_to_dict(index: CardioRiskIndex) -> Dict[str, Any]:
    """Serialize CardioRiskIndex for JSON / API output."""
    return {
        "composite_risk": index.composite_risk,
        "composite_risk_pct": index.composite_risk_pct,
        "risk_tier": index.risk_tier,
        "instruments_available": index.instruments_available,
        "instruments_total": index.instruments_total,
        "coverage": index.coverage,
        "confidence": index.confidence,
        "version": index.version,
        "components": [
            {
                "method": c.method,
                "risk_value": round(c.risk_value, 4),
                "weight": c.weight,
                "normalized_weight": c.normalized_weight,
                "risk_tier": c.risk_tier,
                "normalized_score": c.normalized_score,
                "inverted": c.inverted,
            }
            for c in index.components
        ],
    }
