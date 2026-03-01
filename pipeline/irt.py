"""
Item Response Theory (IRT) scoring — Graded Response Model (GRM).

Samejima, F. (1969). Estimation of latent ability using a response pattern
of graded scores. Psychometrika Monograph Supplement, 34(4, Pt. 2), 1-97.

The GRM models polytomous ordered-category items (e.g., Likert scales).
Each item i has:
  - a_i: discrimination parameter (slope, a_i > 0)
  - b_{i,1}, ..., b_{i,K-1}: boundary parameters for K response categories

Response probability for category k given latent trait θ:
  P*(θ, b_j) = 1 / (1 + exp(-a_i (θ - b_j)))       [cumulative boundary]
  P(X_i=k | θ) = P*(θ, b_{i,k}) - P*(θ, b_{i,k+1})  [category probability]
  where P*(θ, b_0) ≡ 1 and P*(θ, b_K) ≡ 0

Theta estimation: Expected A Posteriori (EAP) with rectangular quadrature
and N(0,1) prior — standard in CAT software (Bock & Mislevy, 1982).

Standard error: SE(θ) = 1 / √I(θ) where I(θ) = Σ_i I_i(θ) is the
total Fisher information.

References:
  - Samejima (1969). Psychometrika Monograph No. 17.
  - Bock & Mislevy (1982). Psychometrika 47(1):29-51.
  - Baker & Kim (2004). Item Response Theory, 2nd ed. CRC Press.
  - Choi & Swartz (2009). Applied Psychological Measurement 33(8):619-632.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GRMItemParams:
    """Parameters for one item under the Graded Response Model."""
    item_id: str
    discrimination: float               # a_i > 0 (slope)
    thresholds: Tuple[float, ...]       # b_{i,1},...,b_{i,K-1} ascending


@dataclass(frozen=True)
class IRTScoreResult:
    """IRT-based score for one scale administration."""
    theta: float             # EAP or MLE estimate of latent trait
    se_theta: float          # standard error of θ
    information: float       # total Fisher information at θ̂
    reliability: float       # marginal reliability: 1 - SE²/σ²_prior
    theta_ci_lower: float    # 95% CI lower bound
    theta_ci_upper: float    # 95% CI upper bound
    n_items_used: int        # items with valid responses
    estimation_method: str   # "eap" or "mle"


# ---------------------------------------------------------------------------
# Core GRM functions
# ---------------------------------------------------------------------------

def _logistic(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def grm_cumulative_prob(theta: float, a: float, b: float) -> float:
    """P*(θ) = logistic(a·(θ - b))."""
    return _logistic(a * (theta - b))


def grm_category_probs(theta: float, item: GRMItemParams) -> List[float]:
    """
    P(X=k|θ) for k = 0, ..., K  (K = len(thresholds)).

    Returns K+1 probabilities summing to 1.0 (floored at 1e-15 for stability).
    """
    K = len(item.thresholds)
    # Cumulative: P*_0 = 1, P*_K = 0
    cum = [1.0]
    for b in item.thresholds:
        cum.append(grm_cumulative_prob(theta, item.discrimination, b))
    cum.append(0.0)

    probs = [max(1e-15, cum[k] - cum[k + 1]) for k in range(K + 1)]
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else probs


def grm_item_information(theta: float, item: GRMItemParams) -> float:
    """
    Fisher information for one GRM item at θ.

    I_i(θ) = Σ_k [P'_k(θ)]² / P_k(θ)

    where P'_k is ∂P(X=k|θ)/∂θ.
    """
    K = len(item.thresholds)
    a = item.discrimination

    # Build cumulative probs and their θ-derivatives
    cum_p = [1.0]
    cum_dp = [0.0]
    for b in item.thresholds:
        p_star = grm_cumulative_prob(theta, a, b)
        cum_p.append(p_star)
        cum_dp.append(a * p_star * (1.0 - p_star))
    cum_p.append(0.0)
    cum_dp.append(0.0)

    info = 0.0
    for k in range(K + 1):
        pk = max(1e-15, cum_p[k] - cum_p[k + 1])
        dpk = cum_dp[k] - cum_dp[k + 1]
        info += (dpk * dpk) / pk
    return info


def total_information(
    theta: float,
    items: Sequence[GRMItemParams],
) -> float:
    """Total test information: I(θ) = Σ_i I_i(θ)."""
    return sum(grm_item_information(theta, item) for item in items)


def se_at_theta(theta: float, items: Sequence[GRMItemParams]) -> float:
    """SE(θ) = 1 / √I(θ). Returns inf when information is zero."""
    info = total_information(theta, items)
    if info <= 1e-15:
        return float("inf")
    return 1.0 / math.sqrt(info)


def marginal_reliability(se: float, prior_var: float = 1.0) -> float:
    """
    Marginal reliability: r = 1 - SE² / σ²_prior.

    Analogous to internal-consistency reliability but derived from the
    measurement precision of the IRT model.  Green et al. (1984).
    """
    if prior_var <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (se * se) / prior_var))


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

def grm_log_likelihood(
    theta: float,
    items: Sequence[GRMItemParams],
    responses: Sequence[int],
) -> float:
    """Log-likelihood of a response vector given θ."""
    ll = 0.0
    for item, resp in zip(items, responses):
        probs = grm_category_probs(theta, item)
        k = max(0, min(int(resp), len(probs) - 1))
        ll += math.log(max(1e-15, probs[k]))
    return ll


# ---------------------------------------------------------------------------
# Theta estimation: EAP
# ---------------------------------------------------------------------------

def estimate_theta_eap(
    items: Sequence[GRMItemParams],
    responses: Sequence[int],
    *,
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
    n_quad: int = 61,
    theta_range: Tuple[float, float] = (-4.0, 4.0),
) -> Tuple[float, float]:
    """
    Expected A Posteriori (EAP) estimation of θ.

    Numerical integration over a rectangular grid with N(prior_mean, prior_sd²)
    prior.  Returns (θ̂_EAP, posterior_SD).
    """
    lo, hi = theta_range
    step = (hi - lo) / max(1, n_quad - 1)
    quad_pts = [lo + i * step for i in range(n_quad)]

    # Log-posterior at each quadrature point
    log_posts: List[float] = []
    for q in quad_pts:
        ll = grm_log_likelihood(q, items, responses)
        log_prior = -0.5 * ((q - prior_mean) / prior_sd) ** 2
        log_posts.append(ll + log_prior)

    # Log-sum-exp for numerical stability
    max_lp = max(log_posts)
    weights = [math.exp(lp - max_lp) for lp in log_posts]
    total_w = sum(weights)
    if total_w <= 0:
        return prior_mean, prior_sd

    weights = [w / total_w for w in weights]

    theta_hat = sum(q * w for q, w in zip(quad_pts, weights))
    var_theta = sum(w * (q - theta_hat) ** 2 for q, w in zip(quad_pts, weights))
    posterior_sd = math.sqrt(max(1e-10, var_theta))
    return theta_hat, posterior_sd


# ---------------------------------------------------------------------------
# Theta estimation: MLE (Newton-Raphson)
# ---------------------------------------------------------------------------

def estimate_theta_mle(
    items: Sequence[GRMItemParams],
    responses: Sequence[int],
    *,
    start: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[float, bool]:
    """
    Maximum Likelihood Estimation of θ via Newton-Raphson.

    Returns (θ̂, converged).  Falls back to bounded estimate
    if Newton step diverges.
    """
    theta = start

    for _ in range(max_iter):
        gradient = 0.0
        neg_hessian = 0.0  # we track -H for convenience (H is typically negative)

        for item, resp in zip(items, responses):
            K = len(item.thresholds)
            a = item.discrimination
            k = max(0, min(int(resp), K))

            # Cumulative probs, first and second θ-derivatives
            cum_p = [1.0]
            cum_dp = [0.0]
            cum_d2p = [0.0]
            for b in item.thresholds:
                p_star = grm_cumulative_prob(theta, a, b)
                dp = a * p_star * (1.0 - p_star)
                d2p = a * a * p_star * (1.0 - p_star) * (1.0 - 2.0 * p_star)
                cum_p.append(p_star)
                cum_dp.append(dp)
                cum_d2p.append(d2p)
            cum_p.append(0.0)
            cum_dp.append(0.0)
            cum_d2p.append(0.0)

            pk = max(1e-15, cum_p[k] - cum_p[k + 1])
            dpk = cum_dp[k] - cum_dp[k + 1]
            d2pk = cum_d2p[k] - cum_d2p[k + 1]

            gradient += dpk / pk
            # Hessian = (d2pk*pk - dpk²) / pk²
            neg_hessian += (dpk * dpk) / (pk * pk) - d2pk / pk

        if neg_hessian < 1e-15:
            break

        delta = gradient / neg_hessian
        delta = max(-2.0, min(2.0, delta))  # step bounding
        theta += delta
        theta = max(-5.0, min(5.0, theta))  # θ bounding

        if abs(delta) < tol:
            return theta, True

    return theta, False


# ---------------------------------------------------------------------------
# Full IRT scoring for a scale
# ---------------------------------------------------------------------------

def score_scale_irt(
    item_params: Sequence[GRMItemParams],
    responses_by_item_id: Dict[str, int],
    *,
    method: str = "eap",
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
) -> Optional[IRTScoreResult]:
    """
    Score a complete scale administration using the GRM.

    Parameters
    ----------
    item_params : sequence of GRMItemParams for this scale
    responses_by_item_id : {item_id: category_index} for answered items
    method : "eap" (default, recommended) or "mle"
    prior_mean, prior_sd : prior for EAP estimation

    Returns None if fewer than 2 items were answered.
    """
    # Match item params to available responses
    used_items: List[GRMItemParams] = []
    used_responses: List[int] = []
    for ip in item_params:
        if ip.item_id in responses_by_item_id:
            used_items.append(ip)
            used_responses.append(int(responses_by_item_id[ip.item_id]))

    if len(used_items) < 2:
        return None

    # Estimate theta
    if method == "mle":
        theta_hat, converged = estimate_theta_mle(used_items, used_responses)
        if not converged:
            # Fall back to EAP
            theta_hat, _ = estimate_theta_eap(
                used_items, used_responses,
                prior_mean=prior_mean, prior_sd=prior_sd,
            )
            method = "eap"
    else:
        theta_hat, _ = estimate_theta_eap(
            used_items, used_responses,
            prior_mean=prior_mean, prior_sd=prior_sd,
        )

    # Compute SE from Fisher information at θ̂
    info = total_information(theta_hat, used_items)
    se = 1.0 / math.sqrt(info) if info > 1e-15 else prior_sd
    rel = marginal_reliability(se, prior_var=prior_sd ** 2)

    return IRTScoreResult(
        theta=round(theta_hat, 4),
        se_theta=round(se, 4),
        information=round(info, 4),
        reliability=round(rel, 4),
        theta_ci_lower=round(theta_hat - 1.96 * se, 4),
        theta_ci_upper=round(theta_hat + 1.96 * se, 4),
        n_items_used=len(used_items),
        estimation_method=method,
    )


# ---------------------------------------------------------------------------
# Default parameter generation from scale metadata
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: Dict[int, Tuple[float, ...]] = {
    # K categories → K-1 thresholds, approximately standard-normal spaced
    2: (0.0,),                                 # binary
    3: (-0.75, 0.75),                          # 3-point
    4: (-1.0, 0.0, 1.0),                       # 4-point
    5: (-1.5, -0.5, 0.5, 1.5),                # 5-point (likert_0_4)
    6: (-1.75, -0.875, 0.0, 0.875, 1.75),     # 6-point
    7: (-2.0, -1.2, -0.4, 0.4, 1.2, 2.0),    # 7-point
    8: (-2.0, -1.4, -0.8, -0.2, 0.4, 1.0, 1.6),  # 8-point
}


def default_item_params(
    item_id: str,
    n_categories: int,
    weight: float = 1.0,
    base_discrimination: float = 1.0,
) -> GRMItemParams:
    """
    Generate plausible default GRM parameters for one item.

    Discrimination is scaled by the item weight in the classical scoring
    formula (higher weight → higher assumed discrimination).

    Thresholds are equally spaced on a standard-normal scale, which produces
    item information functions that peak near θ = 0.

    In a calibrated system these would be estimated from response data.
    """
    n_cat = max(2, min(n_categories, 8))
    thresholds = _DEFAULT_THRESHOLDS.get(n_cat)
    if thresholds is None:
        # Fallback: equally space K-1 thresholds in [-2, 2]
        K = n_cat - 1
        step = 4.0 / max(1, K + 1)
        thresholds = tuple(round(-2.0 + step * (i + 1), 3) for i in range(K))

    a = max(0.2, base_discrimination * weight)
    return GRMItemParams(
        item_id=item_id,
        discrimination=round(a, 3),
        thresholds=thresholds,
    )


def default_scale_params(
    scale_items: Sequence[Tuple[str, float]],
    n_categories: int,
    base_discrimination: float = 1.0,
) -> List[GRMItemParams]:
    """
    Generate default GRM parameters for all items in a scale.

    Parameters
    ----------
    scale_items : [(item_id, weight), ...]
    n_categories : number of response categories (e.g. 5 for likert_0_4)
    base_discrimination : baseline discrimination (1.0 = average)
    """
    return [
        default_item_params(item_id, n_categories, weight, base_discrimination)
        for item_id, weight in scale_items
    ]


# ---------------------------------------------------------------------------
# Utility: convert response_type to category count
# ---------------------------------------------------------------------------

_RESPONSE_TYPE_CATEGORIES: Dict[str, int] = {
    "likert_0_4": 5,
    "likert_0_3": 4,
    "bool_0_1": 2,
    "sex_female_male": 2,
    "age_band_0_4": 5,
    "bmi_band_0_3": 4,
    "waist_points_0_4": 5,
    "family_diabetes_points_0_5": 6,
    "audit_frequency_0_4": 5,
    "audit_typical_0_4": 5,
    "audit_binge_0_4": 5,
    "days_0_7": 8,
    "minutes_0_180": 5,  # treat as ordinal with 5 effective bins
    "minutes_0_960": 5,
}


def categories_for_response_type(response_type: str) -> int:
    """Number of ordinal categories for a given response type."""
    return _RESPONSE_TYPE_CATEGORIES.get(response_type, 5)
