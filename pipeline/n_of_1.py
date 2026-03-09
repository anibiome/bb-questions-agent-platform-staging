"""N-of-1 Adaptive Calibration Engine.

Transitions IRT measurement from population-level to person-specific using
longitudinal within-person data.  Implements a Kalman filter for dynamic
latent-trait tracking with Bayesian shrinkage toward population priors.

Core references:
  Molenaar, 2004.  A Manifesto on Psychology as Idiographic Science.
    Measurement, 2(4), 201-218.
  Jacobson & Truax, 1991.  Clinical significance.
    J Consulting and Clinical Psychology, 59, 12-19.
  Morgan-Lopez et al., 2022.  IRT-based multilevel RCI.
    Int J Methods Psychiatr Res, 31(2), e1906.
  Choi, Grady & Dodd, 2011.  PSER stopping rule.
    Applied Psychological Measurement.

Architecture
------------
Each (user, scale) pair maintains a PersonalCalibration state that tracks:

  1. Kalman posterior  (theta_personal, se_personal)
     A 1D Kalman filter with process noise proportional to the time gap
     between observations.  Handles non-stationarity gracefully:
       predict:  sigma2_pred = sigma2_post + sigma2_process * delta_t
       update:   K = sigma2_pred / (sigma2_pred + se_obs^2)
                 mu_post = mu_pred + K * (theta_obs - mu_pred)
                 sigma2_post = (1 - K) * sigma2_pred

  2. Baseline estimate  (theta_baseline, se_baseline)
     Established at the end of the warming phase.  The RCI is computed
     against this baseline.

  3. Within-person variability  (within_person_sd)
     Estimated from the trajectory after measurement-error correction:
       sigma2_within = max(0, Var(theta_obs) - mean(SE^2))

  4. Reliable Change Index
     Jacobson-Truax with person-specific SE:
       RCI = delta_theta / se_diff
       se_diff = sqrt(se_personal^2 + se_baseline^2)

  5. Calibration phase
     warming (<10 obs) -> calibrating (10-29) -> calibrated (30+)

  6. Measurement sufficiency
     Can we detect a minimally-important difference (0.5 SD) given
     current precision?

All state is immutable (frozen dataclasses).  Each update returns a new
PersonalCalibration --- pure-functional, no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

N_OF_1_VERSION = "n_of_1_v1"

# ---------------------------------------------------------------------------
# Phase thresholds (literature-backed)
# ---------------------------------------------------------------------------
WARMING_MIN_OBS = 10        # Kirtley et al. 2023: bias < 10% for means
CALIBRATED_MIN_OBS = 30     # Schultzberg & Muthen 2018: satisfactory DSEM
MAX_TRAJECTORY_LENGTH = 60  # rolling window for within-person statistics
MIN_OBS_FOR_WITHIN_SD = 10  # minimum observations for variance correction

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_PROCESS_NOISE_SD = 0.10    # per-day process noise SD (k~0.3 regime)
DEFAULT_POPULATION_MU = 0.0        # standard IRT metric
DEFAULT_POPULATION_SIGMA = 1.0     # standard IRT metric
DEFAULT_MID = 0.5                  # minimally important difference (SD units)
RCI_ALPHA = 0.05                   # significance level for RCI


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PersonalCalibration:
    """Point-in-time calibration state for one (user, scale) pair.

    Immutable.  Each update_calibration() call returns a new instance.
    """
    scale_id: str
    n_observations: int

    # Kalman posterior
    theta_personal: float       # posterior mean
    se_personal: float          # posterior SD = sqrt(sigma2_posterior)

    # Population reference (fixed at initialization)
    theta_population_mu: float
    theta_population_sigma: float

    # Baseline (set at end of warming phase)
    theta_baseline: Optional[float]
    se_baseline: Optional[float]

    # Within-person dynamics
    within_person_sd: Optional[float]       # measurement-error corrected
    trajectory_thetas: Tuple[float, ...]    # recent theta history
    trajectory_ses: Tuple[float, ...]       # recent SE history

    # Calibration metadata
    phase: str                  # "warming" | "calibrating" | "calibrated"
    shrinkage: float            # 0 = pure population, 1 = pure personal
    n_effective: float          # effective sample size

    # Hyperparameters (stored for reproducibility)
    process_noise_sd: float

    version: str = N_OF_1_VERSION


@dataclass(frozen=True)
class ReliableChangeResult:
    """Result of Reliable Change Index assessment."""
    rci: float                  # Jacobson-Truax statistic
    p_value: float              # two-tailed p-value
    significant: bool           # |RCI| > z_crit
    direction: str              # "improved" | "declined" | "stable"
    magnitude: str              # "minimal" | "moderate" | "large"
    delta_theta: float          # theta_personal - theta_baseline
    se_diff: float              # sqrt(se_personal^2 + se_baseline^2)
    mid_exceeded: bool          # |delta| > MID


@dataclass(frozen=True)
class MeasurementSufficiency:
    """Assessment of whether current precision is adequate."""
    se_current: float           # current personal SE
    se_target: float            # target SE for detecting MID
    precision_ratio: float      # se_target / se_current (>1 = sufficient)
    sufficient: bool            # can detect MID-sized change
    reliability: float          # 1 - se^2 (on unit-variance metric)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initial_calibration(
    scale_id: str,
    *,
    population_mu: float = DEFAULT_POPULATION_MU,
    population_sigma: float = DEFAULT_POPULATION_SIGMA,
    process_noise_sd: float = DEFAULT_PROCESS_NOISE_SD,
) -> PersonalCalibration:
    """Create the initial calibration state for a new (user, scale) pair.

    Starts at the population prior with maximum uncertainty.
    """
    return PersonalCalibration(
        scale_id=scale_id,
        n_observations=0,
        theta_personal=population_mu,
        se_personal=population_sigma,
        theta_population_mu=population_mu,
        theta_population_sigma=population_sigma,
        theta_baseline=None,
        se_baseline=None,
        within_person_sd=None,
        trajectory_thetas=(),
        trajectory_ses=(),
        phase="warming",
        shrinkage=0.0,
        n_effective=0.0,
        process_noise_sd=process_noise_sd,
    )


# ---------------------------------------------------------------------------
# Core update (Kalman filter)
# ---------------------------------------------------------------------------

def update_calibration(
    prior: PersonalCalibration,
    theta_observed: float,
    se_observed: float,
    *,
    days_since_last: float = 1.0,
) -> PersonalCalibration:
    """Bayesian update of personal calibration with a new IRT observation.

    Implements a 1D Kalman filter:
      predict:  sigma2_pred = sigma2_post + sigma2_process * delta_t
      update:   K = sigma2_pred / (sigma2_pred + se_obs^2)
                mu_post = mu_pred + K * (theta_obs - mu_pred)
                sigma2_post = (1 - K) * sigma2_pred

    Parameters
    ----------
    prior : PersonalCalibration
        Previous calibration state.
    theta_observed : float
        New IRT theta estimate from scale scoring.
    se_observed : float
        Standard error of the new theta estimate.
    days_since_last : float
        Days since previous observation.  Process noise scales linearly.

    Returns
    -------
    PersonalCalibration
        Updated calibration state (new immutable instance).
    """
    se_obs = max(0.01, float(se_observed))  # floor to prevent division by zero
    delta_t = max(0.0, float(days_since_last))
    n_new = prior.n_observations + 1

    # --- Predict step ---
    sigma2_prior = float(prior.se_personal) ** 2
    sigma2_process = float(prior.process_noise_sd) ** 2 * max(1.0, delta_t)
    sigma2_pred = sigma2_prior + sigma2_process
    mu_pred = float(prior.theta_personal)

    # --- Update step ---
    sigma2_obs = se_obs ** 2
    kalman_gain = sigma2_pred / (sigma2_pred + sigma2_obs)
    mu_post = mu_pred + kalman_gain * (float(theta_observed) - mu_pred)
    sigma2_post = (1.0 - kalman_gain) * sigma2_pred
    se_post = math.sqrt(max(1e-12, sigma2_post))

    # --- Update trajectory (rolling window) ---
    new_thetas = _append_to_trajectory(prior.trajectory_thetas, theta_observed)
    new_ses = _append_to_trajectory(prior.trajectory_ses, se_obs)

    # --- Within-person SD (measurement-error corrected) ---
    within_sd = _estimate_within_person_sd(new_thetas, new_ses)

    # --- Shrinkage: fraction of posterior precision from personal data ---
    pop_precision = 1.0 / max(1e-12, float(prior.theta_population_sigma) ** 2)
    post_precision = 1.0 / max(1e-12, sigma2_post)
    personal_precision = max(0.0, post_precision - pop_precision)
    shrinkage = personal_precision / max(1e-12, post_precision)
    shrinkage = max(0.0, min(1.0, shrinkage))

    # --- Effective sample size ---
    n_effective = float(prior.theta_population_sigma) ** 2 / max(1e-12, sigma2_post)

    # --- Phase transitions ---
    phase = _determine_phase(n_new)

    # --- Baseline establishment ---
    theta_baseline = prior.theta_baseline
    se_baseline = prior.se_baseline
    if theta_baseline is None and n_new >= WARMING_MIN_OBS:
        # Establish baseline at the end of warming phase
        theta_baseline = mu_post
        se_baseline = se_post

    return PersonalCalibration(
        scale_id=prior.scale_id,
        n_observations=n_new,
        theta_personal=round(mu_post, 6),
        se_personal=round(se_post, 6),
        theta_population_mu=prior.theta_population_mu,
        theta_population_sigma=prior.theta_population_sigma,
        theta_baseline=round(theta_baseline, 6) if theta_baseline is not None else None,
        se_baseline=round(se_baseline, 6) if se_baseline is not None else None,
        within_person_sd=round(within_sd, 6) if within_sd is not None else None,
        trajectory_thetas=new_thetas,
        trajectory_ses=new_ses,
        phase=phase,
        shrinkage=round(shrinkage, 6),
        n_effective=round(n_effective, 4),
        process_noise_sd=prior.process_noise_sd,
    )


# ---------------------------------------------------------------------------
# Reliable Change Index
# ---------------------------------------------------------------------------

def assess_reliable_change(
    calibration: PersonalCalibration,
    *,
    mid: float = DEFAULT_MID,
    alpha: float = RCI_ALPHA,
) -> Optional[ReliableChangeResult]:
    """Compute the Reliable Change Index against the personal baseline.

    Returns None if no baseline has been established (warming phase).

    The RCI follows Jacobson & Truax (1991) with person-specific SE:
      RCI = delta_theta / se_diff
      se_diff = sqrt(se_personal^2 + se_baseline^2)

    Significance at alpha (two-tailed).
    """
    if calibration.theta_baseline is None or calibration.se_baseline is None:
        return None

    delta = float(calibration.theta_personal) - float(calibration.theta_baseline)
    se_diff = math.sqrt(
        float(calibration.se_personal) ** 2
        + float(calibration.se_baseline) ** 2
    )
    se_diff = max(1e-12, se_diff)

    rci = delta / se_diff

    # Two-tailed p-value from standard normal.
    # Floor at 1e-6: erfc underflows to 0.0 for very large |z|, and we
    # round to 6 decimal places anyway, so 1e-6 is the smallest
    # representable nonzero value.  Matches common reporting convention
    # of "p < 0.000001".
    p_value = max(2.0 * _normal_sf(abs(rci)), 1e-6)

    z_crit = _z_critical(alpha)
    significant = abs(rci) > z_crit

    # Direction.  Threshold at 0.005 (half a hundredth of an SD) rather
    # than machine-epsilon: the Kalman filter's process noise causes tiny
    # posterior drift even with identical observations, and 0.005 is still
    # 100× below any clinically-meaningful difference.
    if abs(delta) < 0.005:
        direction = "stable"
    elif delta > 0:
        direction = "improved"
    else:
        direction = "declined"

    # Magnitude (Cohen's d-like interpretation)
    abs_delta = abs(delta)
    if abs_delta < 0.2:
        magnitude = "minimal"
    elif abs_delta < 0.5:
        magnitude = "moderate"
    else:
        magnitude = "large"

    return ReliableChangeResult(
        rci=round(rci, 4),
        p_value=round(p_value, 6),
        significant=significant,
        direction=direction,
        magnitude=magnitude,
        delta_theta=round(delta, 6),
        se_diff=round(se_diff, 6),
        mid_exceeded=abs_delta >= mid,
    )


# ---------------------------------------------------------------------------
# Measurement sufficiency
# ---------------------------------------------------------------------------

def assess_measurement_sufficiency(
    calibration: PersonalCalibration,
    *,
    mid: float = DEFAULT_MID,
    alpha: float = RCI_ALPHA,
) -> MeasurementSufficiency:
    """Can we detect a MID-sized change with current measurement precision?

    Target SE = MID / z_critical.  If current SE <= target SE, we have
    sufficient precision to detect a minimally-important difference at
    the given alpha level.

    The reliability metric uses the standard IRT formula:
      reliability = 1 - SE^2  (when population SD = 1)
    Clamped to [0, 1].
    """
    z_crit = _z_critical(alpha)
    se_target = mid / max(0.01, z_crit)
    se_current = float(calibration.se_personal)

    # Precision ratio: >1 means sufficient
    precision_ratio = se_target / max(1e-12, se_current)
    sufficient = se_current <= se_target

    # Reliability on unit-variance metric
    pop_var = float(calibration.theta_population_sigma) ** 2
    reliability = max(0.0, min(1.0, 1.0 - (se_current ** 2 / max(1e-12, pop_var))))

    return MeasurementSufficiency(
        se_current=round(se_current, 6),
        se_target=round(se_target, 6),
        precision_ratio=round(precision_ratio, 4),
        sufficient=sufficient,
        reliability=round(reliability, 6),
    )


# ---------------------------------------------------------------------------
# Within-person variability estimation
# ---------------------------------------------------------------------------

def _estimate_within_person_sd(
    thetas: Tuple[float, ...],
    ses: Tuple[float, ...],
) -> Optional[float]:
    """Estimate within-person SD corrected for measurement error.

    sigma2_within = max(0, Var(theta_obs) - mean(SE^2))

    Returns None if insufficient observations (< MIN_OBS_FOR_WITHIN_SD).
    Reference: Kirtley et al. (2023), Behavior Research Methods.
    """
    n = min(len(thetas), len(ses))
    if n < MIN_OBS_FOR_WITHIN_SD:
        return None

    thetas_f = [float(t) for t in thetas[-n:]]
    ses_f = [float(s) for s in ses[-n:]]

    # Observed variance of theta estimates
    mean_theta = sum(thetas_f) / n
    var_obs = sum((t - mean_theta) ** 2 for t in thetas_f) / max(1, n - 1)

    # Mean measurement-error variance
    mean_se2 = sum(s ** 2 for s in ses_f) / n

    # Corrected within-person variance
    var_within = max(0.0, var_obs - mean_se2)
    return math.sqrt(var_within)


def estimate_within_person_sd(
    thetas: Sequence[float],
    ses: Sequence[float],
) -> Optional[float]:
    """Public API for within-person SD estimation.

    Exposed for use outside the calibration update loop (e.g. reporting).
    """
    return _estimate_within_person_sd(tuple(thetas), tuple(ses))


# ---------------------------------------------------------------------------
# Phase management
# ---------------------------------------------------------------------------

def _determine_phase(n_observations: int) -> str:
    """Determine calibration phase from observation count.

    warming    (<10 obs): population prior dominates
    calibrating (10-29):  personal estimates emerging, baseline established
    calibrated  (30+):    personal model reliable
    """
    if n_observations < WARMING_MIN_OBS:
        return "warming"
    if n_observations < CALIBRATED_MIN_OBS:
        return "calibrating"
    return "calibrated"


# ---------------------------------------------------------------------------
# Trajectory management
# ---------------------------------------------------------------------------

def _append_to_trajectory(
    trajectory: Tuple[float, ...],
    value: float,
) -> Tuple[float, ...]:
    """Append a value to the trajectory, capping at MAX_TRAJECTORY_LENGTH."""
    extended = trajectory + (round(float(value), 6),)
    if len(extended) > MAX_TRAJECTORY_LENGTH:
        extended = extended[-MAX_TRAJECTORY_LENGTH:]
    return extended


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _normal_sf(z: float) -> float:
    """Survival function of the standard normal (1 - CDF).

    Uses the complementary error function for numerical precision.
    """
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def _z_critical(alpha: float) -> float:
    """Two-tailed z critical value: z such that P(|Z| > z) = alpha.

    Directly computes the upper quantile via Abramowitz & Stegun 26.2.23.
    For alpha=0.05: z ≈ 1.96.  For alpha=0.01: z ≈ 2.576.
    Accuracy: ~4.5e-4 (sufficient for RCI significance testing).
    """
    p = float(alpha) / 2.0  # one-tailed probability
    if p <= 0.0:
        return float("inf")
    if p >= 0.5:
        return 0.0
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t ** 2) / (1.0 + d1 * t + d2 * t ** 2 + d3 * t ** 3)


def _erfc_inv(p: float) -> float:
    """Inverse complementary error function: x such that erfc(x) = p.

    Uses A&S 26.2.23 for the probit (inverse normal) as initial guess,
    scaled to the erfc domain (z_probit / sqrt(2)), then refines with
    Newton-Raphson on erfc(x) - p = 0.
    Accurate to ~1e-9 for 0 < p < 2.
    """
    if p <= 0.0:
        return float("inf")
    if p >= 2.0:
        return float("-inf")
    if p >= 1.0:
        return -_erfc_inv(2.0 - p)

    # A&S 26.2.23: initial guess for the probit of p/2
    t = math.sqrt(-2.0 * math.log(p / 2.0))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z_probit = t - (c0 + c1 * t + c2 * t ** 2) / (1.0 + d1 * t + d2 * t ** 2 + d3 * t ** 3)

    # Scale from probit domain to erfc domain: erfc_inv(p) = probit(1-p/2) / sqrt(2)
    x = z_probit / math.sqrt(2.0)

    # Newton-Raphson refinement: erfc'(x) = -2/sqrt(pi) * exp(-x^2)
    for _ in range(3):
        err = math.erfc(x) - p
        deriv = -2.0 / math.sqrt(math.pi) * math.exp(-x * x)
        if abs(deriv) < 1e-30:
            break
        x -= err / deriv

    return x


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def calibration_to_dict(cal: PersonalCalibration) -> Dict[str, Any]:
    """Serialize PersonalCalibration for JSON storage."""
    return {
        "scale_id": cal.scale_id,
        "n_observations": cal.n_observations,
        "theta_personal": cal.theta_personal,
        "se_personal": cal.se_personal,
        "theta_population_mu": cal.theta_population_mu,
        "theta_population_sigma": cal.theta_population_sigma,
        "theta_baseline": cal.theta_baseline,
        "se_baseline": cal.se_baseline,
        "within_person_sd": cal.within_person_sd,
        "trajectory_thetas": list(cal.trajectory_thetas),
        "trajectory_ses": list(cal.trajectory_ses),
        "phase": cal.phase,
        "shrinkage": cal.shrinkage,
        "n_effective": cal.n_effective,
        "process_noise_sd": cal.process_noise_sd,
        "version": cal.version,
    }


def calibration_from_dict(d: Dict[str, Any]) -> PersonalCalibration:
    """Deserialize PersonalCalibration from JSON storage."""
    return PersonalCalibration(
        scale_id=str(d["scale_id"]),
        n_observations=int(d["n_observations"]),
        theta_personal=float(d["theta_personal"]),
        se_personal=float(d["se_personal"]),
        theta_population_mu=float(d.get("theta_population_mu", DEFAULT_POPULATION_MU)),
        theta_population_sigma=float(d.get("theta_population_sigma", DEFAULT_POPULATION_SIGMA)),
        theta_baseline=float(d["theta_baseline"]) if d.get("theta_baseline") is not None else None,
        se_baseline=float(d["se_baseline"]) if d.get("se_baseline") is not None else None,
        within_person_sd=float(d["within_person_sd"]) if d.get("within_person_sd") is not None else None,
        trajectory_thetas=tuple(float(t) for t in d.get("trajectory_thetas", ())),
        trajectory_ses=tuple(float(s) for s in d.get("trajectory_ses", ())),
        phase=str(d.get("phase", "warming")),
        shrinkage=float(d.get("shrinkage", 0.0)),
        n_effective=float(d.get("n_effective", 0.0)),
        process_noise_sd=float(d.get("process_noise_sd", DEFAULT_PROCESS_NOISE_SD)),
        version=str(d.get("version", N_OF_1_VERSION)),
    )
