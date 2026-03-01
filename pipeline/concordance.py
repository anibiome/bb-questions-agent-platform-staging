"""
Concordance Study Framework: Adaptive vs Full-Form Validation.

Implements the statistical analysis plan from the clinical validation strategy:
  - Intraclass Correlation Coefficient (ICC) between adaptive and full-form scores
  - Bland-Altman analysis (mean difference + limits of agreement)
  - Cohen's kappa for risk tier classification agreement
  - Score-level per-item agreement

Primary endpoint: ICC(3,1) ≥ 0.85 per instrument.
Secondary endpoints: kappa ≥ 0.80, Bland-Altman limits within clinical range.

This framework is designed to run on paired (adaptive, full-form) score data
collected from the crossover validation study.

References:
  - Shrout & Fleiss (1979). ICC formulas. Psychological Bulletin 86(2):420-428.
  - Bland & Altman (1986). Lancet 327(8476):307-310.
  - Cohen (1960). Kappa statistic. Educational & Psych Measurement 20(1):37-46.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedScore:
    """One paired (adaptive, full-form) score for one participant + instrument."""
    participant_id: str
    instrument_id: str
    timepoint: int             # measurement occasion (0, 1, 2, ...)
    adaptive_score: float       # score from Questions Agent adaptive delivery
    fullform_score: float       # score from standard full-form administration
    adaptive_tier: str          # risk tier from adaptive
    fullform_tier: str          # risk tier from full-form


@dataclass(frozen=True)
class ICCResult:
    """Intraclass Correlation Coefficient result."""
    icc: float                  # ICC(3,1) value
    n_pairs: int                # number of paired observations
    f_value: float              # F-statistic for ICC test
    ci_lower: float             # 95% CI lower bound
    ci_upper: float             # 95% CI upper bound
    passes_threshold: bool      # ICC ≥ threshold (default 0.85)
    threshold: float


@dataclass(frozen=True)
class BlandAltmanResult:
    """Bland-Altman agreement analysis."""
    mean_difference: float           # bias: mean(adaptive - fullform)
    sd_difference: float             # SD of differences
    lower_limit: float               # mean - 1.96 * SD
    upper_limit: float               # mean + 1.96 * SD
    n_outside_limits: int            # count of pairs outside LoA
    proportion_outside: float        # n_outside / n_pairs
    n_pairs: int
    passes_clinical_range: bool      # limits within clinical tolerance


@dataclass(frozen=True)
class KappaResult:
    """Cohen's kappa for categorical agreement."""
    kappa: float                 # kappa statistic
    observed_agreement: float    # P(observed)
    expected_agreement: float    # P(expected by chance)
    n_pairs: int
    categories: Tuple[str, ...]
    passes_threshold: bool       # kappa ≥ threshold (default 0.80)
    threshold: float


@dataclass(frozen=True)
class ConcordanceReport:
    """Full concordance analysis for one instrument."""
    instrument_id: str
    n_participants: int
    n_paired_observations: int
    icc: ICCResult
    bland_altman: BlandAltmanResult
    kappa: KappaResult
    overall_pass: bool           # all three criteria met
    failure_reasons: Tuple[str, ...]


# ---------------------------------------------------------------------------
# ICC(3,1): Two-way mixed, absolute agreement, single measures
# ---------------------------------------------------------------------------

def compute_icc(
    pairs: Sequence[PairedScore],
    *,
    threshold: float = 0.85,
) -> ICCResult:
    """
    Compute ICC(3,1) — two-way mixed, absolute agreement, single measures.

    Uses Shrout & Fleiss (1979) formula for ICC(3,1):
      ICC = (MS_R - MS_E) / (MS_R + (k-1)*MS_E)
    where k=2 (two raters: adaptive and full-form).
    """
    n = len(pairs)
    if n < 3:
        return ICCResult(
            icc=0.0, n_pairs=n, f_value=0.0,
            ci_lower=0.0, ci_upper=0.0,
            passes_threshold=False, threshold=threshold,
        )

    # Extract paired scores
    x = [p.adaptive_score for p in pairs]
    y = [p.fullform_score for p in pairs]
    k = 2  # two methods

    # Grand mean
    grand_mean = sum(x[i] + y[i] for i in range(n)) / (2 * n)

    # Row means (participant)
    row_means = [(x[i] + y[i]) / 2.0 for i in range(n)]

    # Column means (method)
    col_means = [sum(x) / n, sum(y) / n]

    # Sum of squares
    ss_row = k * sum((rm - grand_mean) ** 2 for rm in row_means)
    ss_col = n * sum((cm - grand_mean) ** 2 for cm in col_means)
    ss_total = sum(
        (x[i] - grand_mean) ** 2 + (y[i] - grand_mean) ** 2
        for i in range(n)
    )
    ss_error = ss_total - ss_row - ss_col

    # Mean squares
    df_row = n - 1
    df_error = (n - 1) * (k - 1)
    ms_row = ss_row / max(1, df_row)
    ms_error = ss_error / max(1, df_error)

    # ICC(3,1)
    denom = ms_row + (k - 1) * ms_error
    if denom <= 0:
        icc_val = 0.0
    else:
        icc_val = (ms_row - ms_error) / denom

    icc_val = max(-1.0, min(1.0, icc_val))

    # F-statistic
    f_val = ms_row / max(1e-15, ms_error)

    # Approximate 95% CI using F-distribution approximation
    # (simplified — proper CIs use non-central F, but this is adequate for N≥30)
    fl = max(0, f_val / max(1e-15, _f_critical_approx(n, 0.975)))
    fu = f_val * _f_critical_approx(n, 0.975)
    ci_lower = max(-1.0, (fl - 1) / (fl + k - 1))
    ci_upper = min(1.0, (fu - 1) / (fu + k - 1))

    return ICCResult(
        icc=round(icc_val, 4),
        n_pairs=n,
        f_value=round(f_val, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        passes_threshold=icc_val >= threshold,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Bland-Altman analysis
# ---------------------------------------------------------------------------

def compute_bland_altman(
    pairs: Sequence[PairedScore],
    *,
    clinical_tolerance: Optional[float] = None,
) -> BlandAltmanResult:
    """
    Bland-Altman analysis: mean difference ± 1.96 * SD.

    Parameters
    ----------
    pairs : paired (adaptive, fullform) scores
    clinical_tolerance : if provided, limits of agreement must be within this range
    """
    n = len(pairs)
    if n < 2:
        return BlandAltmanResult(
            mean_difference=0.0, sd_difference=0.0,
            lower_limit=0.0, upper_limit=0.0,
            n_outside_limits=0, proportion_outside=0.0, n_pairs=n,
            passes_clinical_range=True,
        )

    diffs = [p.adaptive_score - p.fullform_score for p in pairs]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / max(1, n - 1)
    sd_diff = math.sqrt(var_diff)

    lower = mean_diff - 1.96 * sd_diff
    upper = mean_diff + 1.96 * sd_diff

    n_outside = sum(1 for d in diffs if d < lower or d > upper)

    passes = True
    if clinical_tolerance is not None:
        passes = (upper - lower) <= clinical_tolerance

    return BlandAltmanResult(
        mean_difference=round(mean_diff, 4),
        sd_difference=round(sd_diff, 4),
        lower_limit=round(lower, 4),
        upper_limit=round(upper, 4),
        n_outside_limits=n_outside,
        proportion_outside=round(n_outside / max(1, n), 4),
        n_pairs=n,
        passes_clinical_range=passes,
    )


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

def compute_kappa(
    pairs: Sequence[PairedScore],
    *,
    threshold: float = 0.80,
) -> KappaResult:
    """
    Cohen's kappa for risk tier classification agreement.

    Uses unweighted kappa (exact categorical agreement).
    """
    n = len(pairs)
    if n < 2:
        return KappaResult(
            kappa=0.0, observed_agreement=0.0, expected_agreement=0.0,
            n_pairs=n, categories=(), passes_threshold=False, threshold=threshold,
        )

    # Collect all categories
    cats = sorted(set(
        [p.adaptive_tier for p in pairs] + [p.fullform_tier for p in pairs]
    ))

    # Build confusion matrix
    matrix: Dict[Tuple[str, str], int] = {}
    for c1 in cats:
        for c2 in cats:
            matrix[(c1, c2)] = 0
    for p in pairs:
        matrix[(p.adaptive_tier, p.fullform_tier)] += 1

    # Observed agreement
    po = sum(matrix.get((c, c), 0) for c in cats) / n

    # Expected agreement (by chance)
    pe = 0.0
    for c in cats:
        row_sum = sum(matrix.get((c, c2), 0) for c2 in cats) / n
        col_sum = sum(matrix.get((c1, c), 0) for c1 in cats) / n
        pe += row_sum * col_sum

    # Kappa
    if abs(1.0 - pe) < 1e-15:
        kappa = 1.0 if po >= 0.99 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    kappa = max(-1.0, min(1.0, kappa))

    return KappaResult(
        kappa=round(kappa, 4),
        observed_agreement=round(po, 4),
        expected_agreement=round(pe, 4),
        n_pairs=n,
        categories=tuple(cats),
        passes_threshold=kappa >= threshold,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Full concordance report
# ---------------------------------------------------------------------------

def generate_concordance_report(
    pairs: Sequence[PairedScore],
    *,
    instrument_id: str,
    icc_threshold: float = 0.85,
    kappa_threshold: float = 0.80,
    clinical_tolerance: Optional[float] = None,
) -> ConcordanceReport:
    """
    Generate a complete concordance analysis report for one instrument.

    Includes ICC, Bland-Altman, and Cohen's kappa analyses.
    """
    participants = set(p.participant_id for p in pairs)

    icc = compute_icc(pairs, threshold=icc_threshold)
    ba = compute_bland_altman(pairs, clinical_tolerance=clinical_tolerance)
    kappa = compute_kappa(pairs, threshold=kappa_threshold)

    failures: List[str] = []
    if not icc.passes_threshold:
        failures.append(f"ICC {icc.icc:.3f} < {icc_threshold}")
    if not ba.passes_clinical_range:
        failures.append(f"Bland-Altman limits ({ba.lower_limit:.2f}, {ba.upper_limit:.2f}) exceed clinical tolerance")
    if not kappa.passes_threshold:
        failures.append(f"Kappa {kappa.kappa:.3f} < {kappa_threshold}")

    return ConcordanceReport(
        instrument_id=instrument_id,
        n_participants=len(participants),
        n_paired_observations=len(pairs),
        icc=icc,
        bland_altman=ba,
        kappa=kappa,
        overall_pass=len(failures) == 0,
        failure_reasons=tuple(failures),
    )


# ---------------------------------------------------------------------------
# Multi-instrument concordance summary
# ---------------------------------------------------------------------------

def generate_multi_instrument_summary(
    reports: Sequence[ConcordanceReport],
) -> Dict[str, Any]:
    """
    Generate a summary across all instruments.

    Used for the concordance paper and FDA submission.
    """
    n_pass = sum(1 for r in reports if r.overall_pass)
    n_fail = len(reports) - n_pass

    return {
        "n_instruments": len(reports),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "all_pass": n_fail == 0,
        "mean_icc": round(sum(r.icc.icc for r in reports) / max(1, len(reports)), 4),
        "min_icc": round(min((r.icc.icc for r in reports), default=0.0), 4),
        "max_icc": round(max((r.icc.icc for r in reports), default=0.0), 4),
        "mean_kappa": round(sum(r.kappa.kappa for r in reports) / max(1, len(reports)), 4),
        "mean_bias": round(
            sum(r.bland_altman.mean_difference for r in reports) / max(1, len(reports)), 4
        ),
        "instruments": {
            r.instrument_id: {
                "icc": r.icc.icc,
                "kappa": r.kappa.kappa,
                "bias": r.bland_altman.mean_difference,
                "pass": r.overall_pass,
                "failures": list(r.failure_reasons),
            }
            for r in reports
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f_critical_approx(n: int, p: float) -> float:
    """Rough F-critical value approximation for ICC CI.

    In a real implementation, use scipy.stats.f.ppf(p, n-1, n-1).
    This approximation is adequate for N ≥ 30.
    """
    # Using the approximation F_{0.975}(df1, df2) ≈ 1 + 2.5/sqrt(df)
    # This is intentionally conservative.
    df = max(1, n - 1)
    if p > 0.5:
        return 1.0 + 2.5 / math.sqrt(df)
    return max(0.01, 1.0 - 2.5 / math.sqrt(df))
