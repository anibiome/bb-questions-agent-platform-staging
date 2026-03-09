import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from questions_agent_platform.pipeline.irt import (
    IRTScoreResult,
    categories_for_response_type,
    default_scale_params,
    score_scale_irt,
)
from questions_agent_platform.pipeline.registry import Scale
from questions_agent_platform.pipeline.response_types import get_response_type

logger = logging.getLogger("questions_agent.scoring")


@dataclass(frozen=True)
class ScaleScoreResult:
    raw_score: float
    normalized_score: float
    answered_count: int
    items_required: int
    risk_tier: Optional[str] = None
    confidence: Optional[str] = None          # high / medium / low
    original_metric_label: Optional[str] = None  # e.g. "FINDRISC (0-26)"
    original_metric_value: Optional[float] = None # e.g. 18.0 (clinically meaningful)
    # IRT measurement layer (GRM)
    theta: Optional[float] = None             # latent trait estimate
    se_theta: Optional[float] = None          # standard error of θ
    irt_reliability: Optional[float] = None   # marginal reliability (1 - SE²/σ²)
    information: Optional[float] = None       # total Fisher information at θ̂
    # Behavioural metadata uncertainty adjustment (Claim Family 5)
    se_theta_adjusted: Optional[float] = None         # SE after metadata modifier
    uncertainty_multiplier: Optional[float] = None     # session multiplier applied
    engagement_quality: Optional[str] = None           # focused | normal | distracted | fatigued


CARDIO_METHOD_FINDRISC = "instrument_findrisc"
CARDIO_METHOD_EZ_CVD = "instrument_ez_cvd"
CARDIO_METHOD_SCORED = "instrument_scored"
CARDIO_METHOD_LEE_NAFLD = "instrument_lee_nafld"
CARDIO_METHOD_IPAQ_SF = "instrument_ipaq_sf"
CARDIO_METHOD_AUDIT_C = "instrument_audit_c"

CARDIO_METHODS = {
    CARDIO_METHOD_FINDRISC,
    CARDIO_METHOD_EZ_CVD,
    CARDIO_METHOD_SCORED,
    CARDIO_METHOD_LEE_NAFLD,
    CARDIO_METHOD_IPAQ_SF,
    CARDIO_METHOD_AUDIT_C,
}

# ---- Vitality battery scoring methods ----
VITALITY_METHOD_WHO5 = "instrument_who5"
VITALITY_METHOD_SF36_VT = "instrument_sf36_vt"
VITALITY_METHOD_PROMIS_FATIGUE_7A = "instrument_promis_fatigue_7a"

VITALITY_METHODS = {
    VITALITY_METHOD_WHO5,
    VITALITY_METHOD_SF36_VT,
    VITALITY_METHOD_PROMIS_FATIGUE_7A,
}

# WHO-5 item IDs
ITEM_WHO5_1 = "vit_who5_cheerful"
ITEM_WHO5_2 = "vit_who5_calm"
ITEM_WHO5_3 = "vit_who5_active"
ITEM_WHO5_4 = "vit_who5_rested"
ITEM_WHO5_5 = "vit_who5_interesting"

# SF-36 Vitality subscale item IDs (RAND v1.0: items 9a, 9e, 9g, 9i)
ITEM_SF36_VT_PEP = "vit_sf36_pep"          # 9a: full of pep (positive)
ITEM_SF36_VT_ENERGY = "vit_sf36_energy"     # 9e: a lot of energy (positive)
ITEM_SF36_VT_WORN = "vit_sf36_worn_out"     # 9g: worn out (negative, recode)
ITEM_SF36_VT_TIRED = "vit_sf36_tired"       # 9i: tired (negative, recode)

# SVS item IDs (6-item version, omitting reverse item 2)
ITEM_SVS_1 = "vit_svs_alive"
ITEM_SVS_3 = "vit_svs_burst"
ITEM_SVS_4 = "vit_svs_spirit"
ITEM_SVS_5 = "vit_svs_forward"
ITEM_SVS_6 = "vit_svs_alert"
ITEM_SVS_7 = "vit_svs_energized"

# PROMIS Fatigue 7a item IDs
ITEM_PF7_TIRED = "vit_pf7_tired"
ITEM_PF7_EXHAUSTION = "vit_pf7_exhaustion"
ITEM_PF7_RUN_OUT = "vit_pf7_run_out"
ITEM_PF7_LIMIT_WORK = "vit_pf7_limit_work"
ITEM_PF7_THINK = "vit_pf7_think"
ITEM_PF7_BATH = "vit_pf7_bath"
ITEM_PF7_EXERCISE = "vit_pf7_exercise"

ITEM_AGE_BAND = "cm_age_band"
ITEM_SEX = "cm_sex_at_birth"
ITEM_BMI_BAND = "cm_bmi_band"
ITEM_WAIST_BAND = "cm_waist_band"
ITEM_ACTIVITY_DAILY = "cm_activity_daily"
ITEM_FRUIT_VEG_DAILY = "cm_fruit_veg_daily"
ITEM_BP_MED = "cm_bp_medication"
ITEM_HIGH_GLUCOSE = "cm_high_glucose_history"
ITEM_FAMILY_DIABETES = "cm_family_diabetes_history"
ITEM_SMOKING = "cm_smoking_current"
ITEM_FAMILY_MI = "cm_family_mi_history"
ITEM_ANEMIA = "cm_anemia_history"
ITEM_CVD = "cm_cvd_history"
ITEM_CHF = "cm_chf_history"
ITEM_PVD = "cm_pvd_history"
ITEM_PROTEINURIA = "cm_proteinuria_history"
ITEM_DYSLIPIDEMIA = "cm_dyslipidemia_history"
ITEM_MENOPAUSE = "cm_menopause_status"
ITEM_IPAQ_VIG_DAYS = "cm_ipaq_vig_days"
ITEM_IPAQ_VIG_MIN = "cm_ipaq_vig_minutes"
ITEM_IPAQ_MOD_DAYS = "cm_ipaq_mod_days"
ITEM_IPAQ_MOD_MIN = "cm_ipaq_mod_minutes"
ITEM_IPAQ_WALK_DAYS = "cm_ipaq_walk_days"
ITEM_IPAQ_WALK_MIN = "cm_ipaq_walk_minutes"
ITEM_IPAQ_SIT_MIN = "cm_ipaq_sit_minutes"
ITEM_ALCOHOL_FREQ = "cm_alcohol_intake_frequency"
ITEM_AUDIT_TYPICAL = "cm_audit_typical_drinks"
ITEM_AUDIT_BINGE = "cm_audit_binge_frequency"


def compute_scale_score(
    scale: Scale,
    latest_answers_by_item_id: Dict[str, float],
) -> Optional[ScaleScoreResult]:
    """
    Computes the score for a single scale using the latest answer per item.

    Returns None when the scale cannot be computed due to insufficient items.
    """
    method = str(scale.method or "").strip().lower()
    scorer = _CUSTOM_SCORERS.get(method)
    if scorer is not None:
        return scorer(scale, latest_answers_by_item_id)
    if method not in ("sum", "mean"):
        raise ValueError(f"Unsupported scoring method: {scale.method}")
    return _compute_generic_scale_score(scale, latest_answers_by_item_id)


def _compute_generic_scale_score(
    scale: Scale,
    latest_answers_by_item_id: Dict[str, float],
) -> Optional[ScaleScoreResult]:
    response_type = get_response_type(scale.response_type)

    weighted_values = []
    weights = []
    for si in scale.items:
        if si.item_id not in latest_answers_by_item_id:
            continue
        v = float(latest_answers_by_item_id[si.item_id])
        v = max(response_type.min_value, min(response_type.max_value, v))
        if si.reverse:
            v = response_type.min_value + response_type.max_value - v
        w = float(si.weight)
        if w <= 0:
            continue
        weighted_values.append(v * w)
        weights.append(w)

    answered_count = len(weights)
    if answered_count < scale.min_items_required:
        return None

    method = str(scale.method or "").strip().lower()
    if method == "sum":
        raw = sum(weighted_values)
    elif method == "mean":
        raw = sum(weighted_values) / (sum(weights) or 1.0)
    else:
        raise ValueError(f"Unsupported scoring method: {scale.method}")

    normalized = _normalize_0_100(raw=raw, min_v=scale.normalize_min, max_v=scale.normalize_max)

    # --- IRT measurement layer (GRM) ---
    irt = _compute_irt_for_scale(scale, latest_answers_by_item_id)

    return ScaleScoreResult(
        raw_score=float(raw),
        normalized_score=float(normalized),
        answered_count=answered_count,
        items_required=len(scale.items),
        confidence=confidence_tier(answered_count, len(scale.items), scale.min_items_required),
        original_metric_label=f"{scale.name} ({scale.normalize_min:.0f}-{scale.normalize_max:.0f})",
        original_metric_value=float(raw),
        theta=irt.theta if irt else None,
        se_theta=irt.se_theta if irt else None,
        irt_reliability=irt.reliability if irt else None,
        information=irt.information if irt else None,
    )


def compute_scale_progress(scale: Scale, answered_item_ids_in_window: Tuple[str, ...]) -> Dict[str, int]:
    answered_set = set(answered_item_ids_in_window)
    required_items = {si.item_id for si in scale.items}
    answered_count = len(required_items & answered_set)
    missing = max(0, scale.min_items_required - answered_count)
    return {
        "answered_count": answered_count,
        "missing_count": missing,
        "items_required": len(scale.items),
        "min_items_required": scale.min_items_required,
    }


def confidence_tier(answered_count: int, items_required: int, min_items_required: int) -> str:
    if answered_count >= items_required and items_required > 0:
        return "high"
    if answered_count >= max(1, min_items_required):
        return "medium"
    return "low"


def infer_risk_tier_from_scale(scale: Scale, *, raw_score: float) -> Optional[str]:
    method = str(scale.method or "").strip().lower()
    if method == CARDIO_METHOD_FINDRISC:
        return _findrisc_tier(raw_score)
    if method == CARDIO_METHOD_EZ_CVD:
        return _ez_cvd_tier(raw_score)
    if method == CARDIO_METHOD_SCORED:
        return _scored_tier(raw_score)
    if method == CARDIO_METHOD_LEE_NAFLD:
        return _lee_nafld_tier(raw_score)
    if method == CARDIO_METHOD_IPAQ_SF:
        return _ipaq_tier(raw_score)
    if method == CARDIO_METHOD_AUDIT_C:
        return _audit_c_tier(raw_score, male=None)
    if method == VITALITY_METHOD_WHO5:
        return _who5_tier(raw_score)
    if method == VITALITY_METHOD_SF36_VT:
        return _sf36_vt_tier(raw_score)
    if method == VITALITY_METHOD_PROMIS_FATIGUE_7A:
        return _promis_fatigue_tier(raw_score)
    return None


def _normalize_0_100(*, raw: float, min_v: float, max_v: float) -> float:
    denom = (float(max_v) - float(min_v)) or 1.0
    normalized = (float(raw) - float(min_v)) * (100.0 / denom)
    return max(0.0, min(100.0, float(normalized)))


def _get_value(latest: Dict[str, float], item_id: str) -> Optional[float]:
    if item_id not in latest:
        return None
    try:
        return float(latest[item_id])
    except Exception:
        return None


def _bool01(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    return 1 if float(value) >= 0.5 else 0


def _score_findrisc(scale: Scale, latest: Dict[str, float]) -> Optional[ScaleScoreResult]:
    age = _get_value(latest, ITEM_AGE_BAND)
    bmi = _get_value(latest, ITEM_BMI_BAND)
    waist = _get_value(latest, ITEM_WAIST_BAND)
    activity = _derive_activity_daily_binary(latest)
    fruit_veg = _bool01(_get_value(latest, ITEM_FRUIT_VEG_DAILY))
    bp_med = _bool01(_get_value(latest, ITEM_BP_MED))
    glucose = _bool01(_get_value(latest, ITEM_HIGH_GLUCOSE))
    family = _get_value(latest, ITEM_FAMILY_DIABETES)

    answered_count = sum(
        1
        for v in (age, bmi, waist, activity, fruit_veg, bp_med, glucose, family)
        if v is not None
    )
    if answered_count < int(scale.min_items_required):
        return None
    raw = float(
        (age or 0.0)
        + (bmi or 0.0)
        + (waist or 0.0)
        + (0.0 if int(activity or 0) == 1 else 2.0)
        + (0.0 if int(fruit_veg or 0) == 1 else 1.0)
        + (2.0 if int(bp_med or 0) == 1 else 0.0)
        + (5.0 if int(glucose or 0) == 1 else 0.0)
        + (family or 0.0)
    )
    return ScaleScoreResult(
        raw_score=raw,
        normalized_score=_normalize_0_100(raw=raw, min_v=scale.normalize_min, max_v=scale.normalize_max),
        answered_count=int(answered_count),
        items_required=8,
        risk_tier=_findrisc_tier(raw),
        confidence=confidence_tier(int(answered_count), 8, scale.min_items_required),
        original_metric_label="FINDRISC (0-26)",
        original_metric_value=raw,
    )


def _findrisc_tier(raw: float) -> str:
    if raw < 7:
        return "low"
    if raw <= 11:
        return "slightly_elevated"
    if raw <= 14:
        return "moderate"
    if raw <= 20:
        return "high"
    return "very_high"


def _score_ez_cvd(scale: Scale, latest: Dict[str, float]) -> Optional[ScaleScoreResult]:
    age_band = _get_value(latest, ITEM_AGE_BAND)
    sex = _get_value(latest, ITEM_SEX)
    smoking = _bool01(_get_value(latest, ITEM_SMOKING))
    diabetes = _bool01(_get_value(latest, ITEM_HIGH_GLUCOSE))
    hypertension = _bool01(_get_value(latest, ITEM_BP_MED))
    family_mi = _bool01(_get_value(latest, ITEM_FAMILY_MI))

    age_factor_answered = age_band is not None and sex is not None
    age_factor = 0.0
    if age_factor_answered:
        male = bool(_bool01(sex))
        threshold_band = 2.0 if male else 3.0
        age_factor = 1.0 if float(age_band) >= threshold_band else 0.0

    raw = float(age_factor)
    answered_count = 1 if age_factor_answered else 0
    for v in (sex, smoking, diabetes, hypertension, family_mi):
        if v is None:
            continue
        raw += float(_bool01(v) or 0)
        answered_count += 1

    if answered_count < int(scale.min_items_required):
        return None

    return ScaleScoreResult(
        raw_score=raw,
        normalized_score=_normalize_0_100(raw=raw, min_v=scale.normalize_min, max_v=scale.normalize_max),
        answered_count=int(answered_count),
        items_required=6,
        risk_tier=_ez_cvd_tier(raw),
        confidence=confidence_tier(int(answered_count), 6, scale.min_items_required),
        original_metric_label="EZ-CVD (0-6)",
        original_metric_value=raw,
    )


def _ez_cvd_tier(raw: float) -> str:
    if raw <= 1:
        return "low"
    if raw <= 3:
        return "moderate"
    return "high"


def _score_scored(scale: Scale, latest: Dict[str, float]) -> Optional[ScaleScoreResult]:
    age_band = _get_value(latest, ITEM_AGE_BAND)
    sex = _get_value(latest, ITEM_SEX)
    anemia = _bool01(_get_value(latest, ITEM_ANEMIA))
    hypertension = _bool01(_get_value(latest, ITEM_BP_MED))
    diabetes = _bool01(_get_value(latest, ITEM_HIGH_GLUCOSE))
    cvd = _bool01(_get_value(latest, ITEM_CVD))
    chf = _bool01(_get_value(latest, ITEM_CHF))
    pvd = _bool01(_get_value(latest, ITEM_PVD))
    proteinuria = _bool01(_get_value(latest, ITEM_PROTEINURIA))

    age_points = 0.0
    age_answered = age_band is not None
    if age_answered:
        ab = float(age_band)
        if ab >= 4.0:
            age_points = 4.0
        elif ab >= 3.0:
            age_points = 3.0
        elif ab >= 2.0:
            age_points = 2.0
        else:
            age_points = 0.0

    female_points = 0.0
    sex_answered = sex is not None
    if sex_answered:
        female_points = 1.0 if int(_bool01(sex) or 0) == 0 else 0.0

    raw = age_points + female_points
    answered_count = (1 if age_answered else 0) + (1 if sex_answered else 0)
    for v in (anemia, hypertension, diabetes, cvd, chf, pvd, proteinuria):
        if v is None:
            continue
        raw += float(v)
        answered_count += 1

    if answered_count < int(scale.min_items_required):
        return None

    return ScaleScoreResult(
        raw_score=float(raw),
        normalized_score=_normalize_0_100(raw=raw, min_v=scale.normalize_min, max_v=scale.normalize_max),
        answered_count=int(answered_count),
        items_required=9,
        risk_tier=_scored_tier(raw),
        confidence=confidence_tier(int(answered_count), 9, scale.min_items_required),
        original_metric_label="SCORED (0-9)",
        original_metric_value=float(raw),
    )


def _scored_tier(raw: float) -> str:
    if raw >= 4.0:
        return "positive_screen"
    return "negative_screen"


def _score_lee_nafld(scale: Scale, latest: Dict[str, float]) -> Optional[ScaleScoreResult]:
    age_band = _get_value(latest, ITEM_AGE_BAND)
    sex = _get_value(latest, ITEM_SEX)
    waist = _get_value(latest, ITEM_WAIST_BAND)
    bmi = _get_value(latest, ITEM_BMI_BAND)
    diabetes = _bool01(_get_value(latest, ITEM_HIGH_GLUCOSE))
    dyslipidemia = _bool01(_get_value(latest, ITEM_DYSLIPIDEMIA))
    alcohol = _get_value(latest, ITEM_ALCOHOL_FREQ)
    activity = _derive_activity_daily_binary(latest)
    menopause = _bool01(_get_value(latest, ITEM_MENOPAUSE))

    answered_count = 0
    raw = 0.0

    if age_band is not None:
        raw += max(0.0, float(age_band))
        answered_count += 1
    if sex is not None:
        raw += 1.0 if int(_bool01(sex) or 0) == 1 else 0.0
        answered_count += 1
    if waist is not None:
        raw += max(0.0, float(waist))
        answered_count += 1
    if bmi is not None:
        raw += max(0.0, float(bmi))
        answered_count += 1
    if diabetes is not None:
        raw += float(diabetes) * 2.0
        answered_count += 1
    if dyslipidemia is not None:
        raw += float(dyslipidemia)
        answered_count += 1
    if alcohol is not None:
        raw += max(0.0, min(3.0, float(alcohol)))
        answered_count += 1
    if activity is not None:
        raw += 0.0 if int(activity) == 1 else 1.0
        answered_count += 1
    if menopause is not None:
        raw += float(menopause)
        answered_count += 1

    if answered_count < int(scale.min_items_required):
        return None

    return ScaleScoreResult(
        raw_score=float(raw),
        normalized_score=_normalize_0_100(raw=raw, min_v=scale.normalize_min, max_v=scale.normalize_max),
        answered_count=int(answered_count),
        items_required=9,
        risk_tier=_lee_nafld_tier(raw),
        confidence=confidence_tier(int(answered_count), 9, scale.min_items_required),
        original_metric_label="Lee NAFLD Index",
        original_metric_value=float(raw),
    )


def _lee_nafld_tier(raw: float) -> str:
    if raw >= 8.0:
        return "high_risk"
    if raw >= 6.0:
        return "elevated_risk"
    return "low_risk"


def _score_ipaq_sf(scale: Scale, latest: Dict[str, float]) -> Optional[ScaleScoreResult]:
    item_ids = (
        ITEM_IPAQ_VIG_DAYS,
        ITEM_IPAQ_VIG_MIN,
        ITEM_IPAQ_MOD_DAYS,
        ITEM_IPAQ_MOD_MIN,
        ITEM_IPAQ_WALK_DAYS,
        ITEM_IPAQ_WALK_MIN,
        ITEM_IPAQ_SIT_MIN,
    )
    values = {item_id: _get_value(latest, item_id) for item_id in item_ids}
    answered_count = sum(1 for v in values.values() if v is not None)
    if answered_count < int(scale.min_items_required):
        return None

    vig_days = float(values.get(ITEM_IPAQ_VIG_DAYS) or 0.0)
    vig_min = float(values.get(ITEM_IPAQ_VIG_MIN) or 0.0)
    mod_days = float(values.get(ITEM_IPAQ_MOD_DAYS) or 0.0)
    mod_min = float(values.get(ITEM_IPAQ_MOD_MIN) or 0.0)
    walk_days = float(values.get(ITEM_IPAQ_WALK_DAYS) or 0.0)
    walk_min = float(values.get(ITEM_IPAQ_WALK_MIN) or 0.0)

    raw = (8.0 * vig_days * vig_min) + (4.0 * mod_days * mod_min) + (3.3 * walk_days * walk_min)
    return ScaleScoreResult(
        raw_score=float(raw),
        normalized_score=_normalize_0_100(raw=raw, min_v=scale.normalize_min, max_v=scale.normalize_max),
        answered_count=int(answered_count),
        items_required=len(item_ids),
        risk_tier=_ipaq_tier(raw),
        confidence=confidence_tier(int(answered_count), len(item_ids), scale.min_items_required),
        original_metric_label="IPAQ-SF MET-min/week",
        original_metric_value=float(raw),
    )


def _ipaq_tier(raw: float) -> str:
    if raw < 600.0:
        return "low_activity"
    if raw <= 3000.0:
        return "moderate_activity"
    return "high_activity"


def _score_audit_c(scale: Scale, latest: Dict[str, float]) -> Optional[ScaleScoreResult]:
    main_ids = (ITEM_ALCOHOL_FREQ, ITEM_AUDIT_TYPICAL, ITEM_AUDIT_BINGE)
    main_values = [_get_value(latest, item_id) for item_id in main_ids]
    answered_count = sum(1 for v in main_values if v is not None)
    if answered_count < int(scale.min_items_required):
        return None
    raw = float(sum(v for v in main_values if v is not None))
    sex = _get_value(latest, ITEM_SEX)
    male = None if sex is None else bool(_bool01(sex))
    return ScaleScoreResult(
        raw_score=raw,
        normalized_score=_normalize_0_100(raw=raw, min_v=scale.normalize_min, max_v=scale.normalize_max),
        answered_count=int(answered_count),
        items_required=3,
        risk_tier=_audit_c_tier(raw, male=male),
        confidence=confidence_tier(int(answered_count), 3, scale.min_items_required),
        original_metric_label="AUDIT-C (0-12)",
        original_metric_value=raw,
    )


def _audit_c_tier(raw: float, male: Optional[bool]) -> str:
    threshold = 4.0 if male else 3.0
    if raw <= 0.0:
        return "no_use"
    if raw < threshold:
        return "low_use"
    if raw >= threshold + 3.0:
        return "high_risk"
    return "positive_screen"


def _derive_activity_daily_binary(latest: Dict[str, float]) -> Optional[int]:
    direct = _bool01(_get_value(latest, ITEM_ACTIVITY_DAILY))
    if direct is not None:
        return direct

    vig_days = _get_value(latest, ITEM_IPAQ_VIG_DAYS)
    vig_min = _get_value(latest, ITEM_IPAQ_VIG_MIN)
    mod_days = _get_value(latest, ITEM_IPAQ_MOD_DAYS)
    mod_min = _get_value(latest, ITEM_IPAQ_MOD_MIN)
    walk_days = _get_value(latest, ITEM_IPAQ_WALK_DAYS)
    walk_min = _get_value(latest, ITEM_IPAQ_WALK_MIN)

    components = (vig_days, vig_min, mod_days, mod_min, walk_days, walk_min)
    if all(v is None for v in components):
        return None

    vig = float(vig_days or 0.0) * float(vig_min or 0.0)
    mod = float(mod_days or 0.0) * float(mod_min or 0.0)
    walk = float(walk_days or 0.0) * float(walk_min or 0.0)
    met = (8.0 * vig) + (4.0 * mod) + (3.3 * walk)
    return 1 if met >= 600.0 else 0


# ═══════════════════════════════════════════════════════════════════════════
# Vitality battery scorers
# ═══════════════════════════════════════════════════════════════════════════

def _score_who5(scale: Scale, latest: Dict[str, float]) -> Optional[ScaleScoreResult]:
    """WHO-5 Well-Being Index.

    5 items scored 0-5, sum × 4 → 0-100 percentage scale.
    Higher = better well-being.
    Cutoffs: ≤28 screens for depression, ≤50 low well-being.
    Reference: Topp et al., Psychother Psychosom, 2015.
    """
    item_ids = (ITEM_WHO5_1, ITEM_WHO5_2, ITEM_WHO5_3, ITEM_WHO5_4, ITEM_WHO5_5)
    values = [_get_value(latest, iid) for iid in item_ids]
    answered_count = sum(1 for v in values if v is not None)
    if answered_count < int(scale.min_items_required):
        return None

    raw_sum = sum(max(0.0, min(5.0, float(v))) for v in values if v is not None)
    # Pro-rate if partially answered: scale sum to 5-item equivalent
    if answered_count < 5:
        raw_sum = raw_sum * (5.0 / answered_count)
    percentage = raw_sum * 4.0  # 0-25 → 0-100

    return ScaleScoreResult(
        raw_score=float(raw_sum),
        normalized_score=max(0.0, min(100.0, percentage)),
        answered_count=answered_count,
        items_required=5,
        risk_tier=_who5_tier(percentage),
        confidence=confidence_tier(answered_count, 5, scale.min_items_required),
        original_metric_label="WHO-5 (0-100)",
        original_metric_value=float(percentage),
    )


def _who5_tier(percentage: float) -> str:
    """WHO-5 well-being tiers from validated cutoffs.

    ≤28: likely depression screen (sensitivity 0.93)
    ≤50: low well-being (warrants clinical evaluation)
    >50: adequate well-being
    >72: high well-being (top quartile in healthy populations)
    """
    if percentage <= 28.0:
        return "likely_depression"
    if percentage <= 50.0:
        return "low_wellbeing"
    if percentage <= 72.0:
        return "adequate_wellbeing"
    return "high_wellbeing"


def _score_sf36_vt(scale: Scale, latest: Dict[str, float]) -> Optional[ScaleScoreResult]:
    """SF-36 Vitality subscale (RAND v1.0, items 9a/9e/9g/9i).

    4 items, 6-point scale (1-6). Positive items (pep, energy) recoded
    1→100, 2→80, ..., 6→0. Negative items (worn out, tired) recoded
    1→0, 2→20, ..., 6→100. Average of 4 recoded values → 0-100.
    Higher = more vitality / less fatigue.
    """
    _RECODE_POSITIVE = {1.0: 100.0, 2.0: 80.0, 3.0: 60.0, 4.0: 40.0, 5.0: 20.0, 6.0: 0.0}
    _RECODE_NEGATIVE = {1.0: 0.0, 2.0: 20.0, 3.0: 40.0, 4.0: 60.0, 5.0: 80.0, 6.0: 100.0}

    pep = _get_value(latest, ITEM_SF36_VT_PEP)
    energy = _get_value(latest, ITEM_SF36_VT_ENERGY)
    worn = _get_value(latest, ITEM_SF36_VT_WORN)
    tired = _get_value(latest, ITEM_SF36_VT_TIRED)

    recoded = []
    answered_count = 0
    for v, recode_map in (
        (pep, _RECODE_POSITIVE),
        (energy, _RECODE_POSITIVE),
        (worn, _RECODE_NEGATIVE),
        (tired, _RECODE_NEGATIVE),
    ):
        if v is None:
            continue
        answered_count += 1
        # Snap to nearest valid response (1-6)
        key = max(1.0, min(6.0, round(float(v))))
        recoded.append(recode_map.get(key, 50.0))

    if answered_count < int(scale.min_items_required):
        return None

    vt_score = sum(recoded) / len(recoded)

    return ScaleScoreResult(
        raw_score=float(vt_score),
        normalized_score=max(0.0, min(100.0, vt_score)),
        answered_count=answered_count,
        items_required=4,
        risk_tier=_sf36_vt_tier(vt_score),
        confidence=confidence_tier(answered_count, 4, scale.min_items_required),
        original_metric_label="SF-36 VT (0-100)",
        original_metric_value=float(vt_score),
    )


def _sf36_vt_tier(score: float) -> str:
    """SF-36 VT tiers based on US population norms (mean ≈ 53.7, SD ≈ 15.4).

    No universal clinical cutoff exists; these are norm-referenced:
    <35: severely low vitality (>1 SD below mean)
    35-50: low vitality (below population average)
    50-65: average vitality
    >65: high vitality (above average)
    """
    if score < 35.0:
        return "severely_low"
    if score < 50.0:
        return "low"
    if score <= 65.0:
        return "average"
    return "high"


# PROMIS Fatigue 7a raw-to-T-score lookup table.
# Source: PROMIS Fatigue User Manual (HealthMeasures, Nov 2025).
# Raw score range: 7-35, T-score: IRT-calibrated, mean=50, SD=10 (US reference).
# Higher T-score = MORE fatigue.
_PROMIS_FATIGUE_7A_RAW_TO_T: Dict[int, float] = {
    7: 33.7,
    8: 36.3,
    9: 38.0,
    10: 39.6,
    11: 40.8,
    12: 41.9,
    13: 43.0,
    14: 43.9,
    15: 44.9,
    16: 45.8,
    17: 46.7,
    18: 47.5,
    19: 48.4,
    20: 49.3,
    21: 50.1,
    22: 51.0,
    23: 51.9,
    24: 52.9,
    25: 53.9,
    26: 54.9,
    27: 56.0,
    28: 57.2,
    29: 58.6,
    30: 60.1,
    31: 61.9,
    32: 64.0,
    33: 66.5,
    34: 70.0,
    35: 75.7,
}


def _score_promis_fatigue_7a(scale: Scale, latest: Dict[str, float]) -> Optional[ScaleScoreResult]:
    """PROMIS Fatigue Short Form 7a.

    7 items scored 1-5 (frequency scale). Sum raw scores → lookup T-score.
    T-score: mean=50, SD=10 (US general population reference).
    Higher T-score = more fatigue.
    Reference: HealthMeasures / PROMIS Fatigue User Manual, Nov 2025.
    """
    item_ids = (
        ITEM_PF7_TIRED, ITEM_PF7_EXHAUSTION, ITEM_PF7_RUN_OUT,
        ITEM_PF7_LIMIT_WORK, ITEM_PF7_THINK, ITEM_PF7_BATH, ITEM_PF7_EXERCISE,
    )
    values = [_get_value(latest, iid) for iid in item_ids]
    answered_count = sum(1 for v in values if v is not None)
    if answered_count < int(scale.min_items_required):
        return None

    # Sum raw responses (each 1-5)
    raw_sum = sum(max(1.0, min(5.0, float(v))) for v in values if v is not None)
    # Pro-rate if partially answered
    if answered_count < 7:
        raw_sum = raw_sum * (7.0 / answered_count)
    raw_int = max(7, min(35, int(round(raw_sum))))

    t_score = _PROMIS_FATIGUE_7A_RAW_TO_T.get(raw_int, 50.0)

    # Normalize: T-score 33.7-75.7 → 0-100, but invert so higher = better
    # (Our state dimension energy_vitality is higher=better)
    inverted_normalized = _normalize_0_100(raw=t_score, min_v=75.7, max_v=33.7)

    return ScaleScoreResult(
        raw_score=float(raw_sum),
        normalized_score=float(inverted_normalized),
        answered_count=answered_count,
        items_required=7,
        risk_tier=_promis_fatigue_tier(t_score),
        confidence=confidence_tier(answered_count, 7, scale.min_items_required),
        original_metric_label="PROMIS Fatigue T-score",
        original_metric_value=float(t_score),
    )


def _promis_fatigue_tier(t_score: float) -> str:
    """PROMIS Fatigue severity tiers from HealthMeasures cutpoints.

    ≤55: within normal limits
    55-60: mild fatigue
    60-70: moderate fatigue
    ≥70: severe fatigue
    """
    if t_score < 55.0:
        return "normal"
    if t_score < 60.0:
        return "mild"
    if t_score < 70.0:
        return "moderate"
    return "severe"


def _compute_irt_for_scale(
    scale: Scale,
    latest_answers_by_item_id: Dict[str, float],
) -> Optional[IRTScoreResult]:
    """
    Compute IRT (GRM) theta + SE for a generic scale alongside the classical score.

    Uses default item parameters derived from scale metadata.  In a calibrated
    system these would come from stored item parameters.
    """
    n_cat = categories_for_response_type(scale.response_type)
    scale_items = [(si.item_id, si.weight) for si in scale.items]
    item_params = default_scale_params(scale_items, n_cat, base_discrimination=1.0)

    # Build integer response dict (category indices)
    responses: Dict[str, int] = {}
    rt = get_response_type(scale.response_type)
    for si in scale.items:
        if si.item_id not in latest_answers_by_item_id:
            continue
        v = float(latest_answers_by_item_id[si.item_id])
        v = max(rt.min_value, min(rt.max_value, v))
        if si.reverse:
            v = rt.min_value + rt.max_value - v
        # Map continuous value to nearest category index
        cat_idx = int(round(v - rt.min_value))
        cat_idx = max(0, min(n_cat - 1, cat_idx))
        responses[si.item_id] = cat_idx

    return score_scale_irt(item_params, responses, method="eap")


_CUSTOM_SCORERS: Dict[str, Callable[[Scale, Dict[str, float]], Optional[ScaleScoreResult]]] = {
    CARDIO_METHOD_FINDRISC: _score_findrisc,
    CARDIO_METHOD_EZ_CVD: _score_ez_cvd,
    CARDIO_METHOD_SCORED: _score_scored,
    CARDIO_METHOD_LEE_NAFLD: _score_lee_nafld,
    CARDIO_METHOD_IPAQ_SF: _score_ipaq_sf,
    CARDIO_METHOD_AUDIT_C: _score_audit_c,
    VITALITY_METHOD_WHO5: _score_who5,
    VITALITY_METHOD_SF36_VT: _score_sf36_vt,
    VITALITY_METHOD_PROMIS_FATIGUE_7A: _score_promis_fatigue_7a,
}
