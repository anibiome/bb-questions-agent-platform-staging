from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ResponseType:
    id: str
    min_value: float
    max_value: float
    options: Tuple[Tuple[float, str], ...]


RESPONSE_TYPES: Dict[str, ResponseType] = {
    "likert_0_4": ResponseType(
        id="likert_0_4",
        min_value=0.0,
        max_value=4.0,
        options=(
            (0.0, "Not at all"),
            (1.0, "A little"),
            (2.0, "Somewhat"),
            (3.0, "A lot"),
            (4.0, "Extremely"),
        ),
    ),
    "likert_0_3": ResponseType(
        id="likert_0_3",
        min_value=0.0,
        max_value=3.0,
        options=(
            (0.0, "Not at all"),
            (1.0, "Several days"),
            (2.0, "More than half the days"),
            (3.0, "Nearly every day"),
        ),
    ),
    "bool_0_1": ResponseType(
        id="bool_0_1",
        min_value=0.0,
        max_value=1.0,
        options=((0.0, "No"), (1.0, "Yes")),
    ),
    "sex_female_male": ResponseType(
        id="sex_female_male",
        min_value=0.0,
        max_value=1.0,
        options=((0.0, "Female"), (1.0, "Male")),
    ),
    "age_band_0_4": ResponseType(
        id="age_band_0_4",
        min_value=0.0,
        max_value=4.0,
        options=(
            (0.0, "<45"),
            (1.0, "45-54"),
            (2.0, "55-64"),
            (3.0, "65-74"),
            (4.0, "75+"),
        ),
    ),
    "bmi_band_0_3": ResponseType(
        id="bmi_band_0_3",
        min_value=0.0,
        max_value=3.0,
        options=((0.0, "<25"), (1.0, "25-29.9"), (2.0, "30-34.9"), (3.0, "35+")),
    ),
    "waist_points_0_4": ResponseType(
        id="waist_points_0_4",
        min_value=0.0,
        max_value=4.0,
        options=((0.0, "Low"), (2.0, "Elevated"), (3.0, "High"), (4.0, "Very high")),
    ),
    "family_diabetes_points_0_5": ResponseType(
        id="family_diabetes_points_0_5",
        min_value=0.0,
        max_value=5.0,
        options=((0.0, "None"), (3.0, "Second-degree"), (5.0, "First-degree")),
    ),
    "audit_frequency_0_4": ResponseType(
        id="audit_frequency_0_4",
        min_value=0.0,
        max_value=4.0,
        options=(
            (0.0, "Never"),
            (1.0, "Monthly or less"),
            (2.0, "2-4 times/month"),
            (3.0, "2-3 times/week"),
            (4.0, "4+ times/week"),
        ),
    ),
    "audit_typical_0_4": ResponseType(
        id="audit_typical_0_4",
        min_value=0.0,
        max_value=4.0,
        options=((0.0, "1-2"), (1.0, "3-4"), (2.0, "5-6"), (3.0, "7-9"), (4.0, "10+")),
    ),
    "audit_binge_0_4": ResponseType(
        id="audit_binge_0_4",
        min_value=0.0,
        max_value=4.0,
        options=(
            (0.0, "Never"),
            (1.0, "Less than monthly"),
            (2.0, "Monthly"),
            (3.0, "Weekly"),
            (4.0, "Daily/almost daily"),
        ),
    ),
    "days_0_7": ResponseType(
        id="days_0_7",
        min_value=0.0,
        max_value=7.0,
        options=(
            (0.0, "0 days"),
            (1.0, "1 day"),
            (2.0, "2 days"),
            (3.0, "3 days"),
            (4.0, "4 days"),
            (5.0, "5 days"),
            (6.0, "6 days"),
            (7.0, "7 days"),
        ),
    ),
    "minutes_0_180": ResponseType(
        id="minutes_0_180",
        min_value=0.0,
        max_value=180.0,
        options=(
            (0.0, "0 min"),
            (10.0, "10 min"),
            (20.0, "20 min"),
            (30.0, "30 min"),
            (45.0, "45 min"),
            (60.0, "60 min"),
            (90.0, "90 min"),
            (120.0, "120 min"),
            (180.0, "180 min"),
        ),
    ),
    "minutes_0_960": ResponseType(
        id="minutes_0_960",
        min_value=0.0,
        max_value=960.0,
        options=(
            (0.0, "0 min"),
            (120.0, "2h"),
            (240.0, "4h"),
            (360.0, "6h"),
            (480.0, "8h"),
            (600.0, "10h"),
            (720.0, "12h"),
            (840.0, "14h"),
            (960.0, "16h"),
        ),
    ),
    # ---- Production gut-brain + wellbeing response types (v3) ----
    "likert_agree_0_6": ResponseType(
        id="likert_agree_0_6", min_value=0.0, max_value=6.0,
        options=(
            (0.0, "Strongly disagree"), (1.0, "Somewhat disagree"),
            (2.0, "A little disagree"), (3.0, "Neither agree or disagree"),
            (4.0, "A little agree"), (5.0, "Somewhat agree"), (6.0, "Strongly agree"),
        ),
    ),
    "likert_agree_0_4": ResponseType(
        id="likert_agree_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Strongly disagree"), (1.0, "Disagree"),
            (2.0, "Neither agree nor disagree"), (3.0, "Agree"), (4.0, "Strongly agree"),
        ),
    ),
    "likert_bfi_0_4": ResponseType(
        id="likert_bfi_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Disagree strongly"), (1.0, "Disagree a little"),
            (2.0, "Neutral, no opinion"), (3.0, "Agree a little"), (4.0, "Agree strongly"),
        ),
    ),
    "dass_0_3": ResponseType(
        id="dass_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "Did not apply to me at all"),
            (1.0, "Applied to me to some of the time"),
            (2.0, "Applied to me a good part of the time"),
            (3.0, "Applied to me most of the time"),
        ),
    ),
    "likert_freq_1_7": ResponseType(
        id="likert_freq_1_7", min_value=1.0, max_value=7.0,
        options=(
            (1.0, "Never true"), (2.0, "Very rarely true"), (3.0, "Rarely true"),
            (4.0, "Sometimes true"), (5.0, "Frequently true"),
            (6.0, "Almost always true"), (7.0, "Always true"),
        ),
    ),
    "freq_0_4": ResponseType(
        id="freq_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "None of the time"), (1.0, "A little of the time"),
            (2.0, "Some of the time"), (3.0, "A good bit of the time"),
            (4.0, "All of the time"),
        ),
    ),
    "freq_mhc_0_4": ResponseType(
        id="freq_mhc_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "None of the time"), (1.0, "A little of the time"),
            (2.0, "Some of the time"), (3.0, "Most of the time"), (4.0, "All the time"),
        ),
    ),
    "affect_0_4": ResponseType(
        id="affect_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Very slightly or not at all"), (1.0, "A little"),
            (2.0, "Moderate"), (3.0, "Quite a bit"), (4.0, "Extremely"),
        ),
    ),
    "agree_0_5": ResponseType(
        id="agree_0_5", min_value=0.0, max_value=5.0,
        options=(
            (0.0, "Definitely disagree"), (1.0, "Mostly disagree"),
            (2.0, "Somewhat disagree"), (3.0, "Somewhat agree"),
            (4.0, "Mostly agree"), (5.0, "Definitely agree"),
        ),
    ),
    "freq_never_almost_0_4": ResponseType(
        id="freq_never_almost_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Never"), (1.0, "Rarely"), (2.0, "Sometimes"),
            (3.0, "Often"), (4.0, "Almost always"),
        ),
    ),
    "freq_never_always_0_4": ResponseType(
        id="freq_never_always_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Never"), (1.0, "Seldom"), (2.0, "Occasionally"),
            (3.0, "Usually"), (4.0, "Always"),
        ),
    ),
    "cope_0_3": ResponseType(
        id="cope_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "I usually don't do this at all"),
            (1.0, "I usually do this a little bit"),
            (2.0, "I usually do this a medium amount"),
            (3.0, "I usually do this a lot"),
        ),
    ),
    "coping_efficacy_0_4": ResponseType(
        id="coping_efficacy_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Cannot do at all"), (1.0, "Cannot do most of the time"),
            (2.0, "Can do sometimes"), (3.0, "Moderately certain can do"),
            (4.0, "Certain can do"),
        ),
    ),
    "truth_0_3": ResponseType(
        id="truth_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "Not true at all"), (1.0, "Hardly true"),
            (2.0, "Moderately true"), (3.0, "Completely true"),
        ),
    ),
    "truth_gse_0_3": ResponseType(
        id="truth_gse_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "Not true at all"), (1.0, "Hardly true"),
            (2.0, "Moderately true"), (3.0, "Exactly true"),
        ),
    ),
    "discomfort_0_6": ResponseType(
        id="discomfort_0_6", min_value=0.0, max_value=6.0,
        options=(
            (0.0, "No discomfort at all"), (1.0, "Minor discomfort"),
            (2.0, "Mild discomfort"), (3.0, "Moderate discomfort"),
            (4.0, "Moderately severe discomfort"), (5.0, "Severe discomfort"),
            (6.0, "Very severe discomfort"),
        ),
    ),
    "freq_dqlq_0_5": ResponseType(
        id="freq_dqlq_0_5", min_value=0.0, max_value=5.0,
        options=(
            (0.0, "Never"), (1.0, "Rarely"), (2.0, "Occasionally"),
            (3.0, "Sometimes"), (4.0, "Frequently"), (5.0, "Usually"),
        ),
    ),
    "freq_ibs_0_3": ResponseType(
        id="freq_ibs_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "Never"), (1.0, "Some of the time"),
            (2.0, "Most of the time"), (3.0, "Always"),
        ),
    ),
    "fmi_0_3": ResponseType(
        id="fmi_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "Rarely"), (1.0, "Occasionally"),
            (2.0, "Fairly often"), (3.0, "Almost always"),
        ),
    ),
    "camsr_0_3": ResponseType(
        id="camsr_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "Not at all"), (1.0, "Sometimes"),
            (2.0, "Often"), (3.0, "Almost always"),
        ),
    ),
    "cfq_0_4": ResponseType(
        id="cfq_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Never"), (1.0, "Very rarely"), (2.0, "Occasionally"),
            (3.0, "Quite often"), (4.0, "Very often"),
        ),
    ),
    "rumination_0_3": ResponseType(
        id="rumination_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "Almost never"), (1.0, "Sometimes"),
            (2.0, "Often"), (3.0, "Almost always"),
        ),
    ),
    "scl_0_4": ResponseType(
        id="scl_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Not at all"), (1.0, "A little bit"),
            (2.0, "Moderately"), (3.0, "Quite a bit"), (4.0, "Extremely"),
        ),
    ),
    "sias_0_4": ResponseType(
        id="sias_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Not at all"), (1.0, "Slightly"), (2.0, "Moderately"),
            (3.0, "Very"), (4.0, "Extremely"),
        ),
    ),
    "agree_rses_0_3": ResponseType(
        id="agree_rses_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "Strongly Disagree"), (1.0, "Disagree"),
            (2.0, "Agree"), (3.0, "Strongly Agree"),
        ),
    ),
    "truth_ssq_0_3": ResponseType(
        id="truth_ssq_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "Definitely false"), (1.0, "Probably false"),
            (2.0, "Probably true"), (3.0, "Definitely true"),
        ),
    ),
    "ucla_0_3": ResponseType(
        id="ucla_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "I never feel this way"), (1.0, "I rarely feel this way"),
            (2.0, "I sometimes feel this way"), (3.0, "I often feel this way"),
        ),
    ),
    "mspss_1_6": ResponseType(
        id="mspss_1_6", min_value=0.0, max_value=5.0,
        options=(
            (0.0, "Very Strongly Disagree"), (1.0, "Strongly disagree"),
            (2.0, "Mildly disagree"), (3.0, "Neutral"),
            (4.0, "Mildly agree"), (5.0, "Strongly agree"),
        ),
    ),
    "visceral_0_5": ResponseType(
        id="visceral_0_5", min_value=0.0, max_value=5.0,
        options=(
            (0.0, "Strongly disagree"), (1.0, "Moderately disagree"),
            (2.0, "Mildly disagree"), (3.0, "Mildly agree"),
            (4.0, "Moderately agree"), (5.0, "Strongly agree"),
        ),
    ),
    "harrill_0_4": ResponseType(
        id="harrill_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "I never think, feel or behave this way"),
            (1.0, "I do less than half the time"),
            (2.0, "I do 50% of the time"),
            (3.0, "I do more than half the time"),
            (4.0, "I always think, feel or behave this way"),
        ),
    ),
    "erq_0_4": ResponseType(
        id="erq_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Strongly disagree"), (1.0, "Disagree a little"),
            (2.0, "Neutral, no opinion"), (3.0, "Agree a little"),
            (4.0, "Totally agree"),
        ),
    ),
    "pss_0_4": ResponseType(
        id="pss_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Never"), (1.0, "Almost never"), (2.0, "Sometimes"),
            (3.0, "Fairly often"), (4.0, "Very often"),
        ),
    ),
    "mindful_freq_0_5": ResponseType(
        id="mindful_freq_0_5", min_value=0.0, max_value=5.0,
        options=(
            (0.0, "Almost never"), (1.0, "Very infrequently"),
            (2.0, "Somewhat infrequently"), (3.0, "Somewhat frequently"),
            (4.0, "Very frequently"), (5.0, "Almost always"),
        ),
    ),
    "memory_freq_0_4": ResponseType(
        id="memory_freq_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Never"), (1.0, "Rarely"), (2.0, "Sometimes"),
            (3.0, "Often"), (4.0, "All the time"),
        ),
    ),
    "agree_brs_0_4": ResponseType(
        id="agree_brs_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Strongly disagree"), (1.0, "Disagree"),
            (2.0, "Neither agree or disagree"), (3.0, "Agree"), (4.0, "Strongly agree"),
        ),
    ),
    "likert_paq_0_6": ResponseType(
        id="likert_paq_0_6", min_value=0.0, max_value=6.0,
        options=(
            (0.0, "Strongly disagree"), (1.0, "Somewhat disagree"),
            (2.0, "A little disagree"), (3.0, "Neither agree or disagree"),
            (4.0, "A little agree"), (5.0, "Somewhat agree"), (6.0, "Strongly agree"),
        ),
    ),
    "satisfaction_0_6": ResponseType(
        id="satisfaction_0_6", min_value=0.0, max_value=6.0,
        options=(
            (0.0, "Not satisfied"), (1.0, "Unhappy"), (2.0, "Mostly dissatisfied"),
            (3.0, "Mixed"), (4.0, "Mostly satisfied"), (5.0, "Pleased"), (6.0, "Delighted"),
        ),
    ),
    "fatigue_0_3": ResponseType(
        id="fatigue_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "Not at all"), (1.0, "Less than usual"),
            (2.0, "No more than usual"), (3.0, "More than usual"),
        ),
    ),
    # ---- Vitality battery response types (v4) ----
    "who5_0_5": ResponseType(
        id="who5_0_5", min_value=0.0, max_value=5.0,
        options=(
            (0.0, "At no time"), (1.0, "Some of the time"),
            (2.0, "Less than half the time"), (3.0, "More than half the time"),
            (4.0, "Most of the time"), (5.0, "All of the time"),
        ),
    ),
    "sf36_vt_1_6": ResponseType(
        id="sf36_vt_1_6", min_value=1.0, max_value=6.0,
        options=(
            (1.0, "All of the time"), (2.0, "Most of the time"),
            (3.0, "A good bit of the time"), (4.0, "Some of the time"),
            (5.0, "A little of the time"), (6.0, "None of the time"),
        ),
    ),
    "svs_1_7": ResponseType(
        id="svs_1_7", min_value=1.0, max_value=7.0,
        options=(
            (1.0, "Not at all true"), (2.0, "Mostly not true"),
            (3.0, "Slightly not true"), (4.0, "Somewhat true"),
            (5.0, "Slightly true"), (6.0, "Mostly true"), (7.0, "Very true"),
        ),
    ),
    "promis_freq_1_5": ResponseType(
        id="promis_freq_1_5", min_value=1.0, max_value=5.0,
        options=(
            (1.0, "Never"), (2.0, "Rarely"), (3.0, "Sometimes"),
            (4.0, "Often"), (5.0, "Always"),
        ),
    ),
    "hads_0_3": ResponseType(
        id="hads_0_3", min_value=0.0, max_value=3.0,
        options=(
            (0.0, "Not at all"), (1.0, "Not often"),
            (2.0, "Sometimes"), (3.0, "Most of the time"),
        ),
    ),
    "ds14_0_5": ResponseType(
        id="ds14_0_5", min_value=0.0, max_value=5.0,
        options=(
            (0.0, "Strongly disagree"), (1.0, "Somewhat disagree"),
            (2.0, "A little disagree"), (3.0, "Neither agree or disagree"),
            (4.0, "A little agree"), (5.0, "Somewhat agree"),
        ),
    ),
    "agree_rs_0_4": ResponseType(
        id="agree_rs_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Strongly disagree"), (1.0, "Disagree"),
            (2.0, "Neither agree nor disagree"), (3.0, "Agree"), (4.0, "Strongly agree"),
        ),
    ),
    "creativity_0_4": ResponseType(
        id="creativity_0_4", min_value=0.0, max_value=4.0,
        options=(
            (0.0, "Much less creative"), (1.0, "Less creative"),
            (2.0, "Neither more or less creative"), (3.0, "More creative"),
            (4.0, "Much more creative"),
        ),
    ),
}


def get_response_type(response_type_id: str) -> ResponseType:
    try:
        return RESPONSE_TYPES[response_type_id]
    except KeyError as e:
        raise ValueError(f"Unsupported response_type: {response_type_id}") from e
