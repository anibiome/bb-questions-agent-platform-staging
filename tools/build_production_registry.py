#!/usr/bin/env python3
"""Build v3 production registry from the ANI BIOME Questions spreadsheet.

Reads:  QUESTIONS_REGISTRY_XLSX env var (or default local path)
Writes: data/registry_production/v3/{items,scales,questionnaires}.json

Also merges v2 cardiometabolic instruments (FINDRISC, EZ-CVD, SCORED,
Lee-NAFLD, IPAQ-SF, AUDIT-C) so the final registry is a superset.

Run:
    export QUESTIONS_REGISTRY_XLSX=/absolute/path/to/Questions_Ani_Confidential.xlsx
    PYTHONPATH=. python3 tools/build_production_registry.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import openpyxl  # type: ignore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
V2_DIR = PROJECT / "data" / "registry_demo" / "v2"
OUT_DIR = PROJECT / "data" / "registry_production" / "v3"
DEFAULT_XLSX = PROJECT / "data" / "registry_source" / "Questions_Ani_Confidential.xlsx"

# ---------------------------------------------------------------------------
# Response-type mapping: spreadsheet option patterns → registry response_type id
# ---------------------------------------------------------------------------
RESPONSE_TYPE_MAP: Dict[str, dict] = {
    # --- 7-point Likert (agree/disagree) ---
    "likert_agree_0_6": {
        "id": "likert_agree_0_6", "min": 0.0, "max": 6.0,
        "options": [
            (0, "Strongly disagree"), (1, "Somewhat disagree"),
            (2, "A little disagree"), (3, "Neither agree or disagree"),
            (4, "A little agree"), (5, "Somewhat agree"), (6, "Strongly agree"),
        ],
        "match_keywords": ["Strongly disagree", "Somewhat disagree", "A little disagree",
                           "Neither agree or disagree", "A little agree", "Somewhat agree", "Strongly agree"],
    },
    # --- 5-point Likert (agree/disagree standard) ---
    "likert_agree_0_4": {
        "id": "likert_agree_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Strongly disagree"), (1, "Disagree"),
            (2, "Neither agree nor disagree"), (3, "Agree"), (4, "Strongly agree"),
        ],
        "match_keywords": ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"],
    },
    # --- 5-point Likert (BIF2 variant) ---
    "likert_bfi_0_4": {
        "id": "likert_bfi_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Disagree strongly"), (1, "Disagree a little"),
            (2, "Neutral, no opinion"), (3, "Agree a little"), (4, "Agree strongly"),
        ],
        "match_keywords": ["Disagree strongly", "Disagree a little", "Neutral, no opinion"],
    },
    # --- DASS-21 (0-3 frequency) ---
    "dass_0_3": {
        "id": "dass_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Did not apply to me at all"),
            (1, "Applied to me to some of the time"),
            (2, "Applied to me a good part of the time"),
            (3, "Applied to me most of the time"),
        ],
        "match_keywords": ["Did not apply to me at all"],
    },
    # --- GAD-7 / PHQ style (0-3) ---
    "likert_0_3": {
        "id": "likert_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Not At All"), (1, "Several Days"),
            (2, "More than half of days"), (3, "Nearly every day"),
        ],
        "match_keywords": ["Several Days", "More than half"],
    },
    # --- CASP-19 (0-3 quality of life) ---
    "casp_0_3": {
        "id": "casp_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Never"), (1, "Not often"),
            (2, "Sometimes"), (3, "Always"),
        ],
        "match_keywords": ["Never", "Not often", "Sometimes", "Always"],
    },
    # --- AAQ (1-7 frequency) ---
    "likert_freq_1_7": {
        "id": "likert_freq_1_7", "min": 1.0, "max": 7.0,
        "options": [
            (1, "Never true"), (2, "Very rarely true"), (3, "Rarely true"),
            (4, "Sometimes true"), (5, "Frequently true"),
            (6, "Almost always true"), (7, "Always true"),
        ],
        "match_keywords": ["Never true", "Very rarely true"],
    },
    # --- Frequency 0-4 (Never..All the time) ---
    "freq_0_4": {
        "id": "freq_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "None of the time"), (1, "A little of the time"),
            (2, "Some of the time"), (3, "A good bit of the time"),
            (4, "All of the time"),
        ],
        "match_keywords": ["None of the time", "A little of the time"],
    },
    # --- MHC frequency 0-4 ---
    "freq_mhc_0_4": {
        "id": "freq_mhc_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "None of the time"), (1, "A little of the time"),
            (2, "Some of the time"), (3, "Most of the time"), (4, "All the time"),
        ],
        "match_keywords": ["None of the time", "Most of the time", "All the time"],
    },
    # --- PANAS affect 0-4 ---
    "affect_0_4": {
        "id": "affect_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Very slightly or not at all"), (1, "A little"),
            (2, "Moderate"), (3, "Quite a bit"), (4, "Extremely"),
        ],
        "match_keywords": ["Very slightly or not at all"],
    },
    # --- PGIS agreement 0-5 ---
    "agree_0_5": {
        "id": "agree_0_5", "min": 0.0, "max": 5.0,
        "options": [
            (0, "Definitely disagree"), (1, "Mostly disagree"),
            (2, "Somewhat disagree"), (3, "Somewhat agree"),
            (4, "Mostly agree"), (5, "Definitely agree"),
        ],
        "match_keywords": ["Definitely disagree", "Mostly disagree", "Definitely agree"],
    },
    # --- 5-point frequency (Never..Almost always) ---
    "freq_never_almost_0_4": {
        "id": "freq_never_almost_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Never"), (1, "Rarely"), (2, "Sometimes"),
            (3, "Often"), (4, "Almost always"),
        ],
        "match_keywords": ["Never", "Rarely", "Sometimes", "Often", "Almost always"],
    },
    # --- 5-point frequency (Never..Always ESCQ) ---
    "freq_never_always_0_4": {
        "id": "freq_never_always_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Never"), (1, "Seldom"), (2, "Occasionally"),
            (3, "Usually"), (4, "Always"),
        ],
        "match_keywords": ["Seldom", "Occasionally", "Usually", "Always"],
    },
    # --- 4-point Cope (0-3) ---
    "cope_0_3": {
        "id": "cope_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "I usually don't do this at all"),
            (1, "I usually do this a little bit"),
            (2, "I usually do this a medium amount"),
            (3, "I usually do this a lot"),
        ],
        "match_keywords": ["I usually don't do this at all"],
    },
    # --- 5-point Coping Self-Efficacy ---
    "coping_efficacy_0_4": {
        "id": "coping_efficacy_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Cannot do at all"), (1, "Cannot do most of the time"),
            (2, "Can do sometimes"), (3, "Moderately certain can do"),
            (4, "Certain can do"),
        ],
        "match_keywords": ["Cannot do at all", "Certain can do"],
    },
    # --- 4-point truth (BPNSNF/GSE) ---
    "truth_0_3": {
        "id": "truth_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Not true at all"), (1, "Hardly true"),
            (2, "Moderately true"), (3, "Completely true"),
        ],
        "match_keywords": ["Not true at all", "Hardly true", "Completely true"],
    },
    # --- 4-point truth variant (GSE Exactly true) ---
    "truth_gse_0_3": {
        "id": "truth_gse_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Not true at all"), (1, "Hardly true"),
            (2, "Moderately true"), (3, "Exactly true"),
        ],
        "match_keywords": ["Not true at all", "Exactly true"],
    },
    # --- GSRS-IBS discomfort 0-6 ---
    "discomfort_0_6": {
        "id": "discomfort_0_6", "min": 0.0, "max": 6.0,
        "options": [
            (0, "No discomfort at all"), (1, "Minor discomfort"),
            (2, "Mild discomfort"), (3, "Moderate discomfort"),
            (4, "Moderately severe discomfort"), (5, "Severe discomfort"),
            (6, "Very severe discomfort"),
        ],
        "match_keywords": ["No discomfort at all", "Minor discomfort"],
    },
    # --- DQLQ frequency 0-5 ---
    "freq_dqlq_0_5": {
        "id": "freq_dqlq_0_5", "min": 0.0, "max": 5.0,
        "options": [
            (0, "Never"), (1, "Rarely"), (2, "Occasionally"),
            (3, "Sometimes"), (4, "Frequently"), (5, "Usually"),
        ],
        "match_keywords": ["Never", "Rarely", "Occasionally", "Sometimes", "Frequently", "Usually"],
    },
    # --- 4-point IBS frequency ---
    "freq_ibs_0_3": {
        "id": "freq_ibs_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Never"), (1, "Some of the time"),
            (2, "Most of the time"), (3, "Always"),
        ],
        "match_keywords": ["Some of the time", "Most of the time"],
    },
    # --- FMI 4-point (0-3 Rarely..Almost always) ---
    "fmi_0_3": {
        "id": "fmi_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Rarely"), (1, "Occasionally"),
            (2, "Fairly often"), (3, "Almost always"),
        ],
        "match_keywords": ["Rarely", "Occasionally", "Fairly often"],
    },
    # --- CAMS-R 4-point ---
    "camsr_0_3": {
        "id": "camsr_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Not at all"), (1, "Sometimes"),
            (2, "Often"), (3, "Almost always"),
        ],
        "match_keywords": ["Not at all", "Sometimes", "Often", "Almost always"],
    },
    # --- CFQ frequency 0-4 ---
    "cfq_0_4": {
        "id": "cfq_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Never"), (1, "Very rarely"), (2, "Occasionally"),
            (3, "Quite often"), (4, "Very often"),
        ],
        "match_keywords": ["Very rarely", "Quite often", "Very often"],
    },
    # --- 4-point rumination (0-3) ---
    "rumination_0_3": {
        "id": "rumination_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Almost never"), (1, "Sometimes"),
            (2, "Often"), (3, "Almost always"),
        ],
        "match_keywords": ["Almost never", "Sometimes", "Often", "Almost always"],
    },
    # --- SCL somatic 0-4 ---
    "scl_0_4": {
        "id": "scl_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Not at all"), (1, "A little bit"),
            (2, "Moderately"), (3, "Quite a bit"), (4, "Extremely"),
        ],
        "match_keywords": ["Not at all", "A little bit", "Moderately", "Quite a bit", "Extremely"],
    },
    # --- SIAS 0-4 (Not at all..Extremely) ---
    "sias_0_4": {
        "id": "sias_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Not at all"), (1, "Slightly"), (2, "Moderately"),
            (3, "Very"), (4, "Extremely"),
        ],
        "match_keywords": ["Not at all", "Slightly", "Moderately", "Very", "Extremely"],
    },
    # --- RSES 4-point agree ---
    "agree_rses_0_3": {
        "id": "agree_rses_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Strongly Disagree"), (1, "Disagree"),
            (2, "Agree"), (3, "Strongly Agree"),
        ],
        "match_keywords": ["Strongly Disagree", "Disagree", "Agree", "Strongly Agree"],
    },
    # --- SSQ-ISEL truth 0-3 ---
    "truth_ssq_0_3": {
        "id": "truth_ssq_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Definitely false"), (1, "Probably false"),
            (2, "Probably true"), (3, "Definitely true"),
        ],
        "match_keywords": ["Definitely false", "Probably false", "Probably true", "Definitely true"],
    },
    # --- UCLA loneliness 0-3 ---
    "ucla_0_3": {
        "id": "ucla_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "I never feel this way"), (1, "I rarely feel this way"),
            (2, "I sometimes feel this way"), (3, "I often feel this way"),
        ],
        "match_keywords": ["I never feel this way", "I rarely feel this way"],
    },
    # --- MSPSS 7-point agreement ---
    "mspss_1_6": {
        "id": "mspss_1_6", "min": 0.0, "max": 5.0,
        "options": [
            (0, "Very Strongly Disagree"), (1, "Strongly disagree"),
            (2, "Mildly disagree"), (3, "Neutral"),
            (4, "Mildly agree"), (5, "Strongly agree"),
        ],
        "match_keywords": ["Very Strongly Disagree", "Mildly disagree", "Mildly agree"],
    },
    # --- Visceral sensitivity 6-point ---
    "visceral_0_5": {
        "id": "visceral_0_5", "min": 0.0, "max": 5.0,
        "options": [
            (0, "Strongly disagree"), (1, "Moderately disagree"),
            (2, "Mildly disagree"), (3, "Mildly agree"),
            (4, "Moderately agree"), (5, "Strongly agree"),
        ],
        "match_keywords": ["Strongly disagree", "Moderately disagree", "Mildly disagree", "Mildly agree"],
    },
    # --- Harrill self-esteem 0-4 ---
    "harrill_0_4": {
        "id": "harrill_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "I never think, feel or behave this way"),
            (1, "I do less than half the time"),
            (2, "I do 50% of the time."),
            (3, "I do more than half the time"),
            (4, "I always think, feel or behave this way"),
        ],
        "match_keywords": ["I never think, feel or behave"],
    },
    # --- ERQ agreement variant 0-4 ---
    "erq_0_4": {
        "id": "erq_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Strongly disagree"), (1, "Disagree a little"),
            (2, "Neutral, no opinion"), (3, "Agree a little"),
            (4, "Totally agree"),
        ],
        "match_keywords": ["Strongly disagree", "Disagree a little", "Totally agree"],
    },
    # --- PSS 0-4 frequency ---
    "pss_0_4": {
        "id": "pss_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Never"), (1, "Almost never"), (2, "Sometimes"),
            (3, "Fairly often"), (4, "Very often"),
        ],
        "match_keywords": ["Never", "Almost never", "Fairly often", "Very often"],
    },
    # --- Day-to-Day (mindfulness) 0-5 ---
    "mindful_freq_0_5": {
        "id": "mindful_freq_0_5", "min": 0.0, "max": 5.0,
        "options": [
            (0, "Almost never"), (1, "Very infrequently"),
            (2, "Somewhat infrequently"), (3, "Somewhat frequently"),
            (4, "Very frequently"), (5, "Almost always"),
        ],
        "match_keywords": ["Very infrequently", "Somewhat infrequently", "Somewhat frequently"],
    },
    # --- Multifactorial memory freq_0_4 ---
    "memory_freq_0_4": {
        "id": "memory_freq_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Never"), (1, "Rarely"), (2, "Sometimes"),
            (3, "Often"), (4, "All the time"),
        ],
        "match_keywords": ["Never", "Rarely", "Sometimes", "Often", "All the time"],
    },
    # --- BRS 5-point agree ---
    "agree_brs_0_4": {
        "id": "agree_brs_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Strongly disagree"), (1, "Disagree"),
            (2, "Neither agree or disagree"), (3, "Agree"), (4, "Strongly agree"),
        ],
        "match_keywords": ["Strongly disagree", "Disagree", "Neither agree or disagree", "Agree", "Strongly agree"],
    },
    # --- PAQ 7-point (same as Mind Age) ---
    "likert_paq_0_6": {
        "id": "likert_paq_0_6", "min": 0.0, "max": 6.0,
        "options": [
            (0, "Strongly disagree"), (1, "Somewhat disagree"),
            (2, "A little disagree"), (3, "Neither agree or disagree"),
            (4, "A little agree"), (5, "Somewhat agree"), (6, "Strongly agree"),
        ],
        "match_keywords": [],  # alias
    },
    # --- QOLS satisfaction 0-6 ---
    "satisfaction_0_6": {
        "id": "satisfaction_0_6", "min": 0.0, "max": 6.0,
        "options": [
            (0, "Not satisfied"), (1, "Unhappy"), (2, "Mostly dissatisfied"),
            (3, "Mixed"), (4, "Mostly satisfied"), (5, "Pleased"), (6, "Delighted"),
        ],
        "match_keywords": ["Unhappy", "Mostly dissatisfied", "Mostly satisfied", "Pleased"],
    },
    # --- BIS sleep days 0-7 ---
    "days_0_7": {
        "id": "days_0_7", "min": 0.0, "max": 7.0,
        "options": [(i, f"{i} day{'s' if i != 1 else ''}") for i in range(8)],
        "match_keywords": ["No days", "1 day", "2 days"],
    },
    # --- CFS fatigue 0-3 variant ---
    "fatigue_0_3": {
        "id": "fatigue_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Not at all"), (1, "Less than usual"),
            (2, "No more than usual"), (3, "More than usual"),
        ],
        "match_keywords": ["Not at all", "Less than usual", "No more than usual", "More than usual"],
    },
    # --- HADS 0-3 (varied per item - use generic) ---
    "hads_0_3": {
        "id": "hads_0_3", "min": 0.0, "max": 3.0,
        "options": [
            (0, "Not at all"), (1, "Not often"),
            (2, "Sometimes"), (3, "Most of the time"),
        ],
        "match_keywords": ["Not at all", "Not often", "Most of the time"],
    },
    # --- DS-14 agreement 0-5 (6-point) ---
    "ds14_0_5": {
        "id": "ds14_0_5", "min": 0.0, "max": 5.0,
        "options": [
            (0, "Strongly disagree"), (1, "Somewhat disagree"),
            (2, "A little disagree"), (3, "Neither agree or disagree"),
            (4, "A little agree"), (5, "Somewhat agree"),
        ],
        "match_keywords": [],  # subtype
    },
    # --- RS agree 0-4 ---
    "agree_rs_0_4": {
        "id": "agree_rs_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Strongly disagree"), (1, "Disagree"),
            (2, "Neither agree nor disagree"), (3, "Agree"), (4, "Strongly agree"),
        ],
        "match_keywords": [],
    },
    # --- K-DOCS creativity 0-4 ---
    "creativity_0_4": {
        "id": "creativity_0_4", "min": 0.0, "max": 4.0,
        "options": [
            (0, "Much less creative"), (1, "Less creative"),
            (2, "Neither more or less creative"), (3, "More creative"),
            (4, "Much more creative"),
        ],
        "match_keywords": ["Much less creative", "Less creative"],
    },
}

# Map instrument source → forced response_type id
INSTRUMENT_RESPONSE_TYPE: Dict[str, str] = {
    "Mind Age": "likert_agree_0_6",
    "MHC": "freq_mhc_0_4",
    "CAMS-R": "camsr_0_3",
    "DASS-21-stress": "dass_0_3",
    "DASS-21-depression": "dass_0_3",
    "FMI": "fmi_0_3",
    "GAD-7": "likert_0_3",
    "CASP-19": "casp_0_3",
    "GSRS-IBS": "discomfort_0_6",
    "PANAS-SF-positive": "affect_0_4",
    "PANAS-SF-negative": "affect_0_4",
    "PGIS": "agree_0_5",
    "Affect Balance Scale": "freq_0_4",
    "AAQ": "likert_freq_1_7",
    "BRS": "agree_brs_0_4",
    "BIF2": "likert_bfi_0_4",
    "Coping Self-Efficacy Scale": "coping_efficacy_0_4",
    "BPNSNF": "truth_0_3",
    "Boundaries Assessment": "freq_never_almost_0_4",
    "ESCQ": "freq_never_always_0_4",
    "QOLS Flanagan": "satisfaction_0_6",
    "MSPSS": "mspss_1_6",
    "GSE": "truth_gse_0_3",
    "DQLQ": "freq_dqlq_0_5",
    "Rumination": "rumination_0_3",
    "Harrill Self-Esteem Inventory": "harrill_0_4",
    "ERQ": "erq_0_4",
    "SIAS": "sias_0_4",
    "SSQ - ISEL": "truth_ssq_0_3",
    "RSES": "agree_rses_0_3",
    "RS": "agree_rs_0_4",
    "Authenticity Scale": "likert_agree_0_4",
    "SRIS": "likert_agree_0_4",
    "PSS": "pss_0_4",
    "Ryff": "likert_agree_0_6",
    "CFQ": "cfq_0_4",
    "IBS": "freq_ibs_0_3",
    "SCL": "scl_0_4",
    "PAQ": "likert_paq_0_6",
    "DS-14": "ds14_0_5",
    "IBDQ": "discomfort_0_6",  # IBDQ uses 7-point but map to main type
    "CFS": "fatigue_0_3",
    "BIS": "days_0_7",
    "HADS": "hads_0_3",
    "Visceral Sensitivity Index": "visceral_0_5",
    "Day-to-Day Experiences": "mindful_freq_0_5",
    "Cope Inventory": "cope_0_3",
    "UCLA Loneliness scale": "ucla_0_3",
    "Multifactorial memory questionnaire": "memory_freq_0_4",
    "Initial Scan": "likert_agree_0_4",  # varies per item
}

# ---------------------------------------------------------------------------
# Questionnaire metadata: source → (module, category, domains)
# ---------------------------------------------------------------------------
QUESTIONNAIRE_META: Dict[str, Dict[str, Any]] = {
    "Mind Age": {"module": "mind_age", "domains": ("cognition", "growth"), "license": "internal"},
    "MHC": {"module": "emotional_health", "domains": ("positive_emotions",), "license": "public_domain"},
    "CAMS-R": {"module": "cognitive_health", "domains": ("mindfulness",), "license": "open_access"},
    "DASS-21-stress": {"module": "mental_health", "domains": ("stress", "anxiety"), "license": "open_access"},
    "DASS-21-depression": {"module": "mental_health", "domains": ("depression",), "license": "open_access"},
    "FMI": {"module": "cognitive_health", "domains": ("mindfulness",), "license": "open_access"},
    "GAD-7": {"module": "mental_health", "domains": ("anxiety",), "license": "open_access", "citation": "Spitzer RL et al. Arch Intern Med. 2006;166(10):1092-7"},
    "CASP-19": {"module": "wellbeing", "domains": ("quality_of_life",), "license": "open_access"},
    "GSRS-IBS": {"module": "gi_health", "domains": ("gi_symptoms",), "license": "academic"},
    "PANAS-SF-positive": {"module": "emotional_health", "domains": ("positive_affect",), "license": "open_access"},
    "PANAS-SF-negative": {"module": "emotional_health", "domains": ("negative_affect",), "license": "open_access"},
    "PGIS": {"module": "self_concept", "domains": ("personal_growth",), "license": "open_access"},
    "Affect Balance Scale": {"module": "mental_health", "domains": ("mood",), "license": "open_access"},
    "AAQ": {"module": "emotional_health", "domains": ("flexibility",), "license": "open_access", "audit_flag": "superseded"},
    "BRS": {"module": "emotional_health", "domains": ("resilience",), "license": "open_access", "citation": "Smith BW et al. Int J Behav Med. 2008;15(3):194-200"},
    "BIF2": {"module": "personality", "domains": ("personality",), "license": "open_access"},
    "Coping Self-Efficacy Scale": {"module": "self_concept", "domains": ("coping",), "license": "open_access"},
    "BPNSNF": {"module": "mental_health", "domains": ("needs_satisfaction",), "license": "open_access"},
    "Boundaries Assessment": {"module": "emotional_health", "domains": ("boundaries",), "license": "internal"},
    "ESCQ": {"module": "emotional_health", "domains": ("emotional_competence",), "license": "academic"},
    "QOLS Flanagan": {"module": "mental_health", "domains": ("quality_of_life",), "license": "open_access"},
    "MSPSS": {"module": "social", "domains": ("social_support",), "license": "open_access", "citation": "Zimet GD et al. J Pers Assess. 1988;52(1):30-41"},
    "GSE": {"module": "self_concept", "domains": ("self_efficacy",), "license": "open_access"},
    "DQLQ": {"module": "gi_health", "domains": ("gi_quality_of_life",), "license": "academic"},
    "Rumination": {"module": "mental_health", "domains": ("rumination",), "license": "open_access"},
    "Harrill Self-Esteem Inventory": {"module": "self_concept", "domains": ("self_esteem",), "license": "internal"},
    "ERQ": {"module": "emotional_health", "domains": ("emotion_regulation",), "license": "open_access", "citation": "Gross JQ, John OP. J Pers Soc Psychol. 2003;85(2):348-62"},
    "SIAS": {"module": "social", "domains": ("social_anxiety",), "license": "open_access"},
    "SSQ - ISEL": {"module": "social", "domains": ("social_support",), "license": "open_access"},
    "RSES": {"module": "self_concept", "domains": ("self_esteem",), "license": "open_access", "citation": "Rosenberg M. Society and the adolescent self-image. 1965"},
    "RS": {"module": "emotional_health", "domains": ("resilience",), "license": "academic", "audit_flag": "redundant_with_cdrisc"},
    "Authenticity Scale": {"module": "self_concept", "domains": ("authenticity",), "license": "open_access"},
    "SRIS": {"module": "self_concept", "domains": ("self_reflection",), "license": "open_access"},
    "PSS": {"module": "emotional_health", "domains": ("perceived_stress",), "license": "open_access", "citation": "Cohen S et al. J Health Soc Behav. 1983;24(4):385-96"},
    "Ryff": {"module": "wellbeing", "domains": ("psychological_wellbeing",), "license": "open_access", "audit_flag": "poor_psychometrics_short"},
    "CFQ": {"module": "cognitive_health", "domains": ("cognitive_failures",), "license": "open_access", "audit_flag": "dated_consider_promis"},
    "IBS": {"module": "gi_health", "domains": ("gi_symptoms",), "license": "academic"},
    "SCL": {"module": "mental_health", "domains": ("somatization",), "license": "academic"},
    "PAQ": {"module": "emotional_health", "domains": ("alexithymia",), "license": "academic"},
    "DS-14": {"module": "emotional_health", "domains": ("type_d_personality",), "license": "academic"},
    "IBDQ": {"module": "gi_health", "domains": ("ibd_quality_of_life",), "license": "academic"},
    "CFS": {"module": "behaviour", "domains": ("fatigue",), "license": "open_access"},
    "BIS": {"module": "behaviour", "domains": ("sleep",), "license": "open_access"},
    "HADS": {"module": "mental_health", "domains": ("anxiety", "depression"), "license": "commercial"},
    "Visceral Sensitivity Index": {"module": "gi_health", "domains": ("visceral_sensitivity",), "license": "open_access"},
    "Day-to-Day Experiences": {"module": "cognitive_health", "domains": ("mindfulness",), "license": "open_access"},
    "Cope Inventory": {"module": "emotional_health", "domains": ("coping",), "license": "open_access"},
    "UCLA Loneliness scale": {"module": "social", "domains": ("loneliness",), "license": "open_access"},
    "Multifactorial memory questionnaire": {"module": "cognitive_health", "domains": ("memory",), "license": "open_access"},
    "Initial Scan": {"module": "initial_scan", "domains": ("screening",), "license": "internal"},
}

# Scale scoring method mapping
INSTRUMENT_SCORING: Dict[str, Dict[str, Any]] = {
    "Mind Age": {"method": "mean", "min": 0, "max": 6},
    "MHC": {"method": "mean", "min": 0, "max": 4},
    "CAMS-R": {"method": "sum", "min": 0, "max": 30},
    "DASS-21-stress": {"method": "sum", "min": 0, "max": 42},  # multiply by 2
    "DASS-21-depression": {"method": "sum", "min": 0, "max": 42},
    "FMI": {"method": "sum", "min": 0, "max": 42},
    "GAD-7": {"method": "sum", "min": 0, "max": 21},
    "CASP-19": {"method": "sum", "min": 0, "max": 57},
    "GSRS-IBS": {"method": "mean", "min": 0, "max": 6},
    "PANAS-SF-positive": {"method": "sum", "min": 9, "max": 45},
    "PANAS-SF-negative": {"method": "sum", "min": 10, "max": 50},
    "PGIS": {"method": "mean", "min": 0, "max": 5},
    "Affect Balance Scale": {"method": "sum", "min": 0, "max": 10},
    "AAQ": {"method": "sum", "min": 7, "max": 49},
    "BRS": {"method": "mean", "min": 0, "max": 4},
    "BIF2": {"method": "mean", "min": 0, "max": 4},  # per subscale
    "Coping Self-Efficacy Scale": {"method": "sum", "min": 0, "max": 130},
    "BPNSNF": {"method": "mean", "min": 0, "max": 3},
    "Boundaries Assessment": {"method": "sum", "min": 0, "max": 76},
    "ESCQ": {"method": "mean", "min": 0, "max": 4},
    "QOLS Flanagan": {"method": "sum", "min": 15, "max": 105},
    "MSPSS": {"method": "mean", "min": 0, "max": 5},
    "GSE": {"method": "sum", "min": 10, "max": 40},
    "DQLQ": {"method": "sum", "min": 0, "max": 45},
    "Rumination": {"method": "sum", "min": 22, "max": 88},
    "Harrill Self-Esteem Inventory": {"method": "sum", "min": 0, "max": 24},
    "ERQ": {"method": "mean", "min": 0, "max": 4},
    "SIAS": {"method": "sum", "min": 0, "max": 80},
    "SSQ - ISEL": {"method": "sum", "min": 0, "max": 120},
    "RSES": {"method": "sum", "min": 0, "max": 30},
    "RS": {"method": "sum", "min": 14, "max": 98},
    "Authenticity Scale": {"method": "mean", "min": 0, "max": 4},
    "SRIS": {"method": "mean", "min": 0, "max": 4},
    "PSS": {"method": "sum", "min": 0, "max": 40},
    "Ryff": {"method": "mean", "min": 0, "max": 6},
    "CFQ": {"method": "sum", "min": 0, "max": 60},
    "IBS": {"method": "sum", "min": 0, "max": 27},
    "SCL": {"method": "sum", "min": 0, "max": 48},
    "PAQ": {"method": "mean", "min": 0, "max": 6},
    "DS-14": {"method": "sum", "min": 0, "max": 56},
    "IBDQ": {"method": "sum", "min": 32, "max": 224},
    "CFS": {"method": "sum", "min": 0, "max": 42},
    "BIS": {"method": "sum", "min": 0, "max": 42},
    "HADS": {"method": "sum", "min": 0, "max": 42},
    "Visceral Sensitivity Index": {"method": "sum", "min": 0, "max": 75},
    "Day-to-Day Experiences": {"method": "mean", "min": 0, "max": 5},
    "Cope Inventory": {"method": "mean", "min": 0, "max": 3},
    "UCLA Loneliness scale": {"method": "sum", "min": 0, "max": 57},
    "Multifactorial memory questionnaire": {"method": "sum", "min": 0, "max": 80},
    "Initial Scan": {"method": "mean", "min": 0, "max": 4},
}

# Tag mapping for Coherence Circle decoherence modes
DOMAIN_TAGS: Dict[str, Tuple[str, ...]] = {
    "mental_health": ("affective",),
    "emotional_health": ("affective",),
    "cognitive_health": ("autonomic", "affective"),
    "gi_health": ("immune", "metabolic"),
    "self_concept": ("affective",),
    "social": ("affective",),
    "personality": ("affective",),
    "behaviour": ("autonomic",),
    "wellbeing": ("affective",),
    "mind_age": ("affective", "autonomic"),
    "initial_scan": ("screening",),
}

CASP19_SOURCE = "CASP-19"
CASP19_QUESTIONNAIRE_ID = "q_casp_19"
CASP19_SCALE_ID = "scale_casp_19"
CASP19_RESPONSE_TYPE = "casp_0_3"
CASP19_ITEM_SPECS: Tuple[Tuple[str, str, bool], ...] = (
    ("item_casp_19_01_age_prevents", "My age prevents me from doing the things I would like to.", False),
    ("item_casp_19_02_out_of_control", "I feel that what happens to me is out of my control.", False),
    ("item_casp_19_03_free_plan_future", "I feel free to plan for the future.", True),
    ("item_casp_19_04_left_out", "I feel left out of things.", False),
    ("item_casp_19_05_do_things_want", "I can do the things that I want to do.", True),
    ("item_casp_19_06_family_responsibilities", "Family responsibilities prevent me from doing what I want to do.", False),
    ("item_casp_19_07_please_myself", "I feel I can please myself what I do.", True),
    ("item_casp_19_08_health_stops", "My health stops me from doing things I want to do.", False),
    ("item_casp_19_09_money_stops", "Shortage of money stops me from doing the things I want to do.", False),
    ("item_casp_19_10_look_forward_each_day", "I look forward to each day.", True),
    ("item_casp_19_11_life_has_meaning", "I feel that my life has meaning.", True),
    ("item_casp_19_12_enjoy_things", "I enjoy the things that I do.", True),
    ("item_casp_19_13_enjoy_company_others", "I enjoy being in the company of others.", True),
    ("item_casp_19_14_happiness_back_on_life", "On balance, I look back on my life with a sense of happiness.", True),
    ("item_casp_19_15_full_of_energy", "I feel full of energy these days.", True),
    ("item_casp_19_16_choose_new_things", "I choose to do things that I have never done before.", True),
    ("item_casp_19_17_life_turnout_satisfied", "I feel satisfied with the way my life has turned out.", True),
    ("item_casp_19_18_life_full_of_opportunities", "I feel that life is full of opportunities.", True),
    ("item_casp_19_19_future_looks_good", "I feel that the future looks good for me.", True),
)
CASP19_REVERSE_ITEM_IDS = {item_id for item_id, _, reverse in CASP19_ITEM_SPECS if reverse}


def build_casp19_items() -> List[dict]:
    items: List[dict] = []
    for item_id, text, _reverse in CASP19_ITEM_SPECS:
        items.append(
            {
                "id": item_id,
                "text": text,
                "response_type": CASP19_RESPONSE_TYPE,
                "tags": ["affective", "quality_of_life"],
                "sensitivity": "low",
                "timeframes_allowed": ["last_7_days"],
                "intrusiveness": "low",
                "declinable": True,
                "onboarding_order": None,
            }
        )
    return items


def build_casp19_scale() -> dict:
    return {
        "id": CASP19_SCALE_ID,
        "questionnaire_id": CASP19_QUESTIONNAIRE_ID,
        "version": "1",
        "name": CASP19_SOURCE,
        "method": "sum",
        "min_items_required": 14,
        "unlock_window_days": 0,
        "retest_interval_days": 90,
        "response_type": CASP19_RESPONSE_TYPE,
        "normalize_min": 0.0,
        "normalize_max": 57.0,
        "items": [
            {
                "item_id": item_id,
                "reverse": reverse,
                "weight": 1.0,
            }
            for item_id, _text, reverse in CASP19_ITEM_SPECS
        ],
        "tags": ["affective", "quality_of_life"],
        "ewma_alpha": 0.2,
        "citation": "",
    }


def build_casp19_questionnaire() -> dict:
    return {
        "id": CASP19_QUESTIONNAIRE_ID,
        "version": "1",
        "name": CASP19_SOURCE,
        "domains": ["quality_of_life"],
        "license": "open_access",
        "source": None,
    }


# ---------------------------------------------------------------------------
# Parse spreadsheet
# ---------------------------------------------------------------------------
def parse_items(ws: Any) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """Return (all_items, items_by_source)."""
    items: List[dict] = []
    by_source: Dict[str, List[dict]] = defaultdict(list)

    for r in range(2, ws.max_row + 1):
        qid = ws.cell(r, 1).value
        text = ws.cell(r, 2).value
        source = ws.cell(r, 3).value
        sub_cat = str(ws.cell(r, 6).value or "")
        reversed_flag = str(ws.cell(r, 8).value or "").strip().lower() == "true"

        if not qid or not text or not source:
            continue

        # Response options
        opts = []
        for c in range(11, 22):
            v = ws.cell(r, c).value
            if v is not None:
                opts.append(str(v))

        resp_type_id = INSTRUMENT_RESPONSE_TYPE.get(str(source), "likert_agree_0_4")

        # Build tags
        module = QUESTIONNAIRE_META.get(str(source), {}).get("module", "unknown")
        decoherence_tags = DOMAIN_TAGS.get(module, ())
        item_tags = list(decoherence_tags)
        if sub_cat:
            tag = sub_cat.lower().replace(" ", "_").replace("-", "_")
            if tag not in item_tags:
                item_tags.append(tag)

        item = {
            "id": str(qid).strip(),
            "text": str(text).strip(),
            "response_type": resp_type_id,
            "tags": tuple(item_tags),
            "sensitivity": "medium" if module in ("mental_health", "gi_health") else "low",
            "timeframes_allowed": ("last_7_days",) if module != "personality" else ("trait",),
            "intrusiveness": "medium" if module in ("mental_health",) else "low",
            "declinable": True,
            "onboarding_order": None,
            "source": str(source),
            "reversed": reversed_flag,
            "subcategory": sub_cat,
        }
        items.append(item)
        by_source[str(source)].append(item)

    return items, by_source


def build_scales(by_source: Dict[str, List[dict]]) -> List[dict]:
    """Build scale definitions from grouped items."""
    scales: List[dict] = []

    for source, items in sorted(by_source.items()):
        if source not in QUESTIONNAIRE_META:
            print(f"  WARN: no metadata for source '{source}', skipping")
            continue

        meta = QUESTIONNAIRE_META[source]
        scoring = INSTRUMENT_SCORING.get(source, {"method": "mean", "min": 0, "max": 4})
        resp_type = INSTRUMENT_RESPONSE_TYPE.get(source, "likert_agree_0_4")

        # Build scale items
        scale_items = []
        for it in items:
            scale_items.append({
                "item_id": it["id"],
                "reverse": it["reversed"],
                "weight": 1.0,
            })

        # Determine tags
        module = meta.get("module", "unknown")
        decoherence_tags = list(DOMAIN_TAGS.get(module, ()))
        for d in meta.get("domains", ()):
            if d not in decoherence_tags:
                decoherence_tags.append(d)

        # Audit flags as tags
        if meta.get("audit_flag"):
            decoherence_tags.append(f"audit:{meta['audit_flag']}")

        scale_id = f"scale_{source.lower().replace(' ', '_').replace('-', '_')}"
        n_items = len(items)
        min_items = max(1, int(n_items * 0.75))  # require 75% completion

        # Retest interval based on module type
        retest_days = 90
        if module in ("behaviour", "gi_health"):
            retest_days = 30
        elif module == "personality":
            retest_days = 180

        scale = {
            "id": scale_id,
            "questionnaire_id": f"q_{source.lower().replace(' ', '_').replace('-', '_')}",
            "version": "1",
            "name": source,
            "method": scoring["method"],
            "min_items_required": min_items,
            "unlock_window_days": 0,
            "retest_interval_days": retest_days,
            "response_type": resp_type,
            "normalize_min": float(scoring["min"]),
            "normalize_max": float(scoring["max"]),
            "items": scale_items,
            "tags": tuple(decoherence_tags),
            "ewma_alpha": 0.2,
            "citation": meta.get("citation", ""),
        }
        scales.append(scale)

    return scales


def build_questionnaires(by_source: Dict[str, List[dict]]) -> List[dict]:
    """Build questionnaire definitions."""
    questionnaires: List[dict] = []
    for source in sorted(by_source.keys()):
        meta = QUESTIONNAIRE_META.get(source, {})
        q = {
            "id": f"q_{source.lower().replace(' ', '_').replace('-', '_')}",
            "version": "1",
            "name": source,
            "domains": list(meta.get("domains", ())),
            "license": meta.get("license", "unknown"),
            "source": meta.get("citation", None),
        }
        questionnaires.append(q)
    return questionnaires


def merge_v2_cardiometabolic(
    items: List[dict],
    scales: List[dict],
    questionnaires: List[dict],
    existing_item_ids: Set[str],
) -> None:
    """Merge v2 cardiometabolic items/scales/questionnaires into lists."""
    v2_items_path = V2_DIR / "items.json"
    v2_scales_path = V2_DIR / "scales.json"
    v2_quest_path = V2_DIR / "questionnaires.json"

    if not v2_items_path.exists():
        print("  WARN: v2 directory not found, skipping cardiometabolic merge")
        return

    with open(v2_items_path) as f:
        v2_items = json.load(f)
    with open(v2_scales_path) as f:
        v2_scales = json.load(f)
    with open(v2_quest_path) as f:
        v2_quests = json.load(f)

    # Add items not already present
    added = 0
    for it in v2_items:
        if it["id"] not in existing_item_ids:
            items.append(it)
            existing_item_ids.add(it["id"])
            added += 1

    # Add scales
    existing_scale_ids = {s["id"] for s in scales}
    for sc in v2_scales:
        if sc["id"] not in existing_scale_ids:
            scales.append(sc)

    # Add questionnaires
    existing_q_ids = {q["id"] for q in questionnaires}
    for q in v2_quests:
        if q["id"] not in existing_q_ids:
            questionnaires.append(q)

    print(f"  Merged {added} cardiometabolic items, {len(v2_scales)} scales, {len(v2_quests)} questionnaires from v2")


def merge_casp19(
    items: List[dict],
    scales: List[dict],
    questionnaires: List[dict],
    existing_item_ids: Set[str],
) -> None:
    """Add CASP-19 definitions to the registry if not already present."""
    existing_scale_ids = {s["id"] for s in scales}
    existing_q_ids = {q["id"] for q in questionnaires}

    added_items = 0
    for item in build_casp19_items():
        if item["id"] in existing_item_ids:
            continue
        items.append(item)
        existing_item_ids.add(item["id"])
        added_items += 1

    if CASP19_SCALE_ID not in existing_scale_ids:
        scales.append(build_casp19_scale())

    if CASP19_QUESTIONNAIRE_ID not in existing_q_ids:
        questionnaires.append(build_casp19_questionnaire())

    print(
        f"  Added CASP-19 registry bundle: {added_items} items, "
        f"{1 if CASP19_SCALE_ID not in existing_scale_ids else 0} scales, "
        f"{1 if CASP19_QUESTIONNAIRE_ID not in existing_q_ids else 0} questionnaires"
    )


def build_response_types_json() -> List[dict]:
    """Build the new response types for the production registry."""
    rts = []
    for rt_id, rt_data in sorted(RESPONSE_TYPE_MAP.items()):
        rts.append({
            "id": rt_id,
            "min_value": rt_data["min"],
            "max_value": rt_data["max"],
            "options": [[v, label] for v, label in rt_data["options"]],
        })
    return rts


def compute_multiplex_map(scales: List[dict]) -> Dict[str, List[str]]:
    """Item → list of scale IDs that use it."""
    m: Dict[str, List[str]] = defaultdict(list)
    for sc in scales:
        for si in sc.get("items", []):
            item_id = si["item_id"] if isinstance(si, dict) else si.item_id
            m[item_id].append(sc["id"])
    return dict(m)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    xlsx_path = Path(
        os.environ.get("QUESTIONS_REGISTRY_XLSX", str(DEFAULT_XLSX))
    ).expanduser()
    if not xlsx_path.exists():
        print(
            "ERROR: input spreadsheet not found. "
            "Set QUESTIONS_REGISTRY_XLSX to an existing file path."
        )
        print(f"Resolved path: {xlsx_path}")
        sys.exit(1)

    print("Building v3 production registry...")
    print(f"  Reading: {xlsx_path}")

    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True)
    ws = wb["Questions"]

    # 1. Parse items
    all_items, by_source = parse_items(ws)
    print(f"  Parsed {len(all_items)} items from {len(by_source)} instruments")

    # 2. Build scales
    scales = build_scales(by_source)
    print(f"  Built {len(scales)} scales")

    # 3. Build questionnaires
    questionnaires = build_questionnaires(by_source)
    print(f"  Built {len(questionnaires)} questionnaires")

    # 4. Convert items to JSON-serializable format (remove internal fields)
    items_json = []
    existing_ids: Set[str] = set()
    for it in all_items:
        existing_ids.add(it["id"])
        items_json.append({
            "id": it["id"],
            "text": it["text"],
            "response_type": it["response_type"],
            "tags": list(it["tags"]),
            "sensitivity": it["sensitivity"],
            "timeframes_allowed": list(it["timeframes_allowed"]),
            "intrusiveness": it["intrusiveness"],
            "declinable": it["declinable"],
            "onboarding_order": it["onboarding_order"],
        })

    # 5. Convert scales to JSON-serializable
    scales_json = []
    for sc in scales:
        scales_json.append({
            "id": sc["id"],
            "questionnaire_id": sc["questionnaire_id"],
            "version": sc["version"],
            "name": sc["name"],
            "response_type": sc["response_type"],
            "min_items_required": sc["min_items_required"],
            "unlock_window_days": sc["unlock_window_days"],
            "retest_interval_days": sc["retest_interval_days"],
            "tags": list(sc["tags"]),
            "scoring": {
                "method": sc["method"],
                "normalize_min": sc["normalize_min"],
                "normalize_max": sc["normalize_max"],
            },
            "items": sc["items"],
            "ewma_alpha": sc["ewma_alpha"],
            "citation": sc["citation"],
        })

    # 6. Merge v2 cardiometabolic
    merge_v2_cardiometabolic(items_json, scales_json, questionnaires, existing_ids)

    # 7. Add CASP-19 bundle
    merge_casp19(items_json, scales_json, questionnaires, existing_ids)

    # 8. Build multiplex map
    mux = compute_multiplex_map(scales_json)
    shared = {k: v for k, v in mux.items() if len(v) > 1}
    print(f"  Multiplex map: {len(mux)} items total, {len(shared)} shared across scales")

    # 9. Write output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "items.json", "w") as f:
        json.dump(items_json, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "scales.json", "w") as f:
        json.dump(scales_json, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "questionnaires.json", "w") as f:
        json.dump(questionnaires, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "response_types.json", "w") as f:
        json.dump(build_response_types_json(), f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "multiplex_map.json", "w") as f:
        json.dump(mux, f, indent=2, ensure_ascii=False)

    # 10. Summary stats
    print("\n=== v3 PRODUCTION REGISTRY ===")
    print(f"  Items:          {len(items_json)}")
    print(f"  Scales:         {len(scales_json)}")
    print(f"  Questionnaires: {len(questionnaires)}")
    print(f"  Response types: {len(RESPONSE_TYPE_MAP)}")
    print(f"  Shared items:   {len(shared)} (multiplex)")
    print(f"  Written to:     {OUT_DIR}")

    # 11. Instrument audit summary
    print("\n=== AUDIT FLAGS ===")
    for sc in scales_json:
        tags = sc.get("tags", [])
        audit_tags = [t for t in tags if str(t).startswith("audit:")]
        if audit_tags:
            print(f"  {sc['name']:35s}  {', '.join(audit_tags)}")


if __name__ == "__main__":
    main()
