from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from questions_agent_platform.contracts import (
    CONTRACT_FUSION_TO_QUESTIONS_CONTEXT,
    CONTRACT_FUSION_TO_QUESTIONS_OUTCOME,
    canonical_contract_version,
)


class QuestionOption(BaseModel):
    value: float
    label: str


class QuestionOut(BaseModel):
    item_id: str
    text: str
    response_type: str
    options: List[QuestionOption]
    tags: List[str]
    sensitivity: str
    progress_hint: Optional[str] = None
    feeds_scales: Optional[List[str]] = None


class DailyQuestionsOut(BaseModel):
    session_id: str
    user_id: str
    date: str
    registry_version: str
    timeframe: Optional[str] = None
    status: Optional[str] = None
    questions: List[QuestionOut]
    extra: Dict[str, Any]
    selection_explain: Dict[str, Any]
    scale_progress: Dict[str, Any]


class AnswerIn(BaseModel):
    client_event_id: str
    item_id: str
    value: Optional[float] = None
    answered_at: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None  # behavioural metadata (latency, edits, etc.)

    @model_validator(mode="after")
    def _validate_value_or_decline(self) -> "AnswerIn":
        raw = self.raw
        permanently_declined = isinstance(raw, dict) and bool(raw.get("permanently_declined"))
        if not permanently_declined and self.value is None:
            raise ValueError("value is required unless raw.permanently_declined=true")
        return self


class SubmitAnswersIn(BaseModel):
    session_id: str
    answers: List[AnswerIn]


class SessionUncertaintyProfileOut(BaseModel):
    session_id: str
    user_id: str
    session_date: str
    session_multiplier: float
    engagement_quality: str
    median_latency_ms: float
    total_edits: int
    skip_count: int
    decline_count: int
    item_count: int


class SubmitAnswersOut(BaseModel):
    inserted_answer_events: int
    new_scale_scores: List[Dict[str, Any]]
    scale_progress: Dict[str, Any]
    session_engagement: Optional[SessionUncertaintyProfileOut] = None


class RequestMoreContextIn(BaseModel):
    session_id: str


class ObservationIn(BaseModel):
    type: str
    observed_at: Optional[str] = None
    features: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[str] = "medium"
    provenance: Dict[str, Any] = Field(default_factory=dict)


class SubmitObservationsIn(BaseModel):
    observations: List[ObservationIn]


class SubmitObservationsOut(BaseModel):
    inserted_observation_events: int
    coupling_outputs: List[Dict[str, Any]] = Field(default_factory=list)


class RegistryUploadIn(BaseModel):
    version: str
    items: List[Dict[str, Any]]
    questionnaires: List[Dict[str, Any]]
    scales: List[Dict[str, Any]]


class PolicyContextIn(BaseModel):
    contract_name: str = CONTRACT_FUSION_TO_QUESTIONS_CONTEXT
    schema_version: str = Field(default_factory=lambda: canonical_contract_version(None))
    anifold_z: Optional[List[float]] = None
    z_uncertainty_diag: Optional[List[float]] = None
    z_velocity: Optional[List[float]] = None
    z_distance_to_attractor: Optional[float] = None

    completion_rate_7d: Optional[float] = None
    completion_rate_14d: Optional[float] = None
    completion_rate_30d: Optional[float] = None
    burden_ms_median_14d: Optional[float] = None

    safety_trigger_active: Optional[bool] = False
    allow_context_batches: Optional[bool] = True

    @field_validator("schema_version", mode="before")
    @classmethod
    def _normalize_schema_version(cls, value: Optional[str]) -> str:
        return canonical_contract_version(value)

    @field_validator("contract_name", mode="before")
    @classmethod
    def _validate_contract_name(cls, value: Optional[str]) -> str:
        v = str(value or CONTRACT_FUSION_TO_QUESTIONS_CONTEXT).strip()
        if v != CONTRACT_FUSION_TO_QUESTIONS_CONTEXT:
            raise ValueError(f"contract_name must be {CONTRACT_FUSION_TO_QUESTIONS_CONTEXT}")
        return v

    @model_validator(mode="after")
    def _validate_vector_consistency(self) -> "PolicyContextIn":
        z = self.anifold_z
        u = self.z_uncertainty_diag
        vel = self.z_velocity
        if isinstance(z, list):
            if isinstance(u, list) and len(u) != len(z):
                raise ValueError("z_uncertainty_diag length must match anifold_z length")
            if isinstance(vel, list) and len(vel) != len(z):
                raise ValueError("z_velocity length must match anifold_z length")
        return self

    model_config = ConfigDict(extra="forbid")


class DailyQuestionsSelectIn(BaseModel):
    schema_version: str = Field(default_factory=lambda: canonical_contract_version(None))
    date: Optional[str] = None
    selection_mode: str = "deterministic"  # deterministic | policy_live | policy_shadow | policy_offline_replay
    k_core: Optional[int] = None
    allow_context_batches: Optional[bool] = None
    identity_mask_id: Optional[str] = None
    include_explanations: Optional[bool] = False
    context: Optional[PolicyContextIn] = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def _normalize_schema_version(cls, value: Optional[str]) -> str:
        return canonical_contract_version(value)

    @field_validator("selection_mode")
    @classmethod
    def _validate_selection_mode(cls, value: str) -> str:
        v = str(value or "").strip().lower()
        allowed = {"deterministic", "policy_live", "policy_shadow", "policy_offline_replay"}
        if v not in allowed:
            raise ValueError(f"selection_mode must be one of {sorted(allowed)}")
        return v

    model_config = ConfigDict(extra="forbid")


class PolicyOutcomeUpdateIn(BaseModel):
    contract_name: str = CONTRACT_FUSION_TO_QUESTIONS_OUTCOME
    schema_version: str = Field(default_factory=lambda: canonical_contract_version(None))
    source_event_id: Optional[str] = None
    decision_id: str
    # Optional Anifold evidence updates (for delayed reward components)
    z_before: Optional[List[float]] = None
    z_after: Optional[List[float]] = None
    uncertainty_before_diag: Optional[List[float]] = None
    uncertainty_after_diag: Optional[List[float]] = None
    reward_overrides: Optional[Dict[str, Any]] = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def _normalize_schema_version(cls, value: Optional[str]) -> str:
        return canonical_contract_version(value)

    @field_validator("contract_name", mode="before")
    @classmethod
    def _validate_contract_name(cls, value: Optional[str]) -> str:
        v = str(value or CONTRACT_FUSION_TO_QUESTIONS_OUTCOME).strip()
        if v != CONTRACT_FUSION_TO_QUESTIONS_OUTCOME:
            raise ValueError(f"contract_name must be {CONTRACT_FUSION_TO_QUESTIONS_OUTCOME}")
        return v

    @model_validator(mode="after")
    def _validate_outcome_vectors(self) -> "PolicyOutcomeUpdateIn":
        zb = self.z_before
        za = self.z_after
        ub = self.uncertainty_before_diag
        ua = self.uncertainty_after_diag
        if isinstance(zb, list) and isinstance(za, list) and len(zb) != len(za):
            raise ValueError("z_before and z_after must have equal length")
        if isinstance(ub, list) and isinstance(ua, list) and len(ub) != len(ua):
            raise ValueError("uncertainty_before_diag and uncertainty_after_diag must have equal length")
        if isinstance(zb, list) and isinstance(ub, list) and len(zb) != len(ub):
            raise ValueError("uncertainty_before_diag length must match z_before length")
        if isinstance(za, list) and isinstance(ua, list) and len(za) != len(ua):
            raise ValueError("uncertainty_after_diag length must match z_after length")
        return self

    model_config = ConfigDict(extra="forbid")


class UserProfilePatchIn(BaseModel):
    mode: Optional[str] = None
    site_config_id: Optional[str] = None
    onboarding_complete: Optional[bool] = None
    active_domains: Optional[List[str]] = None
    queued_domains: Optional[List[str]] = None
    promoted_domains: Optional[List[str]] = None

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = str(value).strip().lower()
        if v not in {"trial", "consumer"}:
            raise ValueError("mode must be one of: trial, consumer")
        return v

    model_config = ConfigDict(extra="forbid")


class SafetyResolveIn(BaseModel):
    resolved_by: str
    resolved_reason: str

    model_config = ConfigDict(extra="forbid")


class ExperimentUpsertIn(BaseModel):
    experiment_id: Optional[str] = None
    name: Optional[str] = None
    status: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    baseline_window_days: Optional[int] = None
    eval_window_days: Optional[int] = None
    target_metrics: Optional[Dict[str, Any]] = None
    stopping_rules: Optional[Dict[str, Any]] = None
    intervention: Optional[Dict[str, Any]] = None

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = str(value).strip().lower()
        if v not in {"active", "paused", "completed", "cancelled"}:
            raise ValueError("status must be one of: active, paused, completed, cancelled")
        return v

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# N-of-1 Calibration
# ---------------------------------------------------------------------------

class CalibrationOut(BaseModel):
    user_id: str
    scale_id: str
    phase: str
    n_observations: int
    theta_personal: float
    se_personal: float
    theta_baseline: Optional[float] = None
    se_baseline: Optional[float] = None
    within_person_sd: Optional[float] = None
    shrinkage: float
    reliable_change: Optional[Dict[str, Any]] = None
    sufficiency: Optional[Dict[str, Any]] = None


class CalibrationListOut(BaseModel):
    user_id: str
    calibrations: List[CalibrationOut]


# ---------------------------------------------------------------------------
# Coherence Assessment
# ---------------------------------------------------------------------------

class CoherenceOut(BaseModel):
    session_id: str
    user_id: str
    date: str
    coherence_score: float
    tier: str
    n_flagged: int
    n_items: int
    signals: List[Dict[str, Any]]


class CoherenceHistoryOut(BaseModel):
    user_id: str
    assessments: List[CoherenceOut]


# ---------------------------------------------------------------------------
# Cardiometabolic Risk Index
# ---------------------------------------------------------------------------

class CardioRiskOut(BaseModel):
    user_id: str
    date: str
    composite_risk: float
    composite_risk_pct: float
    risk_tier: str
    instruments_available: int
    instruments_total: int
    coverage: float
    confidence: str
    components: List[Dict[str, Any]]


class CardioRiskHistoryOut(BaseModel):
    user_id: str
    snapshots: List[CardioRiskOut]


# ---------------------------------------------------------------------------
# Site Configuration
# ---------------------------------------------------------------------------

class SiteConfigOut(BaseModel):
    config_id: str
    display_name: str
    config_type: str
    registry_version: str = "v2"
    locale: str = "en"
    show_clinical_thresholds: bool = False
    show_disease_names: bool = False
    show_risk_tiers: bool = True
    score_display_mode: str = "wellness"
    wellness_language: bool = True
    anifold_enabled: bool = False
    concordance_mode: bool = False
    behavioural_metadata_enabled: bool = True
    anamnesis_enabled: bool = True
    policy_mode: str = "deterministic"
    daily_question_budget: int = 5
    max_extra_batches: int = 3
    item_repeat_cooldown_days: int = 7


class SiteConfigListOut(BaseModel):
    configs: List[SiteConfigOut]


# ---------------------------------------------------------------------------
# Response Metadata
# ---------------------------------------------------------------------------

class ResponseMetadataOut(BaseModel):
    metadata_id: str
    item_id: str
    response_latency_ms: Optional[float] = None
    edit_count: int
    was_skipped: bool
    was_declined: bool
    channel: str
    uncertainty_multiplier: float
    confidence_label: str
    contributing_factors: List[str]


# ---------------------------------------------------------------------------
# Anamnesis Episodes
# ---------------------------------------------------------------------------

class AnamnesisEpisodeOut(BaseModel):
    episode_id: str
    user_id: str
    trigger_date: str
    drift_domain: str
    trigger_type: str
    trigger_value: float
    triggered_scale_ids: List[str]
    status: str
    follow_up_item_ids: List[str]
    priority: int
    max_days: int
    observations_collected: int
    resolution_date: Optional[str] = None
    resolution_reason: Optional[str] = None


class AnamnesisEpisodeListOut(BaseModel):
    user_id: str
    episodes: List[AnamnesisEpisodeOut]


# ---------------------------------------------------------------------------
# Concordance Reports
# ---------------------------------------------------------------------------

class ConcordanceReportOut(BaseModel):
    instrument_id: str
    n_participants: int
    n_paired_observations: int
    icc: Dict[str, Any]
    bland_altman: Dict[str, Any]
    kappa: Dict[str, Any]
    overall_pass: bool
    failure_reasons: List[str]


class ConcordanceSummaryOut(BaseModel):
    reports: List[ConcordanceReportOut]
    summary: Dict[str, Any]
