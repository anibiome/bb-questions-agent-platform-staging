from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class RegistryVersion(Base):
    __tablename__ = "registry_versions"
    version = Column(String, primary_key=True)
    status = Column(String, nullable=False)
    created_at = Column(String, nullable=False)


class User(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True)
    created_at = Column(String, nullable=False)


class UserProfile(Base):
    __tablename__ = "user_profiles"
    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    mode = Column(String, nullable=False, default="consumer")
    site_config_id = Column(String, nullable=False, default="consumer")
    permanently_declined_items_json = Column(Text, nullable=False, default="[]")
    active_domains_json = Column(Text, nullable=False, default='["cardiometabolic"]')
    queued_domains_json = Column(Text, nullable=False, default="[]")
    promoted_domains_json = Column(Text, nullable=False, default='["cardiometabolic"]')
    onboarding_complete = Column(Boolean, nullable=False, default=False)
    state_schema_version = Column(String, nullable=False, default="v1_state_schema")
    updated_at = Column(String, nullable=False)


class DailySession(Base):
    __tablename__ = "daily_sessions"
    session_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    date = Column(Date, nullable=False)
    registry_version = Column(String, ForeignKey("registry_versions.version"), nullable=False)
    timeframe = Column(String, nullable=False, default="last_7_days")
    status = Column(String, nullable=False, default="created")
    selection_mode = Column(String, nullable=False, default="deterministic")
    policy_decision_id = Column(String, nullable=True)
    core_questions_json = Column(Text, nullable=False)
    extra_batches_json = Column(Text, nullable=False)
    extra_batches_used = Column(Integer, nullable=False, default=0)
    selection_explain_json = Column(Text, nullable=False)
    created_at = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "date", name="uq_session_user_date"),)


class AnswerEvent(Base):
    __tablename__ = "answer_events"
    event_id = Column(String, primary_key=True)
    client_event_id = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    session_id = Column(String, ForeignKey("daily_sessions.session_id"), nullable=False)
    item_id = Column(String, nullable=False)
    answered_at = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    raw_json = Column(Text, nullable=True)

    __table_args__ = (UniqueConstraint("user_id", "client_event_id", name="uq_answer_idempotency"),)


class ScaleEvidence(Base):
    __tablename__ = "scale_evidence"
    evidence_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    answer_event_id = Column(String, ForeignKey("answer_events.event_id"), nullable=False)
    session_id = Column(String, ForeignKey("daily_sessions.session_id"), nullable=False)
    item_id = Column(String, nullable=False)
    scale_id = Column(String, nullable=False)
    weight = Column(Float, nullable=False)
    contribution_value = Column(Float, nullable=False)
    timeframe = Column(String, nullable=False)
    window_date = Column(Date, nullable=False)
    created_at = Column(String, nullable=False)


class ScaleBaseline(Base):
    __tablename__ = "scale_baselines"
    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    scale_id = Column(String, primary_key=True)
    mean = Column(Float, nullable=False)
    var = Column(Float, nullable=False)
    n = Column(Integer, nullable=False)
    updated_at = Column(String, nullable=False)


class ScaleScore(Base):
    __tablename__ = "scale_scores"
    score_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    scale_id = Column(String, nullable=False)
    computed_at = Column(String, nullable=False)
    window_start = Column(Date, nullable=False)
    window_end = Column(Date, nullable=False)
    raw_score = Column(Float, nullable=False)
    normalized_score = Column(Float, nullable=False)
    confidence_tier = Column(String, nullable=False)
    items_answered_count = Column(Integer, nullable=False)
    items_required = Column(Integer, nullable=False)
    baseline_mean = Column(Float, nullable=True)
    baseline_std = Column(Float, nullable=True)
    personal_z = Column(Float, nullable=True)
    delta_vs_prev = Column(Float, nullable=True)
    risk_tier = Column(String, nullable=True)
    se_theta = Column(Float, nullable=True)
    se_theta_adjusted = Column(Float, nullable=True)
    uncertainty_multiplier = Column(Float, nullable=True)
    engagement_quality = Column(String, nullable=True)


class ObservationEvent(Base):
    __tablename__ = "observation_events"
    event_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    observed_at = Column(String, nullable=False)
    type = Column(String, nullable=False)
    features_json = Column(Text, nullable=False)
    confidence = Column(String, nullable=False)
    provenance_json = Column(Text, nullable=False)
    coupling_json = Column(Text, nullable=True)


class SafetyEvent(Base):
    __tablename__ = "safety_events"
    event_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    session_id = Column(String, ForeignKey("daily_sessions.session_id"), nullable=True)
    date = Column(Date, nullable=False)
    trigger_source = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    status = Column(String, nullable=False, default="open")
    item_id = Column(String, nullable=True)
    reason_code = Column(String, nullable=False)
    details_json = Column(Text, nullable=True)
    created_at = Column(String, nullable=False)
    resolved_at = Column(String, nullable=True)
    resolved_by = Column(String, nullable=True)
    resolved_reason = Column(String, nullable=True)


class Experiment(Base):
    __tablename__ = "experiments"
    experiment_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    name = Column(String, nullable=False)
    status = Column(String, nullable=False, default="active")
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=True)
    target_metrics_json = Column(Text, nullable=False, default="{}")
    baseline_window_days = Column(Integer, nullable=False, default=14)
    eval_window_days = Column(Integer, nullable=False, default=14)
    stopping_rules_json = Column(Text, nullable=False, default="{}")
    intervention_json = Column(Text, nullable=False, default="{}")
    results_json = Column(Text, nullable=True)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)


class QuestionsProjection(Base):
    __tablename__ = "projections_questions"
    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    timestamp = Column(String, primary_key=True)
    payload_json = Column(Text, nullable=False)


class FollowUpQueueItem(Base):
    __tablename__ = "follow_up_queue"
    queue_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    item_id = Column(String, nullable=False)
    scale_id = Column(String, nullable=True)
    reason_code = Column(String, nullable=False)
    reason_detail_json = Column(Text, nullable=True)
    priority = Column(Float, nullable=False)
    earliest_date = Column(Date, nullable=False)
    timeframe = Column(String, nullable=True)
    status = Column(String, nullable=False, default="pending")
    created_at = Column(String, nullable=False)
    resolved_at = Column(String, nullable=True)
    resolved_reason = Column(String, nullable=True)


class StateSnapshot(Base):
    __tablename__ = "state_snapshots"
    snapshot_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    date = Column(Date, nullable=False)
    timestamp = Column(String, nullable=False)
    state_schema_version = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    x_hat_json = Column(Text, nullable=False)
    x_uncertainty_json = Column(Text, nullable=False)
    coverage_json = Column(Text, nullable=False)
    source = Column(String, nullable=False, default="questions_agent")
    created_at = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "date", "state_schema_version", "model_version", name="uq_state_snapshot_key"),)


class CircleSnapshot(Base):
    __tablename__ = "circle_snapshots"
    snapshot_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    date = Column(Date, nullable=False)
    timestamp = Column(String, nullable=False)
    state_snapshot_id = Column(String, ForeignKey("state_snapshots.snapshot_id"), nullable=True)
    projection_version = Column(String, nullable=False)
    anchor_version = Column(String, nullable=False)
    z_json = Column(Text, nullable=False)
    z_star_json = Column(Text, nullable=False)
    r = Column(Float, nullable=False)
    theta = Column(Float, nullable=False)
    velocity = Column(Float, nullable=False)
    acceleration = Column(Float, nullable=False)
    coherence_score = Column(Float, nullable=True)
    coherence_tier_json = Column(Text, nullable=True)
    uncertainty_json = Column(Text, nullable=False)
    source = Column(String, nullable=False, default="questions_agent")
    created_at = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "date", "projection_version", name="uq_circle_snapshot_key"),)


class EwsFeature(Base):
    __tablename__ = "ews_features"
    feature_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    date = Column(Date, nullable=False)
    timestamp = Column(String, nullable=False)
    window_size_days = Column(Integer, nullable=False)
    var_r = Column(Float, nullable=False)
    ac1_r = Column(Float, nullable=False)
    trend_speed = Column(Float, nullable=False)
    trend_accel = Column(Float, nullable=False)
    recovery_rate = Column(Float, nullable=False)
    ews_score = Column(Float, nullable=False)
    notes_json = Column(Text, nullable=True)
    created_at = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "date", "window_size_days", name="uq_ews_features_user_date_window"),)


class DriftEvent(Base):
    __tablename__ = "drift_events"
    event_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    date = Column(Date, nullable=False)
    timestamp = Column(String, nullable=False)
    coherence_score = Column(Float, nullable=False)
    drift_domain = Column(String, nullable=False)
    triggered_instrument = Column(String, nullable=True)
    ews_score = Column(Float, nullable=True)
    reason_codes_json = Column(Text, nullable=False)
    details_json = Column(Text, nullable=True)
    created_at = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "date", name="uq_drift_events_user_date"),)


class PolicyDecision(Base):
    __tablename__ = "policy_decisions"
    decision_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    date = Column(Date, nullable=False)
    identity_mask_id = Column(String, nullable=True)
    selection_mode = Column(String, nullable=False)
    mode = Column(String, nullable=False)
    policy_version = Column(String, nullable=False)
    feature_version = Column(String, nullable=False)
    context_hash = Column(String, nullable=False)
    candidate_set_hash = Column(String, nullable=False)
    context_json = Column(Text, nullable=True)
    candidate_set_json = Column(Text, nullable=True)
    selected_item_ids_json = Column(Text, nullable=False)
    deterministic_baseline_selected_json = Column(Text, nullable=False)
    propensities_json = Column(Text, nullable=True)
    explanations_json = Column(Text, nullable=False)
    counterfactuals_json = Column(Text, nullable=False)
    created_at = Column(String, nullable=False)


class PolicyOutcome(Base):
    __tablename__ = "policy_outcomes"
    decision_id = Column(String, ForeignKey("policy_decisions.decision_id"), primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    date = Column(Date, nullable=False)
    completion_rate = Column(Float, nullable=True)
    completed_core_count = Column(Integer, nullable=True)
    response_time_ms_median = Column(Float, nullable=True)
    z_before_hash = Column(String, nullable=True)
    z_after_hash = Column(String, nullable=True)
    z_before_json = Column(Text, nullable=True)
    z_after_json = Column(Text, nullable=True)
    z_delta_json = Column(Text, nullable=True)
    z_before_norm = Column(Float, nullable=True)
    z_after_norm = Column(Float, nullable=True)
    z_delta_norm = Column(Float, nullable=True)
    uncertainty_before_mean = Column(Float, nullable=True)
    uncertainty_after_mean = Column(Float, nullable=True)
    uncertainty_before_max = Column(Float, nullable=True)
    uncertainty_after_max = Column(Float, nullable=True)
    reward_components_json = Column(Text, nullable=True)
    total_reward = Column(Float, nullable=True)
    updated_at = Column(String, nullable=False)


class OperationalMetric(Base):
    __tablename__ = "operational_metrics"
    metric_id = Column(String, primary_key=True)
    metric_type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    metric_date = Column(Date, nullable=False)
    latency_ms = Column(Float, nullable=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=True)
    session_id = Column(String, ForeignKey("daily_sessions.session_id"), nullable=True)
    details_json = Column(Text, nullable=True)
    created_at = Column(String, nullable=False)


Index("idx_answer_events_user_time", AnswerEvent.user_id, AnswerEvent.answered_at)
Index("idx_scale_evidence_user_scale_date", ScaleEvidence.user_id, ScaleEvidence.scale_id, ScaleEvidence.window_date)
Index("idx_scale_evidence_answer_event", ScaleEvidence.answer_event_id)
Index("idx_scale_scores_user_scale_time", ScaleScore.user_id, ScaleScore.scale_id, ScaleScore.computed_at)
Index("idx_observation_events_user_time", ObservationEvent.user_id, ObservationEvent.observed_at)
Index("idx_safety_events_user_status_date", SafetyEvent.user_id, SafetyEvent.status, SafetyEvent.date)
Index("idx_safety_events_session", SafetyEvent.session_id)
Index("idx_experiments_user_status_start", Experiment.user_id, Experiment.status, Experiment.start_date)
Index("idx_policy_decisions_user_date", PolicyDecision.user_id, PolicyDecision.date)
Index("idx_follow_up_queue_due", FollowUpQueueItem.user_id, FollowUpQueueItem.status, FollowUpQueueItem.earliest_date, FollowUpQueueItem.priority, FollowUpQueueItem.created_at)
Index("idx_follow_up_queue_item", FollowUpQueueItem.user_id, FollowUpQueueItem.item_id, FollowUpQueueItem.status)
Index("idx_follow_up_queue_scale", FollowUpQueueItem.user_id, FollowUpQueueItem.scale_id, FollowUpQueueItem.status)
Index("idx_state_snapshots_user_date", StateSnapshot.user_id, StateSnapshot.date)
Index("idx_circle_snapshots_user_date", CircleSnapshot.user_id, CircleSnapshot.date)
Index("idx_ews_features_user_date", EwsFeature.user_id, EwsFeature.date)
Index("idx_drift_events_user_date", DriftEvent.user_id, DriftEvent.date)
Index("idx_operational_metrics_type_date_status", OperationalMetric.metric_type, OperationalMetric.metric_date, OperationalMetric.status)


# ---------------------------------------------------------------------------
# N-of-1 Personal Calibration
# ---------------------------------------------------------------------------

class PersonalCalibrationRow(Base):
    __tablename__ = "personal_calibrations"
    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    scale_id = Column(String, primary_key=True)
    n_observations = Column(Integer, nullable=False, default=0)
    phase = Column(String, nullable=False, default="warming")
    theta_personal = Column(Float, nullable=False)
    se_personal = Column(Float, nullable=False)
    theta_baseline = Column(Float, nullable=True)
    se_baseline = Column(Float, nullable=True)
    within_person_sd = Column(Float, nullable=True)
    shrinkage = Column(Float, nullable=False, default=0.0)
    n_effective = Column(Float, nullable=False, default=0.0)
    calibration_json = Column(Text, nullable=False)  # full serialised PersonalCalibration
    updated_at = Column(String, nullable=False)


# ---------------------------------------------------------------------------
# Session Coherence Assessment
# ---------------------------------------------------------------------------

class SessionCoherence(Base):
    __tablename__ = "session_coherence"
    coherence_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    session_id = Column(String, ForeignKey("daily_sessions.session_id"), nullable=False)
    date = Column(Date, nullable=False)
    coherence_score = Column(Float, nullable=False)
    tier = Column(String, nullable=False)
    n_flagged = Column(Integer, nullable=False)
    n_items = Column(Integer, nullable=False)
    signals_json = Column(Text, nullable=False)  # full serialised CoherenceAssessment
    created_at = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "session_id", name="uq_session_coherence"),)


# ---------------------------------------------------------------------------
# Cardiometabolic Risk Index
# ---------------------------------------------------------------------------

class CardioRiskSnapshot(Base):
    __tablename__ = "cardio_risk_snapshots"
    snapshot_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    date = Column(Date, nullable=False)
    composite_risk = Column(Float, nullable=False)
    composite_risk_pct = Column(Float, nullable=False)
    risk_tier = Column(String, nullable=False)
    instruments_available = Column(Integer, nullable=False)
    instruments_total = Column(Integer, nullable=False)
    coverage = Column(Float, nullable=False)
    confidence = Column(String, nullable=False)
    components_json = Column(Text, nullable=False)  # full serialised CardioRiskIndex
    created_at = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "date", name="uq_cardio_risk_user_date"),)


# ---------------------------------------------------------------------------
# Response Metadata (Claim Family 5: Behavioural Uncertainty)
# ---------------------------------------------------------------------------

class ResponseMetadataRow(Base):
    __tablename__ = "response_metadata"
    metadata_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    session_id = Column(String, ForeignKey("daily_sessions.session_id"), nullable=False)
    answer_event_id = Column(String, ForeignKey("answer_events.event_id"), nullable=False)
    item_id = Column(String, nullable=False)
    response_latency_ms = Column(Float, nullable=True)
    edit_count = Column(Integer, nullable=False, default=0)
    was_skipped = Column(Boolean, nullable=False, default=False)
    was_declined = Column(Boolean, nullable=False, default=False)
    channel = Column(String, nullable=False, default="tap")
    voice_hesitation_ms = Column(Float, nullable=True)
    time_of_day_hour = Column(Integer, nullable=True)
    uncertainty_multiplier = Column(Float, nullable=False, default=1.0)
    confidence_label = Column(String, nullable=False, default="medium")
    contributing_factors_json = Column(Text, nullable=False, default="[]")
    raw_components_json = Column(Text, nullable=False, default="{}")
    created_at = Column(String, nullable=False)


class SessionUncertaintyProfileRow(Base):
    __tablename__ = "session_uncertainty_profiles"
    profile_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    session_id = Column(String, ForeignKey("daily_sessions.session_id"), nullable=False)
    session_date = Column(String, nullable=False)
    session_multiplier = Column(Float, nullable=False, default=1.0)
    engagement_quality = Column(String, nullable=False, default="normal")
    median_latency_ms = Column(Float, nullable=False, default=0.0)
    total_edits = Column(Integer, nullable=False, default=0)
    skip_count = Column(Integer, nullable=False, default=0)
    decline_count = Column(Integer, nullable=False, default=0)
    item_count = Column(Integer, nullable=False, default=0)
    created_at = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "session_id", name="uq_session_uncertainty_profile"),)


# ---------------------------------------------------------------------------
# Anamnesis Episodes (Claim Family 4: Drift-Triggered Branching)
# ---------------------------------------------------------------------------

class AnamnesisEpisodeRow(Base):
    __tablename__ = "anamnesis_episodes"
    episode_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    trigger_date = Column(String, nullable=False)
    drift_domain = Column(String, nullable=False)
    trigger_type = Column(String, nullable=False)
    trigger_value = Column(Float, nullable=False)
    triggered_scale_ids_json = Column(Text, nullable=False, default="[]")
    status = Column(String, nullable=False, default="active")
    follow_up_item_ids_json = Column(Text, nullable=False, default="[]")
    priority = Column(Integer, nullable=False, default=100)
    max_days = Column(Integer, nullable=False, default=7)
    observations_collected = Column(Integer, nullable=False, default=0)
    resolution_date = Column(String, nullable=True)
    resolution_reason = Column(String, nullable=True)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

Index("idx_personal_calibrations_user", PersonalCalibrationRow.user_id)
Index("idx_session_coherence_user_date", SessionCoherence.user_id, SessionCoherence.date)
Index("idx_session_coherence_session", SessionCoherence.session_id)
Index("idx_cardio_risk_user_date", CardioRiskSnapshot.user_id, CardioRiskSnapshot.date)
Index("idx_response_metadata_session", ResponseMetadataRow.session_id, ResponseMetadataRow.item_id)
Index("idx_response_metadata_user_session", ResponseMetadataRow.user_id, ResponseMetadataRow.session_id)
Index("idx_session_uncertainty_user_date", SessionUncertaintyProfileRow.user_id, SessionUncertaintyProfileRow.session_date)
Index("idx_anamnesis_episodes_user_status", AnamnesisEpisodeRow.user_id, AnamnesisEpisodeRow.status)
Index("idx_anamnesis_episodes_user_date", AnamnesisEpisodeRow.user_id, AnamnesisEpisodeRow.trigger_date)
