-- Questions Agent (Prod) - PostgreSQL schema

CREATE TABLE IF NOT EXISTS registry_versions (
  version TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_sessions (
  session_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  date DATE NOT NULL,
  registry_version TEXT NOT NULL REFERENCES registry_versions(version),
  timeframe TEXT NOT NULL DEFAULT 'last_7_days',
  status TEXT NOT NULL DEFAULT 'created',
  selection_mode TEXT NOT NULL DEFAULT 'deterministic',
  policy_decision_id TEXT,
  core_questions_json TEXT NOT NULL,
  extra_batches_json TEXT NOT NULL,
  extra_batches_used INTEGER NOT NULL DEFAULT 0,
  selection_explain_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(user_id, date)
);

-- Backward-compatible upgrade (no-op on fresh deploy)
ALTER TABLE daily_sessions ADD COLUMN IF NOT EXISTS selection_mode TEXT NOT NULL DEFAULT 'deterministic';
ALTER TABLE daily_sessions ADD COLUMN IF NOT EXISTS policy_decision_id TEXT;
ALTER TABLE daily_sessions ADD COLUMN IF NOT EXISTS timeframe TEXT NOT NULL DEFAULT 'last_7_days';
ALTER TABLE daily_sessions ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'created';

CREATE TABLE IF NOT EXISTS user_profiles (
  user_id TEXT PRIMARY KEY REFERENCES users(user_id),
  mode TEXT NOT NULL DEFAULT 'consumer',
  permanently_declined_items_json TEXT NOT NULL DEFAULT '[]',
  active_domains_json TEXT NOT NULL DEFAULT '["cardiometabolic"]',
  queued_domains_json TEXT NOT NULL DEFAULT '[]',
  promoted_domains_json TEXT NOT NULL DEFAULT '["cardiometabolic"]',
  onboarding_complete BOOLEAN NOT NULL DEFAULT FALSE,
  state_schema_version TEXT NOT NULL DEFAULT 'v1_state_schema',
  updated_at TEXT NOT NULL
);

ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS active_domains_json TEXT NOT NULL DEFAULT '["cardiometabolic"]';
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS queued_domains_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS promoted_domains_json TEXT NOT NULL DEFAULT '["cardiometabolic"]';

CREATE TABLE IF NOT EXISTS answer_events (
  event_id TEXT PRIMARY KEY,
  client_event_id TEXT NOT NULL,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  session_id TEXT NOT NULL REFERENCES daily_sessions(session_id),
  item_id TEXT NOT NULL,
  answered_at TEXT NOT NULL,
  value DOUBLE PRECISION NOT NULL,
  raw_json TEXT,
  UNIQUE(user_id, client_event_id)
);

CREATE TABLE IF NOT EXISTS scale_evidence (
  evidence_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  answer_event_id TEXT NOT NULL REFERENCES answer_events(event_id),
  session_id TEXT NOT NULL REFERENCES daily_sessions(session_id),
  item_id TEXT NOT NULL,
  scale_id TEXT NOT NULL,
  weight DOUBLE PRECISION NOT NULL,
  contribution_value DOUBLE PRECISION NOT NULL,
  timeframe TEXT NOT NULL,
  window_date DATE NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scale_baselines (
  user_id TEXT NOT NULL REFERENCES users(user_id),
  scale_id TEXT NOT NULL,
  mean DOUBLE PRECISION NOT NULL,
  var DOUBLE PRECISION NOT NULL,
  n INTEGER NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(user_id, scale_id)
);

CREATE TABLE IF NOT EXISTS scale_scores (
  score_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  scale_id TEXT NOT NULL,
  computed_at TEXT NOT NULL,
  window_start DATE NOT NULL,
  window_end DATE NOT NULL,
  raw_score DOUBLE PRECISION NOT NULL,
  normalized_score DOUBLE PRECISION NOT NULL,
  confidence_tier TEXT NOT NULL,
  items_answered_count INTEGER NOT NULL,
  items_required INTEGER NOT NULL,
  baseline_mean DOUBLE PRECISION,
  baseline_std DOUBLE PRECISION,
  personal_z DOUBLE PRECISION,
  delta_vs_prev DOUBLE PRECISION,
  risk_tier TEXT
);

ALTER TABLE scale_scores ADD COLUMN IF NOT EXISTS risk_tier TEXT;

CREATE TABLE IF NOT EXISTS observation_events (
  event_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  observed_at TEXT NOT NULL,
  type TEXT NOT NULL,
  features_json TEXT NOT NULL,
  confidence TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  coupling_json TEXT
);

ALTER TABLE observation_events ADD COLUMN IF NOT EXISTS coupling_json TEXT;

CREATE TABLE IF NOT EXISTS safety_events (
  event_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  session_id TEXT REFERENCES daily_sessions(session_id),
  date DATE NOT NULL,
  trigger_source TEXT NOT NULL,
  severity TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',
  item_id TEXT,
  reason_code TEXT NOT NULL,
  details_json TEXT,
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  resolved_by TEXT,
  resolved_reason TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  start_date DATE NOT NULL,
  end_date DATE,
  target_metrics_json TEXT NOT NULL DEFAULT '{}',
  baseline_window_days INTEGER NOT NULL DEFAULT 14,
  eval_window_days INTEGER NOT NULL DEFAULT 14,
  stopping_rules_json TEXT NOT NULL DEFAULT '{}',
  intervention_json TEXT NOT NULL DEFAULT '{}',
  results_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projections_questions (
  user_id TEXT NOT NULL REFERENCES users(user_id),
  timestamp TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY(user_id, timestamp)
);

CREATE TABLE IF NOT EXISTS follow_up_queue (
  queue_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  item_id TEXT NOT NULL,
  scale_id TEXT,
  reason_code TEXT NOT NULL,
  reason_detail_json TEXT,
  priority DOUBLE PRECISION NOT NULL,
  earliest_date DATE NOT NULL,
  timeframe TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  resolved_reason TEXT
);

CREATE TABLE IF NOT EXISTS state_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  date DATE NOT NULL,
  timestamp TEXT NOT NULL,
  state_schema_version TEXT NOT NULL,
  model_version TEXT NOT NULL,
  x_hat_json TEXT NOT NULL,
  x_uncertainty_json TEXT NOT NULL,
  coverage_json TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'questions_agent',
  created_at TEXT NOT NULL,
  UNIQUE(user_id, date, state_schema_version, model_version)
);

CREATE TABLE IF NOT EXISTS circle_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  date DATE NOT NULL,
  timestamp TEXT NOT NULL,
  state_snapshot_id TEXT REFERENCES state_snapshots(snapshot_id),
  projection_version TEXT NOT NULL,
  anchor_version TEXT NOT NULL,
  z_json TEXT NOT NULL,
  z_star_json TEXT NOT NULL,
  r DOUBLE PRECISION NOT NULL,
  theta DOUBLE PRECISION NOT NULL,
  velocity DOUBLE PRECISION NOT NULL,
  acceleration DOUBLE PRECISION NOT NULL,
  coherence_score DOUBLE PRECISION,
  coherence_tier_json TEXT,
  uncertainty_json TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'questions_agent',
  created_at TEXT NOT NULL,
  UNIQUE(user_id, date, projection_version)
);

ALTER TABLE circle_snapshots ADD COLUMN IF NOT EXISTS coherence_score DOUBLE PRECISION;
ALTER TABLE circle_snapshots ADD COLUMN IF NOT EXISTS coherence_tier_json TEXT;

CREATE TABLE IF NOT EXISTS ews_features (
  feature_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  date DATE NOT NULL,
  timestamp TEXT NOT NULL,
  window_size_days INTEGER NOT NULL,
  var_r DOUBLE PRECISION NOT NULL,
  ac1_r DOUBLE PRECISION NOT NULL,
  trend_speed DOUBLE PRECISION NOT NULL,
  trend_accel DOUBLE PRECISION NOT NULL,
  recovery_rate DOUBLE PRECISION NOT NULL,
  ews_score DOUBLE PRECISION NOT NULL,
  notes_json TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(user_id, date, window_size_days)
);

CREATE TABLE IF NOT EXISTS drift_events (
  event_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  date DATE NOT NULL,
  timestamp TEXT NOT NULL,
  coherence_score DOUBLE PRECISION NOT NULL,
  drift_domain TEXT NOT NULL,
  triggered_instrument TEXT,
  ews_score DOUBLE PRECISION,
  reason_codes_json TEXT NOT NULL,
  details_json TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(user_id, date)
);

-- Policy layer (constrained contextual bandit)
CREATE TABLE IF NOT EXISTS policy_decisions (
  decision_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  date DATE NOT NULL,
  identity_mask_id TEXT,
  selection_mode TEXT NOT NULL,
  mode TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  feature_version TEXT NOT NULL,
  context_hash TEXT NOT NULL,
  candidate_set_hash TEXT NOT NULL,
  context_json TEXT,
  candidate_set_json TEXT,
  selected_item_ids_json TEXT NOT NULL,
  deterministic_baseline_selected_json TEXT NOT NULL,
  propensities_json TEXT,
  explanations_json TEXT NOT NULL,
  counterfactuals_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_policy_decisions_user_date ON policy_decisions(user_id, date);

CREATE TABLE IF NOT EXISTS policy_outcomes (
  decision_id TEXT PRIMARY KEY REFERENCES policy_decisions(decision_id),
  user_id TEXT NOT NULL REFERENCES users(user_id),
  date DATE NOT NULL,
  completion_rate DOUBLE PRECISION,
  completed_core_count INTEGER,
  response_time_ms_median DOUBLE PRECISION,
  z_before_hash TEXT,
  z_after_hash TEXT,
  z_before_json TEXT,
  z_after_json TEXT,
  z_delta_json TEXT,
  z_before_norm DOUBLE PRECISION,
  z_after_norm DOUBLE PRECISION,
  z_delta_norm DOUBLE PRECISION,
  uncertainty_before_mean DOUBLE PRECISION,
  uncertainty_after_mean DOUBLE PRECISION,
  uncertainty_before_max DOUBLE PRECISION,
  uncertainty_after_max DOUBLE PRECISION,
  reward_components_json TEXT,
  total_reward DOUBLE PRECISION,
  updated_at TEXT NOT NULL
);

-- Backward-compatible upgrade columns (no-op on fresh deploy)
ALTER TABLE policy_outcomes ADD COLUMN IF NOT EXISTS z_before_json TEXT;
ALTER TABLE policy_outcomes ADD COLUMN IF NOT EXISTS z_after_json TEXT;
ALTER TABLE policy_outcomes ADD COLUMN IF NOT EXISTS z_delta_json TEXT;
ALTER TABLE policy_outcomes ADD COLUMN IF NOT EXISTS z_before_norm DOUBLE PRECISION;
ALTER TABLE policy_outcomes ADD COLUMN IF NOT EXISTS z_after_norm DOUBLE PRECISION;
ALTER TABLE policy_outcomes ADD COLUMN IF NOT EXISTS z_delta_norm DOUBLE PRECISION;
ALTER TABLE policy_outcomes ADD COLUMN IF NOT EXISTS uncertainty_before_max DOUBLE PRECISION;
ALTER TABLE policy_outcomes ADD COLUMN IF NOT EXISTS uncertainty_after_max DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_answer_events_user_time ON answer_events(user_id, answered_at);
CREATE INDEX IF NOT EXISTS idx_scale_scores_user_scale_time ON scale_scores(user_id, scale_id, computed_at);
CREATE INDEX IF NOT EXISTS idx_observation_events_user_time ON observation_events(user_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_safety_events_user_status_date ON safety_events(user_id, status, date);
CREATE INDEX IF NOT EXISTS idx_safety_events_session ON safety_events(session_id);
CREATE INDEX IF NOT EXISTS idx_experiments_user_status_start ON experiments(user_id, status, start_date);
CREATE INDEX IF NOT EXISTS idx_scale_evidence_user_scale_date ON scale_evidence(user_id, scale_id, window_date);
CREATE INDEX IF NOT EXISTS idx_scale_evidence_answer_event ON scale_evidence(answer_event_id);
CREATE INDEX IF NOT EXISTS idx_follow_up_queue_due ON follow_up_queue(user_id, status, earliest_date, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_follow_up_queue_item ON follow_up_queue(user_id, item_id, status);
CREATE INDEX IF NOT EXISTS idx_follow_up_queue_scale ON follow_up_queue(user_id, scale_id, status);
CREATE INDEX IF NOT EXISTS idx_state_snapshots_user_date ON state_snapshots(user_id, date);
CREATE INDEX IF NOT EXISTS idx_circle_snapshots_user_date ON circle_snapshots(user_id, date);
CREATE INDEX IF NOT EXISTS idx_ews_features_user_date ON ews_features(user_id, date);
CREATE INDEX IF NOT EXISTS idx_drift_events_user_date ON drift_events(user_id, date);
