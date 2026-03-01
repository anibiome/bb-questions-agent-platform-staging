import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple


def init_db(database_path: str) -> None:
    with connect(database_path) as conn:
        _apply_migrations(conn)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version INTEGER PRIMARY KEY,
          applied_at TEXT NOT NULL
        );
        """
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations;").fetchall()}

    migrations = [
        (1, _migration_v1),
        (2, _migration_v2),
        (3, _migration_v3),
        (4, _migration_v4),
        (5, _migration_v5),
        (6, _migration_v6),
        (7, _migration_v7),
        (8, _migration_v8),
        (9, _migration_v9),
        (10, _migration_v10),
    ]

    for version, fn in migrations:
        if version in applied:
            continue
        fn(conn)
        conn.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?);",
            (version, _now_iso()),
        )
        conn.commit()


def _migration_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
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
          user_id TEXT NOT NULL,
          date TEXT NOT NULL,
          registry_version TEXT NOT NULL,
          core_questions_json TEXT NOT NULL,
          extra_batches_json TEXT NOT NULL,
          extra_batches_used INTEGER NOT NULL DEFAULT 0,
          selection_explain_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          UNIQUE(user_id, date),
          FOREIGN KEY(user_id) REFERENCES users(user_id),
          FOREIGN KEY(registry_version) REFERENCES registry_versions(version)
        );

        CREATE TABLE IF NOT EXISTS answer_events (
          event_id TEXT PRIMARY KEY,
          client_event_id TEXT NOT NULL,
          user_id TEXT NOT NULL,
          session_id TEXT NOT NULL,
          item_id TEXT NOT NULL,
          answered_at TEXT NOT NULL,
          value REAL NOT NULL,
          raw_json TEXT,
          UNIQUE(user_id, client_event_id),
          FOREIGN KEY(user_id) REFERENCES users(user_id),
          FOREIGN KEY(session_id) REFERENCES daily_sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS scale_baselines (
          user_id TEXT NOT NULL,
          scale_id TEXT NOT NULL,
          mean REAL NOT NULL,
          var REAL NOT NULL,
          n INTEGER NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(user_id, scale_id),
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS scale_scores (
          score_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          scale_id TEXT NOT NULL,
          computed_at TEXT NOT NULL,
          window_start TEXT NOT NULL,
          window_end TEXT NOT NULL,
          raw_score REAL NOT NULL,
          normalized_score REAL NOT NULL,
          confidence_tier TEXT NOT NULL,
          items_answered_count INTEGER NOT NULL,
          items_required INTEGER NOT NULL,
          baseline_mean REAL,
          baseline_std REAL,
          personal_z REAL,
          delta_vs_prev REAL,
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_answer_events_user_time ON answer_events(user_id, answered_at);
        CREATE INDEX IF NOT EXISTS idx_scale_scores_user_scale_time ON scale_scores(user_id, scale_id, computed_at);

        CREATE TABLE IF NOT EXISTS observation_events (
          event_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          observed_at TEXT NOT NULL,
          type TEXT NOT NULL,
          features_json TEXT NOT NULL,
          confidence TEXT NOT NULL,
          provenance_json TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_observation_events_user_time ON observation_events(user_id, observed_at);

        CREATE TABLE IF NOT EXISTS projections_questions (
          user_id TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          PRIMARY KEY(user_id, timestamp),
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );
        """
    )


def _migration_v2(conn: sqlite3.Connection) -> None:
    # Add policy plumbing (constrained contextual bandit) while keeping backward compatibility.
    # SQLite supports ADD COLUMN but not DROP COLUMN; we keep the original schema intact.
    cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(daily_sessions);").fetchall()}
    if "selection_mode" not in cols:
        conn.execute("ALTER TABLE daily_sessions ADD COLUMN selection_mode TEXT NOT NULL DEFAULT 'deterministic';")
    if "policy_decision_id" not in cols:
        conn.execute("ALTER TABLE daily_sessions ADD COLUMN policy_decision_id TEXT;")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS policy_decisions (
          decision_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          date TEXT NOT NULL,
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
          created_at TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_policy_decisions_user_date ON policy_decisions(user_id, date);

        CREATE TABLE IF NOT EXISTS policy_outcomes (
          decision_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          date TEXT NOT NULL,
          completion_rate REAL,
          completed_core_count INTEGER,
          response_time_ms_median REAL,
          z_before_hash TEXT,
          z_after_hash TEXT,
          uncertainty_before_mean REAL,
          uncertainty_after_mean REAL,
          reward_components_json TEXT,
          total_reward REAL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(decision_id) REFERENCES policy_decisions(decision_id),
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );
        """
    )


def _migration_v3(conn: sqlite3.Connection) -> None:
    # Extend policy_outcomes to store privacy-safe Z/uncertainty summaries (and optional quantized deltas).
    cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(policy_outcomes);").fetchall()}

    def add(col: str, ddl: str) -> None:
        if col in cols:
            return
        conn.execute(ddl)

    add("z_before_json", "ALTER TABLE policy_outcomes ADD COLUMN z_before_json TEXT;")
    add("z_after_json", "ALTER TABLE policy_outcomes ADD COLUMN z_after_json TEXT;")
    add("z_delta_json", "ALTER TABLE policy_outcomes ADD COLUMN z_delta_json TEXT;")
    add("z_before_norm", "ALTER TABLE policy_outcomes ADD COLUMN z_before_norm REAL;")
    add("z_after_norm", "ALTER TABLE policy_outcomes ADD COLUMN z_after_norm REAL;")
    add("z_delta_norm", "ALTER TABLE policy_outcomes ADD COLUMN z_delta_norm REAL;")
    add("uncertainty_before_max", "ALTER TABLE policy_outcomes ADD COLUMN uncertainty_before_max REAL;")
    add("uncertainty_after_max", "ALTER TABLE policy_outcomes ADD COLUMN uncertainty_after_max REAL;")


def _migration_v4(conn: sqlite3.Connection) -> None:
    # P0 production-readiness schema: session state/timeframe, user profile flags, multiplex evidence rows.
    cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(daily_sessions);").fetchall()}
    if "timeframe" not in cols:
        conn.execute("ALTER TABLE daily_sessions ADD COLUMN timeframe TEXT NOT NULL DEFAULT 'last_7_days';")
    if "status" not in cols:
        conn.execute("ALTER TABLE daily_sessions ADD COLUMN status TEXT NOT NULL DEFAULT 'created';")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
          user_id TEXT PRIMARY KEY,
          mode TEXT NOT NULL DEFAULT 'consumer',
          permanently_declined_items_json TEXT NOT NULL DEFAULT '[]',
          onboarding_complete INTEGER NOT NULL DEFAULT 0,
          state_schema_version TEXT NOT NULL DEFAULT 'v1_state_schema',
          updated_at TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS scale_evidence (
          evidence_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          answer_event_id TEXT NOT NULL,
          session_id TEXT NOT NULL,
          item_id TEXT NOT NULL,
          scale_id TEXT NOT NULL,
          weight REAL NOT NULL,
          contribution_value REAL NOT NULL,
          timeframe TEXT NOT NULL,
          window_date TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(user_id),
          FOREIGN KEY(answer_event_id) REFERENCES answer_events(event_id),
          FOREIGN KEY(session_id) REFERENCES daily_sessions(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_scale_evidence_user_scale_date
          ON scale_evidence(user_id, scale_id, window_date);
        CREATE INDEX IF NOT EXISTS idx_scale_evidence_answer_event
          ON scale_evidence(answer_event_id);
        """
    )


def _migration_v5(conn: sqlite3.Connection) -> None:
    cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(scale_scores);").fetchall()}
    if "risk_tier" not in cols:
        conn.execute("ALTER TABLE scale_scores ADD COLUMN risk_tier TEXT;")


def _migration_v6(conn: sqlite3.Connection) -> None:
    # P1 runtime: anamnesis follow-up queue + state/circle daily snapshots.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS follow_up_queue (
          queue_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          item_id TEXT NOT NULL,
          scale_id TEXT,
          reason_code TEXT NOT NULL,
          reason_detail_json TEXT,
          priority REAL NOT NULL,
          earliest_date TEXT NOT NULL,
          timeframe TEXT,
          status TEXT NOT NULL DEFAULT 'pending',
          created_at TEXT NOT NULL,
          resolved_at TEXT,
          resolved_reason TEXT,
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_follow_up_queue_due
          ON follow_up_queue(user_id, status, earliest_date, priority, created_at);
        CREATE INDEX IF NOT EXISTS idx_follow_up_queue_item
          ON follow_up_queue(user_id, item_id, status);
        CREATE INDEX IF NOT EXISTS idx_follow_up_queue_scale
          ON follow_up_queue(user_id, scale_id, status);

        CREATE TABLE IF NOT EXISTS state_snapshots (
          snapshot_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          date TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          state_schema_version TEXT NOT NULL,
          model_version TEXT NOT NULL,
          x_hat_json TEXT NOT NULL,
          x_uncertainty_json TEXT NOT NULL,
          coverage_json TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'questions_agent',
          created_at TEXT NOT NULL,
          UNIQUE(user_id, date, state_schema_version, model_version),
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_state_snapshots_user_date
          ON state_snapshots(user_id, date);

        CREATE TABLE IF NOT EXISTS circle_snapshots (
          snapshot_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          date TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          state_snapshot_id TEXT,
          projection_version TEXT NOT NULL,
          anchor_version TEXT NOT NULL,
          z_json TEXT NOT NULL,
          z_star_json TEXT NOT NULL,
          r REAL NOT NULL,
          theta REAL NOT NULL,
          velocity REAL NOT NULL,
          acceleration REAL NOT NULL,
          uncertainty_json TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'questions_agent',
          created_at TEXT NOT NULL,
          UNIQUE(user_id, date, projection_version),
          FOREIGN KEY(user_id) REFERENCES users(user_id),
          FOREIGN KEY(state_snapshot_id) REFERENCES state_snapshots(snapshot_id)
        );

        CREATE INDEX IF NOT EXISTS idx_circle_snapshots_user_date
          ON circle_snapshots(user_id, date);
        """
    )


def _migration_v7(conn: sqlite3.Connection) -> None:
    # Runtime governance + safety + experiments.
    cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(user_profiles);").fetchall()}
    if "active_domains_json" not in cols:
        conn.execute(
            "ALTER TABLE user_profiles ADD COLUMN active_domains_json TEXT NOT NULL DEFAULT '[\"cardiometabolic\"]';"
        )
    if "queued_domains_json" not in cols:
        conn.execute(
            "ALTER TABLE user_profiles ADD COLUMN queued_domains_json TEXT NOT NULL DEFAULT '[]';"
        )
    if "promoted_domains_json" not in cols:
        conn.execute(
            "ALTER TABLE user_profiles ADD COLUMN promoted_domains_json TEXT NOT NULL DEFAULT '[\"cardiometabolic\"]';"
        )

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS safety_events (
          event_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          session_id TEXT,
          date TEXT NOT NULL,
          trigger_source TEXT NOT NULL,
          severity TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'open',
          item_id TEXT,
          reason_code TEXT NOT NULL,
          details_json TEXT,
          created_at TEXT NOT NULL,
          resolved_at TEXT,
          resolved_by TEXT,
          resolved_reason TEXT,
          FOREIGN KEY(user_id) REFERENCES users(user_id),
          FOREIGN KEY(session_id) REFERENCES daily_sessions(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_safety_events_user_status_date
          ON safety_events(user_id, status, date);
        CREATE INDEX IF NOT EXISTS idx_safety_events_session
          ON safety_events(session_id);

        CREATE TABLE IF NOT EXISTS experiments (
          experiment_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          name TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'active',
          start_date TEXT NOT NULL,
          end_date TEXT,
          target_metrics_json TEXT NOT NULL DEFAULT '{}',
          baseline_window_days INTEGER NOT NULL DEFAULT 14,
          eval_window_days INTEGER NOT NULL DEFAULT 14,
          stopping_rules_json TEXT NOT NULL DEFAULT '{}',
          intervention_json TEXT NOT NULL DEFAULT '{}',
          results_json TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_experiments_user_status_start
          ON experiments(user_id, status, start_date);
        """
    )


def _migration_v8(conn: sqlite3.Connection) -> None:
    # First-class trajectory monitoring persistence.
    cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(circle_snapshots);").fetchall()}
    if "coherence_score" not in cols:
        conn.execute("ALTER TABLE circle_snapshots ADD COLUMN coherence_score REAL;")
    if "coherence_tier_json" not in cols:
        conn.execute("ALTER TABLE circle_snapshots ADD COLUMN coherence_tier_json TEXT;")
    obs_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(observation_events);").fetchall()}
    if "coupling_json" not in obs_cols:
        conn.execute("ALTER TABLE observation_events ADD COLUMN coupling_json TEXT;")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ews_features (
          feature_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          date TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          window_size_days INTEGER NOT NULL,
          var_r REAL NOT NULL,
          ac1_r REAL NOT NULL,
          trend_speed REAL NOT NULL,
          trend_accel REAL NOT NULL,
          recovery_rate REAL NOT NULL,
          ews_score REAL NOT NULL,
          notes_json TEXT,
          created_at TEXT NOT NULL,
          UNIQUE(user_id, date, window_size_days),
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_ews_features_user_date
          ON ews_features(user_id, date);

        CREATE TABLE IF NOT EXISTS drift_events (
          event_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          date TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          coherence_score REAL NOT NULL,
          drift_domain TEXT NOT NULL,
          triggered_instrument TEXT,
          ews_score REAL,
          reason_codes_json TEXT NOT NULL,
          details_json TEXT,
          created_at TEXT NOT NULL,
          UNIQUE(user_id, date),
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_drift_events_user_date
          ON drift_events(user_id, date);
        """
    )


def _migration_v9(conn: sqlite3.Connection) -> None:
    # Wire-up: behavioural metadata, session uncertainty profiles, anamnesis episodes.
    # ALTER existing tables for metadata-adjusted scoring and site config binding.

    # --- ALTER user_profiles: add site_config_id ---
    up_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(user_profiles);").fetchall()}
    if "site_config_id" not in up_cols:
        conn.execute("ALTER TABLE user_profiles ADD COLUMN site_config_id TEXT DEFAULT 'consumer';")

    # --- ALTER scale_scores: add metadata-adjusted columns ---
    ss_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(scale_scores);").fetchall()}
    if "se_theta" not in ss_cols:
        conn.execute("ALTER TABLE scale_scores ADD COLUMN se_theta REAL;")
    if "se_theta_adjusted" not in ss_cols:
        conn.execute("ALTER TABLE scale_scores ADD COLUMN se_theta_adjusted REAL;")
    if "uncertainty_multiplier" not in ss_cols:
        conn.execute("ALTER TABLE scale_scores ADD COLUMN uncertainty_multiplier REAL;")
    if "engagement_quality" not in ss_cols:
        conn.execute("ALTER TABLE scale_scores ADD COLUMN engagement_quality TEXT;")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS response_metadata (
          metadata_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          session_id TEXT NOT NULL,
          answer_event_id TEXT NOT NULL,
          item_id TEXT NOT NULL,
          response_latency_ms REAL,
          edit_count INTEGER NOT NULL DEFAULT 0,
          was_skipped INTEGER NOT NULL DEFAULT 0,
          was_declined INTEGER NOT NULL DEFAULT 0,
          channel TEXT NOT NULL DEFAULT 'tap',
          voice_hesitation_ms REAL,
          time_of_day_hour INTEGER,
          uncertainty_multiplier REAL NOT NULL DEFAULT 1.0,
          confidence_label TEXT NOT NULL DEFAULT 'medium',
          contributing_factors_json TEXT NOT NULL DEFAULT '[]',
          raw_components_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(user_id),
          FOREIGN KEY(session_id) REFERENCES daily_sessions(session_id),
          FOREIGN KEY(answer_event_id) REFERENCES answer_events(event_id)
        );

        CREATE INDEX IF NOT EXISTS idx_response_metadata_session
          ON response_metadata(session_id, item_id);
        CREATE INDEX IF NOT EXISTS idx_response_metadata_user_session
          ON response_metadata(user_id, session_id);

        CREATE TABLE IF NOT EXISTS session_uncertainty_profiles (
          profile_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          session_id TEXT NOT NULL,
          session_date TEXT NOT NULL,
          session_multiplier REAL NOT NULL DEFAULT 1.0,
          engagement_quality TEXT NOT NULL DEFAULT 'normal',
          median_latency_ms REAL NOT NULL DEFAULT 0.0,
          total_edits INTEGER NOT NULL DEFAULT 0,
          skip_count INTEGER NOT NULL DEFAULT 0,
          decline_count INTEGER NOT NULL DEFAULT 0,
          item_count INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL,
          UNIQUE(user_id, session_id),
          FOREIGN KEY(user_id) REFERENCES users(user_id),
          FOREIGN KEY(session_id) REFERENCES daily_sessions(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_session_uncertainty_user_date
          ON session_uncertainty_profiles(user_id, session_date);

        CREATE TABLE IF NOT EXISTS anamnesis_episodes (
          episode_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          trigger_date TEXT NOT NULL,
          drift_domain TEXT NOT NULL,
          trigger_type TEXT NOT NULL,
          trigger_value REAL NOT NULL,
          triggered_scale_ids_json TEXT NOT NULL DEFAULT '[]',
          status TEXT NOT NULL DEFAULT 'active',
          follow_up_item_ids_json TEXT NOT NULL DEFAULT '[]',
          priority INTEGER NOT NULL DEFAULT 100,
          max_days INTEGER NOT NULL DEFAULT 7,
          observations_collected INTEGER NOT NULL DEFAULT 0,
          resolution_date TEXT,
          resolution_reason TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_anamnesis_episodes_user_status
          ON anamnesis_episodes(user_id, status);
        CREATE INDEX IF NOT EXISTS idx_anamnesis_episodes_user_date
          ON anamnesis_episodes(user_id, trigger_date);
        """
    )


def _migration_v10(conn: sqlite3.Connection) -> None:
    """MiniFold sub-circle snapshots + measurement package tracking."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS minifold_snapshots (
          snapshot_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          date TEXT NOT NULL,
          circle_snapshot_id TEXT NOT NULL,
          mode TEXT NOT NULL,
          version TEXT NOT NULL,
          z_json TEXT NOT NULL,
          z_star_json TEXT NOT NULL,
          r REAL NOT NULL DEFAULT 0.0,
          theta REAL NOT NULL DEFAULT 0.0,
          velocity REAL NOT NULL DEFAULT 0.0,
          acceleration REAL NOT NULL DEFAULT 0.0,
          coherence REAL NOT NULL DEFAULT 1.0,
          uncertainty REAL NOT NULL DEFAULT 1.0,
          coverage_ratio REAL NOT NULL DEFAULT 0.0,
          dimensions_json TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL,
          UNIQUE(user_id, date, mode),
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_minifold_user_date_mode
          ON minifold_snapshots(user_id, date, mode);

        CREATE TABLE IF NOT EXISTS user_measurement_packages (
          package_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          measurement_package_id TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending',
          session_count INTEGER NOT NULL DEFAULT 0,
          started_at TEXT,
          completed_at TEXT,
          created_at TEXT NOT NULL,
          UNIQUE(user_id, measurement_package_id),
          FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_user_mpackage_user_status
          ON user_measurement_packages(user_id, status);
        """
    )


@contextmanager
def connect(database_path: str) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(database_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
    finally:
        conn.close()


def ensure_user(conn: sqlite3.Connection, user_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?);",
        (user_id, _now_iso()),
    )


def get_active_registry_version(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute(
        "SELECT version FROM registry_versions WHERE status='active' ORDER BY created_at DESC LIMIT 1;"
    ).fetchone()
    return str(row["version"]) if row else None


def set_registry_version_active(conn: sqlite3.Connection, version: str) -> None:
    conn.execute("UPDATE registry_versions SET status='inactive' WHERE status='active';")
    conn.execute(
        "INSERT OR REPLACE INTO registry_versions(version, status, created_at) VALUES (?, 'active', ?);",
        (version, _now_iso()),
    )


def upsert_registry_version(conn: sqlite3.Connection, version: str, status: str = "inactive") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO registry_versions(version, status, created_at) VALUES (?, ?, ?);",
        (version, status, _now_iso()),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
