"""Initial production schema.

Revision ID: 20260214_0001
Revises:
Create Date: 2026-02-14 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260214_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "registry_versions",
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("version"),
    )

    op.create_table(
        "users",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )

    op.create_table(
        "user_profiles",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("permanently_declined_items_json", sa.Text(), nullable=False),
        sa.Column("onboarding_complete", sa.Boolean(), nullable=False),
        sa.Column("state_schema_version", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("user_id"),
    )

    op.create_table(
        "daily_sessions",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("registry_version", sa.String(), nullable=False),
        sa.Column("timeframe", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("selection_mode", sa.String(), nullable=False),
        sa.Column("policy_decision_id", sa.String(), nullable=True),
        sa.Column("core_questions_json", sa.Text(), nullable=False),
        sa.Column("extra_batches_json", sa.Text(), nullable=False),
        sa.Column("extra_batches_used", sa.Integer(), nullable=False),
        sa.Column("selection_explain_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["registry_version"], ["registry_versions.version"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("session_id"),
        sa.UniqueConstraint("user_id", "date", name="uq_session_user_date"),
    )

    op.create_table(
        "answer_events",
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("client_event_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("item_id", sa.String(), nullable=False),
        sa.Column("answered_at", sa.String(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("raw_json", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["daily_sessions.session_id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint("user_id", "client_event_id", name="uq_answer_idempotency"),
    )

    op.create_table(
        "scale_evidence",
        sa.Column("evidence_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("answer_event_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("item_id", sa.String(), nullable=False),
        sa.Column("scale_id", sa.String(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("contribution_value", sa.Float(), nullable=False),
        sa.Column("timeframe", sa.String(), nullable=False),
        sa.Column("window_date", sa.Date(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["answer_event_id"], ["answer_events.event_id"]),
        sa.ForeignKeyConstraint(["session_id"], ["daily_sessions.session_id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("evidence_id"),
    )

    op.create_table(
        "scale_baselines",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("scale_id", sa.String(), nullable=False),
        sa.Column("mean", sa.Float(), nullable=False),
        sa.Column("var", sa.Float(), nullable=False),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("user_id", "scale_id"),
    )

    op.create_table(
        "scale_scores",
        sa.Column("score_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("scale_id", sa.String(), nullable=False),
        sa.Column("computed_at", sa.String(), nullable=False),
        sa.Column("window_start", sa.Date(), nullable=False),
        sa.Column("window_end", sa.Date(), nullable=False),
        sa.Column("raw_score", sa.Float(), nullable=False),
        sa.Column("normalized_score", sa.Float(), nullable=False),
        sa.Column("confidence_tier", sa.String(), nullable=False),
        sa.Column("items_answered_count", sa.Integer(), nullable=False),
        sa.Column("items_required", sa.Integer(), nullable=False),
        sa.Column("baseline_mean", sa.Float(), nullable=True),
        sa.Column("baseline_std", sa.Float(), nullable=True),
        sa.Column("personal_z", sa.Float(), nullable=True),
        sa.Column("delta_vs_prev", sa.Float(), nullable=True),
        sa.Column("risk_tier", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("score_id"),
    )

    op.create_table(
        "observation_events",
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("observed_at", sa.String(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("features_json", sa.Text(), nullable=False),
        sa.Column("confidence", sa.String(), nullable=False),
        sa.Column("provenance_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("event_id"),
    )

    op.create_table(
        "projections_questions",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("timestamp", sa.String(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("user_id", "timestamp"),
    )

    op.create_table(
        "follow_up_queue",
        sa.Column("queue_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("item_id", sa.String(), nullable=False),
        sa.Column("scale_id", sa.String(), nullable=True),
        sa.Column("reason_code", sa.String(), nullable=False),
        sa.Column("reason_detail_json", sa.Text(), nullable=True),
        sa.Column("priority", sa.Float(), nullable=False),
        sa.Column("earliest_date", sa.Date(), nullable=False),
        sa.Column("timeframe", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("resolved_at", sa.String(), nullable=True),
        sa.Column("resolved_reason", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("queue_id"),
    )

    op.create_table(
        "state_snapshots",
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("timestamp", sa.String(), nullable=False),
        sa.Column("state_schema_version", sa.String(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("x_hat_json", sa.Text(), nullable=False),
        sa.Column("x_uncertainty_json", sa.Text(), nullable=False),
        sa.Column("coverage_json", sa.Text(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint("user_id", "date", "state_schema_version", "model_version", name="uq_state_snapshot_key"),
    )

    op.create_table(
        "circle_snapshots",
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("timestamp", sa.String(), nullable=False),
        sa.Column("state_snapshot_id", sa.String(), nullable=True),
        sa.Column("projection_version", sa.String(), nullable=False),
        sa.Column("anchor_version", sa.String(), nullable=False),
        sa.Column("z_json", sa.Text(), nullable=False),
        sa.Column("z_star_json", sa.Text(), nullable=False),
        sa.Column("r", sa.Float(), nullable=False),
        sa.Column("theta", sa.Float(), nullable=False),
        sa.Column("velocity", sa.Float(), nullable=False),
        sa.Column("acceleration", sa.Float(), nullable=False),
        sa.Column("uncertainty_json", sa.Text(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["state_snapshot_id"], ["state_snapshots.snapshot_id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint("user_id", "date", "projection_version", name="uq_circle_snapshot_key"),
    )

    op.create_table(
        "policy_decisions",
        sa.Column("decision_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("identity_mask_id", sa.String(), nullable=True),
        sa.Column("selection_mode", sa.String(), nullable=False),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("policy_version", sa.String(), nullable=False),
        sa.Column("feature_version", sa.String(), nullable=False),
        sa.Column("context_hash", sa.String(), nullable=False),
        sa.Column("candidate_set_hash", sa.String(), nullable=False),
        sa.Column("context_json", sa.Text(), nullable=True),
        sa.Column("candidate_set_json", sa.Text(), nullable=True),
        sa.Column("selected_item_ids_json", sa.Text(), nullable=False),
        sa.Column("deterministic_baseline_selected_json", sa.Text(), nullable=False),
        sa.Column("propensities_json", sa.Text(), nullable=True),
        sa.Column("explanations_json", sa.Text(), nullable=False),
        sa.Column("counterfactuals_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("decision_id"),
    )

    op.create_table(
        "policy_outcomes",
        sa.Column("decision_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("completion_rate", sa.Float(), nullable=True),
        sa.Column("completed_core_count", sa.Integer(), nullable=True),
        sa.Column("response_time_ms_median", sa.Float(), nullable=True),
        sa.Column("z_before_hash", sa.String(), nullable=True),
        sa.Column("z_after_hash", sa.String(), nullable=True),
        sa.Column("z_before_json", sa.Text(), nullable=True),
        sa.Column("z_after_json", sa.Text(), nullable=True),
        sa.Column("z_delta_json", sa.Text(), nullable=True),
        sa.Column("z_before_norm", sa.Float(), nullable=True),
        sa.Column("z_after_norm", sa.Float(), nullable=True),
        sa.Column("z_delta_norm", sa.Float(), nullable=True),
        sa.Column("uncertainty_before_mean", sa.Float(), nullable=True),
        sa.Column("uncertainty_after_mean", sa.Float(), nullable=True),
        sa.Column("uncertainty_before_max", sa.Float(), nullable=True),
        sa.Column("uncertainty_after_max", sa.Float(), nullable=True),
        sa.Column("reward_components_json", sa.Text(), nullable=True),
        sa.Column("total_reward", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["decision_id"], ["policy_decisions.decision_id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("decision_id"),
    )

    op.create_index("idx_answer_events_user_time", "answer_events", ["user_id", "answered_at"], unique=False)
    op.create_index("idx_scale_evidence_user_scale_date", "scale_evidence", ["user_id", "scale_id", "window_date"], unique=False)
    op.create_index("idx_scale_evidence_answer_event", "scale_evidence", ["answer_event_id"], unique=False)
    op.create_index("idx_scale_scores_user_scale_time", "scale_scores", ["user_id", "scale_id", "computed_at"], unique=False)
    op.create_index("idx_observation_events_user_time", "observation_events", ["user_id", "observed_at"], unique=False)
    op.create_index("idx_policy_decisions_user_date", "policy_decisions", ["user_id", "date"], unique=False)
    op.create_index(
        "idx_follow_up_queue_due",
        "follow_up_queue",
        ["user_id", "status", "earliest_date", "priority", "created_at"],
        unique=False,
    )
    op.create_index("idx_follow_up_queue_item", "follow_up_queue", ["user_id", "item_id", "status"], unique=False)
    op.create_index("idx_follow_up_queue_scale", "follow_up_queue", ["user_id", "scale_id", "status"], unique=False)
    op.create_index("idx_state_snapshots_user_date", "state_snapshots", ["user_id", "date"], unique=False)
    op.create_index("idx_circle_snapshots_user_date", "circle_snapshots", ["user_id", "date"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_circle_snapshots_user_date", table_name="circle_snapshots")
    op.drop_index("idx_state_snapshots_user_date", table_name="state_snapshots")
    op.drop_index("idx_follow_up_queue_scale", table_name="follow_up_queue")
    op.drop_index("idx_follow_up_queue_item", table_name="follow_up_queue")
    op.drop_index("idx_follow_up_queue_due", table_name="follow_up_queue")
    op.drop_index("idx_policy_decisions_user_date", table_name="policy_decisions")
    op.drop_index("idx_observation_events_user_time", table_name="observation_events")
    op.drop_index("idx_scale_scores_user_scale_time", table_name="scale_scores")
    op.drop_index("idx_scale_evidence_answer_event", table_name="scale_evidence")
    op.drop_index("idx_scale_evidence_user_scale_date", table_name="scale_evidence")
    op.drop_index("idx_answer_events_user_time", table_name="answer_events")

    op.drop_table("policy_outcomes")
    op.drop_table("policy_decisions")
    op.drop_table("circle_snapshots")
    op.drop_table("state_snapshots")
    op.drop_table("follow_up_queue")
    op.drop_table("projections_questions")
    op.drop_table("observation_events")
    op.drop_table("scale_scores")
    op.drop_table("scale_baselines")
    op.drop_table("scale_evidence")
    op.drop_table("answer_events")
    op.drop_table("daily_sessions")
    op.drop_table("user_profiles")
    op.drop_table("users")
    op.drop_table("registry_versions")
