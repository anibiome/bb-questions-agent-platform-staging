"""Add response_metadata, session_uncertainty_profiles, anamnesis_episodes tables
and SE adjustment columns on scale_scores.

Revision ID: 20260301_0007
Revises: 20260228_0006
Create Date: 2026-03-01 12:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260301_0007"
down_revision = "20260228_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- Response Metadata (Claim Family 5: Behavioural Uncertainty) ---
    op.create_table(
        "response_metadata",
        sa.Column("metadata_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("answer_event_id", sa.String(), nullable=False),
        sa.Column("item_id", sa.String(), nullable=False),
        sa.Column("response_latency_ms", sa.Float(), nullable=True),
        sa.Column("edit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("was_skipped", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("was_declined", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("channel", sa.String(), nullable=False, server_default="tap"),
        sa.Column("voice_hesitation_ms", sa.Float(), nullable=True),
        sa.Column("time_of_day_hour", sa.Integer(), nullable=True),
        sa.Column("uncertainty_multiplier", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("confidence_label", sa.String(), nullable=False, server_default="medium"),
        sa.Column("contributing_factors_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("raw_components_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.ForeignKeyConstraint(["session_id"], ["daily_sessions.session_id"]),
        sa.ForeignKeyConstraint(["answer_event_id"], ["answer_events.event_id"]),
        sa.PrimaryKeyConstraint("metadata_id"),
    )
    op.create_index(
        "idx_response_metadata_session",
        "response_metadata",
        ["session_id", "item_id"],
        unique=False,
    )
    op.create_index(
        "idx_response_metadata_user_session",
        "response_metadata",
        ["user_id", "session_id"],
        unique=False,
    )

    # --- Session Uncertainty Profiles ---
    op.create_table(
        "session_uncertainty_profiles",
        sa.Column("profile_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("session_date", sa.String(), nullable=False),
        sa.Column("session_multiplier", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("engagement_quality", sa.String(), nullable=False, server_default="normal"),
        sa.Column("median_latency_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("total_edits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skip_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("decline_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("item_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.ForeignKeyConstraint(["session_id"], ["daily_sessions.session_id"]),
        sa.PrimaryKeyConstraint("profile_id"),
        sa.UniqueConstraint("user_id", "session_id", name="uq_session_uncertainty_profile"),
    )
    op.create_index(
        "idx_session_uncertainty_user_date",
        "session_uncertainty_profiles",
        ["user_id", "session_date"],
        unique=False,
    )

    # --- Anamnesis Episodes (Claim Family 4: Drift-Triggered Branching) ---
    op.create_table(
        "anamnesis_episodes",
        sa.Column("episode_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("trigger_date", sa.String(), nullable=False),
        sa.Column("drift_domain", sa.String(), nullable=False),
        sa.Column("trigger_type", sa.String(), nullable=False),
        sa.Column("trigger_value", sa.Float(), nullable=False),
        sa.Column("triggered_scale_ids_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("follow_up_item_ids_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("max_days", sa.Integer(), nullable=False, server_default="7"),
        sa.Column("observations_collected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("resolution_date", sa.String(), nullable=True),
        sa.Column("resolution_reason", sa.String(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("episode_id"),
    )
    op.create_index(
        "idx_anamnesis_episodes_user_status",
        "anamnesis_episodes",
        ["user_id", "status"],
        unique=False,
    )
    op.create_index(
        "idx_anamnesis_episodes_user_date",
        "anamnesis_episodes",
        ["user_id", "trigger_date"],
        unique=False,
    )

    # --- ALTER scale_scores: add SE adjustment columns ---
    op.add_column("scale_scores", sa.Column("se_theta", sa.Float(), nullable=True))
    op.add_column("scale_scores", sa.Column("se_theta_adjusted", sa.Float(), nullable=True))
    op.add_column(
        "scale_scores",
        sa.Column("uncertainty_multiplier", sa.Float(), nullable=True),
    )
    op.add_column("scale_scores", sa.Column("engagement_quality", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("scale_scores", "engagement_quality")
    op.drop_column("scale_scores", "uncertainty_multiplier")
    op.drop_column("scale_scores", "se_theta_adjusted")
    op.drop_column("scale_scores", "se_theta")
    op.drop_index("idx_anamnesis_episodes_user_date", table_name="anamnesis_episodes")
    op.drop_index("idx_anamnesis_episodes_user_status", table_name="anamnesis_episodes")
    op.drop_table("anamnesis_episodes")
    op.drop_index("idx_session_uncertainty_user_date", table_name="session_uncertainty_profiles")
    op.drop_table("session_uncertainty_profiles")
    op.drop_index("idx_response_metadata_user_session", table_name="response_metadata")
    op.drop_index("idx_response_metadata_session", table_name="response_metadata")
    op.drop_table("response_metadata")
