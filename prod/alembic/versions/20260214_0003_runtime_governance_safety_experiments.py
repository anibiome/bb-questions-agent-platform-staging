"""Runtime governance, safety, and experiments.

Revision ID: 20260214_0003
Revises: 20260214_0002
Create Date: 2026-02-14 07:10:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260214_0003"
down_revision = "20260214_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_profiles",
        sa.Column("active_domains_json", sa.Text(), nullable=False, server_default='["cardiometabolic"]'),
    )
    op.add_column(
        "user_profiles",
        sa.Column("queued_domains_json", sa.Text(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "user_profiles",
        sa.Column("promoted_domains_json", sa.Text(), nullable=False, server_default='["cardiometabolic"]'),
    )

    op.create_table(
        "safety_events",
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("trigger_source", sa.String(), nullable=False),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="open"),
        sa.Column("item_id", sa.String(), nullable=True),
        sa.Column("reason_code", sa.String(), nullable=False),
        sa.Column("details_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("resolved_at", sa.String(), nullable=True),
        sa.Column("resolved_by", sa.String(), nullable=True),
        sa.Column("resolved_reason", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["daily_sessions.session_id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index(
        "idx_safety_events_user_status_date",
        "safety_events",
        ["user_id", "status", "date"],
        unique=False,
    )
    op.create_index(
        "idx_safety_events_session",
        "safety_events",
        ["session_id"],
        unique=False,
    )

    op.create_table(
        "experiments",
        sa.Column("experiment_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("target_metrics_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("baseline_window_days", sa.Integer(), nullable=False, server_default="14"),
        sa.Column("eval_window_days", sa.Integer(), nullable=False, server_default="14"),
        sa.Column("stopping_rules_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("intervention_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("results_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("experiment_id"),
    )
    op.create_index(
        "idx_experiments_user_status_start",
        "experiments",
        ["user_id", "status", "start_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_experiments_user_status_start", table_name="experiments")
    op.drop_table("experiments")

    op.drop_index("idx_safety_events_session", table_name="safety_events")
    op.drop_index("idx_safety_events_user_status_date", table_name="safety_events")
    op.drop_table("safety_events")

    op.drop_column("user_profiles", "promoted_domains_json")
    op.drop_column("user_profiles", "queued_domains_json")
    op.drop_column("user_profiles", "active_domains_json")
