"""Add first-class EWS and drift-event persistence tables.

Revision ID: 20260215_0004
Revises: 20260214_0003
Create Date: 2026-02-15 03:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260215_0004"
down_revision = "20260214_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("circle_snapshots", sa.Column("coherence_score", sa.Float(), nullable=True))
    op.add_column("circle_snapshots", sa.Column("coherence_tier_json", sa.Text(), nullable=True))
    op.add_column("observation_events", sa.Column("coupling_json", sa.Text(), nullable=True))

    op.create_table(
        "ews_features",
        sa.Column("feature_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("timestamp", sa.String(), nullable=False),
        sa.Column("window_size_days", sa.Integer(), nullable=False),
        sa.Column("var_r", sa.Float(), nullable=False),
        sa.Column("ac1_r", sa.Float(), nullable=False),
        sa.Column("trend_speed", sa.Float(), nullable=False),
        sa.Column("trend_accel", sa.Float(), nullable=False),
        sa.Column("recovery_rate", sa.Float(), nullable=False),
        sa.Column("ews_score", sa.Float(), nullable=False),
        sa.Column("notes_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("feature_id"),
        sa.UniqueConstraint("user_id", "date", "window_size_days", name="uq_ews_features_user_date_window"),
    )
    op.create_index("idx_ews_features_user_date", "ews_features", ["user_id", "date"], unique=False)

    op.create_table(
        "drift_events",
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("timestamp", sa.String(), nullable=False),
        sa.Column("coherence_score", sa.Float(), nullable=False),
        sa.Column("drift_domain", sa.String(), nullable=False),
        sa.Column("triggered_instrument", sa.String(), nullable=True),
        sa.Column("ews_score", sa.Float(), nullable=True),
        sa.Column("reason_codes_json", sa.Text(), nullable=False),
        sa.Column("details_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint("user_id", "date", name="uq_drift_events_user_date"),
    )
    op.create_index("idx_drift_events_user_date", "drift_events", ["user_id", "date"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_drift_events_user_date", table_name="drift_events")
    op.drop_table("drift_events")
    op.drop_index("idx_ews_features_user_date", table_name="ews_features")
    op.drop_table("ews_features")
    op.drop_column("observation_events", "coupling_json")
    op.drop_column("circle_snapshots", "coherence_tier_json")
    op.drop_column("circle_snapshots", "coherence_score")
