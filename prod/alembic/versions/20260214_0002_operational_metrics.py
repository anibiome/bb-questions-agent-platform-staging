"""Operational metrics table for SLO monitoring.

Revision ID: 20260214_0002
Revises: 20260214_0001
Create Date: 2026-02-14 06:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260214_0002"
down_revision = "20260214_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "operational_metrics",
        sa.Column("metric_id", sa.String(), nullable=False),
        sa.Column("metric_type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("metric_date", sa.Date(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("details_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["daily_sessions.session_id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("metric_id"),
    )
    op.create_index(
        "idx_operational_metrics_type_date_status",
        "operational_metrics",
        ["metric_type", "metric_date", "status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_operational_metrics_type_date_status", table_name="operational_metrics")
    op.drop_table("operational_metrics")

