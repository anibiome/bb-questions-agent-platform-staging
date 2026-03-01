"""Add N-of-1 calibration, session coherence, and cardio risk tables.

Revision ID: 20260227_0005
Revises: 20260215_0004
Create Date: 2026-02-27 12:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260227_0005"
down_revision = "20260215_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- N-of-1 Personal Calibration ---
    op.create_table(
        "personal_calibrations",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("scale_id", sa.String(), nullable=False),
        sa.Column("n_observations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("phase", sa.String(), nullable=False, server_default="warming"),
        sa.Column("theta_personal", sa.Float(), nullable=False),
        sa.Column("se_personal", sa.Float(), nullable=False),
        sa.Column("theta_baseline", sa.Float(), nullable=True),
        sa.Column("se_baseline", sa.Float(), nullable=True),
        sa.Column("within_person_sd", sa.Float(), nullable=True),
        sa.Column("shrinkage", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("n_effective", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("calibration_json", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("user_id", "scale_id"),
    )
    op.create_index(
        "idx_personal_calibrations_user",
        "personal_calibrations",
        ["user_id"],
        unique=False,
    )

    # --- Session Coherence Assessment ---
    op.create_table(
        "session_coherence",
        sa.Column("coherence_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("coherence_score", sa.Float(), nullable=False),
        sa.Column("tier", sa.String(), nullable=False),
        sa.Column("n_flagged", sa.Integer(), nullable=False),
        sa.Column("n_items", sa.Integer(), nullable=False),
        sa.Column("signals_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.ForeignKeyConstraint(["session_id"], ["daily_sessions.session_id"]),
        sa.PrimaryKeyConstraint("coherence_id"),
        sa.UniqueConstraint("user_id", "session_id", name="uq_session_coherence"),
    )
    op.create_index(
        "idx_session_coherence_user_date",
        "session_coherence",
        ["user_id", "date"],
        unique=False,
    )
    op.create_index(
        "idx_session_coherence_session",
        "session_coherence",
        ["session_id"],
        unique=False,
    )

    # --- Cardiometabolic Risk Index ---
    op.create_table(
        "cardio_risk_snapshots",
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("composite_risk", sa.Float(), nullable=False),
        sa.Column("composite_risk_pct", sa.Float(), nullable=False),
        sa.Column("risk_tier", sa.String(), nullable=False),
        sa.Column("instruments_available", sa.Integer(), nullable=False),
        sa.Column("instruments_total", sa.Integer(), nullable=False),
        sa.Column("coverage", sa.Float(), nullable=False),
        sa.Column("confidence", sa.String(), nullable=False),
        sa.Column("components_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint("user_id", "date", name="uq_cardio_risk_user_date"),
    )
    op.create_index(
        "idx_cardio_risk_user_date",
        "cardio_risk_snapshots",
        ["user_id", "date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_cardio_risk_user_date", table_name="cardio_risk_snapshots")
    op.drop_table("cardio_risk_snapshots")
    op.drop_index("idx_session_coherence_session", table_name="session_coherence")
    op.drop_index("idx_session_coherence_user_date", table_name="session_coherence")
    op.drop_table("session_coherence")
    op.drop_index("idx_personal_calibrations_user", table_name="personal_calibrations")
    op.drop_table("personal_calibrations")
