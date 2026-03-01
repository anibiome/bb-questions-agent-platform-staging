"""Add site_config_id column to user_profiles table.

Revision ID: 20260228_0006
Revises: 20260227_0005
Create Date: 2026-02-28 12:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260228_0006"
down_revision = "20260227_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_profiles",
        sa.Column("site_config_id", sa.String(), nullable=False, server_default="consumer"),
    )


def downgrade() -> None:
    op.drop_column("user_profiles", "site_config_id")
