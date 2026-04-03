"""Add isAdmin column to group_memberships.

Revision ID: 003
Revises: 002
Create Date: 2026-04-02
"""

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: str = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "group_memberships",
        sa.Column("isAdmin", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    op.drop_column("group_memberships", "isAdmin")
