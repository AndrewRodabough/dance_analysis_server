"""Add progress column and failed_hidden status to jobs

Revision ID: 002
Revises: 001
Create Date: 2026-03-02
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add progress column (0-100) with default 0
    op.add_column('jobs', sa.Column('progress', sa.Integer(), nullable=True, server_default='0'))

    # Add 'failed_hidden' to the jobstatus enum
    # PostgreSQL requires ALTER TYPE to add a new enum value
    op.execute("ALTER TYPE jobstatus ADD VALUE IF NOT EXISTS 'failed_hidden'")


def downgrade() -> None:
    # Move any failed_hidden jobs back to failed before removing the enum value
    op.execute("UPDATE jobs SET status = 'failed' WHERE status = 'failed_hidden'")

    op.drop_column('jobs', 'progress')

    # Note: PostgreSQL does not support removing values from an enum type.
    # The 'failed_hidden' value will remain in the enum but be unused.
    # To fully remove it, you would need to recreate the enum type.
