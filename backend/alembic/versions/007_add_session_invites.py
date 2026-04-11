"""Add session_invites table for direct session email invites

Revision ID: 007
Revises: 006
Create Date: 2026-04-04
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create the session_invite_status enum type
    op.execute(
        "CREATE TYPE session_invite_status AS ENUM ('pending', 'accepted', 'revoked', 'expired')"
    )

    # Create the session_invites table
    op.create_table(
        "session_invites",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("routine_sessions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "created_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("email", sa.String(255), nullable=False, index=True),
        sa.Column("role", sa.String(50), nullable=False, default="viewer"),
        sa.Column("token", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "accepted",
                "revoked",
                "expired",
                name="session_invite_status",
                native_enum=False,
            ),
            nullable=False,
            default="pending",
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "accepted_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "accepted_by_user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )

    # Create indexes for common queries
    op.create_index(
        "ix_session_invites_session_email",
        "session_invites",
        ["session_id", "email"],
    )
    op.create_index(
        "ix_session_invites_status",
        "session_invites",
        ["status"],
    )


def downgrade() -> None:
    op.drop_table("session_invites")
    op.execute("DROP TYPE session_invite_status")
