"""Refactor session access to separate concerns

This migration separates the concerns of:
1. Direct user access (SessionAccess - who can view/edit what)
2. Group membership access (SessionGroupLink - which groups have what sessions)
3. Access origin tracking (SessionAccessOrigin - how users got access)

Revision ID: 006
Revises: 005
Create Date: 2026-04-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -----------------------
    # 1. Create SessionAccessOrigin table
    # -----------------------
    op.execute("CREATE TYPE session_access_origin_type AS ENUM ('direct', 'group')")

    op.create_table(
        "session_access_origins",
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
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "source_type",
            sa.Enum(
                "direct",
                "group",
                name="session_access_origin_type",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "group_id",
            UUID(as_uuid=True),
            sa.ForeignKey("groups.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.UniqueConstraint(
            "session_id",
            "user_id",
            "source_type",
            "group_id",
            name="uq_session_access_origins_unique",
        ),
    )

    # Add indexes for SessionAccessOrigin
    op.create_index(
        op.f("ix_session_access_origins_session_user"),
        "session_access_origins",
        ["session_id", "user_id"],
    )
    op.create_index(
        op.f("ix_session_access_origins_source_type"),
        "session_access_origins",
        ["source_type"],
    )

    # -----------------------
    # 2. Create SessionGroupLink table
    # -----------------------
    op.create_table(
        "session_group_links",
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
            "group_id",
            UUID(as_uuid=True),
            sa.ForeignKey("groups.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.UniqueConstraint(
            "session_id",
            "group_id",
            name="uq_session_group_links_unique",
        ),
    )

    # -----------------------
    # 3. Backfill SessionAccessOrigin and SessionGroupLink from old SessionAccess
    # -----------------------
    # Create group-based access origins
    op.execute(
        """
        INSERT INTO session_access_origins (session_id, user_id, source_type, group_id, created_at)
        SELECT session_id, user_id, 'group', group_id, created_at
        FROM session_access
        WHERE group_id IS NOT NULL
        """
    )

    # Create direct user access origins from current user_id entries
    op.execute(
        """
        INSERT INTO session_access_origins (session_id, user_id, source_type, created_at)
        SELECT session_id, user_id, 'direct', created_at
        FROM session_access
        WHERE user_id IS NOT NULL AND group_id IS NULL
        """
    )

    # Extract unique session-group pairs to create SessionGroupLink entries
    op.execute(
        """
        INSERT INTO session_group_links (session_id, group_id, created_at)
        SELECT DISTINCT sa.session_id, sa.group_id, MIN(sa.created_at)
        FROM session_access sa
        WHERE sa.group_id IS NOT NULL
        GROUP BY sa.session_id, sa.group_id
        """
    )

    # -----------------------
    # 4. Modify SessionAccess table
    # -----------------------
    # Drop the check constraint
    op.drop_constraint(
        "ck_session_access_one_subject",
        "session_access",
        type_="check",
    )

    # Drop the group_id column
    op.drop_index(
        op.f("ix_session_access_group_id"),
        table_name="session_access",
        if_exists=True,
    )
    op.drop_constraint(
        "session_access_group_id_fkey",
        "session_access",
        type_="foreignkey",
    )
    op.drop_column("session_access", "group_id")

    # Make user_id NOT NULL
    op.alter_column(
        "session_access",
        "user_id",
        existing_type=UUID(as_uuid=True),
        nullable=False,
    )

    # Add unique constraint on (session_id, user_id)
    op.create_unique_constraint(
        "uq_session_access_session_user",
        "session_access",
        ["session_id", "user_id"],
    )


def downgrade() -> None:
    # -----------------------
    # 1. Drop unique constraint from SessionAccess
    # -----------------------
    op.drop_constraint(
        "uq_session_access_session_user",
        "session_access",
        type_="unique",
    )

    # -----------------------
    # 2. Add group_id back to SessionAccess
    # -----------------------
    op.add_column(
        "session_access",
        sa.Column(
            "group_id",
            UUID(as_uuid=True),
            nullable=True,
        ),
    )

    op.create_foreign_key(
        "session_access_group_id_fkey",
        "session_access",
        "groups",
        ["group_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_index(
        op.f("ix_session_access_group_id"),
        "session_access",
        ["group_id"],
    )

    # Make user_id nullable again
    op.alter_column(
        "session_access",
        "user_id",
        existing_type=UUID(as_uuid=True),
        nullable=True,
    )

    # -----------------------
    # 3. Restore data to SessionAccess
    # -----------------------
    # Backfill group_id from SessionAccessOrigin
    op.execute(
        """
        UPDATE session_access sa
        SET group_id = (
            SELECT sao.group_id
            FROM session_access_origins sao
            WHERE sao.session_id = sa.session_id
            AND sao.user_id = sa.user_id
            AND sao.source_type = 'group'
            LIMIT 1
        )
        """
    )

    # Re-add the check constraint
    op.create_check_constraint(
        "ck_session_access_one_subject",
        "session_access",
        "(user_id IS NOT NULL AND group_id IS NULL) OR (user_id IS NULL AND group_id IS NOT NULL)",
    )

    # -----------------------
    # 4. Drop SessionGroupLink table
    # -----------------------
    op.drop_table("session_group_links")

    # -----------------------
    # 5. Drop SessionAccessOrigin table
    # -----------------------
    op.drop_index(
        op.f("ix_session_access_origins_source_type"),
        table_name="session_access_origins",
        if_exists=True,
    )
    op.drop_index(
        op.f("ix_session_access_origins_session_user"),
        table_name="session_access_origins",
        if_exists=True,
    )
    op.drop_table("session_access_origins")

    # -----------------------
    # 6. Drop the enum type
    # -----------------------
    op.execute("DROP TYPE IF EXISTS session_access_origin_type")
