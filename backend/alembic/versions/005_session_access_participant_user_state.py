"""Add session access, participant, and user state layers

Revision ID: 005
Revises: 004
Create Date: 2026-04-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -----------------------
    # 1. Create enums
    # -----------------------
    op.execute("CREATE TYPE session_access_role AS ENUM ('viewer', 'editor', 'admin')")
    op.execute("CREATE TYPE participant_role AS ENUM ('dancer', 'coach')")

    # -----------------------
    # 2. Add owner_id to routine_sessions
    # -----------------------
    op.add_column(
        "routine_sessions",
        sa.Column(
            "owner_id",
            UUID(as_uuid=True),
            nullable=True,
        ),
    )

    # Backfill owner_id with created_by for all existing sessions
    op.execute(
        """
        UPDATE routine_sessions
        SET owner_id = created_by
        """
    )

    # Make owner_id NOT NULL and add FK constraint
    op.alter_column("routine_sessions", "owner_id", nullable=False)
    op.create_foreign_key(
        "routine_sessions_owner_id_fkey",
        "routine_sessions",
        "users",
        ["owner_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        op.f("ix_routine_sessions_owner_id"),
        "routine_sessions",
        ["owner_id"],
    )

    # -----------------------
    # 3. Create session_access table
    # -----------------------
    op.create_table(
        "session_access",
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
            nullable=True,
            index=True,
        ),
        sa.Column(
            "group_id",
            UUID(as_uuid=True),
            sa.ForeignKey("groups.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        sa.Column(
            "role",
            sa.Enum(
                "viewer",
                "editor",
                "admin",
                name="session_access_role",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(user_id IS NOT NULL AND group_id IS NULL) OR (user_id IS NULL AND group_id IS NOT NULL)",
            name="ck_session_access_one_subject",
        ),
    )

    # -----------------------
    # 4. Create session_participants table
    # -----------------------
    op.create_table(
        "session_participants",
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
            "role",
            sa.Enum(
                "dancer",
                "coach",
                name="participant_role",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "dancer_slot_id",
            UUID(as_uuid=True),
            sa.ForeignKey("routine_dancer_slots.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "session_id",
            "user_id",
            name="uq_session_participants_session_user",
        ),
    )

    # -----------------------
    # 5. Create session_user_states table
    # -----------------------
    op.create_table(
        "session_user_states",
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
            "is_archived",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
        sa.Column(
            "is_deleted",
            sa.Boolean(),
            server_default="false",
            nullable=False,
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
            name="uq_session_user_states_session_user",
        ),
    )

    # -----------------------
    # 6. Seed initial SessionAccess for owners
    # -----------------------
    op.execute(
        """
        INSERT INTO session_access (session_id, user_id, role, created_at)
        SELECT id, owner_id, 'admin', now()
        FROM routine_sessions
        """
    )

    # -----------------------
    # 7. Remove group_id from routine_sessions
    # -----------------------
    op.drop_index(
        op.f("ix_routine_sessions_group_id"),
        table_name="routine_sessions",
        if_exists=True,
    )
    op.drop_constraint(
        "routine_sessions_group_id_fkey",
        "routine_sessions",
        type_="foreignkey",
    )
    op.drop_column("routine_sessions", "group_id")


def downgrade() -> None:
    # -----------------------
    # 1. Re-add group_id to routine_sessions
    # -----------------------
    op.add_column(
        "routine_sessions",
        sa.Column(
            "group_id",
            UUID(as_uuid=True),
            nullable=True,
        ),
    )

    # Backfill group_id from session_access records (first group access per session)
    op.execute(
        """
        UPDATE routine_sessions rs
        SET group_id = (
            SELECT group_id
            FROM session_access sa
            WHERE sa.session_id = rs.id
            AND sa.group_id IS NOT NULL
            LIMIT 1
        )
        """
    )

    op.create_foreign_key(
        "routine_sessions_group_id_fkey",
        "routine_sessions",
        "groups",
        ["group_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        op.f("ix_routine_sessions_group_id"),
        "routine_sessions",
        ["group_id"],
    )

    # -----------------------
    # 2. Drop session tables in dependency order
    # -----------------------
    op.drop_table("session_user_states")
    op.drop_table("session_participants")
    op.drop_table("session_access")

    # -----------------------
    # 3. Remove owner_id from routine_sessions
    # -----------------------
    op.drop_index(
        op.f("ix_routine_sessions_owner_id"),
        table_name="routine_sessions",
        if_exists=True,
    )
    op.drop_constraint(
        "routine_sessions_owner_id_fkey",
        "routine_sessions",
        type_="foreignkey",
    )
    op.drop_column("routine_sessions", "owner_id")

    # -----------------------
    # 4. Drop enums
    # -----------------------
    op.execute("DROP TYPE IF EXISTS participant_role")
    op.execute("DROP TYPE IF EXISTS session_access_role")
