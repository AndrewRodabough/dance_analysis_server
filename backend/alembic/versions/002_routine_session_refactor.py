"""Routine session refactor

Revision ID: 002
Revises: 001
Create Date: 2026-03-25
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -----------------------
    # 1. Create routine_sessions
    # -----------------------
    op.create_table(
        "routine_sessions",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "routine_id",
            UUID(as_uuid=True),
            sa.ForeignKey("routines.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "group_id",
            UUID(as_uuid=True),
            sa.ForeignKey("groups.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("label", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        op.f("ix_routine_sessions_routine_id"),
        "routine_sessions",
        ["routine_id"],
    )
    op.create_index(
        op.f("ix_routine_sessions_group_id"),
        "routine_sessions",
        ["group_id"],
    )
    op.create_index(
        op.f("ix_routine_sessions_created_by"),
        "routine_sessions",
        ["created_by"],
    )

    # -----------------------
    # 2. Create routine_dancer_slots
    # -----------------------
    op.create_table(
        "routine_dancer_slots",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "routine_id",
            UUID(as_uuid=True),
            sa.ForeignKey("routines.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("label", sa.String(50), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "routine_id", "label", name="uq_routine_dancer_slots_routine_label"
        ),
    )
    op.create_index(
        op.f("ix_routine_dancer_slots_routine_id"),
        "routine_dancer_slots",
        ["routine_id"],
    )

    # -----------------------
    # 3. Create slot_assignments
    # -----------------------
    op.create_table(
        "slot_assignments",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "routine_session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("routine_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "dancer_slot_id",
            UUID(as_uuid=True),
            sa.ForeignKey("routine_dancer_slots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "routine_session_id",
            "dancer_slot_id",
            name="uq_slot_assignments_session_slot",
        ),
    )
    op.create_index(
        op.f("ix_slot_assignments_routine_session_id"),
        "slot_assignments",
        ["routine_session_id"],
    )
    op.create_index(
        op.f("ix_slot_assignments_dancer_slot_id"),
        "slot_assignments",
        ["dancer_slot_id"],
    )
    op.create_index(
        op.f("ix_slot_assignments_user_id"),
        "slot_assignments",
        ["user_id"],
    )

    # -----------------------
    # 4. Migrate existing data
    # -----------------------

    # Create a RoutineSession for every existing routine, carrying over group_id
    op.execute(
        """
        INSERT INTO routine_sessions (id, routine_id, group_id, created_by, created_at)
        SELECT gen_random_uuid(), id, group_id, created_by, created_at
        FROM routines
        """
    )

    # Add routine_session_id column to videos (nullable for now)
    op.add_column(
        "videos",
        sa.Column(
            "routine_session_id",
            UUID(as_uuid=True),
            nullable=True,
        ),
    )

    # Backfill videos: match via the routine_id -> routine_sessions mapping
    op.execute(
        """
        UPDATE videos
        SET routine_session_id = rs.id
        FROM routine_sessions rs
        WHERE videos.routine_id = rs.routine_id
        """
    )

    # Drop old routine_id FK and column from videos
    op.drop_index(op.f("ix_videos_routine_id"), table_name="videos", if_exists=True)
    op.drop_constraint("videos_routine_id_fkey", "videos", type_="foreignkey")
    op.drop_column("videos", "routine_id")

    # Add FK constraint and index for videos.routine_session_id
    op.create_foreign_key(
        "videos_routine_session_id_fkey",
        "videos",
        "routine_sessions",
        ["routine_session_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        op.f("ix_videos_routine_session_id"),
        "videos",
        ["routine_session_id"],
    )

    # Add routine_session_id column to notes (nullable for now)
    op.add_column(
        "notes",
        sa.Column(
            "routine_session_id",
            UUID(as_uuid=True),
            nullable=True,
        ),
    )

    # Backfill notes
    op.execute(
        """
        UPDATE notes
        SET routine_session_id = rs.id
        FROM routine_sessions rs
        WHERE notes.routine_id = rs.routine_id
        """
    )

    # Make routine_session_id NOT NULL on notes now that data is migrated
    op.alter_column("notes", "routine_session_id", nullable=False)

    # Drop old routine_id FK and column from notes
    op.drop_index(op.f("ix_notes_routine_id"), table_name="notes", if_exists=True)
    op.drop_constraint("notes_routine_id_fkey", "notes", type_="foreignkey")
    op.drop_column("notes", "routine_id")

    # Add FK constraint and index for notes.routine_session_id
    op.create_foreign_key(
        "notes_routine_session_id_fkey",
        "notes",
        "routine_sessions",
        ["routine_session_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        op.f("ix_notes_routine_session_id"),
        "notes",
        ["routine_session_id"],
    )

    # -----------------------
    # 5. Drop group_id from routines
    # -----------------------
    op.drop_index(op.f("ix_routines_group_id"), table_name="routines", if_exists=True)
    op.drop_constraint("routines_group_id_fkey", "routines", type_="foreignkey")
    op.drop_column("routines", "group_id")


def downgrade() -> None:
    # Re-add group_id to routines
    op.add_column(
        "routines",
        sa.Column("group_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "routines_group_id_fkey",
        "routines",
        "groups",
        ["group_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Restore group_id on routines from routine_sessions
    op.execute(
        """
        UPDATE routines
        SET group_id = rs.group_id
        FROM routine_sessions rs
        WHERE routines.id = rs.routine_id
        """
    )

    # Re-add routine_id to notes
    op.add_column(
        "notes",
        sa.Column("routine_id", UUID(as_uuid=True), nullable=True),
    )
    op.execute(
        """
        UPDATE notes
        SET routine_id = rs.routine_id
        FROM routine_sessions rs
        WHERE notes.routine_session_id = rs.id
        """
    )
    op.alter_column("notes", "routine_id", nullable=False)
    op.create_foreign_key(
        "notes_routine_id_fkey",
        "notes",
        "routines",
        ["routine_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_constraint("notes_routine_session_id_fkey", "notes", type_="foreignkey")
    op.drop_index(op.f("ix_notes_routine_session_id"), table_name="notes")
    op.drop_column("notes", "routine_session_id")

    # Re-add routine_id to videos
    op.add_column(
        "videos",
        sa.Column("routine_id", UUID(as_uuid=True), nullable=True),
    )
    op.execute(
        """
        UPDATE videos
        SET routine_id = rs.routine_id
        FROM routine_sessions rs
        WHERE videos.routine_session_id = rs.id
        """
    )
    op.create_foreign_key(
        "videos_routine_id_fkey",
        "videos",
        "routines",
        ["routine_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_constraint("videos_routine_session_id_fkey", "videos", type_="foreignkey")
    op.drop_index(op.f("ix_videos_routine_session_id"), table_name="videos")
    op.drop_column("videos", "routine_session_id")

    # Drop new tables in dependency order
    op.drop_table("slot_assignments")
    op.drop_table("routine_dancer_slots")
    op.drop_table("routine_sessions")
