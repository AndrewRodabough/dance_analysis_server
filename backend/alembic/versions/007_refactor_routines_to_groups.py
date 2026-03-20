"""Refactor routines to group-scoped and drop routine_participants

Revision ID: 007
Revises: 006
Create Date: 2026-03-20
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add group_id to routines (nullable initially for migration, then set NOT NULL)
    op.add_column(
        "routines",
        sa.Column(
            "group_id",
            sa.Integer(),
            sa.ForeignKey("groups.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.create_index(op.f("ix_routines_group_id"), "routines", ["group_id"])

    # Rename owner_id to created_by
    op.alter_column("routines", "owner_id", new_column_name="created_by")

    # Drop routine_participants table
    op.drop_index(op.f("ix_routine_participants_user_id"), table_name="routine_participants")
    op.drop_index(op.f("ix_routine_participants_routine_id"), table_name="routine_participants")
    op.drop_table("routine_participants")

    # Drop routine_role enum
    bind = op.get_bind()
    sa.Enum(name="routine_role").drop(bind, checkfirst=True)


def downgrade() -> None:
    # Re-create routine_role enum
    routine_role_enum = sa.Enum("dancer", "coach", name="routine_role")
    bind = op.get_bind()
    routine_role_enum.create(bind, checkfirst=True)

    # Re-create routine_participants
    op.create_table(
        "routine_participants",
        sa.Column(
            "routine_id",
            sa.Integer(),
            sa.ForeignKey("routines.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("role", routine_role_enum, nullable=False),
    )
    op.create_index(
        op.f("ix_routine_participants_routine_id"),
        "routine_participants",
        ["routine_id"],
    )
    op.create_index(
        op.f("ix_routine_participants_user_id"),
        "routine_participants",
        ["user_id"],
    )

    # Rename created_by back to owner_id
    op.alter_column("routines", "created_by", new_column_name="owner_id")

    # Drop group_id column and index
    op.drop_index(op.f("ix_routines_group_id"), table_name="routines")
    op.drop_column("routines", "group_id")
