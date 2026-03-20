"""Add dances, routines, notes, and routine_participants tables

Revision ID: 004
Revises: 003
Create Date: 2026-03-18
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None

dance_style_enum = sa.Enum(
    "samba", "cha_cha", "rumba", "paso_doble", "jive",
    "waltz", "tango", "viennese_waltz", "foxtrot", "quickstep",
    "american_waltz", "american_tango", "american_foxtrot", "american_viennese_waltz",
    "american_cha_cha", "american_rumba", "swing", "bolero", "mambo",
    name="dance_style",
)
note_type_enum = sa.Enum("critique", "feedback", "complement", name="note_type")
routine_role_enum = sa.Enum("dancer", "coach", name="routine_role")


def upgrade() -> None:
    bind = op.get_bind()
    dance_style_enum.create(bind, checkfirst=True)
    note_type_enum.create(bind, checkfirst=True)
    routine_role_enum.create(bind, checkfirst=True)

    # --- dances ---
    op.create_table(
        "dances",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tempo", sa.Integer(), nullable=False),
        sa.Column("meter", sa.String(length=10), nullable=False),
        sa.Column("style", dance_style_enum, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f("ix_dances_id"), "dances", ["id"], unique=False)
    op.create_index(op.f("ix_dances_style"), "dances", ["style"])

    # --- routines ---
    op.create_table(
        "routines",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "dance_id",
            sa.Integer(),
            sa.ForeignKey("dances.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f("ix_routines_id"), "routines", ["id"], unique=False)
    op.create_index(op.f("ix_routines_owner_id"), "routines", ["owner_id"])
    op.create_index(op.f("ix_routines_dance_id"), "routines", ["dance_id"])

    # --- notes ---
    op.create_table(
        "notes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "author_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "routine_id",
            sa.Integer(),
            sa.ForeignKey("routines.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("note_type", note_type_enum, nullable=False),
        sa.Column("contents", sa.Text(), nullable=False),
        sa.Column(
            "video_id",
            sa.Integer(),
            sa.ForeignKey("videos.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("video_timestamp", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f("ix_notes_id"), "notes", ["id"], unique=False)
    op.create_index(op.f("ix_notes_author_id"), "notes", ["author_id"])
    op.create_index(op.f("ix_notes_routine_id"), "notes", ["routine_id"])
    op.create_index(op.f("ix_notes_video_id"), "notes", ["video_id"])

    # --- routine_participants ---
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


def downgrade() -> None:
    op.drop_index(op.f("ix_routine_participants_user_id"), table_name="routine_participants")
    op.drop_index(op.f("ix_routine_participants_routine_id"), table_name="routine_participants")
    op.drop_table("routine_participants")

    op.drop_index(op.f("ix_notes_video_id"), table_name="notes")
    op.drop_index(op.f("ix_notes_routine_id"), table_name="notes")
    op.drop_index(op.f("ix_notes_author_id"), table_name="notes")
    op.drop_index(op.f("ix_notes_id"), table_name="notes")
    op.drop_table("notes")

    op.drop_index(op.f("ix_routines_dance_id"), table_name="routines")
    op.drop_index(op.f("ix_routines_owner_id"), table_name="routines")
    op.drop_index(op.f("ix_routines_id"), table_name="routines")
    op.drop_table("routines")

    op.drop_index(op.f("ix_dances_style"), table_name="dances")
    op.drop_index(op.f("ix_dances_id"), table_name="dances")
    op.drop_table("dances")

    bind = op.get_bind()
    routine_role_enum.drop(bind, checkfirst=True)
    note_type_enum.drop(bind, checkfirst=True)
    dance_style_enum.drop(bind, checkfirst=True)
