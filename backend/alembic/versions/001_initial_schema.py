"""Initial schema (UUID-based)

Revision ID: 001
Revises: None
Create Date: 2026-03-20
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# -----------------------
# Enum definitions
# -----------------------

# Single-table enums (created automatically by op.create_table)
job_status_enum = sa.Enum(
    "pending",
    "processing",
    "completed",
    "failed",
    "failed_hidden",
    name="jobstatus",
)
video_status_enum = sa.Enum(
    "pending_upload",
    "uploaded",
    "deleted",
    name="video_status",
)
dance_style_enum = sa.Enum(
    "samba",
    "cha_cha",
    "rumba",
    "paso_doble",
    "jive",
    "waltz",
    "tango",
    "viennese_waltz",
    "foxtrot",
    "quickstep",
    "american_waltz",
    "american_tango",
    "american_foxtrot",
    "american_viennese_waltz",
    "american_cha_cha",
    "american_rumba",
    "swing",
    "bolero",
    "mambo",
    name="dance_style",
)
note_type_enum = sa.Enum(
    "critique",
    "feedback",
    "complement",
    name="note_type",
)
note_source_enum = sa.Enum(
    "user",
    "ai",
    "system",
    name="note_source",
)
membership_status_enum = sa.Enum(
    "active",
    "invited",
    "removed",
    name="membership_status",
)
group_invite_status_enum = sa.Enum(
    "pending",
    "accepted",
    "revoked",
    "expired",
    name="group_invite_status",
)

# Multi-table enum: used in group_memberships AND group_invites.
group_role_enum = sa.Enum(
    "owner",
    "coach",
    "member",
    name="group_role",
)


def upgrade() -> None:
    # Ensure gen_random_uuid() is available
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # -----------------------
    # Users
    # -----------------------
    op.create_table(
        "users",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("username", sa.String(50), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column(
            "is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False
        ),
        sa.Column(
            "is_superuser",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.UniqueConstraint("username", name="uq_users_username"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)

    # -----------------------
    # Dances
    # -----------------------
    op.create_table(
        "dances",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("tempo", sa.String(50), nullable=False),
        sa.Column("meter", sa.String(10), nullable=False),
        sa.Column("style", dance_style_enum, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f("ix_dances_style"), "dances", ["style"])

    # -----------------------
    # Groups
    # -----------------------
    op.create_table(
        "groups",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "is_archived", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f("ix_groups_created_by"), "groups", ["created_by"])

    # -----------------------
    # Group memberships
    # -----------------------
    op.create_table(
        "group_memberships",
        sa.Column(
            "group_id",
            UUID(as_uuid=True),
            sa.ForeignKey("groups.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("role", group_role_enum, nullable=False),
        sa.Column("status", membership_status_enum, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # -----------------------
    # Group invites
    # -----------------------
    op.create_table(
        "group_invites",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "group_id",
            UUID(as_uuid=True),
            sa.ForeignKey("groups.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("role", group_role_enum, nullable=True),
        sa.Column("token", sa.String(255), nullable=False),
        sa.Column("status", group_invite_status_enum, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "accepted_by_user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.UniqueConstraint("token", name="uq_group_invites_token"),
    )

    # -----------------------
    # Routines
    # -----------------------
    op.create_table(
        "routines",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column(
            "group_id",
            UUID(as_uuid=True),
            sa.ForeignKey("groups.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "created_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "dance_id",
            UUID(as_uuid=True),
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

    # -----------------------
    # Figures
    # -----------------------
    op.create_table(
        "figures",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("tags", JSONB, nullable=True),
        sa.Column("total_beats", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # -----------------------
    # Steps
    # -----------------------
    op.create_table(
        "steps",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "figure_id",
            UUID(as_uuid=True),
            sa.ForeignKey("figures.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("start_state", JSONB, nullable=False),
        sa.Column("end_state", JSONB, nullable=False),
        sa.Column("keyframes", JSONB, nullable=True),
        sa.Column("duration", sa.Float(), nullable=False),
    )

    # -----------------------
    # Videos
    # -----------------------
    op.create_table(
        "videos",
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
            nullable=True,
        ),
        sa.Column(
            "uploaded_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("storage_key", sa.String(500), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("duration", sa.String(50), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=True),
        sa.Column(
            "status", video_status_enum, server_default="pending_upload", nullable=False
        ),
        sa.Column("original_filename", sa.String(500), nullable=True),
        sa.Column("content_type", sa.String(100), nullable=True),
        sa.UniqueConstraint("storage_key", name="uq_videos_storage_key"),
    )

    # -----------------------
    # Notes
    # -----------------------
    op.create_table(
        "notes",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "author_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "routine_id",
            UUID(as_uuid=True),
            sa.ForeignKey("routines.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("note_type", note_type_enum, nullable=False),
        sa.Column("contents", sa.Text(), nullable=False),
        sa.Column("source", note_source_enum, server_default="user", nullable=False),
        sa.Column(
            "video_id",
            UUID(as_uuid=True),
            sa.ForeignKey("videos.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "video_deleted",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("video_timestamp_ms", sa.Integer(), nullable=True),
        sa.Column("details", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # -----------------------
    # Jobs
    # -----------------------
    op.create_table(
        "jobs",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", job_status_enum, server_default="pending", nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column(
            "video_id",
            UUID(as_uuid=True),
            sa.ForeignKey("videos.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("result_path", sa.String(500), nullable=True),
        sa.Column("data_path", sa.String(500), nullable=True),
        sa.Column(
            "progress", sa.String(50), server_default=sa.text("'0'"), nullable=True
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("attempts", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("job_id", name="uq_jobs_job_id"),
    )


def downgrade() -> None:
    # Drop tables in reverse dependency order
    op.drop_table("jobs")
    op.drop_table("notes")
    op.drop_table("videos")
    op.drop_table("steps")
    op.drop_table("figures")
    op.drop_table("routines")
    op.drop_table("group_invites")
    op.drop_table("group_memberships")
    op.drop_table("groups")
    op.drop_table("dances")
    op.drop_table("users")

    # Drop multi-table enum explicitly
    bind = op.get_bind()
    group_role_enum.drop(bind, checkfirst=True)
