"""Refactor videos to routine-scoped with upload lifecycle status

Revision ID: 008
Revises: 007
Create Date: 2026-03-20
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None

video_status_enum = sa.Enum("pending_upload", "uploaded", "deleted", name="video_status")


def upgrade() -> None:
    bind = op.get_bind()
    video_status_enum.create(bind, checkfirst=True)

    # Add routine_id to videos
    op.add_column(
        "videos",
        sa.Column(
            "routine_id",
            sa.Integer(),
            sa.ForeignKey("routines.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.create_index(op.f("ix_videos_routine_id"), "videos", ["routine_id"])

    # Rename owner_id to uploaded_by
    op.alter_column("videos", "owner_id", new_column_name="uploaded_by")

    # Add status column
    op.add_column(
        "videos",
        sa.Column(
            "status",
            video_status_enum,
            server_default="pending_upload",
            nullable=False,
        ),
    )
    op.create_index(op.f("ix_videos_status"), "videos", ["status"])

    # Add optional metadata columns
    op.add_column(
        "videos",
        sa.Column("original_filename", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "videos",
        sa.Column("content_type", sa.String(length=100), nullable=True),
    )

    # Drop video_permissions table
    op.drop_index(op.f("ix_video_permissions_video_id"), table_name="video_permissions", if_exists=True)
    op.drop_index(op.f("ix_video_permissions_user_id"), table_name="video_permissions", if_exists=True)
    op.drop_table("video_permissions")

    # Drop visibility column and enum
    op.drop_index(op.f("ix_videos_visibility"), table_name="videos", if_exists=True)
    op.drop_column("videos", "visibility")
    sa.Enum(name="video_visibility").drop(bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()

    # Re-create video_visibility enum and column
    video_visibility_enum = sa.Enum("private", "shared", name="video_visibility")
    video_visibility_enum.create(bind, checkfirst=True)
    op.add_column(
        "videos",
        sa.Column(
            "visibility",
            video_visibility_enum,
            server_default="private",
            nullable=False,
        ),
    )
    op.create_index(op.f("ix_videos_visibility"), "videos", ["visibility"])

    # Re-create video_permissions
    op.create_table(
        "video_permissions",
        sa.Column(
            "video_id",
            sa.Integer(),
            sa.ForeignKey("videos.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("can_view", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("can_download", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("can_comment", sa.Boolean(), server_default="false", nullable=False),
    )

    # Drop new columns
    op.drop_column("videos", "content_type")
    op.drop_column("videos", "original_filename")

    op.drop_index(op.f("ix_videos_status"), table_name="videos")
    op.drop_column("videos", "status")

    # Rename uploaded_by back to owner_id
    op.alter_column("videos", "uploaded_by", new_column_name="owner_id")

    op.drop_index(op.f("ix_videos_routine_id"), table_name="videos")
    op.drop_column("videos", "routine_id")

    video_status_enum.drop(bind, checkfirst=True)
