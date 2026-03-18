"""Add video metadata tables and link jobs to videos

Revision ID: 003
Revises: 002
Create Date: 2026-03-15
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None

video_visibility_enum = sa.Enum("private", "shared", name="video_visibility")


def upgrade():
    bind = op.get_bind()
    video_visibility_enum.create(bind, checkfirst=True)

    op.create_table(
        "videos",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("storage_key", sa.String(length=500), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("duration", sa.Integer(), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=True),
        sa.Column(
            "visibility",
            video_visibility_enum,
            nullable=False,
            server_default="private",
        ),
        sa.UniqueConstraint("storage_key", name="uq_videos_storage_key"),
    )
    op.create_index(op.f("ix_videos_owner_id"), "videos", ["owner_id"])
    op.create_index(op.f("ix_videos_visibility"), "videos", ["visibility"])

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
        sa.Column(
            "can_view",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "can_download",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "can_comment",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index(
        op.f("ix_video_permissions_video_id"),
        "video_permissions",
        ["video_id"],
    )
    op.create_index(
        op.f("ix_video_permissions_user_id"),
        "video_permissions",
        ["user_id"],
    )

    op.add_column(
        "jobs",
        sa.Column("video_id", sa.Integer(), nullable=True),
    )
    op.create_index(op.f("ix_jobs_video_id"), "jobs", ["video_id"])
    op.create_foreign_key(
        "fk_jobs_video_id_videos",
        "jobs",
        "videos",
        ["video_id"],
        ["id"],
        ondelete="SET NULL",
    )

    conn = op.get_bind()

    conn.execute(
        sa.text(
            """
            INSERT INTO videos (owner_id, storage_key, created_at, duration, file_size, visibility)
            SELECT DISTINCT
                j.user_id,
                j.video_path,
                COALESCE(j.created_at, now()),
                NULL,
                NULL,
                'private'
            FROM jobs j
            WHERE j.video_path IS NOT NULL
              AND j.video_path <> ''
            ON CONFLICT (storage_key) DO NOTHING
            """
        )
    )

    conn.execute(
        sa.text(
            """
            UPDATE jobs
            SET video_id = v.id
            FROM videos v
            WHERE jobs.video_path = v.storage_key
              AND jobs.user_id = v.owner_id
            """
        )
    )

    conn.execute(
        sa.text(
            """
            INSERT INTO video_permissions (video_id, user_id, can_view, can_download, can_comment)
            SELECT v.id, v.owner_id, TRUE, TRUE, TRUE
            FROM videos v
            ON CONFLICT (video_id, user_id) DO NOTHING
            """
        )
    )

    op.drop_column("jobs", "video_path")


def downgrade():
    op.add_column(
        "jobs",
        sa.Column("video_path", sa.String(length=500), nullable=True),
    )

    conn = op.get_bind()
    conn.execute(
        sa.text(
            """
            UPDATE jobs
            SET video_path = v.storage_key
            FROM videos v
            WHERE jobs.video_id = v.id
            """
        )
    )

    op.drop_constraint("fk_jobs_video_id_videos", "jobs", type_="foreignkey")
    op.drop_index(op.f("ix_jobs_video_id"), table_name="jobs")
    op.drop_column("jobs", "video_id")

    op.drop_index(op.f("ix_video_permissions_user_id"), table_name="video_permissions")
    op.drop_index(op.f("ix_video_permissions_video_id"), table_name="video_permissions")
    op.drop_table("video_permissions")

    op.drop_index(op.f("ix_videos_owner_id"), table_name="videos")
    op.drop_index(op.f("ix_videos_visibility"), table_name="videos")
    op.drop_table("videos")

    video_visibility_enum.drop(op.get_bind(), checkfirst=True)
