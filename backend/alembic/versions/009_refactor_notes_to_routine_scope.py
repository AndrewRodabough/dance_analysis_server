"""Refactor notes to add source, video_timestamp_ms, details, video_deleted

Revision ID: 009
Revises: 008
Create Date: 2026-03-20
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None

note_source_enum = sa.Enum("user", "ai", "system", name="note_source")


def upgrade() -> None:
    bind = op.get_bind()
    note_source_enum.create(bind, checkfirst=True)

    # Add source column
    op.add_column(
        "notes",
        sa.Column("source", note_source_enum, server_default="user", nullable=False),
    )

    # Add video_deleted flag
    op.add_column(
        "notes",
        sa.Column("video_deleted", sa.Boolean(), server_default="false", nullable=False),
    )

    # Add video_timestamp_ms (milliseconds precision)
    op.add_column(
        "notes",
        sa.Column("video_timestamp_ms", sa.Integer(), nullable=True),
    )

    # Migrate existing video_timestamp (seconds) to video_timestamp_ms (milliseconds)
    op.execute("UPDATE notes SET video_timestamp_ms = video_timestamp * 1000 WHERE video_timestamp IS NOT NULL")

    # Drop old video_timestamp column
    op.drop_column("notes", "video_timestamp")

    # Add structured details column (JSONB)
    op.add_column(
        "notes",
        sa.Column("details", JSONB, nullable=True),
    )


def downgrade() -> None:
    # Re-add video_timestamp column
    op.add_column(
        "notes",
        sa.Column("video_timestamp", sa.Integer(), nullable=True),
    )

    # Migrate video_timestamp_ms back to seconds
    op.execute("UPDATE notes SET video_timestamp = video_timestamp_ms / 1000 WHERE video_timestamp_ms IS NOT NULL")

    # Drop new columns
    op.drop_column("notes", "details")
    op.drop_column("notes", "video_timestamp_ms")
    op.drop_column("notes", "video_deleted")
    op.drop_column("notes", "source")

    bind = op.get_bind()
    note_source_enum.drop(bind, checkfirst=True)
