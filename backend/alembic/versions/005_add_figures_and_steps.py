"""Add figures and steps tables

Revision ID: 005
Revises: 004
Create Date: 2026-03-20
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "figures",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("tags", JSON, nullable=True),
        sa.Column("total_beats", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f("ix_figures_id"), "figures", ["id"], unique=False)

    op.create_table(
        "steps",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "figure_id",
            sa.Integer(),
            sa.ForeignKey("figures.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("start_state", JSON, nullable=False),
        sa.Column("end_state", JSON, nullable=False),
        sa.Column("keyframes", JSON, nullable=True),
        sa.Column("duration", sa.Float(), nullable=False),
    )
    op.create_index(op.f("ix_steps_id"), "steps", ["id"], unique=False)
    op.create_index(op.f("ix_steps_figure_id"), "steps", ["figure_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_steps_figure_id"), table_name="steps")
    op.drop_index(op.f("ix_steps_id"), table_name="steps")
    op.drop_table("steps")

    op.drop_index(op.f("ix_figures_id"), table_name="figures")
    op.drop_table("figures")
