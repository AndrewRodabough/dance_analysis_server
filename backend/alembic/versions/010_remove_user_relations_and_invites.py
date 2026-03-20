"""Remove legacy user_relations and invites tables

Revision ID: 010
Revises: 009
Create Date: 2026-03-20
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()

    # Drop invites table
    op.drop_index(op.f("ix_invites_id"), table_name="invites", if_exists=True)
    op.drop_index(op.f("ix_invites_created_by"), table_name="invites", if_exists=True)
    op.drop_table("invites")

    # Drop user_relations table
    op.drop_index(op.f("ix_user_relations_id"), table_name="user_relations", if_exists=True)
    op.drop_index(op.f("ix_user_relations_user_id"), table_name="user_relations", if_exists=True)
    op.drop_index(op.f("ix_user_relations_related_user_id"), table_name="user_relations", if_exists=True)
    op.drop_table("user_relations")

    # Drop legacy enum types
    sa.Enum(name="invite_role").drop(bind, checkfirst=True)
    sa.Enum(name="invite_status").drop(bind, checkfirst=True)
    sa.Enum(name="user_relation_role").drop(bind, checkfirst=True)
    sa.Enum(name="user_relation_status").drop(bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()

    # Re-create enum types
    user_relation_role = sa.Enum("coach", "student", "partner", name="user_relation_role")
    user_relation_status = sa.Enum("pending", "accepted", "rejected", "expired", name="user_relation_status")
    invite_role = sa.Enum("coach", "student", "partner", name="invite_role")
    invite_status = sa.Enum("pending", "accepted", "rejected", "expired", name="invite_status")

    user_relation_role.create(bind, checkfirst=True)
    user_relation_status.create(bind, checkfirst=True)
    invite_role.create(bind, checkfirst=True)
    invite_status.create(bind, checkfirst=True)

    # Re-create user_relations
    op.create_table(
        "user_relations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column(
            "related_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("role", user_relation_role, nullable=False),
        sa.Column("status", user_relation_status, nullable=False, server_default="pending"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(op.f("ix_user_relations_id"), "user_relations", ["id"], unique=False)
    op.create_index(op.f("ix_user_relations_user_id"), "user_relations", ["user_id"])
    op.create_index(op.f("ix_user_relations_related_user_id"), "user_relations", ["related_user_id"])

    # Re-create invites
    op.create_table(
        "invites",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("token", sa.String(length=255), unique=True, nullable=False),
        sa.Column(
            "created_by",
            sa.Integer(),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("phone_number", sa.String(length=32), nullable=True),
        sa.Column("role", invite_role, nullable=False),
        sa.Column("status", invite_status, nullable=False, server_default="pending"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(op.f("ix_invites_id"), "invites", ["id"], unique=False)
    op.create_index(op.f("ix_invites_created_by"), "invites", ["created_by"])
