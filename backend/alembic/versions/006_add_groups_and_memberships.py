"""Add groups, group_memberships, and group_invites tables

Revision ID: 006
Revises: 005
Create Date: 2026-03-20
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None

group_role_enum = sa.Enum("owner", "coach", "member", name="group_role")
membership_status_enum = sa.Enum("active", "invited", "removed", name="membership_status")
invite_status_enum = sa.Enum("pending", "accepted", "revoked", "expired", name="group_invite_status")


def upgrade() -> None:
    bind = op.get_bind()
    group_role_enum.create(bind, checkfirst=True)
    membership_status_enum.create(bind, checkfirst=True)
    invite_status_enum.create(bind, checkfirst=True)

    # --- groups ---
    op.create_table(
        "groups",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_by",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("is_archived", sa.Boolean(), server_default="false", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f("ix_groups_id"), "groups", ["id"], unique=False)
    op.create_index(op.f("ix_groups_created_by"), "groups", ["created_by"])

    # --- group_memberships ---
    op.create_table(
        "group_memberships",
        sa.Column(
            "group_id",
            sa.Integer(),
            sa.ForeignKey("groups.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
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
    op.create_index(op.f("ix_group_memberships_user_id"), "group_memberships", ["user_id"])
    op.create_index(op.f("ix_group_memberships_group_id"), "group_memberships", ["group_id"])

    # --- group_invites ---
    op.create_table(
        "group_invites",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "group_id",
            sa.Integer(),
            sa.ForeignKey("groups.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_by",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("role", group_role_enum, nullable=True),
        sa.Column("token", sa.String(length=255), unique=True, nullable=False),
        sa.Column("status", invite_status_enum, nullable=False),
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
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(op.f("ix_group_invites_id"), "group_invites", ["id"], unique=False)
    op.create_index(op.f("ix_group_invites_group_id"), "group_invites", ["group_id"])
    op.create_index(op.f("ix_group_invites_created_by"), "group_invites", ["created_by"])
    op.create_index(op.f("ix_group_invites_email"), "group_invites", ["email"])
    op.create_index(op.f("ix_group_invites_token"), "group_invites", ["token"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_group_invites_token"), table_name="group_invites")
    op.drop_index(op.f("ix_group_invites_email"), table_name="group_invites")
    op.drop_index(op.f("ix_group_invites_created_by"), table_name="group_invites")
    op.drop_index(op.f("ix_group_invites_group_id"), table_name="group_invites")
    op.drop_index(op.f("ix_group_invites_id"), table_name="group_invites")
    op.drop_table("group_invites")

    op.drop_index(op.f("ix_group_memberships_group_id"), table_name="group_memberships")
    op.drop_index(op.f("ix_group_memberships_user_id"), table_name="group_memberships")
    op.drop_table("group_memberships")

    op.drop_index(op.f("ix_groups_created_by"), table_name="groups")
    op.drop_index(op.f("ix_groups_id"), table_name="groups")
    op.drop_table("groups")

    bind = op.get_bind()
    invite_status_enum.drop(bind, checkfirst=True)
    membership_status_enum.drop(bind, checkfirst=True)
    group_role_enum.drop(bind, checkfirst=True)
