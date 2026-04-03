"""Remove 'owner' from group_role enum, migrating existing owner rows to member + isAdmin=true.

Revision ID: 004
Revises: 003
Create Date: 2026-04-02
"""

from alembic import op

revision: str = "004"
down_revision: str = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Promote existing owner memberships to member + isAdmin before dropping the value
    op.execute(
        """
        UPDATE group_memberships
        SET role = 'member', "isAdmin" = true
        WHERE role = 'owner'
        """
    )

    # Rename old enum, create new one without 'owner', migrate, drop old
    op.execute("ALTER TYPE group_role RENAME TO group_role_old")
    op.execute("CREATE TYPE group_role AS ENUM ('coach', 'member')")
    op.execute(
        """
        ALTER TABLE group_memberships
        ALTER COLUMN role TYPE group_role USING role::text::group_role
        """
    )
    op.execute(
        """
        ALTER TABLE group_invites
        ALTER COLUMN role TYPE group_role USING role::text::group_role
        """
    )
    op.execute("DROP TYPE group_role_old")


def downgrade() -> None:
    op.execute("ALTER TYPE group_role RENAME TO group_role_old")
    op.execute("CREATE TYPE group_role AS ENUM ('owner', 'coach', 'member')")
    op.execute(
        """
        ALTER TABLE group_memberships
        ALTER COLUMN role TYPE group_role USING role::text::group_role
        """
    )
    op.execute(
        """
        ALTER TABLE group_invites
        ALTER COLUMN role TYPE group_role USING role::text::group_role
        """
    )
    op.execute("DROP TYPE group_role_old")
