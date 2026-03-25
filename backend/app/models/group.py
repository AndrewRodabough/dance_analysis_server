"""Group, GroupMembership, and GroupInvite database models."""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class GroupRole(str, PyEnum):
    """Role within a group."""

    OWNER = "owner"
    COACH = "coach"
    MEMBER = "member"


class MembershipStatus(str, PyEnum):
    """Status of a group membership."""

    ACTIVE = "active"
    INVITED = "invited"
    REMOVED = "removed"


class GroupInviteStatus(str, PyEnum):
    """Status of a group invite."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REVOKED = "revoked"
    EXPIRED = "expired"


class Group(Base):
    """A collaborative group for organizing routines, videos, and notes."""

    __tablename__ = "groups"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_by = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    is_archived = Column(Boolean, server_default="false", nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    creator = relationship("User", backref="created_groups", foreign_keys=[created_by])
    memberships = relationship(
        "GroupMembership",
        back_populates="group",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    routine_sessions = relationship(
        "RoutineSession",
        back_populates="group",
    )
    invites = relationship(
        "GroupInvite",
        back_populates="group",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self):
        return f"<Group(id={self.id}, name={self.name})>"


class GroupMembership(Base):
    """Join table linking users to groups with roles."""

    __tablename__ = "group_memberships"

    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("groups.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    role = Column(
        SQLEnum(
            GroupRole,
            values_callable=lambda enum: [e.value for e in enum],
            name="group_role",
        ),
        nullable=False,
    )
    status = Column(
        SQLEnum(
            MembershipStatus,
            values_callable=lambda enum: [e.value for e in enum],
            name="membership_status",
        ),
        nullable=False,
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    group = relationship("Group", back_populates="memberships")
    user = relationship("User", backref="group_memberships")

    def __repr__(self):
        return f"<GroupMembership(group_id={self.group_id}, user_id={self.user_id}, role={self.role})>"


class GroupInvite(Base):
    """Email-first group invite for pre-account users."""

    __tablename__ = "group_invites"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_by = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    email = Column(String(255), nullable=False, index=True)
    role = Column(
        SQLEnum(
            GroupRole,
            values_callable=lambda enum: [e.value for e in enum],
            name="group_role",
        ),
        nullable=True,
    )
    token = Column(String(255), unique=True, nullable=False, index=True)
    status = Column(
        SQLEnum(
            GroupInviteStatus,
            values_callable=lambda enum: [e.value for e in enum],
            name="group_invite_status",
        ),
        nullable=False,
    )
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    accepted_by_user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    group = relationship("Group", back_populates="invites")
    creator = relationship(
        "User", backref="created_group_invites", foreign_keys=[created_by]
    )
    accepted_by_user = relationship("User", foreign_keys=[accepted_by_user_id])

    def __repr__(self):
        return f"<GroupInvite(id={self.id}, group_id={self.group_id}, email={self.email}, status={self.status})>"
