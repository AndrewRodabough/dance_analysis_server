"""SessionInvite database model."""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class SessionInviteStatus(str, PyEnum):
    """Status of a session invite."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REVOKED = "revoked"
    EXPIRED = "expired"


class SessionInvite(Base):
    """Email-first session invite for direct session access."""

    __tablename__ = "session_invites"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("routine_sessions.id", ondelete="CASCADE"),
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
    role = Column(String(50), nullable=False, default="viewer")
    token = Column(String(255), unique=True, nullable=False, index=True)
    status = Column(
        SQLEnum(
            SessionInviteStatus,
            values_callable=lambda enum: [e.value for e in enum],
            name="session_invite_status",
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
    routine_session = relationship("RoutineSession", backref="session_invites")
    creator = relationship(
        "User", backref="created_session_invites", foreign_keys=[created_by]
    )
    accepted_by_user = relationship("User", foreign_keys=[accepted_by_user_id])

    def __repr__(self):
        return f"<SessionInvite(id={self.id}, session_id={self.session_id}, email={self.email}, status={self.status})>"
