"""SessionAccess database model."""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class SessionAccessRole(str, PyEnum):
    """Role for session access."""

    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"


class SessionAccess(Base):
    """Controls direct user access to a session."""

    __tablename__ = "session_access"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "user_id",
            name="uq_session_access_session_user",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("routine_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(
        SQLEnum(
            SessionAccessRole,
            values_callable=lambda enum: [e.value for e in enum],
            name="session_access_role",
        ),
        nullable=False,
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    routine_session = relationship("RoutineSession", back_populates="access_records")
    user = relationship("User", backref="session_access_records")

    def __repr__(self):
        return (
            f"<SessionAccess(id={self.id}, session_id={self.session_id}, "
            f"user_id={self.user_id}, role={self.role})>"
        )
