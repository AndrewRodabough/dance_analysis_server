"""SessionUserState database model for per-user session visibility."""

import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class SessionUserState(Base):
    """Per-user visibility and state for a routine session."""

    __tablename__ = "session_user_states"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "user_id",
            name="uq_session_user_states_session_user",
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
    is_archived = Column(Boolean, server_default="false", nullable=False)
    is_deleted = Column(Boolean, server_default="false", nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    routine_session = relationship("RoutineSession", back_populates="user_states")
    user = relationship("User", backref="session_user_states")

    def __repr__(self):
        return (
            f"<SessionUserState(id={self.id}, session={self.session_id}, "
            f"user={self.user_id}, archived={self.is_archived}, deleted={self.is_deleted})>"
        )
