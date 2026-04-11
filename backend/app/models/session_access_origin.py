"""SessionAccessOrigin database model for tracking access sources."""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class AccessSourceType(str, PyEnum):
    """Source type for session access."""

    DIRECT = "direct"
    GROUP = "group"


class SessionAccessOrigin(Base):
    """Tracks how a user gained access to a session (direct or via group)."""

    __tablename__ = "session_access_origins"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "user_id",
            "source_type",
            "group_id",
            name="uq_session_access_origins_unique",
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
    source_type = Column(
        SQLEnum(
            AccessSourceType,
            values_callable=lambda enum: [e.value for e in enum],
            name="session_access_origin_type",
        ),
        nullable=False,
    )
    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("groups.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    routine_session = relationship("RoutineSession", back_populates="access_origins")
    user = relationship("User", backref="access_origins")
    group = relationship("Group", backref="session_access_origins")

    def __repr__(self):
        return (
            f"<SessionAccessOrigin(id={self.id}, session_id={self.session_id}, "
            f"user_id={self.user_id}, source_type={self.source_type}, "
            f"group_id={self.group_id})>"
        )
