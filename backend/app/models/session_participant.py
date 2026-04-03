"""SessionParticipant database model for participation tracking."""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class ParticipantRole(str, PyEnum):
    """Role of a participant in a session."""

    DANCER = "dancer"
    COACH = "coach"


class SessionParticipant(Base):
    """Represents a user's participation in a routine session."""

    __tablename__ = "session_participants"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "user_id",
            name="uq_session_participants_session_user",
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
            ParticipantRole,
            values_callable=lambda enum: [e.value for e in enum],
            name="participant_role",
        ),
        nullable=False,
    )
    dancer_slot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("routine_dancer_slots.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    routine_session = relationship("RoutineSession", back_populates="participants")
    user = relationship("User", backref="session_participants")
    dancer_slot = relationship("RoutineDancerSlot", backref="session_participants")

    def __repr__(self):
        return (
            f"<SessionParticipant(id={self.id}, session={self.session_id}, "
            f"user={self.user_id}, role={self.role})>"
        )
