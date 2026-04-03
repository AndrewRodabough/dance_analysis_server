"""RoutineSession database model."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class RoutineSession(Base):
    """An instance of a routine used in a specific context."""

    __tablename__ = "routine_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    routine_id = Column(
        UUID(as_uuid=True),
        ForeignKey("routines.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_by = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    owner_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    label = Column(String(255), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    routine = relationship("Routine", back_populates="sessions")
    creator = relationship(
        "User", foreign_keys=[created_by], backref="created_routine_sessions"
    )
    owner = relationship(
        "User", foreign_keys=[owner_id], backref="owned_routine_sessions"
    )
    slot_assignments = relationship(
        "SlotAssignment",
        back_populates="routine_session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    videos = relationship(
        "Video",
        back_populates="routine_session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    notes = relationship(
        "Note",
        back_populates="routine_session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    access_records = relationship(
        "SessionAccess",
        back_populates="routine_session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    participants = relationship(
        "SessionParticipant",
        back_populates="routine_session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    user_states = relationship(
        "SessionUserState",
        back_populates="routine_session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self):
        return f"<RoutineSession(id={self.id}, routine_id={self.routine_id}, owner_id={self.owner_id})>"
