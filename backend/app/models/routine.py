"""Routine database model."""

from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class RoutineRole(str, PyEnum):
    """Role of a user participating in a routine."""
    DANCER = "dancer"
    COACH = "coach"


class Routine(Base):
    """A dance routine with associated dance, participants, and notes."""
    __tablename__ = "routines"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    dance_id = Column(Integer, ForeignKey("dances.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    owner = relationship("User", backref="owned_routines")
    dance = relationship("Dance", backref="routines")
    participants = relationship(
        "RoutineParticipant",
        back_populates="routine",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    notes = relationship(
        "Note",
        back_populates="routine",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self):
        return f"<Routine(id={self.id}, title={self.title}, owner_id={self.owner_id})>"


class RoutineParticipant(Base):
    """Join table linking users to routines with a specific role."""
    __tablename__ = "routine_participants"

    routine_id = Column(Integer, ForeignKey("routines.id", ondelete="CASCADE"), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, index=True)
    role = Column(
        SQLEnum(RoutineRole, values_callable=lambda enum: [e.value for e in enum], name="routine_role"),
        nullable=False,
    )

    routine = relationship("Routine", back_populates="participants")
    user = relationship("User", backref="routine_participations")

    def __repr__(self):
        return f"<RoutineParticipant(routine_id={self.routine_id}, user_id={self.user_id}, role={self.role})>"
