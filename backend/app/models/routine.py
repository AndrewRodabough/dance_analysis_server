"""Routine database model."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Routine(Base):
    """A reusable choreography definition."""

    __tablename__ = "routines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    title = Column(String(255), nullable=False)
    created_by = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dance_id = Column(
        UUID(as_uuid=True),
        ForeignKey("dances.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    creator = relationship("User", backref="created_routines")
    dance = relationship("Dance", backref="routines")
    sessions = relationship(
        "RoutineSession",
        back_populates="routine",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    dancer_slots = relationship(
        "RoutineDancerSlot",
        back_populates="routine",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self):
        return f"<Routine(id={self.id}, title={self.title})>"
