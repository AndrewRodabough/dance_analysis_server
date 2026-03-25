"""RoutineDancerSlot database model."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class RoutineDancerSlot(Base):
    """A named dancer slot within a routine's choreography."""

    __tablename__ = "routine_dancer_slots"
    __table_args__ = (
        UniqueConstraint("routine_id", "label", name="uq_routine_dancer_slots_routine_label"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    routine_id = Column(
        UUID(as_uuid=True),
        ForeignKey("routines.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    label = Column(String(50), nullable=False)
    order_index = Column(Integer, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    routine = relationship("Routine", back_populates="dancer_slots")
    assignments = relationship(
        "SlotAssignment",
        back_populates="dancer_slot",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self):
        return f"<RoutineDancerSlot(id={self.id}, routine_id={self.routine_id}, label={self.label})>"
