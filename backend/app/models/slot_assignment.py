"""SlotAssignment database model."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class SlotAssignment(Base):
    """Assigns a user to a dancer slot for a specific routine session."""

    __tablename__ = "slot_assignments"
    __table_args__ = (
        UniqueConstraint(
            "routine_session_id", "dancer_slot_id",
            name="uq_slot_assignments_session_slot",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    routine_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("routine_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dancer_slot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("routine_dancer_slots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    routine_session = relationship("RoutineSession", back_populates="slot_assignments")
    dancer_slot = relationship("RoutineDancerSlot", back_populates="assignments")
    user = relationship("User", backref="slot_assignments")

    def __repr__(self):
        return (
            f"<SlotAssignment(id={self.id}, session={self.routine_session_id}, "
            f"slot={self.dancer_slot_id}, user={self.user_id})>"
        )
