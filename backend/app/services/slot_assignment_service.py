"""Service for managing slot assignment operations."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.slot_assignment import SlotAssignment
from app.schemas.slot_assignment import SlotAssignmentCreate


class SlotAssignmentService:
    """Service for managing slot assignments within a session."""

    @staticmethod
    def create(
        db: Session, session_id: UUID, data: SlotAssignmentCreate
    ) -> SlotAssignment:
        """Assign a user to a dancer slot for a session."""
        assignment = SlotAssignment(
            routine_session_id=session_id,
            dancer_slot_id=data.dancer_slot_id,
            user_id=data.user_id,
        )
        db.add(assignment)
        db.commit()
        db.refresh(assignment)
        return assignment

    @staticmethod
    def get_by_id(db: Session, assignment_id: UUID) -> Optional[SlotAssignment]:
        """Get a slot assignment by ID."""
        return (
            db.query(SlotAssignment)
            .filter(SlotAssignment.id == assignment_id)
            .first()
        )

    @staticmethod
    def list_for_session(db: Session, session_id: UUID) -> List[SlotAssignment]:
        """List all slot assignments for a session."""
        return (
            db.query(SlotAssignment)
            .filter(SlotAssignment.routine_session_id == session_id)
            .all()
        )

    @staticmethod
    def delete(db: Session, assignment: SlotAssignment) -> bool:
        """Remove a slot assignment."""
        db.delete(assignment)
        db.commit()
        return True
