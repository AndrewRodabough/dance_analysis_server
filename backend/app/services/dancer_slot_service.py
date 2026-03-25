"""Service for managing routine dancer slot operations."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.routine_dancer_slot import RoutineDancerSlot
from app.schemas.routine_dancer_slot import RoutineDancerSlotCreate


class DancerSlotService:
    """Service for managing dancer slots within a routine."""

    @staticmethod
    def create(
        db: Session, routine_id: UUID, data: RoutineDancerSlotCreate
    ) -> RoutineDancerSlot:
        """Create a dancer slot."""
        slot = RoutineDancerSlot(
            routine_id=routine_id,
            label=data.label,
            order_index=data.order_index,
        )
        db.add(slot)
        db.commit()
        db.refresh(slot)
        return slot

    @staticmethod
    def get_by_id(db: Session, slot_id: UUID) -> Optional[RoutineDancerSlot]:
        """Get a dancer slot by ID."""
        return (
            db.query(RoutineDancerSlot)
            .filter(RoutineDancerSlot.id == slot_id)
            .first()
        )

    @staticmethod
    def list_for_routine(db: Session, routine_id: UUID) -> List[RoutineDancerSlot]:
        """List all dancer slots for a routine."""
        return (
            db.query(RoutineDancerSlot)
            .filter(RoutineDancerSlot.routine_id == routine_id)
            .order_by(RoutineDancerSlot.order_index.asc().nulls_last())
            .all()
        )

    @staticmethod
    def delete(db: Session, slot: RoutineDancerSlot) -> bool:
        """Delete a dancer slot."""
        db.delete(slot)
        db.commit()
        return True
