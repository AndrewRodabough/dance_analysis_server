"""Service for managing routine operations."""

from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.routine import Routine
from app.schemas.routine import RoutineCreate, RoutineUpdate


class RoutinesService:
    """Service for managing routines scoped to groups."""

    @staticmethod
    def create_routine(
        db: Session, group_id: int, user_id: int, data: RoutineCreate
    ) -> Routine:
        """Create a routine within a group."""
        routine = Routine(
            title=data.title,
            dance_id=data.dance_id,
            group_id=group_id,
            created_by=user_id,
        )
        db.add(routine)
        db.commit()
        db.refresh(routine)
        return routine

    @staticmethod
    def list_routines(db: Session, group_id: int) -> List[Routine]:
        """List all routines in a group."""
        return (
            db.query(Routine)
            .filter(Routine.group_id == group_id)
            .order_by(Routine.created_at.desc())
            .all()
        )

    @staticmethod
    def get_routine(
        db: Session, group_id: int, routine_id: int
    ) -> Optional[Routine]:
        """Get a routine by ID within a group."""
        return (
            db.query(Routine)
            .filter(Routine.id == routine_id, Routine.group_id == group_id)
            .first()
        )

    @staticmethod
    def update_routine(
        db: Session, routine: Routine, data: RoutineUpdate
    ) -> Routine:
        """Update routine fields."""
        if data.title is not None:
            routine.title = data.title
        if data.dance_id is not None:
            routine.dance_id = data.dance_id
        db.commit()
        db.refresh(routine)
        return routine

    @staticmethod
    def delete_routine(db: Session, routine: Routine) -> bool:
        """Delete a routine."""
        db.delete(routine)
        db.commit()
        return True
