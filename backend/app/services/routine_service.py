"""Service for managing routine operations."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.routine import Routine
from app.schemas.routine import RoutineCreate, RoutineUpdate
from app.services.routine_session_service import RoutineSessionService


class RoutinesService:
    """Service for managing routines as reusable choreography definitions."""

    @staticmethod
    def create_routine(db: Session, user_id: UUID, data: RoutineCreate) -> Routine:
        """Create a routine.

        This is atomic: creates the routine and its default session.
        The default session owner is the routine creator.
        """
        routine = Routine(
            title=data.title,
            dance_id=data.dance_id,
            created_by=user_id,
        )
        db.add(routine)
        db.flush()

        # Create default session for the routine
        # This also grants the creator admin access
        RoutineSessionService.create_default_session(
            db,
            routine.id,
            user_id,
        )

        db.commit()
        db.refresh(routine)
        return routine

    @staticmethod
    def list_user_routines(db: Session, user_id: UUID) -> List[Routine]:
        """List all routines created by a user."""
        return (
            db.query(Routine)
            .filter(Routine.created_by == user_id)
            .order_by(Routine.created_at.desc())
            .all()
        )

    @staticmethod
    def get_by_id(db: Session, routine_id: UUID) -> Optional[Routine]:
        """Get a routine by ID."""
        return db.query(Routine).filter(Routine.id == routine_id).first()

    @staticmethod
    def update_routine(db: Session, routine: Routine, data: RoutineUpdate) -> Routine:
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
        """Delete a routine and all associated sessions and data."""
        db.delete(routine)
        db.commit()
        return True
