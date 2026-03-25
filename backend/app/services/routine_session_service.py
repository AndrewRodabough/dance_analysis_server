"""Service for managing routine session operations."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.routine_session import RoutineSession
from app.schemas.routine_session import RoutineSessionCreate


class RoutineSessionService:
    """Service for managing routine sessions."""

    @staticmethod
    def create(
        db: Session,
        routine_id: UUID,
        created_by: UUID,
        data: RoutineSessionCreate,
    ) -> RoutineSession:
        """Create a routine session."""
        session = RoutineSession(
            routine_id=routine_id,
            group_id=data.group_id,
            created_by=created_by,
            label=data.label,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def get_by_id(db: Session, session_id: UUID) -> Optional[RoutineSession]:
        """Get a routine session by ID."""
        return (
            db.query(RoutineSession)
            .filter(RoutineSession.id == session_id)
            .first()
        )

    @staticmethod
    def list_for_routine(db: Session, routine_id: UUID) -> List[RoutineSession]:
        """List sessions for a routine."""
        return (
            db.query(RoutineSession)
            .filter(RoutineSession.routine_id == routine_id)
            .order_by(RoutineSession.created_at.desc())
            .all()
        )

    @staticmethod
    def list_for_group(db: Session, group_id: UUID) -> List[RoutineSession]:
        """List sessions for a group."""
        return (
            db.query(RoutineSession)
            .filter(RoutineSession.group_id == group_id)
            .order_by(RoutineSession.created_at.desc())
            .all()
        )

    @staticmethod
    def delete(db: Session, session: RoutineSession) -> bool:
        """Delete a routine session."""
        db.delete(session)
        db.commit()
        return True
