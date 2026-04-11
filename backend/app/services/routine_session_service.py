"""Service for managing routine session operations."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.routine_session import RoutineSession
from app.models.session_access import SessionAccessRole
from app.models.session_user_state import SessionUserState
from app.schemas.routine_session import RoutineSessionCreate
from app.services.session_access_service import SessionAccessService


class RoutineSessionService:
    """Service for managing routine sessions."""

    @staticmethod
    def create(
        db: Session,
        routine_id: UUID,
        created_by: UUID,
        data: RoutineSessionCreate,
    ) -> RoutineSession:
        """
        Create a routine session.

        This is atomic: creates the session and grants owner admin access.
        """
        # Create the session with owner = creator
        session = RoutineSession(
            routine_id=routine_id,
            created_by=created_by,
            owner_id=created_by,
            label=data.label,
        )
        db.add(session)
        db.flush()

        # Grant owner admin access
        SessionAccessService.grant_direct_access(
            db,
            session.id,
            created_by,
            role=SessionAccessRole.ADMIN,
        )

        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def create_default_session(
        db: Session,
        routine_id: UUID,
        owner_id: UUID,
    ) -> RoutineSession:
        """
        Create the default session for a routine.

        This is called automatically when a routine is created.
        The default session owner is the routine creator.
        """
        session = RoutineSession(
            routine_id=routine_id,
            created_by=owner_id,
            owner_id=owner_id,
            label=None,  # Default sessions have no label
        )
        db.add(session)
        db.flush()

        # Grant owner admin access
        SessionAccessService.grant_direct_access(
            db,
            session.id,
            owner_id,
            role=SessionAccessRole.ADMIN,
        )

        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def get_by_id(db: Session, session_id: UUID) -> Optional[RoutineSession]:
        """Get a routine session by ID."""
        return db.query(RoutineSession).filter(RoutineSession.id == session_id).first()

    @staticmethod
    def get_default_session(db: Session, routine_id: UUID) -> Optional[RoutineSession]:
        """
        Get the default session for a routine.

        The default session is the one with the earliest created_at timestamp.
        """
        return (
            db.query(RoutineSession)
            .filter(
                RoutineSession.routine_id == routine_id,
            )
            .order_by(RoutineSession.created_at.asc())
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
    def list_for_routine_filtered(
        db: Session,
        routine_id: UUID,
        current_user_id: UUID,
        include_archived: bool = True,
        include_deleted: bool = False,
    ) -> List[RoutineSession]:
        """
        List sessions for a routine with visibility filtering.

        Filters based on current user's session state (archived, deleted).
        """
        sessions = RoutineSessionService.list_for_routine(db, routine_id)

        filtered = []
        for session in sessions:
            user_state = (
                db.query(SessionUserState)
                .filter(
                    SessionUserState.session_id == session.id,
                    SessionUserState.user_id == current_user_id,
                )
                .first()
            )

            # Skip deleted sessions unless explicitly included
            if user_state is not None and user_state.is_deleted and not include_deleted:
                continue

            # Include or skip archived sessions based on flag
            if (
                user_state is not None
                and user_state.is_archived
                and not include_archived
            ):
                continue

            filtered.append(session)

        return filtered

    @staticmethod
    def archive_for_user(
        db: Session, session_id: UUID, user_id: UUID
    ) -> SessionUserState:
        """Archive a session for the current user."""
        user_state = (
            db.query(SessionUserState)
            .filter(
                SessionUserState.session_id == session_id,
                SessionUserState.user_id == user_id,
            )
            .first()
        )

        if user_state is not None:
            user_state.is_archived = True
        else:
            user_state = SessionUserState(
                session_id=session_id,
                user_id=user_id,
                is_archived=True,
                is_deleted=False,
            )
            db.add(user_state)

        db.commit()
        db.refresh(user_state)
        return user_state

    @staticmethod
    def unarchive_for_user(
        db: Session, session_id: UUID, user_id: UUID
    ) -> SessionUserState:
        """Unarchive a session for the current user."""
        user_state = (
            db.query(SessionUserState)
            .filter(
                SessionUserState.session_id == session_id,
                SessionUserState.user_id == user_id,
            )
            .first()
        )

        if user_state is not None:
            user_state.is_archived = False
        else:
            user_state = SessionUserState(
                session_id=session_id,
                user_id=user_id,
                is_archived=False,
                is_deleted=False,
            )
            db.add(user_state)

        db.commit()
        db.refresh(user_state)
        return user_state

    @staticmethod
    def mark_deleted_for_user(
        db: Session, session_id: UUID, user_id: UUID
    ) -> SessionUserState:
        """Mark a session as deleted for the current user."""
        user_state = (
            db.query(SessionUserState)
            .filter(
                SessionUserState.session_id == session_id,
                SessionUserState.user_id == user_id,
            )
            .first()
        )

        if user_state is not None:
            user_state.is_deleted = True
        else:
            user_state = SessionUserState(
                session_id=session_id,
                user_id=user_id,
                is_archived=False,
                is_deleted=True,
            )
            db.add(user_state)

        db.commit()
        db.refresh(user_state)
        return user_state

    @staticmethod
    def restore_for_user(
        db: Session, session_id: UUID, user_id: UUID
    ) -> SessionUserState:
        """Restore a deleted session for the current user."""
        user_state = (
            db.query(SessionUserState)
            .filter(
                SessionUserState.session_id == session_id,
                SessionUserState.user_id == user_id,
            )
            .first()
        )

        if user_state is not None:
            user_state.is_deleted = False
        else:
            user_state = SessionUserState(
                session_id=session_id,
                user_id=user_id,
                is_archived=False,
                is_deleted=False,
            )
            db.add(user_state)

        db.commit()
        db.refresh(user_state)
        return user_state

    @staticmethod
    def delete(db: Session, session: RoutineSession) -> bool:
        """Delete a routine session and all associated data."""
        db.delete(session)
        db.commit()
        return True

    @staticmethod
    def get_user_state(
        db: Session, session_id: UUID, user_id: UUID
    ) -> Optional[SessionUserState]:
        """Get the user state for a session."""
        return (
            db.query(SessionUserState)
            .filter(
                SessionUserState.session_id == session_id,
                SessionUserState.user_id == user_id,
            )
            .first()
        )
