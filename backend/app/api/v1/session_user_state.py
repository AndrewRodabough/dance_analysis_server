"""Session user state management endpoints for archiving and deletion."""

from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.core.authorization import require_session_access
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.routine_session import RoutineSession
from app.models.session_access import SessionAccess
from app.models.session_user_state import SessionUserState
from app.models.user import User
from app.schemas.routine_session import RoutineSessionResponse
from app.schemas.session_user_state import SessionUserStateResponse
from app.services.routine_session_service import RoutineSessionService

router = APIRouter()


@router.post(
    "/sessions/{session_id}/archive",
    response_model=SessionUserStateResponse,
    status_code=status.HTTP_200_OK,
)
def archive_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Archive a session for the current user.

    Archived sessions remain visible but are typically grouped or hidden.
    Requires: user has access to the session.
    """
    require_session_access(db, session_id, current_user.id)
    return RoutineSessionService.archive_for_user(db, session_id, current_user.id)


@router.post(
    "/sessions/{session_id}/unarchive",
    response_model=SessionUserStateResponse,
    status_code=status.HTTP_200_OK,
)
def unarchive_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Unarchive a session for the current user.

    Requires: user has access to the session.
    """
    require_session_access(db, session_id, current_user.id)
    return RoutineSessionService.unarchive_for_user(db, session_id, current_user.id)


@router.post(
    "/sessions/{session_id}/delete",
    response_model=SessionUserStateResponse,
    status_code=status.HTTP_200_OK,
)
def delete_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Mark a session as deleted for the current user.

    Deleted sessions will be excluded from normal session lists.
    Requires: user has access to the session.
    """
    require_session_access(db, session_id, current_user.id)
    return RoutineSessionService.mark_deleted_for_user(db, session_id, current_user.id)


@router.post(
    "/sessions/{session_id}/restore",
    response_model=SessionUserStateResponse,
    status_code=status.HTTP_200_OK,
)
def restore_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Restore a deleted session for the current user.

    Requires: user has access to the session (even if deleted).
    """
    require_session_access(db, session_id, current_user.id)
    return RoutineSessionService.restore_for_user(db, session_id, current_user.id)


@router.get(
    "/me/sessions",
    response_model=list[RoutineSessionResponse],
)
def list_my_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all sessions the current user has access to.

    Excludes sessions the user has soft-deleted (SessionUserState.is_deleted=true).
    Archived sessions are included; callers can check /me/sessions/state for flags.
    Sessions are ordered newest-first.
    """
    return (
        db.query(RoutineSession)
        .join(
            SessionAccess,
            and_(
                SessionAccess.session_id == RoutineSession.id,
                SessionAccess.user_id == current_user.id,
            ),
        )
        .outerjoin(
            SessionUserState,
            and_(
                SessionUserState.session_id == RoutineSession.id,
                SessionUserState.user_id == current_user.id,
            ),
        )
        .filter(
            or_(
                SessionUserState.is_deleted == None,  # noqa: E711
                SessionUserState.is_deleted == False,  # noqa: E712
            )
        )
        .order_by(RoutineSession.created_at.desc())
        .all()
    )


@router.get(
    "/me/sessions/state",
    response_model=list[SessionUserStateResponse],
)
def list_my_session_states(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List per-user state for all sessions the current user has access to.

    Returns an entry for every accessible session. Sessions with no explicit
    state row are returned with id=null and both flags false (the default state).
    """
    accessible_session_ids = [
        row.session_id
        for row in db.query(SessionAccess.session_id)
        .filter(SessionAccess.user_id == current_user.id)
        .all()
    ]

    existing_states = {
        s.session_id: s
        for s in db.query(SessionUserState)
        .filter(
            SessionUserState.user_id == current_user.id,
            SessionUserState.session_id.in_(accessible_session_ids),
        )
        .all()
    }

    result = []
    for session_id in accessible_session_ids:
        if session_id in existing_states:
            result.append(existing_states[session_id])
        else:
            result.append(
                SessionUserStateResponse(
                    id=None,
                    session_id=session_id,
                    user_id=current_user.id,
                    is_archived=False,
                    is_deleted=False,
                    created_at=None,
                    updated_at=None,
                )
            )
    return result
