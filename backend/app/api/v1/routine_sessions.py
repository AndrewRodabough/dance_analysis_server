"""Routine session management endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.authorization import (
    require_group_member,
    require_routine,
    require_session_access,
)
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.schemas.routine_session import RoutineSessionCreate, RoutineSessionResponse
from app.services.routine_session_service import RoutineSessionService

router = APIRouter()


@router.post(
    "/routines/{routine_id}/sessions",
    response_model=RoutineSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session(
    routine_id: UUID,
    data: RoutineSessionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Create a session for a routine.

    - **group_id**: Optional group to scope the session to
    - **label**: Optional human-friendly label
    """
    require_routine(db, routine_id)
    if data.group_id is not None:
        require_group_member(db, data.group_id, current_user.id)
    return RoutineSessionService.create(db, routine_id, current_user.id, data)


@router.get(
    "/routines/{routine_id}/sessions",
    response_model=List[RoutineSessionResponse],
)
def list_routine_sessions(
    routine_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all sessions for a routine."""
    require_routine(db, routine_id)
    return RoutineSessionService.list_for_routine(db, routine_id)


@router.get(
    "/groups/{group_id}/sessions",
    response_model=List[RoutineSessionResponse],
)
def list_group_sessions(
    group_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all sessions within a group. Requires membership."""
    require_group_member(db, group_id, current_user.id)
    return RoutineSessionService.list_for_group(db, group_id)


@router.get(
    "/sessions/{session_id}",
    response_model=RoutineSessionResponse,
)
def get_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get a specific session. Requires access."""
    return require_session_access(db, session_id, current_user.id)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a session. Requires access."""
    rs = require_session_access(db, session_id, current_user.id)
    RoutineSessionService.delete(db, rs)
    return None
