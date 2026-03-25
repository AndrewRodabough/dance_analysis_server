"""Slot assignment management endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.authorization import require_session_access
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.schemas.slot_assignment import SlotAssignmentCreate, SlotAssignmentResponse
from app.services.slot_assignment_service import SlotAssignmentService

router = APIRouter()


@router.post(
    "/sessions/{session_id}/assignments",
    response_model=SlotAssignmentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_assignment(
    session_id: UUID,
    data: SlotAssignmentCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Assign a user to a dancer slot for a session.

    - **dancer_slot_id**: The slot to assign
    - **user_id**: The user to assign to the slot
    """
    require_session_access(db, session_id, current_user.id)
    return SlotAssignmentService.create(db, session_id, data)


@router.get(
    "/sessions/{session_id}/assignments",
    response_model=List[SlotAssignmentResponse],
)
def list_assignments(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all slot assignments for a session."""
    require_session_access(db, session_id, current_user.id)
    return SlotAssignmentService.list_for_session(db, session_id)


@router.delete(
    "/sessions/{session_id}/assignments/{assignment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_assignment(
    session_id: UUID,
    assignment_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Remove a slot assignment. Requires session access."""
    require_session_access(db, session_id, current_user.id)
    assignment = SlotAssignmentService.get_by_id(db, assignment_id)
    if not assignment or assignment.routine_session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
        )
    SlotAssignmentService.delete(db, assignment)
    return None
