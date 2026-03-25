"""Routine management endpoints (top-level)."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.authorization import require_routine_owner
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.schemas.routine import RoutineCreate, RoutineResponse, RoutineUpdate
from app.services.routine_service import RoutinesService

router = APIRouter()


@router.post("", response_model=RoutineResponse, status_code=status.HTTP_201_CREATED)
def create_routine(
    data: RoutineCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Create a routine.

    - **title**: Routine title (1-255 characters)
    - **dance_id**: ID of the dance
    """
    return RoutinesService.create_routine(db, current_user.id, data)


@router.get("", response_model=List[RoutineResponse])
def list_routines(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all routines created by the current user."""
    return RoutinesService.list_user_routines(db, current_user.id)


@router.get("/{routine_id}", response_model=RoutineResponse)
def get_routine(
    routine_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get a specific routine. Requires ownership."""
    return require_routine_owner(db, routine_id, current_user.id)


@router.patch("/{routine_id}", response_model=RoutineResponse)
def update_routine(
    routine_id: UUID,
    data: RoutineUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update a routine. Requires ownership."""
    routine = require_routine_owner(db, routine_id, current_user.id)
    return RoutinesService.update_routine(db, routine, data)


@router.delete("/{routine_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_routine(
    routine_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a routine. Requires ownership."""
    routine = require_routine_owner(db, routine_id, current_user.id)
    RoutinesService.delete_routine(db, routine)
    return None
