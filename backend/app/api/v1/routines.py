"""Routine management endpoints (group-scoped)."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.authorization import require_group_member, require_routine_in_group
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.schemas.routine import RoutineCreate, RoutineResponse, RoutineUpdate
from app.services.routine_service import RoutinesService

router = APIRouter()


@router.post("", response_model=RoutineResponse, status_code=status.HTTP_201_CREATED)
def create_routine(
    group_id: int,
    data: RoutineCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Create a routine within a group.

    - **title**: Routine title (1-255 characters)
    - **dance_id**: ID of the dance
    """
    require_group_member(db, group_id, current_user.id)
    return RoutinesService.create_routine(db, group_id, current_user.id, data)


@router.get("", response_model=List[RoutineResponse])
def list_routines(
    group_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all routines in a group. Requires membership."""
    require_group_member(db, group_id, current_user.id)
    return RoutinesService.list_routines(db, group_id)


@router.get("/{routine_id}", response_model=RoutineResponse)
def get_routine(
    group_id: int,
    routine_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get a specific routine. Requires membership."""
    require_group_member(db, group_id, current_user.id)
    routine = require_routine_in_group(db, group_id, routine_id)
    return routine


@router.patch("/{routine_id}", response_model=RoutineResponse)
def update_routine(
    group_id: int,
    routine_id: int,
    data: RoutineUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update a routine. Requires membership."""
    require_group_member(db, group_id, current_user.id)
    routine = require_routine_in_group(db, group_id, routine_id)
    return RoutinesService.update_routine(db, routine, data)


@router.delete("/{routine_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_routine(
    group_id: int,
    routine_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a routine. Requires membership."""
    require_group_member(db, group_id, current_user.id)
    routine = require_routine_in_group(db, group_id, routine_id)
    RoutinesService.delete_routine(db, routine)
    return None
