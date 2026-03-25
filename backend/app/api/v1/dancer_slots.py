"""Dancer slot management endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.authorization import require_routine_owner
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.schemas.routine_dancer_slot import (
    RoutineDancerSlotCreate,
    RoutineDancerSlotResponse,
)
from app.services.dancer_slot_service import DancerSlotService

router = APIRouter()


@router.post(
    "/routines/{routine_id}/slots",
    response_model=RoutineDancerSlotResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_slot(
    routine_id: UUID,
    data: RoutineDancerSlotCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Create a dancer slot for a routine.

    - **label**: Slot label (e.g. "A", "B", "Lead")
    - **order_index**: Optional ordering hint
    """
    require_routine_owner(db, routine_id, current_user.id)
    return DancerSlotService.create(db, routine_id, data)


@router.get(
    "/routines/{routine_id}/slots",
    response_model=List[RoutineDancerSlotResponse],
)
def list_slots(
    routine_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all dancer slots for a routine."""
    require_routine_owner(db, routine_id, current_user.id)
    return DancerSlotService.list_for_routine(db, routine_id)


@router.delete(
    "/routines/{routine_id}/slots/{slot_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_slot(
    routine_id: UUID,
    slot_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a dancer slot. Requires routine ownership."""
    require_routine_owner(db, routine_id, current_user.id)
    slot = DancerSlotService.get_by_id(db, slot_id)
    if not slot or slot.routine_id != routine_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
        )
    DancerSlotService.delete(db, slot)
    return None
