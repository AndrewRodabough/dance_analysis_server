"""Dance definition endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.schemas.dance import DanceCreate, DanceResponse
from app.services.dance_service import DanceService

router = APIRouter()


@router.get("", response_model=List[DanceResponse])
def list_dances(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all available dance definitions."""
    return DanceService.list_all(db)


@router.post("", response_model=DanceResponse, status_code=status.HTTP_201_CREATED)
def create_dance(
    data: DanceCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Create a dance definition.

    - **style**: Dance style (e.g. waltz, tango)
    - **tempo**: Tempo in BPM
    - **meter**: Time signature (e.g. 3/4, 4/4)
    """
    return DanceService.create(db, data)


@router.get("/{dance_id}", response_model=DanceResponse)
def get_dance(
    dance_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get a dance definition by ID."""
    dance = DanceService.get_by_id(db, dance_id)
    if not dance:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return dance
