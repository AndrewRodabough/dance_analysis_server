"""Pydantic schemas for routine-related requests and responses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.routine import RoutineRole


class RoutineBase(BaseModel):
    """Shared fields for a routine."""
    title: str = Field(..., min_length=1, max_length=255)
    dance_id: int


class RoutineCreate(RoutineBase):
    """Request body for creating a routine."""
    pass


class RoutineResponse(RoutineBase):
    """Response body for routine data."""
    id: int
    owner_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class RoutineParticipantBase(BaseModel):
    """Shared fields for a routine participant."""
    role: RoutineRole


class RoutineParticipantCreate(RoutineParticipantBase):
    """Request body for adding a participant to a routine."""
    routine_id: int
    user_id: int


class RoutineParticipantResponse(RoutineParticipantBase):
    """Response body for routine participant data."""
    routine_id: int
    user_id: int

    model_config = ConfigDict(from_attributes=True)
