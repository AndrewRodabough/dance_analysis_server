"""Pydantic schemas for routine-related requests and responses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RoutineBase(BaseModel):
    """Shared fields for a routine."""
    title: str = Field(..., min_length=1, max_length=255)
    dance_id: int


class RoutineCreate(RoutineBase):
    """Request body for creating a routine."""
    pass


class RoutineUpdate(BaseModel):
    """Request body for updating a routine."""
    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    dance_id: Optional[int] = None


class RoutineResponse(RoutineBase):
    """Response body for routine data."""
    id: int
    group_id: Optional[int] = None
    created_by: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
