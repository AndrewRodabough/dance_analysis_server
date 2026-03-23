"""Pydantic schemas for routine-related requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RoutineBase(BaseModel):
    """Shared fields for a routine."""

    title: str = Field(..., min_length=1, max_length=255)
    dance_id: UUID


class RoutineCreate(RoutineBase):
    """Request body for creating a routine."""

    pass


class RoutineUpdate(BaseModel):
    """Request body for updating a routine."""

    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    dance_id: Optional[UUID] = None


class RoutineResponse(RoutineBase):
    """Response body for routine data."""

    id: UUID
    group_id: Optional[UUID] = None
    created_by: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
