"""Pydantic schemas for routine-session-related requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RoutineSessionCreate(BaseModel):
    """Request body for creating a routine session."""

    group_id: Optional[UUID] = None
    label: Optional[str] = Field(default=None, max_length=255)


class RoutineSessionResponse(BaseModel):
    """Response body for routine session data."""

    id: UUID
    routine_id: UUID
    group_id: Optional[UUID] = None
    created_by: UUID
    label: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
