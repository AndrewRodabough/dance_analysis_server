"""Pydantic schemas for routine-session-related requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RoutineSessionCreate(BaseModel):
    """Request body for creating a routine session."""

    label: Optional[str] = Field(default=None, max_length=255)


class RoutineSessionResponse(BaseModel):
    """Response body for routine session data."""

    id: UUID
    routine_id: UUID
    owner_id: UUID
    created_by: UUID
    label: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class RoutineSessionDetailResponse(RoutineSessionResponse):
    """Detailed response including related data."""

    pass
