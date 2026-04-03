"""Pydantic schemas for dance-related requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.dance import DanceStyle


class DanceBase(BaseModel):
    """Shared fields for a dance definition."""
    tempo: int = Field(..., gt=0, description="Tempo in BPM")
    meter: str = Field(..., min_length=1, max_length=10, description="Time signature, e.g. 3/4")
    style: DanceStyle


class DanceCreate(DanceBase):
    """Request body for creating a dance."""
    pass


class DanceResponse(DanceBase):
    """Response body for dance data."""
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
