"""Pydantic schemas for routine-dancer-slot-related requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RoutineDancerSlotCreate(BaseModel):
    """Request body for creating a dancer slot."""

    label: str = Field(..., min_length=1, max_length=50)
    order_index: Optional[int] = None


class RoutineDancerSlotResponse(BaseModel):
    """Response body for dancer slot data."""

    id: UUID
    routine_id: UUID
    label: str
    order_index: Optional[int] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
