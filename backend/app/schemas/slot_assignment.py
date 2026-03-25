"""Pydantic schemas for slot-assignment-related requests and responses."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class SlotAssignmentCreate(BaseModel):
    """Request body for assigning a user to a dancer slot."""

    dancer_slot_id: UUID
    user_id: UUID


class SlotAssignmentResponse(BaseModel):
    """Response body for slot assignment data."""

    id: UUID
    routine_session_id: UUID
    dancer_slot_id: UUID
    user_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
