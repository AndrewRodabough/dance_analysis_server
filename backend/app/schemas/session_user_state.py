"""Pydantic schemas for session user state."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class SessionUserStateResponse(BaseModel):
    """Response for session user state.

    ``id`` and ``created_at`` are None for sessions that have no explicit state
    row — the caller should treat those as active (not archived, not deleted).
    """

    id: Optional[UUID] = None
    session_id: UUID
    user_id: UUID
    is_archived: bool
    is_deleted: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SessionUserStateCreate(BaseModel):
    """Request to create session user state."""

    is_archived: bool = False
    is_deleted: bool = False


class SessionUserStateUpdate(BaseModel):
    """Request to update session user state."""

    is_archived: Optional[bool] = None
    is_deleted: Optional[bool] = None
