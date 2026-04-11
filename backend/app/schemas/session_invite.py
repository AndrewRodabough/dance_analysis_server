"""Pydantic schemas for session invite requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.session_invite import SessionInviteStatus


class SessionInviteCreate(BaseModel):
    """Request body for creating a session invite."""

    email: EmailStr = Field(..., description="Email address to invite.")
    role: str = Field(
        default="viewer",
        description="Role for the invitee (viewer, editor, or admin).",
    )


class SessionInviteResponse(BaseModel):
    """Response body for a session invite."""

    id: UUID
    session_id: UUID
    created_by: UUID
    email: str
    role: str
    token: str
    status: SessionInviteStatus
    expires_at: datetime
    created_at: datetime
    accepted_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AcceptSessionInviteRequest(BaseModel):
    """Request body for accepting a session invite by token."""

    token: str = Field(..., min_length=1)


class SessionInviteLookupResponse(BaseModel):
    """Public session invite details returned before authentication."""

    session_id: UUID
    routine_name: str
    role: str
    expires_at: datetime


class SessionInvitePendingResponse(BaseModel):
    """One pending session invite in the authenticated user's invite list."""

    token: str
    session_id: UUID
    routine_name: str
    role: str
    expires_at: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
