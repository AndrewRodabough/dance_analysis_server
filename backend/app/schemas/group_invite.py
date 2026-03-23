"""Pydantic schemas for group invite requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.group import GroupInviteStatus, GroupRole


class GroupInviteCreate(BaseModel):
    """Request body for creating a group invite."""

    email: EmailStr = Field(..., description="Email address to invite.")
    role: Optional[GroupRole] = Field(
        default=None,
        description="Optional role for the invitee (defaults to member).",
    )


class GroupInviteResponse(BaseModel):
    """Response body for a group invite."""

    id: UUID
    group_id: UUID
    created_by: UUID
    email: str
    role: Optional[GroupRole] = None
    token: str
    status: GroupInviteStatus
    expires_at: datetime
    created_at: datetime
    accepted_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AcceptInviteRequest(BaseModel):
    """Request body for accepting a group invite by token."""

    token: str = Field(..., min_length=1)
