"""Pydantic schemas for group and membership requests and responses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.group import GroupRole, MembershipStatus


class GroupCreate(BaseModel):
    """Request body for creating a group."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class GroupResponse(BaseModel):
    """Response body for group data."""
    id: int
    name: str
    description: Optional[str] = None
    created_by: int
    is_archived: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class GroupMembershipResponse(BaseModel):
    """Response body for a group membership."""
    group_id: int
    user_id: int
    role: GroupRole
    status: MembershipStatus
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AddMemberRequest(BaseModel):
    """Request body for adding a member to a group."""
    user_id: int
    role: GroupRole = GroupRole.MEMBER
