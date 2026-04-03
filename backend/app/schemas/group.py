"""Pydantic schemas for group and membership requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.group import GroupRole, MembershipStatus


class GroupCreate(BaseModel):
    """Request body for creating a group."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class GroupResponse(BaseModel):
    """Response body for group data."""

    id: UUID
    name: str
    description: Optional[str] = None
    created_by: UUID
    is_archived: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class GroupMembershipResponse(BaseModel):
    """Response body for a group membership."""

    group_id: UUID
    user_id: UUID
    username: str
    role: GroupRole
    isAdmin: bool
    status: MembershipStatus
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AddMemberRequest(BaseModel):
    """Request body for adding a member to a group."""

    user_id: UUID
    role: GroupRole = GroupRole.MEMBER
    isAdmin: bool = False


class AddAdminRequest(AddMemberRequest):
    """Request body for promoting a member to an admin"""

    isAdmin: bool = True
