"""Pydantic schemas for session access-related requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SessionAccessResponse(BaseModel):
    """Response for session access record."""

    id: UUID
    session_id: UUID
    user_id: UUID
    role: str  # SessionAccessRole enum as string
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SessionAccessCreate(BaseModel):
    """Request to create/grant session access."""

    user_id: UUID
    role: str = Field(default="admin", description="Access role: admin, editor, viewer")


class SessionAccessUpdate(BaseModel):
    """Request to update session access."""

    role: Optional[str] = Field(
        default=None, description="Access role: admin, editor, viewer"
    )


class SessionAccessOriginResponse(BaseModel):
    """Response for session access origin."""

    id: UUID
    session_id: UUID
    user_id: UUID
    source_type: str  # 'direct' or 'group'
    group_id: Optional[UUID] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SessionGroupLinkResponse(BaseModel):
    """Response for session-group link."""

    id: UUID
    session_id: UUID
    group_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SessionGroupLinkCreate(BaseModel):
    """Request to link a group to a session."""

    group_id: UUID
    role: str = Field(
        default="admin",
        description="Default role for group members: admin, editor, viewer",
    )


class SessionAccessWithOriginsResponse(BaseModel):
    """Response combining access and its origins."""

    access: SessionAccessResponse
    origins: list[SessionAccessOriginResponse]

    model_config = ConfigDict(from_attributes=True)
