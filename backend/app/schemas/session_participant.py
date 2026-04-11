"""Pydantic schemas for session participant requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SessionParticipantResponse(BaseModel):
    """Response for session participant record."""

    id: UUID
    session_id: UUID
    user_id: UUID
    role: str  # ParticipantRole enum value: dancer, coach
    dancer_slot_id: Optional[UUID] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SessionParticipantCreate(BaseModel):
    """Request to create a session participant."""

    user_id: UUID
    role: str = Field(
        default="dancer",
        description="Participant role: dancer or coach",
    )
    dancer_slot_id: Optional[UUID] = Field(
        default=None,
        description="Optional dancer slot to bind participant to",
    )


class SessionParticipantUpdate(BaseModel):
    """Request to update a session participant."""

    role: Optional[str] = Field(
        default=None,
        description="Participant role: dancer or coach",
    )
    dancer_slot_id: Optional[UUID] = Field(
        default=None,
        description="Dancer slot to bind participant to, or null to unbind",
    )


class SessionParticipantWithUserResponse(SessionParticipantResponse):
    """Response including participant user details."""

    user_email: Optional[str] = None
    user_name: Optional[str] = None


class ParticipantListResponse(BaseModel):
    """Response for list of participants with slot information."""

    participants: list[SessionParticipantResponse]
    available_users: list[UUID] = Field(
        default_factory=list,
        description="User IDs with session access but no participant record",
    )
    total_count: int

    model_config = ConfigDict(from_attributes=True)
