"""Invite request/response schemas."""

from datetime import datetime
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    constr,
    model_validator,
)

from app.models.user import UserRelationRole, UserRelationStatus

PhoneNumberStr = constr(strip_whitespace=True, min_length=7, max_length=32)


class InviteCreate(BaseModel):
    """Payload for creating a new invite."""

    email: Optional[EmailStr] = Field(
        default=None,
        description="Email address to send the invite to.",
    )
    phone_number: Optional[PhoneNumberStr] = Field(
        default=None,
        description="Phone number to notify. Optional until SMS support is enabled.",
    )
    role: UserRelationRole = Field(
        ...,
        description="Relationship role the invitee will assume.",
    )

    @model_validator(mode="after")
    def validate_contact_channel(cls, model: "InviteCreate") -> "InviteCreate":
        """Ensure at least one contact channel is provided."""
        if not model.email and not model.phone_number:
            raise ValueError("Either email or phone_number must be provided.")
        return model


class InviteResponse(BaseModel):
    """Response schema returned after creating or retrieving an invite."""

    id: int
    created_by: int
    email: Optional[EmailStr]
    phone_number: Optional[str]
    role: UserRelationRole
    status: UserRelationStatus
    expires_at: datetime
    created_at: datetime
    target_user_exists: bool = Field(
        default=False,
        description="Indicates whether the invite target already has an account.",
    )

    model_config = ConfigDict(from_attributes=True)
