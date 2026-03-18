"""Endpoint to call to send an invitation to another user."""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_active_user
from app.core.logging import get_logger
from app.database import get_db
from app.models.user import Invite, User, UserRelationRole, UserRelationStatus
from app.schemas.invite import InviteCreate, InviteResponse

router = APIRouter(prefix="/invites", tags=["invites"])

logger = get_logger(__name__)

INVITE_TOKEN_BYTES = 32
DEFAULT_INVITE_TTL_DAYS = 7


def _generate_invite_token() -> str:
    """Return a cryptographically secure invite token."""
    return secrets.token_urlsafe(INVITE_TOKEN_BYTES)


def _calculate_expiration(now: datetime) -> datetime:
    """Compute the default expiration timestamp for an invite."""
    return now + timedelta(days=DEFAULT_INVITE_TTL_DAYS)


def _send_invitation_email_stub(email: str, token: str, role: UserRelationRole) -> None:
    """
    Placeholder for outbound invite emails.

    Replace this with a real SMTP integration once credentials are available.
    """
    logger.info(
        "Stubbed invite email dispatch",
        extra={
            "event_type": "invite_email_stub",
            "email": email,
            "token": token,
            "role": role.value,
        },
    )


def _ensure_unique_contact_invite(
    db: Session,
    *,
    creator_id: int,
    email: Optional[str],
    phone_number: Optional[str],
    role: UserRelationRole,
) -> None:
    """Prevent duplicate pending invites for the same creator/contact/role tuple."""

    if not email and not phone_number:
        raise ValueError("Invite must specify either email or phone number.")

    now = datetime.now(timezone.utc)

    query = db.query(Invite).filter(
        Invite.created_by == creator_id,
        Invite.role == role,
        Invite.status == UserRelationStatus.PENDING,
        Invite.expires_at > now,
    )

    if email:
        query = query.filter(Invite.email == email.lower())
    else:
        query = query.filter(Invite.phone_number == phone_number)

    existing_invite = query.first()

    if existing_invite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An invite is already pending for this contact and role.",
        )

@router.post(
    "",
    response_model=InviteResponse,
    status_code=status.HTTP_200_OK,
    summary="Create a relation invite",
)
def create_invite(
    payload: InviteCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> InviteResponse:
    """
    Create an invite to establish a new relation.

    The caller must be authenticated. Successful requests return HTTP 200 and
    trigger a stubbed email notification (until SMTP is configured).
    """
    _ensure_unique_contact_invite(
        db,
        creator_id=current_user.id,
        email=payload.email,
        phone_number=payload.phone_number,
        role=payload.role,
    )

    target_user_exists = False
    if payload.email:
        target_user_exists = (
            db.query(User).filter(User.email == payload.email).first() is not None
        )

    now = datetime.now(timezone.utc)
    invite = Invite(
        token=_generate_invite_token(),
        created_by=current_user.id,
        email=payload.email,
        phone_number=payload.phone_number,
        role=payload.role,
        status=UserRelationStatus.PENDING,
        expires_at=_calculate_expiration(now),
    )

    db.add(invite)
    db.commit()
    db.refresh(invite)

    if invite.email:
        _send_invitation_email_stub(invite.email, invite.token, invite.role)

    return InviteResponse(
        id=invite.id,
        created_by=invite.created_by,
        email=invite.email,
        phone_number=invite.phone_number,
        role=invite.role,
        status=invite.status,
        expires_at=invite.expires_at,
        created_at=invite.created_at,
        target_user_exists=target_user_exists,
    )
