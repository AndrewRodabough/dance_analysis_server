"""Service for managing group invites (email-first, pre-account)."""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.group import (
    GroupInvite,
    GroupInviteStatus,
    GroupMembership,
    GroupRole,
    MembershipStatus,
)
from app.models.user import User
from app.schemas.group_invite import GroupInviteCreate

logger = get_logger(__name__)

INVITE_TOKEN_BYTES = 32
DEFAULT_INVITE_TTL_DAYS = 7


class GroupInvitesService:
    """Service for managing group invites."""

    @staticmethod
    def create_invite(
        db: Session, group_id: int, actor_user: User, data: GroupInviteCreate
    ) -> GroupInvite:
        """Create a new group invite.

        Does NOT create membership — that happens on accept.
        """
        normalized_email = data.email.lower()

        # Check for existing pending invite for this email + group
        existing = (
            db.query(GroupInvite)
            .filter(
                GroupInvite.group_id == group_id,
                GroupInvite.email == normalized_email,
                GroupInvite.status == GroupInviteStatus.PENDING,
                GroupInvite.expires_at > datetime.now(timezone.utc),
            )
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An invite is already pending for this email.",
            )

        invite = GroupInvite(
            group_id=group_id,
            created_by=actor_user.id,
            email=normalized_email,
            role=data.role or GroupRole.MEMBER,
            token=secrets.token_urlsafe(INVITE_TOKEN_BYTES),
            status=GroupInviteStatus.PENDING,
            expires_at=datetime.now(timezone.utc) + timedelta(days=DEFAULT_INVITE_TTL_DAYS),
        )
        db.add(invite)
        db.commit()
        db.refresh(invite)

        # Stub notification
        GroupInvitesService._send_invite_email_stub(
            normalized_email, invite.token, group_id
        )
        return invite

    @staticmethod
    def accept_invite(db: Session, token: str, accepting_user: User) -> GroupMembership:
        """Accept a group invite by token. Strict email match required.

        Returns 404 for any failure to avoid leaking information.
        """
        generic_404 = HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

        invite = (
            db.query(GroupInvite)
            .filter(GroupInvite.token == token)
            .first()
        )
        if not invite:
            raise generic_404

        # Validate status
        if invite.status != GroupInviteStatus.PENDING:
            raise generic_404

        # Check expiration
        if invite.expires_at < datetime.now(timezone.utc):
            invite.status = GroupInviteStatus.EXPIRED
            db.commit()
            raise generic_404

        # Strict email match (case-insensitive)
        if accepting_user.email.lower() != invite.email.lower():
            raise generic_404

        # Mark invite as accepted
        invite.status = GroupInviteStatus.ACCEPTED
        invite.accepted_at = datetime.now(timezone.utc)
        invite.accepted_by_user_id = accepting_user.id

        # Create or reactivate membership
        membership = (
            db.query(GroupMembership)
            .filter(
                GroupMembership.group_id == invite.group_id,
                GroupMembership.user_id == accepting_user.id,
            )
            .first()
        )
        if membership:
            membership.status = MembershipStatus.ACTIVE
            membership.role = invite.role or GroupRole.MEMBER
        else:
            membership = GroupMembership(
                group_id=invite.group_id,
                user_id=accepting_user.id,
                role=invite.role or GroupRole.MEMBER,
                status=MembershipStatus.ACTIVE,
            )
            db.add(membership)

        db.commit()
        db.refresh(membership)
        return membership

    @staticmethod
    def revoke_invite(db: Session, invite_id: int, group_id: int) -> bool:
        """Revoke a pending invite. Returns True if revoked."""
        invite = (
            db.query(GroupInvite)
            .filter(
                GroupInvite.id == invite_id,
                GroupInvite.group_id == group_id,
                GroupInvite.status == GroupInviteStatus.PENDING,
            )
            .first()
        )
        if not invite:
            return False
        invite.status = GroupInviteStatus.REVOKED
        db.commit()
        return True

    @staticmethod
    def _send_invite_email_stub(email: str, token: str, group_id: int) -> None:
        """Placeholder for outbound invite emails."""
        logger.info(
            "Stubbed group invite email dispatch",
            extra={
                "event_type": "group_invite_email_stub",
                "email": email,
                "token": token,
                "group_id": group_id,
            },
        )
