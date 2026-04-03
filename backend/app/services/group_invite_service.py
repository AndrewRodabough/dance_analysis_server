"""Service for managing group invites (email-first, pre-account)."""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.group import Group
from app.models.group import (
    GroupInvite,
    GroupInviteStatus,
    GroupMembership,
    GroupRole,
    MembershipStatus,
)
from app.models.user import User
from app.schemas.group_invite import GroupInviteCreate, InviteLookupResponse

logger = get_logger(__name__)

INVITE_TOKEN_BYTES = 32
DEFAULT_INVITE_TTL_DAYS = 7


class GroupInvitesService:
    """Service for managing group invites."""

    @staticmethod
    def create_invite(
        db: Session, group_id: UUID, actor_user: User, data: GroupInviteCreate
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
            expires_at=datetime.now(timezone.utc)
            + timedelta(days=DEFAULT_INVITE_TTL_DAYS),
        )
        db.add(invite)
        db.commit()
        db.refresh(invite)

        GroupInvitesService._send_invite_email(normalized_email, invite.token)
        return invite

    @staticmethod
    def lookup_invite(db: Session, token: str) -> InviteLookupResponse:
        """Return public invite details for a token. No auth required.

        Returns 404 for invalid, expired, or revoked tokens.
        """
        invite = db.query(GroupInvite).filter(GroupInvite.token == token).first()

        if not invite or invite.status != GroupInviteStatus.PENDING:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

        if invite.expires_at < datetime.now(timezone.utc):
            invite.status = GroupInviteStatus.EXPIRED
            db.commit()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

        group = db.query(Group).filter(Group.id == invite.group_id).first()
        inviter = db.query(User).filter(User.id == invite.created_by).first()

        return InviteLookupResponse(
            group_name=group.name,
            invited_by=inviter.username,
            role=invite.role,
            expires_at=invite.expires_at,
        )

    @staticmethod
    def accept_invite(db: Session, token: str, accepting_user: User) -> GroupMembership:
        """Accept a group invite by token. Strict email match required.

        Returns 404 for any failure to avoid leaking information.
        """
        generic_404 = HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

        invite = db.query(GroupInvite).filter(GroupInvite.token == token).first()
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
            membership.isAdmin = False
        else:
            membership = GroupMembership(
                group_id=invite.group_id,
                user_id=accepting_user.id,
                role=invite.role or GroupRole.MEMBER,
                isAdmin=False,
                status=MembershipStatus.ACTIVE,
            )
            db.add(membership)

        db.commit()
        db.refresh(membership)
        return membership

    @staticmethod
    def revoke_invite(db: Session, invite_id: UUID, group_id: UUID) -> bool:
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
    def _send_invite_email(email: str, token: str) -> None:
        """Send a group invite email via AWS SES."""
        invite_link = f"{settings.FRONTEND_URL}/invite?token={token}"

        ses = boto3.client(
            "ses",
            region_name=settings.AWS_SES_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

        try:
            ses.send_email(
                Source=f"invite@{settings.EMAIL_DOMAIN}",
                Destination={"ToAddresses": [email]},
                Message={
                    "Subject": {"Data": "You've been invited to join a group"},
                    "Body": {
                        "Text": {
                            "Data": (
                                f"You've been invited to join a group on Dance Coach.\n\n"
                                f"Accept your invite here:\n{invite_link}\n\n"
                                f"This link expires in 7 days."
                            )
                        }
                    },
                },
            )
            logger.info(
                "Invite email sent",
                extra={"event_type": "group_invite_email_sent", "email": email},
            )
        except (ClientError, BotoCoreError):
            logger.exception(
                "Failed to send invite email via SES",
                extra={"event_type": "group_invite_email_failed", "email": email},
            )
