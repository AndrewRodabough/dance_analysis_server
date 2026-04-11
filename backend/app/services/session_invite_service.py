"""Service for managing session invites (email-first, pre-account)."""

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
from app.models.routine import Routine
from app.models.routine_session import RoutineSession
from app.models.session_access import SessionAccess, SessionAccessRole
from app.models.session_invite import SessionInvite, SessionInviteStatus
from app.models.user import User
from app.schemas.session_invite import (
    AcceptSessionInviteRequest,
    SessionInviteCreate,
    SessionInviteLookupResponse,
    SessionInvitePendingResponse,
)

logger = get_logger(__name__)

INVITE_TOKEN_BYTES = 32
DEFAULT_INVITE_TTL_DAYS = 7


class SessionInvitesService:
    """Service for managing session invites."""

    @staticmethod
    def create_invite(
        db: Session, session_id: UUID, actor_user: User, data: SessionInviteCreate
    ) -> SessionInvite:
        """Create a new session invite.

        Does NOT create access immediately — that happens on accept.
        """
        normalized_email = data.email.lower()

        # Check for existing pending invite for this email + session
        existing = (
            db.query(SessionInvite)
            .filter(
                SessionInvite.session_id == session_id,
                SessionInvite.email == normalized_email,
                SessionInvite.status == SessionInviteStatus.PENDING,
                SessionInvite.expires_at > datetime.now(timezone.utc),
            )
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An invite is already pending for this email.",
            )

        # Validate role
        valid_roles = {role.value for role in SessionAccessRole}
        if data.role not in valid_roles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}",
            )

        invite = SessionInvite(
            session_id=session_id,
            created_by=actor_user.id,
            email=normalized_email,
            role=data.role,
            token=secrets.token_urlsafe(INVITE_TOKEN_BYTES),
            status=SessionInviteStatus.PENDING,
            expires_at=datetime.now(timezone.utc)
            + timedelta(days=DEFAULT_INVITE_TTL_DAYS),
        )
        db.add(invite)
        db.commit()
        db.refresh(invite)

        SessionInvitesService._send_invite_email(normalized_email, invite.token)
        return invite

    @staticmethod
    def lookup_invite(db: Session, token: str) -> SessionInviteLookupResponse:
        """Return public invite details for a token. No auth required.

        Returns 404 for invalid, expired, or revoked tokens.
        """
        invite = db.query(SessionInvite).filter(SessionInvite.token == token).first()

        if not invite or invite.status != SessionInviteStatus.PENDING:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

        if invite.expires_at < datetime.now(timezone.utc):
            invite.status = SessionInviteStatus.EXPIRED
            db.commit()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

        routine_name = invite.routine_session.routine.title

        return SessionInviteLookupResponse(
            session_id=invite.session_id,
            routine_name=routine_name,
            role=invite.role,
            expires_at=invite.expires_at,
        )

    @staticmethod
    def accept_invite(db: Session, token: str, accepting_user: User) -> SessionAccess:
        """Accept a session invite by token. Strict email match required.

        Returns 404 for any failure to avoid leaking information.
        Creates or updates SessionAccess record for direct access.
        """
        generic_404 = HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

        invite = db.query(SessionInvite).filter(SessionInvite.token == token).first()
        if not invite:
            raise generic_404

        # Validate status
        if invite.status != SessionInviteStatus.PENDING:
            raise generic_404

        # Check expiration
        if invite.expires_at < datetime.now(timezone.utc):
            invite.status = SessionInviteStatus.EXPIRED
            db.commit()
            raise generic_404

        # Strict email match (case-insensitive)
        if accepting_user.email.lower() != invite.email.lower():
            raise generic_404

        # Mark invite as accepted
        invite.status = SessionInviteStatus.ACCEPTED
        invite.accepted_at = datetime.now(timezone.utc)
        invite.accepted_by_user_id = accepting_user.id

        # Create or update SessionAccess for direct access
        session_access = (
            db.query(SessionAccess)
            .filter(
                SessionAccess.session_id == invite.session_id,
                SessionAccess.user_id == accepting_user.id,
            )
            .first()
        )
        if session_access:
            # Update existing access with the new role
            session_access.role = invite.role
        else:
            # Create new access record
            session_access = SessionAccess(
                session_id=invite.session_id,
                user_id=accepting_user.id,
                role=invite.role,
            )
            db.add(session_access)

        db.commit()
        db.refresh(session_access)
        return session_access

    @staticmethod
    def list_pending_for_user(db: Session, user_email: str) -> list[SessionInvitePendingResponse]:
        """Return all pending session invites addressed to the given email."""
        now = datetime.now(timezone.utc)
        invites = (
            db.query(SessionInvite)
            .filter(
                SessionInvite.email == user_email.lower(),
                SessionInvite.status == SessionInviteStatus.PENDING,
                SessionInvite.expires_at > now,
            )
            .all()
        )

        results = []
        for invite in invites:
            routine_name = invite.routine_session.routine.title
            results.append(
                SessionInvitePendingResponse(
                    token=invite.token,
                    session_id=invite.session_id,
                    routine_name=routine_name,
                    role=invite.role,
                    expires_at=invite.expires_at,
                    created_at=invite.created_at,
                )
            )
        return results

    @staticmethod
    def revoke_invite(db: Session, invite_id: UUID, session_id: UUID) -> bool:
        """Revoke a pending invite. Returns True if revoked."""
        invite = (
            db.query(SessionInvite)
            .filter(
                SessionInvite.id == invite_id,
                SessionInvite.session_id == session_id,
                SessionInvite.status == SessionInviteStatus.PENDING,
            )
            .first()
        )
        if not invite:
            return False
        invite.status = SessionInviteStatus.REVOKED
        db.commit()
        return True

    @staticmethod
    def _send_invite_email(email: str, token: str) -> None:
        """Send a session invite email via AWS SES."""
        invite_link = f"{settings.FRONTEND_URL}/session-invite?token={token}"

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
                    "Subject": {"Data": "You've been invited to a Dance Coach session"},
                    "Body": {
                        "Text": {
                            "Data": (
                                f"You've been invited to join a session on Dance Coach.\n\n"
                                f"Accept your invite here:\n{invite_link}\n\n"
                                f"This link expires in 7 days."
                            )
                        }
                    },
                },
            )
            logger.info(
                "Session invite email sent",
                extra={"event_type": "session_invite_email_sent", "email": email},
            )
        except (ClientError, BotoCoreError):
            logger.exception(
                "Failed to send session invite email via SES",
                extra={"event_type": "session_invite_email_failed", "email": email},
            )
