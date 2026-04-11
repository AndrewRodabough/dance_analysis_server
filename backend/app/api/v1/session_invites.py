"""Session invite management endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.authorization import require_session_capability
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.schemas.session_access import SessionAccessResponse
from app.schemas.session_invite import (
    AcceptSessionInviteRequest,
    SessionInviteCreate,
    SessionInvitePendingResponse,
    SessionInviteResponse,
    SessionInviteLookupResponse,
)
from app.services.session_invite_service import SessionInvitesService

router = APIRouter()


@router.get(
    "/session-invites/lookup",
    response_model=SessionInviteLookupResponse,
)
def lookup_invite(token: str, db: Session = Depends(get_db)):
    """
    Look up public invite details by token. No authentication required.

    Used by the invite landing page to display session info
    before the user logs in or registers.

    Returns 404 for invalid, expired, or revoked tokens.
    """
    return SessionInvitesService.lookup_invite(db, token)


@router.get(
    "/session-invites/pending",
    response_model=list[SessionInvitePendingResponse],
)
def list_pending_invites(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    List all pending session invites for the authenticated user.

    Matches on the user's email address. Returns invites with status=pending
    that have not yet expired.
    """
    return SessionInvitesService.list_pending_for_user(db, current_user.email)


@router.post(
    "/sessions/{session_id}/invites",
    response_model=SessionInviteResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_invite(
    session_id: UUID,
    data: SessionInviteCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Create a session invite for an email address.

    - **email**: Email to invite
    - **role**: Role for the invitee (viewer, editor, or admin)

    Requires session access capability. Does NOT create access immediately.
    """
    require_session_capability(db, session_id, current_user.id, "session:invite:create")
    return SessionInvitesService.create_invite(db, session_id, current_user, data)


@router.post(
    "/session-invites/accept",
    response_model=SessionAccessResponse,
    status_code=status.HTTP_200_OK,
)
def accept_invite(
    data: AcceptSessionInviteRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Accept a session invite by token.

    - **token**: The invite token received

    The accepting user's email must match the invite email (strict).
    Returns 404 for any failure to avoid information leakage.
    Creates a SessionAccess record with direct access.
    """
    return SessionInvitesService.accept_invite(db, data.token, current_user)
