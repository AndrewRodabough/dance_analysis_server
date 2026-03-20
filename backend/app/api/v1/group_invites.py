"""Group invite management endpoints."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.authorization import require_group_capability
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.schemas.group import GroupMembershipResponse
from app.schemas.group_invite import (
    AcceptInviteRequest,
    GroupInviteCreate,
    GroupInviteResponse,
)
from app.services.group_invite_service import GroupInvitesService

router = APIRouter()


@router.post(
    "/groups/{group_id}/invites",
    response_model=GroupInviteResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_invite(
    group_id: int,
    data: GroupInviteCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Create a group invite for an email address.

    - **email**: Email to invite
    - **role**: Optional role for the invitee

    Requires group membership. Does NOT create a membership immediately.
    """
    require_group_capability(db, group_id, current_user.id, "group:invite:create")
    return GroupInvitesService.create_invite(db, group_id, current_user, data)


@router.post(
    "/group-invites/accept",
    response_model=GroupMembershipResponse,
    status_code=status.HTTP_200_OK,
)
def accept_invite(
    data: AcceptInviteRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Accept a group invite by token.

    - **token**: The invite token received

    The accepting user's email must match the invite email (strict).
    Returns 404 for any failure to avoid information leakage.
    """
    return GroupInvitesService.accept_invite(db, data.token, current_user)
