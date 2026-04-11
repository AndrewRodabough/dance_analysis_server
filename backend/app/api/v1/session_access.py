"""Session access and group linkage management endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.authorization import require_session_access, require_session_owner
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.session_access import SessionAccessRole
from app.models.user import User
from app.schemas.session_access import (
    SessionAccessCreate,
    SessionAccessOriginResponse,
    SessionAccessResponse,
    SessionGroupLinkCreate,
    SessionGroupLinkResponse,
)
from app.services.session_access_service import SessionAccessService

router = APIRouter()


# ========================
# Session Access Management
# ========================


@router.get(
    "/sessions/{session_id}/access",
    response_model=List[SessionAccessResponse],
)
def list_session_access(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all users with access to a session.

    Requires: user is the session owner.
    """
    require_session_owner(db, session_id, current_user.id)
    return SessionAccessService.list_session_access(db, session_id)


@router.post(
    "/sessions/{session_id}/access/users",
    response_model=SessionAccessResponse,
    status_code=status.HTTP_201_CREATED,
)
def grant_direct_access(
    session_id: UUID,
    data: SessionAccessCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Grant direct access to a user for a session.

    Creates both the SessionAccess record and tracks the origin.

    Requires: user is the session owner.
    """
    require_session_owner(db, session_id, current_user.id)

    # Validate role
    try:
        role = SessionAccessRole(data.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join([r.value for r in SessionAccessRole])}",
        )

    return SessionAccessService.grant_direct_access(db, session_id, data.user_id, role)


@router.delete(
    "/sessions/{session_id}/access/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def revoke_direct_access(
    session_id: UUID,
    user_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Revoke direct access for a user on a session.

    Removes the direct access origin. If no other origins remain,
    removes the user's access entirely.

    Requires: user is the session owner.
    """
    require_session_owner(db, session_id, current_user.id)

    success = SessionAccessService.revoke_direct_access(db, session_id, user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No direct access found for this user",
        )

    return None


# ========================
# Access Origins (Audit Trail)
# ========================


@router.get(
    "/sessions/{session_id}/access/origins",
    response_model=List[SessionAccessOriginResponse],
)
def list_access_origins(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all access origins for a session.

    Shows how each user gained access (direct or via group).

    Requires: user has access to the session.
    """
    require_session_access(db, session_id, current_user.id)
    return SessionAccessService.list_access_origins(db, session_id)


@router.get(
    "/sessions/{session_id}/access/users/{user_id}/origins",
    response_model=List[SessionAccessOriginResponse],
)
def list_user_origins(
    session_id: UUID,
    user_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all origins for a specific user on a session.

    Shows all ways the user has access (direct, multiple groups, etc.).

    Requires: user has access to the session.
    """
    require_session_access(db, session_id, current_user.id)
    return SessionAccessService.list_user_origins(db, session_id, user_id)


# ========================
# Group Linkage Management
# ========================


@router.get(
    "/sessions/{session_id}/groups",
    response_model=List[SessionGroupLinkResponse],
)
def list_session_groups(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all groups linked to a session.

    Shows which groups have access to this session.

    Requires: user has access to the session.
    """
    require_session_access(db, session_id, current_user.id)
    return SessionAccessService.list_session_groups(db, session_id)


@router.post(
    "/sessions/{session_id}/groups",
    response_model=SessionGroupLinkResponse,
    status_code=status.HTTP_201_CREATED,
)
def link_group_to_session(
    session_id: UUID,
    data: SessionGroupLinkCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Link a group to a session.

    All active group members will gain access to the session at the specified role.
    This is atomic: group link and access records are created together.

    Requires: user is the session owner.
    """
    require_session_owner(db, session_id, current_user.id)

    # Validate role
    try:
        role = SessionAccessRole(data.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join([r.value for r in SessionAccessRole])}",
        )

    return SessionAccessService.link_group_to_session(
        db, session_id, data.group_id, role
    )


@router.delete(
    "/sessions/{session_id}/groups/{group_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def unlink_group_from_session(
    session_id: UUID,
    group_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Unlink a group from a session.

    All group-derived access will be removed for group members.
    If a user has no other access origins, their access is revoked entirely.

    Requires: user is the session owner.
    """
    require_session_owner(db, session_id, current_user.id)

    success = SessionAccessService.unlink_group_from_session(db, session_id, group_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No link found between this group and session",
        )

    return None


@router.delete(
    "/sessions/{session_id}/access/groups/{group_id}/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def revoke_group_member_access(
    session_id: UUID,
    group_id: UUID,
    user_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Revoke group-derived access for a specific user.

    Removes the group access origin for this user. If no other origins remain,
    removes the user's access entirely.

    Requires: user is the session owner.
    """
    require_session_owner(db, session_id, current_user.id)

    success = SessionAccessService.revoke_group_member_access(
        db, session_id, group_id, user_id
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No group-derived access found for this user",
        )

    return None
