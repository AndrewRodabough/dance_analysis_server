"""Group management endpoints."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.authorization import require_group_member
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.schemas.group import (
    AddMemberRequest,
    GroupCreate,
    GroupMembershipResponse,
    GroupResponse,
)
from app.services.group_service import GroupsService

router = APIRouter()


@router.post("", response_model=GroupResponse, status_code=status.HTTP_201_CREATED)
def create_group(
    data: GroupCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Create a new group.

    - **name**: Group name (1-255 characters)
    - **description**: Optional description

    The creator is automatically added as an owner member.
    """
    return GroupsService.create_group(db, current_user, data)


@router.get("", response_model=List[GroupResponse])
def list_groups(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get all groups the current user belongs to."""
    return GroupsService.list_user_groups(db, current_user.id)


@router.get("/{group_id}", response_model=GroupResponse)
def get_group(
    group_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get group details. Requires membership."""
    require_group_member(db, group_id, current_user.id)
    group = GroupsService.get_group(db, group_id)
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return group


@router.get("/{group_id}/members", response_model=List[GroupMembershipResponse])
def list_members(
    group_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all active members of a group. Requires membership."""
    require_group_member(db, group_id, current_user.id)
    return GroupsService.list_members(db, group_id)


@router.post(
    "/{group_id}/members",
    response_model=GroupMembershipResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_member(
    group_id: int,
    data: AddMemberRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Add a member to a group.

    - **user_id**: ID of user to add
    - **role**: Role to assign (default: member)

    Requires membership (v0.0.1: any active member can add).
    """
    require_group_member(db, group_id, current_user.id)
    return GroupsService.add_member(db, group_id, data)


@router.delete("/{group_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_member(
    group_id: int,
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Remove a member from a group. Requires membership."""
    require_group_member(db, group_id, current_user.id)
    removed = GroupsService.remove_member(db, group_id, user_id)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return None
