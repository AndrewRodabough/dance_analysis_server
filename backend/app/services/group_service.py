"""Service for managing group operations."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.group import Group, GroupMembership, GroupRole, MembershipStatus
from app.models.user import User
from app.schemas.group import AddAdminRequest, AddMemberRequest, GroupCreate


class GroupsService:
    """Service for managing group operations."""

    @staticmethod
    def create_group(db: Session, user: User, data: GroupCreate) -> Group:
        """Create a new group and add the creator as admin."""
        group = Group(
            name=data.name,
            description=data.description,
            created_by=user.id,
        )
        db.add(group)
        db.flush()

        membership = GroupMembership(
            group_id=group.id,
            user_id=user.id,
            role=GroupRole.MEMBER,
            isAdmin=True,
            status=MembershipStatus.ACTIVE,
        )
        db.add(membership)
        db.commit()
        db.refresh(group)
        return group

    @staticmethod
    def list_user_groups(db: Session, user_id: UUID) -> List[Group]:
        """List all groups the user is an active member of."""
        return (
            db.query(Group)
            .join(GroupMembership, GroupMembership.group_id == Group.id)
            .filter(
                GroupMembership.user_id == user_id,
                GroupMembership.status == MembershipStatus.ACTIVE,
            )
            .order_by(Group.created_at.desc())
            .all()
        )

    @staticmethod
    def get_group(db: Session, group_id: UUID) -> Optional[Group]:
        """Get a group by ID."""
        return db.query(Group).filter(Group.id == group_id).first()

    @staticmethod
    def list_members(db: Session, group_id: UUID) -> List[GroupMembership]:
        """List all active memberships for a group."""
        return (
            db.query(GroupMembership)
            .filter(
                GroupMembership.group_id == group_id,
                GroupMembership.status == MembershipStatus.ACTIVE,
            )
            .all()
        )

    @staticmethod
    def add_member(
        db: Session, group_id: UUID, data: AddMemberRequest
    ) -> GroupMembership:
        """Add a member to a group."""
        existing = (
            db.query(GroupMembership)
            .filter(
                GroupMembership.group_id == group_id,
                GroupMembership.user_id == data.user_id,
            )
            .first()
        )
        if existing:
            existing.status = MembershipStatus.ACTIVE
            existing.role = data.role
            existing.isAdmin = getattr(data, "isAdmin", False)
            db.commit()
            db.refresh(existing)
            return existing

        membership = GroupMembership(
            group_id=group_id,
            user_id=data.user_id,
            role=data.role,
            isAdmin=getattr(data, "isAdmin", False),
            status=MembershipStatus.ACTIVE,
        )
        db.add(membership)
        db.commit()
        db.refresh(membership)
        return membership

    @staticmethod
    def remove_member(db: Session, group_id: UUID, target_user_id: UUID) -> bool:
        """Remove a member from a group. Returns True if removed."""
        membership = (
            db.query(GroupMembership)
            .filter(
                GroupMembership.group_id == group_id,
                GroupMembership.user_id == target_user_id,
                GroupMembership.status == MembershipStatus.ACTIVE,
            )
            .first()
        )
        if not membership:
            return False
        db.delete(membership)
        db.commit()
        return True
