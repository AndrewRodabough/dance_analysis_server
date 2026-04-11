"""Service for managing session access and group linkages."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.models.group import GroupMembership, MembershipStatus
from app.models.session_access import SessionAccess, SessionAccessRole
from app.models.session_access_origin import AccessSourceType, SessionAccessOrigin
from app.models.session_group_link import SessionGroupLink


class SessionAccessService:
    """Service for managing session access, origins, and group linkages."""

    # ==================== SessionAccess Management ====================

    @staticmethod
    def grant_direct_access(
        db: Session,
        session_id: UUID,
        user_id: UUID,
        role: SessionAccessRole = SessionAccessRole.ADMIN,
    ) -> SessionAccess:
        """
        Grant direct access to a user for a session.

        This is atomic: creates both SessionAccess and SessionAccessOrigin(source_type='direct').
        If access already exists, updates the role if different.
        """
        # Check if access already exists
        existing = (
            db.query(SessionAccess)
            .filter(
                and_(
                    SessionAccess.session_id == session_id,
                    SessionAccess.user_id == user_id,
                )
            )
            .first()
        )

        if existing:
            # Update role if different
            if existing.role != role:
                existing.role = role
                db.commit()
            db.refresh(existing)
            return existing

        # Create new access
        access = SessionAccess(
            session_id=session_id,
            user_id=user_id,
            role=role,
        )
        db.add(access)

        # Create origin record
        origin = SessionAccessOrigin(
            session_id=session_id,
            user_id=user_id,
            source_type=AccessSourceType.DIRECT,
            group_id=None,
        )
        db.add(origin)

        db.commit()
        db.refresh(access)
        return access

    @staticmethod
    def get_access(
        db: Session, session_id: UUID, user_id: UUID
    ) -> Optional[SessionAccess]:
        """Get the effective access record for a user on a session."""
        return (
            db.query(SessionAccess)
            .filter(
                and_(
                    SessionAccess.session_id == session_id,
                    SessionAccess.user_id == user_id,
                )
            )
            .first()
        )

    @staticmethod
    def list_session_access(db: Session, session_id: UUID) -> List[SessionAccess]:
        """List all users with effective access to a session."""
        return (
            db.query(SessionAccess).filter(SessionAccess.session_id == session_id).all()
        )

    @staticmethod
    def revoke_direct_access(db: Session, session_id: UUID, user_id: UUID) -> bool:
        """
        Revoke direct access for a user on a session.

        Removes the direct origin. If no other origins remain,
        deletes the SessionAccess record entirely.
        """
        # Remove direct origin
        origin = (
            db.query(SessionAccessOrigin)
            .filter(
                and_(
                    SessionAccessOrigin.session_id == session_id,
                    SessionAccessOrigin.user_id == user_id,
                    SessionAccessOrigin.source_type == AccessSourceType.DIRECT,
                )
            )
            .first()
        )

        if not origin:
            return False

        db.delete(origin)

        # Check if any other origins remain
        remaining_origins = (
            db.query(SessionAccessOrigin)
            .filter(
                and_(
                    SessionAccessOrigin.session_id == session_id,
                    SessionAccessOrigin.user_id == user_id,
                )
            )
            .first()
        )

        # If no remaining origins, delete effective access
        if not remaining_origins:
            access = (
                db.query(SessionAccess)
                .filter(
                    and_(
                        SessionAccess.session_id == session_id,
                        SessionAccess.user_id == user_id,
                    )
                )
                .first()
            )
            if access:
                db.delete(access)

        db.commit()
        return True

    # ==================== SessionAccessOrigin Management ====================

    @staticmethod
    def list_access_origins(db: Session, session_id: UUID) -> List[SessionAccessOrigin]:
        """List all access origins for a session."""
        return (
            db.query(SessionAccessOrigin)
            .filter(SessionAccessOrigin.session_id == session_id)
            .all()
        )

    @staticmethod
    def list_user_origins(
        db: Session, session_id: UUID, user_id: UUID
    ) -> List[SessionAccessOrigin]:
        """List all origins for a specific user on a session."""
        return (
            db.query(SessionAccessOrigin)
            .filter(
                and_(
                    SessionAccessOrigin.session_id == session_id,
                    SessionAccessOrigin.user_id == user_id,
                )
            )
            .all()
        )

    # ==================== Group Linkage Management ====================

    @staticmethod
    def link_group_to_session(
        db: Session,
        session_id: UUID,
        group_id: UUID,
        role: SessionAccessRole = SessionAccessRole.ADMIN,
    ) -> SessionGroupLink:
        """
        Link a group to a session, granting all active members access.

        This is atomic: creates the link and access records for all members.
        """
        # Check if link already exists
        existing_link = (
            db.query(SessionGroupLink)
            .filter(
                and_(
                    SessionGroupLink.session_id == session_id,
                    SessionGroupLink.group_id == group_id,
                )
            )
            .first()
        )

        if existing_link:
            return existing_link

        # Create the link
        link = SessionGroupLink(session_id=session_id, group_id=group_id)
        db.add(link)
        db.flush()

        # Get all active members of the group
        members = (
            db.query(GroupMembership)
            .filter(
                and_(
                    GroupMembership.group_id == group_id,
                    GroupMembership.status == MembershipStatus.ACTIVE,
                )
            )
            .all()
        )

        # Create access and origin for each member
        for member in members:
            # Check if access already exists
            existing_access = (
                db.query(SessionAccess)
                .filter(
                    and_(
                        SessionAccess.session_id == session_id,
                        SessionAccess.user_id == member.user_id,
                    )
                )
                .first()
            )

            if not existing_access:
                # Create new access
                access = SessionAccess(
                    session_id=session_id,
                    user_id=member.user_id,
                    role=role,
                )
                db.add(access)

            # Create origin record for group-based access
            origin = SessionAccessOrigin(
                session_id=session_id,
                user_id=member.user_id,
                source_type=AccessSourceType.GROUP,
                group_id=group_id,
            )
            db.add(origin)

        db.commit()
        db.refresh(link)
        return link

    @staticmethod
    def unlink_group_from_session(
        db: Session, session_id: UUID, group_id: UUID
    ) -> bool:
        """
        Unlink a group from a session, removing all group-derived access origins.

        For each user with group-derived access, if they have no remaining origins,
        their SessionAccess record is deleted.
        """
        # Find and delete the link
        link = (
            db.query(SessionGroupLink)
            .filter(
                and_(
                    SessionGroupLink.session_id == session_id,
                    SessionGroupLink.group_id == group_id,
                )
            )
            .first()
        )

        if not link:
            return False

        db.delete(link)

        # Find all origins from this group
        origins = (
            db.query(SessionAccessOrigin)
            .filter(
                and_(
                    SessionAccessOrigin.session_id == session_id,
                    SessionAccessOrigin.source_type == AccessSourceType.GROUP,
                    SessionAccessOrigin.group_id == group_id,
                )
            )
            .all()
        )

        # For each origin, remove it and check remaining origins
        for origin in origins:
            db.delete(origin)

            # Check if any other origins remain for this user
            remaining_origins = (
                db.query(SessionAccessOrigin)
                .filter(
                    and_(
                        SessionAccessOrigin.session_id == session_id,
                        SessionAccessOrigin.user_id == origin.user_id,
                    )
                )
                .first()
            )

            # If no remaining origins, delete effective access
            if not remaining_origins:
                access = (
                    db.query(SessionAccess)
                    .filter(
                        and_(
                            SessionAccess.session_id == session_id,
                            SessionAccess.user_id == origin.user_id,
                        )
                    )
                    .first()
                )
                if access:
                    db.delete(access)

        db.commit()
        return True

    @staticmethod
    def revoke_group_member_access(
        db: Session, session_id: UUID, group_id: UUID, user_id: UUID
    ) -> bool:
        """
        Revoke group-derived access for a specific user.

        Removes the group origin. If no other origins remain,
        deletes the SessionAccess record entirely.
        """
        # Remove group origin
        origin = (
            db.query(SessionAccessOrigin)
            .filter(
                and_(
                    SessionAccessOrigin.session_id == session_id,
                    SessionAccessOrigin.user_id == user_id,
                    SessionAccessOrigin.source_type == AccessSourceType.GROUP,
                    SessionAccessOrigin.group_id == group_id,
                )
            )
            .first()
        )

        if not origin:
            return False

        db.delete(origin)

        # Check if any other origins remain
        remaining_origins = (
            db.query(SessionAccessOrigin)
            .filter(
                and_(
                    SessionAccessOrigin.session_id == session_id,
                    SessionAccessOrigin.user_id == user_id,
                )
            )
            .first()
        )

        # If no remaining origins, delete effective access
        if not remaining_origins:
            access = (
                db.query(SessionAccess)
                .filter(
                    and_(
                        SessionAccess.session_id == session_id,
                        SessionAccess.user_id == user_id,
                    )
                )
                .first()
            )
            if access:
                db.delete(access)

        db.commit()
        return True

    @staticmethod
    def list_session_groups(db: Session, session_id: UUID) -> List[SessionGroupLink]:
        """List all groups linked to a session."""
        return (
            db.query(SessionGroupLink)
            .filter(SessionGroupLink.session_id == session_id)
            .all()
        )

    @staticmethod
    def get_group_link(
        db: Session, session_id: UUID, group_id: UUID
    ) -> Optional[SessionGroupLink]:
        """Get the link between a group and session if it exists."""
        return (
            db.query(SessionGroupLink)
            .filter(
                and_(
                    SessionGroupLink.session_id == session_id,
                    SessionGroupLink.group_id == group_id,
                )
            )
            .first()
        )

    # ==================== Sync & Maintenance ====================

    @staticmethod
    def sync_group_member_access(db: Session, session_id: UUID, group_id: UUID) -> None:
        """
        Sync access for a group's members after membership changes.

        Adds access origins for members who gained access,
        removes origins for members who lost access.
        """
        # Get all active members
        active_members = (
            db.query(GroupMembership)
            .filter(
                and_(
                    GroupMembership.group_id == group_id,
                    GroupMembership.status == MembershipStatus.ACTIVE,
                )
            )
            .all()
        )

        active_member_ids = {m.user_id for m in active_members}

        # Get all users with group-derived access
        origins_with_access = (
            db.query(SessionAccessOrigin)
            .filter(
                and_(
                    SessionAccessOrigin.session_id == session_id,
                    SessionAccessOrigin.source_type == AccessSourceType.GROUP,
                    SessionAccessOrigin.group_id == group_id,
                )
            )
            .all()
        )

        users_with_access = {o.user_id for o in origins_with_access}

        # Add access for new members
        for user_id in active_member_ids - users_with_access:
            access = (
                db.query(SessionAccess)
                .filter(
                    and_(
                        SessionAccess.session_id == session_id,
                        SessionAccess.user_id == user_id,
                    )
                )
                .first()
            )

            if not access:
                access = SessionAccess(
                    session_id=session_id,
                    user_id=user_id,
                    role=SessionAccessRole.ADMIN,
                )
                db.add(access)

            origin = SessionAccessOrigin(
                session_id=session_id,
                user_id=user_id,
                source_type=AccessSourceType.GROUP,
                group_id=group_id,
            )
            db.add(origin)

        # Remove access for removed members
        for user_id in users_with_access - active_member_ids:
            origin = (
                db.query(SessionAccessOrigin)
                .filter(
                    and_(
                        SessionAccessOrigin.session_id == session_id,
                        SessionAccessOrigin.user_id == user_id,
                        SessionAccessOrigin.source_type == AccessSourceType.GROUP,
                        SessionAccessOrigin.group_id == group_id,
                    )
                )
                .first()
            )

            if origin:
                db.delete(origin)

                # Check if remaining origins exist
                remaining = (
                    db.query(SessionAccessOrigin)
                    .filter(
                        and_(
                            SessionAccessOrigin.session_id == session_id,
                            SessionAccessOrigin.user_id == user_id,
                        )
                    )
                    .first()
                )

                if not remaining:
                    access = (
                        db.query(SessionAccess)
                        .filter(
                            and_(
                                SessionAccess.session_id == session_id,
                                SessionAccess.user_id == user_id,
                            )
                        )
                        .first()
                    )
                    if access:
                        db.delete(access)

        db.commit()
