"""Centralized authorization helpers for access control.

All non-leaky: return 404 for unauthorized access to avoid leaking existence.
"""

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.group import GroupMembership, MembershipStatus
from app.models.job import Job
from app.models.routine import Routine
from app.models.routine_session import RoutineSession
from app.models.video import Video, VideoStatus


def require_group_member(db: Session, group_id: UUID, user_id: UUID) -> GroupMembership:
    """Require that the user is an active member of the group.

    Raises 404 if not a member or group doesn't exist (non-leaky).
    """
    membership = (
        db.query(GroupMembership)
        .filter(
            GroupMembership.group_id == group_id,
            GroupMembership.user_id == user_id,
            GroupMembership.status == MembershipStatus.ACTIVE,
        )
        .first()
    )
    if not membership:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )
    return membership


def require_group_capability(
    db: Session, group_id: UUID, user_id: UUID, capability: str
) -> GroupMembership:
    """Require that the user has a specific capability within the group.

    v0.0.1: delegates to membership check (any active member has all capabilities).
    Future: role/capability mapping can be inserted here.
    """
    return require_group_member(db, group_id, user_id)


def require_routine_owner(db: Session, routine_id: UUID, user_id: UUID) -> Routine:
    """Require that the user owns the routine.

    Raises 404 if routine doesn't exist or user is not the creator.
    """
    routine = (
        db.query(Routine)
        .filter(Routine.id == routine_id, Routine.created_by == user_id)
        .first()
    )
    if not routine:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )
    return routine


def require_routine(db: Session, routine_id: UUID) -> Routine:
    """Require that a routine exists.

    Raises 404 if routine doesn't exist.
    """
    routine = db.query(Routine).filter(Routine.id == routine_id).first()
    if not routine:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )
    return routine


def require_session(db: Session, session_id: UUID) -> RoutineSession:
    """Require that a routine session exists.

    Raises 404 if session doesn't exist.
    """
    rs = db.query(RoutineSession).filter(RoutineSession.id == session_id).first()
    if not rs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )
    return rs


def require_session_access(
    db: Session, session_id: UUID, user_id: UUID
) -> RoutineSession:
    """Require that the user has access to the session.

    Access is granted if:
    - The user created the session, OR
    - The session belongs to a group the user is an active member of.

    Raises 404 if session doesn't exist or user has no access.
    """
    rs = require_session(db, session_id)

    # Creator always has access
    if rs.created_by == user_id:
        return rs

    # Group-scoped: check membership
    if rs.group_id is not None:
        membership = (
            db.query(GroupMembership)
            .filter(
                GroupMembership.group_id == rs.group_id,
                GroupMembership.user_id == user_id,
                GroupMembership.status == MembershipStatus.ACTIVE,
            )
            .first()
        )
        if membership:
            return rs

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Not found",
    )


def require_video_in_session(
    db: Session, session_id: UUID, video_id: UUID, *, allow_deleted: bool = False
) -> Video:
    """Require that a video belongs to the specified session and is not soft-deleted.

    Raises 404 if video doesn't exist, doesn't belong to the session, or is deleted.
    """
    query = db.query(Video).filter(
        Video.id == video_id, Video.routine_session_id == session_id
    )
    if not allow_deleted:
        query = query.filter(Video.status != VideoStatus.DELETED)
    video = query.first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )
    return video


def require_job_owner(db: Session, job_id: str, user_id: UUID) -> Job:
    """Require that the user owns the specified job.

    Raises 404 if job doesn't exist or doesn't belong to the user (non-leaky).
    """
    job = db.query(Job).filter(Job.job_id == job_id, Job.user_id == user_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )
    return job
