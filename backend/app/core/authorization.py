"""Centralized authorization helpers for group-scoped access control.

All non-leaky: return 404 for unauthorized access to avoid leaking existence.
"""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.group import GroupMembership, MembershipStatus
from app.models.job import Job
from app.models.routine import Routine
from app.models.video import Video, VideoStatus


def require_group_member(db: Session, group_id: int, user_id: int) -> GroupMembership:
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
    db: Session, group_id: int, user_id: int, capability: str
) -> GroupMembership:
    """Require that the user has a specific capability within the group.

    v0.0.1: delegates to membership check (any active member has all capabilities).
    Future: role/capability mapping can be inserted here.
    """
    return require_group_member(db, group_id, user_id)


def require_routine_in_group(db: Session, group_id: int, routine_id: int) -> Routine:
    """Require that a routine belongs to the specified group.

    Raises 404 if routine doesn't exist or doesn't belong to the group.
    """
    routine = (
        db.query(Routine)
        .filter(Routine.id == routine_id, Routine.group_id == group_id)
        .first()
    )
    if not routine:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )
    return routine


def require_video_in_routine(
    db: Session, routine_id: int, video_id: int, *, allow_deleted: bool = False
) -> Video:
    """Require that a video belongs to the specified routine and is not soft-deleted.

    Raises 404 if video doesn't exist, doesn't belong to the routine, or is deleted.
    """
    query = db.query(Video).filter(Video.id == video_id, Video.routine_id == routine_id)
    if not allow_deleted:
        query = query.filter(Video.status != VideoStatus.DELETED)
    video = query.first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )
    return video


def require_job_owner(db: Session, job_id: str, user_id: int) -> Job:
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
