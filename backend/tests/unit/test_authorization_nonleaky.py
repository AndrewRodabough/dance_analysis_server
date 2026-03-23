"""
Unit tests for non-leaky authorization helpers.

Verifies that all authorization checks return 404 (not 403 or 5xx) for:
- Non-existent resources
- Unauthorized access to resources
- Resources that exist but user cannot access

This is a privacy requirement: error responses must not leak whether a group/job/video exists.
"""

import pytest
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.authorization import (
    require_group_capability,
    require_group_member,
    require_job_owner,
    require_routine_in_group,
    require_video_in_routine,
)
from app.models.group import Group, GroupMembership, MembershipStatus
from app.models.job import Job, JobStatus
from app.models.routine import Routine
from app.models.user import User
from app.models.video import Video, VideoStatus


@pytest.fixture
def user1(db: Session):
    """Create a test user (group member)."""
    user = User(
        email="user1@example.com",
        username="user1",
        hashed_password="hashed",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def user2(db: Session):
    """Create a second test user (non-member of most groups)."""
    user = User(
        email="user2@example.com",
        username="user2",
        hashed_password="hashed",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def group1(db: Session, user1):
    """Create a test group with user1 as member."""
    group = Group(name="Test Group", created_by=user1.id)
    db.add(group)
    db.commit()
    db.refresh(group)

    # Add user1 as active member
    membership = GroupMembership(
        group_id=group.id,
        user_id=user1.id,
        role="member",
        status=MembershipStatus.ACTIVE,
    )
    db.add(membership)
    db.commit()
    return group


@pytest.fixture
def routine1(db: Session, group1, user1):
    """Create a test routine in group1."""
    routine = Routine(
        title="Test Routine",
        group_id=group1.id,
        dance_id=1,
        created_by=user1.id,
    )
    db.add(routine)
    db.commit()
    db.refresh(routine)
    return routine


@pytest.fixture
def video1(db: Session, routine1, user1):
    """Create a test video in routine1 (uploaded)."""
    video = Video(
        routine_id=routine1.id,
        uploaded_by=user1.id,
        storage_key="test-key-1",
        status=VideoStatus.UPLOADED,
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


@pytest.fixture
def job1(db: Session, user1):
    """Create a test job owned by user1."""
    job = Job(
        job_id="test-job-1",
        user_id=user1.id,
        filename="test.mp4",
        status=JobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


# ============================================================================
# Test: require_group_member
# ============================================================================


def test_require_group_member_success(db: Session, group1, user1):
    """Member of group can pass the check."""
    membership = require_group_member(db, group1.id, user1.id)
    assert membership is not None
    assert membership.user_id == user1.id
    assert membership.group_id == group1.id


def test_require_group_member_non_member_returns_404(db: Session, group1, user2):
    """Non-member gets 404 (not 403)."""
    with pytest.raises(HTTPException) as exc_info:
        require_group_member(db, group1.id, user2.id)
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


def test_require_group_member_nonexistent_group_returns_404(db: Session, user1):
    """Non-existent group returns 404."""
    with pytest.raises(HTTPException) as exc_info:
        require_group_member(db, 99999, user1.id)
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


def test_require_group_member_generic_error_message(db: Session, group1, user2):
    """Error message is generic (does not leak group existence)."""
    with pytest.raises(HTTPException) as exc_info:
        require_group_member(db, group1.id, user2.id)
    # Should be generic, not "Group 123 not found" or similar
    assert exc_info.value.detail == "Not found"


# ============================================================================
# Test: require_group_capability
# ============================================================================


def test_require_group_capability_delegates_to_membership(db: Session, group1, user1):
    """Capability check delegates to membership (v0.0.1 behavior)."""
    # Should succeed for any active member and any capability
    membership = require_group_capability(
        db, group1.id, user1.id, "group:invite:create"
    )
    assert membership is not None


def test_require_group_capability_non_member_returns_404(db: Session, group1, user2):
    """Non-member gets 404 for any capability."""
    with pytest.raises(HTTPException) as exc_info:
        require_group_capability(db, group1.id, user2.id, "group:invite:create")
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


# ============================================================================
# Test: require_routine_in_group
# ============================================================================


def test_require_routine_in_group_success(db: Session, group1, routine1):
    """Routine in group passes check."""
    routine = require_routine_in_group(db, group1.id, routine1.id)
    assert routine is not None
    assert routine.id == routine1.id


def test_require_routine_in_group_wrong_group_returns_404(db: Session, user2, routine1):
    """Routine with different group returns 404."""
    # Create a second group
    group2 = Group(name="Other Group", created_by=user2.id)
    db.add(group2)
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        require_routine_in_group(db, group2.id, routine1.id)
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


def test_require_routine_in_group_nonexistent_routine_returns_404(db: Session, group1):
    """Non-existent routine returns 404."""
    with pytest.raises(HTTPException) as exc_info:
        require_routine_in_group(db, group1.id, 99999)
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


# ============================================================================
# Test: require_video_in_routine
# ============================================================================


def test_require_video_in_routine_success(db: Session, routine1, video1):
    """Video in routine (uploaded) passes check."""
    video = require_video_in_routine(db, routine1.id, video1.id)
    assert video is not None
    assert video.id == video1.id


def test_require_video_in_routine_deleted_returns_404(db: Session, routine1, video1):
    """Soft-deleted video returns 404 by default."""
    video1.status = VideoStatus.DELETED
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        require_video_in_routine(db, routine1.id, video1.id)
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


def test_require_video_in_routine_deleted_allowed_with_flag(
    db: Session, routine1, video1
):
    """Soft-deleted video can be retrieved with allow_deleted=True."""
    video1.status = VideoStatus.DELETED
    db.commit()

    video = require_video_in_routine(db, routine1.id, video1.id, allow_deleted=True)
    assert video is not None
    assert video.status == VideoStatus.DELETED


def test_require_video_in_routine_nonexistent_returns_404(db: Session, routine1):
    """Non-existent video returns 404."""
    with pytest.raises(HTTPException) as exc_info:
        require_video_in_routine(db, routine1.id, 99999)
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


def test_require_video_in_routine_wrong_routine_returns_404(db: Session, user1, video1):
    """Video from different routine returns 404."""
    # Create another routine
    routine2 = Routine(
        title="Other Routine",
        group_id=video1.id,  # doesn't matter, just needs to differ from video1.routine_id
        dance_id=1,
        created_by=user1.id,
    )
    db.add(routine2)
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        require_video_in_routine(db, routine2.id, video1.id)
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


# ============================================================================
# Test: require_job_owner
# ============================================================================


def test_require_job_owner_success(db: Session, job1, user1):
    """Job owner can pass the check."""
    job = require_job_owner(db, job1.job_id, user1.id)
    assert job is not None
    assert job.user_id == user1.id


def test_require_job_owner_non_owner_returns_404(db: Session, job1, user2):
    """Non-owner gets 404 (not 403)."""
    with pytest.raises(HTTPException) as exc_info:
        require_job_owner(db, job1.job_id, user2.id)
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


def test_require_job_owner_nonexistent_job_returns_404(db: Session, user1):
    """Non-existent job returns 404."""
    with pytest.raises(HTTPException) as exc_info:
        require_job_owner(db, "nonexistent-job-id", user1.id)
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


def test_require_job_owner_generic_error_message(db: Session, job1, user2):
    """Error message is generic (does not leak job existence)."""
    with pytest.raises(HTTPException) as exc_info:
        require_job_owner(db, job1.job_id, user2.id)
    # Should be generic, not "Job not found" or "Not your job"
    assert exc_info.value.detail == "Not found"
