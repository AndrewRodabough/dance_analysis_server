"""
API tests for group invite acceptance with strict email match and non-leaky failure behavior.

Verifies:
1. Invite acceptance requires strict email match (case-insensitive normalized)
2. Non-leaky failures (token invalid/expired/wrong email all return generic 404)
3. Successful acceptance creates/activates membership
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.group import Group, GroupMembership, MembershipStatus
from app.models.group_invite import GroupInvite, InviteStatus
from app.models.user import User


@pytest.fixture
def inviter(db: Session):
    """Create a user who will send invites."""
    user = User(
        email="inviter@example.com",
        username="inviter",
        hashed_password="hashed",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def group_with_inviter(db: Session, inviter):
    """Create a group with inviter as member."""
    group = Group(name="Test Group", created_by=inviter.id)
    db.add(group)
    db.commit()
    db.refresh(group)
    
    membership = GroupMembership(
        group_id=group.id,
        user_id=inviter.id,
        role="owner",
        status=MembershipStatus.ACTIVE,
    )
    db.add(membership)
    db.commit()
    return group


@pytest.fixture
def pending_invite(db: Session, group_with_inviter, inviter):
    """Create a pending group invite."""
    invite = GroupInvite(
        group_id=group_with_inviter.id,
        created_by=inviter.id,
        email="invitee@example.com",
        token="test-token-123",
        status=InviteStatus.PENDING,
        expires_at=None,  # or set a future date
    )
    db.add(invite)
    db.commit()
    db.refresh(invite)
    return invite


def test_accept_invite_success_creates_membership(
    client: TestClient, db: Session, pending_invite
):
    """Accepting a valid invite with matching email creates/activates membership."""
    # First, register a user with the invited email
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "invitee@example.com",
            "username": "invitee",
            "password": "password123",
        },
    )
    assert register_response.status_code == 200
    
    # Get auth token for the new user
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": "invitee@example.com", "password": "password123"},
    )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    
    # Accept the invite
    accept_response = client.post(
        "/api/v1/group-invites/accept",
        json={"token": "test-token-123"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert accept_response.status_code == 200
    
    # Verify membership was created
    invitee = db.query(User).filter(User.email == "invitee@example.com").first()
    membership = (
        db.query(GroupMembership)
        .filter(
            GroupMembership.user_id == invitee.id,
            GroupMembership.group_id == pending_invite.group_id,
            GroupMembership.status == MembershipStatus.ACTIVE,
        )
        .first()
    )
    assert membership is not None


def test_accept_invite_non_leaky_wrong_email(
    client: TestClient, pending_invite
):
    """Email mismatch returns 404 (non-leaky; doesn't reveal token valid for other email)."""
    # Register with a different email
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "different@example.com",
            "username": "different",
            "password": "password123",
        },
    )
    assert register_response.status_code == 200
    
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": "different@example.com", "password": "password123"},
    )
    token = login_response.json()["access_token"]
    
    # Try to accept invite with wrong email
    accept_response = client.post(
        "/api/v1/group-invites/accept",
        json={"token": "test-token-123"},
        headers={"Authorization": f"Bearer {token}"},
    )
    # Should return 404 (non-leaky: doesn't leak that token is valid for another email)
    assert accept_response.status_code == 404


def test_accept_invite_non_leaky_invalid_token(client: TestClient):
    """Invalid/non-existent token returns 404."""
    # Login as any user
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "anyuser@example.com",
            "username": "anyuser",
            "password": "password123",
        },
    )
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": "anyuser@example.com", "password": "password123"},
    )
    token = login_response.json()["access_token"]
    
    # Try to accept non-existent invite
    accept_response = client.post(
        "/api/v1/group-invites/accept",
        json={"token": "invalid-token-xyz"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert accept_response.status_code == 404


def test_accept_invite_email_case_insensitive(
    client: TestClient, db: Session, pending_invite
):
    """Email matching is case-insensitive."""
    # Register with uppercase variant of invited email
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "INVITEE@EXAMPLE.COM",  # Different case
            "username": "invitee_uppercase",
            "password": "password123",
        },
    )
    assert register_response.status_code == 200
    
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": "INVITEE@EXAMPLE.COM", "password": "password123"},
    )
    token = login_response.json()["access_token"]
    
    # Should still accept (case-insensitive match)
    accept_response = client.post(
        "/api/v1/group-invites/accept",
        json={"token": "test-token-123"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert accept_response.status_code == 200


def test_accept_invite_unauthenticated_returns_401(client: TestClient):
    """Unauthenticated request returns 401."""
    accept_response = client.post(
        "/api/v1/group-invites/accept",
        json={"token": "test-token-123"},
    )
    assert accept_response.status_code == 401
