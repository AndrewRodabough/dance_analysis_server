"""Session participants and casting management endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.authorization import require_session_access, require_session_owner
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.session_participant import ParticipantRole, SessionParticipant
from app.models.user import User
from app.schemas.session_participant import (
    ParticipantListResponse,
    SessionParticipantCreate,
    SessionParticipantResponse,
    SessionParticipantUpdate,
    SessionParticipantWithUserResponse,
)
from app.services.session_participant_service import SessionParticipantService

router = APIRouter()


# ========================
# List Participants
# ========================


@router.get(
    "/sessions/{session_id}/participants",
    response_model=ParticipantListResponse,
)
def list_participants(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all participants in a session with available users.

    Requires: user has access to the session.
    """
    require_session_access(db, session_id, current_user.id)

    participants = SessionParticipantService.list_session_participants(db, session_id)
    available_users = SessionParticipantService.list_available_users(db, session_id)

    return ParticipantListResponse(
        participants=participants,
        available_users=[u.id for u in available_users],
        total_count=len(participants),
    )


@router.get(
    "/sessions/{session_id}/participants/with-users",
    response_model=list[SessionParticipantWithUserResponse],
)
def list_participants_with_users(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List participants with their user details.

    Requires: user has access to the session.
    """
    require_session_access(db, session_id, current_user.id)

    participants_with_users = SessionParticipantService.list_participants_with_users(
        db, session_id
    )

    result = []
    for item in participants_with_users:
        participant = item["participant"]
        user = item["user"]
        result.append(
            SessionParticipantWithUserResponse(
                id=participant.id,
                session_id=participant.session_id,
                user_id=participant.user_id,
                role=participant.role.value,
                dancer_slot_id=participant.dancer_slot_id,
                created_at=participant.created_at,
                user_email=user.email if user else None,
                user_name=getattr(user, "name", None) if user else None,
            )
        )

    return result


@router.get(
    "/sessions/{session_id}/available-users",
    response_model=list[dict],
)
def list_available_users(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List users available to add as participants.

    Available = has access + not already a participant + not deleted.

    Requires: user has access to the session.
    """
    require_session_access(db, session_id, current_user.id)

    available_users = SessionParticipantService.list_available_users(db, session_id)

    return [
        {"id": user.id, "email": user.email, "name": getattr(user, "name", None)}
        for user in available_users
    ]


# ========================
# Create Participant
# ========================


@router.post(
    "/sessions/{session_id}/participants",
    response_model=SessionParticipantResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_participant(
    session_id: UUID,
    data: SessionParticipantCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Create a participant in a session.

    This operation is ATOMIC:
    - Creates participant record
    - Binds to slot (if provided)
    - Enforces one participant per user per session
    - Enforces one user per slot per session
    - Validates role is dancer or coach (v1 only)

    Requires: user is the session owner.
    """
    require_session_owner(db, session_id, current_user.id)

    # Validate role
    try:
        role = ParticipantRole(data.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {data.role}. Must be one of: {', '.join([r.value for r in ParticipantRole])}",
        )

    # Create participant atomically
    try:
        participant = SessionParticipantService.create_participant(
            db=db,
            session_id=session_id,
            user_id=data.user_id,
            role=role,
            dancer_slot_id=data.dancer_slot_id,
        )
        return participant
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# ========================
# Get Single Participant
# ========================


@router.get(
    "/sessions/{session_id}/participants/{participant_id}",
    response_model=SessionParticipantResponse,
)
def get_participant(
    session_id: UUID,
    participant_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get a specific participant.

    Requires: user has access to the session.
    """
    require_session_access(db, session_id, current_user.id)

    participant = SessionParticipantService.get_by_id(db, participant_id)
    if not participant or participant.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Participant not found",
        )

    return participant


# ========================
# Update Participant
# ========================


@router.patch(
    "/sessions/{session_id}/participants/{participant_id}",
    response_model=SessionParticipantResponse,
)
def update_participant(
    session_id: UUID,
    participant_id: UUID,
    data: SessionParticipantUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update a participant's role and/or slot binding.

    Can update:
    - role: dancer or coach
    - dancer_slot_id: bind to slot or null to unbind

    Enforces one user per slot constraint.

    Requires: user is the session owner.
    """
    require_session_owner(db, session_id, current_user.id)

    participant = SessionParticipantService.get_by_id(db, participant_id)
    if not participant or participant.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Participant not found",
        )

    try:
        updated = SessionParticipantService.update_participant(db, participant_id, data)
        return updated
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# ========================
# Delete Participant
# ========================


@router.delete(
    "/sessions/{session_id}/participants/{participant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_participant(
    session_id: UUID,
    participant_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a participant from a session.

    Removes:
    - Participant record
    - Slot binding
    - All associated data

    Requires: user is the session owner.
    """
    require_session_owner(db, session_id, current_user.id)

    participant = SessionParticipantService.get_by_id(db, participant_id)
    if not participant or participant.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Participant not found",
        )

    SessionParticipantService.delete_participant(db, participant_id)
    return None


# ========================
# Slot Binding
# ========================


@router.post(
    "/sessions/{session_id}/participants/{participant_id}/bind-slot/{slot_id}",
    response_model=SessionParticipantResponse,
)
def bind_participant_to_slot(
    session_id: UUID,
    participant_id: UUID,
    slot_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Bind a participant to a dancer slot.

    Enforces one user per slot per session.

    Requires: user is the session owner.
    """
    require_session_owner(db, session_id, current_user.id)

    participant = SessionParticipantService.get_by_id(db, participant_id)
    if not participant or participant.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Participant not found",
        )

    try:
        updated = SessionParticipantService.bind_to_slot(db, participant_id, slot_id)
        return updated
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/sessions/{session_id}/participants/{participant_id}/unbind-slot",
    response_model=SessionParticipantResponse,
)
def unbind_participant_from_slot(
    session_id: UUID,
    participant_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Remove a participant's slot binding.

    Participant remains in session, just no longer bound to a slot.

    Requires: user is the session owner.
    """
    require_session_owner(db, session_id, current_user.id)

    participant = SessionParticipantService.get_by_id(db, participant_id)
    if not participant or participant.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Participant not found",
        )

    try:
        updated = SessionParticipantService.unbind_from_slot(db, participant_id)
        return updated
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
