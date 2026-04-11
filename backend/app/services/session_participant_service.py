"""Service for managing session participants and casting state."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.models.routine_dancer_slot import RoutineDancerSlot
from app.models.session_access import SessionAccess
from app.models.session_participant import ParticipantRole, SessionParticipant
from app.models.session_user_state import SessionUserState
from app.models.user import User
from app.schemas.session_participant import (
    SessionParticipantCreate,
    SessionParticipantUpdate,
)


class SessionParticipantService:
    """Service for managing session participants and casting."""

    # ========================
    # Participant CRUD
    # ========================

    @staticmethod
    def create_participant(
        db: Session,
        session_id: UUID,
        user_id: UUID,
        role: ParticipantRole,
        dancer_slot_id: Optional[UUID] = None,
    ) -> SessionParticipant:
        """
        Create a session participant atomically.

        This operation:
        - Creates the participant record
        - Binds to slot if provided
        - Enforces one participant per user per session
        - Enforces one user per slot per session

        Args:
            db: Database session
            session_id: Session ID
            user_id: User ID
            role: Participant role (dancer, coach)
            dancer_slot_id: Optional slot to bind to

        Raises:
            ValueError: If participant already exists or slot is taken

        Returns:
            Created SessionParticipant
        """
        # Check one participant per user per session constraint
        existing = (
            db.query(SessionParticipant)
            .filter(
                and_(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == user_id,
                )
            )
            .first()
        )
        if existing:
            raise ValueError(
                f"User {user_id} already has a participant record in session {session_id}"
            )

        # Check one user per slot constraint if binding
        if dancer_slot_id is not None:
            slot_taken = (
                db.query(SessionParticipant)
                .filter(
                    and_(
                        SessionParticipant.session_id == session_id,
                        SessionParticipant.dancer_slot_id == dancer_slot_id,
                    )
                )
                .first()
            )
            if slot_taken:
                raise ValueError(
                    f"Slot {dancer_slot_id} already has a participant in session {session_id}"
                )

        # Validate role is in v1 set
        if role not in (ParticipantRole.DANCER, ParticipantRole.COACH):
            raise ValueError(f"Invalid role for v1: {role}")

        # Create participant
        participant = SessionParticipant(
            session_id=session_id,
            user_id=user_id,
            role=role,
            dancer_slot_id=dancer_slot_id,
        )
        db.add(participant)
        db.commit()
        db.refresh(participant)
        return participant

    @staticmethod
    def get_by_id(db: Session, participant_id: UUID) -> Optional[SessionParticipant]:
        """Get a participant by ID."""
        return (
            db.query(SessionParticipant)
            .filter(SessionParticipant.id == participant_id)
            .first()
        )

    @staticmethod
    def get_participant(
        db: Session, session_id: UUID, user_id: UUID
    ) -> Optional[SessionParticipant]:
        """Get a participant record for a specific user in a session."""
        return (
            db.query(SessionParticipant)
            .filter(
                and_(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.user_id == user_id,
                )
            )
            .first()
        )

    @staticmethod
    def list_session_participants(
        db: Session, session_id: UUID
    ) -> List[SessionParticipant]:
        """List all participants in a session."""
        return (
            db.query(SessionParticipant)
            .filter(SessionParticipant.session_id == session_id)
            .order_by(SessionParticipant.created_at.asc())
            .all()
        )

    @staticmethod
    def update_participant(
        db: Session,
        participant_id: UUID,
        data: SessionParticipantUpdate,
    ) -> SessionParticipant:
        """
        Update a participant's role and/or slot binding.

        If changing slot binding, enforces one user per slot constraint.
        """
        participant = SessionParticipantService.get_by_id(db, participant_id)
        if not participant:
            raise ValueError(f"Participant {participant_id} not found")

        # If changing slot, check constraint
        if (
            data.dancer_slot_id is not None
            and data.dancer_slot_id != participant.dancer_slot_id
        ):
            slot_taken = (
                db.query(SessionParticipant)
                .filter(
                    and_(
                        SessionParticipant.session_id == participant.session_id,
                        SessionParticipant.dancer_slot_id == data.dancer_slot_id,
                        SessionParticipant.id != participant_id,
                    )
                )
                .first()
            )
            if slot_taken:
                raise ValueError(
                    f"Slot {data.dancer_slot_id} is already taken in this session"
                )

        # Update fields if provided
        if data.role is not None:
            # Validate role is in v1 set
            if data.role not in (
                ParticipantRole.DANCER.value,
                ParticipantRole.COACH.value,
            ):
                raise ValueError(f"Invalid role for v1: {data.role}")
            participant.role = ParticipantRole(data.role)

        if data.dancer_slot_id is not None or (
            hasattr(data, "dancer_slot_id") and data.dancer_slot_id is None
        ):
            participant.dancer_slot_id = data.dancer_slot_id

        db.commit()
        db.refresh(participant)
        return participant

    @staticmethod
    def delete_participant(db: Session, participant_id: UUID) -> bool:
        """Delete a participant and clear their slot binding."""
        participant = SessionParticipantService.get_by_id(db, participant_id)
        if not participant:
            return False

        db.delete(participant)
        db.commit()
        return True

    # ========================
    # Slot Management
    # ========================

    @staticmethod
    def bind_to_slot(
        db: Session,
        participant_id: UUID,
        dancer_slot_id: UUID,
    ) -> SessionParticipant:
        """Bind a participant to a dancer slot."""
        participant = SessionParticipantService.get_by_id(db, participant_id)
        if not participant:
            raise ValueError(f"Participant {participant_id} not found")

        # Check slot constraint
        slot_taken = (
            db.query(SessionParticipant)
            .filter(
                and_(
                    SessionParticipant.session_id == participant.session_id,
                    SessionParticipant.dancer_slot_id == dancer_slot_id,
                    SessionParticipant.id != participant_id,
                )
            )
            .first()
        )
        if slot_taken:
            raise ValueError(f"Slot {dancer_slot_id} is already occupied")

        participant.dancer_slot_id = dancer_slot_id
        db.commit()
        db.refresh(participant)
        return participant

    @staticmethod
    def unbind_from_slot(db: Session, participant_id: UUID) -> SessionParticipant:
        """Remove a participant's slot binding."""
        participant = SessionParticipantService.get_by_id(db, participant_id)
        if not participant:
            raise ValueError(f"Participant {participant_id} not found")

        participant.dancer_slot_id = None
        db.commit()
        db.refresh(participant)
        return participant

    @staticmethod
    def get_slot_occupant(
        db: Session, session_id: UUID, dancer_slot_id: UUID
    ) -> Optional[SessionParticipant]:
        """Get the participant occupying a slot."""
        return (
            db.query(SessionParticipant)
            .filter(
                and_(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.dancer_slot_id == dancer_slot_id,
                )
            )
            .first()
        )

    # ========================
    # Availability & Queries
    # ========================

    @staticmethod
    def list_available_users(db: Session, session_id: UUID) -> List[User]:
        """
        Get users with session access but no participant record.

        Returns all users who can be added as participants.
        """
        # Get all users with access to this session
        users_with_access = (
            db.query(User)
            .join(
                SessionAccess,
                and_(
                    SessionAccess.session_id == session_id,
                    SessionAccess.user_id == User.id,
                ),
            )
            .all()
        )

        # Get users already as participants
        participant_user_ids = (
            db.query(SessionParticipant.user_id)
            .filter(SessionParticipant.session_id == session_id)
            .all()
        )
        participant_ids = {pid[0] for pid in participant_user_ids}

        # Get users with deleted state
        deleted_user_ids = (
            db.query(SessionUserState.user_id)
            .filter(
                and_(
                    SessionUserState.session_id == session_id,
                    SessionUserState.is_deleted,
                )
            )
            .all()
        )
        deleted_ids = {did[0] for did in deleted_user_ids}

        # Filter to only available users
        available = [
            u
            for u in users_with_access
            if u.id not in participant_ids and u.id not in deleted_ids
        ]

        return available

    @staticmethod
    def list_participants_with_users(db: Session, session_id: UUID) -> List[dict]:
        """
        List participants with their user information.

        Returns list of dicts: {"participant": SessionParticipant, "user": User}
        """
        participants = SessionParticipantService.list_session_participants(
            db, session_id
        )
        result = []

        for participant in participants:
            user = db.query(User).filter(User.id == participant.user_id).first()
            result.append(
                {
                    "participant": participant,
                    "user": user,
                }
            )

        return result

    @staticmethod
    def count_participants_in_session(db: Session, session_id: UUID) -> int:
        """Count the number of participants in a session."""
        return (
            db.query(SessionParticipant)
            .filter(SessionParticipant.session_id == session_id)
            .count()
        )

    @staticmethod
    def list_participants_by_role(
        db: Session, session_id: UUID, role: ParticipantRole
    ) -> List[SessionParticipant]:
        """List participants in a session filtered by role."""
        return (
            db.query(SessionParticipant)
            .filter(
                and_(
                    SessionParticipant.session_id == session_id,
                    SessionParticipant.role == role,
                )
            )
            .order_by(SessionParticipant.created_at.asc())
            .all()
        )

    # ========================
    # Reordering
    # ========================

    @staticmethod
    def reorder_participants(
        db: Session,
        session_id: UUID,
        participant_order: List[UUID],
    ) -> List[SessionParticipant]:
        """
        Reorder participants in a session.

        All participant IDs must belong to the session.
        Order is determined by the order of IDs in the list.
        """
        # Verify all participants exist and belong to session
        participants = {}
        for participant_id in participant_order:
            participant = SessionParticipantService.get_by_id(db, participant_id)
            if not participant or participant.session_id != session_id:
                raise ValueError(
                    f"Participant {participant_id} not found in session {session_id}"
                )
            participants[participant_id] = participant

        # Verify all session participants are included
        all_participants = SessionParticipantService.list_session_participants(
            db, session_id
        )
        if len(all_participants) != len(participant_order):
            raise ValueError(
                f"Expected {len(all_participants)} participants, got {len(participant_order)}"
            )

        # In PostgreSQL, we'd use a position column
        # For now, we just return in requested order
        return [participants[pid] for pid in participant_order]

    # ========================
    # Synchronization
    # ========================

    @staticmethod
    def sync_on_access_revoked(
        db: Session,
        session_id: UUID,
        user_id: UUID,
    ) -> bool:
        """
        Sync participant state when user loses access to session.

        If the user is a participant, removes them as participant.
        """
        participant = SessionParticipantService.get_participant(db, session_id, user_id)
        if not participant:
            return False

        # Remove participant
        db.delete(participant)
        db.commit()
        return True

    @staticmethod
    def sync_on_user_deleted(
        db: Session,
        session_id: UUID,
        user_id: UUID,
    ) -> bool:
        """
        Sync participant state when user is marked as deleted from session.

        If the user is a participant, removes them as participant.
        """
        return SessionParticipantService.sync_on_access_revoked(db, session_id, user_id)
