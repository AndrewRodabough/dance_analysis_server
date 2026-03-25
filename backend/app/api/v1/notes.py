"""Note management endpoints (session-scoped)."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.authorization import require_session_access, require_video_in_session
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.schemas.note import NoteResponse, RoutineNoteCreate, VideoNoteCreate
from app.services.note_service import NotesService

router = APIRouter()


# --- Session-level notes ---

@router.post(
    "/sessions/{session_id}/notes",
    response_model=NoteResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session_note(
    session_id: UUID,
    data: RoutineNoteCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Create a session-level note.

    - **note_type**: Type of note (critique, feedback, complement)
    - **contents**: Note contents
    - **details**: Optional structured data (JSON)
    """
    require_session_access(db, session_id, current_user.id)
    return NotesService.create_session_note(db, session_id, current_user.id, data)


@router.get(
    "/sessions/{session_id}/notes",
    response_model=List[NoteResponse],
)
def list_session_notes(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all notes for a session (including migrated video notes)."""
    require_session_access(db, session_id, current_user.id)
    return NotesService.list_session_notes(db, session_id)


@router.delete(
    "/sessions/{session_id}/notes/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_note(
    session_id: UUID,
    note_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a note. Requires session access."""
    require_session_access(db, session_id, current_user.id)
    deleted = NotesService.delete_note(db, note_id, session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return None


# --- Video-level notes ---

@router.post(
    "/sessions/{session_id}/videos/{video_id}/notes",
    response_model=NoteResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_video_note(
    session_id: UUID,
    video_id: UUID,
    data: VideoNoteCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Create a video note with optional timestamp.

    - **note_type**: Type of note
    - **contents**: Note contents
    - **video_timestamp_ms**: Optional timestamp in milliseconds
    - **details**: Optional structured data
    """
    require_session_access(db, session_id, current_user.id)
    require_video_in_session(db, session_id, video_id)
    return NotesService.create_video_note(db, session_id, video_id, current_user.id, data)


@router.get(
    "/sessions/{session_id}/videos/{video_id}/notes",
    response_model=List[NoteResponse],
)
def list_video_notes(
    session_id: UUID,
    video_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all notes for a specific video."""
    require_session_access(db, session_id, current_user.id)
    require_video_in_session(db, session_id, video_id)
    return NotesService.list_video_notes(db, video_id)
