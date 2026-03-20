"""Note management endpoints (group-scoped)."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.authorization import (
    require_group_member,
    require_routine_in_group,
    require_video_in_routine,
)
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.schemas.note import NoteResponse, RoutineNoteCreate, VideoNoteCreate
from app.services.note_service import NotesService

router = APIRouter()


# --- Routine-level notes ---

@router.post(
    "/groups/{group_id}/routines/{routine_id}/notes",
    response_model=NoteResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_routine_note(
    group_id: int,
    routine_id: int,
    data: RoutineNoteCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Create a routine-level note.

    - **note_type**: Type of note (critique, feedback, complement)
    - **contents**: Note contents
    - **details**: Optional structured data (JSON)
    """
    require_group_member(db, group_id, current_user.id)
    require_routine_in_group(db, group_id, routine_id)
    return NotesService.create_routine_note(db, routine_id, current_user.id, data)


@router.get(
    "/groups/{group_id}/routines/{routine_id}/notes",
    response_model=List[NoteResponse],
)
def list_routine_notes(
    group_id: int,
    routine_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all notes for a routine (including migrated video notes)."""
    require_group_member(db, group_id, current_user.id)
    require_routine_in_group(db, group_id, routine_id)
    return NotesService.list_routine_notes(db, routine_id)


@router.delete(
    "/groups/{group_id}/routines/{routine_id}/notes/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_note(
    group_id: int,
    routine_id: int,
    note_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a note. Requires group membership."""
    require_group_member(db, group_id, current_user.id)
    require_routine_in_group(db, group_id, routine_id)
    deleted = NotesService.delete_note(db, note_id, routine_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return None


# --- Video-level notes ---

@router.post(
    "/groups/{group_id}/routines/{routine_id}/videos/{video_id}/notes",
    response_model=NoteResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_video_note(
    group_id: int,
    routine_id: int,
    video_id: int,
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
    require_group_member(db, group_id, current_user.id)
    require_routine_in_group(db, group_id, routine_id)
    require_video_in_routine(db, routine_id, video_id)
    return NotesService.create_video_note(db, routine_id, video_id, current_user.id, data)


@router.get(
    "/groups/{group_id}/routines/{routine_id}/videos/{video_id}/notes",
    response_model=List[NoteResponse],
)
def list_video_notes(
    group_id: int,
    routine_id: int,
    video_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all notes for a specific video."""
    require_group_member(db, group_id, current_user.id)
    require_routine_in_group(db, group_id, routine_id)
    require_video_in_routine(db, routine_id, video_id)
    return NotesService.list_video_notes(db, video_id)
