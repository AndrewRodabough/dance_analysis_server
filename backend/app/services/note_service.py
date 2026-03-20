"""Service for managing note operations."""

from typing import List

from sqlalchemy.orm import Session

from app.models.note import Note, NoteSource
from app.schemas.note import RoutineNoteCreate, VideoNoteCreate


class NotesService:
    """Service for managing notes on routines and videos."""

    @staticmethod
    def create_routine_note(
        db: Session, routine_id: int, author_id: int, data: RoutineNoteCreate
    ) -> Note:
        """Create a routine-level note."""
        note = Note(
            routine_id=routine_id,
            author_id=author_id,
            note_type=data.note_type,
            contents=data.contents,
            source=NoteSource.USER,
            details=data.details,
        )
        db.add(note)
        db.commit()
        db.refresh(note)
        return note

    @staticmethod
    def create_video_note(
        db: Session,
        routine_id: int,
        video_id: int,
        author_id: int,
        data: VideoNoteCreate,
    ) -> Note:
        """Create a video note with optional timestamp."""
        note = Note(
            routine_id=routine_id,
            video_id=video_id,
            author_id=author_id,
            note_type=data.note_type,
            contents=data.contents,
            source=NoteSource.USER,
            video_timestamp_ms=data.video_timestamp_ms,
            details=data.details,
        )
        db.add(note)
        db.commit()
        db.refresh(note)
        return note

    @staticmethod
    def list_routine_notes(db: Session, routine_id: int) -> List[Note]:
        """List all notes for a routine (including migrated video notes)."""
        return (
            db.query(Note)
            .filter(Note.routine_id == routine_id)
            .order_by(Note.created_at.desc())
            .all()
        )

    @staticmethod
    def list_video_notes(db: Session, video_id: int) -> List[Note]:
        """List all notes for a specific video."""
        return (
            db.query(Note)
            .filter(Note.video_id == video_id)
            .order_by(Note.created_at.desc())
            .all()
        )

    @staticmethod
    def delete_note(db: Session, note_id: int, routine_id: int) -> bool:
        """Delete a note. Returns True if deleted."""
        note = (
            db.query(Note)
            .filter(Note.id == note_id, Note.routine_id == routine_id)
            .first()
        )
        if not note:
            return False
        db.delete(note)
        db.commit()
        return True
