"""Pydantic schemas for note-related requests and responses."""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.note import NoteSource, NoteType


class RoutineNoteCreate(BaseModel):
    """Request body for creating a routine-level note."""

    note_type: NoteType
    contents: str = Field(..., min_length=1)
    details: Optional[Dict[str, Any]] = None


class VideoNoteCreate(BaseModel):
    """Request body for creating a video note with timestamp."""

    note_type: NoteType
    contents: str = Field(..., min_length=1)
    video_timestamp_ms: Optional[int] = Field(default=None, ge=0)
    details: Optional[Dict[str, Any]] = None


class NoteResponse(BaseModel):
    """Response body for note data."""

    id: UUID
    author_id: UUID
    routine_session_id: UUID
    video_id: Optional[UUID] = None
    video_deleted: bool = False
    video_timestamp_ms: Optional[int] = None
    note_type: NoteType
    source: NoteSource = NoteSource.USER
    contents: str
    details: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
