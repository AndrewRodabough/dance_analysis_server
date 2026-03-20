"""Pydantic schemas for note-related requests and responses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.note import NoteType


class NoteBase(BaseModel):
    """Shared fields for a note."""
    note_type: NoteType
    contents: str = Field(..., min_length=1)


class NoteCreate(NoteBase):
    """Request body for creating a note."""
    routine_id: int
    video_id: Optional[int] = None
    video_timestamp: Optional[int] = None


class NoteResponse(NoteBase):
    """Response body for note data."""
    id: int
    author_id: int
    routine_id: int
    video_id: Optional[int] = None
    video_timestamp: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
