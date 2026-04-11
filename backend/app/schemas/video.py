"""Pydantic schemas for video-related requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.video import VideoStatus


class VideoRegisterUpload(BaseModel):
    """Request body for registering a new video upload."""

    filename: str = Field(..., min_length=1, max_length=500)
    content_type: str = Field(default="video/mp4", max_length=100)
    file_size: Optional[int] = None


class VideoRegisterResponse(BaseModel):
    """Response after registering a video upload (includes presigned URL)."""

    video: "VideoResponse"
    upload_url: str
    expires_at: datetime


class VideoResponse(BaseModel):
    """Response body for video metadata."""

    id: UUID
    routine_session_id: Optional[UUID] = None
    uploaded_by: UUID
    storage_key: str
    status: VideoStatus
    original_filename: Optional[str] = None
    content_type: Optional[str] = None
    duration: Optional[str] = None
    file_size: Optional[int] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VideoStreamUrlResponse(BaseModel):
    """Response with a presigned stream URL."""

    video_id: UUID
    stream_url: str
    expires_in: int = 3600


# Resolve forward reference
VideoRegisterResponse.model_rebuild()
