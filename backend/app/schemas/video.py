from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.models.video import VideoVisibility


class VideoBase(BaseModel):
    """Shared fields for video metadata."""
    storage_key: str
    visibility: VideoVisibility = VideoVisibility.PRIVATE
    duration: Optional[int] = None  # seconds
    file_size: Optional[int] = None  # bytes


class VideoCreate(VideoBase):
    """Schema used when creating a new video entry."""
    owner_id: int


class VideoUpdate(BaseModel):
    """Schema used when updating mutable video fields."""
    visibility: Optional[VideoVisibility] = None
    duration: Optional[int] = None
    file_size: Optional[int] = None


class VideoResponse(VideoBase):
    """Schema returned to clients with persisted video metadata."""
    id: int
    owner_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VideoPermissionBase(BaseModel):
    """Shared fields for video permission records."""
    can_view: bool = False
    can_download: bool = False
    can_comment: bool = False


class VideoPermissionCreate(VideoPermissionBase):
    """Schema for granting permissions to another user."""
    video_id: int
    user_id: int


class VideoPermissionResponse(VideoPermissionBase):
    """Schema returned to clients describing permissions."""
    video_id: int
    user_id: int

    model_config = ConfigDict(from_attributes=True)
