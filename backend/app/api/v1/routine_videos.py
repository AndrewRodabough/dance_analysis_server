"""Session video management endpoints."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.authorization import require_session_access, require_video_in_session
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.models.video import VideoStatus
from app.schemas.video import (
    VideoStreamUrlResponse,
    VideoRegisterResponse,
    VideoRegisterUpload,
    VideoResponse,
)
from app.services.video_service import VideosService

router = APIRouter()


@router.post("", response_model=VideoRegisterResponse, status_code=status.HTTP_201_CREATED)
def register_upload(
    session_id: UUID,
    data: VideoRegisterUpload,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Register a new video upload.

    Returns a presigned PUT URL for the client to upload the file directly.

    - **filename**: Name of the video file
    - **content_type**: MIME type (default: video/mp4)
    - **file_size**: Optional file size in bytes
    """
    require_session_access(db, session_id, current_user.id)

    video, upload_url, expires_at = VideosService.register_upload(
        db, session_id, current_user.id, data
    )
    return VideoRegisterResponse(
        video=VideoResponse.model_validate(video),
        upload_url=upload_url,
        expires_at=expires_at,
    )


@router.get("", response_model=List[VideoResponse])
def list_videos(
    session_id: UUID,
    video_status: Optional[VideoStatus] = Query(
        None, alias="status", description="Filter by video status"
    ),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    List videos for a session.

    Default: only uploaded videos (visible to all session members).
    Pass ?status=pending_upload to see your own pending uploads.
    """
    require_session_access(db, session_id, current_user.id)

    return VideosService.list_videos(
        db,
        session_id,
        status_filter=video_status,
        caller_user_id=current_user.id,
    )


@router.get("/{video_id}", response_model=VideoResponse)
def get_video(
    session_id: UUID,
    video_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get video metadata. Deleted videos return 404."""
    require_session_access(db, session_id, current_user.id)
    video = require_video_in_session(db, session_id, video_id)
    return video


@router.post("/{video_id}/finalize", response_model=VideoResponse)
def finalize_upload(
    session_id: UUID,
    video_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Mark a video upload as complete.

    Only the uploader can finalize their pending upload (idempotent).
    """
    require_session_access(db, session_id, current_user.id)
    video = require_video_in_session(db, session_id, video_id)

    result = VideosService.finalize_upload(db, video, current_user.id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return result


@router.get("/{video_id}/stream-url", response_model=VideoStreamUrlResponse)
def get_stream_url(
    session_id: UUID,
    video_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get a presigned stream URL. Only works for uploaded videos."""
    require_session_access(db, session_id, current_user.id)
    video = require_video_in_session(db, session_id, video_id)

    url = VideosService.get_stream_url(video)
    if url is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return VideoStreamUrlResponse(video_id=video.id, stream_url=url)


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(
    session_id: UUID,
    video_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Soft delete a video.

    Sets status=deleted and migrates associated notes to session-level.
    """
    require_session_access(db, session_id, current_user.id)
    video = require_video_in_session(db, session_id, video_id)

    deleted = VideosService.soft_delete(db, video)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return None
