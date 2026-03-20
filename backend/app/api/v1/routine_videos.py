"""Routine video management endpoints (group-scoped)."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.authorization import (
    require_group_member,
    require_routine_in_group,
    require_video_in_routine,
)
from app.core.deps import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.models.video import VideoStatus
from app.schemas.video import (
    VideoDownloadResponse,
    VideoRegisterResponse,
    VideoRegisterUpload,
    VideoResponse,
)
from app.services.video_service import VideosService

router = APIRouter()


@router.post("", response_model=VideoRegisterResponse, status_code=status.HTTP_201_CREATED)
def register_upload(
    group_id: int,
    routine_id: int,
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
    require_group_member(db, group_id, current_user.id)
    require_routine_in_group(db, group_id, routine_id)

    video, upload_url, expires_at = VideosService.register_upload(
        db, routine_id, current_user.id, data
    )
    return VideoRegisterResponse(
        video=VideoResponse.model_validate(video),
        upload_url=upload_url,
        expires_at=expires_at,
    )


@router.get("", response_model=List[VideoResponse])
def list_videos(
    group_id: int,
    routine_id: int,
    video_status: Optional[VideoStatus] = Query(
        None, alias="status", description="Filter by video status"
    ),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    List videos for a routine.

    Default: only uploaded videos (visible to all group members).
    Pass ?status=pending_upload to see your own pending uploads.
    """
    require_group_member(db, group_id, current_user.id)
    require_routine_in_group(db, group_id, routine_id)

    return VideosService.list_videos(
        db,
        routine_id,
        status_filter=video_status,
        caller_user_id=current_user.id,
    )


@router.get("/{video_id}", response_model=VideoResponse)
def get_video(
    group_id: int,
    routine_id: int,
    video_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get video metadata. Deleted videos return 404."""
    require_group_member(db, group_id, current_user.id)
    require_routine_in_group(db, group_id, routine_id)
    video = require_video_in_routine(db, routine_id, video_id)
    return video


@router.post("/{video_id}/finalize", response_model=VideoResponse)
def finalize_upload(
    group_id: int,
    routine_id: int,
    video_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Mark a video upload as complete.

    Only the uploader can finalize their pending upload (idempotent).
    """
    require_group_member(db, group_id, current_user.id)
    require_routine_in_group(db, group_id, routine_id)
    video = require_video_in_routine(db, routine_id, video_id)

    result = VideosService.finalize_upload(db, video, current_user.id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return result


@router.get("/{video_id}/download", response_model=VideoDownloadResponse)
def download_video(
    group_id: int,
    routine_id: int,
    video_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get a presigned download URL. Only works for uploaded videos."""
    require_group_member(db, group_id, current_user.id)
    require_routine_in_group(db, group_id, routine_id)
    video = require_video_in_routine(db, routine_id, video_id)

    url = VideosService.get_download_url(video)
    if url is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return VideoDownloadResponse(video_id=video.id, download_url=url)


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(
    group_id: int,
    routine_id: int,
    video_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Soft delete a video.

    Sets status=deleted and migrates associated notes to routine-level.
    """
    require_group_member(db, group_id, current_user.id)
    require_routine_in_group(db, group_id, routine_id)
    video = require_video_in_routine(db, routine_id, video_id)

    deleted = VideosService.soft_delete(db, video)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return None
