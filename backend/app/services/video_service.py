"""Service for managing routine video operations."""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.note import Note
from app.models.video import Video, VideoStatus
from app.schemas.video import VideoRegisterUpload
from app.services.storage import (
    create_presigned_get_url,
    create_presigned_put_url,
    generate_storage_key,
)


class VideosService:
    """Service for managing routine videos with upload lifecycle."""

    @staticmethod
    def register_upload(
        db: Session,
        routine_id: int,
        user_id: int,
        data: VideoRegisterUpload,
    ) -> tuple[Video, str, datetime]:
        """Register a new video upload. Returns (video, upload_url, expires_at)."""
        storage_key = generate_storage_key(user_id, routine_id, data.filename)

        video = Video(
            routine_id=routine_id,
            uploaded_by=user_id,
            storage_key=storage_key,
            status=VideoStatus.PENDING_UPLOAD,
            original_filename=data.filename,
            content_type=data.content_type,
            file_size=data.file_size,
        )
        db.add(video)
        db.commit()
        db.refresh(video)

        upload_url = create_presigned_put_url(storage_key, data.content_type)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)

        return video, upload_url, expires_at

    @staticmethod
    def finalize_upload(db: Session, video: Video, user_id: int) -> Optional[Video]:
        """Mark a pending upload as uploaded. Uploader-only, idempotent.

        Returns None if the caller is not the uploader.
        """
        if video.uploaded_by != user_id:
            return None
        if video.status == VideoStatus.UPLOADED:
            return video  # Idempotent
        if video.status != VideoStatus.PENDING_UPLOAD:
            return None

        video.status = VideoStatus.UPLOADED
        db.commit()
        db.refresh(video)
        return video

    @staticmethod
    def list_videos(
        db: Session,
        routine_id: int,
        *,
        status_filter: Optional[VideoStatus] = None,
        caller_user_id: Optional[int] = None,
    ) -> List[Video]:
        """List videos for a routine.

        Default: only uploaded videos.
        If status_filter=pending_upload: only the caller's pending uploads.
        """
        query = db.query(Video).filter(Video.routine_id == routine_id)

        if status_filter == VideoStatus.PENDING_UPLOAD:
            # Uploader-private: only show caller's pending uploads
            query = query.filter(
                Video.status == VideoStatus.PENDING_UPLOAD,
                Video.uploaded_by == caller_user_id,
            )
        elif status_filter:
            query = query.filter(Video.status == status_filter)
        else:
            # Default: only uploaded videos
            query = query.filter(Video.status == VideoStatus.UPLOADED)

        return query.order_by(Video.created_at.desc()).all()

    @staticmethod
    def get_download_url(video: Video) -> Optional[str]:
        """Get a presigned download URL. Only for uploaded videos."""
        if video.status != VideoStatus.UPLOADED:
            return None
        return create_presigned_get_url(video.storage_key)

    @staticmethod
    def soft_delete(db: Session, video: Video) -> bool:
        """Soft delete a video and migrate its notes to the routine.

        Sets status=deleted, nullifies video_id on notes, sets video_deleted=True.
        """
        if video.status == VideoStatus.DELETED:
            return False

        video.status = VideoStatus.DELETED

        # Migrate associated notes to routine-level
        notes = (
            db.query(Note)
            .filter(Note.video_id == video.id)
            .all()
        )
        for note in notes:
            note.video_id = None
            note.video_deleted = True

        db.commit()
        return True
