"""Service layer for orchestrating job lifecycle events and linked video metadata."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from app.core.logging import log_job_status
from app.models.job import Job, JobStatus
from app.models.video import Video
from app.schemas.job import JobCreate, JobStatusUpdate


class JobService:
    """Business logic for CRUD operations on jobs and their associated videos."""

    # -------------------------------------------------------------------------
    # Creation & Retrieval
    # -------------------------------------------------------------------------
    @staticmethod
    def create_job(
        db: Session,
        user_id: int,
        job_data: JobCreate,
        video_storage_key: Optional[str] = None,
    ) -> Job:
        """
        Create a new job record and optionally link it to a video asset.

        Args:
            db: Database session.
            user_id: ID of the authenticated user requesting the job.
            job_data: DTO describing the upload (currently just filename).
            video_storage_key: Optional storage key if the video artifact already exists.

        Returns:
            The persisted Job instance.
        """
        job_id = str(uuid.uuid4())

        video = (
            JobService._get_or_create_video(db, owner_id=user_id, storage_key=video_storage_key)
            if video_storage_key
            else None
        )

        new_job = Job(
            job_id=job_id,
            user_id=user_id,
            filename=job_data.filename,
            status=JobStatus.PENDING,
            video=video,
        )

        db.add(new_job)
        db.commit()
        db.refresh(new_job)

        log_job_status(job_id, status="created", user_id=user_id, file_name=job_data.filename)
        return new_job

    @staticmethod
    def get_job_by_id(db: Session, job_id: str, user_id: Optional[int] = None) -> Optional[Job]:
        """
        Retrieve a job by its public UUID, optionally enforcing ownership.

        Args:
            db: Database session.
            job_id: Public UUID assigned to the job.
            user_id: If provided, ensures the job belongs to this user.

        Returns:
            Job instance or None.
        """
        query = db.query(Job).filter(Job.job_id == job_id)
        if user_id is not None:
            query = query.filter(Job.user_id == user_id)
        return query.first()

    @staticmethod
    def get_user_jobs(
        db: Session,
        user_id: int,
        status: Optional[JobStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Job]:
        """
        Paginated listing of jobs owned by a specific user.

        Args:
            db: Database session.
            user_id: Owner of the jobs.
            status: Optional filter for job status.
            limit: Max results.
            offset: Pagination offset.

        Returns:
            List of Job instances.
        """
        query = db.query(Job).filter(Job.user_id == user_id)

        if status:
            query = query.filter(Job.status == status)
        else:
            query = query.filter(Job.status != JobStatus.FAILED_HIDDEN)

        return query.order_by(Job.created_at.desc()).limit(limit).offset(offset).all()

    # -------------------------------------------------------------------------
    # Mutations
    # -------------------------------------------------------------------------
    @staticmethod
    def update_job_status(
        db: Session,
        job_id: str,
        status_update: JobStatusUpdate,
    ) -> Optional[Job]:
        """
        Apply status/metadata updates emitted by workers.

        Args:
            db: Database session.
            job_id: Public UUID of the job.
            status_update: Pydantic DTO carrying update payload.

        Returns:
            Updated Job instance or None.
        """
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            return None

        job.status = status_update.status

        if status_update.progress is not None:
            job.progress = status_update.progress

        if status_update.status == JobStatus.PROCESSING and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status_update.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.FAILED_HIDDEN}:
            job.completed_at = datetime.utcnow()

        if status_update.status == JobStatus.COMPLETED:
            job.progress = 100

        if status_update.error_message:
            job.error_message = status_update.error_message
        if status_update.result_path:
            job.result_path = status_update.result_path
        if status_update.data_path:
            job.data_path = status_update.data_path

        db.commit()
        db.refresh(job)

        log_job_status(
            job_id,
            status=status_update.status.value,
            error=status_update.error_message,
        )
        return job

    @staticmethod
    def update_job_video_path(db: Session, job_id: str, storage_key: str) -> Optional[Job]:
        """
        Ensure a job references the canonical video row for the provided storage key.

        Args:
            db: Database session.
            job_id: Public UUID of the job.
            storage_key: Cloud/Object storage identifier of the uploaded video.

        Returns:
            Updated Job or None if job not found.
        """
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            return None

        video = JobService._get_or_create_video(db, owner_id=job.user_id, storage_key=storage_key)
        job.video = video

        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def delete_job(db: Session, job_id: str, user_id: int) -> bool:
        """
        Delete a job if it is owned by the requesting user.

        Args:
            db: Database session.
            job_id: Public UUID of the job.
            user_id: Owner verification.

        Returns:
            True if deleted.
        """
        job = db.query(Job).filter(Job.job_id == job_id, Job.user_id == user_id).first()
        if not job:
            return False

        db.delete(job)
        db.commit()
        return True

    @staticmethod
    def get_next_pending_job(db: Session) -> Optional[Job]:
        """
        Retrieve the oldest pending job for worker consumption (FIFO semantics).
        """
        return (
            db.query(Job)
            .filter(Job.status == JobStatus.PENDING)
            .order_by(Job.created_at.asc())
            .first()
        )

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_or_create_video(db: Session, owner_id: int, storage_key: Optional[str]) -> Optional[Video]:
        """
        Deduplicate video rows by storage key.
        """
        if not storage_key:
            return None

        video = (
            db.query(Video)
            .filter(Video.uploaded_by == owner_id, Video.storage_key == storage_key)
            .first()
        )
        if video:
            return video

        video = Video(uploaded_by=owner_id, storage_key=storage_key)
        db.add(video)
        db.flush()
        return video
