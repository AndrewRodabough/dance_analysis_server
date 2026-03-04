"""Service for managing job queue operations."""

from sqlalchemy.orm import Session
from typing import Optional, List
import uuid
from datetime import datetime

from app.models.job import Job, JobStatus
from app.schemas.job import JobCreate, JobStatusUpdate
from app.core.logging import log_job_status


class JobService:
    """Service for managing job queue operations."""

    @staticmethod
    def create_job(db: Session, user_id: int, job_data: JobCreate, video_path: Optional[str] = None) -> Job:
        """
        Create a new job in the database.

        Args:
            db: Database session
            user_id: ID of the user creating the job
            job_data: Job creation data
            video_path: Optional S3 path to video

        Returns:
            Created Job object
        """
        job_id = str(uuid.uuid4())

        new_job = Job(
            job_id=job_id,
            user_id=user_id,
            filename=job_data.filename,
            video_path=video_path,
            status=JobStatus.PENDING
        )

        db.add(new_job)
        db.commit()
        db.refresh(new_job)

        log_job_status(job_id, status="created", user_id=user_id, file_name=job_data.filename)

        return new_job

    @staticmethod
    def get_job_by_id(db: Session, job_id: str, user_id: Optional[int] = None) -> Optional[Job]:
        """
        Get a job by job_id, optionally ensuring it belongs to the user.

        Args:
            db: Database session
            job_id: UUID string of the job
            user_id: Optional ID of the user requesting the job

        Returns:
            Job object or None if not found or unauthorized
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
        offset: int = 0
    ) -> List[Job]:
        """
        Get all jobs for a user, optionally filtered by status.

        Args:
            db: Database session
            user_id: ID of the user
            status: Optional status filter
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip (pagination)

        Returns:
            List of Job objects
        """
        query = db.query(Job).filter(Job.user_id == user_id)

        if status:
            query = query.filter(Job.status == status)
        else:
            # By default, exclude hidden failed jobs from listing
            query = query.filter(Job.status != JobStatus.FAILED_HIDDEN)

        return query.order_by(Job.created_at.desc()).limit(limit).offset(offset).all()

    @staticmethod
    def update_job_status(
        db: Session,
        job_id: str,
        status_update: JobStatusUpdate
    ) -> Optional[Job]:
        """
        Update job status and related fields.

        Args:
            db: Database session
            job_id: UUID string of the job
            status_update: Status update data

        Returns:
            Updated Job object or None if not found
        """
        job = db.query(Job).filter(Job.job_id == job_id).first()

        if not job:
            return None

        # Update status
        job.status = status_update.status

        # Update progress
        if status_update.progress is not None:
            job.progress = status_update.progress

        # Update timestamps based on status
        if status_update.status == JobStatus.PROCESSING and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status_update.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.FAILED_HIDDEN]:
            job.completed_at = datetime.utcnow()

        # Set progress to 100 on completion
        if status_update.status == JobStatus.COMPLETED:
            job.progress = 100

        # Update optional fields
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
    def update_job_video_path(db: Session, job_id: str, video_path: str) -> Optional[Job]:
        """
        Update the video path for a job.

        Args:
            db: Database session
            job_id: UUID string of the job
            video_path: S3 path to video

        Returns:
            Updated Job object or None if not found
        """
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            return None

        job.video_path = video_path
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def get_next_pending_job(db: Session) -> Optional[Job]:
        """
        Get the next pending job for processing (FIFO queue).

        Returns:
            Oldest pending Job or None if queue is empty
        """
        return db.query(Job).filter(
            Job.status == JobStatus.PENDING
        ).order_by(Job.created_at.asc()).first()

    @staticmethod
    def delete_job(db: Session, job_id: str, user_id: int) -> bool:
        """
        Delete a job.

        Args:
            db: Database session
            job_id: UUID string of the job
            user_id: ID of the user requesting deletion

        Returns:
            True if deleted, False if not found or unauthorized
        """
        job = db.query(Job).filter(
            Job.job_id == job_id,
            Job.user_id == user_id
        ).first()

        if not job:
            return False

        db.delete(job)
        db.commit()
        return True
