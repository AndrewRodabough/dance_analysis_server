"""Job management endpoints - list, view, and manage user jobs."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.core.deps import get_current_active_user
from app.models.user import User
from app.models.job import JobStatus
from app.schemas.job import JobResponse
from app.services.job_service import JobService

router = APIRouter()


@router.get("/jobs", response_model=List[JobResponse])
async def list_user_jobs(
    job_status: Optional[JobStatus] = Query(None, alias="status", description="Filter by job status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all jobs for the current user.

    - **status**: Optional filter by job status (pending, processing, completed, failed)
    - **limit**: Maximum results (1-100, default 50)
    - **offset**: Pagination offset (default 0)
    """
    jobs = JobService.get_user_jobs(
        db=db,
        user_id=current_user.id,
        status=job_status,
        limit=limit,
        offset=offset
    )
    return jobs


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_details(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get details of a specific job."""
    job = JobService.get_job_by_id(db, job_id, current_user.id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    return job


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete a job.

    Note: This will delete job metadata. Associated S3 files are also cleaned up.
    """
    deleted = JobService.delete_job(db, job_id, current_user.id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    return None
