"""Pydantic schemas for job-related requests and responses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.job import JobStatus


class JobCreate(BaseModel):
    """Schema for creating a new job."""
    filename: str = Field(..., min_length=1, max_length=255)


class JobResponse(BaseModel):
    """Schema for job response."""
    id: int
    job_id: str
    user_id: int
    video_id: Optional[int] = None
    status: JobStatus
    filename: str
    progress: Optional[int] = 0
    result_path: Optional[str] = None
    data_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class JobStatusUpdate(BaseModel):
    """Schema for updating job status."""
    status: JobStatus
    progress: Optional[int] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None
    data_path: Optional[str] = None
