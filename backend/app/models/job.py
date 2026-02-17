"""Job database model for tracking video analysis jobs."""

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum as PyEnum
from app.database import Base


class JobStatus(str, PyEnum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), unique=True, index=True, nullable=False)  # UUID string
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Job details
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)
    filename = Column(String(255), nullable=False)

    # Storage paths
    video_path = Column(String(500))  # S3 path to original video
    result_path = Column(String(500))  # S3 path to result video
    data_path = Column(String(500))  # S3 path to JSON data

    # Error tracking
    error_message = Column(Text)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    # Relationship to User
    user = relationship("User", backref="jobs")

    def __repr__(self):
        return f"<Job(id={self.id}, job_id={self.job_id}, status={self.status})>"
