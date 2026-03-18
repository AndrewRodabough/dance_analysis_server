"""Job database model for tracking video analysis jobs."""

from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class JobStatus(str, PyEnum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    FAILED_HIDDEN = "failed_hidden"  # Failed jobs hidden from frontend UI


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), unique=True, index=True, nullable=False)  # UUID string
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Job details
    status = Column(
        SQLEnum(JobStatus, values_callable=lambda enum: [e.value for e in enum], name="jobstatus"),
        default=JobStatus.PENDING,
        nullable=False,
        index=True
    )
    filename = Column(String(255), nullable=False)

    # Video reference and storage paths
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="SET NULL"), nullable=True, index=True)
    result_path = Column(String(500))  # S3 path to result video
    data_path = Column(String(500))  # S3 path to JSON data

    # Progress tracking (0-100)
    progress = Column(Integer, default=0)

    # Error tracking
    error_message = Column(Text)

    # Retry Information
    attempts = Column(Integer)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    # Relationship to User
    user = relationship("User", backref="jobs")
    video = relationship("Video", back_populates="jobs")

    def __repr__(self):
        return f"<Job(id={self.id}, job_id={self.job_id}, status={self.status})>"
