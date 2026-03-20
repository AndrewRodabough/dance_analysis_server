"""Video database model with routine-scoped access and upload lifecycle."""

from enum import Enum as PyEnum

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class VideoStatus(str, PyEnum):
    """Upload lifecycle status for a video."""
    PENDING_UPLOAD = "pending_upload"
    UPLOADED = "uploaded"
    DELETED = "deleted"


class Video(Base):
    """Metadata for a video asset scoped to a routine."""
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    routine_id = Column(Integer, ForeignKey("routines.id", ondelete="CASCADE"), nullable=True, index=True)
    uploaded_by = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    storage_key = Column(String(500), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    duration = Column(Integer)  # Duration in seconds
    file_size = Column(BigInteger)  # File size in bytes

    status = Column(
        SQLEnum(VideoStatus, values_callable=lambda enum: [e.value for e in enum], name="video_status"),
        default=VideoStatus.PENDING_UPLOAD,
        server_default=VideoStatus.PENDING_UPLOAD.value,
        nullable=False,
        index=True,
    )

    original_filename = Column(String(500), nullable=True)
    content_type = Column(String(100), nullable=True)

    # Relationships
    routine = relationship("Routine", back_populates="videos")
    uploader = relationship("User", backref="uploaded_videos")
    jobs = relationship("Job", back_populates="video")
    notes = relationship("Note", back_populates="video")

    def __repr__(self) -> str:
        return (
            f"<Video(id={self.id}, routine_id={self.routine_id}, "
            f"uploaded_by={self.uploaded_by}, status={self.status})>"
        )
