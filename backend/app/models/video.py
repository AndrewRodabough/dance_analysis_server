"""Database models describing stored videos and fine-grained sharing permissions."""

from enum import Enum as PyEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class VideoVisibility(str, PyEnum):
    """Visibility options for a stored video."""
    PRIVATE = "private"
    SHARED = "shared"


class Video(Base):
    """Metadata for an uploaded video asset."""
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    storage_key = Column(String(500), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    duration = Column(Integer)  # Duration in seconds
    file_size = Column(BigInteger)  # File size in bytes

    visibility = Column(
        SQLEnum(VideoVisibility, values_callable=lambda enum: [e.value for e in enum], name="video_visibility"),
        default=VideoVisibility.PRIVATE,
        server_default=VideoVisibility.PRIVATE.value,
        nullable=False,
        index=True,
    )

    owner = relationship("User", backref="videos")
    permissions = relationship(
        "VideoPermission",
        back_populates="video",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    jobs = relationship("Job", back_populates="video")

    def __repr__(self) -> str:
        return (
            f"<Video(id={self.id}, owner_id={self.owner_id}, storage_key='{self.storage_key}', "
            f"visibility={self.visibility})>"
        )


class VideoPermission(Base):
    """Join table linking videos to users with explicit permissions."""
    __tablename__ = "video_permissions"

    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, index=True)

    can_view = Column(Boolean, default=False, server_default="false", nullable=False)
    can_download = Column(Boolean, default=False, server_default="false", nullable=False)
    can_comment = Column(Boolean, default=False, server_default="false", nullable=False)

    video = relationship("Video", back_populates="permissions", passive_deletes=True)
    user = relationship("User", backref="video_permissions")

    def __repr__(self) -> str:
        return (
            f"<VideoPermission(video_id={self.video_id}, user_id={self.user_id}, "
            f"view={self.can_view}, download={self.can_download}, comment={self.can_comment})>"
        )
