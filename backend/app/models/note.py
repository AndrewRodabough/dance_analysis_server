"""Note database model."""

from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, ForeignKey, Integer, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class NoteType(str, PyEnum):
    """Note type enumeration."""
    CRITIQUE = "critique"
    FEEDBACK = "feedback"
    COMPLEMENT = "complement"


class Note(Base):
    """A note attached to a routine, optionally referencing a video timestamp."""
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    author_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    routine_id = Column(Integer, ForeignKey("routines.id", ondelete="CASCADE"), nullable=False, index=True)
    note_type = Column(
        SQLEnum(NoteType, values_callable=lambda enum: [e.value for e in enum], name="note_type"),
        nullable=False,
    )
    contents = Column(Text, nullable=False)

    # Optional video reference for feedback notes
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="SET NULL"), nullable=True, index=True)
    video_timestamp = Column(Integer, nullable=True)  # Seconds into the video

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    author = relationship("User", backref="notes")
    routine = relationship("Routine", back_populates="notes")
    video = relationship("Video", backref="notes")

    def __repr__(self):
        return f"<Note(id={self.id}, author_id={self.author_id}, type={self.note_type})>"
