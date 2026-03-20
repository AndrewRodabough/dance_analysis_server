"""Routine database model."""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Routine(Base):
    """A dance routine scoped to a group."""
    __tablename__ = "routines"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    group_id = Column(Integer, ForeignKey("groups.id", ondelete="CASCADE"), nullable=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    dance_id = Column(Integer, ForeignKey("dances.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    group = relationship("Group", back_populates="routines")
    creator = relationship("User", backref="created_routines")
    dance = relationship("Dance", backref="routines")
    videos = relationship(
        "Video",
        back_populates="routine",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    notes = relationship(
        "Note",
        back_populates="routine",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self):
        return f"<Routine(id={self.id}, title={self.title}, group_id={self.group_id})>"
