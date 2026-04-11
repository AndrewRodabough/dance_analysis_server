"""SessionGroupLink database model."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class SessionGroupLink(Base):
    """Link between a session and a group that has access to it."""

    __tablename__ = "session_group_links"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "group_id",
            name="uq_session_group_links_unique",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("routine_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    routine_session = relationship("RoutineSession", back_populates="group_links")
    group = relationship("Group", backref="session_group_links")

    def __repr__(self):
        return (
            f"<SessionGroupLink(id={self.id}, session_id={self.session_id}, "
            f"group_id={self.group_id})>"
        )
