"""Service for managing dance operations."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.dance import Dance, DanceStyle
from app.schemas.dance import DanceCreate


class DanceService:
    """Service for managing dance operations."""

    @staticmethod
    def create(db: Session, data: DanceCreate) -> Dance:
        """Create a new dance definition."""
        dance = Dance(tempo=data.tempo, meter=data.meter, style=data.style)
        db.add(dance)
        db.commit()
        db.refresh(dance)
        return dance

    @staticmethod
    def list_all(db: Session) -> List[Dance]:
        """List all dance definitions ordered by style."""
        return db.query(Dance).order_by(Dance.style).all()

    @staticmethod
    def get_by_id(db: Session, dance_id: UUID) -> Optional[Dance]:
        """Get a dance by ID."""
        return db.query(Dance).filter(Dance.id == dance_id).first()

    @staticmethod
    def get_by_style(db: Session, style: DanceStyle) -> Optional[Dance]:
        """Get a dance by style."""
        return db.query(Dance).filter(Dance.style == style).first()
