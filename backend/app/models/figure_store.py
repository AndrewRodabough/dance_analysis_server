"""ORM models for persisting Figures and Steps.

Complex dance-technique state (DancerState, Keyframes, etc.) is stored as JSON
so the data structure can evolve without requiring schema migrations.  Only the
fields needed for querying are promoted to typed columns.
"""

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class FigureModel(Base):
    """A named dance figure with its steps stored as related rows."""
    __tablename__ = "figures"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    tags = Column(JSON, nullable=True)
    total_beats = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    steps = relationship(
        "StepModel",
        back_populates="figure",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="StepModel.position",
    )

    def __repr__(self):
        return f"<FigureModel(id={self.id}, name={self.name}, total_beats={self.total_beats})>"


class StepModel(Base):
    """A single step within a figure. Dancer state is stored as JSON."""
    __tablename__ = "steps"

    id = Column(Integer, primary_key=True, index=True)
    figure_id = Column(Integer, ForeignKey("figures.id", ondelete="CASCADE"), nullable=False, index=True)
    position = Column(Integer, nullable=False)  # Ordering within the figure

    start_state = Column(JSON, nullable=False)
    end_state = Column(JSON, nullable=False)
    keyframes = Column(JSON, nullable=True)

    duration = Column(Float, nullable=False)  # Duration in beats

    figure = relationship("FigureModel", back_populates="steps")

    def __repr__(self):
        return f"<StepModel(id={self.id}, figure_id={self.figure_id}, position={self.position})>"
