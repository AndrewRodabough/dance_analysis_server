"""Dance database model."""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.database import Base


class DanceStyle(str, PyEnum):
    """Dance style enumeration."""

    # Latin Dances
    SAMBA = "samba"
    CHA_CHA = "cha_cha"
    RUMBA = "rumba"
    PASO_DOBLE = "paso_doble"
    JIVE = "jive"

    # Standard Dances
    WALTZ = "waltz"
    TANGO = "tango"
    VIENNESE_WALTZ = "viennese_waltz"
    FOXTROT = "foxtrot"
    QUICKSTEP = "quickstep"

    # American Smooth Dances
    AMERICAN_WALTZ = "american_waltz"
    AMERICAN_TANGO = "american_tango"
    AMERICAN_FOXTROT = "american_foxtrot"
    AMERICAN_VIENNESE_WALTZ = "american_viennese_waltz"

    # American Rhythm Dances
    AMERICAN_CHA_CHA = "american_cha_cha"
    AMERICAN_RUMBA = "american_rumba"
    SWING = "swing"
    BOLERO = "bolero"
    MAMBO = "mambo"


class DanceCategory:
    """Categorization mapping for dance styles."""

    LATIN = {
        DanceStyle.SAMBA,
        DanceStyle.CHA_CHA,
        DanceStyle.RUMBA,
        DanceStyle.PASO_DOBLE,
        DanceStyle.JIVE,
    }

    STANDARD = {
        DanceStyle.WALTZ,
        DanceStyle.TANGO,
        DanceStyle.VIENNESE_WALTZ,
        DanceStyle.FOXTROT,
        DanceStyle.QUICKSTEP,
    }

    AMERICAN_SMOOTH = {
        DanceStyle.AMERICAN_WALTZ,
        DanceStyle.AMERICAN_TANGO,
        DanceStyle.AMERICAN_FOXTROT,
        DanceStyle.AMERICAN_VIENNESE_WALTZ,
    }

    AMERICAN_RHYTHM = {
        DanceStyle.AMERICAN_CHA_CHA,
        DanceStyle.AMERICAN_RUMBA,
        DanceStyle.SWING,
        DanceStyle.BOLERO,
        DanceStyle.MAMBO,
    }


class Dance(Base):
    """A dance definition with tempo, meter, and style."""

    __tablename__ = "dances"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tempo = Column(Integer, nullable=False)  # BPM
    meter = Column(String(10), nullable=False)  # e.g. "3/4", "4/4"
    style = Column(
        SQLEnum(
            DanceStyle,
            values_callable=lambda enum: [e.value for e in enum],
            name="dance_style",
        ),
        nullable=False,
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Dance(id={self.id}, style={self.style}, tempo={self.tempo})>"
