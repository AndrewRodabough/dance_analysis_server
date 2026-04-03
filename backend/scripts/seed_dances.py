"""Seed canonical dance styles into the database.

Run once after migrations, or re-run after schema changes (idempotent).

Usage:
    python scripts/seed_dances.py
    docker-compose exec backend python scripts/seed_dances.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import SessionLocal
from app.models.dance import Dance, DanceStyle

# Canonical competition tempos (BPM) and time signatures for each style.
# Sources: WDSF / NDCA syllabus guidelines.
DANCE_SEED_DATA = [
    # --- International Standard ---
    {"style": DanceStyle.WALTZ,           "tempo": 90,  "meter": "3/4"},
    {"style": DanceStyle.TANGO,           "tempo": 132, "meter": "4/4"},
    {"style": DanceStyle.VIENNESE_WALTZ,  "tempo": 180, "meter": "3/4"},
    {"style": DanceStyle.FOXTROT,         "tempo": 120, "meter": "4/4"},
    {"style": DanceStyle.QUICKSTEP,       "tempo": 200, "meter": "4/4"},

    # --- International Latin ---
    {"style": DanceStyle.SAMBA,           "tempo": 104, "meter": "2/4"},
    {"style": DanceStyle.CHA_CHA,         "tempo": 128, "meter": "4/4"},
    {"style": DanceStyle.RUMBA,           "tempo": 104, "meter": "4/4"},
    {"style": DanceStyle.PASO_DOBLE,      "tempo": 124, "meter": "2/4"},
    {"style": DanceStyle.JIVE,            "tempo": 176, "meter": "4/4"},

    # --- American Smooth ---
    {"style": DanceStyle.AMERICAN_WALTZ,          "tempo": 90,  "meter": "3/4"},
    {"style": DanceStyle.AMERICAN_TANGO,          "tempo": 132, "meter": "4/4"},
    {"style": DanceStyle.AMERICAN_FOXTROT,        "tempo": 120, "meter": "4/4"},
    {"style": DanceStyle.AMERICAN_VIENNESE_WALTZ, "tempo": 174, "meter": "3/4"},

    # --- American Rhythm ---
    {"style": DanceStyle.AMERICAN_CHA_CHA, "tempo": 124, "meter": "4/4"},
    {"style": DanceStyle.AMERICAN_RUMBA,   "tempo": 100, "meter": "4/4"},
    {"style": DanceStyle.SWING,            "tempo": 140, "meter": "4/4"},
    {"style": DanceStyle.BOLERO,           "tempo": 96,  "meter": "4/4"},
    {"style": DanceStyle.MAMBO,            "tempo": 188, "meter": "4/4"},
]


def seed_dances() -> None:
    db = SessionLocal()
    try:
        existing_styles = {row.style for row in db.query(Dance.style).all()}

        inserted = 0
        for data in DANCE_SEED_DATA:
            if data["style"] in existing_styles:
                print(f"  skip  {data['style'].value} (already exists)")
                continue
            db.add(Dance(**data))
            inserted += 1
            print(f"  insert {data['style'].value}  {data['tempo']} BPM  {data['meter']}")

        db.commit()
        print(f"\nDone — {inserted} dance(s) inserted, {len(existing_styles)} already present.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_dances()
