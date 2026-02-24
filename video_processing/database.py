"""Database connection for video processing worker."""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://danceuser:dancepass@postgres:5432/dancedb"
)

# Create engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session():
    """Get a new database session. Caller must close it."""
    return SessionLocal()


def get_next_pending_job(db):
    """Get the next pending job (FIFO)."""
    result = db.execute(text("""
        SELECT job_id, filename, video_path, user_id
        FROM jobs
        WHERE status = 'pending'
        ORDER BY created_at ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    """))
    row = result.fetchone()
    if row:
        return {
            'job_id': row[0],
            'filename': row[1],
            'video_path': row[2],
            'user_id': row[3]
        }
    return None


def update_job_status(db, job_id: str, status: str, error_message: str = None,
                      result_path: str = None, data_path: str = None):
    """Update job status in database."""
    from datetime import datetime
    now = datetime.utcnow()

    if status == "processing":
        db.execute(text("""
            UPDATE jobs
            SET status = :status, started_at = :now
            WHERE job_id = :job_id
        """), {"status": status, "now": now, "job_id": job_id})
    elif status in ["completed", "failed"]:
        db.execute(text("""
            UPDATE jobs
            SET status = :status, completed_at = :now, error_message = :error,
                result_path = :result_path, data_path = :data_path
            WHERE job_id = :job_id
        """), {
            "status": status,
            "now": now,
            "error": error_message,
            "result_path": result_path,
            "data_path": data_path,
            "job_id": job_id
        })
    else:
        db.execute(text("""
            UPDATE jobs
            SET status = :status
            WHERE job_id = :job_id
        """), {"status": status, "job_id": job_id})

    db.commit()
