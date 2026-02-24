"""PostgreSQL-based worker for video processing jobs.

This worker polls the PostgreSQL jobs table for pending jobs and processes them.
"""

import time
import logging
import os
from pathlib import Path

from database import get_db_session, get_next_pending_job, update_job_status
from app.tasks import process_job

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))


def run_worker():
    """Main worker loop - polls PostgreSQL for jobs and processes them."""
    logger.info(f"Worker started. Polling every {POLL_INTERVAL} seconds...")
    logger.info(f"Database: {os.getenv('DATABASE_URL', 'not set')}")

    while True:
        db = get_db_session()
        try:
            # Get next pending job
            job = get_next_pending_job(db)

            if job:
                job_id = job['job_id']
                logger.info(f"Processing job {job_id}: {job['filename']}")

                # Mark as processing
                update_job_status(db, job_id, "processing")

                try:
                    # Process the job
                    result = process_job(
                        job_id=job_id,
                        s3_key=job['video_path'],
                        filename=job['filename'],
                        status_callback=lambda status, progress: update_progress(db, job_id, progress)
                    )

                    # Mark as completed
                    update_job_status(
                        db,
                        job_id,
                        "completed",
                    )
                    logger.info(f"Job {job_id} completed successfully")

                except Exception as e:
                    logger.error(f"Job {job_id} failed: {e}", exc_info=True)
                    update_job_status(db, job_id, "failed", error_message=str(e))

            else:
                # No pending jobs, wait before next poll
                time.sleep(POLL_INTERVAL)

        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            time.sleep(POLL_INTERVAL)
        finally:
            db.close()


def update_progress(db, job_id: str, progress: int):
    """Update job progress (currently just logs, could store in DB if needed)."""
    logger.info(f"Job {job_id} progress: {progress}%")


if __name__ == "__main__":
    run_worker()
