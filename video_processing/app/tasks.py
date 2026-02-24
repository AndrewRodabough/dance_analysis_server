"""Video processing tasks - GPU stage + analysis."""

from pathlib import Path
import json
from typing import Dict, Callable, Optional
import logging
import boto3
from botocore.client import Config
import os
import tempfile
import numpy as np

from app.analysis.pipelines.orchestrator import run_analysis_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# S3 Configuration
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "dance-videos")
USE_MOCK_ANALYSIS = os.getenv("USE_MOCK_ANALYSIS", "false").lower() == "true"

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

TEMP_DIR = Path("/workspace/temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def process_job(
    job_id: str,
    s3_key: str,
    filename: str,
    options: Dict = None,
    status_callback: Optional[Callable[[str, int], None]] = None,
) -> Dict:
    """
    Process a video analysis job (called by PostgreSQL worker).

    Args:
        job_id: Unique job identifier
        s3_key: S3 key where video is stored
        filename: Original filename
        options: Processing options
        status_callback: Optional callback(status, progress) for status updates

    Returns:
        Dict with result paths and metadata
    """
    def update_status(status: str, progress: int):
        if status_callback:
            status_callback(status, progress)
        logger.info(f"[{job_id}] {status} ({progress}%)")

    local_video_path = TEMP_DIR / f"{job_id}_input.mp4"

    try:
        logger.info(f"Processing job {job_id}: {s3_key}")

        # Download video from S3
        update_status('Downloading video from S3', 10)
        s3_client.download_file(S3_BUCKET, s3_key, str(local_video_path))

        # Run analysis pipeline
        update_status('Running analysis pipeline', 20)
        results = run_analysis_pipeline(
            local_video_path=local_video_path,
            visualization_video_path=Path("/workspace/outputs") / job_id / "video_visualization.mp4",
            keypoints_2d_output_path=Path("/workspace/outputs") / job_id / "keypoints_2d.json",
            keypoints_3d_output_path=Path("/workspace/outputs") / job_id / "keypoints_3d.json",
            update_status=update_status,
        )

        # Upload results to S3 and DB
        update_status('Uploading Results', 95)
        # TODO: Implement upload logic

        # Cleanup
        if local_video_path.exists():
            local_video_path.unlink()

        update_status('Complete', 100)
        
        return {
            'status': 'success',
        }

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
        if local_video_path.exists():
            local_video_path.unlink()
        raise