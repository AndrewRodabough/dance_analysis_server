"""RQ worker tasks for video processing - GPU stage + analysis."""

from pathlib import Path
import json
from typing import Dict, Optional
import logging
from rq import get_current_job
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


def generate_feedback(
    job_id: str,
    s3_video_key: str,
    local_video_path: Path
) -> Dict:
    """
    Generate visualization and feedback from video.
    
    Orchestrates the complete analysis pipeline through three stages:
    1. Pose Estimation - Load keypoints and create pose objects
    2. Feature Extraction - Analyze pose and calculate metrics
    3. Report Generation - Create reports and upload results

    Args:
        job_id: Unique job identifier
        s3_video_key: S3 key where original video is stored
        local_video_path: Path where video should be downloaded

    Returns:
        Dict with final results and S3 paths
    """
    
    # Normalize to Path in case a string path is passed in from RQ.
    local_video_path = Path(local_video_path)
    # Create parent directory if it doesn't exist
    local_video_path.parent.mkdir(parents=True, exist_ok=True)

    job = get_current_job()
    job.meta['status'] = 'Downloading video from S3'
    job.meta['progress'] = 10
    job.save_meta()

    logger.info(f"Downloading {s3_video_key} to {local_video_path}")
    s3_client.download_file(S3_BUCKET, s3_video_key, str(local_video_path))
    logger.info(f"Download complete: {local_video_path.stat().st_size} bytes")

    return run_analysis_pipeline(
        job_id=job_id,
        s3_bucket=S3_BUCKET,
        s3_client=s3_client,
        local_video_path=local_video_path,
        redis_connection=job.connection,
        visualization_video_path=Path("/workspace/outputs") / job_id / "video_visualization.mp4"
    )