"""RQ worker tasks for video processing - GPU stage + analysis."""

from pathlib import Path
import json
from typing import Dict
import logging
from rq import get_current_job
import boto3
from botocore.client import Config
import os
import tempfile
import numpy as np

from app.analysis.feedback_pipeline import run_feedback_pipeline

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


def extract_keypoints(s3_key: str, job_id: str, options: Dict = None) -> Dict:
    """
    GPU Stage: Extract 2D and 3D keypoints from video.
    
    Downloads video from S3, runs pose estimation, saves keypoints to S3,
    then enqueues the analysis stage.

    Args:
        s3_key: S3 key where video is stored
        job_id: Unique job identifier
        options: Processing options (apply_smoothing, etc.)

    Returns:
        Dict with keypoints paths and metadata
    """
    job = get_current_job()
    local_video_path = TEMP_DIR / f"{job_id}_input.mp4"

    try:
        logger.info(f"[GPU] Extracting keypoints from S3: {s3_key} (job: {job_id})")

        # Step 1: Download video from S3
        job.meta['status'] = 'Downloading video from S3'
        job.meta['progress'] = 10
        job.save_meta()

        logger.info(f"Downloading {s3_key} to {local_video_path}")
        s3_client.download_file(S3_BUCKET, s3_key, str(local_video_path))
        logger.info(f"Download complete: {local_video_path.stat().st_size} bytes")

        # Step 2: Run pose estimation (or use mock data)
        job.meta['status'] = 'Extracting pose data (GPU)'
        job.meta['progress'] = 30
        job.save_meta()

        options = options or {}
        apply_smoothing = options.get('apply_smoothing', True)

        if USE_MOCK_ANALYSIS:
            logger.info(f"Using MOCK keypoints for job {job_id}")
            # Load pre-generated keypoints from test_outputs
            test_outputs = Path("/workspace/test_outputs")
            
            with open(test_outputs / "keypoints_2d.json", 'r') as f:
                keypoints_2d = np.array(json.load(f))
            
            with open(test_outputs / "keypoints_3d.json", 'r') as f:
                keypoints_3d = np.array(json.load(f))
            
            logger.info(f"Loaded mock keypoints: {len(keypoints_2d)} frames")
        else:
            logger.info(f"Running GPU pose estimation for {job_id}")
            from app.analysis.pose_estimation.pose_estimation import pose_estimation
            
            keypoints_2d, keypoints_3d, keypoints_3d_before, scores = pose_estimation(
                str(local_video_path),
                apply_smoothing=apply_smoothing
            )
            logger.info(f"Pose estimation complete: {len(keypoints_2d)} frames")

        # Step 3: Save keypoints locally
        job.meta['status'] = 'Saving keypoints'
        job.meta['progress'] = 60
        job.save_meta()

        json_2d_path = TEMP_DIR / f"{job_id}_keypoints_2d.json"
        json_3d_path = TEMP_DIR / f"{job_id}_keypoints_3d.json"

        logger.info(f"Saving keypoints to {json_2d_path} and {json_3d_path}")
        with open(json_2d_path, 'w') as f:
            # Convert to numpy array first to handle nested structures, then to list
            data_2d = np.asarray(keypoints_2d).tolist()
            json.dump(data_2d, f)

        with open(json_3d_path, 'w') as f:
            data_3d = np.asarray(keypoints_3d).tolist()
            json.dump(data_3d, f)

        # Step 4: Run analysis sequentially on this worker
        job.meta['status'] = 'Running analysis'
        job.meta['progress'] = 80
        job.save_meta()

        analysis_result = generate_feedback(
            job_id=job_id,
            s3_video_key=s3_key,
            num_frames=len(keypoints_2d),
            local_keypoints_2d_path=json_2d_path,
            local_keypoints_3d_path=json_3d_path,
            local_video_path=local_video_path
        )

        # Step 5: Cleanup local files
        logger.info(f"Cleaning up temporary files for job {job_id}")
        local_video_path.unlink(missing_ok=True)
        json_2d_path.unlink(missing_ok=True)
        json_3d_path.unlink(missing_ok=True)

        job.meta['status'] = 'Complete'
        job.meta['progress'] = 100
        job.save_meta()

        result = {
            'status': 'success',
            'num_frames': len(keypoints_2d),
            's3_results': analysis_result.get('s3_results') if isinstance(analysis_result, dict) else None
        }

        logger.info(f"[GPU] Job {job_id} completed")
        return result

    except Exception as e:
        logger.error(f"Error extracting keypoints: {e}", exc_info=True)
        job.meta['status'] = 'Failed'
        job.meta['error'] = str(e)
        job.save_meta()

        # Cleanup on failure
        if local_video_path.exists():
            local_video_path.unlink()

        raise


def generate_feedback(
    job_id: str,
    s3_video_key: str,
    num_frames: int,
    local_keypoints_2d_path: Path = None,
    local_keypoints_3d_path: Path = None,
    local_video_path: Path = None
) -> Dict:
    """
    Generate visualization and feedback from keypoints.
    
    Orchestrates the complete analysis pipeline through modular components.

    Args:
        job_id: Unique job identifier
        s3_video_key: S3 key where original video is stored
        num_frames: Number of frames in video
        local_keypoints_2d_path: Path to 2D keypoints JSON file
        local_keypoints_3d_path: Path to 3D keypoints JSON file
        local_video_path: Optional path to video file

    Returns:
        Dict with final results and S3 paths
    """
    return run_feedback_pipeline(
        job_id=job_id,
        s3_bucket=S3_BUCKET,
        s3_client=s3_client,
        local_keypoints_2d_path=local_keypoints_2d_path,
        local_keypoints_3d_path=local_keypoints_3d_path,
        local_video_path=local_video_path,
    )
