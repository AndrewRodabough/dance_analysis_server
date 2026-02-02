"""RQ worker tasks for video processing - GPU stage (pose estimation only)."""

from pathlib import Path
import json
from typing import Dict
import logging
from rq import get_current_job, Queue
from redis import Redis
import boto3
from botocore.client import Config
import os

logger = logging.getLogger(__name__)

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

# Redis connection for enqueuing next stage
redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))

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
            import numpy as np
            test_outputs = Path("/workspace/test_outputs")
            
            with open(test_outputs / "keypoints_2d.json", 'r') as f:
                keypoints_2d = np.array(json.load(f))
            
            with open(test_outputs / "keypoints_3d.json", 'r') as f:
                keypoints_3d = np.array(json.load(f))
            
            logger.info(f"Loaded mock keypoints: {len(keypoints_2d)} frames")
        else:
            logger.info(f"Running GPU pose estimation for {job_id}")
            from pose_estimation import pose_estimation
            
            keypoints_2d, keypoints_3d = pose_estimation(
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
            json.dump(keypoints_2d.tolist(), f)

        with open(json_3d_path, 'w') as f:
            json.dump(keypoints_3d.tolist(), f)

        # Step 4: Upload keypoints to S3
        job.meta['status'] = 'Uploading keypoints to S3'
        job.meta['progress'] = 80
        job.save_meta()

        logger.info(f"Uploading keypoints to S3 for job {job_id}")
        s3_client.upload_file(
            str(json_2d_path),
            S3_BUCKET,
            f"results/{job_id}/keypoints_2d.json"
        )
        logger.info(f"Uploaded keypoints_2d.json")

        s3_client.upload_file(
            str(json_3d_path),
            S3_BUCKET,
            f"results/{job_id}/keypoints_3d.json"
        )
        logger.info(f"Uploaded keypoints_3d.json")

        # Step 5: Cleanup local files
        logger.info(f"Cleaning up temporary files for job {job_id}")
        local_video_path.unlink(missing_ok=True)
        json_2d_path.unlink(missing_ok=True)
        json_3d_path.unlink(missing_ok=True)

        # Step 6: Enqueue analysis stage (runs on backend CPU workers)
        job.meta['status'] = 'Keypoints extracted, queuing analysis'
        job.meta['progress'] = 90
        job.save_meta()

        analysis_queue = Queue('analysis', connection=redis_conn)
        analysis_job = analysis_queue.enqueue(
            'app.tasks.generate_feedback',
            job_id=job_id,
            s3_video_key=s3_key,
            num_frames=len(keypoints_2d),
            job_timeout='10m'
        )
        
        logger.info(f"Enqueued analysis job {analysis_job.id} for {job_id}")

        job.meta['status'] = 'Keypoints complete, analysis queued'
        job.meta['progress'] = 100
        job.meta['analysis_job_id'] = analysis_job.id
        job.save_meta()

        result = {
            'status': 'keypoints_extracted',
            'num_frames': len(keypoints_2d),
            'analysis_job_id': analysis_job.id,
            's3_keypoints': {
                'keypoints_2d': f"results/{job_id}/keypoints_2d.json",
                'keypoints_3d': f"results/{job_id}/keypoints_3d.json"
            }
        }

        logger.info(f"[GPU] Job {job_id} completed, analysis stage queued")
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
