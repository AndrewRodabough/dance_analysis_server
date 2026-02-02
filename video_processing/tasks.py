"""RQ worker tasks for video processing."""

from pathlib import Path
import json
from typing import Dict
import logging
from rq import get_current_job
import boto3
from botocore.client import Config
import os

logger = logging.getLogger(__name__)

# S3 Configuration
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "dance-videos")

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


def process_video(s3_key: str, job_id: str, options: Dict = None) -> Dict:
    """
    Process video from S3: download, analyze, upload results.

    Args:
        s3_key: S3 key where video is stored
        job_id: Unique job identifier
        options: Processing options (apply_smoothing, etc.)

    Returns:
        Dict with processing results and S3 paths
    """
    job = get_current_job()

    try:
        logger.info(f"Processing video from S3: {s3_key} (job: {job_id})")

        # Step 1: Download video from S3
        job.meta['status'] = 'Downloading video from S3'
        job.meta['progress'] = 5
        job.save_meta()

        local_video_path = TEMP_DIR / f"{job_id}_input.mp4"
        logger.info(f"Downloading {s3_key} to {local_video_path}")
        s3_client.download_file(S3_BUCKET, s3_key, str(local_video_path))
        logger.info(f"Download complete: {local_video_path.stat().st_size} bytes")

        # Step 2: Run pose estimation
        job.meta['status'] = 'Extracting pose data'
        job.meta['progress'] = 30
        job.save_meta()

        from pose_estimation import pose_estimation
        from video_generation import generate_visualization

        options = options or {}
        apply_smoothing = options.get('apply_smoothing', True)

        logger.info(f"Starting pose estimation for {job_id}")
        keypoints_2d, keypoints_3d = pose_estimation(
            str(local_video_path),
            apply_smoothing=apply_smoothing
        )
        logger.info(f"Pose estimation complete: {len(keypoints_2d)} frames")

        # Step 3: Save results locally
        job.meta['status'] = 'Saving pose data'
        job.meta['progress'] = 60
        job.save_meta()

        json_2d_path = TEMP_DIR / f"{job_id}_keypoints_2d.json"
        json_3d_path = TEMP_DIR / f"{job_id}_keypoints_3d.json"

        logger.info(f"Saving keypoints to {json_2d_path} and {json_3d_path}")
        with open(json_2d_path, 'w') as f:
            json.dump(keypoints_2d.tolist(), f)

        with open(json_3d_path, 'w') as f:
            json.dump(keypoints_3d.tolist(), f)

        # Step 4: Generate visualization
        job.meta['status'] = 'Generating visualization'
        job.meta['progress'] = 70
        job.save_meta()

        viz_video_path = TEMP_DIR / f"{job_id}_visualization.mp4"
        logger.info(f"Generating visualization video to {viz_video_path}")
        generate_visualization(
            str(local_video_path),
            keypoints_2d,
            keypoints_3d,
            str(viz_video_path)
        )
        logger.info(f"Visualization complete: {viz_video_path.stat().st_size} bytes")

        # Step 5: Upload results to S3
        job.meta['status'] = 'Uploading results to S3'
        job.meta['progress'] = 90
        job.save_meta()

        logger.info(f"Uploading results to S3 for job {job_id}")
        # Upload keypoints
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

        # Upload visualization
        s3_client.upload_file(
            str(viz_video_path),
            S3_BUCKET,
            f"results/{job_id}/visualization.mp4"
        )
        logger.info(f"Uploaded visualization.mp4")

        # Step 6: Cleanup local files
        logger.info(f"Cleaning up temporary files for job {job_id}")
        local_video_path.unlink(missing_ok=True)
        json_2d_path.unlink(missing_ok=True)
        json_3d_path.unlink(missing_ok=True)
        viz_video_path.unlink(missing_ok=True)

        job.meta['status'] = 'Complete'
        job.meta['progress'] = 100
        job.save_meta()

        result = {
            'status': 'success',
            'num_frames': len(keypoints_2d),
            's3_results': {
                'keypoints_2d': f"results/{job_id}/keypoints_2d.json",
                'keypoints_3d': f"results/{job_id}/keypoints_3d.json",
                'visualization': f"results/{job_id}/visualization.mp4"
            }
        }

        logger.info(f"Job {job_id} completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        job.meta['status'] = 'Failed'
        job.meta['error'] = str(e)
        job.save_meta()

        # Cleanup on failure
        if local_video_path.exists():
            local_video_path.unlink()

        raise
