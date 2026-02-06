"""Upload analysis results to S3."""

import logging
from pathlib import Path
from typing import Callable, Dict, Any

logger = logging.getLogger(__name__)


def safe_upload(
    s3_client,
    local_path: Path,
    s3_key: str,
    s3_bucket: str,
    description: str = None
) -> bool:
    """
    Upload file to S3, ignoring urllib3 header parsing errors.
    
    Args:
        s3_client: Boto3 S3 client
        local_path: Path to local file
        s3_key: S3 key/path for the file
        s3_bucket: S3 bucket name
        description: Description of file for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client.upload_file(str(local_path), s3_bucket, s3_key)
        logger.info(f"Uploaded {description or s3_key}")
        return True
    except Exception as e:
        # Check if it's the urllib3 header parsing issue (file actually uploaded)
        if "HeaderParsingError" in str(type(e).__name__) or "HeaderParsingError" in str(e):
            logger.warning(f"Header parsing warning for {description} (likely uploaded successfully)")
            return True
        else:
            logger.error(f"Failed to upload {description}: {e}")
            return False


def upload_results(
    s3_client,
    job_id: str,
    s3_bucket: str,
    keypoints_2d_path: Path,
    keypoints_3d_path: Path,
    feedback_path: Path,
    scores_path: Path,
    features_path: Path,
    judge_path: Path,
    viz_video_path: Path = None
) -> Dict[str, str]:
    """
    Upload all analysis results to S3.
    
    Args:
        s3_client: Boto3 S3 client
        job_id: Job identifier
        s3_bucket: S3 bucket name
        keypoints_2d_path: Path to 2D keypoints file
        keypoints_3d_path: Path to 3D keypoints file
        feedback_path: Path to feedback file
        scores_path: Path to scores file
        features_path: Path to features file
        judge_path: Path to judge file
        viz_video_path: Optional path to visualization video
        
    Returns:
        Dictionary with S3 paths for all uploaded files
    """
    logger.info(f"Uploading results to S3 for job {job_id}")
    
    s3_results = {}
    
    # Upload keypoints
    s3_key = f"results/{job_id}/keypoints_2d.json"
    if safe_upload(s3_client, keypoints_2d_path, s3_key, s3_bucket, "keypoints_2d.json"):
        s3_results['keypoints_2d'] = s3_key
    
    s3_key = f"results/{job_id}/keypoints_3d.json"
    if safe_upload(s3_client, keypoints_3d_path, s3_key, s3_bucket, "keypoints_3d.json"):
        s3_results['keypoints_3d'] = s3_key
    
    # Upload visualization if it exists
    if viz_video_path and viz_video_path.exists():
        s3_key = f"results/{job_id}/visualization.mp4"
        if safe_upload(s3_client, viz_video_path, s3_key, s3_bucket, "visualization.mp4"):
            s3_results['visualization'] = s3_key
    
    # Upload analysis results
    s3_key = f"results/{job_id}/feedback.txt"
    if safe_upload(s3_client, feedback_path, s3_key, s3_bucket, "feedback.txt"):
        s3_results['feedback'] = s3_key
    
    s3_key = f"results/{job_id}/scores.json"
    if safe_upload(s3_client, scores_path, s3_key, s3_bucket, "scores.json"):
        s3_results['scores'] = s3_key
    
    s3_key = f"results/{job_id}/features.txt"
    if safe_upload(s3_client, features_path, s3_key, s3_bucket, "features.txt"):
        s3_results['features'] = s3_key
    
    s3_key = f"results/{job_id}/judge.json"
    if safe_upload(s3_client, judge_path, s3_key, s3_bucket, "judge.json"):
        s3_results['judge'] = s3_key
    
    logger.info(f"Uploaded {len(s3_results)} files to S3")
    return s3_results
