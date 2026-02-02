"""RQ worker tasks for backend analysis - CPU stage (visualization & feedback)."""

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


def generate_feedback(job_id: str, s3_video_key: str, num_frames: int) -> Dict:
    """
    CPU Stage: Generate visualization and feedback from keypoints.
    
    Downloads keypoints from S3, downloads original video, generates visualization
    and feedback, then uploads results to S3.

    Args:
        job_id: Unique job identifier
        s3_video_key: S3 key where original video is stored
        num_frames: Number of frames in video

    Returns:
        Dict with final results and S3 paths
    """
    job = get_current_job()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            logger.info(f"[CPU] Generating feedback for job {job_id}")

            # Step 1: Download keypoints from S3
            job.meta['status'] = 'Downloading keypoints from S3'
            job.meta['progress'] = 10
            job.save_meta()

            json_2d_path = temp_path / "keypoints_2d.json"
            json_3d_path = temp_path / "keypoints_3d.json"

            logger.info(f"Downloading keypoints for job {job_id}")
            s3_client.download_file(
                S3_BUCKET,
                f"results/{job_id}/keypoints_2d.json",
                str(json_2d_path)
            )
            s3_client.download_file(
                S3_BUCKET,
                f"results/{job_id}/keypoints_3d.json",
                str(json_3d_path)
            )

            # Load keypoints
            with open(json_2d_path, 'r') as f:
                keypoints_2d = np.array(json.load(f))
            
            with open(json_3d_path, 'r') as f:
                keypoints_3d = np.array(json.load(f))

            logger.info(f"Loaded keypoints: {len(keypoints_2d)} frames")

            # Step 2: Download original video
            job.meta['status'] = 'Downloading original video'
            job.meta['progress'] = 30
            job.save_meta()

            video_path = temp_path / f"{job_id}_input.mp4"
            logger.info(f"Downloading original video from {s3_video_key}")
            s3_client.download_file(S3_BUCKET, s3_video_key, str(video_path))
            logger.info(f"Downloaded video: {video_path.stat().st_size} bytes")

            # Step 3: Generate visualization video
            job.meta['status'] = 'Generating visualization video'
            job.meta['progress'] = 50
            job.save_meta()

            viz_video_path = temp_path / "visualization.mp4"
            
            # TODO: Import actual video generation function
            # from app.analysis.video_generation import generate_visualization
            # For now, create a placeholder
            logger.info(f"Generating visualization for {job_id}")
            
            # Placeholder: Copy original video (replace with actual visualization)
            import shutil
            shutil.copy(video_path, viz_video_path)
            logger.info(f"Visualization complete (placeholder): {viz_video_path.stat().st_size} bytes")

            # Step 4: Generate feedback text
            job.meta['status'] = 'Generating feedback'
            job.meta['progress'] = 70
            job.save_meta()

            feedback_path = temp_path / "feedback.txt"
            
            # TODO: Implement actual feedback generation
            feedback_text = f"""Dance Analysis Feedback - Job {job_id}
            
Frames analyzed: {num_frames}
Keypoints extracted: {keypoints_2d.shape[1]} joints per frame

Analysis Summary:
- Pose detection successful
- 3D reconstruction complete
- Visualization generated

Next steps:
- Review visualization video
- Check keypoint data for accuracy
"""
            
            with open(feedback_path, 'w') as f:
                f.write(feedback_text)
            
            logger.info(f"Generated feedback: {len(feedback_text)} characters")

            # Step 5: Generate scores (placeholder)
            job.meta['status'] = 'Calculating scores'
            job.meta['progress'] = 85
            job.save_meta()

            scores_path = temp_path / "scores.json"
            scores = {
                'overall_score': 85.5,
                'confidence': 0.92,
                'frames_analyzed': num_frames,
                'quality_metrics': {
                    'pose_detection_rate': 0.98,
                    'tracking_stability': 0.95
                }
            }
            
            with open(scores_path, 'w') as f:
                json.dump(scores, f, indent=2)
            
            logger.info(f"Generated scores")

            # Step 6: Upload all results to S3
            job.meta['status'] = 'Uploading results to S3'
            job.meta['progress'] = 95
            job.save_meta()

            logger.info(f"Uploading final results to S3 for job {job_id}")
            
            s3_client.upload_file(
                str(viz_video_path),
                S3_BUCKET,
                f"results/{job_id}/visualization.mp4"
            )
            logger.info(f"Uploaded visualization.mp4")

            s3_client.upload_file(
                str(feedback_path),
                S3_BUCKET,
                f"results/{job_id}/feedback.txt"
            )
            logger.info(f"Uploaded feedback.txt")

            s3_client.upload_file(
                str(scores_path),
                S3_BUCKET,
                f"results/{job_id}/scores.json"
            )
            logger.info(f"Uploaded scores.json")

            # Mark complete
            job.meta['status'] = 'Complete'
            job.meta['progress'] = 100
            job.save_meta()

            result = {
                'status': 'success',
                'num_frames': num_frames,
                's3_results': {
                    'keypoints_2d': f"results/{job_id}/keypoints_2d.json",
                    'keypoints_3d': f"results/{job_id}/keypoints_3d.json",
                    'visualization': f"results/{job_id}/visualization.mp4",
                    'feedback': f"results/{job_id}/feedback.txt",
                    'scores': f"results/{job_id}/scores.json"
                }
            }

            logger.info(f"[CPU] Job {job_id} analysis complete")
            return result

        except Exception as e:
            logger.error(f"Error generating feedback: {e}", exc_info=True)
            job.meta['status'] = 'Failed'
            job.meta['error'] = str(e)
            job.save_meta()
            raise
