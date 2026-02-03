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
    
    Load keypoints from S3, load video and generate visualization (optional),
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
            logger.info(f"Generating feedback for job {job_id}")



            # ------- Load Keypoints ------- #
            job.meta['status'] = 'Loading keypoints'
            job.meta['progress'] = 10
            job.save_meta()

            json_2d_path = local_keypoints_2d_path
            json_3d_path = local_keypoints_3d_path

            with open(json_2d_path, 'r') as f:
                keypoints_2d = np.array(json.load(f))
            
            with open(json_3d_path, 'r') as f:
                keypoints_3d = np.array(json.load(f))

            logger.info(f"Loaded keypoints: {len(keypoints_2d)} frames")



            # ------- Generate Video (optional) ------- #
            video_path = local_video_path
            viz_video_path = temp_path / "visualization.mp4"
            
            if video_path:
                # Generate visualization video
                job.meta['status'] = 'Generating visualization video'
                job.meta['progress'] = 20
                job.save_meta()

                # TODO: Generate Visualization

                logger.info(f"Generated visualization video")
            


            # ------- Feature Extraction ------- #
            job.meta['status'] = 'Extracting Features'
            job.meta['progress'] = 30
            job.save_meta()

            features_path = temp_path / "features.txt"
            
            features = '' # TODO: Implement actual feature extraction

            with open(features_path, 'w') as f:
                json.dump(features, f, indent=2)

            logger.info(f"Extracted Features")



            # ------- Judge Heuristics ------- #
            job.meta['status'] = 'Judging Heuristics'
            job.meta['progress'] = 60
            job.save_meta()

            judge_path = temp_path / "judge.json"

            judge = '' # TODO: Implement actual feature extraction

            with open(judge_path, 'w') as f:
                json.dump(judge, f, indent=2)

            logger.info(f"Judged heuristics")



            # ------- Score Calculation ------- #
            job.meta['status'] = 'Calculating Scores'
            job.meta['progress'] = 80
            job.save_meta()

            scores_path = temp_path / "scores.json"

            scores = '' # TODO: Implement actual score calculation
            
            with open(scores_path, 'w') as f:
                json.dump(scores, f, indent=2)
            
            logger.info(f"Calculated scores")



            # ------- Generate Report ------- #
            job.meta['status'] = 'Generating Report'
            job.meta['progress'] = 90
            job.save_meta()

            feedback_path = temp_path / "feedback.txt"

            feedback_text = '' # TODO: Implement actual feedback generation 
            
            # write feedback to file
            with open(feedback_path, 'w') as f:
                f.write(feedback_text)
            
            logger.info(f"Generated Report: {len(feedback_text)} characters")



            # ------- Upload Results to S3 (including keypoints) ------- #
            job.meta['status'] = 'Uploading results to S3'
            job.meta['progress'] = 95
            job.save_meta()

            logger.info(f"Uploading final results to S3 for job {job_id}")

            # Upload files with retry logic for urllib3 header parsing issues
            def safe_upload(local_path, s3_key, description):
                """Upload file to S3, ignoring urllib3 header parsing errors."""
                try:
                    s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
                    logger.info(f"Uploaded {description}")
                except Exception as e:
                    # Check if it's the urllib3 header parsing issue (file actually uploaded)
                    if "HeaderParsingError" in str(type(e).__name__) or "HeaderParsingError" in str(e):
                        logger.warning(f"Header parsing warning for {description} (likely uploaded successfully)")
                    else:
                        logger.error(f"Failed to upload {description}: {e}")
                        raise

            safe_upload(json_2d_path, f"results/{job_id}/keypoints_2d.json", "keypoints_2d.json")
            safe_upload(json_3d_path, f"results/{job_id}/keypoints_3d.json", "keypoints_3d.json")
            
            if viz_video_path.exists():
                safe_upload(viz_video_path, f"results/{job_id}/visualization.mp4", "visualization.mp4")

            safe_upload(feedback_path, f"results/{job_id}/feedback.txt", "feedback.txt")
            safe_upload(scores_path, f"results/{job_id}/scores.json", "scores.json")
            safe_upload(features_path, f"results/{job_id}/features.txt", "features.txt")
            safe_upload(judge_path, f"results/{job_id}/judge.json", "judge.json")

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
