"""Pipeline Orchestrator - Coordinates all three analysis stages."""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any

try:
    from rq import Job
except ImportError:
    # For local development without RQ
    Job = None

from .pose_estimation_pipeline import run_pose_estimation_pipeline
from .feature_extraction_pipeline import run_feature_extraction_pipeline
from .report_generation_pipeline import run_report_generation_pipeline

logger = logging.getLogger(__name__)


def run_analysis_pipeline(
    job_id: str,
    s3_bucket: str,
    s3_client,
    local_keypoints_2d_path: Optional[Path] = None,
    local_keypoints_3d_path: Optional[Path] = None,
    local_video_path: Optional[Path] = None,
    visualization_video_path: Optional[Path] = None,
    redis_connection: Optional[Any] = None,
) -> Dict:
    """
    Orchestrate the complete video analysis pipeline.
    
    Chains three stages together:
    1. Pose Estimation Pipeline - Load keypoints and create VectorizedPoseData objects
    2. Feature Extraction Pipeline - Analyze pose and calculate metrics from skeleton objects
    3. Report Generation Pipeline - Create reports and upload results
    
    Automatically handles skeleton conversion using coco_w (2D) and human_17 (3D) formats.
    
    Args:
        job_id: Unique job identifier
        s3_bucket: S3 bucket name
        s3_client: Boto3 S3 client
        local_keypoints_2d_path: Path to 2D keypoints JSON
        local_keypoints_3d_path: Path to 3D keypoints JSON
        local_video_path: Optional path to video file for keypoint generation
        visualization_video_path: Optional path to visualization video file
        redis_connection: Optional Redis connection for fetching job. If provided, will fetch
                         job from job_id. If not provided, job progress tracking is disabled.
        
    Returns:
        Dictionary with complete analysis results including:
            - status: 'success' or error state
            - job_id: Job identifier
            - stage1_result: Pose estimation result with VectorizedPoseData objects
            - stage2_result: Feature extraction result
            - stage3_result: Report generation result
    """
    # Fetch job from job_id if redis connection is provided
    job = None
    if redis_connection and Job:
        try:
            job = Job.fetch(job_id, connection=redis_connection)
        except Exception as e:
            logger.warning(f"Failed to fetch job {job_id}: {e}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            logger.info(f"Starting complete analysis pipeline for job {job_id}")
            
            # ============================================================================
            # STAGE 1: Pose Estimation Pipeline
            # ============================================================================
            logger.info("=" * 80)
            logger.info("STAGE 1: POSE ESTIMATION")
            logger.info("=" * 80)
            
            if job:
                job.meta['stage'] = 'pose_estimation'
                job.meta['progress'] = 10
                job.save_meta()
            
            stage1_result = run_pose_estimation_pipeline(
                job_id=job_id,
                local_keypoints_2d_path=local_keypoints_2d_path,
                local_keypoints_3d_path=local_keypoints_3d_path,
                local_video_path=local_video_path
            )
            
            pose_data_2d = stage1_result['pose_data_2d']
            pose_data_3d = stage1_result['pose_data_3d']
            
            # ============================================================================
            # STAGE 2: Feature Extraction Pipeline
            # ============================================================================
            logger.info("=" * 80)
            logger.info("STAGE 2: FEATURE EXTRACTION & ANALYSIS")
            logger.info("=" * 80)
            
            if job:
                job.meta['stage'] = 'feature_extraction'
                job.meta['progress'] = 40
                job.save_meta()
            
            stage2_result = run_feature_extraction_pipeline(
                job_id=job_id,
                pose_data_2d=pose_data_2d,
                pose_data_3d=pose_data_3d,
            )
            
            # ============================================================================
            # STAGE 3: Report Generation Pipeline
            # ============================================================================
            logger.info("=" * 80)
            logger.info("STAGE 3: REPORT GENERATION & UPLOAD")
            logger.info("=" * 80)
            
            if job:
                job.meta['stage'] = 'report_generation'
                job.meta['progress'] = 70
                job.save_meta()
            
            stage3_result = run_report_generation_pipeline(
                job_id=job_id,
                s3_bucket=s3_bucket,
                s3_client=s3_client,
                local_video_path=local_video_path,
                visualization_video_path=visualization_video_path
            )
            
            # ============================================================================
            # Final Results
            # ============================================================================
            logger.info("=" * 80)
            logger.info("ANALYSIS COMPLETE ✓")
            logger.info("=" * 80)
            
            if job:
                job.meta['stage'] = 'complete'
                job.meta['progress'] = 100
                job.save_meta()
            
            final_result = {
                'status': 'success',
                'job_id': job_id,
                'stage1_result': stage1_result,
                'stage2_result': stage2_result,
                'stage3_result': stage3_result,
                }
            
            logger.info(f"Pipeline completed successfully for job {job_id}")
            return final_result
            
        except Exception as e:
            logger.error(f"Pipeline failed at one of the stages: {e}", exc_info=True)
            
            if job:
                job.meta['status'] = 'failed'
                job.meta['error'] = str(e)
                job.save_meta()
            
            raise
