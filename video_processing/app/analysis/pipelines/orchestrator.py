"""Pipeline Orchestrator - Coordinates all three analysis stages."""

import logging
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional, Any

from .pose_estimation_pipeline import run_pose_estimation_pipeline
from .feature_extraction_pipeline import run_feature_extraction_pipeline
from .report_generation_pipeline import run_report_generation_pipeline

logger = logging.getLogger(__name__)


def run_analysis_pipeline(
    local_keypoints_2d_path: Optional[Path] = None,
    local_keypoints_3d_path: Optional[Path] = None,
    local_video_path: Optional[Path] = None,
    visualization_video_path: Optional[Path] = None,
    update_status: Optional[Callable[[str, int], None]] = None,
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
   
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # ============================================================================
            # STAGE 1: Pose Estimation Pipeline
            # ============================================================================

            if update_status:
                update_status('pose_estimation', 21)
            
            stage1_result = run_pose_estimation_pipeline(
                local_keypoints_2d_path=local_keypoints_2d_path,
                local_keypoints_3d_path=local_keypoints_3d_path,
                local_video_path=local_video_path
            )
            
            pose_data_2d = stage1_result['pose_data_2d']
            pose_data_3d = stage1_result['pose_data_3d']
            
            # ============================================================================
            # STAGE 2: Feature Extraction Pipeline
            # ============================================================================
            if update_status:
                update_status('feature_extraction', 70)
            
            stage2_result = run_feature_extraction_pipeline(
                pose_data_2d=pose_data_2d,
                pose_data_3d=pose_data_3d,
            )
            
            # ============================================================================
            # STAGE 3: Report Generation Pipeline
            # ============================================================================
        
            if update_status:
                update_status('report_generation', 90)
            
            stage3_result = run_report_generation_pipeline(
                pose_data_2d,
                pose_data_3d,
                local_video_path=local_video_path,
                visualization_video_path=visualization_video_path
            )
            
            # ============================================================================
            # Results
            # ============================================================================
            
            final_result = {
                'status': 'success',
                'stage1_result': stage1_result,
                'stage2_result': stage2_result,
                'stage3_result': stage3_result,
                }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
