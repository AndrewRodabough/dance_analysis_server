"""Pipeline Orchestrator - Coordinates all three analysis stages."""

import logging
import tempfile
import json
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
    keypoints_2d_output_path: Optional[Path] = None,
    keypoints_3d_output_path: Optional[Path] = None,
    update_status: Optional[Callable[[str, int], None]] = None,
) -> Dict:
    """
    Orchestrate the complete video analysis pipeline.
    
    Chains three stages together:
    1. Pose Estimation Pipeline - Load keypoints and create VectorizedPoseData objects
    2. Feature Extraction Pipeline - Analyze pose and calculate metrics from skeleton objects
    3. Report Generation Pipeline - Create reports and upload results
    
    Automatically handles skeleton conversion using coco_17 (2D) and human_17 (3D) formats.
    
    Args:
        local_keypoints_2d_path: Path to 2D keypoints JSON
        local_keypoints_3d_path: Path to 3D keypoints JSON
        local_video_path: Optional path to video file for keypoint generation
        visualization_video_path: Optional path to visualization video file
        keypoints_2d_output_path: Optional path to save 2D keypoints JSON after processing
        keypoints_3d_output_path: Optional path to save 3D keypoints JSON after processing
        update_status: Optional callback function(status: str, progress: int) for progress tracking
        
    Returns:
        Dictionary with complete analysis results including:
            - status: 'success' or error state
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
            
            # Save keypoints if output paths provided
            if keypoints_2d_output_path:
                keypoints_2d_output_path = Path(keypoints_2d_output_path)
                keypoints_2d_output_path.parent.mkdir(parents=True, exist_ok=True)
                keypoints_2d_json = pose_data_2d.skeleton.data.tolist()
                with open(keypoints_2d_output_path, 'w') as f:
                    json.dump(keypoints_2d_json, f)
                logger.info(f"Saved 2D keypoints to {keypoints_2d_output_path}")
            
            if keypoints_3d_output_path:
                keypoints_3d_output_path = Path(keypoints_3d_output_path)
                keypoints_3d_output_path.parent.mkdir(parents=True, exist_ok=True)
                keypoints_3d_json = pose_data_3d.skeleton.data.tolist()
                with open(keypoints_3d_output_path, 'w') as f:
                    json.dump(keypoints_3d_json, f)
                logger.info(f"Saved 3D keypoints to {keypoints_3d_output_path}")
            
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
