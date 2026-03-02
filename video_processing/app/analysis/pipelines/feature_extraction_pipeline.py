"""Feature Extraction Pipeline - Stage 2: Analyze pose data and calculate metrics."""

import json
import logging
from pathlib import Path
from typing import Dict

from ..feature_extraction.feature_extraction import extract_features
from shared.skeletons.pose_data import VectorizedPoseData

logger = logging.getLogger(__name__)


def run_feature_extraction_pipeline(
    pose_data_2d,
    pose_data_3d,
    fps: float = 60.0,
) -> Dict:
    """
    Stage 2: Feature Extraction Pipeline
    
    Analyzes pose data to extract features, judge heuristics, and calculate scores.
    
    Args:
        pose_data_2d: VectorizedPoseData object with 2D pose from Stage 1
        pose_data_3d: VectorizedPoseData object with 3D pose from Stage 1
        fps: Frame rate in frames per second (default 60). Used for velocity 
             threshold normalization in 2D analysis.
        
    Returns:
        Dictionary containing:
            - features: Extracted feature data
            - pose_data_2d: Reference to the 2D pose data
            - pose_data_3d: Reference to the 3D pose data
            - features_path: Path to saved features.json (if temp_dir provided)
    """
    logger.info(f"[STAGE 2] Feature Extraction Pipeline: Analyzing pose data (fps={fps})")
    
    try:
        # Validate input types
        if not isinstance(pose_data_3d, VectorizedPoseData):
            raise TypeError(f"pose_data_3d must be VectorizedPoseData, got {type(pose_data_3d)}")
        if not isinstance(pose_data_2d, VectorizedPoseData):
            raise TypeError(f"pose_data_2d must be VectorizedPoseData, got {type(pose_data_2d)}")
        
        logger.debug(f"Received 2D pose data: {pose_data_2d.num_frames} frames, {pose_data_2d.num_joints} joints")
        logger.debug(f"Received 3D pose data: {pose_data_3d.num_frames} frames, {pose_data_3d.num_joints} joints")
        
        # Step 1: Extract Features from 2D pose data (side-view analysis)
        logger.info("Extracting features from 2D pose data (side-view)")
        features = extract_features(pose_data_2d=pose_data_2d, use_2d_analysis=True, fps=fps)
        logger.info(f"✓ Features extracted from {pose_data_2d.num_frames} frames")
                
        logger.info(f"[STAGE 2] ✓ Feature extraction complete")
        return features
        
    except Exception as e:
        logger.error(f"[STAGE 2] ✗ Feature extraction failed: {e}", exc_info=True)
        raise
