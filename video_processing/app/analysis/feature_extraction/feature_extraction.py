"""Extract features from pose data."""

import logging
from typing import Dict, Any

from shared.skeletons.pose_data import VectorizedPoseData
from .leg_straightening_timing import analyze_cha_cha_walk, log_analysis_summary

logger = logging.getLogger(__name__)


def extract_features(pose_data_3d: VectorizedPoseData) -> Dict[str, Any]:
    """
    Extract features from 3D pose data.
    
    Args:
        pose_data_3d: VectorizedPoseData object with 3D keypoints
        
    Returns:
        Dictionary containing extracted features
    """
    try:        
        results = analyze_cha_cha_walk(pose_data_3d)
        log_analysis_summary(results)

        """
        features = {
            "walks_straightening": results,
        }
        """
        
        return {}
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise
