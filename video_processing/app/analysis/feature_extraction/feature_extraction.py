"""Extract features from pose data."""

import logging
from typing import Dict, Any, Optional

from shared.skeletons.pose_data import VectorizedPoseData
from .leg_straightening_timing import (
    analyze_cha_cha_walk, 
    analyze_cha_cha_walk_2d,
    log_analysis_summary, 
    probe_data_ranges,
    probe_data_ranges_2d
)

logger = logging.getLogger(__name__)


def extract_features(
    pose_data_3d: Optional[VectorizedPoseData] = None,
    pose_data_2d: Optional[VectorizedPoseData] = None,
    use_2d_analysis: bool = False
) -> Dict[str, Any]:
    """
    Extract features from pose data (2D or 3D).
    
    Args:
        pose_data_3d: VectorizedPoseData object with 3D keypoints (optional)
        pose_data_2d: VectorizedPoseData object with 2D keypoints (optional)
        use_2d_analysis: If True, use 2D side-view analysis instead of 3D
        
    Returns:
        Dictionary containing extracted features
    """
    try:
        if use_2d_analysis:
            if pose_data_2d is None:
                raise ValueError("pose_data_2d must be provided when use_2d_analysis=True")
            
            logger.info("Running 2D side-view cha-cha walk analysis...")
            probe_data_ranges_2d(pose_data_2d)
            results = analyze_cha_cha_walk_2d(pose_data_2d)
            log_analysis_summary(results)
            features = {
                "2d_walks": results,
            }
        else:
            if pose_data_3d is None:
                raise ValueError("pose_data_3d must be provided when use_2d_analysis=False")
            
            logger.info("Running 3D cha-cha walk analysis...")
            probe_data_ranges(pose_data_3d)
            results = analyze_cha_cha_walk(pose_data_3d)
            log_analysis_summary(results)

            features = {
                "3d_walks": results,
            }

        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise
