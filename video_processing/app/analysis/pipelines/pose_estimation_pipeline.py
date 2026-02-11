"""Pose Estimation Pipeline - Stage 1: Load keypoints and create pose objects."""

import logging
import json
from pathlib import Path
from typing import Optional, Dict
import numpy as np

from shared.skeletons.skeleton_registry import SkeletonRegistry
from shared.skeletons.skeleton import VectorizedSkeleton
from shared.skeletons.pose_data import VectorizedPoseData
from ..pose_estimation.pose_estimation_motionbert import pose_estimation

logger = logging.getLogger(__name__)

# Define skeleton config paths
SKELETON_CONFIGS_DIR = Path(__file__).parent.parent.parent.parent.parent / "configs" / "skeletons"


def _load_skeleton_from_config(skeleton_name: str) -> VectorizedSkeleton:
    """
    Load a skeleton configuration from JSON and create a VectorizedSkeleton.
    
    Args:
        skeleton_name: Name of the skeleton (e.g., 'coco_w', 'human_17')
        
    Returns:
        VectorizedSkeleton instance initialized with the configuration
    """
    config_path = SKELETON_CONFIGS_DIR / f"{skeleton_name}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Skeleton config not found: {config_path}")
    
    # Load skeleton config from JSON
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    joint_names = config['joints']
    bones = config['bones']
    
    logger.debug(f"Loaded skeleton config '{skeleton_name}': {len(joint_names)} joints, {len(bones)} bones")
    
    # Create and return VectorizedSkeleton
    skeleton = VectorizedSkeleton(joint_names, bones)
    return skeleton


def _load_keypoints_from_json_file(keypoints_path: Path) -> np.ndarray:
    """
    Load keypoint data from a JSON file.
    
    Expected format: List of frames, each containing keypoint data.
    
    Args:
        keypoints_path: Path to the keypoints JSON file
        
    Returns:
        Numpy array of shape (Frames, Joints, Channels)
    """
    with open(keypoints_path, 'r') as f:
        data = json.load(f)
    
    # Convert to numpy array
    # Assuming data is a list of lists representing (frames, joints, channels)
    keypoints = np.array(data, dtype=np.float32)
    
    logger.debug(f"Loaded keypoints from {keypoints_path}: shape {keypoints.shape}")
    return keypoints


def run_pose_estimation_pipeline(
    job_id: str,
    local_keypoints_2d_path: Optional[Path] = None,
    local_keypoints_3d_path: Optional[Path] = None,
    local_video_path: Optional[Path] = None,
) -> Dict:
    """
    Stage 1: Pose Estimation Pipeline
    
    Loads pre-extracted keypoints (from GPU stage) and creates skeleton/pose objects
    for downstream analysis. Returns VectorizedPoseData objects wrapping the skeletons.
    
    For JSON files: Uses coco_w skeleton for 2D, human_17 for 3D.
    For video: Generates keypoints and wraps in coco_w (2D) and human_17 (3D) skeletons.
    
    Args:
        job_id: Unique job identifier
        local_keypoints_2d_path: Path to 2D keypoints JSON file
        local_keypoints_3d_path: Path to 3D keypoints JSON file
        local_video_path: Path to video file
        
    Returns:
        Dictionary containing:
            - pose_data_2d: VectorizedPoseData object for 2D pose
            - pose_data_3d: VectorizedPoseData object for 3D pose
    """
    logger.info(f"[STAGE 1] Pose Estimation Pipeline: Loading keypoints for job {job_id}")
    
    try:
        if local_keypoints_2d_path and local_keypoints_2d_path.exists() \
            and local_keypoints_3d_path and local_keypoints_3d_path.exists():
            
            # Load keypoints from JSON files
            logger.info(f"Loading keypoints from JSON files")
            logger.debug(f"  2D: {local_keypoints_2d_path}")
            logger.debug(f"  3D: {local_keypoints_3d_path}")
            
            # Load raw keypoint data
            keypoints_2d_raw = _load_keypoints_from_json_file(local_keypoints_2d_path)
            keypoints_3d_raw = _load_keypoints_from_json_file(local_keypoints_3d_path)
            
            # Load skeleton configurations (for now: hardcoded coco_w for 2D, human_17 for 3D)
            logger.debug("Loading skeleton configs: coco_w for 2D, human_17 for 3D")
            skeleton_2d = _load_skeleton_from_config('coco_w')
            skeleton_3d = _load_skeleton_from_config('human_17')
            
            # Load keypoint data into skeletons
            skeleton_2d.load_data(keypoints_2d_raw)
            skeleton_3d.load_data(keypoints_3d_raw)
            
            # For now, create confidence arrays (all 1.0 if not in the JSON yet)
            # TODO: Extract confidence from JSON when available
            num_frames_2d = skeleton_2d.num_frames
            num_joints_2d = skeleton_2d.num_joints
            confidence_2d = np.ones((num_frames_2d, num_joints_2d), dtype=np.float32)
            
            num_frames_3d = skeleton_3d.num_frames
            num_joints_3d = skeleton_3d.num_joints
            confidence_3d = np.ones((num_frames_3d, num_joints_3d), dtype=np.float32)
            
            # Create VectorizedPoseData objects
            pose_data_2d = VectorizedPoseData(skeleton_2d, confidence_2d)
            pose_data_3d = VectorizedPoseData(skeleton_3d, confidence_3d)
            
            logger.info(f"✓ Loaded {num_frames_2d} frames of 2D pose data ({num_joints_2d} joints)")
            logger.info(f"✓ Loaded {num_frames_3d} frames of 3D pose data ({num_joints_3d} joints)")
            
        elif local_video_path and local_video_path.exists():
            # Generate keypoints from video
            logger.info(f"Generating keypoints from video: {local_video_path}")
            keypoints_2d_raw, keypoints_3d_raw, _ = pose_estimation(local_video_path, apply_smoothing=False)
            
            # Load skeleton configurations  
            logger.debug("Loading skeleton configs: coco_w for 2D, human_17 for 3D")
            skeleton_2d = _load_skeleton_from_config('coco_w')
            skeleton_3d = _load_skeleton_from_config('human_17')
            
            # Load keypoint data into skeletons
            skeleton_2d.load_data(keypoints_2d_raw)
            skeleton_3d.load_data(keypoints_3d_raw)
            
            # Create confidence arrays (extract from video generation or set to 1.0)
            # TODO: Extract actual confidence scores from video processing
            num_frames_2d = skeleton_2d.num_frames
            num_joints_2d = skeleton_2d.num_joints
            confidence_2d = np.ones((num_frames_2d, num_joints_2d), dtype=np.float32)
            
            num_frames_3d = skeleton_3d.num_frames
            num_joints_3d = skeleton_3d.num_joints
            confidence_3d = np.ones((num_frames_3d, num_joints_3d), dtype=np.float32)
            
            # Create VectorizedPoseData objects
            pose_data_2d = VectorizedPoseData(skeleton_2d, confidence_2d)
            pose_data_3d = VectorizedPoseData(skeleton_3d, confidence_3d)
            
            logger.info(f"✓ Generated {num_frames_2d} frames of 2D pose data ({num_joints_2d} joints)")
            logger.info(f"✓ Generated {num_frames_3d} frames of 3D pose data ({num_joints_3d} joints)")
            
        else:
            raise FileNotFoundError("No valid keypoints or video file found for pose estimation")
        
        logger.info(f"[STAGE 1] ✓ Pose estimation complete")
        
        return {
            'pose_data_2d': pose_data_2d,
            'pose_data_3d': pose_data_3d,
        }
        
    except Exception as e:
        logger.error(f"[STAGE 1] ✗ Pose estimation failed: {e}", exc_info=True)
        raise
