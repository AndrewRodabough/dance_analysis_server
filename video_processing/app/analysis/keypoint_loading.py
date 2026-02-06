"""Load keypoints and create skeleton objects."""

import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np

from shared.skeletons.skeleton_registry import SkeletonRegistry
from shared.skeletons.skeleton import VectorizedSkeleton
from shared.skeletons.pose_data import VectorizedPoseData

logger = logging.getLogger(__name__)


def load_skeleton_registry():
    """Load skeleton configurations from shared configs."""
    registry = SkeletonRegistry()
    config_dir = Path(__file__).parent.parent.parent.parent / "shared" / "configs" / "skeletons"
    
    # Load COCO-WholeBody for 2D
    coco_w_path = config_dir / "coco_w.json"
    registry.load_from_json(str(coco_w_path))
    logger.info(f"Loaded COCO-WholeBody config from {coco_w_path}")
    
    # Load Human3.6M for 3D
    human_17_path = config_dir / "human_17.json"
    registry.load_from_json(str(human_17_path))
    logger.info(f"Loaded Human3.6M config from {human_17_path}")
    
    return registry


def normalize_keypoints_format(keypoints: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Normalize keypoints to (frames, people, joints, channels) format.
    
    Handles multiple input formats:
    - (frames, joints, channels) -> single person -> (frames, 1, joints, channels)
    - (frames, people, joints, channels) -> multi-person (unchanged)
    
    Args:
        keypoints: Keypoints array in various formats
        
    Returns:
        Tuple of (normalized_keypoints, num_people)
    """
    if keypoints.ndim == 3:
        # Single person format: (frames, joints, channels)
        logger.info(f"Detected single-person format: {keypoints.shape}")
        keypoints = keypoints[:, np.newaxis, :, :]  # Insert person dimension
        num_people = 1
    elif keypoints.ndim == 4:
        # Multi-person format: (frames, people, joints, channels)
        logger.info(f"Detected multi-person format: {keypoints.shape}")
        num_people = keypoints.shape[1]
    else:
        raise ValueError(f"Invalid keypoints shape: {keypoints.shape}. Expected 3D or 4D array.")
    
    logger.info(f"Normalized keypoints shape: {keypoints.shape} ({num_people} people)")
    return keypoints, num_people


def extract_person_data(keypoints_all: np.ndarray, person_idx: int) -> np.ndarray:
    """
    Extract data for a specific person from multi-person keypoints.
    
    Args:
        keypoints_all: Multi-person keypoints (frames, people, joints, channels)
        person_idx: Index of person to extract
        
    Returns:
        Single-person keypoints (frames, joints, channels)
    """
    return keypoints_all[:, person_idx, :, :]


def load_keypoints_from_json(
    json_2d_path: Path,
    json_3d_path: Path
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load 2D and 3D keypoints from JSON files.
    
    Supports both single-person and multi-person formats:
    - Single: (frames, joints, channels)
    - Multi: (frames, people, joints, channels)
    
    Args:
        json_2d_path: Path to 2D keypoints JSON file
        json_3d_path: Path to 3D keypoints JSON file
        
    Returns:
        Tuple of (keypoints_2d, keypoints_3d, num_people)
        All normalized to (frames, people, joints, channels) format
    """
    with open(json_2d_path, 'r') as f:
        keypoints_2d = np.array(json.load(f))
    
    with open(json_3d_path, 'r') as f:
        keypoints_3d = np.array(json.load(f))
    
    # Normalize both to (frames, people, joints, channels) format
    keypoints_2d, num_people_2d = normalize_keypoints_format(keypoints_2d)
    keypoints_3d, num_people_3d = normalize_keypoints_format(keypoints_3d)
    
    if num_people_2d != num_people_3d:
        logger.warning(f"Person count mismatch: 2D has {num_people_2d}, 3D has {num_people_3d}")
        num_people = min(num_people_2d, num_people_3d)
        # Trim both to match minimum
        keypoints_2d = keypoints_2d[:, :num_people, :, :]
        keypoints_3d = keypoints_3d[:, :num_people, :, :]
    else:
        num_people = num_people_2d
    
    logger.info(f"Loaded keypoints: 2D {keypoints_2d.shape}, 3D {keypoints_3d.shape}, {num_people} people")
    return keypoints_2d, keypoints_3d, num_people


def create_skeleton_objects(
    keypoints_2d: np.ndarray,
    keypoints_3d: np.ndarray,
    person_idx: int = 0
) -> Tuple[VectorizedSkeleton, VectorizedSkeleton]:
    """
    Create skeleton objects from keypoint data.
    
    Args:
        keypoints_2d: 2D keypoints array (frames, people, joints, channels) or (frames, joints, channels)
        keypoints_3d: 3D keypoints array (frames, people, joints, channels) or (frames, joints, channels)
        person_idx: Index of person to create skeleton for (default: 0, first person)
        
    Returns:
        Tuple of (skeleton_2d, skeleton_3d)
    """
    registry = load_skeleton_registry()
    
    # Normalize to multi-person format if needed
    keypoints_2d_norm, _ = normalize_keypoints_format(keypoints_2d)
    keypoints_3d_norm, _ = normalize_keypoints_format(keypoints_3d)
    
    # Extract specific person
    kp_2d_person = extract_person_data(keypoints_2d_norm, person_idx)
    kp_3d_person = extract_person_data(keypoints_3d_norm, person_idx)
    
    logger.info(f"Creating skeletons for person {person_idx}: 2D {kp_2d_person.shape}, 3D {kp_3d_person.shape}")
    
    # Load COCO-WholeBody (133 joints for 2D)
    coco_w_joints, coco_w_bones = registry.get("coco_w")
    skeleton_2d = VectorizedSkeleton(coco_w_joints, coco_w_bones)
    skeleton_2d.load_data(kp_2d_person)
    logger.info(f"Created 2D skeleton (COCO-WholeBody): {skeleton_2d.num_joints} joints, {skeleton_2d.num_frames} frames")
    
    # Load Human3.6M (17 joints for 3D)
    human_17_joints, human_17_bones = registry.get("human_17")
    skeleton_3d = VectorizedSkeleton(human_17_joints, human_17_bones)
    skeleton_3d.load_data(kp_3d_person)
    logger.info(f"Created 3D skeleton (Human3.6M): {skeleton_3d.num_joints} joints, {skeleton_3d.num_frames} frames")
    
    return skeleton_2d, skeleton_3d


def create_all_skeleton_objects(
    keypoints_2d: np.ndarray,
    keypoints_3d: np.ndarray,
) -> Dict[int, Tuple[VectorizedSkeleton, VectorizedSkeleton]]:
    """
    Create skeleton objects for all people in the keypoints.
    
    Args:
        keypoints_2d: 2D keypoints array (frames, people, joints, channels) or (frames, joints, channels)
        keypoints_3d: 3D keypoints array (frames, people, joints, channels) or (frames, joints, channels)
        
    Returns:
        Dictionary mapping person_idx to (skeleton_2d, skeleton_3d) tuples
    """
    # Normalize to multi-person format if needed
    keypoints_2d_norm, num_people = normalize_keypoints_format(keypoints_2d)
    keypoints_3d_norm, _ = normalize_keypoints_format(keypoints_3d)
    
    all_skeletons = {}
    for person_idx in range(num_people):
        logger.info(f"Creating skeletons for person {person_idx}/{num_people}")
        all_skeletons[person_idx] = create_skeleton_objects(
            keypoints_2d_norm,
            keypoints_3d_norm,
            person_idx=person_idx
        )
    
    return all_skeletons


def create_pose_data_objects(
    skeleton_2d: VectorizedSkeleton,
    skeleton_3d: VectorizedSkeleton,
    confidence_2d_value: float = 0.95,
    confidence_3d_value: float = 0.95
) -> Tuple[VectorizedPoseData, VectorizedPoseData]:
    """
    Create PoseData objects with confidence scores.
    
    Args:
        skeleton_2d: 2D skeleton object
        skeleton_3d: 3D skeleton object
        confidence_2d_value: Default confidence value for 2D (0-1)
        confidence_3d_value: Default confidence value for 3D (0-1)
        
    Returns:
        Tuple of (pose_data_2d, pose_data_3d)
    """
    # Create confidence arrays
    confidence_2d = np.ones((skeleton_2d.num_frames, skeleton_2d.num_joints), dtype=np.float32) * confidence_2d_value
    confidence_3d = np.ones((skeleton_3d.num_frames, skeleton_3d.num_joints), dtype=np.float32) * confidence_3d_value
    
    pose_data_2d = VectorizedPoseData(skeleton_2d, confidence_2d)
    pose_data_3d = VectorizedPoseData(skeleton_3d, confidence_3d)
    
    logger.info(f"Created PoseData objects with confidence scores")
    return pose_data_2d, pose_data_3d
