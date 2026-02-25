"""Extract feet position features - determine if feet are in front or behind the body."""

import logging
import numpy as np
from typing import Dict, Tuple

from shared.skeletons.pose_data import VectorizedPoseData

logger = logging.getLogger(__name__)

# Human3.6M joint indices
H36M_PELVIS = 0
H36M_R_HIP = 1
H36M_L_HIP = 4
H36M_R_ANKLE = 3
H36M_L_ANKLE = 6


def extract_feet_position(pose_data_3d: VectorizedPoseData) -> Dict[str, np.ndarray]:
    """
    Extract feet position relative to body for all frames.
    
    Determines whether each foot (ankle) is in front of or behind the body
    by calculating the forward direction from hip geometry and using plane normals.
    This approach works regardless of the person's rotation in 3D space.
    
    Args:
        pose_data_3d: VectorizedPoseData with 3D keypoints (Human3.6M skeleton)
        
    Returns:
        Dictionary with:
        - 'feet_position': np.ndarray of shape (num_frames, 2) with bool values
          [frame, leg] where leg: 0=left, 1=right; value: 0=behind, 1=in_front
        - 'forward_vectors': np.ndarray of forward direction vectors for debugging
        - 'ankle_projections': np.ndarray of ankle projections onto forward direction
    """
    try:
        # Validate skeleton type
        if pose_data_3d.num_joints != 17:
            logger.warning(
                f"Expected Human3.6M skeleton (17 joints), got {pose_data_3d.num_joints}. "
                "Feet position extraction may not work correctly."
            )
        
        # Extract skeleton data
        skeleton = pose_data_3d.skeleton
        keypoints = skeleton.data  # Shape: (num_frames, num_joints, 3)
        
        num_frames = skeleton.num_frames
        
        # Get joint indices safely
        left_ankle_idx = H36M_L_ANKLE if skeleton.num_joints > H36M_L_ANKLE else None
        right_ankle_idx = H36M_R_ANKLE if skeleton.num_joints > H36M_R_ANKLE else None
        pelvis_idx = H36M_PELVIS if skeleton.num_joints > H36M_PELVIS else None
        left_hip_idx = H36M_L_HIP if skeleton.num_joints > H36M_L_HIP else None
        right_hip_idx = H36M_R_HIP if skeleton.num_joints > H36M_R_HIP else None
        
        if None in [left_ankle_idx, right_ankle_idx, pelvis_idx, left_hip_idx, right_hip_idx]:
            raise ValueError("Could not find required joint indices (pelvis, hips, ankles)")
        
        # Extract joint positions
        # Shape: (num_frames, 3)
        pelvis_pos = keypoints[:, pelvis_idx, :]
        left_hip_pos = keypoints[:, left_hip_idx, :]
        right_hip_pos = keypoints[:, right_hip_idx, :]
        left_ankle_pos = keypoints[:, left_ankle_idx, :]
        right_ankle_pos = keypoints[:, right_ankle_idx, :]
        
        # Calculate forward direction using hip geometry
        # Hip line vector: from left hip to right hip
        # Shape: (num_frames, 3)
        hip_line = right_hip_pos - left_hip_pos
        
        # Hip center: midpoint between hips
        # Shape: (num_frames, 3)
        hip_center = (left_hip_pos + right_hip_pos) / 2.0
        
        # Vector from hip center to pelvis (torso direction)
        # Shape: (num_frames, 3)
        torso_direction = pelvis_pos - hip_center
        
        # Forward direction: cross product of hip line with vertical up vector
        # This gives us the direction perpendicular to hips in horizontal plane
        # Shape: (num_frames, 3)
        up_vector = np.array([0, 1, 0])  # Y is typically up in 3D coordinate systems
        up_vectors = np.tile(up_vector, (num_frames, 1))
        
        # Cross product: hip_line × up = forward direction
        forward_raw = np.cross(hip_line, up_vectors)
        
        # Normalize forward vectors (handle zero vectors)
        forward_magnitudes = np.linalg.norm(forward_raw, axis=1, keepdims=True)
        forward_magnitudes = np.where(forward_magnitudes < 1e-6, 1.0, forward_magnitudes)
        forward_direction = forward_raw / forward_magnitudes
        
        # Ensure forward direction points away from torso (opposite to pelvis relative to hip center)
        # If dot product is positive, pelvis is in forward direction, so flip to point away from pelvis
        torso_dot = np.sum(forward_direction * torso_direction, axis=1, keepdims=True)
        forward_direction = np.where(torso_dot > 0, -forward_direction, forward_direction)
        
        # Calculate ankle positions relative to pelvis
        # Shape: (num_frames, 3)
        left_ankle_rel = left_ankle_pos - pelvis_pos
        right_ankle_rel = right_ankle_pos - pelvis_pos
        
        # Project ankle vectors onto forward direction
        # Positive projection = in front, negative = behind
        # Shape: (num_frames,)
        left_projection = np.sum(left_ankle_rel * forward_direction, axis=1)
        right_projection = np.sum(right_ankle_rel * forward_direction, axis=1)
        
        # Convert to binary: 1 if in front (positive projection), 0 if behind
        left_in_front = (left_projection > 0).astype(np.float32)
        right_in_front = (right_projection > 0).astype(np.float32)
        
        # Combine into output array
        # Shape: (num_frames, 2) where dim1: [0]=left, [1]=right
        feet_position = np.stack([left_in_front, right_in_front], axis=1)
        
        logger.info(
            f"Extracted feet position for {num_frames} frames using hip geometry. "
            f"Left foot in front: {np.sum(left_in_front)} frames. "
            f"Right foot in front: {np.sum(right_in_front)} frames."
        )
        
        return {
            'feet_position': feet_position,
            'forward_vectors': forward_direction,
            'ankle_projections': np.stack([left_projection, right_projection], axis=1),
        }
        
    except Exception as e:
        logger.error(f"Error extracting feet position: {e}")
        raise


def get_feet_position_summary(feet_position_result: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Get summary statistics for feet position.
    
    Args:
        feet_position_result: Output from extract_feet_position()
        
    Returns:
        Dictionary with summary statistics
    """
    feet_pos = feet_position_result['feet_position']
    
    left_pct_front = np.mean(feet_pos[:, 0]) * 100
    right_pct_front = np.mean(feet_pos[:, 1]) * 100
    both_front_pct = np.mean((feet_pos[:, 0] == 1) & (feet_pos[:, 1] == 1)) * 100
    
    # Calculate average projections if available
    result = {
        'left_foot_pct_in_front': float(left_pct_front),
        'right_foot_pct_in_front': float(right_pct_front),
        'both_feet_pct_in_front': float(both_front_pct),
    }
    
    if 'ankle_projections' in feet_position_result:
        projections = feet_position_result['ankle_projections']
        result['avg_left_projection'] = float(np.mean(projections[:, 0]))
        result['avg_right_projection'] = float(np.mean(projections[:, 1]))
    
    return result
