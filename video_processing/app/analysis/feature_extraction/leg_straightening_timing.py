"""Analyze leg straightening timing for rumba/cha-cha walk using hips, knees, and ankles."""

import logging
import numpy as np
from enum import Enum
from typing import Dict, Tuple, List

from shared.skeletons.pose_data import VectorizedPoseData

logger = logging.getLogger(__name__)

# Analysis thresholds
VELOCITY_THRESHOLD = 0.05  # Below this = "planted" foot
HIP_DISTANCE_THRESHOLD = 0.1  # Below this = weight fully transferred
KNEE_STRAIGHT_ANGLE = np.pi * 0.9  # Above this = "straight" leg (162 degrees)

class WalkingState(Enum):
    RELEASE = 0
    PASSING = 1
    EXTENSION = 2
    ARRIVAL = 3
    COMPLETED = 4

STARTING_STATE = WalkingState.RELEASE


def compute_weight_transfer_offsets(
    pose_data_3d: VectorizedPoseData,
) -> np.ndarray:
    """
    Compute X/Z offsets from the mid-hip to each ankle.

    Returns:
        Array of shape (frames, 2, 2) where:
        - axis 0 = frame
        - axis 1 = leg (0=left, 1=right)
        - axis 2 = offset components (x, z)
    """

    left_hip_idx = pose_data_3d.skeleton.name_to_idx["left_hip"]
    left_ankle_idx = pose_data_3d.skeleton.name_to_idx["left_ankle"]
    right_hip_idx = pose_data_3d.skeleton.name_to_idx["right_hip"]
    right_ankle_idx = pose_data_3d.skeleton.name_to_idx["right_ankle"]

    if pose_data_3d.skeleton.data is None:
        raise ValueError("pose_data_3d must have loaded skeleton data")

    skeleton_data = pose_data_3d.skeleton.data
    if skeleton_data.ndim != 3 or skeleton_data.shape[2] < 3:
        raise ValueError("pose_data_3d.skeleton.data must have shape (frames, joints, 3)")

    mid_hip = (skeleton_data[:, left_hip_idx, :] + skeleton_data[:, right_hip_idx, :]) * 0.5
    left_offset = mid_hip - skeleton_data[:, left_ankle_idx, :]
    right_offset = mid_hip - skeleton_data[:, right_ankle_idx, :]

    offsets = np.zeros((skeleton_data.shape[0], 2, 2), dtype=np.float32)
    offsets[:, 0, 0] = left_offset[:, 0]
    offsets[:, 0, 1] = left_offset[:, 2]
    offsets[:, 1, 0] = right_offset[:, 0]
    offsets[:, 1, 1] = right_offset[:, 2]

    return offsets

def compute_ankle_to_ankle_distance_xz(
    pose_data_3d: VectorizedPoseData,
) -> np.ndarray:
    """
    Compute X/Z ankle-to-ankle distance per frame.

    Returns:
        Array of shape (frames,) with distance in the XZ plane.
    """
    left_ankle_idx = pose_data_3d.skeleton.name_to_idx["left_ankle"]
    right_ankle_idx = pose_data_3d.skeleton.name_to_idx["right_ankle"]

    if pose_data_3d.skeleton.data is None:
        raise ValueError("pose_data_3d must have loaded skeleton data")

    skeleton_data = pose_data_3d.skeleton.data
    if skeleton_data.ndim != 3 or skeleton_data.shape[2] < 3:
        raise ValueError("pose_data_3d.skeleton.data must have shape (frames, joints, 3)")

    left_ankle = skeleton_data[:, left_ankle_idx, :]
    right_ankle = skeleton_data[:, right_ankle_idx, :]
    diff = left_ankle - right_ankle

    return np.sqrt(diff[:, 0] ** 2 + diff[:, 2] ** 2).astype(np.float32)

def extract_leg_straightening_timing(pose_data_3d: VectorizedPoseData) -> Dict[str, np.ndarray]:

    angles = pose_data_3d.get_weighted_bone_angles(threshold=0)
    velocities = pose_data_3d.get_weighted_joint_velocities(threshold=0)
    weight_transfer = compute_weight_transfer_offsets(pose_data_3d)
    ankle_distance = compute_ankle_to_ankle_distance_xz(pose_data_3d)

    ankle_velocities = velocities[:, [pose_data_3d.skeleton.name_to_idx["left_ankle"], pose_data_3d.skeleton.name_to_idx["right_ankle"]], :]
    knee_angles = angles[:, [pose_data_3d.skeleton.name_to_idx["left_knee"], pose_data_3d.skeleton.name_to_idx["right_knee"]]]

    current_state = STARTING_STATE

    for frame_idx in range(pose_data_3d.num_frames):
        pass