"""
Anatomical constraint filter for 3D pose estimation.
Fixes depth ambiguity issues where joints bend in impossible directions.
"""
import numpy as np
from typing import List, Tuple, Optional
from enum import IntEnum


class BodyJoint(IntEnum):
    """COCO Wholebody joint indices for main body"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle at p2 formed by vectors p1->p2 and p2->p3.
    Returns angle in degrees.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 180.0  # Degenerate case
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    # Calculate angle
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def detect_person_orientation(keypoints_3d: np.ndarray, scores: np.ndarray, 
                              confidence_threshold: float = 0.3) -> str:
    """
    Detect if person is facing toward or away from camera using shoulder orientation.
    
    Uses the cross product of shoulder vectors to determine body orientation:
    - If shoulders point "forward" (toward camera), person is facing camera
    - If shoulders point "backward" (away from camera), person is facing away
    
    Args:
        keypoints_3d: 3D keypoints array, shape (133, 3) or (N, 133, 3)
        scores: confidence scores
        confidence_threshold: minimum confidence required
    
    Returns:
        'forward' if facing camera, 'backward' if facing away, 'unknown' if uncertain
    """
    # Remove batch dimension if present
    if len(keypoints_3d.shape) == 3:
        keypoints_3d = keypoints_3d[0]
    if len(scores.shape) == 2:
        scores = scores[0]
    
    # Need shoulders and hips to determine orientation
    left_shoulder_idx = BodyJoint.LEFT_SHOULDER
    right_shoulder_idx = BodyJoint.RIGHT_SHOULDER
    left_hip_idx = BodyJoint.LEFT_HIP
    right_hip_idx = BodyJoint.RIGHT_HIP
    
    # Check if we have confident detections
    required_joints = [left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx]
    if not all(scores[j] > confidence_threshold for j in required_joints):
        return 'unknown'
    
    # Get joint positions
    left_shoulder = keypoints_3d[left_shoulder_idx]
    right_shoulder = keypoints_3d[right_shoulder_idx]
    left_hip = keypoints_3d[left_hip_idx]
    right_hip = keypoints_3d[right_hip_idx]
    
    # Vector from left to right shoulder
    shoulder_vec = right_shoulder - left_shoulder
    
    # Vector from shoulder midpoint to hip midpoint (torso direction)
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    torso_vec = hip_mid - shoulder_mid
    
    # Cross product gives the normal to the torso plane
    # This normal points either forward (toward camera) or backward (away)
    normal = np.cross(shoulder_vec, torso_vec)
    
    # The z-component tells us orientation:
    # - Positive z: normal points toward camera -> person facing camera
    # - Negative z: normal points away from camera -> person facing away
    
    if abs(normal[2]) < 0.01:
        # Person is very side-on, can't determine orientation reliably
        return 'unknown'
    
    return 'forward' if normal[2] > 0 else 'backward'


def detect_impossible_elbow(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray,
                           is_left: bool = True, facing: str = 'forward') -> bool:
    """
    Detect if elbow is bending in an anatomically impossible direction.
    
    Human elbows can only bend in one direction (towards the body's front).
    We check if the elbow is bending backwards using cross product to determine
    which side of the arm plane the wrist is on.
    
    Args:
        shoulder: 3D position of shoulder joint
        elbow: 3D position of elbow joint
        wrist: 3D position of wrist joint
        is_left: True for left arm, False for right arm
        facing: 'forward' (toward camera), 'backward' (away from camera), or 'unknown'
    
    Returns True if elbow bend is impossible (needs correction).
    """
    # Vector from shoulder to elbow
    upper_arm = elbow - shoulder
    # Vector from elbow to wrist
    forearm = wrist - elbow
    
    # Cross product gives normal to the plane of the arm
    cross = np.cross(upper_arm, forearm)
    
    # The expected direction of the cross product depends on:
    # 1. Which arm (left/right)
    # 2. Which direction person is facing
    
    if facing == 'forward':
        # Person facing camera
        # Left elbow: cross.z should be positive (bending inward)
        # Right elbow: cross.z should be negative (bending inward)
        if is_left:
            return cross[2] < -0.01  # Left elbow bending backwards (impossible)
        else:
            return cross[2] > 0.01   # Right elbow bending backwards (impossible)
    
    elif facing == 'backward':
        # Person facing away from camera
        # The signs are reversed!
        # Left elbow: cross.z should be negative (bending inward/forward relative to body)
        # Right elbow: cross.z should be positive (bending inward/forward relative to body)
        if is_left:
            return cross[2] > 0.01   # Left elbow bending wrong way
        else:
            return cross[2] < -0.01  # Right elbow bending wrong way
    
    else:
        # Unknown orientation - use angle-based check as fallback
        # If elbow angle is too extreme (< 30° or > 170°), it's likely wrong
        angle = calculate_angle(shoulder, elbow, wrist)
        return angle < 30.0 or angle > 170.0


def detect_impossible_knee(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray,
                          is_left: bool = True, facing: str = 'forward') -> bool:
    """
    Detect if knee is bending in an anatomically impossible direction.
    
    Human knees can only bend backwards (towards the body's back).
    Similar to elbow but opposite direction.
    
    Args:
        hip: 3D position of hip joint
        knee: 3D position of knee joint
        ankle: 3D position of ankle joint
        is_left: True for left leg, False for right leg
        facing: 'forward' (toward camera), 'backward' (away from camera), or 'unknown'
    
    Returns True if knee bend is impossible (needs correction).
    """
    # Vector from hip to knee
    thigh = knee - hip
    # Vector from knee to ankle
    shin = ankle - knee
    
    # Cross product gives normal to the plane of the leg
    cross = np.cross(thigh, shin)
    
    if facing == 'forward':
        # Person facing camera
        # Knees should bend backward (away from camera)
        # Left knee: cross.z should be negative (bending backward)
        # Right knee: cross.z should be positive (bending backward)
        if is_left:
            return cross[2] > 0.01   # Left knee bending forward (impossible)
        else:
            return cross[2] < -0.01  # Right knee bending forward (impossible)
    
    elif facing == 'backward':
        # Person facing away from camera
        # Knees still bend backward relative to body, but signs are reversed
        # Left knee: cross.z should be positive
        # Right knee: cross.z should be negative
        if is_left:
            return cross[2] < -0.01  # Left knee bending wrong way
        else:
            return cross[2] > 0.01   # Right knee bending wrong way
    
    else:
        # Unknown orientation - use angle-based check
        angle = calculate_angle(hip, knee, ankle)
        return angle < 30.0 or angle > 170.0


def reflect_joint_across_limb_axis(parent: np.ndarray, joint: np.ndarray, child: np.ndarray) -> np.ndarray:
    """
    Reflect joint across the line connecting parent to child.
    
    This is the geometrically correct way to fix impossible bends.
    The joint moves PERPENDICULAR to the limb axis, not along the camera Z-axis.
    
    For example, for an elbow:
    - parent = shoulder
    - joint = elbow (to be corrected)
    - child = wrist
    
    The elbow is reflected across the shoulder-wrist line, moving it to the 
    opposite side of the arm axis. This preserves bone lengths and moves
    the joint in the anatomically correct direction.
    
    Args:
        parent: Parent joint (shoulder/hip)
        joint: Joint to correct (elbow/knee)  
        child: Child joint (wrist/ankle)
    
    Returns:
        Corrected joint position, reflected across the parent-child axis
    """
    # Direction vector from parent to child (the limb axis)
    limb_axis = child - parent
    limb_length = np.linalg.norm(limb_axis)
    
    if limb_length < 1e-6:
        # Degenerate case: parent and child at same position
        return joint.copy()
    
    # Normalize the limb axis
    limb_axis_unit = limb_axis / limb_length
    
    # Vector from parent to the joint
    parent_to_joint = joint - parent
    
    # Project the joint position onto the limb axis
    # This gives us the closest point on the limb axis to the joint
    projection_length = np.dot(parent_to_joint, limb_axis_unit)
    projection_point = parent + projection_length * limb_axis_unit
    
    # The perpendicular vector from the limb axis to the joint
    # This is what we need to flip
    perpendicular = joint - projection_point
    
    # Reflect: the corrected joint is on the opposite side of the limb axis
    # at the same distance
    corrected = projection_point - perpendicular
    
    return corrected


def flip_joint_depth(parent: np.ndarray, joint: np.ndarray, child: np.ndarray, 
                     preserve_distances: bool = True) -> np.ndarray:
    """
    DEPRECATED: Use reflect_joint_across_limb_axis instead.
    
    This function incorrectly flipped only the Z coordinate, which moves
    the joint parallel to the camera axis rather than perpendicular to the limb.
    
    Kept for backwards compatibility but now just calls the correct function.
    """
    return reflect_joint_across_limb_axis(parent, joint, child)
    
    return corrected


def apply_anatomical_constraints(keypoints_3d: np.ndarray, 
                                 scores: np.ndarray,
                                 confidence_threshold: float = 0.3,
                                 verbose: bool = False) -> np.ndarray:
    """
    Apply anatomical constraints to fix impossible joint angles.
    
    Args:
        keypoints_3d: 3D keypoints array, shape (133, 3) or (N, 133, 3)
        scores: confidence scores, shape (133,) or (N, 133)
        confidence_threshold: minimum confidence to process a joint
        verbose: print debug information
    
    Returns:
        Corrected keypoints_3d array with same shape as input
    """
    # Handle batch dimension
    is_batch = len(keypoints_3d.shape) == 3
    if not is_batch:
        keypoints_3d = keypoints_3d[np.newaxis, ...]
        scores = scores[np.newaxis, ...]
    
    corrected = keypoints_3d.copy()
    
    for person_idx in range(keypoints_3d.shape[0]):
        kps = corrected[person_idx]
        conf = scores[person_idx]
        
        # Detect person orientation first
        facing = detect_person_orientation(kps, conf, confidence_threshold)
        
        if verbose:
            print(f"Person {person_idx}: Detected facing {facing}")
        
        corrections_made = []
        
        # Check left elbow
        if (conf[BodyJoint.LEFT_SHOULDER] > confidence_threshold and
            conf[BodyJoint.LEFT_ELBOW] > confidence_threshold and
            conf[BodyJoint.LEFT_WRIST] > confidence_threshold):
            
            shoulder = kps[BodyJoint.LEFT_SHOULDER]
            elbow = kps[BodyJoint.LEFT_ELBOW]
            wrist = kps[BodyJoint.LEFT_WRIST]
            
            if detect_impossible_elbow(shoulder, elbow, wrist, is_left=True, facing=facing):
                corrected[person_idx][BodyJoint.LEFT_ELBOW] = flip_joint_depth(shoulder, elbow, wrist)
                corrections_made.append("LEFT_ELBOW")
        
        # Check right elbow
        if (conf[BodyJoint.RIGHT_SHOULDER] > confidence_threshold and
            conf[BodyJoint.RIGHT_ELBOW] > confidence_threshold and
            conf[BodyJoint.RIGHT_WRIST] > confidence_threshold):
            
            shoulder = kps[BodyJoint.RIGHT_SHOULDER]
            elbow = kps[BodyJoint.RIGHT_ELBOW]
            wrist = kps[BodyJoint.RIGHT_WRIST]
            
            if detect_impossible_elbow(shoulder, elbow, wrist, is_left=False, facing=facing):
                corrected[person_idx][BodyJoint.RIGHT_ELBOW] = flip_joint_depth(shoulder, elbow, wrist)
                corrections_made.append("RIGHT_ELBOW")
        
        # Check left knee
        if (conf[BodyJoint.LEFT_HIP] > confidence_threshold and
            conf[BodyJoint.LEFT_KNEE] > confidence_threshold and
            conf[BodyJoint.LEFT_ANKLE] > confidence_threshold):
            
            hip = kps[BodyJoint.LEFT_HIP]
            knee = kps[BodyJoint.LEFT_KNEE]
            ankle = kps[BodyJoint.LEFT_ANKLE]
            
            if detect_impossible_knee(hip, knee, ankle, is_left=True, facing=facing):
                corrected[person_idx][BodyJoint.LEFT_KNEE] = flip_joint_depth(hip, knee, ankle)
                corrections_made.append("LEFT_KNEE")
        
        # Check right knee
        if (conf[BodyJoint.RIGHT_HIP] > confidence_threshold and
            conf[BodyJoint.RIGHT_KNEE] > confidence_threshold and
            conf[BodyJoint.RIGHT_ANKLE] > confidence_threshold):
            
            hip = kps[BodyJoint.RIGHT_HIP]
            knee = kps[BodyJoint.RIGHT_KNEE]
            ankle = kps[BodyJoint.RIGHT_ANKLE]
            
            if detect_impossible_knee(hip, knee, ankle, is_left=False, facing=facing):
                corrected[person_idx][BodyJoint.RIGHT_KNEE] = flip_joint_depth(hip, knee, ankle)
                corrections_made.append("RIGHT_KNEE")
        
        if verbose and corrections_made:
            print(f"Person {person_idx}: Corrected {', '.join(corrections_made)}")
    
    # Remove batch dimension if input didn't have it
    if not is_batch:
        corrected = corrected[0]
    
    return corrected


def apply_constraints_to_sequence(keypoints_3d_list: List[np.ndarray],
                                  scores_list: List[np.ndarray],
                                  confidence_threshold: float = 0.3,
                                  temporal_window: int = 5,
                                  verbose: bool = False) -> List[np.ndarray]:
    """
    Apply anatomical constraints to a sequence of frames with temporal consistency.
    
    Args:
        keypoints_3d_list: List of 3D keypoint arrays
        scores_list: List of confidence score arrays
        confidence_threshold: minimum confidence to process a joint
        temporal_window: number of frames to consider for temporal consistency
        verbose: print debug information
    
    Returns:
        List of corrected 3D keypoint arrays
    """
    corrected_list = []
    correction_history = {}  # Track which joints were corrected in recent frames
    
    for frame_idx, (kps, scores) in enumerate(zip(keypoints_3d_list, scores_list)):
        if verbose and frame_idx == 0:
            print(f"Applying anatomical constraints with temporal consistency to {len(keypoints_3d_list)} frames...")
        
        # Apply constraints for this frame
        corrected = apply_anatomical_constraints(
            kps, scores, confidence_threshold, 
            verbose=(verbose and frame_idx < 3)
        )
        
        # Temporal consistency check: Don't flip if it creates oscillation
        if frame_idx > 0:
            corrected = _apply_temporal_consistency(
                corrected, corrected_list[-1], kps, 
                frame_idx, correction_history, temporal_window
            )
        
        corrected_list.append(corrected)
    
    if verbose:
        print(f"Anatomical constraint correction complete with temporal smoothing")
    
    return corrected_list


def _apply_temporal_consistency(current_corrected: np.ndarray, 
                                previous_corrected: np.ndarray,
                                current_original: np.ndarray,
                                frame_idx: int,
                                correction_history: dict,
                                window_size: int) -> np.ndarray:
    """
    Apply temporal consistency to prevent oscillations in joint corrections.
    
    If a joint would flip back and forth between frames, prefer stability.
    """
    # Check major joints that might have been corrected
    joints_to_check = [
        BodyJoint.LEFT_ELBOW, BodyJoint.RIGHT_ELBOW,
        BodyJoint.LEFT_KNEE, BodyJoint.RIGHT_KNEE
    ]
    
    result = current_corrected.copy()
    
    # Handle batch dimension
    if len(result.shape) == 3:
        result_single = result[0]
        prev_single = previous_corrected[0] if len(previous_corrected.shape) == 3 else previous_corrected
        orig_single = current_original[0] if len(current_original.shape) == 3 else current_original
    else:
        result_single = result
        prev_single = previous_corrected
        orig_single = current_original
    
    for joint_idx in joints_to_check:
        # Check if this joint was "corrected" (z changed significantly)
        z_change = abs(result_single[joint_idx, 2] - orig_single[joint_idx, 2])
        
        if z_change > 0.01:  # Joint was flipped
            # Check if it would flip back from previous frame
            prev_z = prev_single[joint_idx, 2]
            new_z = result_single[joint_idx, 2]
            
            # If new position is far from previous, might be oscillating
            z_jump = abs(new_z - prev_z)
            
            # Track correction history
            if joint_idx not in correction_history:
                correction_history[joint_idx] = []
            
            correction_history[joint_idx].append(frame_idx)
            
            # Check recent history for oscillation pattern
            recent_corrections = [f for f in correction_history[joint_idx] 
                                if frame_idx - f < window_size]
            
            # If we've been correcting this joint frequently, it might be oscillating
            # In that case, prefer the previous frame's position for stability
            if len(recent_corrections) > window_size // 2 and z_jump > 0.05:
                # Stabilize: use previous frame's z-coordinate
                result_single[joint_idx, 2] = prev_z
    
    if len(result.shape) == 3:
        result[0] = result_single
    else:
        result = result_single
    
    return result
