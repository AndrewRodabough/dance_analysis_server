"""Analyze leg straightening timing for rumba/cha-cha walk using hips, knees, and ankles."""

import logging
import numpy as np
from enum import Enum
from typing import Any, Dict, Tuple, List, Optional

from shared.skeletons.pose_data import VectorizedPoseData

logger = logging.getLogger(__name__)

def deg_to_rad(angle_in_degrees):
    return angle_in_degrees * np.pi / 180

class WalkingState(Enum):
    RELEASE = 0
    PASSING = 1
    EXTENSION = 2
    ARRIVAL = 3
    COMPLETED = 4

# 3D Thresholds (in meters)
VELOCITY_THRESHOLD = 0.06      # m/s: Minimum speed to consider foot "moving"
PLANT_THRESHOLD = 0.04         # m/s: Maximum speed to consider foot "planted"
HIP_DISTANCE_THRESHOLD = 0.4   # Below this = weight fully transferred
BREAKOUT_THRESHOLD = 0.16      # m: Min horizontal distance to trigger next Release
PASSING_MIN_DIST = 0.1        # m: Max distance between ankles to be "Passing"
PASSING_ENTRANCE_DIST = 0.25

# 2D Thresholds (in pixels, calibrated from actual test video data)
# Data ranges observed: velocity avg=5.7 max=242, hip-ankle X-axis min=0.07 max=155, ankle-ankle min=0.5 max=252
VELOCITY_THRESHOLD_2D = 4.0       # pixels/frame: Minimum speed to consider foot "moving" (above avg 5.7)
PLANT_THRESHOLD_2D = 4.0          # pixels/frame: Maximum speed to consider foot "planted"
HIP_DISTANCE_THRESHOLD_2D = 50.0  # pixels: Below this = weight fully transferred (observed min near 0, used for final alignment)
BREAKOUT_THRESHOLD_2D = 20.0      # pixels: Min horizontal distance to trigger next Release
PASSING_MIN_DIST_2D = 20.0        # pixels: Max distance between ankles to be "Passing"
PASSING_ENTRANCE_DIST_2D = 40.0   # pixels: Distance threshold to enter passing state
ANKLE_FORWARD_THRESHOLD_2D = 45.0 # pixels: Min hip-ankle offset for extension (must reach well forward in cycle)

STRAIGHT_LEG_MIN = deg_to_rad(172)  # Minimum angle to be considered "Straight"
FLEXED_LEG_MAX = deg_to_rad(160)    # Maximum angle to be considered "Flexed" during passing
RELEASE_DRIVE_MAX = deg_to_rad(170)
HYPER_EXTENSION_DEVIATION_THRESHOLD_2D = 5.0  # pixels: Min backward deviation from hip-ankle line to classify as hyper-extended

STARTING_STATE = WalkingState.RELEASE
      
def analyze_cha_cha_walk(pose_data_3d: VectorizedPoseData) -> Dict[str, Any]:
    """
    Analyzes a sequence of 3D poses for Cha Cha Forward Walk technique.
    Returns a dictionary containing the frame-by-frame State and any identified Faults.
    """
    
    angles = pose_data_3d.get_weighted_bone_angles(threshold=0)
    velocities = pose_data_3d.get_weighted_joint_velocities(threshold=0)
    weight_transfer = compute_weight_transfer_offsets(pose_data_3d)
    ankle_distances = compute_ankle_to_ankle_distance_xz(pose_data_3d)

    left_knee_idx = pose_data_3d.skeleton.name_to_idx["L_Knee"]
    right_knee_idx = pose_data_3d.skeleton.name_to_idx["R_Knee"]
    left_ankle_idx = pose_data_3d.skeleton.name_to_idx["L_Ankle"]
    right_ankle_idx = pose_data_3d.skeleton.name_to_idx["R_Ankle"]

    ankle_velocities = velocities[:, [left_ankle_idx, right_ankle_idx], :]
    knee_angles = angles[:, [left_knee_idx, right_knee_idx]]

    current_state = STARTING_STATE
    current_standing_leg = 0  # 0=left, 1=right
    current_moving_leg = 1 - current_standing_leg

    states = []
    faults = []

    # --- 3. ANALYSIS LOOP ---
    for f_idx in range(pose_data_3d.num_frames):
        states.append(current_state.name)

        # helper variables for readability
        active_knee_angle = knee_angles[f_idx, current_moving_leg]
        standing_knee_angle = knee_angles[f_idx, current_standing_leg]
        active_ankle_velocity = magnitude(ankle_velocities[f_idx, current_moving_leg])
        standing_ankle_velocity = magnitude(ankle_velocities[f_idx, current_standing_leg])
        active_ankle_hip_offset = np.linalg.norm(weight_transfer[f_idx, current_moving_leg])
        standing_ankle_hip_offset = np.linalg.norm(weight_transfer[f_idx, current_standing_leg])
        ankle_distance = ankle_distances[f_idx]

        # ------------------------------------------------------------------
        # STATE: RELEASE (The Drive)
        # ------------------------------------------------------------------
        if current_state == WalkingState.RELEASE:
            # OPTIMIZATION: Ignore faults if we aren't moving yet (Standing Still)
            is_moving = active_ankle_velocity > 0.02
            
            if is_moving:
                # CHECK 1: Standing leg must stay locked (The Anchor)
                if standing_knee_angle < STRAIGHT_LEG_MIN:
                    faults.append({"frame": f_idx, "type": "SOFT_STANDING_LEG_IN_DRIVE"})

                # CHECK 2: Active leg must flex to drive (The Push)
                if active_knee_angle > RELEASE_DRIVE_MAX:
                    faults.append({"frame": f_idx, "type": "NO_DRIVE_ACTION"})

            # TRANSITION LOGIC (Improved)
            # We move to PASSING if:
            # 1. We are moving fast enough (Velocity) OR
            # 2. The ankles are getting close together (Distance)
            
            approaching_pass = ankle_distance < PASSING_ENTRANCE_DIST
            
            if active_ankle_velocity > VELOCITY_THRESHOLD or (is_moving and approaching_pass):
                current_state = WalkingState.PASSING


        # ------------------------------------------------------------------
        # STATE: PASSING (The Bow and Arrow)
        # ------------------------------------------------------------------
        elif current_state == WalkingState.PASSING:
            # CHECK 3: The "Broken Bow" (Standing leg softens)
            if standing_knee_angle < STRAIGHT_LEG_MIN:
                faults.append({"frame": f_idx, "type": "DROPPED_HEIGHT_IN_PASSING"})

            # CHECK 4: Stiff Passing Leg (The Arrow isn't drawn)
            # Only check this if we are physically passing the standing leg
            if ankle_distance < PASSING_MIN_DIST:
                if active_knee_angle > FLEXED_LEG_MAX:
                    faults.append({"frame": f_idx, "type": "STIFF_PASSING_LEG"})

            # TRANSITION: Active foot moves past the standing foot
            if ankle_distance < PASSING_MIN_DIST:  # Added buffer to prevent jittery transitions
                 current_state = WalkingState.EXTENSION


        # ------------------------------------------------------------------
        # STATE: EXTENSION (The Reach)
        # ------------------------------------------------------------------
        elif current_state == WalkingState.EXTENSION:
            # CHECK 5: Early Locking / Stumping
            # Leg is straight, but foot is still moving fast (hasn't landed)
            if active_knee_angle > deg_to_rad(178) and active_ankle_velocity > PLANT_THRESHOLD:
                faults.append({"frame": f_idx, "type": "EARLY_LOCK_STUMPING"})

            # TRANSITION: Foot slows down (plants) AND is sufficiently forward
            if active_ankle_velocity < PLANT_THRESHOLD and active_ankle_hip_offset > 0.34 and ankle_distance > 0.34:
                current_state = WalkingState.ARRIVAL


        # ------------------------------------------------------------------
        # STATE: ARRIVAL (The Moment of Truth)
        # ------------------------------------------------------------------
        elif current_state == WalkingState.ARRIVAL:
            # CHECK 6: The "Double Straight" (International Latin Standard)
            if active_knee_angle < STRAIGHT_LEG_MIN or standing_knee_angle < STRAIGHT_LEG_MIN:
                faults.append({"frame": f_idx, "type": "SOFT_KNEE_ARRIVAL"})

            # TRANSITION: Weight Transfer (Hips move over the Active Ankle)
            if active_ankle_hip_offset < HIP_DISTANCE_THRESHOLD:
                # OFFICIAL HANDOVER
                current_standing_leg, current_moving_leg = current_moving_leg, current_standing_leg
                current_state = WalkingState.COMPLETED


        # ------------------------------------------------------------------
        # STATE: COMPLETED (The Settle)
        # ------------------------------------------------------------------
        elif current_state == WalkingState.COMPLETED:
            # Note: 'stand_k' now refers to the NEW standing leg (the one that just arrived)
            
            # CHECK 7: Buckling after arrival
            if standing_knee_angle < STRAIGHT_LEG_MIN:
                faults.append({"frame": f_idx, "type": "BUCKLED_STANDING_LEG"})

            # TRANSITION: Hips "break out" forward to start next step
            # We track the distance between the NEW standing leg and the hips
            if standing_ankle_hip_offset > BREAKOUT_THRESHOLD: 
                current_state = WalkingState.RELEASE


    return {
        "states": states, 
        "faults": faults,
        "final_standing_leg": "Right" if current_standing_leg == 1 else "Left"
    }

# --- HELPER FUNCTIONS ---

def magnitude(vec):
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

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

    left_hip_idx = pose_data_3d.skeleton.name_to_idx["L_Hip"]
    left_ankle_idx = pose_data_3d.skeleton.name_to_idx["L_Ankle"]
    right_hip_idx = pose_data_3d.skeleton.name_to_idx["R_Hip"]
    right_ankle_idx = pose_data_3d.skeleton.name_to_idx["R_Ankle"]

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
    left_ankle_idx = pose_data_3d.skeleton.name_to_idx["L_Ankle"]
    right_ankle_idx = pose_data_3d.skeleton.name_to_idx["R_Ankle"]

    if pose_data_3d.skeleton.data is None:
        raise ValueError("pose_data_3d must have loaded skeleton data")

    skeleton_data = pose_data_3d.skeleton.data
    if skeleton_data.ndim != 3 or skeleton_data.shape[2] < 3:
        raise ValueError("pose_data_3d.skeleton.data must have shape (frames, joints, 3)")

    left_ankle = skeleton_data[:, left_ankle_idx, :]
    right_ankle = skeleton_data[:, right_ankle_idx, :]
    diff = left_ankle - right_ankle

    return np.sqrt(diff[:, 0] ** 2 + diff[:, 2] ** 2).astype(np.float32)
 # Angle must drop below this to show "Drive"


from itertools import groupby
from collections import Counter

def log_analysis_summary(analysis_result: Dict[str, Any]):
    """
    Parses the analysis output to print a clean, human-readable timeline.
    Groups consecutive frames of the same state and summarizes faults within them.
    """
    states = analysis_result["states"]
    faults = analysis_result["faults"]
    
    print(f"\n{'='*20} CHA CHA WALK ANALYSIS TIMELINE {'='*20}")
    
    # 1. Group consecutive states into segments
    # Result: [(StateName, StartFrame, EndFrame), ...]
    segments = []
    current_idx = 0
    for state_name, group in groupby(states):
        length = len(list(group))
        segments.append({
            "state": state_name,
            "start": current_idx,
            "end": current_idx + length - 1
        })
        current_idx += length

    # 2. Map faults to these segments
    # We want to know which faults happened during which state segment
    for segment in segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        state_name = segment["state"]
        
        # Filter faults that happened in this frame range
        segment_faults = [
            f for f in faults 
            if seg_start <= f["frame"] <= seg_end
        ]
        
        # Print the State Header
        print(f"\n[Frames {seg_start:03d} - {seg_end:03d}]  STATE: {state_name}")
        
        # 3. Summarize the Faults (Deduplicate)
        if not segment_faults:
            print("    ✔  Technique Clean")
        else:
            # Count how many frames triggered each fault type
            fault_counts = Counter(f["type"] for f in segment_faults)
            
            for fault_type, count in fault_counts.items():
                # Visual Indicator for severity
                severity = "WARNING" if count < 3 else "FAIL" 
                print(f"    ❌ {fault_type} (triggered in {count} frames)")
                
                # Context helper (Why it triggered)
                if fault_type == "SOFT_STANDING_LEG_IN_DRIVE":
                    print("       -> Standing leg bent before the push was finished.")
                elif fault_type == "STIFF_PASSING_LEG":
                    print("       -> Knee angle was too straight while crossing legs.")
                elif fault_type == "EARLY_LOCK_STUMPING":
                    print("       -> Leg locked straight while foot was still moving fast.")
                elif fault_type == "SOFT_KNEE_ARRIVAL":
                    print("       -> Weight transferred, but leg wasn't 180° straight.")

    print(f"\n{'='*60}\n")


def probe_data_ranges(pose_data_3d):
    # 1. Calculate the raw features
    velocities = pose_data_3d.get_weighted_joint_velocities(threshold=0)
    offsets = compute_weight_transfer_offsets(pose_data_3d) # (Frames, 2, 2)
    ankle_dists = compute_ankle_to_ankle_distance_xz(pose_data_3d) # (Frames,)

    # 2. Extract magnitudes
    # Speed of Left Ankle
    l_ank_idx = pose_data_3d.skeleton.name_to_idx["L_Ankle"]
    speeds = np.linalg.norm(velocities[:, l_ank_idx, :], axis=1)
    
    # Hip-to-Ankle Distance
    hip_dists = np.linalg.norm(offsets[:, 0, :], axis=1) # Left leg offsets

    print(f"{'='*10} DATA UNIT PROBE {'='*10}")
    print(f"Velocities (Speed):")
    print(f"  Min: {np.min(speeds):.4f}")
    print(f"  Max: {np.max(speeds):.4f}  <-- If this is > 10, you are in Pixels")
    print(f"  Avg: {np.mean(speeds):.4f}")
    
    print(f"\nHip-to-Ankle Distance (Weight Transfer):")
    print(f"  Min: {np.min(hip_dists):.4f}  <-- This should be your NEW 'HIP_DISTANCE_THRESHOLD'")
    print(f"  Max: {np.max(hip_dists):.4f}")
    
    print(f"\nAnkle-to-Ankle Distance (Passing):")
    print(f"  Min: {np.min(ankle_dists):.4f} <-- This should be your NEW 'PASSING_MIN_DIST'")
    print(f"  Max: {np.max(ankle_dists):.4f}")
    print(f"{'='*35}")


# ============================================================================
# 2D ANALYSIS FUNCTIONS FOR SIDE VIEW (COCO-17 FORMAT)
# ============================================================================

def probe_data_ranges_2d(pose_data_2d):
    """
    Probe 2D data ranges to help calibrate thresholds.
    Useful for understanding pixel-based vs normalized coordinate scales.
    """
    # 1. Calculate the raw features
    velocities = pose_data_2d.get_weighted_joint_velocities(threshold=0)
    offsets = compute_weight_transfer_offsets_2d(pose_data_2d)  # (Frames, 2, 2)
    ankle_dists = compute_ankle_to_ankle_distance_2d(pose_data_2d)  # (Frames,)

    # 2. Extract magnitudes
    # Speed of Left Ankle
    l_ank_idx = pose_data_2d.skeleton.name_to_idx["left_ankle"]
    speeds = np.linalg.norm(velocities[:, l_ank_idx, :], axis=1)
    
    # Hip-to-Ankle Distance (X-axis only for horizontal alignment)
    hip_dists_x = np.abs(offsets[:, 0, 0])  # Left leg X offset only

    print(f"{'='*10} 2D DATA UNIT PROBE {'='*10}")
    print(f"Velocities (Speed in 2D):")
    print(f"  Min: {np.min(speeds):.4f}")
    print(f"  Max: {np.max(speeds):.4f}")
    print(f"  Avg: {np.mean(speeds):.4f}")
    print(f"  Current Thresholds: VELOCITY={VELOCITY_THRESHOLD_2D}, PLANT={PLANT_THRESHOLD_2D}")
    
    print(f"\nHip-to-Ankle Distance (X-axis only for weight transfer):")
    print(f"  Min: {np.min(hip_dists_x):.4f}")
    print(f"  Max: {np.max(hip_dists_x):.4f}")
    print(f"  Current Threshold: HIP_DISTANCE={HIP_DISTANCE_THRESHOLD_2D}, ANKLE_FORWARD={ANKLE_FORWARD_THRESHOLD_2D}")
    
    print(f"\nAnkle-to-Ankle Distance (Passing):")
    print(f"  Min: {np.min(ankle_dists):.4f}")
    print(f"  Max: {np.max(ankle_dists):.4f}")
    print(f"  Current Thresholds: PASSING_MIN={PASSING_MIN_DIST_2D}, PASSING_ENTRANCE={PASSING_ENTRANCE_DIST_2D}")
    print(f"{'='*38}")


def magnitude_2d(vec):
    """Calculate magnitude of a 2D vector."""
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)


def compute_weight_transfer_offsets_2d(
    pose_data_2d: VectorizedPoseData,
) -> np.ndarray:
    """
    Compute X/Y offsets from the mid-hip to each ankle for 2D side view.
    
    In side view:
    - X axis = forward/backward (horizontal) - USED for weight transfer detection
    - Y axis = up/down (vertical) - stored but not used for weight transfer
    
    Weight transfer is detected when X offset is small (hip horizontally aligned over ankle).

    Returns:
        Array of shape (frames, 2, 2) where:
        - axis 0 = frame
        - axis 1 = leg (0=left, 1=right)
        - axis 2 = offset components (x, y)
    """
    left_hip_idx = pose_data_2d.skeleton.name_to_idx["left_hip"]
    left_ankle_idx = pose_data_2d.skeleton.name_to_idx["left_ankle"]
    right_hip_idx = pose_data_2d.skeleton.name_to_idx["right_hip"]
    right_ankle_idx = pose_data_2d.skeleton.name_to_idx["right_ankle"]

    if pose_data_2d.skeleton.data is None:
        raise ValueError("pose_data_2d must have loaded skeleton data")

    skeleton_data = pose_data_2d.skeleton.data
    if skeleton_data.ndim != 3 or skeleton_data.shape[2] < 2:
        raise ValueError("pose_data_2d.skeleton.data must have shape (frames, joints, 2)")

    mid_hip = (skeleton_data[:, left_hip_idx, :] + skeleton_data[:, right_hip_idx, :]) * 0.5
    left_offset = mid_hip - skeleton_data[:, left_ankle_idx, :]
    right_offset = mid_hip - skeleton_data[:, right_ankle_idx, :]

    offsets = np.zeros((skeleton_data.shape[0], 2, 2), dtype=np.float32)
    offsets[:, 0, 0] = left_offset[:, 0]   # left x (horizontal)
    offsets[:, 0, 1] = left_offset[:, 1]   # left y (vertical)
    offsets[:, 1, 0] = right_offset[:, 0]  # right x (horizontal)
    offsets[:, 1, 1] = right_offset[:, 1]  # right y (vertical)

    return offsets


def compute_ankle_to_ankle_distance_2d(
    pose_data_2d: VectorizedPoseData,
) -> np.ndarray:
    """
    Compute horizontal (X-axis) ankle-to-ankle distance per frame for side view.
    In side view, this represents forward/backward separation.

    Returns:
        Array of shape (frames,) with distance in pixels (or normalized units).
    """
    left_ankle_idx = pose_data_2d.skeleton.name_to_idx["left_ankle"]
    right_ankle_idx = pose_data_2d.skeleton.name_to_idx["right_ankle"]

    if pose_data_2d.skeleton.data is None:
        raise ValueError("pose_data_2d must have loaded skeleton data")

    skeleton_data = pose_data_2d.skeleton.data
    if skeleton_data.ndim != 3 or skeleton_data.shape[2] < 2:
        raise ValueError("pose_data_2d.skeleton.data must have shape (frames, joints, 2)")

    left_ankle = skeleton_data[:, left_ankle_idx, :]
    right_ankle = skeleton_data[:, right_ankle_idx, :]
    diff = left_ankle - right_ankle

    # For side view, use full 2D distance (both horizontal and vertical components)
    return np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2).astype(np.float32)


def determine_walk_direction_2d(pose_data_2d: VectorizedPoseData) -> int:
    """
    Determine the horizontal direction of walking from 2D side-view data.

    Uses the net horizontal displacement of the mid-hip across all frames to
    determine direction. Returns +1 if walking to the right (+X direction)
    or -1 if walking to the left (-X direction).

    Args:
        pose_data_2d: VectorizedPoseData with 2D skeleton data (frames, joints, 2)

    Returns:
        +1 for rightward walk, -1 for leftward walk.
    """
    left_hip_idx = pose_data_2d.skeleton.name_to_idx["left_hip"]
    right_hip_idx = pose_data_2d.skeleton.name_to_idx["right_hip"]
    skeleton_data = pose_data_2d.skeleton.data

    if skeleton_data.shape[0] < 2:
        return 1  # Default to right if not enough frames to determine direction

    mid_hip_x = (skeleton_data[:, left_hip_idx, 0] + skeleton_data[:, right_hip_idx, 0]) * 0.5
    total_displacement = mid_hip_x[-1] - mid_hip_x[0]

    return 1 if total_displacement >= 0 else -1


def compute_knee_deviation_2d(pose_data_2d: VectorizedPoseData) -> np.ndarray:
    """
    Compute the signed horizontal deviation of each knee from the hip-ankle line.

    For each frame and each leg, computes how far the knee deviates horizontally
    from the straight line connecting the hip to the ankle.  A positive value
    means the knee is to the right (+X) of that line; a negative value means it
    is to the left (-X).

    This deviation is used to distinguish a bent leg (knee forward of the
    hip-ankle line) from a hyper-extended leg (knee behind the line):
      - Walking RIGHT (+1): positive deviation = bent/flexed, negative = hyper-extended
      - Walking LEFT  (-1): negative deviation = bent/flexed, positive = hyper-extended

    Args:
        pose_data_2d: VectorizedPoseData with 2D skeleton data (frames, joints, 2)

    Returns:
        Array of shape (frames, 2) where axis 1 is leg (0=left, 1=right).
        Values are signed horizontal deviations in pixels (or normalized units).
    """
    left_hip_idx = pose_data_2d.skeleton.name_to_idx["left_hip"]
    left_knee_idx = pose_data_2d.skeleton.name_to_idx["left_knee"]
    left_ankle_idx = pose_data_2d.skeleton.name_to_idx["left_ankle"]
    right_hip_idx = pose_data_2d.skeleton.name_to_idx["right_hip"]
    right_knee_idx = pose_data_2d.skeleton.name_to_idx["right_knee"]
    right_ankle_idx = pose_data_2d.skeleton.name_to_idx["right_ankle"]

    skeleton_data = pose_data_2d.skeleton.data
    num_frames = skeleton_data.shape[0]
    deviations = np.zeros((num_frames, 2), dtype=np.float32)

    leg_joints = [
        (left_hip_idx, left_knee_idx, left_ankle_idx),
        (right_hip_idx, right_knee_idx, right_ankle_idx),
    ]

    for leg_idx, (hip_idx, knee_idx, ankle_idx) in enumerate(leg_joints):
        hip = skeleton_data[:, hip_idx, :]      # (frames, 2)
        knee = skeleton_data[:, knee_idx, :]    # (frames, 2)
        ankle = skeleton_data[:, ankle_idx, :]  # (frames, 2)

        hip_to_ankle = ankle - hip              # (frames, 2)
        hip_to_knee = knee - hip                # (frames, 2)

        # Parameterize the projection of the knee onto the hip-ankle segment
        length_sq = np.sum(hip_to_ankle ** 2, axis=1)       # (frames,)
        safe_length_sq = np.where(length_sq == 0, 1.0, length_sq)
        t = np.sum(hip_to_knee * hip_to_ankle, axis=1) / safe_length_sq  # (frames,)

        # Closest point on the hip-ankle line to the knee
        closest_x = hip[:, 0] + t * hip_to_ankle[:, 0]     # (frames,)

        # Signed horizontal deviation: positive = knee is to the right of the line
        deviations[:, leg_idx] = knee[:, 0] - closest_x

    return deviations


def analyze_cha_cha_walk_2d(pose_data_2d: VectorizedPoseData, walk_direction: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyzes a sequence of 2D poses from side view for Cha Cha Forward Walk technique.
    
    This version works with COCO-17 format 2D keypoints and adapts the 3D analysis 
    logic to work with side-view projections. Both legs are analyzed simultaneously.
    
    Bent vs. hyper-extended leg differentiation:
        A bent (flexed) knee deviates forward (in the direction of travel) from the
        straight hip-ankle line. A hyper-extended knee deviates backward (opposite to
        the direction of travel). The walk_direction parameter controls which side is
        "forward". When not supplied it is auto-detected from mid-hip displacement.

    Args:
        pose_data_2d: VectorizedPoseData with 2D skeleton data (frames, joints, 2)
                     Expected format: COCO-17 with lowercase joint names
        walk_direction: +1 if the subject walks to the right (+X), -1 if to the left
                        (-X). Auto-detected from hip displacement when None.
                     
    Returns:
        Dictionary containing:
            - "states": List of state names per frame
            - "faults": List of detected faults with frame numbers and types
            - "final_standing_leg": Which leg is supporting at the end
            - "walk_direction": The walk direction used (+1 or -1)
    """
    # Determine walk direction (right = +1, left = -1)
    if walk_direction is None:
        walk_direction = determine_walk_direction_2d(pose_data_2d)

    # Compute features using 2D data
    velocities = pose_data_2d.get_weighted_joint_velocities(threshold=0)
    weight_transfer = compute_weight_transfer_offsets_2d(pose_data_2d)
    ankle_distances = compute_ankle_to_ankle_distance_2d(pose_data_2d)
    knee_deviations = compute_knee_deviation_2d(pose_data_2d)  # (frames, 2), 0=left, 1=right

    # Compute knee angles directly from named joints to avoid angle-triplet index mismatches
    left_knee_angles = pose_data_2d.skeleton.get_angle("left_hip", "left_knee", "left_ankle")
    right_knee_angles = pose_data_2d.skeleton.get_angle("right_hip", "right_knee", "right_ankle")
    knee_angles = np.stack([left_knee_angles, right_knee_angles], axis=1)  # (frames, 2)

    # Get joint indices (COCO-17 format uses lowercase names)
    left_ankle_idx = pose_data_2d.skeleton.name_to_idx["left_ankle"]
    right_ankle_idx = pose_data_2d.skeleton.name_to_idx["right_ankle"]
    left_knee_idx = pose_data_2d.skeleton.name_to_idx["left_knee"]
    right_knee_idx = pose_data_2d.skeleton.name_to_idx["right_knee"]

    ankle_velocities = velocities[:, [left_ankle_idx, right_ankle_idx], :]

    # Initialize state machine
    current_state = STARTING_STATE
    current_standing_leg = 0  # 0=left, 1=right
    current_moving_leg = 1 - current_standing_leg

    states = []
    faults = []

    # Analysis loop - same logic as 3D but with 2D metrics
    for f_idx in range(pose_data_2d.num_frames):
        states.append(current_state.name)

        # Helper variables for readability
        active_knee_angle = knee_angles[f_idx, current_moving_leg]
        standing_knee_angle = knee_angles[f_idx, current_standing_leg]
        active_ankle_velocity = magnitude_2d(ankle_velocities[f_idx, current_moving_leg])
        standing_ankle_velocity = magnitude_2d(ankle_velocities[f_idx, current_standing_leg])
        # Use X-axis only for weight transfer (horizontal alignment in side view)
        active_ankle_hip_offset = abs(weight_transfer[f_idx, current_moving_leg, 0])
        standing_ankle_hip_offset = abs(weight_transfer[f_idx, current_standing_leg, 0])
        ankle_distance = ankle_distances[f_idx]

        # Knee deviation projected onto the walk direction axis.
        # A negative value means the knee is behind the hip-ankle line (hyper-extended).
        active_knee_deviation = knee_deviations[f_idx, current_moving_leg] * walk_direction
        standing_knee_deviation = knee_deviations[f_idx, current_standing_leg] * walk_direction
        standing_is_hyper_extended = standing_knee_deviation < -HYPER_EXTENSION_DEVIATION_THRESHOLD_2D
        active_is_hyper_extended = active_knee_deviation < -HYPER_EXTENSION_DEVIATION_THRESHOLD_2D

        # Check for low confidence - skip fault detection if data is unreliable
        confidence = pose_data_2d.confidence[f_idx]
        left_conf = confidence[left_ankle_idx] * confidence[left_knee_idx]
        right_conf = confidence[right_ankle_idx] * confidence[right_knee_idx]
        is_confident = (left_conf > 0.09) and (right_conf > 0.09)  # 0.3^2 threshold

        # ------------------------------------------------------------------
        # STATE: RELEASE (The Drive)
        # ------------------------------------------------------------------
        if current_state == WalkingState.RELEASE:
            is_moving = active_ankle_velocity > PLANT_THRESHOLD_2D  # Use plant threshold as minimum movement

            if is_moving and is_confident:
                # CHECK 1: Standing leg must stay locked
                if standing_knee_angle < STRAIGHT_LEG_MIN:
                    faults.append({"frame": f_idx, "type": "SOFT_STANDING_LEG_IN_DRIVE"})

                # CHECK 1b: Standing leg is hyper-extended (knee pushed backward)
                if standing_is_hyper_extended:
                    faults.append({"frame": f_idx, "type": "HYPER_EXTENDED_STANDING_LEG"})

                # CHECK 2: Active leg must flex to drive
                if active_knee_angle > RELEASE_DRIVE_MAX:
                    faults.append({"frame": f_idx, "type": "NO_DRIVE_ACTION"})

            # TRANSITION LOGIC
            approaching_pass = ankle_distance < PASSING_ENTRANCE_DIST_2D

            if active_ankle_velocity > VELOCITY_THRESHOLD_2D or (is_moving and approaching_pass):
                current_state = WalkingState.PASSING

        # ------------------------------------------------------------------
        # STATE: PASSING (The Bow and Arrow)
        # ------------------------------------------------------------------
        elif current_state == WalkingState.PASSING:
            if is_confident:
                # CHECK 3: Standing leg softens (dropped height)
                if standing_knee_angle < STRAIGHT_LEG_MIN:
                    faults.append({"frame": f_idx, "type": "DROPPED_HEIGHT_IN_PASSING"})

                # CHECK 3b: Standing leg is hyper-extended
                if standing_is_hyper_extended:
                    faults.append({"frame": f_idx, "type": "HYPER_EXTENDED_STANDING_LEG"})

                # CHECK 4: Stiff Passing Leg
                if ankle_distance < PASSING_MIN_DIST_2D:
                    if active_knee_angle > FLEXED_LEG_MAX:
                        faults.append({"frame": f_idx, "type": "STIFF_PASSING_LEG"})

            # TRANSITION: Active foot moves past the standing foot
            if ankle_distance < PASSING_MIN_DIST_2D:
                current_state = WalkingState.EXTENSION

        # ------------------------------------------------------------------
        # STATE: EXTENSION (The Reach)
        # ------------------------------------------------------------------
        elif current_state == WalkingState.EXTENSION:
            if is_confident:
                # CHECK 5: Early Locking / Stumping
                if active_knee_angle > deg_to_rad(178) and active_ankle_velocity > PLANT_THRESHOLD_2D:
                    faults.append({"frame": f_idx, "type": "EARLY_LOCK_STUMPING"})

            # TRANSITION: Foot slows down (plants) AND is sufficiently forward
            if active_ankle_velocity < PLANT_THRESHOLD_2D and active_ankle_hip_offset > ANKLE_FORWARD_THRESHOLD_2D:
                current_state = WalkingState.ARRIVAL

        # ------------------------------------------------------------------
        # STATE: ARRIVAL (The Moment of Truth)
        # ------------------------------------------------------------------
        elif current_state == WalkingState.ARRIVAL:
            if is_confident:
                # CHECK 6: The "Double Straight"
                if active_knee_angle < STRAIGHT_LEG_MIN or standing_knee_angle < STRAIGHT_LEG_MIN:
                    faults.append({"frame": f_idx, "type": "SOFT_KNEE_ARRIVAL"})

                # CHECK 6b: Hyper-extended legs at arrival
                if active_is_hyper_extended:
                    faults.append({"frame": f_idx, "type": "HYPER_EXTENDED_ACTIVE_LEG"})
                if standing_is_hyper_extended:
                    faults.append({"frame": f_idx, "type": "HYPER_EXTENDED_STANDING_LEG"})

            # TRANSITION: Weight Transfer (Hips move over the Active Ankle)
            if active_ankle_hip_offset < HIP_DISTANCE_THRESHOLD_2D:
                # OFFICIAL HANDOVER
                current_standing_leg, current_moving_leg = current_moving_leg, current_standing_leg
                current_state = WalkingState.RELEASE

            elif standing_ankle_velocity > VELOCITY_THRESHOLD_2D:
                current_state = WalkingState.PASSING
                current_standing_leg, current_moving_leg = current_moving_leg, current_standing_leg
                faults.append({"frame": f_idx, "type": "WEIGHT_TRANSFER_WITH_MOVING_FOOT"})

    return {
        "states": states,
        "faults": faults,
        "final_standing_leg": "Right" if current_standing_leg == 1 else "Left",
        "walk_direction": walk_direction,
    }