"""Analyze leg straightening timing for rumba/cha-cha walk using hips, knees, and ankles."""

import logging
import numpy as np
from enum import Enum
from typing import Any, Dict, Tuple, List

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

VELOCITY_THRESHOLD = 0.05      # m/s: Minimum speed to consider foot "moving"
PLANT_THRESHOLD = 0.02         # m/s: Maximum speed to consider foot "planted"
HIP_DISTANCE_THRESHOLD = 0.1   # Below this = weight fully transferred
BREAKOUT_THRESHOLD = 0.12      # m: Min horizontal distance to trigger next Release
PASSING_MIN_DIST = 0.10        # m: Max distance between ankles to be "Passing"

STRAIGHT_LEG_MIN = deg_to_rad(172)  # Minimum angle to be considered "Straight"
FLEXED_LEG_MAX = deg_to_rad(160)    # Maximum angle to be considered "Flexed" during passing
RELEASE_DRIVE_MAX = deg_to_rad(170)

STARTING_STATE = WalkingState.COMPLETED
      
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
            # CHECK 1: Standing leg must stay locked (The Anchor)
            if standing_knee_angle < STRAIGHT_LEG_MIN:
                faults.append({"frame": f_idx, "type": "SOFT_STANDING_LEG_IN_DRIVE"})

            # CHECK 2: Active leg must flex to drive (The Push)
            # If moving fast but knee is still locked, they are "falling", not driving
            if active_ankle_velocity > VELOCITY_THRESHOLD and active_knee_angle > RELEASE_DRIVE_MAX:
                faults.append({"frame": f_idx, "type": "NO_DRIVE_ACTION"})

            # TRANSITION: Significant forward velocity
            if active_ankle_velocity > VELOCITY_THRESHOLD:
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
            if ankle_distance > PASSING_MIN_DIST:
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
            if active_ankle_velocity < PLANT_THRESHOLD and active_ankle_hip_offset > 0.15:
                current_state = WalkingState.ARRIVAL


        # ------------------------------------------------------------------
        # STATE: ARRIVAL (The Moment of Truth)
        # ------------------------------------------------------------------
        elif current_state == WalkingState.ARRIVAL:
            # CHECK 6: The "Double Straight" (International Latin Standard)
            if active_knee_angle < STRAIGHT_LEG_MIN or standing_knee_angle < STRAIGHT_LEG_MIN:
                faults.append({"frame": f_idx, "type": "SOFT_KNEE_ARRIVAL"})

            # TRANSITION: Weight Transfer (Hips move over the Active Ankle)
            if abs(active_ankle_hip_offset) < HIP_DISTANCE_THRESHOLD:
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
            if abs(standing_ankle_hip_offset) > BREAKOUT_THRESHOLD: 
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
    l_ank_idx = pose_data_3d.skeleton.name_to_idx["left_ankle"]
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


"""
def cha_cha_walks_analysis(pose_data_3d: VectorizedPoseData) -> Dict[str, np.ndarray]:

    angles = pose_data_3d.get_weighted_bone_angles(threshold=0)
    velocities = pose_data_3d.get_weighted_joint_velocities(threshold=0)
    weight_transfer = compute_weight_transfer_offsets(pose_data_3d)
    ankle_distance = compute_ankle_to_ankle_distance_xz(pose_data_3d)

    ankle_velocities = velocities[:, [pose_data_3d.skeleton.name_to_idx["left_ankle"], pose_data_3d.skeleton.name_to_idx["right_ankle"]], :]
    knee_angles = angles[:, [pose_data_3d.skeleton.name_to_idx["left_knee"], pose_data_3d.skeleton.name_to_idx["right_knee"]]]

    current_state = STARTING_STATE
    current_standing_leg = 0  # 0=left, 1=right
    current_moving_leg = 1 - current_standing_leg
    states = []

    logger.info(f"Frame {0}: Phase STARTING - Starting in state COMPLETED")

    for current_frame_idx in range(pose_data_3d.num_frames):
        states.append(current_state)

        # helper variables for readability
        active_knee_angle = knee_angles[current_frame_idx, current_moving_leg]
        standing_knee_angle = knee_angles[current_frame_idx, current_standing_leg]
        active_ankle_velocity = ankle_velocities[current_frame_idx, current_moving_leg]
        standing_ankle_velocity = ankle_velocities[current_frame_idx, current_standing_leg] # for logging only
        active_ankle_hip_offset = weight_transfer[current_frame_idx, current_moving_leg]
        standing_ankle_hip_offset = weight_transfer[current_frame_idx, current_standing_leg] # for logging only

        logger.info(f""Frame {current_frame_idx}: Current state {current_state.name}\n
                    active knee angle {active_knee_angle:.2f}, standing knee angle {standing_knee_angle:.2f}\n
                    active ankle velocity {active_ankle_velocity:.2f}, standing ankle velocity {standing_ankle_velocity:.2f}\n
                    active ankle-hip offset {active_ankle_hip_offset:.2f}, standing ankle-hip offset {standing_ankle_hip_offset:.2f}"")


        if current_state == WalkingState.RELEASE:
            # ACTION: Back leg pushes; hips drive forward.
            # KNEE-ACTIVE: 165 -> 150 (Unlocking to drive)
            # KNEE-STANDING: 180 (Anchor)
            # TRANSITION: Moving ankle velocity exceeds a threshold, indicating movement.

            if active_knee_angle > deg_to_rad(170):
                logger.info(f"Frame {current_frame_idx}: Phase RELEASE - Active knee angle {active_knee_angle:.2f} exceeds 170 degrees")
            if standing_knee_angle < deg_to_rad(170):
                logger.info(f"Frame {current_frame_idx}: Phase RELEASE - Standing knee angle {standing_knee_angle:.2f} below 170 degrees")

            if active_ankle_velocity > VELOCITY_THRESHOLD:
                logger.info(f"Frame {current_frame_idx}: Phase RELEASE transition to PASSING - ankle velocity {active_ankle_velocity:.2f} exceeds threshold")
                current_state = WalkingState.PASSING


        elif current_state == WalkingState.PASSING:
            # ACTION: "Bow and Arrow" (knees are closest together).
            # KNEE-ACTIVE: 150 -> 135 (Deep flexion for floor clearance)
            # KNEE-STANDING: 180 (The "Bow" pillar)
            # TRANSITION: Moving ankle passes the standing ankle on the motion axis.

            if active_knee_angle > deg_to_rad(150):
                logger.info(f"Frame {current_frame_idx}: Phase PASSING - Active knee angle {active_knee_angle:.2f} exceeds 150 degrees")
            if standing_knee_angle < deg_to_rad(170):
                logger.info(f"Frame {current_frame_idx}: Phase PASSING - Standing knee angle {standing_knee_angle:.2f} below 170 degrees")

            if is_past_standing_leg(current_frame_idx):
                logger.info(f"Frame {current_frame_idx}: Phase PASSING transition to EXTENSION - active ankle has passed standing ankle")
                current_state = WalkingState.EXTENSION


        elif current_state == WalkingState.EXTENSION:
            # ACTION: Reaching for the floor; active straightening begins.
            # KNEE-ACTIVE: 135 -> 175 (The "Arrow" firing forward)
            # KNEE-STANDING: 180 (The Anchor)
            # TRANSITION: Ankle velocity drops toward zero, indicating the foot has hit the floor.

            if active_knee_angle > deg_to_rad(170):
                logger.info(f"Frame {current_frame_idx}: Phase EXTENSION - Active knee angle {active_knee_angle:.2f} exceeds 170 degrees")
            if standing_knee_angle < deg_to_rad(170):
                logger.info(f"Frame {current_frame_idx}: Phase EXTENSION - Standing knee angle {standing_knee_angle:.2f} below 170 degrees")

            if active_ankle_velocity < VELOCITY_THRESHOLD:
                logger.info(f"Frame {current_frame_idx}: Phase EXTENSION transition to ARRIVAL - ankle velocity {active_ankle_velocity:.2f} below threshold {VELOCITY_THRESHOLD:.2f}")
                current_state = WalkingState.ARRIVAL


        elif current_state == WalkingState.ARRIVAL:
            # ACTION: The "Double Straight" check; weight is splitting.
            # KNEE-ACTIVE: 175 -> 180 (Snap to full extension)
            # KNEE-STANDING: 180 (Must still be straight for International style)
            # TRANSITION: Center of Mass (Hips) moves directly over the active ankle.

            if active_knee_angle < deg_to_rad(170):
                logger.info(f"Frame {current_frame_idx}: Phase ARRIVAL - Active knee angle {active_knee_angle:.2f} below 170 degrees")
            if standing_knee_angle < deg_to_rad(170):
                logger.info(f"Frame {current_frame_idx}: Phase EXTENSION - Standing knee angle {standing_knee_angle:.2f} below 170 degrees")

            if active_ankle_hip_offset < HIP_DISTANCE_THRESHOLD: 
                logger.info(f"Frame {current_frame_idx}: Phase ARRIVAL transition to COMPLETED - ankle-hip offset {active_ankle_hip_offset:.2f} below threshold {HIP_DISTANCE_THRESHOLD:.2f}")
                
                current_state = WalkingState.COMPLETED
                
                # legs swap roles
                current_standing_leg, current_moving_leg = current_moving_leg, current_standing_leg


        elif current_state == WalkingState.COMPLETED:
            # ACTION: Hips settle laterally (Cuban Motion); back leg prepares to release.
            # KNEE-NEW_STANDING: 180 (Locked over the weight)
            # KNEE-NEW_MOVING: 180 -> 165 (Softening because weight has left it)
            # TRANSITION: Hips move forward again, breaking the vertical alignment.

            if standing_ankle_hip_offset > HIP_DISTANCE_THRESHOLD: 

                current_state = WalkingState.RELEASE


        else:
            raise ValueError(f"Unknown state: {current_state}")
  
 
 """