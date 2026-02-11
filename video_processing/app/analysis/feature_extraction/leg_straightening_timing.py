"""Analyze leg straightening timing for rumba/cha-cha walk using hips, knees, and ankles."""

import logging
import numpy as np
from typing import Dict, Tuple, List

from shared.skeletons.pose_data import VectorizedPoseData

logger = logging.getLogger(__name__)

# Human3.6M joint indices
H36M_PELVIS = 0
H36M_R_HIP = 1
H36M_L_HIP = 4
H36M_R_KNEE = 2
H36M_L_KNEE = 5
H36M_R_ANKLE = 3
H36M_L_ANKLE = 6

# Analysis thresholds
VELOCITY_THRESHOLD = 0.05  # Below this = "planted" foot
HIP_DISTANCE_THRESHOLD = 0.1  # Below this = weight fully transferred
KNEE_STRAIGHT_ANGLE = np.pi * 0.9  # Above this = "straight" leg (162 degrees)


def extract_leg_straightening_timing(pose_data_3d: VectorizedPoseData) -> Dict[str, np.ndarray]:
    """
    Analyze leg straightening timing for rumba/cha-cha walk.
    
    Uses ankle velocity (floor contact proxy), hip-ankle distance (weight transfer proxy),
    and knee angle (leg straightness) to identify dance phases.
    
    Args:
        pose_data_3d: VectorizedPoseData with 3D keypoints (Human3.6M skeleton)
        
    Returns:
        Dictionary with timing analysis:
        - 'left_phases': Phase classification for left leg [frames]
        - 'right_phases': Phase classification for right leg [frames] 
        - 'left_metrics': Detailed metrics for left leg
        - 'right_metrics': Detailed metrics for right leg
        - 'summary': Overall timing analysis
    """
    try:
        # Validate skeleton type
        if pose_data_3d.num_joints != 17:
            logger.warning(
                f"Expected Human3.6M skeleton (17 joints), got {pose_data_3d.num_joints}. "
                "Leg analysis may not work correctly."
            )
        
        skeleton = pose_data_3d.skeleton
        num_frames = skeleton.num_frames
        
        if num_frames < 2:
            logger.warning("Need at least 2 frames for velocity analysis")
            return _empty_result()
        
        # Validate joint indices exist
        max_joint_idx = max(H36M_L_HIP, H36M_R_HIP, H36M_L_KNEE, H36M_R_KNEE, 
                           H36M_L_ANKLE, H36M_R_ANKLE)
        if skeleton.num_joints <= max_joint_idx:
            logger.error(f"Skeleton has {skeleton.num_joints} joints but need at least {max_joint_idx + 1}")
            return _empty_result()
        
        # Apply smoothing to reduce pose estimation jitter
        smoothed_data = smooth_skeleton_data(skeleton.data)
        
        # Temporarily replace skeleton data with smoothed version for velocity calculation
        original_data = skeleton.data.copy()
        skeleton.data = smoothed_data
        
        # Get joint velocities for ankle movement analysis
        velocities = skeleton.get_joint_velocities()
        
        # Restore original data
        skeleton.data = original_data
        
        # Analyze each leg separately
        left_analysis = _analyze_single_leg(
            skeleton, velocities, num_frames, "left",
            H36M_L_HIP, H36M_L_KNEE, H36M_L_ANKLE
        )
        
        right_analysis = _analyze_single_leg(
            skeleton, velocities, num_frames, "right", 
            H36M_R_HIP, H36M_R_KNEE, H36M_R_ANKLE
        )
        
        # Generate summary
        summary = _generate_timing_summary(left_analysis, right_analysis, num_frames)
        
        return {
            'left_phases': left_analysis['phases'],
            'right_phases': right_analysis['phases'], 
            'left_metrics': left_analysis['metrics'],
            'right_metrics': right_analysis['metrics'],
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Error analyzing leg straightening timing: {e}")
        raise


def _analyze_single_leg(skeleton, velocities, num_frames: int, leg_name: str,
                       hip_idx: int, knee_idx: int, ankle_idx: int) -> Dict:
    """Analyze timing phases for a single leg."""
    
    # 1. ANKLE VELOCITY - Proxy for floor contact
    ankle_velocity = velocities[:, ankle_idx, :]  # Shape: (frames, 3)
    ankle_speed = np.linalg.norm(ankle_velocity, axis=1)  # Shape: (frames,)
    
    # 2. HIP-ANKLE HORIZONTAL DISTANCE - Proxy for weight transfer
    hip_pos = skeleton.data[:, hip_idx, :]  # Shape: (frames, 3)
    ankle_pos = skeleton.data[:, ankle_idx, :]  # Shape: (frames, 3)
    
    # Calculate horizontal distance (ignore Y/vertical)
    hip_ankle_diff = hip_pos - ankle_pos  # Shape: (frames, 3)
    horizontal_distance = np.sqrt(hip_ankle_diff[:, 0]**2 + hip_ankle_diff[:, 2]**2)  # XZ plane
    
    # 3. KNEE ANGLE - Proxy for leg straightness
    try:
        # Get knee angles using hip-knee-ankle triplet
        hip_name = skeleton.idx_to_name[hip_idx]
        knee_name = skeleton.idx_to_name[knee_idx] 
        ankle_name = skeleton.idx_to_name[ankle_idx]
        
        knee_angles = skeleton.get_angle(hip_name, knee_name, ankle_name)  # Shape: (frames,)
    except Exception as e:
        logger.warning(f"Could not calculate knee angles for {leg_name} leg: {e}")
        knee_angles = np.full(num_frames, np.pi, dtype=np.float32)  # Assume straight
    
    # 4. PHASE CLASSIFICATION
    phases = _classify_dance_phases(ankle_speed, horizontal_distance, knee_angles)
    
    # 5. METRICS COMPILATION
    metrics = {
        'ankle_speed': ankle_speed,
        'hip_ankle_distance': horizontal_distance,
        'knee_angles': knee_angles,
        'is_planted': ankle_speed < VELOCITY_THRESHOLD,
        'weight_transferred': horizontal_distance < HIP_DISTANCE_THRESHOLD,
        'leg_straight': knee_angles > KNEE_STRAIGHT_ANGLE
    }
    
    return {
        'phases': phases,
        'metrics': metrics
    }


def _classify_dance_phases(ankle_speed: np.ndarray, hip_distance: np.ndarray, 
                          knee_angles: np.ndarray) -> np.ndarray:
    """
    Classify dance phases based on the three proxies.
    
    Phase encoding:
    0 = Swing (moving leg forward)
    1 = Contact (touching floor) 
    2 = Action (rolling onto foot)
    3 = Arrival (standing on leg)
    """
    phases = np.zeros(len(ankle_speed), dtype=np.int32)
    
    is_planted = ankle_speed < VELOCITY_THRESHOLD
    weight_transferred = hip_distance < HIP_DISTANCE_THRESHOLD
    leg_straight = knee_angles > KNEE_STRAIGHT_ANGLE
    
    for i in range(len(phases)):
        if not is_planted[i]:
            # High velocity = Swing phase
            phases[i] = 0  # Swing
        elif is_planted[i] and not weight_transferred[i]:
            # Planted but hip not over ankle = Contact/Action
            if leg_straight[i]:
                phases[i] = 2  # Action (rolling onto straight leg)
            else:
                phases[i] = 1  # Contact (just touched, leg not straight)
        else:
            # Planted and weight transferred = Arrival
            phases[i] = 3  # Arrival
    
    return phases


def _generate_timing_summary(left_analysis: Dict, right_analysis: Dict, 
                           num_frames: int) -> Dict:
    """Generate phase-conditional summary statistics for timing analysis."""
    
    summary = {}
    
    for side, data in [('left', left_analysis), ('right', right_analysis)]:
        phases = data['phases']
        metrics = data['metrics']
        
        # Phase distribution percentages
        phase_names = ['swing', 'contact', 'action', 'arrival']
        for phase_idx, phase_name in enumerate(phase_names):
            count = np.sum(phases == phase_idx)
            summary[f'{side}_{phase_name}_pct'] = (count / num_frames) * 100
        
        # Extract indices for phase-specific analysis
        arrival_indices = np.where(phases == 3)[0]  # 3 = Arrival (Full Stance)
        contact_indices = np.where(phases == 1)[0]  # 1 = Contact (Initial Landing)
        action_indices = np.where(phases == 2)[0]   # 2 = Action (Rolling through)
        
        # Calculate straightness ONLY during Arrival (The "Lock")
        if len(arrival_indices) > 0:
            avg_arrival_angle = np.mean(metrics['knee_angles'][arrival_indices])
            summary[f'{side}_arrival_straightness_deg'] = float(np.degrees(avg_arrival_angle))
        else:
            summary[f'{side}_arrival_straightness_deg'] = 0.0
            
        # Calculate bend ONLY during Contact (The "Cushion")
        if len(contact_indices) > 0:
            avg_contact_angle = np.mean(metrics['knee_angles'][contact_indices])
            summary[f'{side}_contact_bend_deg'] = float(np.degrees(avg_contact_angle))
        else:
            summary[f'{side}_contact_bend_deg'] = 0.0
            
        # Calculate roll-through quality during Action
        if len(action_indices) > 0:
            avg_action_distance = np.mean(metrics['hip_ankle_distance'][action_indices])
            summary[f'{side}_action_weight_transfer'] = float(avg_action_distance)
        else:
            summary[f'{side}_action_weight_transfer'] = 0.0
            
        # Overall movement quality
        summary[f'{side}_avg_ankle_speed'] = float(np.mean(metrics['ankle_speed']))
        summary[f'{side}_planted_pct'] = float(np.mean(metrics['is_planted']) * 100)
    
    return summary


def smooth_skeleton_data(skeleton_data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply simple moving average to reduce skeletal jitter from pose estimation.
    
    Args:
        skeleton_data: Shape (frames, joints, 3) - raw keypoint data
        window_size: Size of smoothing window (must be odd)
        
    Returns:
        Smoothed skeleton data with same shape
    """
    num_frames = skeleton_data.shape[0]
    
    # If we have fewer frames than window size, just return original data
    if num_frames < window_size:
        logger.debug(f"Skipping smoothing: {num_frames} frames < {window_size} window size")
        return skeleton_data.copy()
        
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
        
    kernel = np.ones(window_size) / window_size
    smoothed_data = np.zeros_like(skeleton_data)
    
    # Apply along the time axis (axis 0)
    for joint in range(skeleton_data.shape[1]):
        for coord in range(3):
            # Use 'same' mode to preserve original length
            smoothed_data[:, joint, coord] = np.convolve(
                skeleton_data[:, joint, coord], kernel, mode='same'
            )
    
    return smoothed_data


def _empty_result() -> Dict:
    """Return empty result for edge cases."""
    return {
        'left_phases': np.array([], dtype=np.int32),
        'right_phases': np.array([], dtype=np.int32),
        'left_metrics': {},
        'right_metrics': {},
        'summary': {}
    }


def get_leg_timing_interpretation(phases: np.ndarray) -> List[str]:
    """
    Convert numeric phase codes to human-readable descriptions.
    
    Args:
        phases: Array of phase codes (0=swing, 1=contact, 2=action, 3=arrival)
        
    Returns:
        List of phase names for each frame
    """
    phase_names = ['swing', 'contact', 'action', 'arrival']
    return [phase_names[phase] for phase in phases]


def analyze_step_timing(left_phases: np.ndarray, right_phases: np.ndarray) -> Dict:
    """
    Analyze the timing relationship between left and right legs.
    
    Args:
        left_phases: Phase sequence for left leg
        right_phases: Phase sequence for right leg
        
    Returns:
        Dictionary with step timing analysis
    """
    # Find phase transitions (when phase changes)
    left_transitions = np.where(np.diff(left_phases) != 0)[0] + 1
    right_transitions = np.where(np.diff(right_phases) != 0)[0] + 1
    
    # Count simultaneous arrivals (both legs in arrival phase)
    both_arrival = np.sum((left_phases == 3) & (right_phases == 3))
    
    # Alternation quality (legs should be in different phases)
    alternation_frames = np.sum(left_phases != right_phases)
    alternation_pct = (alternation_frames / len(left_phases)) * 100
    
    return {
        'left_transition_count': len(left_transitions),
        'right_transition_count': len(right_transitions), 
        'simultaneous_arrival_frames': int(both_arrival),
        'alternation_percentage': float(alternation_pct),
        'left_transitions': left_transitions.tolist(),
        'right_transitions': right_transitions.tolist()
    }


def generate_coach_feedback(summary: Dict) -> List[str]:
    """
    Generate human-readable dance coaching feedback based on phase analysis.
    
    Args:
        summary: Dictionary from _generate_timing_summary with phase-conditional metrics
        
    Returns:
        List of coaching feedback strings
    """
    feedback = []
    
    # Target angles for good technique
    target_straight = 170.0  # Degrees - minimum for "locked" leg
    target_contact_min = 150.0  # Degrees - minimum bend for soft landing
    target_contact_max = 165.0  # Degrees - maximum bend (not sitting)
    
    # 1. Check Stance Leg (Arrival Phase) - Should be straight
    for side in ['left', 'right']:
        arrival_angle = summary.get(f'{side}_arrival_straightness_deg', 0)
        
        if arrival_angle < 160:
            feedback.append(
                f"⚠️ {side.title()} Leg: You are standing on a bent knee ({arrival_angle:.0f}°). "
                "Push the knee back to create a strong line!"
            )
        elif arrival_angle < target_straight:
            feedback.append(
                f"ℹ️ {side.title()} Leg: Almost straight ({arrival_angle:.0f}°), "
                "but try to lock it more for better posture."
            )
        else:
            feedback.append(f"✅ {side.title()} Leg: Great straight leg line ({arrival_angle:.0f}°)!")
    
    # 2. Check Landing (Contact Phase) - Should be soft but not sitting
    for side in ['left', 'right']:
        contact_angle = summary.get(f'{side}_contact_bend_deg', 0)
        
        if contact_angle == 0.0:
            continue  # No contact phase detected
            
        if contact_angle > 175:
            feedback.append(
                f"⚠️ {side.title()} Leg: You are landing too stiff ({contact_angle:.0f}°). "
                "Soften the knee when you step to absorb impact."
            )
        elif contact_angle < target_contact_min:
            feedback.append(
                f"⚠️ {side.title()} Leg: Too much bend on landing ({contact_angle:.0f}°). "
                "Don't sit into the leg - maintain more height."
            )
        elif contact_angle <= target_contact_max:
            feedback.append(
                f"✅ {side.title()} Leg: Good soft landing ({contact_angle:.0f}°)!"
            )
    
    # 3. Check overall phase distribution
    for side in ['left', 'right']:
        arrival_pct = summary.get(f'{side}_arrival_pct', 0)
        contact_pct = summary.get(f'{side}_contact_pct', 0)
        
        if arrival_pct < 15:
            feedback.append(
                f"📊 {side.title()} Leg: Very little time standing on this leg ({arrival_pct:.0f}%). "
                "Work on weight transfer and commitment to each step."
            )
        elif contact_pct < 10:
            feedback.append(
                f"📊 {side.title()} Leg: Landing very briefly ({contact_pct:.0f}%). "
                "This could indicate good control or insufficient settling time."
            )
    
    # 4. Movement quality assessment
    avg_left_speed = summary.get('left_avg_ankle_speed', 0)
    avg_right_speed = summary.get('right_avg_ankle_speed', 0)
    
    if avg_left_speed < 0.02 and avg_right_speed < 0.02:
        feedback.append(
            "📊 Overall: Very controlled movement - good stability! "
            "Consider adding more dynamic range if this is choreographed dance."
        )
    elif avg_left_speed > 0.1 or avg_right_speed > 0.1:
        feedback.append(
            "📊 Overall: High movement speed detected. "
            "Ensure you're finding moments of stillness and balance."
        )
    
    if not feedback:
        feedback.append("✅ Overall: Movement analysis complete - no major issues detected!")
    
    return feedback