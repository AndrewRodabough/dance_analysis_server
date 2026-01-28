"""
Test script to visualize anatomical constraint corrections.
Run this after analyzing a video to see which joints were corrected.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def load_keypoints(filepath):
    """Load keypoints from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [np.array(frame) for frame in data]


def visualize_joint_corrections(keypoints_3d_list, frame_idx=0):
    """
    Visualize specific joints to check for impossible bends.
    Shows before/after if you have both datasets.
    """
    # Joint indices
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 7
    LEFT_WRIST = 9
    RIGHT_SHOULDER = 6
    RIGHT_ELBOW = 8
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    LEFT_KNEE = 13
    LEFT_ANKLE = 15
    RIGHT_HIP = 12
    RIGHT_KNEE = 14
    RIGHT_ANKLE = 16
    
    if frame_idx >= len(keypoints_3d_list):
        print(f"Frame {frame_idx} not available. Total frames: {len(keypoints_3d_list)}")
        return
    
    kps = keypoints_3d_list[frame_idx]
    if len(kps.shape) == 3:
        kps = kps[0]  # Take first person
    
    fig = plt.figure(figsize=(15, 5))
    
    # Left arm
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title(f'Left Arm (Frame {frame_idx})')
    
    left_arm = kps[[LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]]
    ax1.plot(left_arm[:, 0], left_arm[:, 1], left_arm[:, 2], 'b-o', linewidth=2, markersize=8)
    ax1.scatter([kps[LEFT_SHOULDER, 0]], [kps[LEFT_SHOULDER, 1]], [kps[LEFT_SHOULDER, 2]], 
                c='red', s=100, label='Shoulder')
    ax1.scatter([kps[LEFT_ELBOW, 0]], [kps[LEFT_ELBOW, 1]], [kps[LEFT_ELBOW, 2]], 
                c='green', s=100, label='Elbow')
    ax1.scatter([kps[LEFT_WRIST, 0]], [kps[LEFT_WRIST, 1]], [kps[LEFT_WRIST, 2]], 
                c='blue', s=100, label='Wrist')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m) - Depth')
    ax1.legend()
    ax1.view_init(elev=15, azim=45)
    
    # Right arm
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title(f'Right Arm (Frame {frame_idx})')
    
    right_arm = kps[[RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]]
    ax2.plot(right_arm[:, 0], right_arm[:, 1], right_arm[:, 2], 'r-o', linewidth=2, markersize=8)
    ax2.scatter([kps[RIGHT_SHOULDER, 0]], [kps[RIGHT_SHOULDER, 1]], [kps[RIGHT_SHOULDER, 2]], 
                c='red', s=100, label='Shoulder')
    ax2.scatter([kps[RIGHT_ELBOW, 0]], [kps[RIGHT_ELBOW, 1]], [kps[RIGHT_ELBOW, 2]], 
                c='green', s=100, label='Elbow')
    ax2.scatter([kps[RIGHT_WRIST, 0]], [kps[RIGHT_WRIST, 1]], [kps[RIGHT_WRIST, 2]], 
                c='blue', s=100, label='Wrist')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m) - Depth')
    ax2.legend()
    ax2.view_init(elev=15, azim=45)
    
    # Left leg
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title(f'Left Leg (Frame {frame_idx})')
    
    left_leg = kps[[LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]]
    ax3.plot(left_leg[:, 0], left_leg[:, 1], left_leg[:, 2], 'g-o', linewidth=2, markersize=8)
    ax3.scatter([kps[LEFT_HIP, 0]], [kps[LEFT_HIP, 1]], [kps[LEFT_HIP, 2]], 
                c='red', s=100, label='Hip')
    ax3.scatter([kps[LEFT_KNEE, 0]], [kps[LEFT_KNEE, 1]], [kps[LEFT_KNEE, 2]], 
                c='green', s=100, label='Knee')
    ax3.scatter([kps[LEFT_ANKLE, 0]], [kps[LEFT_ANKLE, 1]], [kps[LEFT_ANKLE, 2]], 
                c='blue', s=100, label='Ankle')
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m) - Depth')
    ax3.legend()
    ax3.view_init(elev=15, azim=45)
    
    plt.tight_layout()
    plt.savefig(f'test_outputs/joint_check_frame_{frame_idx}.png', dpi=150)
    print(f"Saved visualization to test_outputs/joint_check_frame_{frame_idx}.png")
    plt.show()


def check_impossible_bends_in_sequence(keypoints_3d_list):
    """
    Scan through entire sequence to find potential impossible bends.
    Reports statistics on which joints and frames have issues.
    """
    from app.analysis.pose_estimation.anatomical_constraints import (
        detect_impossible_elbow, detect_impossible_knee, BodyJoint
    )
    
    stats = {
        'left_elbow': 0,
        'right_elbow': 0,
        'left_knee': 0,
        'right_knee': 0
    }
    
    problematic_frames = {
        'left_elbow': [],
        'right_elbow': [],
        'left_knee': [],
        'right_knee': []
    }
    
    for frame_idx, kps in enumerate(keypoints_3d_list):
        if len(kps.shape) == 3:
            kps = kps[0]  # Take first person
        
        # Check left elbow
        try:
            if detect_impossible_elbow(
                kps[BodyJoint.LEFT_SHOULDER],
                kps[BodyJoint.LEFT_ELBOW],
                kps[BodyJoint.LEFT_WRIST],
                is_left=True
            ):
                stats['left_elbow'] += 1
                problematic_frames['left_elbow'].append(frame_idx)
        except:
            pass
        
        # Check right elbow
        try:
            if detect_impossible_elbow(
                kps[BodyJoint.RIGHT_SHOULDER],
                kps[BodyJoint.RIGHT_ELBOW],
                kps[BodyJoint.RIGHT_WRIST],
                is_left=False
            ):
                stats['right_elbow'] += 1
                problematic_frames['right_elbow'].append(frame_idx)
        except:
            pass
        
        # Check left knee
        try:
            if detect_impossible_knee(
                kps[BodyJoint.LEFT_HIP],
                kps[BodyJoint.LEFT_KNEE],
                kps[BodyJoint.LEFT_ANKLE],
                is_left=True
            ):
                stats['left_knee'] += 1
                problematic_frames['left_knee'].append(frame_idx)
        except:
            pass
        
        # Check right knee
        try:
            if detect_impossible_knee(
                kps[BodyJoint.RIGHT_HIP],
                kps[BodyJoint.RIGHT_KNEE],
                kps[BodyJoint.RIGHT_ANKLE],
                is_left=False
            ):
                stats['right_knee'] += 1
                problematic_frames['right_knee'].append(frame_idx)
        except:
            pass
    
    print("\n" + "="*60)
    print("IMPOSSIBLE BEND DETECTION REPORT")
    print("="*60)
    print(f"Total frames analyzed: {len(keypoints_3d_list)}")
    print()
    
    for joint, count in stats.items():
        if count > 0:
            percentage = (count / len(keypoints_3d_list)) * 100
            print(f"{joint.upper():20s}: {count:4d} frames ({percentage:.1f}%)")
            
            # Show first few problematic frames
            frames = problematic_frames[joint][:5]
            print(f"                      First occurrences: {frames}")
    
    if sum(stats.values()) == 0:
        print("✓ No impossible bends detected!")
    
    print("="*60 + "\n")
    
    return stats, problematic_frames


if __name__ == "__main__":
    import sys
    
    # Load the keypoints
    keypoints_path = Path("test_outputs/keypoints_3d.json")
    
    if not keypoints_path.exists():
        print(f"Error: {keypoints_path} not found")
        print("Run the analysis first: python test_scripts/test_api_video_upload.py <video_file>")
        sys.exit(1)
    
    print(f"Loading keypoints from {keypoints_path}...")
    keypoints_3d_list = load_keypoints(keypoints_path)
    print(f"Loaded {len(keypoints_3d_list)} frames")
    
    # Check for impossible bends
    stats, problematic_frames = check_impossible_bends_in_sequence(keypoints_3d_list)
    
    # Visualize a few frames
    if len(keypoints_3d_list) > 0:
        print("\nGenerating visualizations...")
        
        # Visualize first frame
        visualize_joint_corrections(keypoints_3d_list, frame_idx=0)
        
        # Visualize a problematic frame if any exist
        all_problematic = []
        for frames in problematic_frames.values():
            all_problematic.extend(frames)
        
        if all_problematic:
            problem_frame = all_problematic[0]
            print(f"\nVisualizing problematic frame {problem_frame}...")
            visualize_joint_corrections(keypoints_3d_list, frame_idx=problem_frame)
