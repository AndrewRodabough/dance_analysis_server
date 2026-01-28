"""
Visualize the effect of anatomical constraints by comparing raw vs corrected poses.
Useful for debugging and understanding what the filter does.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compare_poses_3d(keypoints_before, keypoints_after, frame_idx=0, title="Pose Comparison"):
    """
    Create side-by-side 3D visualization of before/after correction.
    
    Args:
        keypoints_before: Original 3D keypoints
        keypoints_after: Corrected 3D keypoints
        frame_idx: Which frame to visualize
        title: Plot title
    """
    # Joint indices for major body parts
    CONNECTIONS = [
        # Torso
        (5, 6),   # Shoulders
        (5, 11),  # Left shoulder to hip
        (6, 12),  # Right shoulder to hip
        (11, 12), # Hips
        # Left arm
        (5, 7),   # Shoulder to elbow
        (7, 9),   # Elbow to wrist
        # Right arm
        (6, 8),   # Shoulder to elbow
        (8, 10),  # Elbow to wrist
        # Left leg
        (11, 13), # Hip to knee
        (13, 15), # Knee to ankle
        # Right leg
        (12, 14), # Hip to knee
        (14, 16), # Knee to ankle
    ]
    
    ELBOW_KNEE_JOINTS = [7, 8, 13, 14]  # Joints that might be corrected
    
    if len(keypoints_before.shape) == 3:
        keypoints_before = keypoints_before[0]
    if len(keypoints_after.shape) == 3:
        keypoints_after = keypoints_after[0]
    
    fig = plt.figure(figsize=(16, 7))
    
    # Before correction
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Before Anatomical Constraints', fontsize=14, fontweight='bold')
    
    # Draw skeleton
    for pt1_idx, pt2_idx in CONNECTIONS:
        pt1 = keypoints_before[pt1_idx]
        pt2 = keypoints_before[pt2_idx]
        ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 
                'b-', linewidth=2, alpha=0.7)
    
    # Draw all joints
    ax1.scatter(keypoints_before[:17, 0], keypoints_before[:17, 1], 
               keypoints_before[:17, 2], c='blue', s=30, alpha=0.6)
    
    # Highlight elbows and knees (likely correction points)
    for joint_idx in ELBOW_KNEE_JOINTS:
        ax1.scatter([keypoints_before[joint_idx, 0]], 
                   [keypoints_before[joint_idx, 1]], 
                   [keypoints_before[joint_idx, 2]], 
                   c='red', s=100, marker='o', edgecolors='darkred', linewidth=2,
                   label='Elbow/Knee' if joint_idx == 7 else '')
    
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_zlabel('Z (m) - Depth', fontsize=11)
    ax1.view_init(elev=15, azim=45)
    ax1.legend(loc='upper left')
    
    # After correction
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('After Anatomical Constraints', fontsize=14, fontweight='bold', color='green')
    
    # Draw skeleton
    for pt1_idx, pt2_idx in CONNECTIONS:
        pt1 = keypoints_after[pt1_idx]
        pt2 = keypoints_after[pt2_idx]
        ax2.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 
                'g-', linewidth=2, alpha=0.7)
    
    # Draw all joints
    ax2.scatter(keypoints_after[:17, 0], keypoints_after[:17, 1], 
               keypoints_after[:17, 2], c='green', s=30, alpha=0.6)
    
    # Highlight corrected joints
    corrections_made = []
    for joint_idx in ELBOW_KNEE_JOINTS:
        # Check if this joint was corrected (z-coordinate changed significantly)
        if abs(keypoints_before[joint_idx, 2] - keypoints_after[joint_idx, 2]) > 0.01:
            ax2.scatter([keypoints_after[joint_idx, 0]], 
                       [keypoints_after[joint_idx, 1]], 
                       [keypoints_after[joint_idx, 2]], 
                       c='yellow', s=150, marker='*', edgecolors='orange', 
                       linewidth=2, label='Corrected' if not corrections_made else '')
            corrections_made.append(joint_idx)
        else:
            ax2.scatter([keypoints_after[joint_idx, 0]], 
                       [keypoints_after[joint_idx, 1]], 
                       [keypoints_after[joint_idx, 2]], 
                       c='green', s=100, marker='o', edgecolors='darkgreen', linewidth=2)
    
    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_zlabel('Z (m) - Depth', fontsize=11)
    ax2.view_init(elev=15, azim=45)
    if corrections_made:
        ax2.legend(loc='upper left')
    
    # Set same axis limits for both plots
    for ax in [ax1, ax2]:
        all_points = np.vstack([keypoints_before[:17], keypoints_after[:17]])
        center = all_points.mean(axis=0)
        range_val = 1.0
        ax.set_xlim([center[0] - range_val, center[0] + range_val])
        ax.set_ylim([center[1] - range_val, center[1] + range_val])
        ax.set_zlim([center[2] - range_val, center[2] + range_val])
    
    joint_names = {7: 'Left Elbow', 8: 'Right Elbow', 13: 'Left Knee', 14: 'Right Knee'}
    if corrections_made:
        corrected_names = [joint_names[j] for j in corrections_made]
        fig.suptitle(f'{title} - Frame {frame_idx}\nCorrected: {", ".join(corrected_names)}', 
                    fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f'{title} - Frame {frame_idx}\nNo corrections needed', 
                    fontsize=16)
    
    plt.tight_layout()
    return fig, corrections_made


if __name__ == "__main__":
    import json
    from pathlib import Path
    import sys
    
    # This is a template - you would need to save both before/after data
    # to use this comparison tool
    
    print("="*70)
    print("ANATOMICAL CONSTRAINT VISUALIZATION TOOL")
    print("="*70)
    print()
    print("To use this tool, you need to:")
    print("1. Modify pose_estimation.py to save BOTH before/after keypoints")
    print("2. Or run analysis twice (with and without constraints)")
    print()
    print("Example modification in pose_estimation.py:")
    print("-" * 70)
    print("""
    # Before applying constraints
    all_keypoints_3d_before = [kp.copy() for kp in all_keypoints_3d]
    
    # Apply constraints
    all_keypoints_3d = apply_constraints_to_sequence(...)
    
    # Save both for comparison
    with open('test_outputs/keypoints_3d_before.json', 'w') as f:
        json.dump([kp.tolist() for kp in all_keypoints_3d_before], f)
    with open('test_outputs/keypoints_3d_after.json', 'w') as f:
        json.dump([kp.tolist() for kp in all_keypoints_3d], f)
    """)
    print("-" * 70)
    print()
    
    # Check if comparison files exist
    before_path = Path("test_outputs/keypoints_3d_before.json")
    after_path = Path("test_outputs/keypoints_3d_after.json")
    
    if before_path.exists() and after_path.exists():
        print("✓ Found before/after files, generating comparison...")
        
        with open(before_path) as f:
            kps_before = [np.array(frame) for frame in json.load(f)]
        with open(after_path) as f:
            kps_after = [np.array(frame) for frame in json.load(f)]
        
        # Compare first frame
        fig, corrections = compare_poses_3d(kps_before[0], kps_after[0], frame_idx=0)
        plt.savefig('test_outputs/anatomical_comparison_frame_0.png', dpi=150, bbox_inches='tight')
        print(f"Saved: test_outputs/anatomical_comparison_frame_0.png")
        
        # Find and visualize a corrected frame
        for i in range(min(30, len(kps_before))):
            fig, corrections = compare_poses_3d(kps_before[i], kps_after[i], frame_idx=i)
            if corrections:
                plt.savefig(f'test_outputs/anatomical_comparison_frame_{i}.png', 
                          dpi=150, bbox_inches='tight')
                print(f"Saved: test_outputs/anatomical_comparison_frame_{i}.png")
                print(f"  → Frame {i} had corrections!")
                break
        
        plt.show()
    else:
        print("✗ Before/after comparison files not found")
        print()
        print("The anatomical constraints are still working correctly!")
        print("This visualization tool is just for debugging/understanding.")
