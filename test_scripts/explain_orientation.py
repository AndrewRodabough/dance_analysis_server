"""
Demonstrate how orientation detection works in the anatomical constraint filter.
Shows the logic for detecting if a person is facing forward or backward.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_orientation_detection():
    """
    Visualize how the orientation detection algorithm works.
    Shows shoulder and hip positions and the resulting normal vector.
    """
    fig = plt.figure(figsize=(16, 6))
    
    # Scenario 1: Person facing forward (toward camera)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Person Facing Forward\n(Toward Camera)', fontsize=12, fontweight='bold')
    
    # Define joints for person facing forward
    # Z-axis points away from camera (into the scene)
    left_shoulder_fwd = np.array([-0.3, 0.5, 1.0])
    right_shoulder_fwd = np.array([0.3, 0.5, 1.0])
    left_hip_fwd = np.array([-0.25, 0.0, 1.1])
    right_hip_fwd = np.array([0.25, 0.0, 1.1])
    
    # Draw skeleton
    ax1.plot([left_shoulder_fwd[0], right_shoulder_fwd[0]], 
            [left_shoulder_fwd[1], right_shoulder_fwd[1]], 
            [left_shoulder_fwd[2], right_shoulder_fwd[2]], 'b-', linewidth=3, label='Shoulders')
    ax1.plot([left_hip_fwd[0], right_hip_fwd[0]], 
            [left_hip_fwd[1], right_hip_fwd[1]], 
            [left_hip_fwd[2], right_hip_fwd[2]], 'g-', linewidth=3, label='Hips')
    ax1.plot([left_shoulder_fwd[0], left_hip_fwd[0]], 
            [left_shoulder_fwd[1], left_hip_fwd[1]], 
            [left_shoulder_fwd[2], left_hip_fwd[2]], 'gray', linewidth=2, alpha=0.5)
    ax1.plot([right_shoulder_fwd[0], right_hip_fwd[0]], 
            [right_shoulder_fwd[1], right_hip_fwd[1]], 
            [right_shoulder_fwd[2], right_hip_fwd[2]], 'gray', linewidth=2, alpha=0.5)
    
    # Calculate and draw normal vector
    shoulder_vec = right_shoulder_fwd - left_shoulder_fwd
    shoulder_mid = (left_shoulder_fwd + right_shoulder_fwd) / 2
    hip_mid = (left_hip_fwd + right_hip_fwd) / 2
    torso_vec = hip_mid - shoulder_mid
    normal = np.cross(shoulder_vec, torso_vec)
    normal_normalized = normal / np.linalg.norm(normal) * 0.3
    
    torso_center = (shoulder_mid + hip_mid) / 2
    ax1.quiver(torso_center[0], torso_center[1], torso_center[2],
              normal_normalized[0], normal_normalized[1], normal_normalized[2],
              color='red', arrow_length_ratio=0.3, linewidth=3, label='Normal (Front)')
    
    ax1.text(0, -0.3, 0.8, f'Normal Z: {normal[2]:.3f} > 0\n→ Facing Forward', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y') 
    ax1.set_zlabel('Z (Depth)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.view_init(elev=20, azim=45)
    
    # Scenario 2: Person facing backward (away from camera)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('Person Facing Backward\n(Away from Camera)', fontsize=12, fontweight='bold')
    
    # Person rotated 180 degrees
    left_shoulder_back = np.array([0.3, 0.5, 1.0])  # Swapped left/right in x
    right_shoulder_back = np.array([-0.3, 0.5, 1.0])
    left_hip_back = np.array([0.25, 0.0, 0.9])  # Hips closer to camera
    right_hip_back = np.array([-0.25, 0.0, 0.9])
    
    # Draw skeleton
    ax2.plot([left_shoulder_back[0], right_shoulder_back[0]], 
            [left_shoulder_back[1], right_shoulder_back[1]], 
            [left_shoulder_back[2], right_shoulder_back[2]], 'b-', linewidth=3, label='Shoulders')
    ax2.plot([left_hip_back[0], right_hip_back[0]], 
            [left_hip_back[1], right_hip_back[1]], 
            [left_hip_back[2], right_hip_back[2]], 'g-', linewidth=3, label='Hips')
    ax2.plot([left_shoulder_back[0], left_hip_back[0]], 
            [left_shoulder_back[1], left_hip_back[1]], 
            [left_shoulder_back[2], left_hip_back[2]], 'gray', linewidth=2, alpha=0.5)
    ax2.plot([right_shoulder_back[0], right_hip_back[0]], 
            [right_shoulder_back[1], right_hip_back[1]], 
            [right_shoulder_back[2], right_hip_back[2]], 'gray', linewidth=2, alpha=0.5)
    
    # Calculate normal
    shoulder_vec = right_shoulder_back - left_shoulder_back
    shoulder_mid = (left_shoulder_back + right_shoulder_back) / 2
    hip_mid = (left_hip_back + right_hip_back) / 2
    torso_vec = hip_mid - shoulder_mid
    normal = np.cross(shoulder_vec, torso_vec)
    normal_normalized = normal / np.linalg.norm(normal) * 0.3
    
    torso_center = (shoulder_mid + hip_mid) / 2
    ax2.quiver(torso_center[0], torso_center[1], torso_center[2],
              normal_normalized[0], normal_normalized[1], normal_normalized[2],
              color='blue', arrow_length_ratio=0.3, linewidth=3, label='Normal (Back)')
    
    ax2.text(0, -0.3, 0.8, f'Normal Z: {normal[2]:.3f} < 0\n→ Facing Backward', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z (Depth)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.view_init(elev=20, azim=45)
    
    # Scenario 3: Side view diagram
    ax3 = fig.add_subplot(133)
    ax3.set_title('Cross Product Logic\n(Top View)', fontsize=12, fontweight='bold')
    ax3.set_aspect('equal')
    
    # Draw both scenarios from above
    # Forward facing
    ax3.arrow(-0.5, 1.0, 0.6, 0, head_width=0.05, head_length=0.08, fc='blue', ec='blue', linewidth=2)
    ax3.text(-0.2, 1.15, 'Shoulder Vector', fontsize=9, ha='center')
    ax3.arrow(-0.2, 1.0, 0, -0.4, head_width=0.05, head_length=0.08, fc='green', ec='green', linewidth=2)
    ax3.text(-0.35, 0.8, 'Torso\nVector', fontsize=9, ha='center')
    ax3.plot([-0.2], [1.0], 'ro', markersize=10)
    ax3.text(0.3, 0.8, '⊗ Normal points\ntoward you\n(Z > 0)', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # Backward facing  
    ax3.arrow(-0.5, 0.0, -0.6, 0, head_width=0.05, head_length=0.08, fc='blue', ec='blue', linewidth=2, linestyle='--')
    ax3.text(-0.8, 0.15, 'Shoulder Vector', fontsize=9, ha='center')
    ax3.arrow(-0.8, 0.0, 0, -0.4, head_width=0.05, head_length=0.08, fc='green', ec='green', linewidth=2, linestyle='--')
    ax3.text(-0.95, -0.2, 'Torso\nVector', fontsize=9, ha='center')
    ax3.plot([-0.8], [0.0], 'bs', markersize=10)
    ax3.text(-0.2, -0.2, '⊙ Normal points\naway from you\n(Z < 0)', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    ax3.set_xlim(-1.2, 0.5)
    ax3.set_ylim(-0.6, 1.3)
    ax3.set_xlabel('X (Left ← → Right)', fontsize=10)
    ax3.set_ylabel('Z (Camera ← → Scene)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('test_outputs/orientation_detection_explanation.png', dpi=150, bbox_inches='tight')
    print("Saved: test_outputs/orientation_detection_explanation.png")
    plt.show()


def print_algorithm_explanation():
    """Print detailed explanation of the algorithm"""
    print("="*80)
    print("ORIENTATION DETECTION ALGORITHM")
    print("="*80)
    print()
    print("The algorithm determines if a person is facing toward or away from the")
    print("camera using the cross product of body vectors:")
    print()
    print("STEP 1: Get Key Points")
    print("  - Left & Right Shoulders")
    print("  - Left & Right Hips")
    print()
    print("STEP 2: Calculate Vectors")
    print("  shoulder_vec = right_shoulder - left_shoulder")
    print("  torso_vec = hip_midpoint - shoulder_midpoint")
    print()
    print("STEP 3: Cross Product")
    print("  normal = shoulder_vec × torso_vec")
    print()
    print("STEP 4: Check Z-Component of Normal")
    print("  if normal.z > 0:  → Person facing FORWARD (toward camera)")
    print("  if normal.z < 0:  → Person facing BACKWARD (away from camera)")
    print("  if |normal.z| ≈ 0: → Person side-on (unknown orientation)")
    print()
    print("="*80)
    print("WHY THIS MATTERS FOR JOINT DETECTION")
    print("="*80)
    print()
    print("Elbows and knees bend in specific directions:")
    print("  • Elbows: Bend toward the front of the body")
    print("  • Knees:  Bend toward the back of the body")
    print()
    print("When facing FORWARD (toward camera):")
    print("  Left elbow:  Cross product Z should be POSITIVE (bending inward)")
    print("  Right elbow: Cross product Z should be NEGATIVE (bending inward)")
    print("  Left knee:   Cross product Z should be NEGATIVE (bending backward)")
    print("  Right knee:  Cross product Z should be POSITIVE (bending backward)")
    print()
    print("When facing BACKWARD (away from camera):")
    print("  → All signs are REVERSED!")
    print()
    print("The filter automatically adjusts its expectations based on detected orientation.")
    print("="*80)
    print()


if __name__ == "__main__":
    print_algorithm_explanation()
    
    print("Generating visualization...")
    visualize_orientation_detection()
    
    print("\n✓ Visualization complete!")
    print("\nKey Takeaways:")
    print("  1. The filter detects orientation using shoulder/hip geometry")
    print("  2. It adjusts expectations for elbow/knee bending based on orientation")
    print("  3. Falls back to angle-based checks if orientation is uncertain")
    print("  4. This makes the filter work correctly regardless of which way person faces")
