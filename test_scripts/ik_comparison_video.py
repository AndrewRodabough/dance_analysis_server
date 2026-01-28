"""
Side-by-side 3D visualization comparing poses before and after IK correction.
Generates a video with two 3D skeletons: original (left) and corrected (right).
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json


# COCO wholebody skeleton connections (body only)
SKELETON = [
    # Body
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
]

# Joints that might be corrected by IK
CORRECTABLE_JOINTS = {
    7: 'L_ELBOW',
    8: 'R_ELBOW', 
    13: 'L_KNEE',
    14: 'R_KNEE'
}


def draw_skeleton_3d_comparison(keypoints_before, keypoints_after, scores, 
                                 fixed_limits=None, frame_idx=0):
    """
    Draw side-by-side 3D skeletons: before and after IK correction.
    
    Args:
        keypoints_before: Original 3D keypoints (133, 3) or (N, 133, 3)
        keypoints_after: Corrected 3D keypoints
        scores: Confidence scores
        fixed_limits: Dict with 'center' and 'range' for consistent axis limits
        frame_idx: Frame number for title
    
    Returns:
        BGR image of the comparison plot
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Convert to numpy arrays and remove batch dimension if present
    keypoints_before = np.asarray(keypoints_before)
    keypoints_after = np.asarray(keypoints_after)
    scores = np.asarray(scores)
    
    if len(keypoints_before.shape) == 3:
        keypoints_before = keypoints_before[0]
    if len(keypoints_after.shape) == 3:
        keypoints_after = keypoints_after[0]
    if len(scores.shape) == 2:
        scores = scores[0]
    
    # Find which joints were corrected
    corrected_joints = []
    for joint_idx in CORRECTABLE_JOINTS:
        if joint_idx < len(keypoints_before):
            diff = np.linalg.norm(keypoints_after[joint_idx] - keypoints_before[joint_idx])
            if diff > 0.01:  # Significant change
                corrected_joints.append(joint_idx)
    
    # Calculate axis limits if not provided
    if fixed_limits is None:
        all_points = np.vstack([keypoints_before[:17], keypoints_after[:17]])
        valid_mask = scores[:17] > 0.3
        if valid_mask.sum() > 0:
            center = keypoints_before[:17][valid_mask].mean(axis=0)
        else:
            center = keypoints_before[:17].mean(axis=0)
        range_val = 1.0
    else:
        center = fixed_limits['center']
        range_val = fixed_limits['range']
    
    # Left plot: Before correction
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('BEFORE IK Correction', fontsize=12, fontweight='bold', color='red')
    _draw_skeleton_on_axis(ax1, keypoints_before, scores, corrected_joints, 
                           highlight_color='red', is_before=True)
    _set_axis_limits(ax1, center, range_val)
    ax1.view_init(elev=15, azim=45)
    
    # Right plot: After correction
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('AFTER IK Correction', fontsize=12, fontweight='bold', color='green')
    _draw_skeleton_on_axis(ax2, keypoints_after, scores, corrected_joints,
                           highlight_color='green', is_before=False)
    _set_axis_limits(ax2, center, range_val)
    ax2.view_init(elev=15, azim=45)
    
    # Add info about corrections
    if corrected_joints:
        joint_names = [CORRECTABLE_JOINTS.get(j, str(j)) for j in corrected_joints]
        fig.suptitle(f'Frame {frame_idx} - Corrected: {", ".join(joint_names)}', 
                    fontsize=11, y=0.98)
    else:
        fig.suptitle(f'Frame {frame_idx} - No corrections needed', fontsize=11, y=0.98)
    
    plt.tight_layout()
    
    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    
    return img


def _draw_skeleton_on_axis(ax, keypoints, scores, corrected_joints, 
                           highlight_color='red', is_before=True):
    """Draw skeleton on a matplotlib 3D axis"""
    
    # Draw skeleton connections
    for connection in SKELETON:
        pt1_idx, pt2_idx = connection
        if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
            continue
        
        score1 = float(scores[pt1_idx]) if pt1_idx < len(scores) else 0.0
        score2 = float(scores[pt2_idx]) if pt2_idx < len(scores) else 0.0
        
        if score1 > 0.3 and score2 > 0.3:
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            
            # Highlight limbs connected to corrected joints
            is_corrected_limb = pt1_idx in corrected_joints or pt2_idx in corrected_joints
            
            if is_corrected_limb:
                color = highlight_color
                linewidth = 3
                alpha = 1.0
            else:
                color = 'blue'
                linewidth = 2
                alpha = 0.7
            
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 
                   color=color, linewidth=linewidth, alpha=alpha)
    
    # Draw keypoints
    for i in range(min(17, len(keypoints))):
        if i < len(scores) and scores[i] > 0.3:
            kp = keypoints[i]
            
            if i in corrected_joints:
                # Highlight corrected joint
                ax.scatter([kp[0]], [kp[1]], [kp[2]], 
                          c=highlight_color, s=100, marker='o', 
                          edgecolors='black', linewidth=2)
            else:
                ax.scatter([kp[0]], [kp[1]], [kp[2]], c='blue', s=30, alpha=0.7)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')


def _set_axis_limits(ax, center, range_val):
    """Set consistent axis limits"""
    ax.set_xlim([center[0] - range_val, center[0] + range_val])
    ax.set_ylim([center[1] - range_val, center[1] + range_val])
    ax.set_zlim([center[2] - range_val, center[2] + range_val])


def generate_comparison_video(video_path: str, 
                              keypoints_before_list: list,
                              keypoints_after_list: list,
                              scores_list: list,
                              output_path: str):
    """
    Generate a video showing before/after IK correction side by side.
    
    Args:
        video_path: Path to original video (for FPS)
        keypoints_before_list: List of original 3D keypoints per frame
        keypoints_after_list: List of corrected 3D keypoints per frame
        scores_list: List of confidence scores per frame
        output_path: Where to save the comparison video
    """
    print(f"Generating IK comparison video...")
    
    # Get video FPS
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    cap.release()
    
    # Helper to extract keypoints for first person
    def extract_person_keypoints(kps, scrs):
        """Extract keypoints and scores for first person, handling various shapes."""
        kps = np.asarray(kps)
        scrs = np.asarray(scrs)
        
        # Handle batch dimension [N, 133, 3] -> [133, 3]
        if len(kps.shape) == 3:
            kps = kps[0]
        if len(scrs.shape) == 2:
            scrs = scrs[0]
        
        # Ensure we have at least 17 keypoints
        if len(kps) < 17:
            return None, None
        
        return kps, scrs
    
    # Calculate fixed axis limits across all frames for consistency
    print("Computing fixed axis limits...")
    all_keypoints = []
    for kps_before, kps_after, scores in zip(keypoints_before_list, keypoints_after_list, scores_list):
        kps_before, scores_b = extract_person_keypoints(kps_before, scores)
        kps_after, _ = extract_person_keypoints(kps_after, scores)
        
        if kps_before is None or kps_after is None or scores_b is None:
            continue
        
        valid_mask = scores_b[:17] > 0.3
        if valid_mask.sum() > 0:
            all_keypoints.append(kps_before[:17][valid_mask])
            all_keypoints.append(kps_after[:17][valid_mask])
    
    if all_keypoints:
        all_kps = np.vstack(all_keypoints)
        center = np.median(all_kps, axis=0)
        distances = np.abs(all_kps - center)
        range_val = max(np.percentile(distances, 95, axis=0).max() * 1.2, 1.0)
        fixed_limits = {'center': center, 'range': range_val}
    else:
        fixed_limits = None
    
    # Generate first frame to get dimensions
    first_img = draw_skeleton_3d_comparison(
        keypoints_before_list[0], keypoints_after_list[0], 
        scores_list[0], fixed_limits, frame_idx=0
    )
    height, width = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not create video writer")
        return
    
    # Count corrections
    total_corrections = 0
    frames_with_corrections = 0
    
    # Generate all frames
    num_frames = len(keypoints_before_list)
    for frame_idx in range(num_frames):
        if frame_idx % 50 == 0:
            print(f"  Processing frame {frame_idx}/{num_frames}")
        
        kps_before = np.asarray(keypoints_before_list[frame_idx])
        kps_after = np.asarray(keypoints_after_list[frame_idx])
        scores = np.asarray(scores_list[frame_idx])
        
        # Check for corrections - extract first person
        if len(kps_before.shape) == 3:
            kps_b = kps_before[0]
            kps_a = kps_after[0]
        else:
            kps_b = kps_before
            kps_a = kps_after
        
        frame_corrections = 0
        for joint_idx in CORRECTABLE_JOINTS:
            if joint_idx < len(kps_b):
                diff = np.linalg.norm(kps_a[joint_idx] - kps_b[joint_idx])
                if diff > 0.01:
                    frame_corrections += 1
        
        if frame_corrections > 0:
            frames_with_corrections += 1
            total_corrections += frame_corrections
        
        img = draw_skeleton_3d_comparison(
            kps_before, kps_after, scores, fixed_limits, frame_idx
        )
        out.write(img)
    
    out.release()
    
    print(f"Saved comparison video to: {output_path}")
    print(f"  Total frames: {num_frames}")
    print(f"  Frames with corrections: {frames_with_corrections} ({100*frames_with_corrections/num_frames:.1f}%)")
    print(f"  Total joint corrections: {total_corrections}")


def compare_from_json_files(before_path: str, after_path: str, scores_path: str,
                            video_path: str, output_path: str):
    """
    Generate comparison video from saved JSON keypoint files.
    
    Args:
        before_path: Path to keypoints_3d_before.json
        after_path: Path to keypoints_3d_after.json (or keypoints_3d.json)
        scores_path: Path to scores.json (optional, will use dummy scores if not found)
        video_path: Path to original video
        output_path: Where to save comparison video
    """
    print(f"Loading keypoints...")
    
    with open(before_path, 'r') as f:
        keypoints_before = [np.array(frame) for frame in json.load(f)]
    
    with open(after_path, 'r') as f:
        keypoints_after = [np.array(frame) for frame in json.load(f)]
    
    # Try to load scores, or create dummy scores
    try:
        with open(scores_path, 'r') as f:
            scores = [np.array(frame) for frame in json.load(f)]
    except FileNotFoundError:
        print(f"  Scores file not found, using dummy scores")
        scores = [np.ones(kps.shape[:-1]) for kps in keypoints_before]
    
    print(f"  Loaded {len(keypoints_before)} frames")
    
    generate_comparison_video(video_path, keypoints_before, keypoints_after, 
                              scores, output_path)


if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("IK CORRECTION COMPARISON VIDEO GENERATOR")
    print("="*70)
    print()
    print("Usage:")
    print("  python ik_comparison_video.py                    # Auto-detect files")
    print("  python ik_comparison_video.py <video_path>       # Specify video")
    print("  python ik_comparison_video.py --generate-sample  # Create sample data")
    print()
    
    # Check for sample generation mode
    if len(sys.argv) > 1 and sys.argv[1] == '--generate-sample':
        print("Generating sample comparison data...")
        
        # Create sample before/after data
        import os
        os.makedirs("test_outputs", exist_ok=True)
        
        # Generate 60 frames of sample data
        num_frames = 60
        sample_before = []
        sample_after = []
        sample_scores = []
        
        for i in range(num_frames):
            # Create a standing pose with slight movement
            t = i / 30.0  # Time in seconds
            
            # 17 body keypoints
            keypoints = np.zeros((17, 3))
            
            # Head
            keypoints[0] = [0, 0.8, 0]  # Nose
            keypoints[1] = [-0.05, 0.85, 0]  # L eye
            keypoints[2] = [0.05, 0.85, 0]   # R eye
            keypoints[3] = [-0.1, 0.8, 0]    # L ear
            keypoints[4] = [0.1, 0.8, 0]     # R ear
            
            # Shoulders
            keypoints[5] = [-0.2, 0.5, 0]    # L shoulder
            keypoints[6] = [0.2, 0.5, 0]     # R shoulder
            
            # Arms with wrong elbow depth (simulate the problem)
            # Left arm
            keypoints[7] = [-0.3, 0.3, 0.15 + 0.05*np.sin(t*2)]  # L elbow (wrong - in front)
            keypoints[9] = [-0.35, 0.1, 0]   # L wrist
            
            # Right arm
            keypoints[8] = [0.3, 0.3, -0.15 + 0.05*np.sin(t*2)]  # R elbow (wrong - behind)
            keypoints[10] = [0.35, 0.1, 0]   # R wrist
            
            # Hips
            keypoints[11] = [-0.15, 0, 0]    # L hip
            keypoints[12] = [0.15, 0, 0]     # R hip
            
            # Legs with wrong knee depth
            keypoints[13] = [-0.15, -0.4, 0.1 + 0.03*np.sin(t*3)]   # L knee (wrong)
            keypoints[15] = [-0.15, -0.8, 0]  # L ankle
            
            keypoints[14] = [0.15, -0.4, -0.1 + 0.03*np.sin(t*3)]   # R knee (wrong)
            keypoints[16] = [0.15, -0.8, 0]   # R ankle
            
            sample_before.append(keypoints.copy())
            
            # Create "corrected" version by reflecting joints
            from app.analysis.pose_estimation.anatomical_constraints import reflect_joint_across_limb_axis
            
            corrected = keypoints.copy()
            # Correct left elbow
            corrected[7] = reflect_joint_across_limb_axis(keypoints[5], keypoints[7], keypoints[9])
            # Correct right elbow  
            corrected[8] = reflect_joint_across_limb_axis(keypoints[6], keypoints[8], keypoints[10])
            # Correct left knee
            corrected[13] = reflect_joint_across_limb_axis(keypoints[11], keypoints[13], keypoints[15])
            # Correct right knee
            corrected[14] = reflect_joint_across_limb_axis(keypoints[12], keypoints[14], keypoints[16])
            
            sample_after.append(corrected)
            sample_scores.append(np.ones(17))
        
        # Save sample data
        with open("test_outputs/keypoints_3d_before.json", 'w') as f:
            json.dump([kp.tolist() for kp in sample_before], f)
        with open("test_outputs/keypoints_3d.json", 'w') as f:
            json.dump([kp.tolist() for kp in sample_after], f)
        with open("test_outputs/scores.json", 'w') as f:
            json.dump([s.tolist() for s in sample_scores], f)
        
        print("Sample data saved to test_outputs/")
        print()
        print("Now generating comparison video from sample...")
        
        generate_comparison_video(
            "",  # No video needed for sample
            sample_before, sample_after, sample_scores,
            "test_outputs/ik_comparison_sample.avi"
        )
        
        print()
        print("Done! Open test_outputs/ik_comparison_sample.avi to see the comparison.")
        sys.exit(0)
    
    # Check for required files
    before_path = Path("test_outputs/keypoints_3d_before.json")
    after_path = Path("test_outputs/keypoints_3d.json")
    
    if not before_path.exists():
        print("Before-IK keypoints not found.")
        print()
        print("Options:")
        print("  1. Enable ENABLE_ANATOMICAL_CONSTRAINTS=True in pose_estimation.py")
        print("     then run analysis - it will save both before/after automatically")
        print()
        print("  2. Generate sample data to test the visualization:")
        print("     python ik_comparison_video.py --generate-sample")
        sys.exit(0)
    
    if not after_path.exists():
        print(f"Error: {after_path} not found")
        sys.exit(1)
    
    # Find the original video
    video_path = None
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        uploads = Path("uploads")
        videos = list(uploads.glob("*.mp4")) + list(uploads.glob("*.avi"))
        if videos:
            video_path = str(videos[0])
            print(f"Using video: {video_path}")
    
    if not video_path:
        print("No video found. Using default FPS of 30.")
        video_path = ""
    
    output_path = "test_outputs/ik_comparison.avi"
    
    compare_from_json_files(
        str(before_path),
        str(after_path),
        "test_outputs/scores.json",
        video_path,
        output_path
    )
