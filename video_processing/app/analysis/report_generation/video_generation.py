import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D

# H36M 17-keypoint format (used by MotionBERT/human3d)
# Keypoint order:
#  0: Pelvis (Hip center)
#  1: Right Hip
#  2: Right Knee  
#  3: Right Ankle
#  4: Left Hip
#  5: Left Knee
#  6: Left Ankle
#  7: Spine
#  8: Thorax (Chest)
#  9: Neck/Nose
# 10: Head
# 11: Left Shoulder
# 12: Left Elbow
# 13: Left Wrist
# 14: Right Shoulder
# 15: Right Elbow
# 16: Right Wrist
# Extended with feet (17-22):
# 17: Left Big Toe
# 18: Left Small Toe
# 19: Left Heel
# 20: Right Big Toe
# 21: Right Small Toe
# 22: Right Heel

SKELETON_H36M = [
    # Spine chain
    [0, 7],    # pelvis to spine
    [7, 8],    # spine to thorax
    [8, 9],    # thorax to neck
    [9, 10],   # neck to head
    # Right leg
    [0, 1],    # pelvis to right hip
    [1, 2],    # right hip to right knee
    [2, 3],    # right knee to right ankle
    # Left leg
    [0, 4],    # pelvis to left hip
    [4, 5],    # left hip to left knee
    [5, 6],    # left knee to left ankle
    # Left arm
    [8, 11],   # thorax to left shoulder
    [11, 12],  # left shoulder to left elbow
    [12, 13],  # left elbow to left wrist
    # Right arm
    [8, 14],   # thorax to right shoulder
    [14, 15],  # right shoulder to right elbow
    [15, 16],  # right elbow to right wrist
]

# Extended skeleton with feet (23 keypoints: 17 body + 6 feet)
SKELETON_H36M_WITH_FEET = SKELETON_H36M + [
    # Left foot (connected to left ankle, index 6)
    [6, 17],   # left ankle to left big toe
    [6, 18],   # left ankle to left small toe
    [6, 19],   # left ankle to left heel
    [17, 18],  # left big toe to left small toe (foot outline)
    [19, 17],  # left heel to left big toe (foot sole)
    # Right foot (connected to right ankle, index 3)
    [3, 20],   # right ankle to right big toe
    [3, 21],   # right ankle to right small toe
    [3, 22],   # right ankle to right heel
    [20, 21],  # right big toe to right small toe (foot outline)
    [22, 20],  # right heel to right big toe (foot sole)
]

# COCO 17 body keypoint skeleton connections (for comparison/legacy)
SKELETON_COCO17 = [
    # Torso
    [5, 6],    # shoulders
    [5, 11],   # left shoulder to left hip
    [6, 12],   # right shoulder to right hip
    [11, 12],  # hips
    # Left arm
    [5, 7],    # left shoulder to left elbow
    [7, 9],    # left elbow to left wrist
    # Right arm
    [6, 8],    # right shoulder to right elbow
    [8, 10],   # right elbow to right wrist
    # Left leg
    [11, 13],  # left hip to left knee
    [13, 15],  # left knee to left ankle
    # Right leg
    [12, 14],  # right hip to right knee
    [14, 16],  # right knee to right ankle
    # Head
    [0, 1],    # nose to left eye
    [0, 2],    # nose to right eye
    [1, 3],    # left eye to left ear
    [2, 4],    # right eye to right ear
    [3, 5],    # left ear to left shoulder
    [4, 6],    # right ear to right shoulder
]

# Default 17-keypoint skeleton is H36M (used by MotionBERT)
SKELETON_17 = SKELETON_H36M

# COCO wholebody 133 skeleton connections (first 17 are body, same as above)
SKELETON_133 = [
    # Body (same as SKELETON_17)
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
]

# Default to 133 for backward compatibility with 2D visualization
SKELETON = SKELETON_133


def get_skeleton_for_keypoint_count(num_keypoints: int):
    """Return appropriate skeleton connections for keypoint count"""
    if num_keypoints <= 17:
        return SKELETON_17
    elif num_keypoints <= 23:
        # 23 keypoints = H36M body (17) + feet (6)
        return SKELETON_H36M_WITH_FEET
    else:
        return SKELETON_133


def _restructure_keypoints(keypoints_2d_list, keypoints_3d_list):
    """Restructure data: convert from list of [N, 133, 2/3] to list of list of person dicts"""
    estimation_2d = []
    estimation_3d = []
    
    for frame_idx, (kp2d, kp3d) in enumerate(zip(keypoints_2d_list, keypoints_3d_list)):
        # kp2d shape: [N, 133, 2] where N is number of people
        # kp3d shape: [N, 133, 3]
        num_people = kp2d.shape[0] if len(kp2d.shape) > 2 else 1
        
        frame_2d = []
        frame_3d = []
        
        for person_idx in range(num_people):
            if len(kp2d.shape) == 2:
                # Single person, no batch dimension
                person_kp2d = kp2d
                person_kp3d = kp3d
            else:
                person_kp2d = kp2d[person_idx]
                person_kp3d = kp3d[person_idx]
            
            # For 2D, we need keypoints and scores
            # Scores are the confidence values (assume from last channel or generate from data)
            if person_kp2d.shape[-1] == 3:
                # Has confidence as 3rd channel
                scores_2d = person_kp2d[:, 2]
                keypoints_2d = person_kp2d[:, :2]
            else:
                # Generate scores (placeholder - model should provide these)
                scores_2d = np.ones(person_kp2d.shape[0])
                keypoints_2d = person_kp2d
            
            if person_kp3d.shape[-1] == 4:
                # Has confidence as 4th channel
                scores_3d = person_kp3d[:, 3]
                keypoints_3d = person_kp3d[:, :3]
            else:
                scores_3d = np.ones(person_kp3d.shape[0])
                keypoints_3d = person_kp3d
            
            frame_2d.append({
                'keypoints': keypoints_2d,
                'scores': scores_2d
            })
            
            frame_3d.append({
                'keypoints': keypoints_3d,
                'scores': scores_3d
            })
        
        estimation_2d.append(frame_2d)
        estimation_3d.append(frame_3d)
    
    return estimation_2d, estimation_3d


def draw_bbox(frame, bbox):
    """Draw bounding box on frame"""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame


def draw_skeleton_3d(keypoints_3d, scores, fig_size=(6, 6), fixed_limits=None):
    """Draw 3D pose skeleton and return as image array
    
    Args:
        keypoints_3d: 3D keypoint coordinates
        scores: confidence scores for each keypoint
        fig_size: figure size tuple
        fixed_limits: dict with 'center' and 'range' to use fixed axis limits across frames
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Use perspective projection instead of orthographic
    # focal_length: smaller = more perspective distortion, larger = more orthographic
    ax.set_proj_type('persp', focal_length=0.3)
    
    # Remove batch dimension if present
    if len(keypoints_3d.shape) == 3:
        keypoints_3d = keypoints_3d[0]
    if len(scores.shape) == 2:
        scores = scores[0]
    
    # Flip Z-axis to make the pose upright (Z points upward, not Y)
    keypoints_3d = keypoints_3d.copy()
    keypoints_3d[:, 2] = -keypoints_3d[:, 2]
    
    # Select appropriate skeleton for keypoint count
    skeleton = get_skeleton_for_keypoint_count(len(keypoints_3d))
    
    # Draw skeleton connections
    for connection in skeleton:
        pt1_idx, pt2_idx = connection
        if pt1_idx >= len(keypoints_3d) or pt2_idx >= len(keypoints_3d):
            continue
            
        score1 = float(scores[pt1_idx]) if pt1_idx < len(scores) else 0.0
        score2 = float(scores[pt2_idx]) if pt2_idx < len(scores) else 0.0
        
        if score1 > 0.3 and score2 > 0.3:
            pt1 = keypoints_3d[pt1_idx]
            pt2 = keypoints_3d[pt2_idx]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'g-', linewidth=2)
    
    # Draw keypoints
    valid_points = scores > 0.3
    valid_kps = keypoints_3d[valid_points]
    if len(valid_kps) > 0:
        ax.scatter(valid_kps[:, 0], valid_kps[:, 1], valid_kps[:, 2], c='r', s=20)
    
    # Set labels and limits
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Pose')
    
    # Use fixed axis limits to prevent jitter, or calculate from current frame
    if fixed_limits is not None:
        center = fixed_limits['center']
        range_val = fixed_limits['range']
        ax.set_xlim([center[0] - range_val, center[0] + range_val])
        ax.set_ylim([center[1] - range_val, center[1] + range_val])
        ax.set_zlim([center[2] - range_val, center[2] + range_val])
    elif len(valid_kps) > 0:
        center = valid_kps.mean(axis=0)
        range_val = 1.0
        ax.set_xlim([center[0] - range_val, center[0] + range_val])
        ax.set_ylim([center[1] - range_val, center[1] + range_val])
        ax.set_zlim([center[2] - range_val, center[2] + range_val])
    
    # Set viewing angle: elev=elevation (up/down), azim=azimuth (rotation around vertical)
    # elev=15, azim=45 gives a front-facing view with strong depth perspective
    ax.view_init(elev=15, azim=45)
    
    # Convert plot to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    
    return img


def draw_skeleton(frame, person_data, debug=False):
    """Draw 2D pose skeleton on frame"""
    keypoints = person_data['keypoints']  # Shape: (1, num_keypoints, 2 or 3) or (num_keypoints, 2 or 3)
    scores = person_data['scores']  # Shape: (1, num_keypoints) or (num_keypoints,)
    
    # Draw bbox if available
    if 'bbox' in person_data:
        frame = draw_bbox(frame, person_data['bbox'])
    
    # Remove batch dimension if present
    if len(keypoints.shape) == 3:
        keypoints = keypoints[0]  # (1, 133, 2) -> (133, 2)
    if len(scores.shape) == 2:
        scores = scores[0]  # (1, 133) -> (133,)
    
    # Ensure scores is 1D
    if len(scores.shape) > 1:
        scores = scores.flatten()
    
    if debug:
        print(f"After squeezing - keypoints shape: {keypoints.shape}, scores shape: {scores.shape}")
        print(f"First 5 keypoints: {keypoints[:5]}")
        print(f"First 5 scores: {scores[:5]}")
        print(f"Max score: {scores.max()}, Min score: {scores.min()}")
        high_conf = (scores > 0.3).sum()
        print(f"Number of keypoints with score > 0.3: {high_conf}")
    
    # Draw skeleton connections
    drawn_lines = 0
    for connection in SKELETON:
        pt1_idx, pt2_idx = connection
        if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
            continue
            
        pt1 = keypoints[pt1_idx]
        pt2 = keypoints[pt2_idx]
        score1 = float(scores[pt1_idx]) if pt1_idx < len(scores) else 0.0
        score2 = float(scores[pt2_idx]) if pt2_idx < len(scores) else 0.0
        
        # Only draw if both points are confident
        if score1 > 0.3 and score2 > 0.3:
            pt1_coords = (int(pt1[0]), int(pt1[1]))
            pt2_coords = (int(pt2[0]), int(pt2[1]))
            cv2.line(frame, pt1_coords, pt2_coords, (0, 255, 0), 2)
            drawn_lines += 1
    
    # Draw keypoints
    drawn_points = 0
    for i, kp in enumerate(keypoints):
        if i < len(scores):
            score = float(scores[i])
            if score > 0.3:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                drawn_points += 1
    
    if debug:
        print(f"Drew {drawn_lines} lines and {drawn_points} points")
    
    return frame


def generate_visualization_videos(filepath: str, keypoints_2d_list: list, keypoints_3d_list: list, output_dir: str):
    """Generate 2D and 3D visualization videos"""
    # Restructure keypoints data for visualization
    estimation_2d, estimation_3d = _restructure_keypoints(keypoints_2d_list, keypoints_3d_list)
    
    # Create visualization video
    print("Creating visualization video...")
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {filepath}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Estimation data has {len(estimation_2d)} frames")
    
    # Create video writer with XVID codec (more compatible)
    output_path = Path(output_dir) / "pose_visualization.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not create video writer, trying MJPEG codec...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Error: Failed to create video writer")
            cap.release()
            return
    
    frame_idx = 0
    frames_written = 0
    poses_drawn = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw poses for this frame
        if frame_idx < len(estimation_2d):
            frame_poses = estimation_2d[frame_idx]
            if frame_idx == 0:  # Debug first frame
                print(f"Frame 0 has {len(frame_poses)} people")
                if len(frame_poses) > 0:
                    print(f"First person keypoints shape: {frame_poses[0]['keypoints'].shape}")
                    print(f"First person scores shape: {frame_poses[0]['scores'].shape}")
            
            for person in frame_poses:
                frame = draw_skeleton(frame, person, debug=(frame_idx == 0))
                poses_drawn += 1
        
        out.write(frame)
        frames_written += 1
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Saved visualization video to {output_path}")
    print(f"Wrote {frames_written} frames")
    print(f"Drew {poses_drawn} total poses across all frames")
    
    # Create side-by-side visualization with 3D
    print("Creating side-by-side 2D/3D visualization...")
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {filepath} for side-by-side")
        return
    
    # Create side-by-side video
    output_path_3d = Path(output_dir) / "pose_visualization_3d_sidebyside.avi"
    
    # Read first frame to determine 3D visualization size
    ret, first_frame = cap.read()
    if not ret:
        print("Could not read first frame")
        cap.release()
        return
    
    # Calculate fixed axis limits across all frames to prevent jitter
    fixed_limits = None
    if len(estimation_3d) > 0:
        print("Computing fixed axis limits across all frames to prevent jitter...")
        all_valid_keypoints = []
        for frame_3d in estimation_3d:
            for person in frame_3d:
                keypoints = person['keypoints']
                scores = person['scores']
                
                # Remove batch dimension if present
                if len(keypoints.shape) == 3:
                    keypoints = keypoints[0]
                if len(scores.shape) == 2:
                    scores = scores[0]
                
                # Flip Z-axis (same as in draw_skeleton_3d)
                keypoints = keypoints.copy()
                keypoints[:, 2] = -keypoints[:, 2]
                
                # Collect valid keypoints
                valid_mask = scores > 0.3
                valid_kps = keypoints[valid_mask]
                if len(valid_kps) > 0:
                    all_valid_keypoints.append(valid_kps)
        
        if len(all_valid_keypoints) > 0:
            all_keypoints_array = np.vstack(all_valid_keypoints)
            # Use median instead of mean to be more robust to outliers
            center = np.median(all_keypoints_array, axis=0)
            # Compute range to include 95% of the data
            distances = np.abs(all_keypoints_array - center)
            range_val = np.percentile(distances, 95, axis=0).max()
            # Add some padding
            range_val = max(range_val * 1.2, 1.0)
            
            fixed_limits = {
                'center': center,
                'range': range_val
            }
            print(f"Fixed axis limits: center={center}, range={range_val}")
    
    # Generate a sample 3D plot to get dimensions
    if len(estimation_3d) > 0 and len(estimation_3d[0]) > 0:
        sample_3d = draw_skeleton_3d(estimation_3d[0][0]['keypoints'], 
                                     estimation_3d[0][0]['scores'],
                                     fixed_limits=fixed_limits)
        # Resize 3D visualization to match video height
        aspect_ratio = sample_3d.shape[1] / sample_3d.shape[0]
        new_width = int(height * aspect_ratio)
        sample_3d_resized = cv2.resize(sample_3d, (new_width, height))
        
        combined_width = width + new_width
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_3d = cv2.VideoWriter(str(output_path_3d), fourcc, fps, (combined_width, height))
        
        if not out_3d.isOpened():
            print("Error: Could not create 3D video writer")
            cap.release()
            return
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        frames_written_3d = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw 2D poses on left side
            if frame_idx < len(estimation_2d):
                for person in estimation_2d[frame_idx]:
                    frame = draw_skeleton(frame, person)
            
            # Create 3D visualization for right side
            if frame_idx < len(estimation_3d) and len(estimation_3d[frame_idx]) > 0:
                # Use first person for 3D visualization
                person_3d = estimation_3d[frame_idx][0]
                viz_3d = draw_skeleton_3d(person_3d['keypoints'], 
                                         person_3d['scores'],
                                         fixed_limits=fixed_limits)
                viz_3d = cv2.resize(viz_3d, (new_width, height))
            else:
                # Empty 3D visualization
                viz_3d = np.zeros((height, new_width, 3), dtype=np.uint8)
            
            # Combine side by side
            combined = np.hstack([frame, viz_3d])
            out_3d.write(combined)
            frames_written_3d += 1
            frame_idx += 1
        
        cap.release()
        out_3d.release()
        print(f"Saved side-by-side visualization to {output_path_3d}")
        print(f"Wrote {frames_written_3d} frames with 3D visualization")
    else:
        print("No 3D data available for visualization")
        cap.release()
