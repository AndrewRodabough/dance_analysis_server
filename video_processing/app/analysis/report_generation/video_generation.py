import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
from shared.skeletons.pose_data import VectorizedPoseData


def get_limb_color(joint_name: str):
    """
    Returns color (BGR format for OpenCV) based on limb laterality.
    Blue (255, 0, 0) for left limbs
    Green (0, 255, 0) for right limbs
    Gray (128, 128, 128) for midline/neutral joints
    """
    if "left" in joint_name.lower():
        return (255, 0, 0)  # Blue for left
    elif "right" in joint_name.lower():
        return (0, 255, 0)  # Green for right
    else:
        return (128, 128, 128)  # Gray for neutral/midline


def get_bone_color(skeleton, p1_idx: int, p2_idx: int):
    """
    Determines bone color based on the joint names of bone endpoints.
    Prioritizes left/right classification; uses gray for neutral bones.
    """
    p1_name = skeleton.idx_to_name[p1_idx]
    p2_name = skeleton.idx_to_name[p2_idx]
    
    # Check if either endpoint is left
    if "left" in p1_name.lower() or "left" in p2_name.lower():
        return (255, 0, 0)  # Blue for left
    # Check if either endpoint is right
    elif "right" in p1_name.lower() or "right" in p2_name.lower():
        return (0, 255, 0)  # Green for right
    else:
        return (128, 128, 128)  # Gray for neutral


def get_3d_limits(pose_3d: VectorizedPoseData):
    """
    Computes fixed axis limits across all frames using vectorized operations 
    to prevent the 3D plot from jittering.
    """
    # data shape: (Frames, Joints, 3)
    data = pose_3d.skeleton.data.copy()

    # confidence shape: (Frames, Joints)
    mask = pose_3d.confidence > 0.3
    
    # Extract only confident points across all frames
    valid_kps = data[mask]
    
    if len(valid_kps) == 0:
        return None
        
    # Use median to be robust to outliers
    center = np.median(valid_kps, axis=0)
    
    # Compute range to include 95% of the data
    distances = np.abs(valid_kps - center)
    range_val = np.percentile(distances, 95, axis=0).max()
    
    # Add padding and enforce a minimum range
    range_val = max(range_val * 1.2, 1.0)
    
    print(f"Computed fixed 3D limits: center={center}, range={range_val}")
    return {'center': center, 'range': range_val}


def draw_skeleton_2d(frame, pose_2d: VectorizedPoseData, frame_idx: int):
    """Draws a 2D pose skeleton directly onto the cv2 frame with color-coding for left/right limbs."""
    if frame_idx >= pose_2d.num_frames:
        return frame
        
    keypoints = pose_2d.skeleton.data[frame_idx]  # Shape: (Joints, 2)
    confidences = pose_2d.confidence[frame_idx]   # Shape: (Joints,)
    bones = pose_2d.skeleton.bones_index          # Shape: (2, Num_Bones)
    
    # Validate that data dimensions match expectations
    num_joints = keypoints.shape[0]
    if num_joints != confidences.shape[0]:
        print(f"Warning: Keypoint count mismatch at frame {frame_idx}: "
              f"{num_joints} keypoints vs {confidences.shape[0]} confidences")
        return frame
    
    # 1. Draw Bones with bounds checking and color coding
    for i in range(bones.shape[1]):
        p1_idx, p2_idx = bones[0, i], bones[1, i]
        
        # Validate bone indices are within range
        if p1_idx >= num_joints or p2_idx >= num_joints:
            print(f"Warning: Invalid bone index at frame {frame_idx}: "
                  f"bone[{p1_idx}, {p2_idx}] exceeds {num_joints} joints")
            continue
        
        # Only draw if both endpoints are confident
        if confidences[p1_idx] > 0.3 and confidences[p2_idx] > 0.3:
            pt1 = (int(keypoints[p1_idx, 0]), int(keypoints[p1_idx, 1]))
            pt2 = (int(keypoints[p2_idx, 0]), int(keypoints[p2_idx, 1]))
            
            # Validate coordinates are within frame bounds
            h, w = frame.shape[:2]
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                # Get color based on left/right limb
                color = get_bone_color(pose_2d.skeleton, p1_idx, p2_idx)
                cv2.line(frame, pt1, pt2, color, 2)
            
    # 2. Draw Joints with bounds checking
    for i, pt in enumerate(keypoints):
        if i >= len(confidences):
            break
        if confidences[i] > 0.3:
            x, y = int(pt[0]), int(pt[1])
            h, w = frame.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                # Get color based on joint laterality
                joint_name = pose_2d.skeleton.idx_to_name[i]
                color = get_limb_color(joint_name)
                cv2.circle(frame, (x, y), 3, color, -1)
            
    return frame


def get_3d_bone_color(skeleton, p1_idx: int, p2_idx: int):
    """
    Determines bone color for 3D matplotlib visualization (RGB format).
    Blue for left limbs, Green for right limbs, Gray for neutral.
    """
    p1_name = skeleton.idx_to_name[p1_idx]
    p2_name = skeleton.idx_to_name[p2_idx]
    
    # Check if either endpoint is left
    if "left" in p1_name.lower() or "left" in p2_name.lower():
        return (0, 0, 1)  # Blue for left (RGB)
    # Check if either endpoint is right
    elif "right" in p1_name.lower() or "right" in p2_name.lower():
        return (0, 1, 0)  # Green for right (RGB)
    else:
        return (0.5, 0.5, 0.5)  # Gray for neutral (RGB)


def get_3d_joint_color(skeleton, joint_idx: int):
    """
    Determines joint color for 3D matplotlib visualization (RGB format).
    Blue for left joints, Green for right joints, Red for neutral.
    """
    joint_name = skeleton.idx_to_name[joint_idx]
    if "left" in joint_name.lower():
        return (0, 0, 1)  # Blue for left (RGB)
    elif "right" in joint_name.lower():
        return (0, 1, 0)  # Green for right (RGB)
    else:
        return (1, 0, 0)  # Red for neutral/midline (RGB)


def draw_skeleton_3d(pose_3d: VectorizedPoseData, frame_idx: int, fig_size=(6, 6), fixed_limits=None):
    """Draws the 3D pose skeleton for a single person and returns an image array."""
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('persp', focal_length=0.3)
    
    if frame_idx < pose_3d.num_frames:
        keypoints = pose_3d.skeleton.data[frame_idx].copy()
        confidences = pose_3d.confidence[frame_idx]
        bones = pose_3d.skeleton.bones_index
        
        # Validate data dimensions
        num_joints = keypoints.shape[0]
        if num_joints != confidences.shape[0]:
            print(f"Warning: 3D keypoint count mismatch at frame {frame_idx}: "
                  f"{num_joints} keypoints vs {confidences.shape[0]} confidences")
            plt.close(fig)
            return np.zeros((int(fig_size[0]*100), int(fig_size[1]*100), 3), dtype=np.uint8)

        # 1. Draw Bones with bounds checking and color coding
        for i in range(bones.shape[1]):
            p1_idx, p2_idx = bones[0, i], bones[1, i]
            
            # Validate bone indices
            if p1_idx >= num_joints or p2_idx >= num_joints:
                print(f"Warning: Invalid 3D bone index at frame {frame_idx}: "
                      f"bone[{p1_idx}, {p2_idx}] exceeds {num_joints} joints")
                continue
                
            if confidences[p1_idx] > 0.3 and confidences[p2_idx] > 0.3:
                pt1, pt2 = keypoints[p1_idx], keypoints[p2_idx]
                # Get color based on left/right limb
                color = get_3d_bone_color(pose_3d.skeleton, p1_idx, p2_idx)
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 
                       color=color, linewidth=2)
                
        # 2. Draw Joints with validation and color coding
        for i in range(num_joints):
            if confidences[i] > 0.3:
                pt = keypoints[i]
                color = get_3d_joint_color(pose_3d.skeleton, i)
                ax.scatter([pt[0]], [pt[1]], [pt[2]], c=[color], s=40, alpha=0.8)
            
    # Set labels and limits
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Pose')
    
    if fixed_limits is not None:
        center, range_val = fixed_limits['center'], fixed_limits['range']
        ax.set_xlim([center[0] - range_val, center[0] + range_val])
        ax.set_ylim([center[1] - range_val, center[1] + range_val])
        ax.set_zlim([center[2] - range_val, center[2] + range_val])
        
    ax.view_init(elev=15, azim=90)
    
    # Convert plot to cv2 image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    
    return img


def generate_side_by_side_video(
    filepath: str,
    pose_2d: VectorizedPoseData,
    pose_3d: VectorizedPoseData,
    output_path: str,
):
    """Generates a single video with original+2D on the left and 3D on the right."""
    print("Creating side-by-side 2D/3D visualization...")
    
    # Validate pose data integrity before processing
    print(f"Pose 2D: {pose_2d.num_frames} frames, {pose_2d.skeleton.num_joints} joints")
    print(f"Pose 3D: {pose_3d.num_frames} frames, {pose_3d.skeleton.num_joints} joints")
    
    # Check for data consistency
    if pose_2d.skeleton.num_joints != pose_2d.skeleton.data.shape[1]:
        print(f"ERROR: 2D skeleton config mismatch - "
              f"expected {pose_2d.skeleton.num_joints} joints, "
              f"got {pose_2d.skeleton.data.shape[1]} in data")
        return
    
    if pose_3d.skeleton.num_joints != pose_3d.skeleton.data.shape[1]:
        print(f"ERROR: 3D skeleton config mismatch - "
              f"expected {pose_3d.skeleton.num_joints} joints, "
              f"got {pose_3d.skeleton.data.shape[1]} in data")
        return
    
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {filepath}")
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1. Pre-compute fixed limits for the 3D plot to stop jittering
    fixed_limits = get_3d_limits(pose_3d)
    
    # 2. Generate a single 3D frame to calculate the dynamic width
    sample_3d = draw_skeleton_3d(pose_3d, 0, fixed_limits=fixed_limits)
    aspect_ratio = sample_3d.shape[1] / sample_3d.shape[0]
    new_width = int(height * aspect_ratio)
    
    # 3. Setup Video Writer (Using mp4v for better cross-platform compatibility)
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path / "pose_visualization_combined.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_width = width + new_width
    fourcc_candidates = ['mp4v', 'avc1', 'H264']
    out = None
    for fourcc_tag in fourcc_candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (combined_width, height))
        if writer.isOpened():
            out = writer
            break
        writer.release()

    if out is None:
        print(f"Error: Could not open VideoWriter for {output_path}")
        cap.release()
        return
    
    frame_idx = 0
    frames_written = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    text_origin = (12, 28)
    
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw 2D poses onto the original frame
        if frame_idx < pose_2d.num_frames:
            frame = draw_skeleton_2d(frame, pose_2d, frame_idx)
                
        # Generate 3D visualization and resize it to match the video height
        if frame_idx < pose_3d.num_frames:
            viz_3d = draw_skeleton_3d(pose_3d, frame_idx, fixed_limits=fixed_limits)
            viz_3d = cv2.resize(viz_3d, (new_width, height))
        else:
            viz_3d = np.zeros((height, new_width, 3), dtype=np.uint8)
            
        # Combine side-by-side and write
        combined_frame = np.hstack([frame, viz_3d])
        label = f"Frame {frame_idx}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            font_thickness,
        )
        x, y = text_origin
        cv2.rectangle(
            combined_frame,
            (x - 6, y - text_height - 6),
            (x + text_width + 6, y + baseline + 6),
            bg_color,
            thickness=-1,
        )
        cv2.putText(
            combined_frame,
            label,
            (x, y),
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )
        out.write(combined_frame)
        
        frames_written += 1
        frame_idx += 1
        
        if frames_written % 100 == 0:
            print(f"Processed {frames_written}/{total_frames} frames...")
            
    cap.release()
    out.release()
    if frames_written == 0:
        print(f"Warning: No frames written to {output_path}")
    else:
        print(f"Success! Saved combined visualization to {output_path}")