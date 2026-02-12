import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
from shared.skeletons.pose_data import VectorizedPoseData

def get_3d_limits(pose_3d: VectorizedPoseData):
    """
    Computes fixed axis limits across all frames using vectorized operations 
    to prevent the 3D plot from jittering.
    """
    # data shape: (Frames, Joints, 3)
    data = pose_3d.skeleton.data.copy()
    # Flip axes so positive Y is up in the plot (image coords are Y-down).
    data[..., 1] = -data[..., 1]
    data[..., 2] = -data[..., 2]  # Flip Z-axis to make pose upright
    
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
    """Draws a 2D pose skeleton directly onto the cv2 frame."""
    if frame_idx >= pose_2d.num_frames:
        return frame
        
    keypoints = pose_2d.skeleton.data[frame_idx]  # Shape: (Joints, 2)
    confidences = pose_2d.confidence[frame_idx]   # Shape: (Joints,)
    bones = pose_2d.skeleton.bones_index          # Shape: (2, Num_Bones)
    
    # 1. Draw Bones
    for i in range(bones.shape[1]):
        p1_idx, p2_idx = bones[0, i], bones[1, i]
        
        # Only draw if both endpoints are confident
        if confidences[p1_idx] > 0.3 and confidences[p2_idx] > 0.3:
            pt1 = (int(keypoints[p1_idx, 0]), int(keypoints[p1_idx, 1]))
            pt2 = (int(keypoints[p2_idx, 0]), int(keypoints[p2_idx, 1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
    # 2. Draw Joints
    for i, pt in enumerate(keypoints):
        if confidences[i] > 0.3:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
            
    return frame


def draw_skeleton_3d(pose_3d: VectorizedPoseData, frame_idx: int, fig_size=(6, 6), fixed_limits=None):
    """Draws the 3D pose skeleton for a single person and returns an image array."""
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('persp', focal_length=0.3)
    
    if frame_idx < pose_3d.num_frames:
        keypoints = pose_3d.skeleton.data[frame_idx].copy()
        confidences = pose_3d.confidence[frame_idx]
        bones = pose_3d.skeleton.bones_index
        
        # Flip axes so positive Y is up in the plot (image coords are Y-down).
        keypoints[:, 1] = -keypoints[:, 1]
        keypoints[:, 2] = -keypoints[:, 2]
        
        # 1. Draw Bones
        for i in range(bones.shape[1]):
            p1_idx, p2_idx = bones[0, i], bones[1, i]
            if confidences[p1_idx] > 0.3 and confidences[p2_idx] > 0.3:
                pt1, pt2 = keypoints[p1_idx], keypoints[p2_idx]
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'g-', linewidth=2)
                
        # 2. Draw Joints
        valid_mask = confidences > 0.3
        valid_kps = keypoints[valid_mask]
        if len(valid_kps) > 0:
            ax.scatter(valid_kps[:, 0], valid_kps[:, 1], valid_kps[:, 2], c='r', s=20)
            
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
        
    ax.view_init(elev=15, azim=45)
    
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