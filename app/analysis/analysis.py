import json
import numpy as np
import cv2
from pathlib import Path
from app.analysis.pose_estimation.pose_estimation import pose_estimation 

# COCO wholebody skeleton connections
SKELETON = [
    # Body
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
]

def draw_bbox(frame, bbox):
    """Draw bounding box on frame"""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame

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

def analyze_video(filepath: str):
    print(f"Analyzing video: {filepath}")

    # pose processing
    estimation_2d, estimation_3d = pose_estimation(filepath)

    # Save results to test_scripts folder
    test_scripts_dir = Path("test_scripts")
    test_scripts_dir.mkdir(exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    # Save 2D estimation
    with open(test_scripts_dir / "estimation_2d.json", "w") as f:
        json.dump(convert_to_serializable(estimation_2d), f, indent=2)
    print(f"Saved 2D estimation to {test_scripts_dir / 'estimation_2d.json'}")
    
    # Save 3D estimation
    with open(test_scripts_dir / "estimation_3d.json", "w") as f:
        json.dump(convert_to_serializable(estimation_3d), f, indent=2)
    print(f"Saved 3D estimation to {test_scripts_dir / 'estimation_3d.json'}")
    
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
    output_path = test_scripts_dir / "pose_visualization.avi"
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

    # other data / features Extraction

    # anaysis

    # report generation