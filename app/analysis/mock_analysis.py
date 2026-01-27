"""
Mock analysis module for local development without GPU/Docker dependencies.
Simulates the behavior of the real analysis pipeline.
"""
import json
import time
from pathlib import Path


def analyze_video_mock(filepath: str, save_keypoints_path: str):
    """
    Mock video analysis that returns fake pose data without running actual models.
    Useful for API development without GPU/Docker dependencies.
    
    Args:
        filepath: Path to the video file (not actually processed in mock mode)
    """
    print(f"[MOCK] Analyzing video: {filepath}")
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Create output directory
    output_dir = Path(save_keypoints_path)
    output_dir.mkdir(exist_ok=True)
    
    # Generate mock 2D pose data
    # MediaPipe outputs 33 keypoints per person per frame
    # Format: list of frames, each containing keypoints with x, y, visibility
    num_frames = 30  # Mock 30 frames (~1 second at 30fps)
    
    mock_2d_data = []
    for frame_idx in range(num_frames):
        frame_keypoints = []
        for kp_idx in range(33):  # 33 MediaPipe keypoints
            # Create realistic-looking motion by varying positions slightly
            frame_keypoints.append({
                "x": 320 + (kp_idx * 20) + (frame_idx * 2),
                "y": 240 + (kp_idx * 15) + (frame_idx * 1.5),
                "visibility": 0.85 + (0.15 * (kp_idx % 3) / 3)  # Vary visibility
            })
        mock_2d_data.append(frame_keypoints)
    
    # Generate mock 3D pose data
    # Format: list of frames, each containing keypoints with x, y, z, visibility
    mock_3d_data = []
    for frame_idx in range(num_frames):
        frame_keypoints = []
        for kp_idx in range(33):
            # 3D coordinates in meters (world coordinates)
            frame_keypoints.append({
                "x": -0.5 + (kp_idx * 0.05) + (frame_idx * 0.01),
                "y": 0.8 + (kp_idx * 0.04) + (frame_idx * 0.008),
                "z": -1.0 + (kp_idx * 0.03),
                "visibility": 0.85 + (0.15 * (kp_idx % 3) / 3)
            })
        mock_3d_data.append(frame_keypoints)
    
    # Save 2D estimation
    estimation_2d_path = output_dir / "estimation_2d.json"
    with open(estimation_2d_path, "w") as f:
        json.dump(mock_2d_data, f, indent=2)
    print(f"[MOCK] Saved 2D estimation to {estimation_2d_path}")
    
    # Save 3D estimation
    estimation_3d_path = output_dir / "estimation_3d.json"
    with open(estimation_3d_path, "w") as f:
        json.dump(mock_3d_data, f, indent=2)
    print(f"[MOCK] Saved 3D estimation to {estimation_3d_path}")
    
    # Create dummy video files (empty files to simulate output)
    video_2d_path = output_dir / "pose_visualization.avi"
    video_3d_path = output_dir / "pose_visualization_3d_sidebyside.avi"
    
    video_2d_path.touch()
    video_3d_path.touch()
    
    print(f"[MOCK] Created mock visualization videos")
    print(f"[MOCK] Analysis complete")
    
    return {
        "status": "success",
        "mode": "mock",
        "frames_processed": num_frames,
        "output_files": {
            "2d_data": str(estimation_2d_path),
            "3d_data": str(estimation_3d_path),
            "2d_video": str(video_2d_path),
            "3d_video": str(video_3d_path)
        }
    }
