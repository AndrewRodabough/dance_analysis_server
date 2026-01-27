import json
import os
import numpy as np
from pathlib import Path
from app.analysis.pose_estimation.pose_estimation import pose_estimation
from app.analysis.mock_analysis import analyze_video_mock

# Toggle between real and mock analysis via environment variable
USE_MOCK = os.getenv("USE_MOCK_ANALYSIS", "false").lower() == "true"

def analyze_video(filepath: str, save_keypoints_path: str = 'test_outputs'):
    # Use mock analysis if enabled (for local development without GPU)
    if USE_MOCK:
        return analyze_video_mock(filepath)
    
    print(f"Analyzing video: {filepath}")

    # pose processing (with smoothing applied)
    keypoints_2d_list, keypoints_3d_list = pose_estimation(filepath, apply_smoothing=True)

    if save_keypoints_path != '':
        save_keypoints_to_json(save_keypoints_path, keypoints_2d_list, keypoints_3d_list)

    # data / features Extraction


    # anaysis


    # report generation


def save_keypoints_to_json(folder_path: str, keypoints_2d, keypoints_3d):
    """Save keypoints data to JSON files"""
    os.makedirs(folder_path, exist_ok=True)
    keypoints_2d_path = Path(folder_path) / "keypoints_2d.json"
    keypoints_3d_path = Path(folder_path) / "keypoints_3d.json"

    with open(keypoints_2d_path, 'w') as f2d:
        json.dump([kp.tolist() for kp in keypoints_2d], f2d)

    with open(keypoints_3d_path, 'w') as f3d:
        json.dump([kp.tolist() for kp in keypoints_3d], f3d)

    print(f"Saved 2D keypoints to {keypoints_2d_path}")
    print(f"Saved 3D keypoints to {keypoints_3d_path}")