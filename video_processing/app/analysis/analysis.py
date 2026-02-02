import json
import os
import numpy as np
from pathlib import Path
from app.analysis.pose_estimation.pose_estimation import pose_estimation, ENABLE_ANATOMICAL_CONSTRAINTS
from app.analysis.mock_analysis import analyze_video_mock
from app.analysis.report_generation.video_generation import generate_visualization_videos

# Toggle between real and mock analysis via environment variable
USE_MOCK = os.getenv("USE_MOCK_ANALYSIS", "false").lower() == "true"

def analyze_video(filepath: str, save_keypoints_path: str = 'test_outputs', generate_video: bool = True):
    # Use mock analysis if enabled (for local development without GPU)
    if USE_MOCK:
        return analyze_video_mock(filepath, save_keypoints_path)
    
    print(f"Analyzing video: {filepath}")

    # pose processing (with smoothing applied)
    # Returns: 2d keypoints, 3d keypoints (after IK), 3d keypoints (before IK), scores
    keypoints_2d_list, keypoints_3d_list, keypoints_3d_before, scores_list = pose_estimation(filepath, apply_smoothing=True)

    if save_keypoints_path != '':
        save_keypoints_to_json(save_keypoints_path, keypoints_2d_list, keypoints_3d_list, 
                               keypoints_3d_before, scores_list)
        
        if generate_video:
            generate_visualization_videos(filepath, keypoints_2d_list, keypoints_3d_list, save_keypoints_path)
            
            # IK comparison video generation disabled for now - needs debugging
            # if ENABLE_ANATOMICAL_CONSTRAINTS:
            #     from test_scripts.ik_comparison_video import generate_comparison_video
            #     comparison_output = str(Path(save_keypoints_path) / "ik_comparison.avi")
            #     generate_comparison_video(
            #         filepath, keypoints_3d_before, keypoints_3d_list, 
            #         scores_list, comparison_output
            #     )

    # data / features Extraction


    # anaysis


    # report generation




def save_keypoints_to_json(folder_path: str, keypoints_2d, keypoints_3d, 
                           keypoints_3d_before=None, scores=None):
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
    
    # Save before-IK keypoints for comparison visualization
    if keypoints_3d_before is not None:
        keypoints_3d_before_path = Path(folder_path) / "keypoints_3d_before.json"
        with open(keypoints_3d_before_path, 'w') as f:
            json.dump([kp.tolist() for kp in keypoints_3d_before], f)
        print(f"Saved 3D keypoints (before IK) to {keypoints_3d_before_path}")
    
    # Save scores for confidence visualization
    if scores is not None:
        scores_path = Path(folder_path) / "scores.json"
        with open(scores_path, 'w') as f:
            json.dump([s.tolist() for s in scores], f)
        print(f"Saved scores to {scores_path}")