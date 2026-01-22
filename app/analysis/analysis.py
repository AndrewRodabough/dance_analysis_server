import json
import numpy as np
from pathlib import Path
from app.analysis.pose_estimation.pose_estimation import pose_estimation
from app.analysis.video_generation import generate_visualization_videos

def analyze_video(filepath: str):
    print(f"Analyzing video: {filepath}")

    # pose processing (with smoothing applied)
    keypoints_2d_list, keypoints_3d_list = pose_estimation(filepath, apply_smoothing=True)

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
        json.dump(convert_to_serializable(keypoints_2d_list), f, indent=2)
    print(f"Saved 2D estimation to {test_scripts_dir / 'estimation_2d.json'}")
    
    # Save 3D estimation
    with open(test_scripts_dir / "estimation_3d.json", "w") as f:
        json.dump(convert_to_serializable(keypoints_3d_list), f, indent=2)
    print(f"Saved 3D estimation to {test_scripts_dir / 'estimation_3d.json'}")
    
    # Generate visualization videos
    generate_visualization_videos(filepath, keypoints_2d_list, keypoints_3d_list, test_scripts_dir)


    # data / features Extraction


    # anaysis


    # report generation