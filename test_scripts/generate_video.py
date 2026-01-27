#!/usr/bin/env python3
"""
Generate visualization videos from pose estimation JSON files.

Usage:
    python generate_video.py --video path/to/video.mp4 --pose2d path/to/estimation_2d.json --pose3d path/to/estimation_3d.json --output path/to/output_dir
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path to import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.analysis.report_generation.video_generation import generate_visualization_videos


def load_json(filepath: str):
    """Load JSON file and return data"""
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Generate pose visualization videos from JSON data')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--pose2d', required=True, help='Path to 2D pose JSON file')
    parser.add_argument('--pose3d', required=True, help='Path to 3D pose JSON file')
    parser.add_argument('--output', required=True, help='Output directory for generated videos')
    
    args = parser.parse_args()
    
    # Validate input files exist
    video_path = Path(args.video)
    pose2d_path = Path(args.pose2d)
    pose3d_path = Path(args.pose3d)
    output_dir = Path(args.output)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not pose2d_path.exists():
        print(f"Error: 2D pose JSON not found: {pose2d_path}")
        sys.exit(1)
    
    if not pose3d_path.exists():
        print(f"Error: 3D pose JSON not found: {pose3d_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pose data
    print(f"Loading pose data...")
    print(f"  2D poses: {pose2d_path}")
    print(f"  3D poses: {pose3d_path}")
    
    keypoints_2d_list = load_json(str(pose2d_path))
    keypoints_3d_list = load_json(str(pose3d_path))
    
    print(f"Loaded {len(keypoints_2d_list)} frames of 2D data")
    print(f"Loaded {len(keypoints_3d_list)} frames of 3D data")
    
    # Generate visualization videos
    print(f"\nGenerating visualization videos...")
    print(f"  Input video: {video_path}")
    print(f"  Output directory: {output_dir}")
    
    generate_visualization_videos(
        str(video_path),
        keypoints_2d_list,
        keypoints_3d_list,
        output_dir
    )
    
    print(f"\n✓ Done! Videos saved to {output_dir}")
    print(f"  - pose_visualization.avi")
    print(f"  - pose_visualization_3d_sidebyside.avi")


if __name__ == "__main__":
    main()
