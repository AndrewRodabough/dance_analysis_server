"""
Example script demonstrating how to load and work with pose estimation data.
"""
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.mappers import load_pose_data


def main():
    """Load and explore pose estimation results."""
    
    # Load pose data from MediaPipe JSON files
    print("Loading pose data from JSON files...")
    pose_sequence = load_pose_data(
        json_2d_path="test_scripts/estimation_2d.json",
        json_3d_path="test_scripts/estimation_3d.json",
        model_type="mediapipe",
        fps=30.0,  # Example FPS
        video_width=1920,
        video_height=1080
    )
    
    print(f"\n=== Pose Sequence Summary ===")
    print(f"Model: {pose_sequence.model_name}")
    print(f"Number of frames: {pose_sequence.num_frames}")
    print(f"Duration: {pose_sequence.duration:.2f} seconds" if pose_sequence.duration else "Duration: Unknown")
    print(f"FPS: {pose_sequence.fps}")
    print(f"Resolution: {pose_sequence.video_width}x{pose_sequence.video_height}")
    
    # Examine first frame
    first_frame = pose_sequence.get_frame(0)
    print(f"\n=== First Frame ===")
    print(f"Frame number: {first_frame.frame_number}")
    print(f"Timestamp: {first_frame.timestamp:.3f}s" if first_frame.timestamp else "Timestamp: Unknown")
    print(f"Number of people detected: {first_frame.num_people}")
    
    # Examine first person in first frame
    if first_frame.num_people > 0:
        first_person = first_frame.get_person(0)
        print(f"\n=== First Person ===")
        print(f"Person ID: {first_person.person_id}")
        print(f"Number of keypoints: {first_person.num_keypoints}")
        
        # Show first few keypoints
        print(f"\n=== Sample Keypoints (first 5) ===")
        for i in range(min(5, first_person.num_keypoints)):
            kp_2d = first_person.get_keypoint_2d(i)
            kp_3d = first_person.get_keypoint_3d(i)
            print(f"Keypoint {i}:")
            print(f"  2D: ({kp_2d.x:.2f}, {kp_2d.y:.2f})")
            if kp_3d:
                print(f"  3D: ({kp_3d.x:.4f}, {kp_3d.y:.4f}, {kp_3d.z:.4f})")
    
    # Get trajectory of first person across all frames
    print(f"\n=== Person Trajectory ===")
    trajectory = pose_sequence.get_person_trajectory(person_index=0)
    print(f"Tracked person across {len(trajectory)} frames")
    
    # Show movement of a specific keypoint (e.g., nose - usually index 0)
    print(f"\n=== Nose Movement (first 10 frames) ===")
    for i in range(min(10, len(trajectory))):
        nose_2d = trajectory[i].get_keypoint_2d(0)
        print(f"Frame {i}: ({nose_2d.x:.2f}, {nose_2d.y:.2f})")
    
    # Save to new format (optional)
    print(f"\n=== Saving to Standardized Format ===")
    output_path = "test_scripts/pose_sequence_standardized.json"
    pose_sequence.to_json(output_path)
    print(f"Saved standardized pose data to: {output_path}")
    
    # Demonstrate loading from standardized format
    print(f"\n=== Loading from Standardized Format ===")
    from app.models import PoseSequence
    reloaded = PoseSequence.from_json(output_path)
    print(f"Reloaded {reloaded.num_frames} frames from standardized format")
    print(f"Model: {reloaded.model_name}")


if __name__ == "__main__":
    main()
