"""
Example demonstrating semantic access to keypoints by body part name.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.mappers import load_pose_data
from app.models import COCOWholebodyKeypoint, get_keypoint_name


def main():
    """Demonstrate semantic keypoint access."""
    
    # Load pose data
    print("Loading pose data...")
    pose_seq = load_pose_data(
        json_2d_path="test_scripts/estimation_2d.json",
        json_3d_path="test_scripts/estimation_3d.json",
        model_type="mediapipe",
        fps=30.0
    )
    
    print(f"\n{'='*60}")
    print("SEMANTIC KEYPOINT ACCESS EXAMPLES")
    print(f"{'='*60}\n")
    
    # Get a specific frame
    frame = pose_seq.get_frame(10)
    person = frame.get_person(0)
    
    # Example 1: Access keypoints by semantic name
    print("1. Accessing keypoints by body part name:")
    print("-" * 40)
    
    nose = person.get_body_part("NOSE")
    left_elbow = person.get_body_part("LEFT_ELBOW")
    right_shoulder = person.get_body_part("RIGHT_SHOULDER")
    left_hip = person.get_body_part("LEFT_HIP")
    
    print(f"Nose position:           ({nose.x:.1f}, {nose.y:.1f})")
    print(f"Left elbow position:     ({left_elbow.x:.1f}, {left_elbow.y:.1f})")
    print(f"Right shoulder position: ({right_shoulder.x:.1f}, {right_shoulder.y:.1f})")
    print(f"Left hip position:       ({left_hip.x:.1f}, {left_hip.y:.1f})")
    
    # Example 2: Access 3D coordinates
    print(f"\n2. Accessing 3D coordinates:")
    print("-" * 40)
    
    nose_3d = person.get_body_part("NOSE", dimension="3d")
    left_elbow_3d = person.get_body_part("LEFT_ELBOW", dimension="3d")
    
    if nose_3d:
        print(f"Nose 3D:       ({nose_3d.x:.3f}, {nose_3d.y:.3f}, {nose_3d.z:.3f}) meters")
    if left_elbow_3d:
        print(f"Left elbow 3D: ({left_elbow_3d.x:.3f}, {left_elbow_3d.y:.3f}, {left_elbow_3d.z:.3f}) meters")
    
    # Example 3: Check body alignment
    print(f"\n3. Analyzing body alignment:")
    print("-" * 40)
    
    left_shoulder = person.get_body_part("LEFT_SHOULDER")
    right_shoulder = person.get_body_part("RIGHT_SHOULDER")
    
    shoulder_width = abs(right_shoulder.x - left_shoulder.x)
    shoulder_alignment = abs(right_shoulder.y - left_shoulder.y)
    
    print(f"Shoulder width (pixels):     {shoulder_width:.1f}")
    print(f"Shoulder height difference:  {shoulder_alignment:.1f}")
    print(f"Shoulders level: {'Yes' if shoulder_alignment < 20 else 'No (tilted)'}")
    
    # Example 4: Track elbow movement across frames
    print(f"\n4. Tracking left elbow movement over time:")
    print("-" * 40)
    
    trajectory = pose_seq.get_person_trajectory(0)
    print(f"Frame | Left Elbow Position")
    print("-" * 40)
    
    for frame_idx in [0, 10, 20, 30, 40]:
        if frame_idx < len(trajectory):
            person_at_frame = trajectory[frame_idx]
            elbow = person_at_frame.get_body_part("LEFT_ELBOW")
            print(f"{frame_idx:5d} | ({elbow.x:7.1f}, {elbow.y:7.1f})")
    
    # Example 5: Calculate joint angles (simplified)
    print(f"\n5. Calculating left arm angle (elbow):")
    print("-" * 40)
    
    shoulder = person.get_body_part("LEFT_SHOULDER")
    elbow = person.get_body_part("LEFT_ELBOW")
    wrist = person.get_body_part("LEFT_WRIST")
    
    import math
    
    # Vector from shoulder to elbow
    v1_x = elbow.x - shoulder.x
    v1_y = elbow.y - shoulder.y
    
    # Vector from elbow to wrist
    v2_x = wrist.x - elbow.x
    v2_y = wrist.y - elbow.y
    
    # Calculate angle using dot product
    dot_product = v1_x * v2_x + v1_y * v2_y
    mag1 = math.sqrt(v1_x**2 + v1_y**2)
    mag2 = math.sqrt(v2_x**2 + v2_y**2)
    
    if mag1 > 0 and mag2 > 0:
        cos_angle = dot_product / (mag1 * mag2)
        # Clamp to [-1, 1] to avoid numerical issues
        cos_angle = max(-1, min(1, cos_angle))
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        print(f"Left elbow angle: {angle_deg:.1f}°")
        print(f"Arm position: {'Extended' if angle_deg > 150 else 'Bent'}")
    
    # Example 6: Available keypoint names
    print(f"\n6. All available body part names:")
    print("-" * 40)
    
    from app.models import KeypointSchema
    schema = KeypointSchema("coco_wholebody")
    
    print("\nMain body keypoints:")
    body_parts = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE"
    ]
    
    for part in body_parts:
        idx = schema.get_index(part)
        print(f"  {part:20s} -> index {idx}")
    
    print("\nAdditional features:")
    print("  - Face landmarks: 68 points (indices 23-90)")
    print("  - Left hand: 21 points (indices 91-111)")
    print("  - Right hand: 21 points (indices 112-132)")
    print("  - Feet: 6 points (toes, heels)")
    
    # Example 7: Check if person is facing camera
    print(f"\n7. Estimating person orientation:")
    print("-" * 40)
    
    left_ear = person.get_body_part("LEFT_EAR")
    right_ear = person.get_body_part("RIGHT_EAR")
    nose = person.get_body_part("NOSE")
    
    # Simple heuristic: if both ears are visible and roughly equidistant from nose,
    # person is likely facing camera
    left_ear_visible = left_ear.x > 0
    right_ear_visible = right_ear.x > 0
    
    if left_ear_visible and right_ear_visible:
        left_dist = abs(nose.x - left_ear.x)
        right_dist = abs(nose.x - right_ear.x)
        symmetry = abs(left_dist - right_dist) / max(left_dist, right_dist)
        
        if symmetry < 0.3:
            print("Person is likely facing the camera (frontal view)")
        else:
            print(f"Person is turned {'left' if left_dist < right_dist else 'right'} (profile view)")
    else:
        print("Person is in profile (only one ear visible)")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
