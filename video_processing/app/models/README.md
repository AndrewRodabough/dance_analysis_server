# Pose Data Models

This directory contains data structures and mappers for working with pose estimation results from various models.

## Overview

The core design provides a **consistent, model-agnostic interface** for pose data, regardless of which model (MediaPipe, MMPose, etc.) generated it.

## Data Structures

### Hierarchy
```
PoseSequence              # Complete video analysis
  └─ FramePose           # Single frame
      └─ PersonPose      # Single person in that frame
          ├─ Keypoint2D  # 2D pixel coordinates
          └─ Keypoint3D  # 3D world coordinates (optional)
```

### Classes

**`Keypoint2D`** - 2D keypoint with pixel coordinates
- `x`, `y`: Pixel coordinates
- `visibility`: Optional confidence score (0-1)

**`Keypoint3D`** - 3D keypoint with world coordinates
- `x`, `y`, `z`: World coordinates (typically meters)
- `visibility`: Optional confidence score (0-1)

**`PersonPose`** - Pose data for one person in one frame
- `keypoints_2d`: List of 2D keypoints
- `keypoints_3d`: Optional list of 3D keypoints
- `person_id`: Identifier for tracking
- Methods: `get_keypoint_2d(index)`, `get_keypoint_3d(index)`

**`FramePose`** - All people detected in a single frame
- `frame_number`: Frame index
- `people`: List of PersonPose objects
- `timestamp`: Optional timestamp in seconds
- Methods: `get_person(index)`

**`PoseSequence`** - Complete pose estimation for entire video
- `frames`: List of FramePose objects
- `fps`, `video_width`, `video_height`: Video metadata
- `model_name`: Which model generated this data
- Methods: `get_frame(index)`, `get_person_trajectory(person_index)`
- I/O: `to_json()`, `from_json()`

## Mappers

Mappers convert model-specific output formats into the standard `PoseSequence` format.

### MediaPipeMapper

Handles MediaPipe pose estimation output with **automatic structure detection**:

**Input formats supported:**
- Multiple people: `[frames][people][keypoints][coords]`
- Single person: `[frames][keypoints][coords]` (people dimension collapsed)

The mapper automatically detects which structure is present per frame and normalizes it.

**Usage:**
```python
from app.models.mappers import load_pose_data

# Load from JSON files
pose_seq = load_pose_data(
    json_2d_path="estimation_2d.json",
    json_3d_path="estimation_3d.json",
    model_type="mediapipe",
    fps=30.0,
    video_width=1920,
    video_height=1080
)

# Access data
print(f"Frames: {pose_seq.num_frames}")
print(f"Duration: {pose_seq.duration} seconds")

# Get first frame
frame = pose_seq.get_frame(0)
print(f"People in frame: {frame.num_people}")

# Get person trajectory
trajectory = pose_seq.get_person_trajectory(person_index=0)
```

### MMPoseMapper

Placeholder for future MMPose integration.

## Usage Examples

See `examples/load_pose_data.py` for a complete working example.

### Basic Usage
```python
from app.models import PoseSequence
from app.models.mappers import MediaPipeMapper

# Load from files
pose_seq = MediaPipeMapper.from_json_files(
    "estimation_2d.json",
    "estimation_3d.json",
    fps=30.0
)

# Examine a specific frame
frame_10 = pose_seq.get_frame(10)
for person_idx, person in enumerate(frame_10.people):
    nose = person.get_keypoint_2d(0)  # First keypoint is typically nose
    print(f"Person {person_idx} nose at: ({nose.x}, {nose.y})")

# Track a person across time
person_0_trajectory = pose_seq.get_person_trajectory(0)
for frame_idx, person_pose in enumerate(person_0_trajectory):
    nose = person_pose.get_keypoint_2d(0)
    print(f"Frame {frame_idx}: nose at ({nose.x}, {nose.y})")

# Save to standardized format
pose_seq.to_json("output.json")

# Load from standardized format
reloaded = PoseSequence.from_json("output.json")
```

### Semantic Keypoint Access

Access keypoints by **body part name** instead of numeric index:

```python
from app.models.mappers import load_pose_data

pose_seq = load_pose_data(
    "estimation_2d.json",
    "estimation_3d.json",
    model_type="mediapipe"
)

frame = pose_seq.get_frame(10)
person = frame.get_person(0)

# Access by semantic name (much more readable!)
nose = person.get_body_part("NOSE")
left_elbow = person.get_body_part("LEFT_ELBOW")
right_shoulder = person.get_body_part("RIGHT_SHOULDER")

print(f"Nose: ({nose.x}, {nose.y})")
print(f"Left elbow: ({left_elbow.x}, {left_elbow.y})")

# Access 3D coordinates
left_elbow_3d = person.get_body_part("LEFT_ELBOW", dimension="3d")
print(f"Left elbow 3D: ({left_elbow_3d.x}, {left_elbow_3d.y}, {left_elbow_3d.z})")

# Calculate body measurements
left_shoulder = person.get_body_part("LEFT_SHOULDER")
right_shoulder = person.get_body_part("RIGHT_SHOULDER")
shoulder_width = abs(right_shoulder.x - left_shoulder.x)

# Track specific body part over time
trajectory = pose_seq.get_person_trajectory(0)
for person_pose in trajectory:
    elbow = person_pose.get_body_part("LEFT_ELBOW")
    # Analyze elbow position...
```

**Available body parts:**
- **Body**: `NOSE`, `LEFT_EYE`, `RIGHT_EYE`, `LEFT_EAR`, `RIGHT_EAR`
- **Upper body**: `LEFT_SHOULDER`, `RIGHT_SHOULDER`, `LEFT_ELBOW`, `RIGHT_ELBOW`, `LEFT_WRIST`, `RIGHT_WRIST`
- **Lower body**: `LEFT_HIP`, `RIGHT_HIP`, `LEFT_KNEE`, `RIGHT_KNEE`, `LEFT_ANKLE`, `RIGHT_ANKLE`
- **Feet**: `LEFT_BIG_TOE`, `LEFT_SMALL_TOE`, `LEFT_HEEL`, `RIGHT_BIG_TOE`, `RIGHT_SMALL_TOE`, `RIGHT_HEEL`
- **Hands**: 21 points per hand (e.g., `LEFT_HAND_THUMB_1`, `RIGHT_HAND_INDEX_4`)
- **Face**: 68 facial landmarks (indices 23-90)

See `examples/semantic_access_example.py` for complete examples including:
- Joint angle calculation
- Body alignment analysis
- Movement tracking
- Orientation detection

## Known Issues

### MediaPipe Data Inconsistency
The current MediaPipe pipeline has a bug where it collapses the "people" dimension when only one person is detected. The mapper handles this automatically, but ideally the pipeline should be fixed to maintain consistent structure:

**Current behavior:**
- 2+ people detected: `[frames][people][keypoints][coords]`
- 1 person detected: `[frames][keypoints][coords]` ❌

**Expected behavior:**
- Always: `[frames][people][keypoints][coords]` ✓

## Adding New Model Support

To add support for a new pose estimation model:

1. Create a new mapper class (e.g., `YourModelMapper`)
2. Implement `from_raw_output()` or similar method
3. Convert to `PoseSequence` format
4. Update `load_pose_data()` function to support the new model type

Example:
```python
class YourModelMapper:
    @staticmethod
    def from_raw_output(model_output: dict) -> PoseSequence:
        frames = []
        # Parse model-specific format
        # Create Keypoint2D/3D, PersonPose, FramePose objects
        # ...
        return PoseSequence(frames=frames, model_name="YourModel")
```
