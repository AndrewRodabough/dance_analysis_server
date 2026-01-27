# Quick Reference: Semantic Keypoint Access

## Basic Pattern

```python
# Load data
from app.models.mappers import load_pose_data
pose_seq = load_pose_data("2d.json", "3d.json", model_type="mediapipe")

# Get a person
frame = pose_seq.get_frame(frame_number)
person = frame.get_person(0)

# Access body parts
keypoint = person.get_body_part("BODY_PART_NAME", dimension="2d")  # or "3d"
```

## Body Part Names (COCO Wholebody)

### Head & Face
```python
"NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR"
```

### Upper Body
```python
"LEFT_SHOULDER", "RIGHT_SHOULDER"
"LEFT_ELBOW", "RIGHT_ELBOW"
"LEFT_WRIST", "RIGHT_WRIST"
```

### Lower Body
```python
"LEFT_HIP", "RIGHT_HIP"
"LEFT_KNEE", "RIGHT_KNEE"
"LEFT_ANKLE", "RIGHT_ANKLE"
```

### Feet
```python
"LEFT_BIG_TOE", "LEFT_SMALL_TOE", "LEFT_HEEL"
"RIGHT_BIG_TOE", "RIGHT_SMALL_TOE", "RIGHT_HEEL"
```

### Hands (21 points each)
```python
# Left hand
"LEFT_HAND_WRIST"
"LEFT_HAND_THUMB_1" through "LEFT_HAND_THUMB_4"
"LEFT_HAND_INDEX_1" through "LEFT_HAND_INDEX_4"
"LEFT_HAND_MIDDLE_1" through "LEFT_HAND_MIDDLE_4"
"LEFT_HAND_RING_1" through "LEFT_HAND_RING_4"
"LEFT_HAND_PINKY_1" through "LEFT_HAND_PINKY_4"

# Right hand (same pattern with RIGHT_HAND_*)
```

## Common Use Cases

### Check Person X's Body Part Y
```python
person = pose_seq.get_frame(10).get_person(0)
elbow = person.get_body_part("LEFT_ELBOW")
print(f"Left elbow at: ({elbow.x}, {elbow.y})")
```

### Track Movement Over Time
```python
trajectory = pose_seq.get_person_trajectory(person_index=0)
for frame_idx, person_pose in enumerate(trajectory):
    shoulder = person_pose.get_body_part("LEFT_SHOULDER")
    print(f"Frame {frame_idx}: shoulder at ({shoulder.x}, {shoulder.y})")
```

### Calculate Distances
```python
left_shoulder = person.get_body_part("LEFT_SHOULDER")
right_shoulder = person.get_body_part("RIGHT_SHOULDER")
shoulder_width = abs(right_shoulder.x - left_shoulder.x)
```

### Calculate Angles
```python
import math

shoulder = person.get_body_part("LEFT_SHOULDER")
elbow = person.get_body_part("LEFT_ELBOW")
wrist = person.get_body_part("LEFT_WRIST")

# Vector math for angle at elbow
v1_x, v1_y = elbow.x - shoulder.x, elbow.y - shoulder.y
v2_x, v2_y = wrist.x - elbow.x, wrist.y - elbow.y

dot = v1_x * v2_x + v1_y * v2_y
mag1 = math.sqrt(v1_x**2 + v1_y**2)
mag2 = math.sqrt(v2_x**2 + v2_y**2)

angle_rad = math.acos(dot / (mag1 * mag2))
angle_deg = math.degrees(angle_rad)
```

### 3D Position
```python
nose_3d = person.get_body_part("NOSE", dimension="3d")
print(f"3D position: ({nose_3d.x}, {nose_3d.y}, {nose_3d.z}) meters")
```

### Check Alignment
```python
left_hip = person.get_body_part("LEFT_HIP")
right_hip = person.get_body_part("RIGHT_HIP")
hip_level_diff = abs(left_hip.y - right_hip.y)
is_level = hip_level_diff < 20  # pixels
```

## Alternative: Using Numeric Indices

If you prefer or need numeric indices:

```python
from app.models import COCOWholebodyKeypoint

# Get index for a body part
elbow_idx = COCOWholebodyKeypoint.LEFT_ELBOW.value  # Returns 7

# Access by index
elbow = person.get_keypoint_2d(elbow_idx)
```

## Schema Information

```python
from app.models import KeypointSchema, get_keypoint_name

# Create schema helper
schema = KeypointSchema("coco_wholebody")

# Get name from index
name = schema.get_name(7)  # Returns "LEFT_ELBOW"

# Get index from name
index = schema.get_index("LEFT_ELBOW")  # Returns 7

# Get groups
body_indices = schema.get_body_keypoints()  # [0-22]
face_indices = schema.get_face_keypoints()  # [23-90]
left_hand_indices = schema.get_left_hand_keypoints()  # [91-111]
right_hand_indices = schema.get_right_hand_keypoints()  # [112-132]
```
