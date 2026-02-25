# Multi-Person Support in Dance Analysis Pipeline

## Overview

The pipeline now fully supports videos with **one or multiple dancers**. The JSON format is:

```
[frames][people][joints][channels]
```

This hierarchical structure allows the pipeline to:
- Auto-detect single vs multi-person videos
- Process each person independently while sharing video analysis
- Generate per-person feedback and scores

## Data Format

### Single-Person Format (Legacy)
```json
[
  [  // Frame 0
    [x, y, confidence],  // Joint 0
    [x, y, confidence],  // Joint 1
    ...
  ],
  [  // Frame 1
    ...
  ]
]
// Shape: (frames, joints, channels)
```

### Multi-Person Format (New)
```json
[
  [  // Frame 0
    [  // Person 0
      [x, y, confidence],  // Joint 0
      [x, y, confidence],  // Joint 1
      ...
    ],
    [  // Person 1
      [x, y, confidence],  // Joint 0
      ...
    ]
  ],
  [  // Frame 1
    ...
  ]
]
// Shape: (frames, people, joints, channels)
```

## Implementation

### Core Functions in `keypoint_loading.py`

#### `normalize_keypoints_format(keypoints)`
Automatically detects and normalizes data to 4D format `(frames, people, joints, channels)`.

**Handles:**
- Single-person 3D arrays → expands to 4D with 1 person
- Multi-person 4D arrays → passes through unchanged
- Logs format detection for debugging

**Returns:** `(normalized_keypoints, num_people)`

#### `extract_person_data(keypoints_all, person_idx)`
Extracts single-person data from multi-person array.

**Example:**
```python
multi_person_kp = np.array(...)  # Shape: (100, 3, 133, 3)
person_0_kp = extract_person_data(multi_person_kp, 0)  # Shape: (100, 133, 3)
```

#### `load_keypoints_from_json(json_2d_path, json_3d_path)`
Loads keypoints from JSON, auto-detects format, handles mismatches.

**Returns:** `(keypoints_2d, keypoints_3d, num_people)`
- All normalized to `(frames, people, joints, channels)`
- If 2D and 3D have different person counts, uses minimum

#### `create_skeleton_objects(keypoints_2d, keypoints_3d, person_idx)`
Creates skeleton for a specific person from multi-person data.

**Key features:**
- Accepts both 3D and 4D arrays (auto-normalizes)
- Extracts single person automatically
- Creates COCO-WholeBody (133 joints, 2D) and Human3.6M (17 joints, 3D) skeletons

#### `create_all_skeleton_objects(keypoints_2d, keypoints_3d)`
Convenience function to create skeletons for all people at once.

**Returns:** `Dict[person_idx -> (skeleton_2d, skeleton_3d)]`

### Pipeline Integration

#### Single-Person Analysis (Default)
```python
result = run_feedback_pipeline(
    job_id="job_123",
    s3_bucket="results",
    s3_client=client,
    local_keypoints_2d_path=Path("2d.json"),
    local_keypoints_3d_path=Path("3d.json"),
    # person_idx=0 by default (first person)
)
```

#### Multi-Person Analysis
```python
# Process specific person
result = run_feedback_pipeline(
    job_id="job_123",
    s3_bucket="results",
    s3_client=client,
    local_keypoints_2d_path=Path("2d.json"),
    local_keypoints_3d_path=Path("3d.json"),
    person_idx=1  # Process second person
)
```

#### Batch Process All People
```python
results = process_all_people(
    job_id="job_123",
    s3_bucket="results",
    s3_client=client,
    local_keypoints_2d_path=Path("2d.json"),
    local_keypoints_3d_path=Path("3d.json"),
)
# Results structure:
# {
#   'status': 'success',
#   'job_id': 'job_123',
#   'num_people': 3,
#   'people_results': {
#     0: { 'status': 'success', ... },
#     1: { 'status': 'success', ... },
#     2: { 'status': 'success', ... }
#   }
# }
```

## Error Handling

### Mismatched Person Counts
If 2D and 3D have different numbers of people, the pipeline:
1. Logs a warning
2. Uses the **minimum count** for both
3. Trims both arrays to match

**Example:**
- 2D has 3 people, 3D has 2 people
- Pipeline uses 2 people, trims 2D accordingly

### Out-of-Range Person Index
If requesting person index that doesn't exist:
```python
kp_2d, kp_3d, num_people = load_keypoints_from_json(...)  # num_people = 2
skeleton = create_skeleton_objects(..., person_idx=5)  # IndexError!
```

**Solution:** Always validate:
```python
if person_idx >= num_people:
    raise ValueError(f"Person {person_idx} out of range [0, {num_people})")
```

## Mismatch Handling

### Different Joint Counts Between 2D and 3D
- 2D uses COCO-WholeBody (133 joints)
- 3D uses Human3.6M (17 joints)
- This is **expected and handled** - each skeleton has its own joint set

### Different Frame Counts
Not currently handled - assumes both have same frame count. If they differ, will cause errors in skeleton creation.

## Testing

### Multi-Person Tests (12 new tests)
Located in `tests/test_multi_person_support.py`:

1. **Format Normalization** (4 tests)
   - Single-person format conversion
   - Multi-person format passthrough
   - Invalid format detection
   - Value preservation

2. **Data Extraction** (3 tests)
   - Extract first person
   - Extract middle person
   - Extract last person

3. **JSON Loading** (3 tests)
   - Single-person JSON loading
   - Multi-person JSON loading
   - Mismatched person count handling

4. **Pipeline Support** (2 tests)
   - Create skeleton for specific person
   - Person index validation

**All 106 tests passing:**
- 94 original tests (skeletons, registry, pose data, config loading)
- 12 new multi-person tests

### Running Tests
```bash
# Run all tests
pytest -v

# Run only multi-person tests
pytest tests/test_multi_person_support.py -v

# Run shared skeleton tests + multi-person
pytest shared/tests/ tests/test_multi_person_support.py -v
```

## Usage Examples

### Example 1: Single Dancer
```python
# Load single-person JSON
kp_2d, kp_3d, num_people = load_keypoints_from_json(
    Path("dancer_001_2d.json"),
    Path("dancer_001_3d.json")
)
# num_people = 1

# Automatically processes person 0
skeleton_2d, skeleton_3d = create_skeleton_objects(kp_2d, kp_3d)
```

### Example 2: Multiple Dancers
```python
# Load multi-person JSON
kp_2d, kp_3d, num_people = load_keypoints_from_json(
    Path("group_2d.json"),  # 4 dancers per frame
    Path("group_3d.json")
)
# num_people = 4

# Process all dancers
all_skeletons = create_all_skeleton_objects(kp_2d, kp_3d)
for person_idx, (skel_2d, skel_3d) in all_skeletons.items():
    print(f"Person {person_idx}: {skel_2d.num_frames} frames")

# Or process individually
for person_idx in range(num_people):
    skeleton_2d, skeleton_3d = create_skeleton_objects(
        kp_2d, kp_3d, person_idx=person_idx
    )
    # Generate feedback for this person...
```

### Example 3: Multi-Person Pipeline
```python
# Process all people in a video
results = process_all_people(
    job_id="group_dance_001",
    s3_bucket="dance-analysis",
    s3_client=boto3.client('s3'),
    local_keypoints_2d_path=Path("2d.json"),
    local_keypoints_3d_path=Path("3d.json"),
    local_video_path=Path("video.mp4")
)

# Results include per-person analysis
for person_idx, person_result in results['people_results'].items():
    print(f"Person {person_idx}: {person_result['status']}")
    # Access S3 paths: person_result['s3_results']['feedback_url']
```

## Architecture Benefits

1. **Backward Compatible**: Single-person data (3D arrays) still works
2. **Flexible**: Processes any number of people per frame
3. **Robust**: Handles mismatches gracefully
4. **Testable**: 12 new unit tests verify all scenarios
5. **Observable**: Detailed logging for debugging

## Future Enhancements

1. **Parallel Processing**: Use multiprocessing for independent per-person analysis
2. **Group Metrics**: Compare dancers to each other (e.g., synchronization)
3. **Occlusion Handling**: Better handling of partially visible dancers
4. **Confidence Weighting**: Filter low-confidence keypoints per person
5. **Skeleton Alignment**: Allow different skeleton types per person
