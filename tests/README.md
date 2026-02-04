# Unit Tests

This directory contains unit tests for the pose estimation data structures and mappers.

## Philosophy

These tests focus on **meaningful coverage** rather than 100% coverage:

- ✅ Test business logic and critical functionality
- ✅ Test edge cases and error conditions
- ✅ Test data transformations and serialization
- ❌ Don't test Python language features (dataclass getters/setters)
- ❌ Don't test trivial assignments

## Test Structure

- `test_pose_data.py` - Tests for core pose data structures (Keypoint2D, Keypoint3D, PersonPose, FramePose, PoseSequence)
- `test_mappers.py` - Tests for mapper system that converts raw model outputs to PoseSequence format
- `fixtures.py` - Test fixtures providing sample data in various formats, including edge cases

## Running Tests

### Using unittest (built-in)

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_pose_data
python -m unittest tests.test_mappers

# Run specific test class
python -m unittest tests.test_mappers.TestMediaPipeMapper

# Run specific test method
python -m unittest tests.test_mappers.TestMediaPipeMapper.test_single_person_structure_detection
```

### Using pytest (recommended for development)

```bash
# Install pytest if not already installed
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_mappers.py

# Run tests matching a pattern
pytest tests/ -k "mapper"

# Run with coverage report
pytest tests/ --cov=app.models --cov-report=html --cov-report=term-missing
```

## Test Coverage

### pose_data.py Tests

**Core Functionality:**

- ✅ Keypoint2D/3D serialization (to_dict, to_list)
- ✅ PersonPose keypoint access and structure
- ✅ FramePose multi-person support
- ✅ PoseSequence trajectory tracking
- ✅ JSON serialization roundtrip (save/load integrity)
- ✅ Dictionary-based construction (from_dict)

**Edge Cases:**

- ✅ Empty keypoint lists (detection failures)
- ✅ Mismatched 2D/3D keypoint counts
- ✅ Out-of-bounds access (raises IndexError)
- ✅ Missing optional fields (person_id, timestamp, 3D data)
- ✅ Extreme coordinate values (negative, zero, very large)
- ✅ Empty sequences and frames
- ✅ Zero/negative FPS handling
- ✅ Person trajectories with varying people counts

### mappers.py Tests

**Core Functionality:**

- ✅ MediaPipe single-person structure detection (collapsed format)
- ✅ MediaPipe multi-person structure detection
- ✅ Keypoint value parsing accuracy
- ✅ Timestamp calculation from FPS
- ✅ JSON file loading
- ✅ Mapper extensibility via dependency injection
- ✅ Custom mapper implementation example
- ✅ Factory pattern for mapper selection
- ✅ Output structure consistency
- ✅ Serialization roundtrip

**Edge Cases:**

- ✅ Empty frames (no people detected)
- ✅ Mismatched 2D/3D frame counts
- ✅ Completely empty arrays
- ✅ Single keypoint per person
- ✅ Mixed structure across frames (single → multi → single person)
- ✅ Zero/negative FPS handling
- ✅ Data shape validation

## Critical Test Cases

### Structure Detection (Most Important!)

The MediaPipe mapper handles an inconsistent data structure where the "people" dimension is collapsed when only one person is detected:

```python
# Single person: [frames][keypoints][coords] - people dimension missing!
single_person = [
    [[100, 200], [150, 250]],  # Frame 0: keypoints directly
]

# Multiple people: [frames][people][keypoints][coords]
multi_person = [
    [                          # Frame 0
        [[100, 200], [150, 250]],  # Person 0
        [[400, 500], [450, 550]],  # Person 1
    ]
]
```

Tests ensure the mapper correctly detects which structure each frame uses.

### Edge Case Testing

Real-world scenarios that must be handled:

1. **Empty frames** - No people detected in some frames
2. **Varying people count** - 1 person → 2 people → 1 person across frames
3. **Mismatched arrays** - 2D has 10 frames, 3D has 8 frames
4. **Extreme values** - Coordinates outside screen bounds, negative depth
5. **Minimal data** - Only required fields, all optionals missing

## Mapper Extensibility

The test suite includes examples of how to extend the mapper system for new models:

1. **Custom Mapper** - See `TestMapperExtensibility.test_custom_mapper_injection` for an example of creating a mapper for a different model format

2. **Factory Pattern** - See `TestMapperExtensibility.test_mapper_factory_pattern` for using a factory to select mappers

3. **Protocol** - The `MapperProtocol` class defines the interface all mappers should implement

## Adding New Mappers

To add support for a new model (e.g., MMPose, OpenPose):

1. Create a new mapper class in `app/models/mappers.py`:

   ```python
   class NewModelMapper:
       @staticmethod
       def from_raw_arrays(keypoints_2d_list, keypoints_3d_list, **kwargs) -> PoseSequence:
           # Implementation here
           pass
   ```

2. Add test fixtures in `tests/fixtures.py` for the new format:

   ```python
   NEW_MODEL_DATA_2D = [...]
   NEW_MODEL_DATA_3D = [...]

   def create_new_model_data() -> Tuple[List, List]:
       return NEW_MODEL_DATA_2D, NEW_MODEL_DATA_3D
   ```

3. Create test cases in `tests/test_mappers.py`:

   ```python
   class TestNewModelMapper(unittest.TestCase):
       def test_structure_detection(self):
           kp_2d, kp_3d = create_new_model_data()
           sequence = NewModelMapper.from_raw_arrays(kp_2d, kp_3d)
           # Assertions...

       def test_edge_cases(self):
           # Test empty frames, etc.
   ```

4. Update the `load_pose_data` function to support the new model type

## Test Data

The `fixtures.py` module provides sample data in multiple formats:

**Standard Test Cases:**

- MediaPipe single-person format (collapsed structure)
- MediaPipe multi-person format
- Alternative model format (for testing extensibility)

**Edge Case Fixtures:**

- Empty frame data (no people detected)
- Extreme coordinate values (negative, zero, offscreen)
- Single keypoint per person
- Varying people count across frames

This ensures mappers correctly handle different data structures and edge cases.

## Test Quality Principles

1. **Test behavior, not implementation** - Focus on what the code does, not how
2. **Test real scenarios** - Edge cases should reflect actual real-world failures
3. **Meaningful assertions** - Every assertion should verify something important
4. **Clear test names** - Should describe what's being tested and why it matters
5. **Minimal setup** - Only create the data needed for the specific test

## Current Test Statistics

- **Total tests**: 45
- **Test files**: 2 (test_pose_data.py, test_mappers.py)
- **Test classes**: 9
- **Edge case coverage**: High
- **All tests passing**: ✅

Last updated: 2024
