"""
Test fixtures for pose estimation tests.
Provides sample data in various formats for testing mappers and data structures.

Includes both standard test cases and edge cases:
- Standard: Single/multi-person detection with normal data
- Edge cases: Empty frames, mismatched arrays, extreme values
"""
from typing import List, Tuple

# Sample MediaPipe-style data: Single person detected (collapsed structure)
MEDIAPIPE_SINGLE_PERSON_2D = [
    # Frame 0: Single person, 5 keypoints (simplified from 133)
    [[100.0, 200.0], [150.0, 250.0], [200.0, 300.0], [250.0, 350.0], [300.0, 400.0]],
    # Frame 1
    [[102.0, 202.0], [152.0, 252.0], [202.0, 302.0], [252.0, 352.0], [302.0, 402.0]],
]

MEDIAPIPE_SINGLE_PERSON_3D = [
    # Frame 0: Single person, 5 keypoints
    [[0.1, 0.2, 0.5], [0.15, 0.25, 0.55], [0.2, 0.3, 0.6], [0.25, 0.35, 0.65], [0.3, 0.4, 0.7]],
    # Frame 1
    [[0.102, 0.202, 0.502], [0.152, 0.252, 0.552], [0.202, 0.302, 0.602], [0.252, 0.352, 0.652], [0.302, 0.402, 0.702]],
]


# Sample MediaPipe-style data: Multiple people detected
MEDIAPIPE_MULTI_PERSON_2D = [
    # Frame 0: 2 people, 5 keypoints each
    [
        [[100.0, 200.0], [150.0, 250.0], [200.0, 300.0], [250.0, 350.0], [300.0, 400.0]],  # Person 0
        [[400.0, 500.0], [450.0, 550.0], [500.0, 600.0], [550.0, 650.0], [600.0, 700.0]],  # Person 1
    ],
    # Frame 1: 2 people
    [
        [[102.0, 202.0], [152.0, 252.0], [202.0, 302.0], [252.0, 352.0], [302.0, 402.0]],  # Person 0
        [[402.0, 502.0], [452.0, 552.0], [502.0, 602.0], [552.0, 652.0], [602.0, 702.0]],  # Person 1
    ],
]

MEDIAPIPE_MULTI_PERSON_3D = [
    # Frame 0: 2 people, 5 keypoints each
    [
        [[0.1, 0.2, 0.5], [0.15, 0.25, 0.55], [0.2, 0.3, 0.6], [0.25, 0.35, 0.65], [0.3, 0.4, 0.7]],  # Person 0
        [[0.4, 0.5, 0.9], [0.45, 0.55, 0.95], [0.5, 0.6, 1.0], [0.55, 0.65, 1.05], [0.6, 0.7, 1.1]],  # Person 1
    ],
    # Frame 1: 2 people
    [
        [[0.102, 0.202, 0.502], [0.152, 0.252, 0.552], [0.202, 0.302, 0.602], [0.252, 0.352, 0.652], [0.302, 0.402, 0.702]],
        [[0.402, 0.502, 0.902], [0.452, 0.552, 0.952], [0.502, 0.602, 1.002], [0.552, 0.652, 1.052], [0.602, 0.702, 1.102]],
    ],
]


# Alternative mapper format (for testing extensibility)
# This simulates a different model's output format
ALTERNATIVE_FORMAT_DATA = {
    "frames": [
        {
            "frame_id": 0,
            "timestamp": 0.0,
            "detections": [
                {
                    "person_id": 0,
                    "keypoints_2d": [[100.0, 200.0], [150.0, 250.0], [200.0, 300.0]],
                    "keypoints_3d": [[0.1, 0.2, 0.5], [0.15, 0.25, 0.55], [0.2, 0.3, 0.6]],
                    "confidence": [0.9, 0.85, 0.92],
                }
            ]
        },
        {
            "frame_id": 1,
            "timestamp": 0.033,
            "detections": [
                {
                    "person_id": 0,
                    "keypoints_2d": [[102.0, 202.0], [152.0, 252.0], [202.0, 302.0]],
                    "keypoints_3d": [[0.102, 0.202, 0.502], [0.152, 0.252, 0.552], [0.202, 0.302, 0.602]],
                    "confidence": [0.91, 0.86, 0.93],
                }
            ]
        }
    ],
    "metadata": {
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "model": "AlternativeModel"
    }
}


def create_mediapipe_single_person_data() -> Tuple[List, List]:
    """Returns MediaPipe-style data for a single person (collapsed structure)."""
    return MEDIAPIPE_SINGLE_PERSON_2D, MEDIAPIPE_SINGLE_PERSON_3D


def create_mediapipe_multi_person_data() -> Tuple[List, List]:
    """Returns MediaPipe-style data for multiple people."""
    return MEDIAPIPE_MULTI_PERSON_2D, MEDIAPIPE_MULTI_PERSON_3D


def create_alternative_format_data() -> dict:
    """Returns data in an alternative format for testing custom mappers."""
    return ALTERNATIVE_FORMAT_DATA


# Edge case fixtures for robust testing

EMPTY_FRAME_DATA_2D = [[]]  # 1 frame with no people detected
EMPTY_FRAME_DATA_3D = [[]]


EXTREME_COORDINATES_2D = [
    # Frame with extreme coordinate values (offscreen, negative, very large)
    [[-1000.0, -500.0], [0.0, 0.0], [10000.0, 8000.0], [1920.0, 1080.0]]
]
EXTREME_COORDINATES_3D = [
    [[-1.0, -0.5, -2.0], [0.0, 0.0, 0.0], [10.0, 8.0, 5.0], [1.0, 1.0, 1.0]]
]


SINGLE_KEYPOINT_2D = [[[100.0, 200.0]]]  # 1 frame, 1 person, 1 keypoint
SINGLE_KEYPOINT_3D = [[[0.1, 0.2, 0.5]]]


VARYING_PEOPLE_COUNT_2D = [
    [[100.0, 200.0]],  # Frame 0: 1 person (collapsed structure)
    [[[200.0, 300.0]], [[400.0, 500.0]]],  # Frame 1: 2 people (multi-person structure)
    [[300.0, 400.0]]  # Frame 2: 1 person again
]
VARYING_PEOPLE_COUNT_3D = [
    [[0.1, 0.2, 0.5]],
    [[[0.2, 0.3, 0.6]], [[0.4, 0.5, 0.9]]],
    [[0.3, 0.4, 0.7]]
]


def create_empty_frame_data() -> Tuple[List, List]:
    """Returns data with empty frames (no people detected)."""
    return EMPTY_FRAME_DATA_2D, EMPTY_FRAME_DATA_3D


def create_extreme_coordinate_data() -> Tuple[List, List]:
    """Returns data with extreme coordinate values for edge case testing."""
    return EXTREME_COORDINATES_2D, EXTREME_COORDINATES_3D


def create_single_keypoint_data() -> Tuple[List, List]:
    """Returns data with only one keypoint per person."""
    return SINGLE_KEYPOINT_2D, SINGLE_KEYPOINT_3D


def create_varying_people_count_data() -> Tuple[List, List]:
    """Returns data where the number of people varies across frames."""
    return VARYING_PEOPLE_COUNT_2D, VARYING_PEOPLE_COUNT_3D
