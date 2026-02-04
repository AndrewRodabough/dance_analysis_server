"""
Keypoint schemas for different pose estimation formats.
Maps numeric indices to semantic body part names.
"""
from enum import Enum
from typing import Dict, List, Tuple


class COCOWholebodyKeypoint(Enum):
    """
    COCO Wholebody format (133 keypoints total).
    
    Layout:
    - 0-16: Body keypoints (17 points)
    - 17-22: Feet keypoints (6 points)
    - 23-90: Face keypoints (68 points)
    - 91-111: Left hand keypoints (21 points)
    - 112-132: Right hand keypoints (21 points)
    """
    
    # Body (17 keypoints) - indices 0-16
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    # Feet (6 keypoints) - indices 17-22
    LEFT_BIG_TOE = 17
    LEFT_SMALL_TOE = 18
    LEFT_HEEL = 19
    RIGHT_BIG_TOE = 20
    RIGHT_SMALL_TOE = 21
    RIGHT_HEEL = 22
    
    # Face landmarks (68 keypoints) - indices 23-90
    # Simplified grouping for common use
    FACE_START = 23
    FACE_END = 90
    
    # Left Hand (21 keypoints) - indices 91-111
    LEFT_HAND_START = 91
    LEFT_HAND_WRIST = 91
    LEFT_HAND_THUMB_1 = 92
    LEFT_HAND_THUMB_2 = 93
    LEFT_HAND_THUMB_3 = 94
    LEFT_HAND_THUMB_4 = 95
    LEFT_HAND_INDEX_1 = 96
    LEFT_HAND_INDEX_2 = 97
    LEFT_HAND_INDEX_3 = 98
    LEFT_HAND_INDEX_4 = 99
    LEFT_HAND_MIDDLE_1 = 100
    LEFT_HAND_MIDDLE_2 = 101
    LEFT_HAND_MIDDLE_3 = 102
    LEFT_HAND_MIDDLE_4 = 103
    LEFT_HAND_RING_1 = 104
    LEFT_HAND_RING_2 = 105
    LEFT_HAND_RING_3 = 106
    LEFT_HAND_RING_4 = 107
    LEFT_HAND_PINKY_1 = 108
    LEFT_HAND_PINKY_2 = 109
    LEFT_HAND_PINKY_3 = 110
    LEFT_HAND_PINKY_4 = 111
    LEFT_HAND_END = 111
    
    # Right Hand (21 keypoints) - indices 112-132
    RIGHT_HAND_START = 112
    RIGHT_HAND_WRIST = 112
    RIGHT_HAND_THUMB_1 = 113
    RIGHT_HAND_THUMB_2 = 114
    RIGHT_HAND_THUMB_3 = 115
    RIGHT_HAND_THUMB_4 = 116
    RIGHT_HAND_INDEX_1 = 117
    RIGHT_HAND_INDEX_2 = 118
    RIGHT_HAND_INDEX_3 = 119
    RIGHT_HAND_INDEX_4 = 120
    RIGHT_HAND_MIDDLE_1 = 121
    RIGHT_HAND_MIDDLE_2 = 122
    RIGHT_HAND_MIDDLE_3 = 123
    RIGHT_HAND_MIDDLE_4 = 124
    RIGHT_HAND_RING_1 = 125
    RIGHT_HAND_RING_2 = 126
    RIGHT_HAND_RING_3 = 127
    RIGHT_HAND_RING_4 = 128
    RIGHT_HAND_PINKY_1 = 129
    RIGHT_HAND_PINKY_2 = 130
    RIGHT_HAND_PINKY_3 = 131
    RIGHT_HAND_PINKY_4 = 132
    RIGHT_HAND_END = 132


# Skeleton connections for visualization
COCO_WHOLEBODY_SKELETON: List[Tuple[int, int]] = [
    # Body connections
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),  # Legs
    (5, 11), (6, 12), (5, 6),  # Torso
    (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),  # Head
]


class KeypointSchema:
    """Helper class to work with keypoint schemas."""
    
    def __init__(self, schema_type: str = "coco_wholebody"):
        """
        Initialize with a specific schema type.
        
        Args:
            schema_type: Type of keypoint schema ("coco_wholebody", etc.)
        """
        self.schema_type = schema_type
        
        if schema_type == "coco_wholebody":
            self.keypoint_enum = COCOWholebodyKeypoint
            self.skeleton = COCO_WHOLEBODY_SKELETON
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")
        
        # Create reverse mapping: index -> name
        self._index_to_name = {kp.value: kp.name for kp in self.keypoint_enum}
        self._name_to_index = {kp.name: kp.value for kp in self.keypoint_enum}
    
    def get_name(self, index: int) -> str:
        """Get the body part name for a keypoint index."""
        return self._index_to_name.get(index, f"KEYPOINT_{index}")
    
    def get_index(self, name: str) -> int:
        """Get the keypoint index for a body part name."""
        return self._name_to_index.get(name.upper())
    
    def get_body_keypoints(self) -> List[int]:
        """Get indices of main body keypoints (excluding face/hands)."""
        return list(range(0, 23))  # Body + feet
    
    def get_face_keypoints(self) -> List[int]:
        """Get indices of face keypoints."""
        return list(range(23, 91))
    
    def get_left_hand_keypoints(self) -> List[int]:
        """Get indices of left hand keypoints."""
        return list(range(91, 112))
    
    def get_right_hand_keypoints(self) -> List[int]:
        """Get indices of right hand keypoints."""
        return list(range(112, 133))
    
    def get_keypoint_groups(self) -> Dict[str, List[int]]:
        """Get all keypoint groups."""
        return {
            "body": self.get_body_keypoints(),
            "face": self.get_face_keypoints(),
            "left_hand": self.get_left_hand_keypoints(),
            "right_hand": self.get_right_hand_keypoints(),
        }


class H36MKeypoint(Enum):
    """
    Human3.6M (H36M) 17-keypoint format.
    Used by MotionBERT and other 3D pose estimators.
    """
    PELVIS = 0
    RIGHT_HIP = 1
    RIGHT_KNEE = 2
    RIGHT_ANKLE = 3
    LEFT_HIP = 4
    LEFT_KNEE = 5
    LEFT_ANKLE = 6
    SPINE = 7
    THORAX = 8
    NECK = 9
    HEAD = 10
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 12
    LEFT_WRIST = 13
    RIGHT_SHOULDER = 14
    RIGHT_ELBOW = 15
    RIGHT_WRIST = 16


# H36M skeleton connections
H36M_SKELETON: List[Tuple[int, int]] = [
    # Spine chain
    (0, 7),    # pelvis to spine
    (7, 8),    # spine to thorax
    (8, 9),    # thorax to neck
    (9, 10),   # neck to head
    # Right leg
    (0, 1),    # pelvis to right hip
    (1, 2),    # right hip to right knee
    (2, 3),    # right knee to right ankle
    # Left leg
    (0, 4),    # pelvis to left hip
    (4, 5),    # left hip to left knee
    (5, 6),    # left knee to left ankle
    # Left arm
    (8, 11),   # thorax to left shoulder
    (11, 12),  # left shoulder to left elbow
    (12, 13),  # left elbow to left wrist
    # Right arm
    (8, 14),   # thorax to right shoulder
    (14, 15),  # right shoulder to right elbow
    (15, 16),  # right elbow to right wrist
]


# Pre-instantiated default schema
DEFAULT_SCHEMA = KeypointSchema("coco_wholebody")


def get_keypoint_name(index: int, schema_type: str = "coco_wholebody") -> str:
    """
    Convenience function to get keypoint name.
    
    Args:
        index: Keypoint index
        schema_type: Schema to use (coco_wholebody or h36m)
    
    Returns:
        Human-readable name for the keypoint
    """
    if schema_type == "coco_wholebody":
        return DEFAULT_SCHEMA.get_name(index)
    elif schema_type == "h36m":
        return H36MKeypoint(index).name if 0 <= index <= 16 else f"KEYPOINT_{index}"
    else:
        schema = KeypointSchema(schema_type)
        return schema.get_name(index)
