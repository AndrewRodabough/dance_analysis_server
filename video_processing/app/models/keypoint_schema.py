"""
Keypoint schemas for different pose estimation formats.
Maps numeric indices to semantic body part names.
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Any, Union
import itertools
        
class VectorizedSkeleton:
    """
    High-performance skeleton data structure.
    Separates TOPOLOGY (init) from MOTION DATA (load_data).
    """
    def __init__(self, joint_names: List[str], str_bone_tuples: List[Tuple[str, str]]):
        """ 
        Defines the RIGID structure. No frame data is allocated here.
        """
        self.num_joints = len(joint_names)
        
        # 1. Topology Metadata
        self.name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(joint_names)}
        self.idx_to_name: List[str] = joint_names
        
        # 2. Conversion: String Tuples -> Int Tuples
        idx_bone_tuples: List[Tuple[int, int]] = []
        valid_bones = [] # Keep track of valid bones for graph building
        
        for parent_name, child_name in str_bone_tuples:
            if parent_name in self.name_to_idx and child_name in self.name_to_idx:
                p_idx = self.name_to_idx[parent_name]
                c_idx = self.name_to_idx[child_name]
                idx_bone_tuples.append((p_idx, c_idx))
                valid_bones.append((p_idx, c_idx))

        # 3. EDGE INDEX (Matrix Ops) -> Shape: (2, Num_Bones)
        if idx_bone_tuples:
            self.bones_index = np.array(idx_bone_tuples, dtype=np.int32).T
        else:
            self.bones_index = np.zeros((2, 0), dtype=np.int32)
            
        # 4. ADJACENCY LIST (Graph Ops) -> Shape: List of Lists
        self.bones = [[] for _ in range(self.num_joints)]
        for p_idx, c_idx in valid_bones:
            self.bones[p_idx].append(c_idx)
            self.bones[c_idx].append(p_idx) # Bidirectional

        # 5. ANGLE INDEX (Matrix Ops) -> Shape: (3, Num_Angles)
        idx_joint_triplets: List[Tuple[int, int, int]] = []
        
        for pivot_idx in range(self.num_joints):
            neighbors = self.bones[pivot_idx]
            if len(neighbors) >= 2:
                for start, end in itertools.combinations(neighbors, 2):
                    idx_joint_triplets.append((start, pivot_idx, end))

        if idx_joint_triplets:
            self.joints_index = np.array(idx_joint_triplets, dtype=np.int32).T
        else:
            self.joints_index = np.zeros((3, 0), dtype=np.int32)

        # Placeholder for data (initialized as None)
        self.data: Optional[np.ndarray] = None
        self.num_frames = 0

    def load_data(self, raw_data: np.ndarray):
        """
        Allocates memory and loads motion data.
        :param raw_data: Numpy array of shape (Frames, Joints, 3)
        """
        frames, input_joints, channels = raw_data.shape
        
        # Validation
        if input_joints != self.num_joints:
            raise ValueError(f"Skeleton expects {self.num_joints} joints, but input has {input_joints}.")

        # Update dimensions
        self.num_frames = frames
        
        # Allocate & Copy (Safe deep copy to ensure contiguous memory)
        self.data = np.ascontiguousarray(raw_data, dtype=np.float32)

        print(f"Loaded {self.num_frames} frames. Skeleton is ready for vector math.")

    def get_bone_lengths(self):
        """Example of vector math working on the loaded data"""
        if self.data is None: raise RuntimeError("No data loaded!")
        
        # Fancy Indexing using the pre-computed topology
        parents = self.data[:, self.bones_index[0], :]
        children = self.data[:, self.bones_index[1], :]
        return np.linalg.norm(children - parents, axis=2)

    def get_bone_angles(self):
        """
        Calculate angles at each joint using vectorized operations.
        Returns angles in radians for all joint triplets across all frames.
        Shape: (Frames, Num_Angles)
        """
        if self.data is None: raise RuntimeError("No data loaded!")
        
        # Fancy Indexing using the pre-computed angle topology (start, pivot, end)
        start_joints = self.data[:, self.joints_index[0], :]   # Shape: (Frames, Num_Angles, 3)
        pivot_joints = self.data[:, self.joints_index[1], :]   # Shape: (Frames, Num_Angles, 3)
        end_joints = self.data[:, self.joints_index[2], :]     # Shape: (Frames, Num_Angles, 3)
        
        # Compute vectors from pivot to start and pivot to end
        vec_a = start_joints - pivot_joints  # Shape: (Frames, Num_Angles, 3)
        vec_b = end_joints - pivot_joints    # Shape: (Frames, Num_Angles, 3)
        
        # Normalize vectors
        norm_a = np.linalg.norm(vec_a, axis=2, keepdims=True)  # Shape: (Frames, Num_Angles, 1)
        norm_b = np.linalg.norm(vec_b, axis=2, keepdims=True)
        
        # Avoid division by zero
        norm_a = np.where(norm_a == 0, 1, norm_a)
        norm_b = np.where(norm_b == 0, 1, norm_b)
        
        vec_a_normalized = vec_a / norm_a
        vec_b_normalized = vec_b / norm_b
        
        # Compute dot product
        dot_product = np.sum(vec_a_normalized * vec_b_normalized, axis=2)  # Shape: (Frames, Num_Angles)
        
        # Clamp to [-1, 1] to avoid numerical errors with arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Compute angles in radians
        angles = np.arccos(dot_product)  # Shape: (Frames, Num_Angles)
        
        return angles



class COCOWholebodyKeypoint(Enum):
    """
    COCO Wholebody format (133 keypoints total).
    used by models like RTMW.
    
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

class Human36MKeypoint(Enum):
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
    SPINE = 7 # Spine / torso
    NECK = 8 # Neck base 
    HEAD = 9 # Head / jaw
    SITE = 10 # Head top / crown
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 12
    LEFT_WRIST = 13
    RIGHT_SHOULDER = 14
    RIGHT_ELBOW = 15
    RIGHT_WRIST = 16

HUMAN36_SKELETON: List[Tuple[int, int]] = [



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
