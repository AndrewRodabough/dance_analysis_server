import numpy as np
from typing import Dict, List, Tuple, Optional, Union
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

    def load_data(self, raw_data: Union[np.ndarray, list]):
        """
        Allocates memory and loads motion data.
        :param raw_data: Numpy array or nested list of shape (Frames, Joints, Channels)
        """
        # 1. Convert immediately. 
        # If it's a list, it becomes an array. If it's already an array, it ensures it's contiguous.
        data_array = np.ascontiguousarray(raw_data, dtype=np.float32)
        
        # 2. Safely extract the shape now that we guarantee it's a NumPy array
        frames, input_joints, channels = data_array.shape
        
        # Validation
        if input_joints != self.num_joints:
            raise ValueError(f"Skeleton expects {self.num_joints} joints, but input has {input_joints}.")

        # Update dimensions
        self.num_frames = frames
        
        # Assign the validated array
        self.data = data_array

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
        start_joints = self.data[:, self.joints_index[0], :]   # Shape: (Frames, Num_Angles, Channels)
        pivot_joints = self.data[:, self.joints_index[1], :]   # Shape: (Frames, Num_Angles, Channels)
        end_joints = self.data[:, self.joints_index[2], :]     # Shape: (Frames, Num_Angles, Channels)
        
        # Compute vectors from pivot to start and pivot to end
        vec_a = start_joints - pivot_joints  # Shape: (Frames, Num_Angles, Channels)
        vec_b = end_joints - pivot_joints    # Shape: (Frames, Num_Angles, Channels)
        
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

    def get_joint_velocities(self, fps: float = None):
        """
        Get velocities of all joints across all frames.
        Velocity is calculated as the change in position between consecutive frames.
        
        Args:
            fps: Frames per second for FPS normalization. If provided, velocities are 
                 scaled to be independent of frame rate. Default is None (raw pixel/meter differences).
        
        Returns:
            Numpy array of shape (Frames, Joints, Channels) with joint velocities.
            Frame 0 velocity is approximated using frame 1 velocity.
        """
        if self.data is None: 
            raise RuntimeError("No data loaded!")
        
        if self.num_frames == 0:
            return np.zeros((0, self.num_joints, 3), dtype=np.float32)
        
        if self.num_frames == 1:
            # Single frame: no velocity calculation possible
            return np.zeros_like(self.data)
        
        # Initialize velocities array with same shape as data
        velocities = np.zeros_like(self.data)  # Shape: (Frames, Joints, Channels)
        
        # For frames 1 and onwards, calculate velocity as position difference
        velocities[1:] = self.data[1:] - self.data[:-1]
        
        # Approximate frame 0 velocity using frame 1 velocity
        velocities[0] = velocities[1]
        
        # Apply FPS normalization if provided
        if fps is not None and fps > 0:
            # Normalize to 60 FPS baseline (current calibration was done at 60 FPS)
            fps_normalization_factor = 60.0 / fps
            velocities = velocities * fps_normalization_factor
        
        return velocities

    def get_joint_velocity(self, joint_name: str, fps: float = None):
        """
        Get velocity of a specific joint across all frames.
        
        Args:
            joint_name: Name of the joint
            fps: Frames per second for FPS normalization. If provided, velocities are 
                 scaled to be independent of frame rate. Default is None (raw differences).
            
        Returns:
            Numpy array of shape (Frames, Channels) with joint velocity.
            Frame 0 velocity is approximated using frame 1 velocity.
        """
        if self.data is None: 
            raise RuntimeError("No data loaded!")
        
        if joint_name not in self.name_to_idx:
            raise ValueError(f"Joint name not found: {joint_name}")
        
        joint_idx = self.name_to_idx[joint_name]
        
        if self.num_frames == 0:
            return np.zeros((0, 3), dtype=np.float32)
        
        if self.num_frames == 1:
            # Single frame: no velocity calculation possible
            return np.zeros((1, 3), dtype=np.float32)
        
        # Initialize velocity array
        velocity = np.zeros((self.num_frames, 3), dtype=np.float32)
        
        # For frames 1 and onwards, calculate velocity as position difference
        velocity[1:] = self.data[1:, joint_idx, :] - self.data[:-1, joint_idx, :]
        
        # Approximate frame 0 velocity using frame 1 velocity
        velocity[0] = velocity[1]
        
        # Apply FPS normalization if provided
        if fps is not None and fps > 0:
            # Normalize to 60 FPS baseline
            fps_normalization_factor = 60.0 / fps
            velocity = velocity * fps_normalization_factor
        
        return velocity

    def get_bone_length(self, parent_name: str, child_name: str):
        """
        Get the length of a specific bone across all frames.
        
        Args:
            parent_name: Name of the parent joint
            child_name: Name of the child joint
            
        Returns:
            Numpy array of shape (Frames,) with bone lengths
        """
        if self.data is None: raise RuntimeError("No data loaded!")
        
        # Get joint indices
        if parent_name not in self.name_to_idx or child_name not in self.name_to_idx:
            raise ValueError(f"Joint names not found: {parent_name}, {child_name}")
        
        parent_idx = self.name_to_idx[parent_name]
        child_idx = self.name_to_idx[child_name]
        
        # Extract joint positions
        parent_pos = self.data[:, parent_idx, :]  # Shape: (Frames, Channels)
        child_pos = self.data[:, child_idx, :]    # Shape: (Frames, Channels)
        
        # Compute bone length
        return np.linalg.norm(child_pos - parent_pos, axis=1)

    def get_angle(self, start_name: str, pivot_name: str, end_name: str):
        """
        Get the angle at a specific joint across all frames.
        
        Args:
            start_name: Name of the first joint
            pivot_name: Name of the pivot joint (where angle is measured)
            end_name: Name of the third joint
            
        Returns:
            Numpy array of shape (Frames,) with angles in radians
        """
        if self.data is None: raise RuntimeError("No data loaded!")
        
        # Get joint indices
        if (start_name not in self.name_to_idx or 
            pivot_name not in self.name_to_idx or 
            end_name not in self.name_to_idx):
            raise ValueError(f"Joint names not found: {start_name}, {pivot_name}, {end_name}")
        
        start_idx = self.name_to_idx[start_name]
        pivot_idx = self.name_to_idx[pivot_name]
        end_idx = self.name_to_idx[end_name]
        
        # Extract joint positions
        start_pos = self.data[:, start_idx, :]  # Shape: (Frames, Channels)
        pivot_pos = self.data[:, pivot_idx, :]  # Shape: (Frames, Channels)
        end_pos = self.data[:, end_idx, :]      # Shape: (Frames, Channels)
        
        # Compute vectors from pivot
        vec_a = start_pos - pivot_pos
        vec_b = end_pos - pivot_pos
        
        # Normalize vectors
        norm_a = np.linalg.norm(vec_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(vec_b, axis=1, keepdims=True)
        
        # Avoid division by zero
        norm_a = np.where(norm_a == 0, 1, norm_a)
        norm_b = np.where(norm_b == 0, 1, norm_b)
        
        vec_a_normalized = vec_a / norm_a
        vec_b_normalized = vec_b / norm_b
        
        # Compute dot product
        dot_product = np.sum(vec_a_normalized * vec_b_normalized, axis=1)
        
        # Clamp to [-1, 1] to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Compute angle in radians
        return np.arccos(dot_product)

    def get_angle_2d_signed(self, start_name: str, pivot_name: str, end_name: str):
        """
        Get the SIGNED angle at a knee joint in 2D (side view). 
        Uses cross product to determine bend direction.
        
        In image coordinates (X=right, Y=down, with up=-Y for analysis):
        - Positive cross product (counter-clockwise from hip to ankle) = backward lean/bend
        - Negative cross product (clockwise from hip to ankle) = forward lean/hyperextension
        
        Args:
            start_name: Name of the first joint (hip)
            pivot_name: Name of the pivot joint (knee)
            end_name: Name of the third joint (ankle)
            
        Returns:
            Dictionary with keys:
                - "unsigned_angle": angle in radians [0, π]
                - "cross_product_z": Z component of 2D cross product (sign indicates direction)
                - "is_forward_bend": bool, True if bending forward (hyperextension tendency)
        """
        if self.data is None:
            raise RuntimeError("No data loaded!")
        
        # Get joint indices
        if (start_name not in self.name_to_idx or 
            pivot_name not in self.name_to_idx or 
            end_name not in self.name_to_idx):
            raise ValueError(f"Joint names not found: {start_name}, {pivot_name}, {end_name}")
        
        start_idx = self.name_to_idx[start_name]
        pivot_idx = self.name_to_idx[pivot_name]
        end_idx = self.name_to_idx[end_name]
        
        # Extract joint positions (only first 2 channels for 2D)
        start_pos = self.data[:, start_idx, :2]  # Shape: (Frames, 2)
        pivot_pos = self.data[:, pivot_idx, :2]  # Shape: (Frames, 2)
        end_pos = self.data[:, end_idx, :2]      # Shape: (Frames, 2)
        
        # Compute vectors from pivot to start and pivot to end
        vec_a = start_pos - pivot_pos  # Hip->Knee in image coords
        vec_b = end_pos - pivot_pos     # Ankle->Knee in image coords
        
        # Normalize vectors
        norm_a = np.linalg.norm(vec_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(vec_b, axis=1, keepdims=True)
        
        # Avoid division by zero
        norm_a = np.where(norm_a == 0, 1, norm_a)
        norm_b = np.where(norm_b == 0, 1, norm_b)
        
        vec_a_normalized = vec_a / norm_a
        vec_b_normalized = vec_b / norm_b
        
        # Compute unsigned angle using dot product
        dot_product = np.sum(vec_a_normalized * vec_b_normalized, axis=1)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        unsigned_angle = np.arccos(dot_product)
        
        # Compute 2D cross product to determine bend direction
        # In 2D: cross(vec_a, vec_b) = vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0]
        cross_product_z = vec_a[:, 0] * vec_b[:, 1] - vec_a[:, 1] * vec_b[:, 0]
        
        # Determine if bend is forward (negative cross product = clockwise = forward bend)
        is_forward_bend = cross_product_z < 0
        
        return {
            "unsigned_angle": unsigned_angle,
            "cross_product_z": cross_product_z,
            "is_forward_bend": is_forward_bend
        }