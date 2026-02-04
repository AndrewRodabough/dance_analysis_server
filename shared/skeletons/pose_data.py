"""
Core data structures for pose estimation results.
Provides a consistent interface regardless of the underlying model (MediaPipe, MMPose, etc.)
"""
from dataclasses import dataclass
from typing import List, Optional
import json
import numpy as np

class VectorizedPoseData:
    """
    Wrapper combining VectorizedSkeleton with confidence scores.
    Keeps skeleton focused on geometry while bundling quality metrics.
    """
    
    def __init__(self, skeleton, confidence: np.ndarray):
        """
        Args:
            skeleton: VectorizedSkeleton with loaded motion data
            confidence: Confidence scores, shape (Frames, Joints)
        """
        from .skeleton import VectorizedSkeleton
        
        if not isinstance(skeleton, VectorizedSkeleton):
            raise TypeError("skeleton must be a VectorizedSkeleton instance")
        
        if skeleton.data is None:
            raise ValueError("Skeleton must have loaded data before creating VectorizedPoseData")
        
        # Validate confidence shape
        expected_shape = (skeleton.num_frames, skeleton.num_joints)
        if confidence.shape != expected_shape:
            raise ValueError(
                f"Confidence shape {confidence.shape} doesn't match "
                f"expected {expected_shape} (frames, joints)"
            )
        
        self.skeleton = skeleton
        self.confidence = confidence.astype(np.float32)
    
    @property
    def num_frames(self) -> int:
        """Number of frames in the sequence."""
        return self.skeleton.num_frames
    
    @property
    def num_joints(self) -> int:
        """Number of joints in the skeleton."""
        return self.skeleton.num_joints
    
    def get_high_confidence_mask(self, threshold: float = 0.5) -> np.ndarray:
        """
        Returns boolean mask for joints above confidence threshold.
        
        Args:
            threshold: Confidence threshold (0-1)
            
        Returns:
            Boolean array of shape (Frames, Joints)
        """
        return self.confidence > threshold
    
    def get_weighted_bone_lengths(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get bone lengths, masking out bones with low-confidence endpoints.
        
        Args:
            threshold: Minimum confidence for both endpoints
            
        Returns:
            Bone lengths array, NaN where confidence is too low
        """
        lengths = self.skeleton.get_bone_lengths()
        
        # Get confidence for parent and child joints
        parent_conf = self.confidence[:, self.skeleton.bones_index[0]]
        child_conf = self.confidence[:, self.skeleton.bones_index[1]]
        
        # Both endpoints must meet threshold
        valid_mask = (parent_conf >= threshold) & (child_conf >= threshold)
        
        # Mask invalid bones with NaN
        result = lengths.copy()
        result[~valid_mask] = np.nan
        
        return result
    
    def get_weighted_bone_angles(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get bone angles, masking out angles with low-confidence joints.
        
        Args:
            threshold: Minimum confidence for all three joints
            
        Returns:
            Angles array in radians, NaN where confidence is too low
        """
        angles = self.skeleton.get_bone_angles()
        
        # Get confidence for all three joints in each angle
        start_conf = self.confidence[:, self.skeleton.joints_index[0]]
        pivot_conf = self.confidence[:, self.skeleton.joints_index[1]]
        end_conf = self.confidence[:, self.skeleton.joints_index[2]]
        
        # All three joints must meet threshold
        valid_mask = (start_conf >= threshold) & (pivot_conf >= threshold) & (end_conf >= threshold)
        
        # Mask invalid angles with NaN
        result = angles.copy()
        result[~valid_mask] = np.nan
        
        return result
    
    def get_joint_confidence(self, joint_name: str) -> np.ndarray:
        """
        Get confidence scores for a specific joint across all frames.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            Array of shape (Frames,) with confidence values
        """
        if joint_name not in self.skeleton.name_to_idx:
            raise ValueError(f"Joint '{joint_name}' not found in skeleton")
        
        joint_idx = self.skeleton.name_to_idx[joint_name]
        return self.confidence[:, joint_idx]
    
    def get_average_confidence(self) -> float:
        """Get average confidence across all frames and joints."""
        return np.mean(self.confidence).item()
    
    def get_frame_confidence(self) -> np.ndarray:
        """
        Get average confidence per frame.
        
        Returns:
            Array of shape (Frames,) with average confidence per frame
        """
        return np.mean(self.confidence, axis=1)
    
    def filter_low_confidence_frames(self, threshold: float = 0.5) -> 'VectorizedPoseData':
        """
        Return a new VectorizedPoseData with only high-confidence frames.
        
        Args:
            threshold: Minimum average confidence per frame
            
        Returns:
            New VectorizedPoseData instance with filtered frames
        """
        if self.skeleton.data is None:
            raise ValueError("Skeleton data must be loaded before filtering frames")
        
        frame_conf = self.get_frame_confidence()
        valid_frames = frame_conf >= threshold
        
        # Filter skeleton data and confidence
        filtered_data = self.skeleton.data[valid_frames]
        filtered_conf = self.confidence[valid_frames]
        
        # Create new skeleton with filtered data
        from .skeleton import VectorizedSkeleton
        new_skeleton = VectorizedSkeleton(
            self.skeleton.idx_to_name,
            [(self.skeleton.idx_to_name[p], self.skeleton.idx_to_name[c]) 
             for p, c in self.skeleton.bones_index.T]
        )
        new_skeleton.load_data(filtered_data)
        
        return VectorizedPoseData(new_skeleton, filtered_conf)


class ScenePoseData:
    """
    Scene-level container for multi-person pose data with video metadata.
    Holds all pose data for a complete video/scene.
    """
    
    def __init__(
        self,
        people: List[VectorizedPoseData],
        fps: float,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
        model_name: Optional[str] = None,
        video_path: Optional[str] = None
    ):
        """
        Args:
            people: List of VectorizedPoseData, one per tracked person
            fps: Frames per second of the video
            video_width: Video width in pixels
            video_height: Video height in pixels
            model_name: Name of the pose estimation model used
            video_path: Path to the source video file
        """
        if not people:
            raise ValueError("At least one person must be provided")
        
        # Validate all people have same number of frames
        num_frames = people[0].num_frames
        if not all(p.num_frames == num_frames for p in people):
            raise ValueError("All people must have the same number of frames")
        
        self.people = people
        self.fps = fps
        self.video_width = video_width
        self.video_height = video_height
        self.model_name = model_name
        self.video_path = video_path
    
    @property
    def num_people(self) -> int:
        """Number of tracked people in the scene."""
        return len(self.people)
    
    @property
    def num_frames(self) -> int:
        """Number of frames in the video."""
        return self.people[0].num_frames
    
    @property
    def duration(self) -> float:
        """Duration of the video in seconds."""
        return self.num_frames / self.fps
    
    def get_person(self, person_index: int) -> VectorizedPoseData:
        """
        Get pose data for a specific person.
        
        Args:
            person_index: Index of the person (0-based)
            
        Returns:
            VectorizedPoseData for that person
        """
        if person_index < 0 or person_index >= self.num_people:
            raise IndexError(f"Person index {person_index} out of range [0, {self.num_people})")
        return self.people[person_index]
    
    def get_primary_person(self) -> VectorizedPoseData:
        """
        Get the primary person (usually person 0 or highest confidence).
        
        Returns:
            VectorizedPoseData for the primary person
        """
        if self.num_people == 1:
            return self.people[0]
        
        # Find person with highest average confidence
        best_idx = 0
        best_conf = self.people[0].get_average_confidence()
        
        for i, person in enumerate(self.people[1:], 1):
            conf = person.get_average_confidence()
            if conf > best_conf:
                best_conf = conf
                best_idx = i
        
        return self.people[best_idx]
    
    def get_frame_timestamp(self, frame_index: int) -> float:
        """
        Get timestamp for a specific frame in seconds.
        
        Args:
            frame_index: Frame number (0-based)
            
        Returns:
            Timestamp in seconds
        """
        if frame_index < 0 or frame_index >= self.num_frames:
            raise IndexError(f"Frame index {frame_index} out of range [0, {self.num_frames})")
        return frame_index / self.fps
    
    def get_scene_average_confidence(self) -> float:
        """Get average confidence across all people and frames."""
        return np.mean([person.get_average_confidence() for person in self.people]).item()
    
    def filter_by_confidence(self, threshold: float = 0.5) -> 'ScenePoseData':
        """
        Create a new ScenePoseData with low-confidence frames removed from all people.
        
        Args:
            threshold: Minimum average confidence per frame
            
        Returns:
            New ScenePoseData with filtered frames
        """
        filtered_people = [person.filter_low_confidence_frames(threshold) for person in self.people]
        
        return ScenePoseData(
            people=filtered_people,
            fps=self.fps,
            video_width=self.video_width,
            video_height=self.video_height,
            model_name=self.model_name,
            video_path=self.video_path
        )
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "num_people": self.num_people,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "duration": self.duration,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "model_name": self.model_name,
            "video_path": self.video_path,
            "scene_average_confidence": self.get_scene_average_confidence(),
            "people_average_confidence": [p.get_average_confidence() for p in self.people]
        }
    
    def __repr__(self) -> str:
        return (
            f"ScenePoseData(people={self.num_people}, frames={self.num_frames}, "
            f"fps={self.fps:.2f}, duration={self.duration:.2f}s, "
            f"avg_conf={self.get_scene_average_confidence():.3f})"
        )
