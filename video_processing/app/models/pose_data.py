"""
Core data structures for pose estimation results.
Provides a consistent interface regardless of the underlying model (MediaPipe, MMPose, etc.)
"""
from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class Keypoint2D:
    """2D keypoint with pixel coordinates."""
    x: float
    y: float
    visibility: Optional[float] = None  # Confidence score (0-1)
    
    def to_dict(self) -> dict:
        result = {"x": self.x, "y": self.y}
        if self.visibility is not None:
            result["visibility"] = self.visibility
        return result
    
    def to_list(self) -> List[float]:
        """Returns [x, y] format for compatibility."""
        return [self.x, self.y]


@dataclass
class Keypoint3D:
    """3D keypoint with world coordinates (typically in meters)."""
    x: float
    y: float
    z: float
    visibility: Optional[float] = None  # Confidence score (0-1)
    
    def to_dict(self) -> dict:
        result = {"x": self.x, "y": self.y, "z": self.z}
        if self.visibility is not None:
            result["visibility"] = self.visibility
        return result
    
    def to_list(self) -> List[float]:
        """Returns [x, y, z] format for compatibility."""
        return [self.x, self.y, self.z]


@dataclass
class PersonPose:
    """Pose data for a single person in a single frame."""
    keypoints_2d: List[Keypoint2D]
    keypoints_3d: Optional[List[Keypoint3D]] = None
    person_id: Optional[int] = None  # For tracking multiple people
    keypoint_schema: Optional[str] = "coco_wholebody"  # Schema type for semantic access
    
    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints_2d)
    
    def get_keypoint_2d(self, index: int) -> Keypoint2D:
        """Get a specific 2D keypoint by index."""
        return self.keypoints_2d[index]
    
    def get_keypoint_3d(self, index: int) -> Optional[Keypoint3D]:
        """Get a specific 3D keypoint by index."""
        if self.keypoints_3d is None:
            return None
        return self.keypoints_3d[index]
    
    def get_keypoint_by_name(self, name: str, dimension: str = "2d"):
        """Get a keypoint by semantic name (e.g., 'LEFT_ELBOW', 'NOSE').
        
        Args:
            name: Semantic name of the body part (from COCOWholebodyKeypoint)
            dimension: Either '2d' or '3d'
        
        Returns:
            Keypoint2D or Keypoint3D object, or None if not found/available
        
        Examples:
            >>> person.get_keypoint_by_name('NOSE')
            >>> person.get_keypoint_by_name('LEFT_ELBOW', '3d')
        """
        from .keypoint_schema import KeypointSchema
        schema = KeypointSchema(self.keypoint_schema)
        index = schema.get_index(name)
        
        if index is None:
            return None
        
        if dimension == "2d":
            return self.get_keypoint_2d(index)
        elif dimension == "3d":
            return self.get_keypoint_3d(index)
        else:
            raise ValueError(f"Invalid dimension: {dimension}. Use '2d' or '3d'")
    
    def get_body_part(self, part: str, dimension: str = "2d"):
        """Alias for get_keypoint_by_name for more natural API."""
        return self.get_keypoint_by_name(part, dimension)
    
    def to_dict(self) -> dict:
        result = {
            "keypoints_2d": [kp.to_dict() for kp in self.keypoints_2d],
        }
        if self.keypoints_3d is not None:
            result["keypoints_3d"] = [kp.to_dict() for kp in self.keypoints_3d]
        if self.person_id is not None:
            result["person_id"] = self.person_id
        return result


@dataclass
class FramePose:
    """Pose data for all people detected in a single frame."""
    frame_number: int
    people: List[PersonPose]
    timestamp: Optional[float] = None  # Timestamp in seconds
    
    @property
    def num_people(self) -> int:
        return len(self.people)
    
    def get_person(self, person_index: int) -> PersonPose:
        """Get pose data for a specific person in this frame."""
        return self.people[person_index]
    
    def to_dict(self) -> dict:
        result = {
            "frame_number": self.frame_number,
            "people": [person.to_dict() for person in self.people],
        }
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        return result


@dataclass
class PoseSequence:
    """Complete pose estimation sequence for a video."""
    frames: List[FramePose]
    fps: Optional[float] = None
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    model_name: Optional[str] = None  # Track which model generated this data
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds if fps is available."""
        if self.fps is not None:
            return self.num_frames / self.fps
        return None
    
    def get_frame(self, frame_number: int) -> FramePose:
        """Get pose data for a specific frame."""
        return self.frames[frame_number]
    
    def get_person_trajectory(self, person_index: int = 0) -> List[PersonPose]:
        """Get the trajectory of a specific person across all frames."""
        return [frame.people[person_index] for frame in self.frames 
                if person_index < len(frame.people)]
    
    def to_dict(self) -> dict:
        result = {
            "frames": [frame.to_dict() for frame in self.frames],
            "num_frames": self.num_frames,
        }
        if self.fps is not None:
            result["fps"] = self.fps
            result["duration"] = self.duration
        if self.video_width is not None:
            result["video_width"] = self.video_width
        if self.video_height is not None:
            result["video_height"] = self.video_height
        if self.model_name is not None:
            result["model_name"] = self.model_name
        return result
    
    def to_json(self, filepath: str, indent: int = 2):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)
    
    @classmethod
    def from_dict(cls, data: dict) -> "PoseSequence":
        """Load from dictionary."""
        frames = []
        for frame_data in data["frames"]:
            people = []
            for person_data in frame_data["people"]:
                keypoints_2d = [
                    Keypoint2D(**kp) for kp in person_data["keypoints_2d"]
                ]
                keypoints_3d = None
                if "keypoints_3d" in person_data:
                    keypoints_3d = [
                        Keypoint3D(**kp) for kp in person_data["keypoints_3d"]
                    ]
                people.append(PersonPose(
                    keypoints_2d=keypoints_2d,
                    keypoints_3d=keypoints_3d,
                    person_id=person_data.get("person_id")
                ))
            frames.append(FramePose(
                frame_number=frame_data["frame_number"],
                people=people,
                timestamp=frame_data.get("timestamp")
            ))
        
        return cls(
            frames=frames,
            fps=data.get("fps"),
            video_width=data.get("video_width"),
            video_height=data.get("video_height"),
            model_name=data.get("model_name")
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> "PoseSequence":
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
