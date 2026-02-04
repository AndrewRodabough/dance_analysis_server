"""
Mappers to convert raw model outputs into standard PoseSequence format.
Each mapper handles a specific model's output format (MediaPipe, MMPose, etc.)
"""
import json
from typing import List, Tuple

from .pose_data import FramePose, Keypoint2D, Keypoint3D, PersonPose, PoseSequence


class MediaPipeMapper:
    """
    Mapper for MediaPipe pose estimation output.

    Current format from pose_estimation.py:
    - keypoints_2d_list: List[List[List[List[float]]]]
      Shape: [num_frames, num_people, num_keypoints, 2]  # [x, y]
    - keypoints_3d_list: List[List[List[List[float]]]]
      Shape: [num_frames, num_people, num_keypoints, 3]  # [x, y, z]
    """

    @staticmethod
    def from_raw_arrays(
        keypoints_2d_list: List,
        keypoints_3d_list: List,
        fps: float = None,
        video_width: int = None,
        video_height: int = None
    ) -> PoseSequence:
        """
        Convert raw MediaPipe output arrays to PoseSequence.
        
        Handles inconsistent data structure where the 'people' dimension may be
        collapsed when only one person is detected:
        - Multiple people: [frames][people][keypoints][coords]
        - Single person: [frames][keypoints][coords] (people dimension missing)
        
        Args:
            keypoints_2d_list: Raw 2D keypoints
            keypoints_3d_list: Raw 3D keypoints
            fps: Video frame rate
            video_width: Video width in pixels
            video_height: Video height in pixels
        
        Returns:
            PoseSequence with structured data
        """
        frames = []
        
        for frame_idx, (frame_2d, frame_3d) in enumerate(zip(keypoints_2d_list, keypoints_3d_list)):
            people = []
            
            # Detect structure: check if this frame has multiple people or single person
            # If frame_2d[0] is a list of 2 numbers [x, y], then structure is [keypoints][coords]
            # If frame_2d[0] is a list of lists, then structure is [people][keypoints][coords]
            
            is_single_person = (
                len(frame_2d) > 0 and 
                isinstance(frame_2d[0], list) and 
                len(frame_2d[0]) == 2 and
                isinstance(frame_2d[0][0], (int, float))  # Check if it's a coordinate, not another list
            )
            
            if is_single_person:
                # Structure: [keypoints][coords] - single person detected
                # Wrap in a list to normalize to [people][keypoints][coords]
                person_2d = frame_2d
                person_3d = frame_3d
                
                keypoints_2d = [
                    Keypoint2D(x=kp[0], y=kp[1])
                    for kp in person_2d
                ]
                
                keypoints_3d = [
                    Keypoint3D(x=kp[0], y=kp[1], z=kp[2])
                    for kp in person_3d
                ]
                
                people.append(PersonPose(
                    keypoints_2d=keypoints_2d,
                    keypoints_3d=keypoints_3d,
                    person_id=0
                ))
            else:
                # Structure: [people][keypoints][coords] - multiple people detected
                for person_idx, (person_2d, person_3d) in enumerate(zip(frame_2d, frame_3d)):
                    keypoints_2d = [
                        Keypoint2D(x=kp[0], y=kp[1])
                        for kp in person_2d
                    ]
                    
                    keypoints_3d = [
                        Keypoint3D(x=kp[0], y=kp[1], z=kp[2])
                        for kp in person_3d
                    ]
                    
                    people.append(PersonPose(
                        keypoints_2d=keypoints_2d,
                        keypoints_3d=keypoints_3d,
                        person_id=person_idx
                    ))

            timestamp = frame_idx / fps if fps else None
            frames.append(FramePose(
                frame_number=frame_idx,
                people=people,
                timestamp=timestamp
            ))

        return PoseSequence(
            frames=frames,
            fps=fps,
            video_width=video_width,
            video_height=video_height,
            model_name="MediaPipe"
        )

    @staticmethod
    def from_json_files(
        json_2d_path: str,
        json_3d_path: str,
        fps: float | None = None,
        video_width: int | None= None,
        video_height: int | None = None
    ) -> PoseSequence:
        """
        Load MediaPipe results from JSON files and convert to PoseSequence.

        Args:
            json_2d_path: Path to 2D keypoints JSON file
            json_3d_path: Path to 3D keypoints JSON file
            fps: Video frame rate
            video_width: Video width in pixels
            video_height: Video height in pixels

        Returns:
            PoseSequence with structured data
        """
        with open(json_2d_path, 'r') as f:
            keypoints_2d_list = json.load(f)

        with open(json_3d_path, 'r') as f:
            keypoints_3d_list = json.load(f)

        return MediaPipeMapper.from_raw_arrays(
            keypoints_2d_list,
            keypoints_3d_list,
            fps=fps,
            video_width=video_width,
            video_height=video_height
        )


class MMPoseMapper:
    """
    Mapper for MMPose output format.
    To be implemented when MMPose integration is added.
    """

    @staticmethod
    def from_raw_output(mmpose_output: dict) -> PoseSequence:
        """
        Convert MMPose output to PoseSequence.

        Args:
            mmpose_output: Raw MMPose detection results

        Returns:
            PoseSequence with structured data
        """
        # TODO: Implement based on actual MMPose output format
        raise NotImplementedError("MMPose mapper not yet implemented")


def load_pose_data(
    json_2d_path: str,
    json_3d_path: str | None = None,
    model_type: str = "mediapipe",
    **kwargs
) -> PoseSequence:
    """
    Universal loader that automatically uses the correct mapper.

    Args:
        json_2d_path: Path to 2D keypoints file
        json_3d_path: Path to 3D keypoints file (optional)
        model_type: Type of model used ("mediapipe", "mmpose", etc.)
        **kwargs: Additional metadata (fps, video_width, video_height, etc.)

    Returns:
        PoseSequence with structured data

    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()

    if model_type == "mediapipe":
        if json_3d_path is None:
            raise ValueError("MediaPipe requires both 2D and 3D keypoint files")
        return MediaPipeMapper.from_json_files(
            json_2d_path,
            json_3d_path,
            **kwargs
        )
    elif model_type == "mmpose":
        return MMPoseMapper.from_raw_output(json_2d_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
