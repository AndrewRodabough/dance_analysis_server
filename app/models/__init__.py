"""
Data models for pose estimation results.
"""
from .pose_data import Keypoint2D, Keypoint3D, PersonPose, FramePose, PoseSequence
from .keypoint_schema import (
    COCOWholebodyKeypoint,
    KeypointSchema,
    get_keypoint_name,
    COCO_WHOLEBODY_SKELETON,
)

__all__ = [
    "Keypoint2D",
    "Keypoint3D", 
    "PersonPose",
    "FramePose",
    "PoseSequence",
    "COCOWholebodyKeypoint",
    "KeypointSchema",
    "get_keypoint_name",
    "COCO_WHOLEBODY_SKELETON",
]
