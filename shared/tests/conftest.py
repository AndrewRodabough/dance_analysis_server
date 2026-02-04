"""Shared test fixtures for skeleton module tests."""
import pytest
import numpy as np
from shared.skeletons.skeleton import VectorizedSkeleton
from shared.skeletons.pose_data import VectorizedPoseData


@pytest.fixture
def simple_skeleton():
    """Create a simple 3-joint skeleton: parent -> mid -> child"""
    joint_names = ["parent", "mid", "child"]
    bones = [("parent", "mid"), ("mid", "child")]
    return VectorizedSkeleton(joint_names, bones)


@pytest.fixture
def simple_skeleton_with_data(simple_skeleton):
    """Create a simple skeleton with 10 frames of mock data (3D coordinates)"""
    # 10 frames, 3 joints, 3 channels (x, y, z)
    data = np.array([
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # Frame 0
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # Frame 1
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # Frame 2
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # Frame 3
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # Frame 4
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # Frame 5
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # Frame 6
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # Frame 7
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # Frame 8
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # Frame 9
    ], dtype=np.float32)
    simple_skeleton.load_data(data)
    return simple_skeleton


@pytest.fixture
def complex_skeleton():
    """Create a more complex skeleton (5 joints with multiple branches)"""
    joint_names = ["root", "left_arm", "left_hand", "right_arm", "right_hand"]
    bones = [
        ("root", "left_arm"),
        ("left_arm", "left_hand"),
        ("root", "right_arm"),
        ("right_arm", "right_hand"),
    ]
    return VectorizedSkeleton(joint_names, bones)


@pytest.fixture
def complex_skeleton_with_data(complex_skeleton):
    """Create a complex skeleton with 5 frames of varied motion data"""
    np.random.seed(42)
    data = np.random.randn(5, 5, 3).astype(np.float32)
    complex_skeleton.load_data(data)
    return complex_skeleton


@pytest.fixture
def pose_data_with_confidence(simple_skeleton_with_data):
    """Create VectorizedPoseData with confidence scores"""
    confidence = np.array([
        [0.9, 0.95, 0.85],
        [0.8, 0.9, 0.75],
        [0.95, 0.92, 0.88],
        [0.7, 0.8, 0.65],
        [0.85, 0.88, 0.9],
        [0.9, 0.95, 0.85],
        [0.8, 0.9, 0.75],
        [0.95, 0.92, 0.88],
        [0.7, 0.8, 0.65],
        [0.85, 0.88, 0.9],
    ], dtype=np.float32)
    return VectorizedPoseData(simple_skeleton_with_data, confidence)


@pytest.fixture
def isolated_joints_skeleton():
    """Create a skeleton with no bones (isolated joints)"""
    joint_names = ["joint1", "joint2", "joint3"]
    bones = []
    return VectorizedSkeleton(joint_names, bones)
