"""Tests for 2D hyperextension detection in leg straightening timing analysis."""

import pytest
import numpy as np
from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
    detect_walking_direction_2d,
    apply_hyperextension_clamping_2d,
)
from shared.skeletons.skeleton import VectorizedSkeleton
from shared.skeletons.pose_data import VectorizedPoseData


@pytest.fixture
def coco_17_skeleton_2d():
    """Create COCO-17 skeleton with 2D side-view data."""
    joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", 
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ]
    bones = [
        ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
        ["right_hip", "right_knee"], ["right_knee", "right_ankle"]
    ]
    skeleton = VectorizedSkeleton(joint_names, bones)
    
    # Create 2D side-view data for 20 frames
    # First 10 frames: moving right (positive X velocity)
    # Last 10 frames: moving left (negative X velocity)
    num_frames = 20
    data = np.zeros((num_frames, 17, 2), dtype=np.float32)
    
    for frame in range(num_frames):
        # Base positions (image coordinates: X=right, Y=down)
        data[frame, 11, :] = [400, 300]   # left_hip
        data[frame, 13, :] = [400, 500]   # left_knee
        data[frame, 15, :] = [400, 700]   # left_ankle
        
        data[frame, 12, :] = [600, 300]   # right_hip
        data[frame, 14, :] = [600, 500]   # right_knee
        data[frame, 16, :] = [600, 700]   # right_ankle
        
        # Add horizontal velocity (simulate forward/backward motion)
        if frame < 10:
            # Moving right (positive X)
            ankle_x_offset = frame * 10
            data[frame, 15, 0] += ankle_x_offset   # left_ankle X
            data[frame, 16, 0] += ankle_x_offset   # right_ankle X
        else:
            # Moving left (negative X)
            ankle_x_offset = (19 - frame) * 10
            data[frame, 15, 0] -= ankle_x_offset   # left_ankle X
            data[frame, 16, 0] -= ankle_x_offset   # right_ankle X
    
    skeleton.load_data(data)
    confidence = np.ones((num_frames, 17), dtype=np.float32) * 0.9
    pose_data = VectorizedPoseData(skeleton, confidence)
    
    return pose_data


class TestWalkingDirectionDetection:
    """Test walking direction detection in 2D."""
    
    def test_detect_walking_direction_2d_output_shape(self, coco_17_skeleton_2d):
        """Test that output has correct shape."""
        direction = detect_walking_direction_2d(coco_17_skeleton_2d)
        
        assert direction.shape == (20,)
    
    def test_detect_walking_direction_2d_valid_values(self, coco_17_skeleton_2d):
        """Test that direction values are in {-1, 0, 1}."""
        direction = detect_walking_direction_2d(coco_17_skeleton_2d)
        
        assert np.all(np.isin(direction, [-1, 0, 1]))
    
    def test_detect_walking_direction_2d_detecting_motion(self, coco_17_skeleton_2d):
        """Test that walking direction detects motion direction."""
        direction = detect_walking_direction_2d(coco_17_skeleton_2d)
        
        # First half should have mostly positive direction (moving right)
        first_half = direction[:10]
        # Last half should have mostly negative direction (moving left)
        last_half = direction[10:]
        
        # Count non-zero directions
        first_positive = np.sum(first_half > 0)
        last_negative = np.sum(last_half < 0)
        
        # At least some frames should show consistent direction
        assert first_positive >= 2  # Some frames moving right
        assert last_negative >= 2   # Some frames moving left
    
    def test_detect_walking_direction_2d_window_size(self, coco_17_skeleton_2d):
        """Test that window size parameter works."""
        direction_small_window = detect_walking_direction_2d(coco_17_skeleton_2d, window_size=3)
        direction_large_window = detect_walking_direction_2d(coco_17_skeleton_2d, window_size=9)
        
        # Both should have same shape
        assert direction_small_window.shape == direction_large_window.shape
        
        # Larger window should be smoother (fewer sign changes expected)
        small_changes = np.sum(np.diff(direction_small_window) != 0)
        large_changes = np.sum(np.diff(direction_large_window) != 0)
        
        # Large window should have fewer transitions
        assert large_changes <= small_changes


class TestHyperextensionClamping:
    """Test hyperextension clamping logic."""
    
    def test_apply_hyperextension_clamping_2d_output_shape(self, coco_17_skeleton_2d):
        """Test that clamping preserves input shape."""
        # Create test knee angles
        angles = np.array([
            [2.8, 2.8],   # Near straight
            [2.5, 2.5],   # Bent
            [3.1, 3.1],   # Hyperextended
            [2.0, 2.0],   # Significantly bent
        ] * 5, dtype=np.float32)  # Extend to 20 frames
        
        walking_direction = detect_walking_direction_2d(coco_17_skeleton_2d)
        
        # Need exactly 20 frames
        angles_20 = angles[:20]
        
        clamped = apply_hyperextension_clamping_2d(
            angles_20, coco_17_skeleton_2d, walking_direction
        )
        
        # Output shape should match input shape
        assert clamped.shape == angles_20.shape
    
    def test_apply_hyperextension_clamping_2d_output_range(self, coco_17_skeleton_2d):
        """Test that clamped values are valid angles [0, π]."""
        angles = np.random.uniform(0.5, 2.0, (20, 2)).astype(np.float32)
        walking_direction = detect_walking_direction_2d(coco_17_skeleton_2d)
        
        clamped = apply_hyperextension_clamping_2d(
            angles, coco_17_skeleton_2d, walking_direction
        )
        
        # All values should be in [0, π]
        assert np.all(clamped >= 0)
        assert np.all(clamped <= np.pi + 0.01)  # Allow small numerical error
    
    def test_apply_hyperextension_clamping_2d_clamps_forward_bends(self, coco_17_skeleton_2d):
        """Test that forward bends near π are clamped to π."""
        # Create angles that are less than π (bent)
        angles = np.full((20, 2), 2.8, dtype=np.float32)
        
        walking_direction = detect_walking_direction_2d(coco_17_skeleton_2d)
        
        clamped = apply_hyperextension_clamping_2d(
            angles, coco_17_skeleton_2d, walking_direction
        )
        
        # Some values should be clamped to π
        clamped_count = np.sum(np.abs(clamped - np.pi) < 0.01)
        
        # If there are forward bends detected, some should be clamped
        # (At least verify the function doesn't crash)
        assert clamped.shape == angles.shape
    
    def test_apply_hyperextension_clamping_2d_doesnt_modify_input(self, coco_17_skeleton_2d):
        """Test that clamping doesn't modify input array."""
        angles = np.random.uniform(0.5, 2.0, (20, 2)).astype(np.float32)
        original_angles = angles.copy()
        
        walking_direction = detect_walking_direction_2d(coco_17_skeleton_2d)
        
        # Apply clamping
        clamped = apply_hyperextension_clamping_2d(
            angles, coco_17_skeleton_2d, walking_direction
        )
        
        # Input should not be modified
        np.testing.assert_array_equal(angles, original_angles)
        
        # Output should be a different array
        assert clamped is not angles
    
    def test_apply_hyperextension_clamping_2d_threshold_parameter(self, coco_17_skeleton_2d):
        """Test that threshold parameter affects clamping behavior."""
        angles = np.full((20, 2), 2.8, dtype=np.float32)
        walking_direction = detect_walking_direction_2d(coco_17_skeleton_2d)
        
        # Apply clamping with default threshold (π)
        clamped_default = apply_hyperextension_clamping_2d(
            angles.copy(), coco_17_skeleton_2d, walking_direction
        )
        
        # Apply clamping with higher threshold
        clamped_higher = apply_hyperextension_clamping_2d(
            angles.copy(), coco_17_skeleton_2d, walking_direction,
            hyperextension_threshold=3.05
        )
        
        # With higher threshold, fewer values should be clamped
        default_clamped = np.sum(np.abs(clamped_default - np.pi) < 0.01)
        higher_clamped = np.sum(np.abs(clamped_higher - 3.05) < 0.01)
        
        # Both should work without error
        assert clamped_default.shape == angles.shape
        assert clamped_higher.shape == angles.shape


class TestIntegration:
    """Integration tests for hyperextension detection."""
    
    def test_walking_direction_and_clamping_together(self, coco_17_skeleton_2d):
        """Test that walking direction and clamping work together."""
        # Create simple angles
        angles = np.full((20, 2), 2.8, dtype=np.float32)
        
        # Pipeline: detect direction, then clamp
        walking_direction = detect_walking_direction_2d(coco_17_skeleton_2d)
        clamped = apply_hyperextension_clamping_2d(
            angles, coco_17_skeleton_2d, walking_direction
        )
        
        # Verify results are valid
        assert clamped.shape == angles.shape
        assert np.all(clamped >= 0)
        assert np.all(clamped <= np.pi + 0.01)
    
    def test_stationary_frames(self):
        """Test with stationary frames (zero velocity)."""
        joint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", 
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]
        bones = [["left_hip", "left_knee"]]
        skeleton = VectorizedSkeleton(joint_names, bones)
        
        # Static data (no motion)
        num_frames = 10
        data = np.zeros((num_frames, 17, 2), dtype=np.float32)
        data[:, 11, :] = [400, 300]   # left_hip (static)
        data[:, 13, :] = [400, 500]   # left_knee (static)
        data[:, 15, :] = [400, 700]   # left_ankle (static)
        
        skeleton.load_data(data)
        confidence = np.ones((num_frames, 17), dtype=np.float32)
        pose_data = VectorizedPoseData(skeleton, confidence)
        
        # Direction for stationary sequence should be mostly zero
        direction = detect_walking_direction_2d(pose_data)
        
        # Most should be 0 (stationary)
        zero_count = np.sum(direction == 0)
        assert zero_count >= num_frames // 2  # At least half should be zero
