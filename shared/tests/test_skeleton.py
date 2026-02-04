"""Unit tests for VectorizedSkeleton class."""
import pytest
import numpy as np
from shared.skeletons.skeleton import VectorizedSkeleton


class TestSkeletonInitialization:
    """Test skeleton initialization and topology setup."""

    def test_init_with_valid_joints_and_bones(self):
        """Test initialization with valid joint names and bones."""
        joint_names = ["a", "b", "c"]
        bones = [("a", "b"), ("b", "c")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        assert skel.num_joints == 3
        assert skel.name_to_idx == {"a": 0, "b": 1, "c": 2}
        assert skel.idx_to_name == ["a", "b", "c"]

    def test_init_with_invalid_bone_references(self):
        """Test that invalid bone references are skipped."""
        joint_names = ["a", "b", "c"]
        bones = [("a", "b"), ("b", "nonexistent"), ("a", "c")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        # Should only have valid bones
        assert skel.bones_index.shape[1] == 2  # Only 2 valid bones

    def test_init_with_no_bones(self):
        """Test skeleton with no bones (isolated joints)."""
        joint_names = ["a", "b", "c"]
        bones = []
        skel = VectorizedSkeleton(joint_names, bones)
        
        assert skel.num_joints == 3
        assert skel.bones_index.shape == (2, 0)
        assert all(len(neighbors) == 0 for neighbors in skel.bones)

    def test_adjacency_list_generation_bidirectional(self):
        """Test that adjacency list is bidirectional."""
        joint_names = ["a", "b", "c"]
        bones = [("a", "b"), ("b", "c")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        # Check bidirectionality
        assert 1 in skel.bones[0]  # a -> b
        assert 0 in skel.bones[1]  # b -> a
        assert 2 in skel.bones[1]  # b -> c
        assert 1 in skel.bones[2]  # c -> b

    def test_bones_index_shape(self):
        """Test that bones_index has correct shape."""
        joint_names = ["a", "b", "c", "d"]
        bones = [("a", "b"), ("b", "c"), ("c", "d")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        assert skel.bones_index.shape == (2, 3)
        assert skel.bones_index.dtype == np.int32

    def test_joints_index_generation(self):
        """Test that angle indices are generated correctly for joints with 2+ neighbors."""
        joint_names = ["root", "left", "right"]
        bones = [("root", "left"), ("root", "right")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        # Root has 2 neighbors, should have 1 angle (left-root-right)
        assert skel.joints_index.shape[1] == 1
        assert skel.joints_index.shape[0] == 3

    def test_single_joint_skeleton(self):
        """Test skeleton with a single joint."""
        joint_names = ["alone"]
        bones = []
        skel = VectorizedSkeleton(joint_names, bones)
        
        assert skel.num_joints == 1
        assert skel.bones_index.shape == (2, 0)
        assert skel.joints_index.shape == (3, 0)

    def test_data_initialized_as_none(self):
        """Test that data is None before loading."""
        joint_names = ["a", "b"]
        bones = [("a", "b")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        assert skel.data is None
        assert skel.num_frames == 0


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_valid_data(self, simple_skeleton):
        """Test loading valid motion data."""
        data = np.ones((10, 3, 3), dtype=np.float32)
        simple_skeleton.load_data(data)
        
        assert simple_skeleton.num_frames == 10
        assert simple_skeleton.data.shape == (10, 3, 3)
        assert simple_skeleton.data.dtype == np.float32

    def test_load_data_mismatched_joint_count(self, simple_skeleton):
        """Test that loading with wrong joint count raises ValueError."""
        data = np.ones((10, 5, 3), dtype=np.float32)  # Wrong joint count
        
        with pytest.raises(ValueError, match="expects 3 joints"):
            simple_skeleton.load_data(data)

    def test_data_dtype_conversion_to_float32(self, simple_skeleton):
        """Test that data is converted to float32."""
        data = np.ones((10, 3, 3), dtype=np.int32)
        simple_skeleton.load_data(data)
        
        assert simple_skeleton.data.dtype == np.float32

    def test_data_contiguity_after_loading(self, simple_skeleton):
        """Test that loaded data is contiguous in memory."""
        data = np.ones((10, 3, 3), dtype=np.float32)
        simple_skeleton.load_data(data)
        
        assert simple_skeleton.data.flags['C_CONTIGUOUS']


class TestBoneLengthCalculation:
    """Test bone length calculations."""

    def test_get_bone_lengths_constant_positions(self, simple_skeleton_with_data):
        """Test bone lengths with constant positions."""
        lengths = simple_skeleton_with_data.get_bone_lengths()
        
        # All joints constant, so all bone lengths should be sqrt(3)
        assert lengths.shape == (10, 2)  # 10 frames, 2 bones
        expected_length = np.sqrt(3)
        np.testing.assert_array_almost_equal(lengths, expected_length)

    def test_get_bone_length_specific_bone(self, simple_skeleton_with_data):
        """Test getting length of a specific bone."""
        length = simple_skeleton_with_data.get_bone_length("parent", "mid")
        
        assert length.shape == (10,)
        expected_length = np.sqrt(3)
        np.testing.assert_array_almost_equal(length, expected_length)

    def test_get_bone_length_invalid_joint_names(self, simple_skeleton_with_data):
        """Test error with invalid joint names."""
        with pytest.raises(ValueError):
            simple_skeleton_with_data.get_bone_length("nonexistent", "parent")

    def test_bone_length_zero_vector(self):
        """Test bone length with coincident joints."""
        joint_names = ["a", "b"]
        bones = [("a", "b")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        # Both joints at same position
        data = np.array([[[0, 0, 0], [0, 0, 0]]], dtype=np.float32)
        skel.load_data(data)
        
        length = skel.get_bone_length("a", "b")
        assert length[0] == 0

    def test_get_bone_lengths_error_without_data(self, simple_skeleton):
        """Test that calling without data raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No data loaded"):
            simple_skeleton.get_bone_lengths()

    def test_get_bone_length_variable_positions(self, complex_skeleton_with_data):
        """Test bone lengths with variable positions."""
        lengths = complex_skeleton_with_data.get_bone_lengths()
        
        # Lengths should vary across frames
        assert lengths.shape[0] == 5  # 5 frames
        assert not np.allclose(lengths[0], lengths[1])  # Should differ


class TestAngleCalculation:
    """Test angle calculations."""

    def test_get_bone_angles_shape(self, complex_skeleton_with_data):
        """Test that bone angles have correct shape."""
        angles = complex_skeleton_with_data.get_bone_angles()
        
        # Complex skeleton has one angle at root (left-root-right)
        assert angles.shape[0] == 5  # 5 frames
        assert angles.shape[1] >= 1  # At least 1 angle

    def test_get_angle_specific_triplet(self, simple_skeleton_with_data):
        """Test getting angle for specific joint triplet."""
        angle = simple_skeleton_with_data.get_angle("parent", "mid", "child")
        
        assert angle.shape == (10,)
        assert np.all(angle >= 0)  # Angles should be non-negative
        assert np.all(angle <= np.pi)  # Angles should be <= pi

    def test_angle_at_collinear_points(self):
        """Test angle calculation at collinear points."""
        joint_names = ["a", "b", "c"]
        bones = [("a", "b"), ("b", "c")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        # Collinear points: (0,0,0), (1,0,0), (2,0,0)
        data = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=np.float32)
        skel.load_data(data)
        
        angle = skel.get_angle("a", "b", "c")
        # Should be close to π (collinear)
        np.testing.assert_almost_equal(angle[0], np.pi, decimal=5)

    def test_angle_error_without_data(self, simple_skeleton):
        """Test that calling without data raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No data loaded"):
            simple_skeleton.get_angle("parent", "mid", "child")

    def test_angle_invalid_joint_names(self, simple_skeleton_with_data):
        """Test error with invalid joint names in angle query."""
        with pytest.raises(ValueError):
            simple_skeleton_with_data.get_angle("nonexistent", "mid", "child")

    def test_angle_clamping_to_avoid_numerical_errors(self):
        """Test that dot products are clamped to [-1, 1]."""
        joint_names = ["a", "b", "c"]
        bones = [("a", "b"), ("b", "c")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        # Create data that might cause numerical issues
        data = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]]] * 5, dtype=np.float32)
        skel.load_data(data)
        
        angle = skel.get_angle("a", "b", "c")
        # Should not raise or produce NaN
        assert not np.any(np.isnan(angle))
        assert np.all(np.isfinite(angle))

    def test_right_angle_detection(self):
        """Test detection of 90-degree angles."""
        joint_names = ["a", "b", "c"]
        bones = [("a", "b"), ("b", "c")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        # Right angle: (0,0,0), (1,0,0), (1,1,0)
        data = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0]]], dtype=np.float32)
        skel.load_data(data)
        
        angle = skel.get_angle("a", "b", "c")
        # Should be close to π/2
        np.testing.assert_almost_equal(angle[0], np.pi / 2, decimal=5)

    def test_zero_vector_handling(self):
        """Test angle calculation with zero vectors."""
        joint_names = ["a", "b", "c"]
        bones = [("a", "b"), ("b", "c")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        # Zero vectors: pivot and all others at same point
        data = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.float32)
        skel.load_data(data)
        
        angle = skel.get_angle("a", "b", "c")
        # Should handle gracefully (return 0 or similar)
        assert not np.any(np.isnan(angle))


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_frame(self, simple_skeleton):
        """Test with single frame of data."""
        data = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]]], dtype=np.float32)
        simple_skeleton.load_data(data)
        
        lengths = simple_skeleton.get_bone_lengths()
        assert lengths.shape == (1, 2)

    def test_skeleton_with_cycles(self):
        """Test topology with cycles (should still work)."""
        joint_names = ["a", "b", "c"]
        bones = [("a", "b"), ("b", "c"), ("c", "a")]  # Cycle
        skel = VectorizedSkeleton(joint_names, bones)
        
        assert skel.num_joints == 3
        # All bones should be registered
        assert skel.bones_index.shape[1] == 3

    def test_large_skeleton(self):
        """Test with a larger skeleton."""
        joint_names = [f"joint_{i}" for i in range(50)]
        bones = [(f"joint_{i}", f"joint_{i+1}") for i in range(49)]
        skel = VectorizedSkeleton(joint_names, bones)
        
        assert skel.num_joints == 50
        # Load data
        data = np.random.randn(10, 50, 3).astype(np.float32)
        skel.load_data(data)
        
        lengths = skel.get_bone_lengths()
        assert lengths.shape == (10, 49)

    def test_high_dimensional_channels(self):
        """Test with more than 3 channels."""
        joint_names = ["a", "b"]
        bones = [("a", "b")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        # 5 channels instead of 3
        data = np.ones((5, 2, 5), dtype=np.float32)
        skel.load_data(data)
        
        length = skel.get_bone_length("a", "b")
        assert length.shape == (5,)
