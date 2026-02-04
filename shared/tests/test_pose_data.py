"""Unit tests for VectorizedPoseData class."""
import pytest
import numpy as np
from shared.skeletons.skeleton import VectorizedSkeleton
from shared.skeletons.pose_data import VectorizedPoseData


class TestPoseDataInitialization:
    """Test VectorizedPoseData initialization and validation."""

    def test_init_with_valid_skeleton_and_confidence(self, simple_skeleton_with_data):
        """Test initialization with valid skeleton and confidence scores."""
        confidence = np.ones((10, 3), dtype=np.float32) * 0.9
        pose_data = VectorizedPoseData(simple_skeleton_with_data, confidence)
        
        assert pose_data.num_frames == 10
        assert pose_data.num_joints == 3
        assert pose_data.skeleton is simple_skeleton_with_data

    def test_init_type_checking_skeleton(self, simple_skeleton_with_data):
        """Test type checking for skeleton parameter."""
        confidence = np.ones((10, 3), dtype=np.float32)
        
        with pytest.raises(TypeError, match="VectorizedSkeleton"):
            VectorizedPoseData("not_a_skeleton", confidence)

    def test_init_error_when_skeleton_has_no_data(self, simple_skeleton):
        """Test error when skeleton has no loaded data."""
        confidence = np.ones((10, 3), dtype=np.float32)
        
        with pytest.raises(ValueError, match="must have loaded data"):
            VectorizedPoseData(simple_skeleton, confidence)

    def test_init_confidence_shape_validation(self, simple_skeleton_with_data):
        """Test validation of confidence shape."""
        # Wrong shape
        confidence = np.ones((5, 3), dtype=np.float32)  # Wrong frame count
        
        with pytest.raises(ValueError, match="doesn't match"):
            VectorizedPoseData(simple_skeleton_with_data, confidence)

    def test_init_confidence_dtype_conversion(self, simple_skeleton_with_data):
        """Test that confidence is converted to float32."""
        confidence = np.ones((10, 3), dtype=np.int32)
        pose_data = VectorizedPoseData(simple_skeleton_with_data, confidence)
        
        assert pose_data.confidence.dtype == np.float32

    def test_properties_num_frames_and_joints(self, pose_data_with_confidence):
        """Test num_frames and num_joints properties."""
        assert pose_data_with_confidence.num_frames == 10
        assert pose_data_with_confidence.num_joints == 3


class TestConfidenceFiltering:
    """Test confidence-based filtering and masking."""

    def test_get_high_confidence_mask_basic(self, pose_data_with_confidence):
        """Test getting high confidence mask."""
        mask = pose_data_with_confidence.get_high_confidence_mask(threshold=0.8)
        
        assert mask.shape == (10, 3)
        assert mask.dtype == bool
        assert np.any(mask)  # Some values should be True

    def test_get_high_confidence_mask_various_thresholds(self, pose_data_with_confidence):
        """Test mask with various thresholds."""
        mask_high = pose_data_with_confidence.get_high_confidence_mask(threshold=0.9)
        mask_low = pose_data_with_confidence.get_high_confidence_mask(threshold=0.5)
        
        # Lower threshold should have more True values
        assert np.sum(mask_low) >= np.sum(mask_high)

    def test_get_weighted_bone_lengths(self, pose_data_with_confidence):
        """Test getting weighted bone lengths with masking."""
        lengths = pose_data_with_confidence.get_weighted_bone_lengths(threshold=0.7)
        
        assert lengths.shape[0] == 10  # 10 frames
        # Some values should be NaN due to low confidence
        assert np.any(np.isnan(lengths)) or np.all(~np.isnan(lengths))

    def test_weighted_bone_lengths_low_confidence_masked(self):
        """Test that low-confidence bone endpoints are masked as NaN."""
        joint_names = ["a", "b"]
        bones = [("a", "b")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        data = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.float32)
        skel.load_data(data)
        
        # Low confidence for child
        confidence = np.array([[0.9, 0.3]], dtype=np.float32)
        pose_data = VectorizedPoseData(skel, confidence)
        
        lengths = pose_data.get_weighted_bone_lengths(threshold=0.5)
        # Should be NaN because child has low confidence
        assert np.isnan(lengths[0, 0])

    def test_get_weighted_bone_angles(self, pose_data_with_confidence):
        """Test getting weighted bone angles with masking."""
        angles = pose_data_with_confidence.get_weighted_bone_angles(threshold=0.7)
        
        # Should have same number of frames as input
        assert angles.shape[0] == 10

    def test_weighted_angles_all_joints_required_above_threshold(self):
        """Test that all three joints must be above threshold."""
        joint_names = ["a", "b", "c"]
        bones = [("a", "b"), ("b", "c")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        data = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0]]], dtype=np.float32)
        skel.load_data(data)
        
        # Middle joint (pivot) has low confidence
        confidence = np.array([[0.9, 0.3, 0.9]], dtype=np.float32)
        pose_data = VectorizedPoseData(skel, confidence)
        
        angles = pose_data.get_weighted_bone_angles(threshold=0.5)
        # Should be NaN because pivot has low confidence
        assert np.isnan(angles[0, 0])


class TestJointQueries:
    """Test joint-specific queries."""

    def test_get_joint_confidence_specific_joint(self, pose_data_with_confidence):
        """Test getting confidence for a specific joint."""
        conf = pose_data_with_confidence.get_joint_confidence("parent")
        
        assert conf.shape == (10,)
        assert np.all((conf >= 0) & (conf <= 1))

    def test_get_joint_confidence_invalid_joint(self, pose_data_with_confidence):
        """Test error for non-existent joint."""
        with pytest.raises(ValueError, match="not found"):
            pose_data_with_confidence.get_joint_confidence("nonexistent")

    def test_get_average_confidence(self, pose_data_with_confidence):
        """Test getting average confidence across all data."""
        avg_conf = pose_data_with_confidence.get_average_confidence()
        
        assert 0 <= avg_conf <= 1
        # Should match manual calculation
        expected = np.mean(pose_data_with_confidence.confidence)
        np.testing.assert_almost_equal(avg_conf, expected)

    def test_get_frame_confidence(self, pose_data_with_confidence):
        """Test getting average confidence per frame."""
        frame_conf = pose_data_with_confidence.get_frame_confidence()
        
        assert frame_conf.shape == (10,)
        assert np.all((frame_conf >= 0) & (frame_conf <= 1))

    def test_get_frame_confidence_values(self, pose_data_with_confidence):
        """Test that frame confidence matches manual calculation."""
        frame_conf = pose_data_with_confidence.get_frame_confidence()
        
        expected = np.mean(pose_data_with_confidence.confidence, axis=1)
        np.testing.assert_array_almost_equal(frame_conf, expected)


class TestFrameFiltering:
    """Test frame filtering based on confidence."""

    def test_filter_low_confidence_frames(self, pose_data_with_confidence):
        """Test filtering frames by confidence threshold."""
        filtered = pose_data_with_confidence.filter_low_confidence_frames(threshold=0.8)
        
        assert isinstance(filtered, VectorizedPoseData)
        assert filtered.num_frames <= pose_data_with_confidence.num_frames

    def test_filter_returns_new_instance(self, pose_data_with_confidence):
        """Test that filtering returns new instance."""
        filtered = pose_data_with_confidence.filter_low_confidence_frames(threshold=0.8)
        
        # Should be different objects
        assert filtered is not pose_data_with_confidence

    def test_filter_reduces_frame_count(self):
        """Test that filtering reduces frame count appropriately."""
        joint_names = ["a", "b"]
        bones = [("a", "b")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        # 5 frames
        data = np.ones((5, 2, 3), dtype=np.float32)
        skel.load_data(data)
        
        # Varying confidence: [0.9, 0.5, 0.9, 0.5, 0.9]
        confidence = np.array([
            [0.9, 0.9],
            [0.4, 0.4],
            [0.95, 0.95],
            [0.3, 0.3],
            [0.9, 0.9]
        ], dtype=np.float32)
        pose_data = VectorizedPoseData(skel, confidence)
        
        filtered = pose_data.filter_low_confidence_frames(threshold=0.8)
        # Should keep only frames 0, 2, 4 (3 frames)
        assert filtered.num_frames == 3

    def test_filter_preserves_confidence_values(self):
        """Test that filtering preserves confidence values in kept frames."""
        joint_names = ["a", "b"]
        bones = [("a", "b")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        data = np.ones((3, 2, 3), dtype=np.float32)
        skel.load_data(data)
        
        confidence = np.array([
            [0.9, 0.8],
            [0.4, 0.3],
            [0.95, 0.87]
        ], dtype=np.float32)
        pose_data = VectorizedPoseData(skel, confidence)
        
        filtered = pose_data.filter_low_confidence_frames(threshold=0.8)
        
        # Kept frames should be 0 and 2
        np.testing.assert_array_almost_equal(
            filtered.confidence[0],
            confidence[0]
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zero_confidence(self, simple_skeleton_with_data):
        """Test with all zero confidence scores."""
        confidence = np.zeros((10, 3), dtype=np.float32)
        pose_data = VectorizedPoseData(simple_skeleton_with_data, confidence)
        
        mask = pose_data.get_high_confidence_mask(threshold=0.5)
        assert not np.any(mask)

    def test_all_max_confidence(self, simple_skeleton_with_data):
        """Test with all maximum confidence scores."""
        confidence = np.ones((10, 3), dtype=np.float32)
        pose_data = VectorizedPoseData(simple_skeleton_with_data, confidence)
        
        mask = pose_data.get_high_confidence_mask(threshold=0.5)
        assert np.all(mask)

    def test_single_frame_pose_data(self, simple_skeleton):
        """Test with single frame."""
        data = np.ones((1, 3, 3), dtype=np.float32)
        simple_skeleton.load_data(data)
        
        confidence = np.array([[0.9, 0.8, 0.85]], dtype=np.float32)
        pose_data = VectorizedPoseData(simple_skeleton, confidence)
        
        assert pose_data.num_frames == 1
        frame_conf = pose_data.get_frame_confidence()
        assert frame_conf.shape == (1,)

    def test_single_joint_pose_data(self, isolated_joints_skeleton):
        """Test with single joint."""
        data = np.ones((5, 3, 3), dtype=np.float32)
        isolated_joints_skeleton.load_data(data)
        
        confidence = np.array([[0.9, 0.8, 0.85], [0.8, 0.7, 0.9], [0.85, 0.9, 0.8], [0.7, 0.8, 0.75], [0.95, 0.92, 0.88]], dtype=np.float32)
        pose_data = VectorizedPoseData(isolated_joints_skeleton, confidence)
        
        assert pose_data.num_joints == 3
        conf = pose_data.get_joint_confidence("joint1")
        assert conf.shape == (5,)

    def test_boundary_confidence_values(self, simple_skeleton_with_data):
        """Test with boundary confidence values (0 and 1)."""
        confidence = np.array([
            [0.0, 0.5, 1.0],
            [1.0, 0.0, 0.5],
            [0.5, 1.0, 0.0],
        ] + [[0.5, 0.5, 0.5]] * 7, dtype=np.float32)
        pose_data = VectorizedPoseData(simple_skeleton_with_data, confidence)
        
        # Should handle without errors
        mask = pose_data.get_high_confidence_mask(threshold=0.5)
        assert mask.shape == (10, 3)

    def test_nan_handling_in_calculations(self):
        """Test that NaN values in confidence are handled correctly."""
        joint_names = ["a", "b"]
        bones = [("a", "b")]
        skel = VectorizedSkeleton(joint_names, bones)
        
        data = np.ones((2, 2, 3), dtype=np.float32)
        skel.load_data(data)
        
        # NaN in confidence
        confidence = np.array([[np.nan, 0.9], [0.8, 0.7]], dtype=np.float32)
        pose_data = VectorizedPoseData(skel, confidence)
        
        # Should handle NaN in filtering
        mask = pose_data.get_high_confidence_mask(threshold=0.5)
        # NaN comparisons are False
        assert mask[0, 0] == False


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_workflow_load_filter_query(self, complex_skeleton_with_data):
        """Test end-to-end workflow: load -> filter -> query."""
        confidence = np.random.uniform(0.5, 1.0, size=(5, 5)).astype(np.float32)
        pose_data = VectorizedPoseData(complex_skeleton_with_data, confidence)
        
        # Filter high-confidence frames
        filtered = pose_data.filter_low_confidence_frames(threshold=0.8)
        
        # Query average confidence
        avg_conf = filtered.get_average_confidence()
        assert 0 <= avg_conf <= 1

    def test_consecutive_filters(self, simple_skeleton_with_data):
        """Test applying multiple filters consecutively."""
        confidence = np.random.uniform(0, 1, size=(10, 3)).astype(np.float32)
        pose_data = VectorizedPoseData(simple_skeleton_with_data, confidence)
        
        # First filter
        filtered1 = pose_data.filter_low_confidence_frames(threshold=0.7)
        # Second filter
        filtered2 = filtered1.filter_low_confidence_frames(threshold=0.8)
        
        # Should have fewer frames each time
        assert filtered2.num_frames <= filtered1.num_frames
