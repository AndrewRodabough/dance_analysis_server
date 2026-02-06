"""Test multi-person support in keypoint loading and pipeline."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Add workspace to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processing.app.analysis.keypoint_loading import (
    normalize_keypoints_format,
    extract_person_data,
    load_keypoints_from_json,
)


class TestNormalizeKeypointsFormat:
    """Test format normalization for single and multi-person data."""
    
    def test_single_person_3d_format(self):
        """Convert 3D (frames, joints, channels) to 4D (frames, people, joints, channels)."""
        # Create single-person data: 10 frames, 133 joints, 3 channels
        data = np.random.rand(10, 133, 3)
        
        normalized, num_people = normalize_keypoints_format(data)
        
        assert normalized.shape == (10, 1, 133, 3), f"Expected (10, 1, 133, 3), got {normalized.shape}"
        assert num_people == 1
    
    def test_multi_person_4d_format(self):
        """4D data (frames, people, joints, channels) stays unchanged."""
        # Create multi-person data: 10 frames, 3 people, 133 joints, 3 channels
        data = np.random.rand(10, 3, 133, 3)
        
        normalized, num_people = normalize_keypoints_format(data)
        
        assert normalized.shape == (10, 3, 133, 3)
        assert num_people == 3
    
    def test_invalid_format(self):
        """Raise error for invalid data dimensions."""
        data = np.random.rand(10, 133)  # 2D data
        
        with pytest.raises(ValueError, match="Invalid keypoints shape"):
            normalize_keypoints_format(data)
    
    def test_single_person_preserves_values(self):
        """Ensure values are preserved when converting format."""
        data = np.array([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # Frame 0: 2 joints, 3 channels each
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],  # Frame 1
        ])
        
        normalized, _ = normalize_keypoints_format(data)
        
        # Check shape
        assert normalized.shape == (2, 1, 2, 3)
        
        # Check values preserved
        np.testing.assert_array_equal(normalized[:, 0, :, :], data)


class TestExtractPersonData:
    """Test extraction of individual person data from multi-person array."""
    
    def test_extract_first_person(self):
        """Extract data for first person."""
        # 5 frames, 3 people, 10 joints, 3 channels
        all_people = np.arange(5 * 3 * 10 * 3).reshape(5, 3, 10, 3)
        
        person_0 = extract_person_data(all_people, 0)
        
        assert person_0.shape == (5, 10, 3)
        np.testing.assert_array_equal(person_0, all_people[:, 0, :, :])
    
    def test_extract_middle_person(self):
        """Extract data for middle person."""
        all_people = np.arange(5 * 3 * 10 * 3).reshape(5, 3, 10, 3)
        
        person_1 = extract_person_data(all_people, 1)
        
        assert person_1.shape == (5, 10, 3)
        np.testing.assert_array_equal(person_1, all_people[:, 1, :, :])
    
    def test_extract_last_person(self):
        """Extract data for last person."""
        all_people = np.arange(5 * 3 * 10 * 3).reshape(5, 3, 10, 3)
        
        person_2 = extract_person_data(all_people, 2)
        
        assert person_2.shape == (5, 10, 3)
        np.testing.assert_array_equal(person_2, all_people[:, 2, :, :])


class TestLoadKeypointsFromJsonMultiPerson:
    """Test loading and normalizing keypoints from JSON with multi-person support."""
    
    def test_load_single_person_json(self):
        """Load single-person keypoints from JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create single-person 2D and 3D data
            kp_2d = np.random.rand(10, 133, 3).tolist()
            kp_3d = np.random.rand(10, 17, 3).tolist()
            
            kp_2d_path = temp_path / "2d.json"
            kp_3d_path = temp_path / "3d.json"
            
            with open(kp_2d_path, 'w') as f:
                json.dump(kp_2d, f)
            with open(kp_3d_path, 'w') as f:
                json.dump(kp_3d, f)
            
            # Load and normalize
            kp_2d_norm, kp_3d_norm, num_people = load_keypoints_from_json(
                kp_2d_path, kp_3d_path
            )
            
            # Should be expanded to 4D with 1 person
            assert kp_2d_norm.shape == (10, 1, 133, 3)
            assert kp_3d_norm.shape == (10, 1, 17, 3)
            assert num_people == 1
    
    def test_load_multi_person_json(self):
        """Load multi-person keypoints from JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multi-person 2D and 3D data
            kp_2d = np.random.rand(10, 3, 133, 3).tolist()  # 3 people
            kp_3d = np.random.rand(10, 3, 17, 3).tolist()   # 3 people
            
            kp_2d_path = temp_path / "2d.json"
            kp_3d_path = temp_path / "3d.json"
            
            with open(kp_2d_path, 'w') as f:
                json.dump(kp_2d, f)
            with open(kp_3d_path, 'w') as f:
                json.dump(kp_3d, f)
            
            # Load and normalize
            kp_2d_norm, kp_3d_norm, num_people = load_keypoints_from_json(
                kp_2d_path, kp_3d_path
            )
            
            # Should remain 4D with 3 people
            assert kp_2d_norm.shape == (10, 3, 133, 3)
            assert kp_3d_norm.shape == (10, 3, 17, 3)
            assert num_people == 3
    
    def test_load_mismatched_person_counts(self):
        """Handle mismatched person counts between 2D and 3D."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mismatched data: 3 people in 2D, 2 in 3D
            kp_2d = np.random.rand(10, 3, 133, 3).tolist()
            kp_3d = np.random.rand(10, 2, 17, 3).tolist()
            
            kp_2d_path = temp_path / "2d.json"
            kp_3d_path = temp_path / "3d.json"
            
            with open(kp_2d_path, 'w') as f:
                json.dump(kp_2d, f)
            with open(kp_3d_path, 'w') as f:
                json.dump(kp_3d, f)
            
            # Load should handle gracefully
            kp_2d_norm, kp_3d_norm, num_people = load_keypoints_from_json(
                kp_2d_path, kp_3d_path
            )
            
            # Should use minimum count
            assert kp_2d_norm.shape[1] == 2  # min(3, 2)
            assert kp_3d_norm.shape[1] == 2
            assert num_people == 2


class TestMultiPersonPipelineSupport:
    """Test that pipeline functions support multi-person data."""
    
    def test_create_skeleton_for_specific_person(self):
        """Create skeleton for a specific person from multi-person data."""
        # This requires actual skeleton configs to be loaded
        # For now, verify the function signature and error handling
        from video_processing.app.analysis.keypoint_loading import create_skeleton_objects
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multi-person test data
            kp_2d = np.random.rand(10, 2, 133, 3)  # 2 people
            kp_3d = np.random.rand(10, 2, 17, 3)
            
            # Should be able to extract person 0
            try:
                skeleton_2d, skeleton_3d = create_skeleton_objects(
                    kp_2d, kp_3d, person_idx=0
                )
                # If no error, we successfully created skeletons for person 0
                assert skeleton_2d.num_frames == 10
                assert skeleton_3d.num_frames == 10
            except FileNotFoundError:
                # Expected if skeleton configs not available in test environment
                pytest.skip("Skeleton configs not available in test environment")
    
    def test_person_index_validation(self):
        """Validate person index is within valid range."""
        from video_processing.app.analysis.keypoint_loading import create_skeleton_objects
        
        kp_2d = np.random.rand(10, 2, 133, 3)  # 2 people
        kp_3d = np.random.rand(10, 2, 17, 3)
        
        # Should fail for out-of-range person index
        with pytest.raises((FileNotFoundError, IndexError)):
            # Index 5 is out of range for 2 people
            create_skeleton_objects(kp_2d, kp_3d, person_idx=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
