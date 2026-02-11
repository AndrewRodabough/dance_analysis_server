"""Tests for feet position extraction."""

import pytest
import numpy as np
from video_processing.app.analysis.feature_extraction.feet_position import (
    extract_feet_position,
    get_feet_position_summary,
    H36M_PELVIS,
    H36M_L_ANKLE,
    H36M_R_ANKLE,
    H36M_L_HIP,
    H36M_R_HIP,
)
from shared.skeletons.skeleton import VectorizedSkeleton
from shared.skeletons.pose_data import VectorizedPoseData


@pytest.fixture
def human_17_skeleton_with_feet_data():
    """Create Human3.6M skeleton with test data where feet are at different forward/back positions."""
    joint_names = [
        "Pelvis", "R_Hip", "R_Knee", "R_Ankle", 
        "L_Hip", "L_Knee", "L_Ankle", 
        "Spine", "Neck", "Head", "Site", 
        "L_Shoulder", "L_Elbow", "L_Wrist", 
        "R_Shoulder", "R_Elbow", "R_Wrist"
    ]
    bones = [
        ["Pelvis", "Spine"], 
        ["Spine", "Neck"], 
        ["Neck", "Head"], 
        ["Head", "Site"],
        ["Pelvis", "L_Hip"], ["L_Hip", "L_Knee"], ["L_Knee", "L_Ankle"],
        ["Pelvis", "R_Hip"], ["R_Hip", "R_Knee"], ["R_Knee", "R_Ankle"],
        ["Neck", "L_Shoulder"], ["L_Shoulder", "L_Elbow"], ["L_Elbow", "L_Wrist"],
        ["Neck", "R_Shoulder"], ["R_Shoulder", "R_Elbow"], ["R_Elbow", "R_Wrist"]
    ]
    
    skeleton = VectorizedSkeleton(joint_names, bones)
    
    # Create test data: 5 frames, 17 joints, 3D coordinates
    data = np.zeros((5, 17, 3), dtype=np.float32)
    
    # Setup basic pose: person facing forward (+Z direction)
    # Hips are at X=-0.5 (left) and X=+0.5 (right), pelvis at center
    hip_y = 1.0  # Hip height
    pelvis_offset_back = -0.2  # Pelvis slightly behind hip line (defines forward direction)
    
    for frame in range(5):
        # Hip positions (form the forward direction reference)
        data[frame, H36M_L_HIP, :] = [-0.5, hip_y, 0]    # Left hip
        data[frame, H36M_R_HIP, :] = [0.5, hip_y, 0]     # Right hip  
        data[frame, H36M_PELVIS, :] = [0, hip_y, pelvis_offset_back]  # Pelvis behind hips
    
    # Frame 0: Both feet in front (positive Z from pelvis)
    data[0, H36M_L_ANKLE, :] = [-0.5, 0, 0.5]  # In front
    data[0, H36M_R_ANKLE, :] = [0.5, 0, 0.5]   # In front
    
    # Frame 1: Both feet behind (negative Z from pelvis)
    data[1, H36M_L_ANKLE, :] = [-0.5, 0, -0.8]  # Behind
    data[1, H36M_R_ANKLE, :] = [0.5, 0, -0.8]   # Behind
    
    # Frame 2: Left in front, right behind
    data[2, H36M_L_ANKLE, :] = [-0.5, 0, 0.5]   # In front
    data[2, H36M_R_ANKLE, :] = [0.5, 0, -0.8]   # Behind
    
    # Frame 3: Right in front, left behind
    data[3, H36M_L_ANKLE, :] = [-0.5, 0, -0.8]  # Behind
    data[3, H36M_R_ANKLE, :] = [0.5, 0, 0.5]    # In front
    
    # Frame 4: Both exactly at pelvis level (edge case)
    data[4, H36M_L_ANKLE, :] = [-0.5, 0, pelvis_offset_back]  # Same Z as pelvis
    data[4, H36M_R_ANKLE, :] = [0.5, 0, pelvis_offset_back]   # Same Z as pelvis
    
    skeleton.load_data(data)
    confidence = np.ones((5, 17), dtype=np.float32) * 0.9
    pose_data = VectorizedPoseData(skeleton, confidence)
    
    return pose_data


class TestFeetPositionExtraction:
    """Test feet position extraction functionality."""
    
    def test_extract_feet_position_shape(self, human_17_skeleton_with_feet_data):
        """Test output shape is correct."""
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        
        feet_pos = result['feet_position']
        assert feet_pos.shape == (5, 2)  # 5 frames, 2 feet (left, right)
    
    def test_both_feet_in_front(self, human_17_skeleton_with_feet_data):
        """Test detection when both feet are in front."""
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        feet_pos = result['feet_position']
        
        # Frame 0: both in front
        assert feet_pos[0, 0] == 1.0  # Left in front
        assert feet_pos[0, 1] == 1.0  # Right in front
    
    def test_both_feet_behind(self, human_17_skeleton_with_feet_data):
        """Test detection when both feet are behind."""
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        feet_pos = result['feet_position']
        
        # Frame 1: both behind
        assert feet_pos[1, 0] == 0.0  # Left behind
        assert feet_pos[1, 1] == 0.0  # Right behind
    
    def test_mixed_feet_positions(self, human_17_skeleton_with_feet_data):
        """Test detection with mixed feet positions."""
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        feet_pos = result['feet_position']
        
        # Frame 2: left in front, right behind
        assert feet_pos[2, 0] == 1.0  # Left in front
        assert feet_pos[2, 1] == 0.0  # Right behind
        
        # Frame 3: right in front, left behind
        assert feet_pos[3, 0] == 0.0  # Left behind
        assert feet_pos[3, 1] == 1.0  # Right in front
    
    def test_feet_equal_to_body(self, human_17_skeleton_with_feet_data):
        """Test edge case where feet are at same depth as body."""
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        feet_pos = result['feet_position']
        
        # Frame 4: both equal to pelvis (not > so should be 0)
        assert feet_pos[4, 0] == 0.0  # Left not in front
        assert feet_pos[4, 1] == 0.0  # Right not in front
    
    def test_debug_arrays_present(self, human_17_skeleton_with_feet_data):
        """Test that debug arrays are returned."""
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        
        assert 'forward_vectors' in result
        assert 'ankle_projections' in result
        assert result['forward_vectors'].shape == (5, 3)     # 5 frames, 3D vectors
        assert result['ankle_projections'].shape == (5, 2)   # 5 frames, 2 ankles
    
    def test_output_dtype(self, human_17_skeleton_with_feet_data):
        """Test output is float32."""
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        feet_pos = result['feet_position']
        
        assert feet_pos.dtype == np.float32
    
    def test_values_binary(self, human_17_skeleton_with_feet_data):
        """Test all values are 0 or 1."""
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        feet_pos = result['feet_position']
        
        assert np.all((feet_pos == 0) | (feet_pos == 1))


class TestFeetPositionSummary:
    """Test summary statistics calculation."""
    
    def test_summary_statistics(self, human_17_skeleton_with_feet_data):
        """Test summary statistics."""
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        summary = get_feet_position_summary(result)
        
        assert 'left_foot_pct_in_front' in summary
        assert 'right_foot_pct_in_front' in summary
        assert 'both_feet_pct_in_front' in summary
        assert 'avg_left_projection' in summary
        assert 'avg_right_projection' in summary
    
    def test_summary_percentages_valid(self, human_17_skeleton_with_feet_data):
        """Test summary percentages are between 0-100."""
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        summary = get_feet_position_summary(result)
        
        # Check percentages are between 0-100
        for key in ['left_foot_pct_in_front', 'right_foot_pct_in_front', 'both_feet_pct_in_front']:
            value = summary[key]
            assert 0 <= value <= 100, f"{key} = {value} is not in [0, 100]"
        
        # Projections can be negative (behind) or positive (in front), so just check they exist
        assert 'avg_left_projection' in summary
        assert 'avg_right_projection' in summary
    
    def test_summary_left_foot_percentage(self, human_17_skeleton_with_feet_data):
        """Test left foot percentage calculation.
        
        Frame 0: in front (1)
        Frame 1: behind (0)
        Frame 2: in front (1)
        Frame 3: behind (0)
        Frame 4: behind (0)
        
        Percentage in front: 2/5 = 40%
        """
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        summary = get_feet_position_summary(result)
        
        assert np.isclose(summary['left_foot_pct_in_front'], 40.0)
    
    def test_summary_right_foot_percentage(self, human_17_skeleton_with_feet_data):
        """Test right foot percentage calculation.
        
        Frame 0: in front (1)
        Frame 1: behind (0)
        Frame 2: behind (0)
        Frame 3: in front (1)
        Frame 4: behind (0)
        
        Percentage in front: 2/5 = 40%
        """
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        summary = get_feet_position_summary(result)
        
        assert np.isclose(summary['right_foot_pct_in_front'], 40.0)
    
    def test_summary_both_feet_percentage(self, human_17_skeleton_with_feet_data):
        """Test both feet in front percentage.
        
        Only Frame 0 has both in front.
        Percentage: 1/5 = 20%
        """
        result = extract_feet_position(human_17_skeleton_with_feet_data)
        summary = get_feet_position_summary(result)
        
        assert np.isclose(summary['both_feet_pct_in_front'], 20.0)


class TestFeetPositionEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_frame(self):
        """Test with single frame."""
        joint_names = [
            "Pelvis", "R_Hip", "R_Knee", "R_Ankle", 
            "L_Hip", "L_Knee", "L_Ankle", 
            "Spine", "Neck", "Head", "Site", 
            "L_Shoulder", "L_Elbow", "L_Wrist", 
            "R_Shoulder", "R_Elbow", "R_Wrist"
        ]
        bones = [["Pelvis", "Spine"]]  # Minimal
        
        skeleton = VectorizedSkeleton(joint_names, bones)
        data = np.zeros((1, 17, 3), dtype=np.float32)
        
        # Setup hip geometry
        data[0, H36M_L_HIP, :] = [-0.5, 1.0, 0]
        data[0, H36M_R_HIP, :] = [0.5, 1.0, 0]
        data[0, H36M_PELVIS, :] = [0, 1.0, -0.2]  # Behind hips
        
        # Ankles: left in front, right behind
        data[0, H36M_L_ANKLE, :] = [-0.5, 0, 0.5]  # In front
        data[0, H36M_R_ANKLE, :] = [0.5, 0, -0.8]  # Behind
        
        skeleton.load_data(data)
        confidence = np.ones((1, 17), dtype=np.float32)
        pose_data = VectorizedPoseData(skeleton, confidence)
        
        result = extract_feet_position(pose_data)
        assert result['feet_position'].shape == (1, 2)
        assert result['feet_position'][0, 0] == 1.0  # Left in front
        assert result['feet_position'][0, 1] == 0.0  # Right behind
    
    def test_many_frames(self):
        """Test with many frames."""
        joint_names = [
            "Pelvis", "R_Hip", "R_Knee", "R_Ankle", 
            "L_Hip", "L_Knee", "L_Ankle", 
            "Spine", "Neck", "Head", "Site", 
            "L_Shoulder", "L_Elbow", "L_Wrist", 
            "R_Shoulder", "R_Elbow", "R_Wrist"
        ]
        bones = [["Pelvis", "Spine"]]
        
        skeleton = VectorizedSkeleton(joint_names, bones)
        num_frames = 100
        data = np.zeros((num_frames, 17, 3), dtype=np.float32)
        
        # Setup consistent hip geometry for all frames
        for i in range(num_frames):
            data[i, H36M_L_HIP, :] = [-0.5, 1.0, 0]
            data[i, H36M_R_HIP, :] = [0.5, 1.0, 0]
            data[i, H36M_PELVIS, :] = [0, 1.0, -0.2]  # Behind hips
            
            # Left: alternates front/back
            left_z = 0.5 if i % 2 == 0 else -0.8
            data[i, H36M_L_ANKLE, :] = [-0.5, 0, left_z]
            
            # Right: constant in front
            data[i, H36M_R_ANKLE, :] = [0.5, 0, 0.5]
        
        skeleton.load_data(data)
        confidence = np.ones((num_frames, 17), dtype=np.float32)
        pose_data = VectorizedPoseData(skeleton, confidence)
        
        result = extract_feet_position(pose_data)
        assert result['feet_position'].shape == (num_frames, 2)
        
        # Right always in front
        assert np.all(result['feet_position'][:, 1] == 1.0)
        
        # Left alternates (50%)
        assert np.isclose(np.mean(result['feet_position'][:, 0]), 0.5)
