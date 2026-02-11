"""Tests for leg straightening timing analysis."""

import pytest
import numpy as np
from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
    extract_leg_straightening_timing,
    get_leg_timing_interpretation,
    analyze_step_timing,
    H36M_PELVIS,
    H36M_L_HIP,
    H36M_R_HIP,
    H36M_L_KNEE,
    H36M_R_KNEE,
    H36M_L_ANKLE,
    H36M_R_ANKLE,
    VELOCITY_THRESHOLD,
    HIP_DISTANCE_THRESHOLD,
    KNEE_STRAIGHT_ANGLE,
)
from shared.skeletons.skeleton import VectorizedSkeleton
from shared.skeletons.pose_data import VectorizedPoseData


@pytest.fixture
def human_17_skeleton_with_walk_data():
    """Create Human3.6M skeleton with rumba walk simulation data."""
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
    
    # Create rumba walk simulation: 8 frames showing one step cycle
    num_frames = 8
    data = np.zeros((num_frames, 17, 3), dtype=np.float32)
    
    # Base positions
    hip_height = 1.0
    knee_height = 0.5
    ankle_height = 0.0
    
    for frame in range(num_frames):
        # Pelvis stays relatively stable
        data[frame, H36M_PELVIS, :] = [0, hip_height, 0]
        
        # Left leg cycle: swing -> contact -> action -> arrival
        phase = frame % 4
        if phase == 0:  # Swing phase
            # Hip forward, knee bent, ankle moving fast
            data[frame, H36M_L_HIP, :] = [-0.3, hip_height, 0.2]  # Hip forward
            data[frame, H36M_L_KNEE, :] = [-0.2, knee_height, 0.4]  # Knee bent
            data[frame, H36M_L_ANKLE, :] = [-0.1, ankle_height, 0.6]  # Ankle forward
        elif phase == 1:  # Contact phase
            # Foot touches, but hip not yet over ankle
            data[frame, H36M_L_HIP, :] = [-0.3, hip_height, 0]
            data[frame, H36M_L_KNEE, :] = [-0.2, knee_height, 0.1] 
            data[frame, H36M_L_ANKLE, :] = [-0.3, ankle_height, 0.3]  # Ankle stops
        elif phase == 2:  # Action phase  
            # Hip moving over ankle, leg straightening
            data[frame, H36M_L_HIP, :] = [-0.3, hip_height, 0.2]
            data[frame, H36M_L_KNEE, :] = [-0.3, knee_height, 0.25]  # Straighter
            data[frame, H36M_L_ANKLE, :] = [-0.3, ankle_height, 0.3]  # Planted
        else:  # Arrival phase
            # Hip over ankle, leg straight
            data[frame, H36M_L_HIP, :] = [-0.3, hip_height, 0.3]
            data[frame, H36M_L_KNEE, :] = [-0.3, knee_height, 0.15]  # Nearly straight
            data[frame, H36M_L_ANKLE, :] = [-0.3, ankle_height, 0.3]  # Planted
        
        # Right leg (opposite phase, offset by 2)
        right_phase = (frame + 2) % 4
        if right_phase == 0:  # Swing phase
            data[frame, H36M_R_HIP, :] = [0.3, hip_height, 0.2]
            data[frame, H36M_R_KNEE, :] = [0.2, knee_height, 0.4]
            data[frame, H36M_R_ANKLE, :] = [0.1, ankle_height, 0.6]
        elif right_phase == 1:  # Contact phase
            data[frame, H36M_R_HIP, :] = [0.3, hip_height, 0]
            data[frame, H36M_R_KNEE, :] = [0.2, knee_height, 0.1]
            data[frame, H36M_R_ANKLE, :] = [0.3, ankle_height, 0.3]
        elif right_phase == 2:  # Action phase
            data[frame, H36M_R_HIP, :] = [0.3, hip_height, 0.2] 
            data[frame, H36M_R_KNEE, :] = [0.3, knee_height, 0.25]
            data[frame, H36M_R_ANKLE, :] = [0.3, ankle_height, 0.3]
        else:  # Arrival phase
            data[frame, H36M_R_HIP, :] = [0.3, hip_height, 0.3]
            data[frame, H36M_R_KNEE, :] = [0.3, knee_height, 0.15]
            data[frame, H36M_R_ANKLE, :] = [0.3, ankle_height, 0.3]
    
    skeleton.load_data(data)
    confidence = np.ones((num_frames, 17), dtype=np.float32) * 0.9
    pose_data = VectorizedPoseData(skeleton, confidence)
    
    return pose_data


class TestLegStraighteningTiming:
    """Test leg straightening timing analysis."""
    
    def test_extract_leg_timing_shape(self, human_17_skeleton_with_walk_data):
        """Test output shape and structure."""
        result = extract_leg_straightening_timing(human_17_skeleton_with_walk_data)
        
        assert 'left_phases' in result
        assert 'right_phases' in result
        assert 'left_metrics' in result
        assert 'right_metrics' in result
        assert 'summary' in result
        
        assert result['left_phases'].shape == (8,)
        assert result['right_phases'].shape == (8,)
    
    def test_phase_values_in_range(self, human_17_skeleton_with_walk_data):
        """Test that phase values are in expected range."""
        result = extract_leg_straightening_timing(human_17_skeleton_with_walk_data)
        
        left_phases = result['left_phases']
        right_phases = result['right_phases']
        
        # Phases should be 0-3 (swing, contact, action, arrival)
        assert np.all(left_phases >= 0)
        assert np.all(left_phases <= 3)
        assert np.all(right_phases >= 0)
        assert np.all(right_phases <= 3)
    
    def test_metrics_structure(self, human_17_skeleton_with_walk_data):
        """Test that metrics contain expected data."""
        result = extract_leg_straightening_timing(human_17_skeleton_with_walk_data)
        
        left_metrics = result['left_metrics']
        right_metrics = result['right_metrics']
        
        expected_keys = [
            'ankle_speed', 'hip_ankle_distance', 'knee_angles',
            'is_planted', 'weight_transferred', 'leg_straight'
        ]
        
        for key in expected_keys:
            assert key in left_metrics
            assert key in right_metrics
            assert left_metrics[key].shape == (8,)
            assert right_metrics[key].shape == (8,)
    
    def test_summary_statistics(self, human_17_skeleton_with_walk_data):
        """Test summary statistics structure."""
        result = extract_leg_straightening_timing(human_17_skeleton_with_walk_data)
        
        summary = result['summary']
        
        # Phase percentages
        phase_keys = ['swing', 'contact', 'action', 'arrival']
        for phase in phase_keys:
            assert f'left_{phase}_pct' in summary
            assert f'right_{phase}_pct' in summary
            assert 0 <= summary[f'left_{phase}_pct'] <= 100
            assert 0 <= summary[f'right_{phase}_pct'] <= 100
        
        # Phase-conditional metrics (new approach)
        phase_conditional_keys = [
            'left_arrival_straightness_deg', 'right_arrival_straightness_deg',
            'left_contact_bend_deg', 'right_contact_bend_deg',
            'left_action_weight_transfer', 'right_action_weight_transfer',
            'left_avg_ankle_speed', 'right_avg_ankle_speed',
            'left_planted_pct', 'right_planted_pct'
        ]
        for key in phase_conditional_keys:
            assert key in summary
    
    def test_single_frame_handling(self):
        """Test handling of single frame data."""
        joint_names = [
            "Pelvis", "R_Hip", "R_Knee", "R_Ankle", 
            "L_Hip", "L_Knee", "L_Ankle", 
            "Spine", "Neck", "Head", "Site", 
            "L_Shoulder", "L_Elbow", "L_Wrist", 
            "R_Shoulder", "R_Elbow", "R_Wrist"
        ]
        bones = [["Pelvis", "Spine"]]
        skeleton = VectorizedSkeleton(joint_names, bones)
        
        # Single frame
        data = np.zeros((1, 17, 3), dtype=np.float32)
        skeleton.load_data(data)
        confidence = np.ones((1, 17), dtype=np.float32)
        pose_data = VectorizedPoseData(skeleton, confidence)
        
        result = extract_leg_straightening_timing(pose_data)
        
        # Should return empty results gracefully
        assert result['left_phases'].shape == (0,)
        assert result['right_phases'].shape == (0,)
    
    def test_velocity_threshold_logic(self):
        """Test that velocity threshold correctly identifies planted feet."""
        joint_names = [
            "Pelvis", "R_Hip", "R_Knee", "R_Ankle", 
            "L_Hip", "L_Knee", "L_Ankle", 
            "Spine", "Neck", "Head", "Site", 
            "L_Shoulder", "L_Elbow", "L_Wrist", 
            "R_Shoulder", "R_Elbow", "R_Wrist"
        ]
        bones = [["L_Hip", "L_Knee"], ["L_Knee", "L_Ankle"]]
        skeleton = VectorizedSkeleton(joint_names, bones)
        
        # Create data where ankle moves then stops
        data = np.zeros((4, 17, 3), dtype=np.float32)
        data[0, H36M_L_ANKLE, :] = [0, 0, 0]     # Start
        data[1, H36M_L_ANKLE, :] = [0.5, 0, 0]   # Moving (high velocity)
        data[2, H36M_L_ANKLE, :] = [0.51, 0, 0]  # Barely moving (low velocity)
        data[3, H36M_L_ANKLE, :] = [0.51, 0, 0]  # Stopped (zero velocity)
        
        # Set other joints to reasonable positions
        for frame in range(4):
            data[frame, H36M_L_HIP, :] = [0, 1, 0]
            data[frame, H36M_L_KNEE, :] = [0, 0.5, 0]
        
        skeleton.load_data(data)
        confidence = np.ones((4, 17), dtype=np.float32)
        pose_data = VectorizedPoseData(skeleton, confidence)
        
        result = extract_leg_straightening_timing(pose_data)
        left_metrics = result['left_metrics']
        
        # Check that is_planted logic works
        # Frame 0: should be planted (velocity approximated from frame 1)
        # Frame 1: high velocity, not planted
        # Frame 2: low velocity, planted
        # Frame 3: zero velocity, planted
        assert left_metrics['is_planted'][1] == False  # Moving
        assert left_metrics['is_planted'][2] == True   # Stopped
        assert left_metrics['is_planted'][3] == True   # Stopped


class TestTimingInterpretation:
    """Test timing interpretation utilities."""
    
    def test_get_leg_timing_interpretation(self):
        """Test phase code to name conversion."""
        phases = np.array([0, 1, 2, 3, 0, 1])
        names = get_leg_timing_interpretation(phases)
        
        expected = ['swing', 'contact', 'action', 'arrival', 'swing', 'contact']
        assert names == expected
    
    def test_analyze_step_timing(self):
        """Test step timing relationship analysis."""
        left_phases = np.array([0, 1, 2, 3, 0, 1, 2, 3])   # Regular cycle
        right_phases = np.array([2, 3, 0, 1, 2, 3, 0, 1])  # Offset cycle
        
        timing = analyze_step_timing(left_phases, right_phases)
        
        assert 'left_transition_count' in timing
        assert 'right_transition_count' in timing
        assert 'alternation_percentage' in timing
        assert 'simultaneous_arrival_frames' in timing
        
        # Should have good alternation (legs in different phases)
        assert timing['alternation_percentage'] > 50
        
        # Should have same number of transitions
        assert timing['left_transition_count'] == timing['right_transition_count']


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data(self):
        """Test with empty data."""
        joint_names = [
            "Pelvis", "R_Hip", "R_Knee", "R_Ankle", 
            "L_Hip", "L_Knee", "L_Ankle", 
            "Spine", "Neck", "Head", "Site", 
            "L_Shoulder", "L_Elbow", "L_Wrist", 
            "R_Shoulder", "R_Elbow", "R_Wrist"
        ]
        bones = [["Pelvis", "Spine"]]
        skeleton = VectorizedSkeleton(joint_names, bones)
        
        # Empty data
        data = np.zeros((0, 17, 3), dtype=np.float32)
        skeleton.load_data(data)
        confidence = np.ones((0, 17), dtype=np.float32)
        pose_data = VectorizedPoseData(skeleton, confidence)
        
        result = extract_leg_straightening_timing(pose_data)
        
        assert result['left_phases'].shape == (0,)
        assert result['right_phases'].shape == (0,)
        assert result['summary'] == {}
    
    def test_minimal_motion(self):
        """Test with very small movements."""
        joint_names = [
            "Pelvis", "R_Hip", "R_Knee", "R_Ankle", 
            "L_Hip", "L_Knee", "L_Ankle", 
            "Spine", "Neck", "Head", "Site", 
            "L_Shoulder", "L_Elbow", "L_Wrist", 
            "R_Shoulder", "R_Elbow", "R_Wrist"
        ]
        bones = [["L_Hip", "L_Knee"], ["L_Knee", "L_Ankle"]]
        skeleton = VectorizedSkeleton(joint_names, bones)
        
        # Minimal motion (should be classified as planted/arrival)
        data = np.zeros((5, 17, 3), dtype=np.float32)
        for frame in range(5):
            data[frame, H36M_L_HIP, :] = [0, 1, 0]
            data[frame, H36M_L_KNEE, :] = [0, 0.5, 0] 
            data[frame, H36M_L_ANKLE, :] = [0, 0, 0]
        
        skeleton.load_data(data)
        confidence = np.ones((5, 17), dtype=np.float32)
        pose_data = VectorizedPoseData(skeleton, confidence)
        
        result = extract_leg_straightening_timing(pose_data)
        
        # Should mostly be in planted/stable phases (not swing)
        left_phases = result['left_phases']
        swing_frames = np.sum(left_phases == 0)  # Phase 0 = swing
        assert swing_frames <= 1  # At most frame 0 (velocity approximation)
    
    def test_invalid_skeleton(self):
        """Test error handling with wrong skeleton format."""
        joint_names = ["a", "b", "c"]  # Wrong skeleton
        bones = [["a", "b"]]
        skeleton = VectorizedSkeleton(joint_names, bones)
        
        data = np.zeros((3, 3, 3), dtype=np.float32)
        skeleton.load_data(data)
        confidence = np.ones((3, 3), dtype=np.float32)
        pose_data = VectorizedPoseData(skeleton, confidence)
        
        # Should handle gracefully with warning
        result = extract_leg_straightening_timing(pose_data)
        
        # Should still return structured result
        assert 'left_phases' in result
        assert 'right_phases' in result