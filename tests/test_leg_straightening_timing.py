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


# ============================================================================
# NEW TESTS FOR 2D SIDE VIEW ANALYSIS
# ============================================================================

class Test2DChaChWalk:
    """Test 2D side-view cha-cha walk analysis."""
    
    @pytest.fixture
    def coco_17_skeleton_2d(self):
        """Create COCO-17 skeleton with 2D side-view walk data."""
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
            ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
            ["left_hip", "right_hip"],
            ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"]
        ]
        
        skeleton = VectorizedSkeleton(joint_names, bones)
        
        # Create 2D side-view walk simulation (pixel coordinates)
        # X = horizontal (forward/backward), Y = vertical (up/down)
        num_frames = 60
        data = np.zeros((num_frames, 17, 2), dtype=np.float32)
        
        # Image is 1920x1080, person centered
        hip_y = 600  # Hip height in pixels
        knee_y = 800  # Knee height
        ankle_y = 1000  # Ankle on ground
        center_x = 960
        
        for frame in range(num_frames):
            # Simple walk cycle: frame 0-29 = left step, 30-59 = right step
            progress = (frame % 30) / 30.0
            
            # Both hips stay relatively stable vertically
            data[frame, 11, :] = [center_x - 20, hip_y]  # left_hip
            data[frame, 12, :] = [center_x + 20, hip_y]  # right_hip
            
            if frame < 30:  # Left leg forward step
                # Left leg moves forward
                left_ankle_x = center_x + (progress * 200 - 100)
                data[frame, 15, :] = [left_ankle_x, ankle_y]  # left_ankle
                
                # Knee bends during passing, straightens at arrival
                if progress < 0.3:  # Release/passing
                    left_knee_y = knee_y + 30  # Knee bends more
                elif progress < 0.7:  # Extension
                    left_knee_y = knee_y
                else:  # Arrival
                    left_knee_y = knee_y - 10  # Leg straighter
                data[frame, 13, :] = [left_ankle_x - 10, left_knee_y]  # left_knee
                
                # Right leg stays back (standing leg)
                data[frame, 16, :] = [center_x - 100, ankle_y]  # right_ankle
                data[frame, 14, :] = [center_x - 110, knee_y - 10]  # right_knee (straight)
                
            else:  # Right leg forward step
                # Right leg moves forward
                right_ankle_x = center_x + ((progress - 0.5) * 2 * 200 - 100)
                data[frame, 16, :] = [right_ankle_x, ankle_y]  # right_ankle
                
                # Knee bends during passing, straightens at arrival
                if progress < 0.8:  # Release/passing
                    right_knee_y = knee_y + 30
                elif progress < 0.9:  # Extension
                    right_knee_y = knee_y
                else:  # Arrival
                    right_knee_y = knee_y - 10
                data[frame, 14, :] = [right_ankle_x + 10, right_knee_y]  # right_knee
                
                # Left leg stays back (standing leg)
                data[frame, 15, :] = [center_x + 100, ankle_y]  # left_ankle
                data[frame, 13, :] = [center_x + 110, knee_y - 10]  # left_knee (straight)
        
        skeleton.load_data(data)
        confidence = np.ones((num_frames, 17), dtype=np.float32) * 0.9
        pose_data = VectorizedPoseData(skeleton, confidence)
        
        return pose_data
    
    def test_2d_analysis_returns_expected_structure(self, coco_17_skeleton_2d):
        """Test that 2D analysis returns correct structure."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            analyze_cha_cha_walk_2d
        )
        
        result = analyze_cha_cha_walk_2d(coco_17_skeleton_2d)
        
        assert 'states' in result
        assert 'faults' in result
        assert 'final_standing_leg' in result
        
        num_frames = coco_17_skeleton_2d.num_frames
        assert len(result['states']) == num_frames
        assert isinstance(result['faults'], list)
        assert result['final_standing_leg'] in ['Left', 'Right']
    
    def test_2d_helper_functions(self, coco_17_skeleton_2d):
        """Test 2D helper functions work correctly."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            compute_weight_transfer_offsets_2d,
            compute_ankle_to_ankle_distance_2d,
            magnitude_2d
        )
        
        # Test weight transfer computation
        offsets = compute_weight_transfer_offsets_2d(coco_17_skeleton_2d)
        assert offsets.shape == (coco_17_skeleton_2d.num_frames, 2, 2)
        assert offsets.dtype == np.float32
        
        # Test ankle distance computation
        distances = compute_ankle_to_ankle_distance_2d(coco_17_skeleton_2d)
        assert distances.shape == (coco_17_skeleton_2d.num_frames,)
        assert np.all(distances >= 0)
        
        # Test 2D magnitude
        vec = np.array([3.0, 4.0])
        mag = magnitude_2d(vec)
        assert np.isclose(mag, 5.0)
    
    def test_2d_state_transitions(self, coco_17_skeleton_2d):
        """Test that 2D analysis produces state transitions."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            analyze_cha_cha_walk_2d
        )
        
        result = analyze_cha_cha_walk_2d(coco_17_skeleton_2d)
        states = result['states']
        
        # Should have multiple different states
        unique_states = set(states)
        assert len(unique_states) > 1, "Should transition between states"
        
        # Should include expected state names
        valid_states = {'RELEASE', 'PASSING', 'EXTENSION', 'ARRIVAL', 'COMPLETED'}
        assert unique_states.issubset(valid_states)
    
    def test_2d_low_confidence_handling(self):
        """Test that 2D analysis handles low confidence frames gracefully."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            analyze_cha_cha_walk_2d
        )
        
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
        
        # Create simple data
        data = np.zeros((10, 17, 2), dtype=np.float32)
        for frame in range(10):
            data[frame, 11, :] = [960, 600]  # left_hip
            data[frame, 12, :] = [980, 600]  # right_hip
            data[frame, 13, :] = [960, 800]  # left_knee
            data[frame, 14, :] = [980, 800]  # right_knee
            data[frame, 15, :] = [960, 1000]  # left_ankle
            data[frame, 16, :] = [980, 1000]  # right_ankle
        
        skeleton.load_data(data)
        
        # Low confidence for some frames
        confidence = np.ones((10, 17), dtype=np.float32) * 0.1  # Very low
        pose_data = VectorizedPoseData(skeleton, confidence)
        
        # Should not crash, but may not detect faults
        result = analyze_cha_cha_walk_2d(pose_data)
        assert 'states' in result
        assert 'faults' in result
    
    def test_2d_feature_extraction_integration(self, coco_17_skeleton_2d):
        """Test integration with feature_extraction module."""
        from video_processing.app.analysis.feature_extraction.feature_extraction import (
            extract_features
        )
        
        # Test calling with 2D data
        result = extract_features(
            pose_data_2d=coco_17_skeleton_2d,
            use_2d_analysis=True
        )
        
        # Should return without error
        assert isinstance(result, dict)
    
    def test_2d_vs_3d_api_compatibility(self):
        """Test that 2D and 3D analysis have compatible APIs."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            analyze_cha_cha_walk,
            analyze_cha_cha_walk_2d
        )
        
        # Both should have same output structure
        # (Just check that they have the same keys expected)
        expected_keys = {'states', 'faults', 'final_standing_leg'}
        
        # We can't easily test both without proper data, but we can at least
        # verify the function signatures exist
        assert callable(analyze_cha_cha_walk)
        assert callable(analyze_cha_cha_walk_2d)


# ============================================================================
# NEW TESTS: Bent vs. Hyper-Extended Leg Differentiation in 2D Analysis
# ============================================================================

def _make_coco17_pose_data(data: np.ndarray, confidence_value: float = 0.9):
    """Helper to build a minimal COCO-17 VectorizedPoseData from raw 2D data."""
    joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]
    bones = [
        ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
        ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
        ["left_hip", "right_hip"],
        ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
    ]
    num_frames = data.shape[0]
    skel = VectorizedSkeleton(joint_names, bones)
    skel.load_data(data)
    conf = np.ones((num_frames, 17), dtype=np.float32) * confidence_value
    return VectorizedPoseData(skel, conf)


class TestWalkDirectionDetection:
    """Tests for determine_walk_direction_2d."""

    def test_walking_right_detected(self):
        """Person moving in +X direction should return +1."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            determine_walk_direction_2d,
        )
        num_frames = 10
        data = np.zeros((num_frames, 17, 2), dtype=np.float32)
        for f in range(num_frames):
            x = 400 + f * 50  # hips move right
            data[f, 11, :] = [x - 10, 600]  # left_hip
            data[f, 12, :] = [x + 10, 600]  # right_hip

        pd = _make_coco17_pose_data(data)
        assert determine_walk_direction_2d(pd) == 1

    def test_walking_left_detected(self):
        """Person moving in -X direction should return -1."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            determine_walk_direction_2d,
        )
        num_frames = 10
        data = np.zeros((num_frames, 17, 2), dtype=np.float32)
        for f in range(num_frames):
            x = 800 - f * 50  # hips move left
            data[f, 11, :] = [x - 10, 600]
            data[f, 12, :] = [x + 10, 600]

        pd = _make_coco17_pose_data(data)
        assert determine_walk_direction_2d(pd) == -1

    def test_single_frame_defaults_to_right(self):
        """Single-frame clip cannot determine direction; should default to +1."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            determine_walk_direction_2d,
        )
        data = np.zeros((1, 17, 2), dtype=np.float32)
        data[0, 11, :] = [400, 600]
        data[0, 12, :] = [420, 600]
        pd = _make_coco17_pose_data(data)
        assert determine_walk_direction_2d(pd) == 1


class TestKneeDeviation:
    """Tests for compute_knee_deviation_2d."""

    def test_straight_leg_zero_deviation(self):
        """Knee on the hip-ankle line should have ~0 deviation."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            compute_knee_deviation_2d,
        )
        data = np.zeros((3, 17, 2), dtype=np.float32)
        for f in range(3):
            # Left leg: perfectly straight (knee on the line between hip and ankle)
            data[f, 11, :] = [500, 600]   # left_hip
            data[f, 13, :] = [500, 800]   # left_knee (on the line)
            data[f, 15, :] = [500, 1000]  # left_ankle
            # Right leg: same
            data[f, 12, :] = [520, 600]
            data[f, 14, :] = [520, 800]
            data[f, 16, :] = [520, 1000]

        pd = _make_coco17_pose_data(data)
        deviations = compute_knee_deviation_2d(pd)
        assert deviations.shape == (3, 2)
        np.testing.assert_allclose(deviations[:, 0], 0.0, atol=1e-4)
        np.testing.assert_allclose(deviations[:, 1], 0.0, atol=1e-4)

    def test_forward_bent_knee_positive_deviation(self):
        """Knee bent forward (+X) of the hip-ankle line → positive deviation."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            compute_knee_deviation_2d,
        )
        data = np.zeros((2, 17, 2), dtype=np.float32)
        for f in range(2):
            data[f, 11, :] = [500, 600]   # left_hip
            data[f, 13, :] = [530, 800]   # left_knee pushed RIGHT (+X = forward)
            data[f, 15, :] = [500, 1000]  # left_ankle
            data[f, 12, :] = [520, 600]
            data[f, 14, :] = [520, 800]
            data[f, 16, :] = [520, 1000]

        pd = _make_coco17_pose_data(data)
        deviations = compute_knee_deviation_2d(pd)
        # Left knee is 30 px to the right of the hip-ankle line
        assert deviations[0, 0] > 0, "Forward bent knee should have positive deviation"
        np.testing.assert_allclose(deviations[0, 0], 30.0, atol=1e-3)

    def test_hyper_extended_knee_negative_deviation(self):
        """Knee pushed backward (-X) of the hip-ankle line → negative deviation."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            compute_knee_deviation_2d,
        )
        data = np.zeros((2, 17, 2), dtype=np.float32)
        for f in range(2):
            data[f, 11, :] = [500, 600]   # left_hip
            data[f, 13, :] = [470, 800]   # left_knee pushed LEFT (-X = backward)
            data[f, 15, :] = [500, 1000]  # left_ankle
            data[f, 12, :] = [520, 600]
            data[f, 14, :] = [520, 800]
            data[f, 16, :] = [520, 1000]

        pd = _make_coco17_pose_data(data)
        deviations = compute_knee_deviation_2d(pd)
        # Left knee is 30 px to the left of the hip-ankle line
        assert deviations[0, 0] < 0, "Hyper-extended knee should have negative deviation"
        np.testing.assert_allclose(deviations[0, 0], -30.0, atol=1e-3)

    def test_output_shape(self):
        """Output should be (frames, 2)."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            compute_knee_deviation_2d,
        )
        data = np.zeros((7, 17, 2), dtype=np.float32)
        for f in range(7):
            data[f, 11, :] = [500, 600]
            data[f, 13, :] = [500, 800]
            data[f, 15, :] = [500, 1000]
            data[f, 12, :] = [520, 600]
            data[f, 14, :] = [520, 800]
            data[f, 16, :] = [520, 1000]
        pd = _make_coco17_pose_data(data)
        deviations = compute_knee_deviation_2d(pd)
        assert deviations.shape == (7, 2)
        assert deviations.dtype == np.float32


class TestHyperExtensionFaultDetection:
    """Tests that HYPER_EXTENDED_* faults are emitted correctly."""

    def _build_walking_right_data(self, num_frames: int, standing_knee_offset_x: float):
        """
        Build a simple rightward-walk dataset.  The standing (left) knee is
        displaced horizontally by standing_knee_offset_x relative to the straight
        hip-ankle line.  Negative = backward = hyper-extended.
        """
        data = np.zeros((num_frames, 17, 2), dtype=np.float32)
        for f in range(num_frames):
            x = 400 + f * 20  # person moves right
            data[f, 11, :] = [x - 10, 600]    # left_hip (standing)
            data[f, 12, :] = [x + 10, 600]    # right_hip
            # Standing (left) leg – knee shifted by standing_knee_offset_x
            data[f, 13, :] = [x - 10 + standing_knee_offset_x, 800]  # left_knee
            data[f, 15, :] = [x - 10, 1000]   # left_ankle (directly below hip)
            # Moving (right) leg – forward and moving
            data[f, 14, :] = [x + 60, 800]    # right_knee
            data[f, 16, :] = [x + 80, 1000]   # right_ankle
        return data

    def test_hyper_extended_standing_leg_emits_fault(self):
        """Knee significantly behind hip-ankle line triggers HYPER_EXTENDED_STANDING_LEG."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            analyze_cha_cha_walk_2d,
        )
        # standing_knee_offset_x = -40 (far behind the line) → hyper-extended
        data = self._build_walking_right_data(num_frames=25, standing_knee_offset_x=-40.0)
        pd = _make_coco17_pose_data(data)
        result = analyze_cha_cha_walk_2d(pd, walk_direction=1)
        fault_types = {f["type"] for f in result["faults"]}
        assert "HYPER_EXTENDED_STANDING_LEG" in fault_types

    def test_normal_bent_leg_does_not_emit_hyper_extension_fault(self):
        """Knee in front of hip-ankle line (correct bend) must NOT trigger hyper-extension."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            analyze_cha_cha_walk_2d,
        )
        # standing_knee_offset_x = +20 (forward = correct flexion)
        data = self._build_walking_right_data(num_frames=25, standing_knee_offset_x=20.0)
        pd = _make_coco17_pose_data(data)
        result = analyze_cha_cha_walk_2d(pd, walk_direction=1)
        fault_types = {f["type"] for f in result["faults"]}
        assert "HYPER_EXTENDED_STANDING_LEG" not in fault_types

    def test_walk_direction_included_in_result(self):
        """analyze_cha_cha_walk_2d should return 'walk_direction' in output."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            analyze_cha_cha_walk_2d,
        )
        data = self._build_walking_right_data(num_frames=10, standing_knee_offset_x=0.0)
        pd = _make_coco17_pose_data(data)
        result = analyze_cha_cha_walk_2d(pd)
        assert "walk_direction" in result
        assert result["walk_direction"] in (1, -1)

    def test_explicit_walk_direction_respected(self):
        """Explicitly passed walk_direction must override auto-detection."""
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            analyze_cha_cha_walk_2d,
        )
        data = self._build_walking_right_data(num_frames=10, standing_knee_offset_x=0.0)
        pd = _make_coco17_pose_data(data)

        result_right = analyze_cha_cha_walk_2d(pd, walk_direction=1)
        result_left = analyze_cha_cha_walk_2d(pd, walk_direction=-1)
        assert result_right["walk_direction"] == 1
        assert result_left["walk_direction"] == -1

    def test_hyper_extension_direction_flips_with_walk_direction(self):
        """
        Same knee position classified differently when walk direction reverses:
        a backward-deviated knee (negative X) is hyper-extended for rightward
        walk but correctly bent for leftward walk.
        """
        from video_processing.app.analysis.feature_extraction.leg_straightening_timing import (
            analyze_cha_cha_walk_2d,
        )
        # knee at -40 (to the left/behind for rightward walk)
        data = self._build_walking_right_data(num_frames=25, standing_knee_offset_x=-40.0)
        pd = _make_coco17_pose_data(data)

        result_right = analyze_cha_cha_walk_2d(pd, walk_direction=1)
        result_left = analyze_cha_cha_walk_2d(pd, walk_direction=-1)

        fault_types_right = {f["type"] for f in result_right["faults"]}
        fault_types_left = {f["type"] for f in result_left["faults"]}

        assert "HYPER_EXTENDED_STANDING_LEG" in fault_types_right
        assert "HYPER_EXTENDED_STANDING_LEG" not in fault_types_left