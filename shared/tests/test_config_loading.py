"""Unit tests for loading actual skeleton configuration files."""
import pytest
import os
import numpy as np
from shared.skeletons.skeleton_registry import SkeletonRegistry
from shared.skeletons.skeleton import VectorizedSkeleton
from shared.skeletons.pose_data import VectorizedPoseData


class TestConfigFileLoading:
    """Test loading real skeleton configuration files."""

    @pytest.fixture
    def config_dir(self):
        """Get path to configs directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # shared/tests -> shared -> project_root
        shared_dir = os.path.dirname(current_dir)
        project_root = os.path.dirname(shared_dir)
        return os.path.join(project_root, "shared", "configs", "skeletons")

    def test_coco_w_config_exists(self, config_dir):
        """Test that coco_w.json exists."""
        config_path = os.path.join(config_dir, "coco_w.json")
        assert os.path.exists(config_path), f"Config file not found: {config_path}"

    def test_human_17_config_exists(self, config_dir):
        """Test that human_17.json exists."""
        config_path = os.path.join(config_dir, "human_17.json")
        assert os.path.exists(config_path), f"Config file not found: {config_path}"

    def test_load_coco_w_config(self, config_dir):
        """Test loading COCO-WholeBody configuration."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "coco_w.json")
        
        name = registry.load_from_json(config_path)
        assert name == "coco_w"
        
        joints, bones = registry.get("coco_w")
        assert len(joints) == 133
        assert len(bones) > 0

    def test_load_human_17_config(self, config_dir):
        """Test loading Human3.6M configuration."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "human_17.json")
        
        name = registry.load_from_json(config_path)
        assert name == "human_17"
        
        joints, bones = registry.get("human_17")
        assert len(joints) == 17
        assert len(bones) == 16

    def test_coco_w_skeleton_instantiation(self, config_dir):
        """Test creating VectorizedSkeleton from COCO-W config."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "coco_w.json")
        registry.load_from_json(config_path)
        
        joints, bones = registry.get("coco_w")
        skeleton = VectorizedSkeleton(joints, bones)
        
        assert skeleton.num_joints == 133
        assert skeleton.data is None

    def test_human_17_skeleton_instantiation(self, config_dir):
        """Test creating VectorizedSkeleton from Human3.6M config."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "human_17.json")
        registry.load_from_json(config_path)
        
        joints, bones = registry.get("human_17")
        skeleton = VectorizedSkeleton(joints, bones)
        
        assert skeleton.num_joints == 17

    def test_coco_w_skeleton_with_data(self, config_dir):
        """Test loading data into COCO-W skeleton."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "coco_w.json")
        registry.load_from_json(config_path)
        
        joints, bones = registry.get("coco_w")
        skeleton = VectorizedSkeleton(joints, bones)
        
        # Create mock data: 5 frames, 133 joints, 3D coordinates
        data = np.random.randn(5, 133, 3).astype(np.float32)
        skeleton.load_data(data)
        
        assert skeleton.num_frames == 5
        lengths = skeleton.get_bone_lengths()
        assert lengths.shape == (5, len(bones))

    def test_human_17_skeleton_with_data(self, config_dir):
        """Test loading data into Human3.6M skeleton."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "human_17.json")
        registry.load_from_json(config_path)
        
        joints, bones = registry.get("human_17")
        skeleton = VectorizedSkeleton(joints, bones)
        
        # Create mock data: 10 frames, 17 joints, 3D coordinates
        data = np.random.randn(10, 17, 3).astype(np.float32)
        skeleton.load_data(data)
        
        assert skeleton.num_frames == 10
        lengths = skeleton.get_bone_lengths()
        assert lengths.shape == (10, len(bones))

    def test_coco_w_pose_data_creation(self, config_dir):
        """Test creating PoseData from COCO-W skeleton."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "coco_w.json")
        registry.load_from_json(config_path)
        
        joints, bones = registry.get("coco_w")
        skeleton = VectorizedSkeleton(joints, bones)
        
        # Mock data and confidence
        data = np.random.randn(5, 133, 3).astype(np.float32)
        skeleton.load_data(data)
        
        confidence = np.random.uniform(0.5, 1.0, size=(5, 133)).astype(np.float32)
        pose_data = VectorizedPoseData(skeleton, confidence)
        
        assert pose_data.num_frames == 5
        assert pose_data.num_joints == 133

    def test_human_17_pose_data_creation(self, config_dir):
        """Test creating PoseData from Human3.6M skeleton."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "human_17.json")
        registry.load_from_json(config_path)
        
        joints, bones = registry.get("human_17")
        skeleton = VectorizedSkeleton(joints, bones)
        
        # Mock data and confidence
        data = np.random.randn(10, 17, 3).astype(np.float32)
        skeleton.load_data(data)
        
        confidence = np.random.uniform(0.5, 1.0, size=(10, 17)).astype(np.float32)
        pose_data = VectorizedPoseData(skeleton, confidence)
        
        assert pose_data.num_frames == 10
        assert pose_data.num_joints == 17
        assert pose_data.get_average_confidence() > 0

    def test_coco_w_joint_name_mapping(self, config_dir):
        """Test that COCO-W joint names are correctly mapped."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "coco_w.json")
        registry.load_from_json(config_path)
        
        joints, bones = registry.get("coco_w")
        skeleton = VectorizedSkeleton(joints, bones)
        
        # Check key joint names exist
        assert "Nose" in skeleton.name_to_idx
        assert "L_Eye" in skeleton.name_to_idx
        assert "R_Eye" in skeleton.name_to_idx
        assert "L_Shoulder" in skeleton.name_to_idx
        assert "R_Shoulder" in skeleton.name_to_idx

    def test_human_17_joint_name_mapping(self, config_dir):
        """Test that Human3.6M joint names are correctly mapped."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "human_17.json")
        registry.load_from_json(config_path)
        
        joints, bones = registry.get("human_17")
        skeleton = VectorizedSkeleton(joints, bones)
        
        # Check key joint names exist
        assert "Pelvis" in skeleton.name_to_idx
        assert "Spine" in skeleton.name_to_idx
        assert "Neck" in skeleton.name_to_idx
        assert "Head" in skeleton.name_to_idx
        assert "L_Shoulder" in skeleton.name_to_idx
        assert "R_Shoulder" in skeleton.name_to_idx

    def test_human_17_bone_connectivity(self, config_dir):
        """Test that Human3.6M skeleton bones form correct connectivity."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "human_17.json")
        registry.load_from_json(config_path)
        
        joints, bones = registry.get("human_17")
        skeleton = VectorizedSkeleton(joints, bones)
        
        # Check spine chain: Pelvis -> Spine -> Neck -> Head
        pelvis_idx = skeleton.name_to_idx["Pelvis"]
        spine_idx = skeleton.name_to_idx["Spine"]
        neck_idx = skeleton.name_to_idx["Neck"]
        head_idx = skeleton.name_to_idx["Head"]
        
        # Should be connected
        assert spine_idx in skeleton.bones[pelvis_idx]
        assert neck_idx in skeleton.bones[spine_idx]
        assert head_idx in skeleton.bones[neck_idx]

    def test_coco_w_specific_angles(self, config_dir):
        """Test calculating angles in COCO-W skeleton."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "coco_w.json")
        registry.load_from_json(config_path)
        
        joints, bones = registry.get("coco_w")
        skeleton = VectorizedSkeleton(joints, bones)
        
        # Create data with straight arm: L_Shoulder -> L_Elbow -> L_Wrist
        data = np.array([
            [[0, 0, 0] for _ in range(133)]
        ], dtype=np.float32)
        
        # Set positions for shoulder-elbow-wrist line
        l_shoulder_idx = skeleton.name_to_idx["L_Shoulder"]
        l_elbow_idx = skeleton.name_to_idx["L_Elbow"]
        l_wrist_idx = skeleton.name_to_idx["L_Wrist"]
        
        # Straight line: (0,0,0) -> (1,0,0) -> (2,0,0)
        data[0, l_shoulder_idx] = [0, 0, 0]
        data[0, l_elbow_idx] = [1, 0, 0]
        data[0, l_wrist_idx] = [2, 0, 0]
        
        skeleton.load_data(data)
        
        # Calculate angle at elbow
        angle = skeleton.get_angle("L_Shoulder", "L_Elbow", "L_Wrist")
        # Should be close to π (straight line = 180 degrees)
        assert np.isclose(angle[0], np.pi, atol=0.01)

    def test_human_17_elbow_angle(self, config_dir):
        """Test calculating elbow angle in Human3.6M skeleton."""
        registry = SkeletonRegistry()
        config_path = os.path.join(config_dir, "human_17.json")
        registry.load_from_json(config_path)
        
        joints, bones = registry.get("human_17")
        skeleton = VectorizedSkeleton(joints, bones)
        
        # Create data with bent elbow
        data = np.zeros((1, 17, 3), dtype=np.float32)
        
        l_shoulder_idx = skeleton.name_to_idx["L_Shoulder"]
        l_elbow_idx = skeleton.name_to_idx["L_Elbow"]
        l_wrist_idx = skeleton.name_to_idx["L_Wrist"]
        
        # Right angle: (0,0,0) -> (1,0,0) -> (1,1,0)
        data[0, l_shoulder_idx] = [0, 0, 0]
        data[0, l_elbow_idx] = [1, 0, 0]
        data[0, l_wrist_idx] = [1, 1, 0]
        
        skeleton.load_data(data)
        
        angle = skeleton.get_angle("L_Shoulder", "L_Elbow", "L_Wrist")
        # Should be close to π/2 (90 degrees)
        assert np.isclose(angle[0], np.pi / 2, atol=0.01)


class TestMultipleConfigLoading:
    """Test loading multiple configs simultaneously."""

    @pytest.fixture
    def config_dir(self):
        """Get path to configs directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # shared/tests -> shared -> project_root
        shared_dir = os.path.dirname(current_dir)
        project_root = os.path.dirname(shared_dir)
        return os.path.join(project_root, "shared", "configs", "skeletons")

    def test_load_both_configs(self, config_dir):
        """Test loading both configs into same registry."""
        registry = SkeletonRegistry()
        
        coco_path = os.path.join(config_dir, "coco_w.json")
        human_path = os.path.join(config_dir, "human_17.json")
        
        registry.load_from_json(coco_path)
        registry.load_from_json(human_path)
        
        # Both should be in registry
        coco_joints, coco_bones = registry.get("coco_w")
        human_joints, human_bones = registry.get("human_17")
        
        assert len(coco_joints) == 133
        assert len(human_joints) == 17

    def test_create_skeletons_from_both_configs(self, config_dir):
        """Test creating skeletons from both configs."""
        registry = SkeletonRegistry()
        
        coco_path = os.path.join(config_dir, "coco_w.json")
        human_path = os.path.join(config_dir, "human_17.json")
        
        registry.load_from_json(coco_path)
        registry.load_from_json(human_path)
        
        # Create both skeletons
        coco_joints, coco_bones = registry.get("coco_w")
        human_joints, human_bones = registry.get("human_17")
        
        coco_skeleton = VectorizedSkeleton(coco_joints, coco_bones)
        human_skeleton = VectorizedSkeleton(human_joints, human_bones)
        
        # Load different data sizes
        coco_data = np.random.randn(5, 133, 3).astype(np.float32)
        human_data = np.random.randn(10, 17, 3).astype(np.float32)
        
        coco_skeleton.load_data(coco_data)
        human_skeleton.load_data(human_data)
        
        assert coco_skeleton.num_frames == 5
        assert human_skeleton.num_frames == 10
