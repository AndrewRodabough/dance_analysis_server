"""Unit tests for SkeletonRegistry class."""
import pytest
import json
import os
import tempfile
from shared.skeletons.skeleton_registry import SkeletonRegistry


class TestSkeletonRegistration:
    """Test skeleton registration and retrieval."""

    def test_manual_registration(self):
        """Test manual registration of skeleton formats."""
        registry = SkeletonRegistry()
        joints = ["a", "b", "c"]
        bones = [("a", "b"), ("b", "c")]
        
        registry.register("test_skeleton", joints, bones)
        
        assert "test_skeleton" in registry._registry
        assert registry._registry["test_skeleton"]["joints"] == joints

    def test_register_multiple_skeletons(self):
        """Test registering multiple skeleton formats."""
        registry = SkeletonRegistry()
        
        registry.register("skeleton1", ["a", "b"], [("a", "b")])
        registry.register("skeleton2", ["x", "y", "z"], [("x", "y"), ("y", "z")])
        
        assert len(registry._registry) == 2
        assert "skeleton1" in registry._registry
        assert "skeleton2" in registry._registry

    def test_get_registered_skeleton(self):
        """Test retrieving a registered skeleton."""
        registry = SkeletonRegistry()
        joints = ["a", "b", "c"]
        bones = [("a", "b"), ("b", "c")]
        
        registry.register("test", joints, bones)
        retrieved_joints, retrieved_bones = registry.get("test")
        
        assert retrieved_joints == joints
        assert retrieved_bones == bones

    def test_get_nonexistent_skeleton_error(self):
        """Test error when retrieving non-existent skeleton."""
        registry = SkeletonRegistry()
        
        with pytest.raises(ValueError, match="not found"):
            registry.get("nonexistent")


class TestJSONSerialization:
    """Test JSON I/O operations."""

    def test_save_to_json(self):
        """Test saving skeleton to JSON file."""
        registry = SkeletonRegistry()
        joints = ["a", "b", "c"]
        bones = [("a", "b"), ("b", "c")]
        registry.register("test", joints, bones)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            registry.save_to_json("test", temp_path)
            
            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert data["name"] == "test"
            assert data["joints"] == joints
            # Bones should be converted to lists
            assert data["bones"] == [["a", "b"], ["b", "c"]]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_from_json(self):
        """Test loading skeleton from JSON file."""
        # Create a temporary JSON file
        test_data = {
            "name": "loaded_skeleton",
            "joints": ["x", "y", "z"],
            "bones": [["x", "y"], ["y", "z"]]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            registry = SkeletonRegistry()
            name = registry.load_from_json(temp_path)
            
            assert name == "loaded_skeleton"
            assert "loaded_skeleton" in registry._registry
            joints, bones = registry.get("loaded_skeleton")
            assert joints == ["x", "y", "z"]
            # Bones should be converted back to tuples
            assert bones == [("x", "y"), ("y", "z")]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_json_contains_required_fields(self):
        """Test that saved JSON contains all required fields."""
        registry = SkeletonRegistry()
        registry.register("test", ["a", "b"], [("a", "b")])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            registry.save_to_json("test", temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert "name" in data
            assert "joints" in data
            assert "bones" in data
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_nonexistent_file_error(self):
        """Test error when loading non-existent JSON file."""
        registry = SkeletonRegistry()
        
        with pytest.raises(FileNotFoundError):
            registry.load_from_json("/nonexistent/path/skeleton.json")

    def test_load_malformed_json_error(self):
        """Test error handling for malformed JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name
        
        try:
            registry = SkeletonRegistry()
            with pytest.raises(json.JSONDecodeError):
                registry.load_from_json(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_save_nonexistent_skeleton_error(self):
        """Test error when saving non-existent skeleton."""
        registry = SkeletonRegistry()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="not found"):
                registry.save_to_json("nonexistent", temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestDataConversion:
    """Test tuple-to-list and list-to-tuple conversions."""

    def test_tuple_to_list_conversion_json_save(self):
        """Test that tuples are converted to lists for JSON serialization."""
        registry = SkeletonRegistry()
        bones_input = [("a", "b"), ("b", "c"), ("c", "d")]
        registry.register("test", ["a", "b", "c", "d"], bones_input)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            registry.save_to_json("test", temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            # Should be lists, not tuples
            assert all(isinstance(bone, list) for bone in data["bones"])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_list_to_tuple_conversion_json_load(self):
        """Test that lists are converted back to tuples when loading from JSON."""
        test_data = {
            "name": "test",
            "joints": ["a", "b", "c"],
            "bones": [["a", "b"], ["b", "c"]]  # Lists
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            registry = SkeletonRegistry()
            registry.load_from_json(temp_path)
            
            joints, bones = registry.get("test")
            # Should be tuples
            assert all(isinstance(bone, tuple) for bone in bones)
            assert bones == [("a", "b"), ("b", "c")]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_roundtrip_save_and_load(self):
        """Test that save followed by load preserves skeleton data."""
        original_registry = SkeletonRegistry()
        original_joints = ["shoulder", "elbow", "wrist"]
        original_bones = [("shoulder", "elbow"), ("elbow", "wrist")]
        original_registry.register("arm", original_joints, original_bones)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            original_registry.save_to_json("arm", temp_path)
            
            # Load
            new_registry = SkeletonRegistry()
            new_registry.load_from_json(temp_path)
            
            # Verify data matches
            loaded_joints, loaded_bones = new_registry.get("arm")
            assert loaded_joints == original_joints
            assert loaded_bones == original_bones
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_skeleton(self):
        """Test registering skeleton with no bones."""
        registry = SkeletonRegistry()
        registry.register("no_bones", ["a", "b", "c"], [])
        
        joints, bones = registry.get("no_bones")
        assert len(bones) == 0

    def test_single_joint_skeleton(self):
        """Test skeleton with single joint."""
        registry = SkeletonRegistry()
        registry.register("single", ["only_one"], [])
        
        joints, bones = registry.get("single")
        assert joints == ["only_one"]
        assert bones == []

    def test_large_skeleton_registration(self):
        """Test registering large skeleton."""
        registry = SkeletonRegistry()
        joints = [f"joint_{i}" for i in range(100)]
        bones = [(f"joint_{i}", f"joint_{i+1}") for i in range(99)]
        
        registry.register("large", joints, bones)
        
        loaded_joints, loaded_bones = registry.get("large")
        assert len(loaded_joints) == 100
        assert len(loaded_bones) == 99

    def test_special_characters_in_names(self):
        """Test skeleton with special characters in joint names."""
        registry = SkeletonRegistry()
        joints = ["joint-1", "joint_2", "joint.3"]
        bones = [("joint-1", "joint_2"), ("joint_2", "joint.3")]
        
        registry.register("special", joints, bones)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            registry.save_to_json("special", temp_path)
            
            new_registry = SkeletonRegistry()
            new_registry.load_from_json(temp_path)
            
            loaded_joints, loaded_bones = new_registry.get("special")
            assert loaded_joints == joints
            assert loaded_bones == bones
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_overwrite_existing_skeleton(self):
        """Test registering with same name overwrites previous."""
        registry = SkeletonRegistry()
        
        registry.register("test", ["a", "b"], [("a", "b")])
        registry.register("test", ["x", "y", "z"], [("x", "y")])
        
        joints, bones = registry.get("test")
        assert joints == ["x", "y", "z"]
        assert bones == [("x", "y")]
