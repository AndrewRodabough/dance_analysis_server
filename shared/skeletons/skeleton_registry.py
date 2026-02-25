import json
import os
from typing import List, Tuple, Dict, Any

class SkeletonRegistry:
    """
    Central store for skeleton definitions.
    Loads/Saves formats (COCO, H36M, OpenPose) to JSON.
    """
    def __init__(self):
        # Internal cache of loaded configs
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, joints: List[str], bones: List[Tuple[str, str]]):
        """Manually register a format in memory"""
        self._registry[name] = {
            "joints": joints,
            "bones": bones
        }

    def save_to_json(self, name: str, file_path: str):
        """Exports a registered skeleton to a JSON file"""
        if name not in self._registry:
            raise ValueError(f"Skeleton '{name}' not found.")
        
        data = {
            "name": name,
            "joints": self._registry[name]["joints"],
            # JSON doesn't support tuples, convert to lists
            "bones": [list(b) for b in self._registry[name]["bones"]]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved '{name}' to {file_path}")

    def load_from_json(self, file_path: str):
        """Loads a JSON file and registers it"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        name = data.get("name", "unknown")
        joints = data["joints"]
        # Convert lists back to tuples
        bones = [tuple(b) for b in data["bones"]]
        
        self.register(name, joints, bones)
        print(f"Loaded '{name}' from JSON.")
        return name

    def get(self, name: str):
        """
        Returns the arguments needed to instantiate VectorizedSkeleton
        """
        if name not in self._registry:
            raise ValueError(f"Skeleton '{name}' not found in registry.")
            
        config = self._registry[name]
        return config["joints"], config["bones"]