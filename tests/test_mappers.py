"""
Unit tests for mappers.

Tests the mapper system for converting raw model outputs to PoseSequence format.
Uses dependency injection to support testing multiple mapper types.
"""
import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Protocol

from app.models.mappers import MediaPipeMapper, load_pose_data
from app.models.pose_data import (
    FramePose,
    Keypoint2D,
    Keypoint3D,
    PersonPose,
    PoseSequence,
)

try:
    from .fixtures import (
        create_alternative_format_data,
        create_mediapipe_multi_person_data,
        create_mediapipe_single_person_data,
    )
except ImportError:
    from fixtures import (
        create_alternative_format_data,
        create_mediapipe_multi_person_data,
        create_mediapipe_single_person_data,
    )


class MapperProtocol(Protocol):
    """Protocol defining the interface that all mappers should implement."""

    @staticmethod
    def from_raw_arrays(
        keypoints_2d_list: List,
        keypoints_3d_list: List,
        fps: float = None,
        video_width: int = None,
        video_height: int = None
    ) -> PoseSequence:
        """Convert raw arrays to PoseSequence."""
        ...


class TestMediaPipeMapper(unittest.TestCase):
    """Test MediaPipeMapper for converting MediaPipe output to PoseSequence."""

    def test_single_person_structure_detection(self):
        """Test that mapper correctly handles collapsed single-person structure."""
        kp_2d, kp_3d = create_mediapipe_single_person_data()

        sequence = MediaPipeMapper.from_raw_arrays(
            kp_2d, kp_3d, fps=30.0, video_width=1920, video_height=1080
        )

        # Verify basic properties
        self.assertEqual(sequence.num_frames, 2)
        self.assertEqual(sequence.fps, 30.0)
        self.assertEqual(sequence.video_width, 1920)
        self.assertEqual(sequence.video_height, 1080)
        self.assertEqual(sequence.model_name, "MediaPipe")

        # Verify frame structure
        frame0 = sequence.get_frame(0)
        self.assertEqual(frame0.num_people, 1)
        self.assertEqual(frame0.frame_number, 0)
        self.assertIsNotNone(frame0.timestamp)

        # Verify person data
        person = frame0.get_person(0)
        self.assertEqual(person.person_id, 0)
        self.assertEqual(person.num_keypoints, 5)

    def test_single_person_keypoint_values(self):
        """Test that keypoint values are correctly parsed for single person."""
        kp_2d, kp_3d = create_mediapipe_single_person_data()

        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d)

        # Check first keypoint of first frame
        person = sequence.get_frame(0).get_person(0)

        kp2d_0 = person.get_keypoint_2d(0)
        self.assertEqual(kp2d_0.x, 100.0)
        self.assertEqual(kp2d_0.y, 200.0)

        kp3d_0 = person.get_keypoint_3d(0)
        self.assertEqual(kp3d_0.x, 0.1)
        self.assertEqual(kp3d_0.y, 0.2)
        self.assertEqual(kp3d_0.z, 0.5)

    def test_multi_person_structure_detection(self):
        """Test that mapper correctly handles multi-person structure."""
        kp_2d, kp_3d = create_mediapipe_multi_person_data()

        sequence = MediaPipeMapper.from_raw_arrays(
            kp_2d, kp_3d, fps=30.0, video_width=1920, video_height=1080
        )

        # Verify basic properties
        self.assertEqual(sequence.num_frames, 2)
        self.assertEqual(sequence.model_name, "MediaPipe")

        # Verify frame structure - should have 2 people
        frame0 = sequence.get_frame(0)
        self.assertEqual(frame0.num_people, 2)

        # Verify both people are tracked
        person0 = frame0.get_person(0)
        person1 = frame0.get_person(1)

        self.assertEqual(person0.person_id, 0)
        self.assertEqual(person1.person_id, 1)
        self.assertEqual(person0.num_keypoints, 5)
        self.assertEqual(person1.num_keypoints, 5)

    def test_multi_person_keypoint_values(self):
        """Test that keypoint values are correctly parsed for multiple people."""
        kp_2d, kp_3d = create_mediapipe_multi_person_data()

        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d)

        frame0 = sequence.get_frame(0)

        # Check first person's first keypoint
        person0 = frame0.get_person(0)
        kp2d_p0 = person0.get_keypoint_2d(0)
        self.assertEqual(kp2d_p0.x, 100.0)
        self.assertEqual(kp2d_p0.y, 200.0)

        # Check second person's first keypoint (should be different)
        person1 = frame0.get_person(1)
        kp2d_p1 = person1.get_keypoint_2d(0)
        self.assertEqual(kp2d_p1.x, 400.0)
        self.assertEqual(kp2d_p1.y, 500.0)

    def test_timestamp_calculation(self):
        """Test that timestamps are correctly calculated from fps."""
        kp_2d, kp_3d = create_mediapipe_single_person_data()

        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d, fps=30.0)

        frame0 = sequence.get_frame(0)
        frame1 = sequence.get_frame(1)

        self.assertAlmostEqual(frame0.timestamp, 0.0, places=5)
        self.assertAlmostEqual(frame1.timestamp, 1.0 / 30.0, places=5)

    def test_timestamp_none_without_fps(self):
        """Test that timestamps are None when fps is not provided."""
        kp_2d, kp_3d = create_mediapipe_single_person_data()

        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d)

        frame0 = sequence.get_frame(0)
        self.assertIsNone(frame0.timestamp)

    def test_trajectory_across_frames(self):
        """Test tracking a person's movement across frames."""
        kp_2d, kp_3d = create_mediapipe_single_person_data()

        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d)
        trajectory = sequence.get_person_trajectory(person_index=0)

        self.assertEqual(len(trajectory), 2)

        # Verify movement (coordinates should change slightly)
        kp_frame0 = trajectory[0].get_keypoint_2d(0)
        kp_frame1 = trajectory[1].get_keypoint_2d(0)

        self.assertEqual(kp_frame0.x, 100.0)
        self.assertEqual(kp_frame1.x, 102.0)  # Moved slightly

    def test_from_json_files(self):
        """Test loading from JSON files."""
        kp_2d, kp_3d = create_mediapipe_single_person_data()

        # Create temporary JSON files
        with tempfile.NamedTemporaryFile(mode='w', suffix='_2d.json', delete=False) as f:
            json.dump(kp_2d, f)
            temp_2d_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='_3d.json', delete=False) as f:
            json.dump(kp_3d, f)
            temp_3d_path = f.name

        try:
            sequence = MediaPipeMapper.from_json_files(
                temp_2d_path,
                temp_3d_path,
                fps=30.0,
                video_width=1920,
                video_height=1080
            )

            # Verify it loaded correctly
            self.assertEqual(sequence.num_frames, 2)
            self.assertEqual(sequence.fps, 30.0)
            self.assertEqual(sequence.model_name, "MediaPipe")

            frame0 = sequence.get_frame(0)
            self.assertEqual(frame0.num_people, 1)

        finally:
            # Clean up
            Path(temp_2d_path).unlink()
            Path(temp_3d_path).unlink()

    def test_data_shape_validation(self):
        """Test that mapper produces correct data shapes."""
        kp_2d, kp_3d = create_mediapipe_multi_person_data()

        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d)

        # Verify structure consistency across all frames
        for frame_idx in range(sequence.num_frames):
            frame = sequence.get_frame(frame_idx)

            # Each frame should have the same number of people
            self.assertEqual(frame.num_people, 2)

            for person_idx in range(frame.num_people):
                person = frame.get_person(person_idx)

                # Each person should have same number of 2D and 3D keypoints
                self.assertEqual(person.num_keypoints, 5)
                self.assertEqual(len(person.keypoints_3d), 5)

                # Verify all keypoints have proper structure
                for kp_idx in range(person.num_keypoints):
                    kp_2d = person.get_keypoint_2d(kp_idx)
                    kp_3d = person.get_keypoint_3d(kp_idx)

                    # 2D keypoint should have x, y
                    self.assertIsNotNone(kp_2d.x)
                    self.assertIsNotNone(kp_2d.y)

                    # 3D keypoint should have x, y, z
                    self.assertIsNotNone(kp_3d.x)
                    self.assertIsNotNone(kp_3d.y)
                    self.assertIsNotNone(kp_3d.z)

    def test_empty_frame_data(self):
        """Test that mapper handles frames with no people detected."""
        # Empty frame structure: [frame][empty people list]
        kp_2d = [[]]  # 1 frame, no people
        kp_3d = [[]]

        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d, fps=30.0)

        self.assertEqual(sequence.num_frames, 1)
        frame = sequence.get_frame(0)
        self.assertEqual(frame.num_people, 0)

    def test_mismatched_2d_3d_frame_counts(self):
        """Test that mismatched 2D/3D frame counts doesn't crash (uses min length)."""
        kp_2d, _ = create_mediapipe_single_person_data()
        # Create matching structure but with fewer frames
        kp_3d = [[[0.1, 0.2, 0.5], [0.15, 0.25, 0.55], [0.2, 0.3, 0.6], [0.25, 0.35, 0.65], [0.3, 0.4, 0.7]]]  # Only 1 frame

        # zip() will stop at the shorter sequence
        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d)

        # Should only process 1 frame (min of 2 and 1)
        self.assertEqual(sequence.num_frames, 1)

    def test_structure_detection_edge_case_empty_array(self):
        """Test structure detection with completely empty arrays."""
        kp_2d = []
        kp_3d = []

        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d)

        self.assertEqual(sequence.num_frames, 0)
        self.assertEqual(len(sequence.frames), 0)

    def test_structure_detection_with_single_keypoint(self):
        """Test edge case where person has only 1 keypoint."""
        kp_2d = [[[100.0, 200.0]]]  # 1 frame, 1 person, 1 keypoint
        kp_3d = [[[0.1, 0.2, 0.5]]]

        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d)

        frame = sequence.get_frame(0)
        self.assertEqual(frame.num_people, 1)
        person = frame.get_person(0)
        self.assertEqual(person.num_keypoints, 1)

    def test_mixed_structure_across_frames(self):
        """Test that each frame's structure is independently detected."""
        # Frame 0: single person (collapsed), Frame 1: multiple people
        kp_2d = [
            [[100.0, 200.0], [150.0, 250.0]],  # Collapsed structure
            [[[200.0, 300.0]], [[400.0, 500.0]]]  # Multi-person structure
        ]
        kp_3d = [
            [[0.1, 0.2, 0.5], [0.15, 0.25, 0.55]],
            [[[0.2, 0.3, 0.6]], [[0.4, 0.5, 0.9]]]
        ]

        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d)

        # Frame 0 should have 1 person (collapsed)
        self.assertEqual(sequence.get_frame(0).num_people, 1)

        # Frame 1 should have 2 people
        self.assertEqual(sequence.get_frame(1).num_people, 2)

    def test_zero_fps_handling(self):
        """Test that zero fps is handled (though invalid)."""
        kp_2d, kp_3d = create_mediapipe_single_person_data()

        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d, fps=0.0)

        # Should not crash during creation
        self.assertEqual(sequence.fps, 0.0)

        # First frame timestamp should be 0/0 which will raise ZeroDivisionError
        # but only when accessed, not during creation
        frame = sequence.get_frame(0)
        # We can't compute timestamp with fps=0, but it shouldn't crash the mapper

    def test_negative_fps_handling(self):
        """Test that negative fps is handled (though invalid)."""
        kp_2d, kp_3d = create_mediapipe_single_person_data()

        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d, fps=-30.0)

        self.assertEqual(sequence.fps, -30.0)
        # First frame timestamp will be 0/-30 = -0.0 (which equals 0)
        # Second frame will be negative
        frame0 = sequence.get_frame(0)
        frame1 = sequence.get_frame(1)
        # frame0 is 0/-30 = -0.0 (equals 0 in Python)
        self.assertEqual(frame0.timestamp, -0.0)
        # frame1 is 1/-30 = negative
        self.assertLess(frame1.timestamp, 0)


class TestLoadPoseData(unittest.TestCase):
    """Test the universal load_pose_data function."""

    def test_load_mediapipe_model(self):
        """Test loading with model_type='mediapipe'."""
        kp_2d, kp_3d = create_mediapipe_single_person_data()

        # Create temporary JSON files
        with tempfile.NamedTemporaryFile(mode='w', suffix='_2d.json', delete=False) as f:
            json.dump(kp_2d, f)
            temp_2d_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='_3d.json', delete=False) as f:
            json.dump(kp_3d, f)
            temp_3d_path = f.name

        try:
            sequence = load_pose_data(
                temp_2d_path,
                temp_3d_path,
                model_type="mediapipe",
                fps=30.0
            )

            self.assertEqual(sequence.model_name, "MediaPipe")
            self.assertEqual(sequence.num_frames, 2)

        finally:
            Path(temp_2d_path).unlink()
            Path(temp_3d_path).unlink()

    def test_mediapipe_requires_3d_file(self):
        """Test that MediaPipe loader requires 3D file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='_2d.json', delete=False) as f:
            json.dump([[]], f)
            temp_2d_path = f.name

        try:
            with self.assertRaises(ValueError) as context:
                load_pose_data(temp_2d_path, None, model_type="mediapipe")

            self.assertIn("requires both 2D and 3D", str(context.exception))

        finally:
            Path(temp_2d_path).unlink()

    def test_unsupported_model_type(self):
        """Test that unsupported model type raises error."""
        with self.assertRaises(ValueError) as context:
            load_pose_data("dummy.json", model_type="unknown_model")

        self.assertIn("Unsupported model type", str(context.exception))


class TestMapperExtensibility(unittest.TestCase):
    """Test that the mapper system is extensible via dependency injection."""

    def test_custom_mapper_injection(self):
        """Test that we can inject a custom mapper for a different model format."""

        # Define a custom mapper for an alternative format
        class AlternativeMapper:
            """Example mapper for a different model's output format."""

            @staticmethod
            def from_raw_data(data: dict) -> PoseSequence:
                """Convert alternative format to PoseSequence."""
                frames = []

                for frame_data in data["frames"]:
                    people = []

                    for detection in frame_data["detections"]:
                        keypoints_2d = [
                            Keypoint2D(x=kp[0], y=kp[1], visibility=detection["confidence"][i])
                            for i, kp in enumerate(detection["keypoints_2d"])
                        ]

                        keypoints_3d = [
                            Keypoint3D(x=kp[0], y=kp[1], z=kp[2], visibility=detection["confidence"][i])
                            for i, kp in enumerate(detection["keypoints_3d"])
                        ]

                        people.append(PersonPose(
                            keypoints_2d=keypoints_2d,
                            keypoints_3d=keypoints_3d,
                            person_id=detection["person_id"]
                        ))

                    frames.append(FramePose(
                        frame_number=frame_data["frame_id"],
                        people=people,
                        timestamp=frame_data["timestamp"]
                    ))

                return PoseSequence(
                    frames=frames,
                    fps=data["metadata"]["fps"],
                    video_width=data["metadata"]["width"],
                    video_height=data["metadata"]["height"],
                    model_name=data["metadata"]["model"]
                )

        # Test the custom mapper
        alt_data = create_alternative_format_data()
        sequence = AlternativeMapper.from_raw_data(alt_data)

        # Verify it works
        self.assertEqual(sequence.num_frames, 2)
        self.assertEqual(sequence.model_name, "AlternativeModel")
        self.assertEqual(sequence.fps, 30.0)

        frame0 = sequence.get_frame(0)
        self.assertEqual(frame0.num_people, 1)

        person = frame0.get_person(0)
        self.assertEqual(person.num_keypoints, 3)

        # Verify visibility was set from confidence
        kp = person.get_keypoint_2d(0)
        self.assertEqual(kp.visibility, 0.9)

    def test_mapper_factory_pattern(self):
        """Test using a factory pattern for selecting mappers."""

        def get_mapper(model_type: str):
            """Factory function to get appropriate mapper."""
            mappers = {
                "mediapipe": MediaPipeMapper,
                # Could add more mappers here:
                # "mmpose": MMPoseMapper,
                # "openpose": OpenPoseMapper,
            }

            if model_type not in mappers:
                raise ValueError(f"Unknown model type: {model_type}")

            return mappers[model_type]

        # Test factory
        mapper = get_mapper("mediapipe")
        self.assertEqual(mapper, MediaPipeMapper)

        # Test with actual data
        kp_2d, kp_3d = create_mediapipe_single_person_data()
        sequence = mapper.from_raw_arrays(kp_2d, kp_3d)

        self.assertEqual(sequence.model_name, "MediaPipe")
        self.assertEqual(sequence.num_frames, 2)


class TestMapperConsistency(unittest.TestCase):
    """Test that all mappers produce consistent output structure."""

    def test_mediapipe_output_structure(self):
        """Test MediaPipe mapper produces valid PoseSequence."""
        kp_2d, kp_3d = create_mediapipe_single_person_data()
        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d, fps=30.0)

        # All mappers should produce these properties
        self.assertIsInstance(sequence, PoseSequence)
        self.assertIsNotNone(sequence.frames)
        self.assertGreater(sequence.num_frames, 0)
        self.assertIsNotNone(sequence.model_name)

        # Frames should have consistent structure
        for frame in sequence.frames:
            self.assertIsInstance(frame, FramePose)
            self.assertIsNotNone(frame.people)

            for person in frame.people:
                self.assertIsInstance(person, PersonPose)
                self.assertIsNotNone(person.keypoints_2d)
                self.assertGreater(len(person.keypoints_2d), 0)

    def test_mapper_roundtrip_serialization(self):
        """Test that mapper output can be serialized and deserialized."""
        kp_2d, kp_3d = create_mediapipe_multi_person_data()
        sequence = MediaPipeMapper.from_raw_arrays(kp_2d, kp_3d, fps=30.0)

        # Convert to dict
        seq_dict = sequence.to_dict()

        # Reconstruct from dict
        reloaded = PoseSequence.from_dict(seq_dict)

        # Verify integrity
        self.assertEqual(sequence.num_frames, reloaded.num_frames)
        self.assertEqual(sequence.fps, reloaded.fps)
        self.assertEqual(sequence.model_name, reloaded.model_name)

        # Verify frame-level data
        for i in range(sequence.num_frames):
            orig_frame = sequence.get_frame(i)
            reload_frame = reloaded.get_frame(i)

            self.assertEqual(orig_frame.num_people, reload_frame.num_people)


if __name__ == "__main__":
    unittest.main()
