"""
Unit tests for pose_data models.

Tests the core data structures: Keypoint2D, Keypoint3D, PersonPose, FramePose, PoseSequence
Focuses on business logic, edge cases, and critical functionality.
"""
import tempfile
import unittest
from pathlib import Path

from app.models.pose_data import (
    FramePose,
    Keypoint2D,
    Keypoint3D,
    PersonPose,
    PoseSequence,
)


class TestKeypoint2D(unittest.TestCase):
    """Test Keypoint2D data structure - focus on serialization and edge cases."""

    def test_serialization_with_and_without_visibility(self):
        """Test that visibility is correctly included/excluded in serialization."""
        kp_without = Keypoint2D(x=100.5, y=200.3)
        kp_with = Keypoint2D(x=100.5, y=200.3, visibility=0.95)

        dict_without = kp_without.to_dict()
        dict_with = kp_with.to_dict()

        self.assertNotIn("visibility", dict_without)
        self.assertEqual(dict_with["visibility"], 0.95)
        self.assertEqual(kp_without.to_list(), [100.5, 200.3])

    def test_extreme_coordinate_values(self):
        """Test that extreme values are handled correctly."""
        kp = Keypoint2D(x=-1000.0, y=10000.0, visibility=0.0)
        self.assertEqual(kp.x, -1000.0)
        self.assertEqual(kp.y, 10000.0)
        self.assertEqual(kp.visibility, 0.0)

    def test_zero_coordinates(self):
        """Test edge case of zero coordinates (valid for offscreen/invisible points)."""
        kp = Keypoint2D(x=0.0, y=0.0, visibility=0.0)
        self.assertEqual(kp.to_list(), [0.0, 0.0])


class TestKeypoint3D(unittest.TestCase):
    """Test Keypoint3D data structure - focus on serialization and edge cases."""

    def test_serialization_with_and_without_visibility(self):
        """Test that visibility is correctly included/excluded in serialization."""
        kp_without = Keypoint3D(x=0.1, y=0.2, z=0.5)
        kp_with = Keypoint3D(x=0.1, y=0.2, z=0.5, visibility=0.92)

        dict_without = kp_without.to_dict()
        dict_with = kp_with.to_dict()

        self.assertNotIn("visibility", dict_without)
        self.assertEqual(dict_with["visibility"], 0.92)
        self.assertEqual(kp_without.to_list(), [0.1, 0.2, 0.5])

    def test_negative_depth_values(self):
        """Test that negative Z coordinates are handled (behind camera)."""
        kp = Keypoint3D(x=0.1, y=0.2, z=-0.5, visibility=0.8)
        self.assertEqual(kp.z, -0.5)
        self.assertEqual(kp.to_list(), [0.1, 0.2, -0.5])


class TestPersonPose(unittest.TestCase):
    """Test PersonPose data structure - focus on access patterns and edge cases."""

    def setUp(self):
        self.keypoints_2d = [
            Keypoint2D(x=100.0, y=200.0),
            Keypoint2D(x=150.0, y=250.0),
            Keypoint2D(x=200.0, y=300.0),
        ]
        self.keypoints_3d = [
            Keypoint3D(x=0.1, y=0.2, z=0.5),
            Keypoint3D(x=0.15, y=0.25, z=0.55),
            Keypoint3D(x=0.2, y=0.3, z=0.6),
        ]

    def test_get_keypoint_out_of_bounds(self):
        """Test that accessing out-of-bounds keypoint raises IndexError."""
        person = PersonPose(keypoints_2d=self.keypoints_2d)

        with self.assertRaises(IndexError):
            person.get_keypoint_2d(10)

    def test_get_keypoint_3d_returns_none_when_unavailable(self):
        """Test that get_keypoint_3d returns None when 3D data not available."""
        person = PersonPose(keypoints_2d=self.keypoints_2d)
        self.assertIsNone(person.get_keypoint_3d(0))

    def test_empty_keypoints_list(self):
        """Test that PersonPose can handle empty keypoints (detection failure)."""
        person = PersonPose(keypoints_2d=[])
        self.assertEqual(person.num_keypoints, 0)

    def test_mismatched_2d_3d_lengths(self):
        """Test behavior when 2D and 3D keypoint counts don't match."""
        # This is a real-world scenario - model might output different counts
        person = PersonPose(
            keypoints_2d=self.keypoints_2d,
            keypoints_3d=[Keypoint3D(x=0.1, y=0.2, z=0.5)]  # Only 1 instead of 3
        )

        # Should not crash - just access what's available
        self.assertEqual(person.num_keypoints, 3)
        self.assertIsNotNone(person.get_keypoint_3d(0))

        # Accessing beyond 3D length should raise IndexError
        with self.assertRaises(IndexError):
            person.get_keypoint_3d(2)

    def test_to_dict_handles_optional_fields(self):
        """Test that to_dict correctly handles optional fields."""
        person_minimal = PersonPose(keypoints_2d=self.keypoints_2d)
        person_full = PersonPose(
            keypoints_2d=self.keypoints_2d,
            keypoints_3d=self.keypoints_3d,
            person_id=42
        )

        dict_minimal = person_minimal.to_dict()
        dict_full = person_full.to_dict()

        self.assertNotIn("keypoints_3d", dict_minimal)
        self.assertNotIn("person_id", dict_minimal)

        self.assertIn("keypoints_3d", dict_full)
        self.assertEqual(dict_full["person_id"], 42)


class TestFramePose(unittest.TestCase):
    """Test FramePose data structure - focus on multi-person scenarios."""

    def setUp(self):
        self.person1 = PersonPose(
            keypoints_2d=[Keypoint2D(x=100.0, y=200.0)],
            person_id=0
        )
        self.person2 = PersonPose(
            keypoints_2d=[Keypoint2D(x=400.0, y=500.0)],
            person_id=1
        )

    def test_empty_frame_no_people_detected(self):
        """Test frame with no people detected (common in real videos)."""
        frame = FramePose(frame_number=0, people=[])
        self.assertEqual(frame.num_people, 0)

    def test_get_person_out_of_bounds(self):
        """Test that accessing non-existent person raises IndexError."""
        frame = FramePose(frame_number=0, people=[self.person1])

        with self.assertRaises(IndexError):
            frame.get_person(5)

    def test_negative_timestamp(self):
        """Test that negative timestamps are handled (shouldn't occur but should work)."""
        frame = FramePose(frame_number=0, people=[self.person1], timestamp=-0.5)
        self.assertEqual(frame.timestamp, -0.5)

    def test_to_dict_preserves_timestamp(self):
        """Test that timestamp is correctly serialized when present."""
        frame_with_time = FramePose(
            frame_number=5,
            people=[self.person1],
            timestamp=0.167
        )
        frame_without_time = FramePose(
            frame_number=5,
            people=[self.person1]
        )

        self.assertEqual(frame_with_time.to_dict()["timestamp"], 0.167)
        self.assertNotIn("timestamp", frame_without_time.to_dict())


class TestPoseSequence(unittest.TestCase):
    """Test PoseSequence data structure - focus on trajectories and edge cases."""

    def setUp(self):
        person1_frame0 = PersonPose(
            keypoints_2d=[Keypoint2D(x=100.0, y=200.0)],
            person_id=0
        )
        person1_frame1 = PersonPose(
            keypoints_2d=[Keypoint2D(x=102.0, y=202.0)],
            person_id=0
        )
        person2_frame1 = PersonPose(
            keypoints_2d=[Keypoint2D(x=300.0, y=400.0)],
            person_id=1
        )

        self.frame0 = FramePose(frame_number=0, people=[person1_frame0], timestamp=0.0)
        self.frame1 = FramePose(frame_number=1, people=[person1_frame1, person2_frame1], timestamp=0.033)

    def test_empty_sequence(self):
        """Test that empty sequence is handled correctly."""
        sequence = PoseSequence(frames=[])
        self.assertEqual(sequence.num_frames, 0)
        self.assertIsNone(sequence.duration)

    def test_duration_calculation_edge_cases(self):
        """Test duration calculation with different fps values."""
        sequence = PoseSequence(frames=[self.frame0, self.frame1], fps=30.0)
        self.assertAlmostEqual(sequence.duration, 2.0 / 30.0, places=5)

        # Zero fps should not crash (though invalid)
        sequence_zero = PoseSequence(frames=[self.frame0, self.frame1], fps=0.0)
        with self.assertRaises(ZeroDivisionError):
            _ = sequence_zero.duration

        # None fps should return None duration
        sequence_none = PoseSequence(frames=[self.frame0, self.frame1])
        self.assertIsNone(sequence_none.duration)

    def test_get_frame_out_of_bounds(self):
        """Test that accessing non-existent frame raises IndexError."""
        sequence = PoseSequence(frames=[self.frame0, self.frame1])

        with self.assertRaises(IndexError):
            sequence.get_frame(10)

    def test_get_person_trajectory_with_varying_people_count(self):
        """Test trajectory when person appears/disappears across frames."""
        # Frame 0 has 1 person, Frame 1 has 2 people
        sequence = PoseSequence(frames=[self.frame0, self.frame1])

        # Person 0 appears in both frames
        trajectory_0 = sequence.get_person_trajectory(person_index=0)
        self.assertEqual(len(trajectory_0), 2)

        # Person 1 only appears in frame 1
        trajectory_1 = sequence.get_person_trajectory(person_index=1)
        self.assertEqual(len(trajectory_1), 1)
        self.assertEqual(trajectory_1[0].keypoints_2d[0].x, 300.0)

        # Person 2 doesn't exist
        trajectory_2 = sequence.get_person_trajectory(person_index=2)
        self.assertEqual(len(trajectory_2), 0)

    def test_json_serialization_roundtrip_with_all_metadata(self):
        """Test saving to JSON and loading back preserves all data."""
        sequence = PoseSequence(
            frames=[self.frame0, self.frame1],
            fps=30.0,
            video_width=1920,
            video_height=1080,
            model_name="TestModel"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            sequence.to_json(temp_path)
            loaded_sequence = PoseSequence.from_json(temp_path)

            # Verify metadata
            self.assertEqual(loaded_sequence.num_frames, sequence.num_frames)
            self.assertEqual(loaded_sequence.fps, sequence.fps)
            self.assertEqual(loaded_sequence.video_width, sequence.video_width)
            self.assertEqual(loaded_sequence.video_height, sequence.video_height)
            self.assertEqual(loaded_sequence.model_name, sequence.model_name)

            # Verify frame data integrity
            for i in range(sequence.num_frames):
                orig_frame = sequence.get_frame(i)
                loaded_frame = loaded_sequence.get_frame(i)

                self.assertEqual(orig_frame.frame_number, loaded_frame.frame_number)
                self.assertAlmostEqual(orig_frame.timestamp, loaded_frame.timestamp, places=5)
                self.assertEqual(orig_frame.num_people, loaded_frame.num_people)

        finally:
            Path(temp_path).unlink()

    def test_json_serialization_with_minimal_data(self):
        """Test that serialization works with minimal optional fields."""
        minimal_person = PersonPose(keypoints_2d=[Keypoint2D(x=1.0, y=2.0)])
        minimal_frame = FramePose(frame_number=0, people=[minimal_person])
        sequence = PoseSequence(frames=[minimal_frame])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            sequence.to_json(temp_path)
            loaded = PoseSequence.from_json(temp_path)

            self.assertEqual(loaded.num_frames, 1)
            self.assertIsNone(loaded.fps)
            self.assertIsNone(loaded.model_name)

        finally:
            Path(temp_path).unlink()

    def test_from_dict_with_malformed_data(self):
        """Test that from_dict handles edge cases in data structure."""
        # Missing person_id (should default to None)
        data = {
            "frames": [
                {
                    "frame_number": 0,
                    "people": [
                        {
                            "keypoints_2d": [{"x": 100.0, "y": 200.0}],
                            # person_id missing
                        }
                    ]
                }
            ]
        }

        sequence = PoseSequence.from_dict(data)
        person = sequence.get_frame(0).get_person(0)
        self.assertIsNone(person.person_id)

    def test_to_dict_includes_computed_properties(self):
        """Test that to_dict includes computed properties like duration."""
        sequence = PoseSequence(
            frames=[self.frame0, self.frame1],
            fps=30.0
        )

        result = sequence.to_dict()

        self.assertIn("num_frames", result)
        self.assertIn("duration", result)
        self.assertEqual(result["num_frames"], 2)
        self.assertIsNotNone(result["duration"])


if __name__ == "__main__":
    unittest.main()
