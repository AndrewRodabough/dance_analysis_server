"""
Ballroom dance technique data model.

This module provides a clean, extensible data model for representing dancer states,
steps (transitions), and figures. It supports both ideal syllabus technique and
measured/annotated real performance.

Core concepts:
  - DancerState: A complete snapshot of a dancer at a moment in time
  - Step: A transition from one DancerState to another
  - Figure: A sequence of Steps with timing information
  - Performance: Real or ideal performance data for comparison and analysis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Direction(Enum):
    """
    Named directions as convenient constants.

    These provide sensible defaults relative to Line of Dance (LOD).
    For arbitrary directions, use degrees directly (0-360) in Placement.direction.

    Degrees: 0° = Forward (LOD), 90° = Right, 180° = Back, 270° = Left
    """
    FORWARD = 0
    DIAG_FWD_RIGHT = 45
    RIGHT = 90
    DIAG_BACK_RIGHT = 135
    BACK = 180
    DIAG_BACK_LEFT = 225
    LEFT = 270
    DIAG_FWD_LEFT = 315


class FootContact(Enum):
    """Which part of the foot is in contact with the floor."""
    HEEL = "heel"
    BALL = "ball"
    TOE = "toe"


@dataclass
class Placement:
    """Where a foot is positioned relative to the body or a reference point."""
    direction: float  # Direction in degrees (0-360). Can use Direction enum values or arbitrary degrees.
    distance: float   # Normalized step size (e.g., 1.0 = standard step width)

    def __post_init__(self):
        if self.distance < 0:
            raise ValueError(f"distance must be non-negative, got {self.distance}")

        # Convert Direction enum to float if needed
        if isinstance(self.direction, Direction):
            self.direction = float(self.direction.value)
        elif not isinstance(self.direction, (int, float)):
            raise TypeError(f"direction must be a float or Direction enum, got {type(self.direction)}")

        # Normalize direction to [0, 360)
        self.direction = self.direction % 360.0

@dataclass
class LegState:
    knee_angle: float
    hip_angle: float
    turnout: float

@dataclass
class HipState:
    rotation: float
    tilt: float
    forward: float

@dataclass
class FootState:
    """Complete state of a single foot."""
    placement: Placement
    alignment: float      # Foot direction (degrees, 0-360)
    weight: float         # Weight distribution (0.0-1.0)
    contact: FootContact

    def __post_init__(self):
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0.0, 1.0], got {self.weight}")
        if not 0.0 <= self.alignment < 360.0:
            raise ValueError(f"alignment must be in [0.0, 360.0), got {self.alignment}")


@dataclass
class BodyState:
    """Complete state of the dancer's body (torso, frame)."""
    alignment: float      # Facing direction (degrees, 0-360)
    sway: float  # degrees
    rotation: float       # Torso rotation relative to alignment (degrees, for CBM/CBMP)

    def __post_init__(self):
        if not 0.0 <= self.alignment < 360.0:
            raise ValueError(f"alignment must be in [0.0, 360.0), got {self.alignment}")
        if not -1.0 <= self.sway <= 1.0:
            raise ValueError(f"sway must be in [-1.0, 1.0], got {self.sway}")

@dataclass
class ArmState:
    shoulder_angle: float   # elevation from torso (degrees)
    shoulder_rotation: float  # forward/back rotation
    elbow_angle: float      # 0 = straight, positive = bent
    wrist_angle: float      # optional styling/detail
    hand_height: float      # relative vertical position

@dataclass
class DancerState:
    """
    Complete snapshot of a dancer at a moment in time.

    Includes positions, alignments, and weight distribution for both feet,
    plus body orientation and posture.
    """

    body: BodyState
    left_foot: FootState
    right_foot: FootState
    left_leg: LegState
    right_leg: LegState
    left_arm: ArmState
    right_arm: ArmState
    hips: HipState

    def __post_init__(self):
        # Validate that weights sum to 1.0 (allowing small floating-point error)
        total_weight = self.left_foot.weight + self.right_foot.weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"left.weight + right.weight must equal 1.0, "
                f"got {self.left_foot.weight} + {self.right_foot.weight} = {total_weight}"
            )

@dataclass
class Keyframe:
    """A snapshot of DancerState at a specific point in a Step's progression."""
    t: float          # Normalized time (0.0 at start, 1.0 at end)
    state: DancerState

    def __post_init__(self):
        if not 0.0 <= self.t <= 1.0:
            raise ValueError(f"t must be in [0.0, 1.0], got {self.t}")


@dataclass
class Step:
    """
    A transition from one DancerState to another.

    A Step represents the movement of one foot (or synchronized movement of both feet),
    including all intermediate states if keyframes are provided. Steps are the building
    blocks of Figures and are used for analysis, comparison, and feedback.
    """
    start: DancerState
    end: DancerState
    duration: float                    # Duration in beats
    keyframes: List[Keyframe] = field(default_factory=list)

    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")

        # Validate keyframes are ordered and within [0, 1]
        if self.keyframes:
            for i, kf in enumerate(self.keyframes):
                if i > 0 and kf.t <= self.keyframes[i - 1].t:
                    raise ValueError(
                        f"keyframes must be strictly increasing in time, "
                        f"got t={self.keyframes[i - 1].t} then t={kf.t}"
                    )


@dataclass
class Figure:
    """
    A named sequence of Steps with optional timing information.

    Figures are the choreographic building blocks (e.g., "Feather Step", "Natural Turn").
    They combine multiple Steps with timing notation (e.g., S, Q, Q for Slow, Quick, Quick).
    """
    name: str
    steps: List[Step]
    tags: Optional[List[str]] = None    # e.g., ["standard", "waltz", "basic"]
    total_beats: int = 0 # Number of beats in a figure

    def __post_init__(self):
        if not self.name:
            raise ValueError("Figure name cannot be empty")
        if not self.steps:
            raise ValueError("Figure must contain at least one Step")
        if abs(sum(x.duration for x in self.steps) - self.total_beats) > 1e-6:
            raise ValueError("Duration of each step in a figure must sum to match counts in a figure")


@dataclass
class Performance:
    """
    A sequence of Figures representing either ideal or real performance.

    Used for storing both syllabus technique (ideal) and annotated real performance
    data for comparison and ML-based feedback.
    """
    figures: List[Figure]
    source: str  # e.g., "ideal", "video", "manual", "calculated"

    def __post_init__(self):
        if not self.figures:
            raise ValueError("Performance must contain at least one Step")
        if not self.source:
            raise ValueError("Performance source cannot be empty")


@dataclass
class StepComparison:
    """
    Comparison between an expected (ideal) Step and an actual (performed) Step.

    Used for analysis, feedback generation, and ML training. Differences can be
    computed from the start/end states.
    """
    expected: Step
    actual: Step
    notes: Optional[str] = None

# ============================================================================
# Example: Simple Weight Transfer Step
# ============================================================================

def example_weight_transfer_step() -> Step:
    """
    Construct a simple step: weight transfer from left foot to right foot.

    This demonstrates the core structure:
    - Start: Weight on left foot (left: 0.9, right: 0.1)
    - End:   Weight on right foot (left: 0.1, right: 0.9)
    - Includes full body state: legs, arms, hips (typical for Latin dance)
    """
    # Starting position: weight predominantly on left foot
    start_state = DancerState(
        body=BodyState(
            alignment=0.0,
            sway=0.0,
            rotation=0.0,
        ),
        left_foot=FootState(
            placement=Placement(direction=Direction.FORWARD.value, distance=1.0),
            alignment=0.0,
            weight=0.9,
            contact=FootContact.BALL,
        ),
        right_foot=FootState(
            placement=Placement(direction=Direction.FORWARD.value, distance=0.5),
            alignment=0.0,
            weight=0.1,
            contact=FootContact.BALL,
        ),
        left_leg=LegState(
            knee_angle=15.0,  # Slight bend
            hip_angle=0.0,
            turnout=0.0,
        ),
        right_leg=LegState(
            knee_angle=25.0,  # Slightly more bent (supporting)
            hip_angle=0.0,
            turnout=0.0,
        ),
        left_arm=ArmState(
            shoulder_angle=0.0,
            shoulder_rotation=0.0,
            elbow_angle=90.0,
            wrist_angle=0.0,
            hand_height=0.5,
        ),
        right_arm=ArmState(
            shoulder_angle=0.0,
            shoulder_rotation=0.0,
            elbow_angle=90.0,
            wrist_angle=0.0,
            hand_height=0.5,
        ),
        hips=HipState(
            rotation=0.0,
            tilt=0.0,
            forward=0.0,
        ),
    )

    # Ending position: weight transferred to right foot
    end_state = DancerState(
        body=BodyState(
            alignment=0.0,
            sway=0.0,
            rotation=0.0,
        ),
        left_foot=FootState(
            placement=Placement(direction=Direction.FORWARD.value, distance=0.5),
            alignment=0.0,
            weight=0.1,
            contact=FootContact.BALL,
        ),
        right_foot=FootState(
            placement=Placement(direction=Direction.FORWARD.value, distance=1.0),
            alignment=0.0,
            weight=0.9,
            contact=FootContact.BALL,
        ),
        left_leg=LegState(
            knee_angle=25.0,  # More bent (supporting)
            hip_angle=0.0,
            turnout=0.0,
        ),
        right_leg=LegState(
            knee_angle=15.0,  # Slight bend
            hip_angle=0.0,
            turnout=0.0,
        ),
        left_arm=ArmState(
            shoulder_angle=0.0,
            shoulder_rotation=0.0,
            elbow_angle=90.0,
            wrist_angle=0.0,
            hand_height=0.5,
        ),
        right_arm=ArmState(
            shoulder_angle=0.0,
            shoulder_rotation=0.0,
            elbow_angle=90.0,
            wrist_angle=0.0,
            hand_height=0.5,
        ),
        hips=HipState(
            rotation=0.0,
            tilt=0.0,
            forward=0.0,
        ),
    )

    # Create the step with intermediate keyframe showing mid-transfer
    mid_state = DancerState(
        body=BodyState(
            alignment=0.0,
            sway=0.0,
            rotation=0.0,
        ),
        left_foot=FootState(
            placement=Placement(direction=Direction.FORWARD.value, distance=0.75),
            alignment=0.0,
            weight=0.5,
            contact=FootContact.BALL,
        ),
        right_foot=FootState(
            placement=Placement(direction=Direction.FORWARD.value, distance=0.75),
            alignment=0.0,
            weight=0.5,
            contact=FootContact.BALL,
        ),
        left_leg=LegState(
            knee_angle=20.0,  # Equal bend during transfer
            hip_angle=0.0,
            turnout=0.0,
        ),
        right_leg=LegState(
            knee_angle=20.0,  # Equal bend during transfer
            hip_angle=0.0,
            turnout=0.0,
        ),
        left_arm=ArmState(
            shoulder_angle=0.0,
            shoulder_rotation=0.0,
            elbow_angle=90.0,
            wrist_angle=0.0,
            hand_height=0.5,
        ),
        right_arm=ArmState(
            shoulder_angle=0.0,
            shoulder_rotation=0.0,
            elbow_angle=90.0,
            wrist_angle=0.0,
            hand_height=0.5,
        ),
        hips=HipState(
            rotation=0.0,
            tilt=0.0,
            forward=0.0,
        ),
    )

    step = Step(
        start=start_state,
        end=end_state,
        duration=1.0,  # One beat
        keyframes=[Keyframe(t=0.5, state=mid_state)],
    )

    return step


if __name__ == "__main__":
    # Demonstrate the model with a simple example
    step = example_weight_transfer_step()
    print(f"Step created: {step.duration} beat(s)")
    print(f"Start state - Left weight: {step.start.left_foot.weight}, Right weight: {step.start.right_foot.weight}")
    print(f"End state   - Left weight: {step.end.left_foot.weight}, Right weight: {step.end.right_foot.weight}")
    print(f"Keyframes: {len(step.keyframes)}")
