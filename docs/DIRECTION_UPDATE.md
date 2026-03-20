# Direction System Update

## Overview

The `Direction` enum has been redesigned to serve as a **convenient set of named constants** while supporting **arbitrary degree-based directions** as the primary mechanism.

This change makes the system more flexible for:
- Precise angle measurements from pose estimation
- Custom choreography with non-standard angles
- Continuous interpolation between directions
- Integration with motion capture data

## Key Changes

### Before
```python
# Direction was an Enum with fixed values
class Direction(Enum):
    FORWARD = 0
    DIAG_FWD_RIGHT = 45
    # ...

# Placement required Direction enum
placement = Placement(direction=Direction.FORWARD, distance=1.0)
```

### After
```python
# Direction is still an Enum, but now for convenient defaults
class Direction(Enum):
    FORWARD = 0
    DIAG_FWD_RIGHT = 45
    # ...

# Placement accepts degrees directly (preferred)
placement = Placement(direction=0.0, distance=1.0)      # ✓ Degrees
placement = Placement(direction=45.5, distance=1.0)     # ✓ Any float
placement = Placement(direction=Direction.FORWARD.value, distance=1.0)  # ✓ Enum for reference

# Direction wrapping is automatic
placement = Placement(direction=405, distance=1.0)  # Becomes 45°
placement = Placement(direction=-45, distance=1.0)  # Becomes 315°
```

## Using the Direction System

### Option 1: Named Constants (Recommended for Common Directions)

Use `Direction` enum values when you need standard compass directions:

```python
from app.models.technique import Direction, Placement, FootState

# Use Direction enum values for clarity
forward = Placement(direction=Direction.FORWARD.value, distance=1.0)
diagonal = Placement(direction=Direction.DIAG_FWD_RIGHT.value, distance=0.7)
back = Placement(direction=Direction.BACK.value, distance=1.0)

foot = FootState(
    placement=forward,
    alignment=0.0,
    weight=0.5,
    contact=FootContact.BALL,
)
```

**Available Named Constants:**

| Name | Degrees | Use Case |
|------|---------|----------|
| `FORWARD` | 0° | Along Line of Dance (LOD) |
| `DIAG_FWD_RIGHT` | 45° | Forward-right diagonal |
| `RIGHT` | 90° | Perpendicular right |
| `DIAG_BACK_RIGHT` | 135° | Back-right diagonal |
| `BACK` | 180° | Opposite direction of LOD |
| `DIAG_BACK_LEFT` | 225° | Back-left diagonal |
| `LEFT` | 270° | Perpendicular left |
| `DIAG_FWD_LEFT` | 315° | Forward-left diagonal |

### Option 2: Raw Degrees (Recommended for Measurements)

Use raw degrees directly when working with pose estimation or precise measurements:

```python
# From pose estimation output
foot_angle = 32.5  # degrees from video analysis

placement = Placement(direction=foot_angle, distance=1.0)
```

### Option 3: Hybrid (Use Both)

You can mix approaches in the same code:

```python
from app.models.technique import Direction, Placement

# Standard position
start_foot = Placement(direction=Direction.FORWARD.value, distance=1.0)

# Measured from video (precise)
mid_foot = Placement(direction=22.3, distance=1.2)

# Another standard
end_foot = Placement(direction=Direction.BACK.value, distance=0.5)
```

## Direction Normalization

All directions are automatically normalized to the range **[0°, 360°)**:

```python
from app.models.technique import Placement

# These all work and are normalized automatically
p1 = Placement(direction=0, distance=1.0)         # → 0°
p2 = Placement(direction=360, distance=1.0)       # → 0° (wraps)
p3 = Placement(direction=405, distance=1.0)       # → 45° (wraps)
p4 = Placement(direction=-45, distance=1.0)       # → 315° (wraps)
p5 = Placement(direction=720.5, distance=1.0)     # → 0.5° (wraps)

# Works with floats for precision
p6 = Placement(direction=45.5, distance=1.0)      # → 45.5°
p7 = Placement(direction=157.25, distance=1.0)    # → 157.25°
```

## Coordinate System Reference

The direction system uses a standard mathematical convention relative to Line of Dance:

```
                    0° (FORWARD / LOD)
                           ↑
                           |
                           |
    270° (LEFT) ←──────────●──────────→ 90° (RIGHT)
                           |
                           |
                           ↓
                   180° (BACK / ALOD)

Diagonals:
  45° = Forward-Right        225° = Back-Left
  135° = Back-Right          315° = Forward-Left
```

## Practical Examples

### Example 1: Precise Motion Capture Data

```python
from app.models.technique import Placement, FootState, FootContact

# From motion capture system
measured_angle = 23.7  # degrees

foot = FootState(
    placement=Placement(direction=measured_angle, distance=1.0),
    alignment=0.0,
    weight=0.5,
    contact=FootContact.BALL,
)
```

### Example 2: Choreographed Step Sequence

```python
from app.models.technique import Direction, Placement

# Feather step: forward, diagonal, forward
steps = [
    Placement(direction=Direction.FORWARD.value, distance=1.0),
    Placement(direction=Direction.DIAG_FWD_RIGHT.value, distance=0.7),
    Placement(direction=Direction.FORWARD.value, distance=1.0),
]
```

### Example 3: Custom Angle (Non-Standard Direction)

```python
from app.models.technique import Placement

# Choreography with 60° angle
placement = Placement(direction=60.0, distance=1.0)
```

### Example 4: Interpolation Between Angles

```python
from app.models.technique import Placement

# Interpolate smoothly between two directions
start_angle = 0.0
end_angle = 90.0
interpolation_factor = 0.5

current_angle = start_angle + (end_angle - start_angle) * interpolation_factor
placement = Placement(direction=current_angle, distance=1.0)  # → 45°
```

## Type Flexibility

The `Placement.direction` field accepts multiple input types:

| Type | Example | Notes |
|------|---------|-------|
| `int` | `Placement(direction=45, ...)` | Automatically converted to float |
| `float` | `Placement(direction=45.5, ...)` | Preferred for precision |
| `Direction` enum | `Placement(direction=Direction.FORWARD.value, ...)` | Use `.value` to extract degrees |

```python
from app.models.technique import Direction, Placement

# All equivalent
p1 = Placement(direction=45, distance=1.0)
p2 = Placement(direction=45.0, distance=1.0)
p3 = Placement(direction=Direction.DIAG_FWD_RIGHT.value, distance=1.0)

# Invalid (will raise TypeError)
p4 = Placement(direction=Direction.FORWARD, distance=1.0)  # ✗ Don't pass enum directly
# Use .value: direction=Direction.FORWARD.value ✓
```

## Migration Guide

If you have existing code using `Direction` enums:

### Old Code
```python
from app.models.technique import Direction, Placement

placement = Placement(direction=Direction.FORWARD, distance=1.0)
```

### New Code
```python
from app.models.technique import Direction, Placement

# Option 1: Extract the value
placement = Placement(direction=Direction.FORWARD.value, distance=1.0)

# Option 2: Use raw degrees
placement = Placement(direction=0.0, distance=1.0)

# Option 3: Use both (mix and match)
standard_placement = Placement(direction=Direction.FORWARD.value, distance=1.0)
measured_placement = Placement(direction=23.5, distance=1.0)  # From pose estimation
```

## Benefits of This Approach

1. **Flexibility**: Support arbitrary angles from measurements
2. **Precision**: No loss of precision with float degrees
3. **Backwards Compatible**: Named constants still available
4. **Simplicity**: No need to create custom enums for new angles
5. **ML-Friendly**: Easy integration with continuous neural network outputs
6. **Interpolation-Ready**: Smooth transitions between angles

## Common Patterns

### Pattern 1: From Video Analysis

```python
def create_foot_from_pose_estimate(angle_deg: float, distance: float):
    """Create FootState from pose estimation data."""
    return FootState(
        placement=Placement(direction=angle_deg, distance=distance),
        alignment=0.0,
        weight=0.5,
        contact=FootContact.BALL,
    )

# Usage
foot = create_foot_from_pose_estimate(angle_deg=35.2, distance=1.1)
```

### Pattern 2: Validation and Normalization

```python
from app.models.technique import Placement

def validate_direction(degrees: float) -> float:
    """Validate and normalize a direction."""
    p = Placement(direction=degrees, distance=1.0)
    return p.direction

# Usage
angle = validate_direction(405)  # Returns 45.0
angle = validate_direction(-90)  # Returns 270.0
```

### Pattern 3: Direction Comparison

```python
def direction_difference(angle1: float, angle2: float) -> float:
    """Compute shortest angular distance between two directions."""
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)

# Usage
diff = direction_difference(10, 350)  # Returns 20 (not 340)
```

## Summary

- **Use degrees directly** as the primary method
- **Use Direction enum constants** for clarity on standard directions
- **Automatic normalization** handles wrapping and edge cases
- **Full float precision** for measurements and interpolation
- **No breaking changes** - Direction enum still works (use `.value`)
