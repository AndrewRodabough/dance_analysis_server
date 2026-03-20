"""Mapper between domain Figure/Step dataclasses and FigureModel/StepModel ORM records.

Serialization: dataclass tree → JSON-safe dicts  (via _serialize)
Deserialization: JSON dicts → dataclass tree      (via _deserialize)

Both helpers are driven entirely by dataclass introspection and type hints,
so when fields are added, removed, or renamed on the domain objects the mapper
adapts automatically.  The only manual touchpoints are figure_to_record() and
record_to_figure(), which map the "promoted" SQL columns (name, tags, etc.).
"""

import dataclasses
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import List, Optional, Type, TypeVar, Union, get_args, get_origin, get_type_hints

from app.models.figure import DancerState, Figure, Keyframe, Step
from app.models.figure_store import FigureModel, StepModel

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Generic serialization helpers
# ---------------------------------------------------------------------------

def _serialize(obj):
    """Recursively convert a dataclass tree to a JSON-serializable dict.

    - Dataclasses → dicts (keyed by field name)
    - Enums → their value
    - Lists → mapped recursively
    - Primitives → passed through
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _serialize(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    return obj


def _deserialize(cls: Type[T], data) -> T:
    """Reconstruct a typed object from a JSON-safe value using *cls* as the target type.

    Handles dataclasses, Enums, Optional, List, and primitives.
    Missing fields that have a default on the dataclass are filled automatically,
    so old JSON records survive schema additions gracefully.
    """
    if data is None:
        return None

    origin = get_origin(cls)

    # Optional[X] → unwrap and recurse
    if origin is Union:
        non_none = [a for a in get_args(cls) if a is not type(None)]
        if len(non_none) == 1:
            return _deserialize(non_none[0], data)
        return data

    # List[X] → map elements
    if origin is list:
        args = get_args(cls)
        if args and isinstance(data, list):
            return [_deserialize(args[0], item) for item in data]
        return data

    # Dataclass → reconstruct from dict
    if is_dataclass(cls):
        hints = get_type_hints(cls)
        kwargs = {}
        for f in fields(cls):
            if f.name in data:
                kwargs[f.name] = _deserialize(hints[f.name], data[f.name])
            elif f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                kwargs[f.name] = f.default_factory()
            # Otherwise omit — let the dataclass raise for missing required fields
        return cls(**kwargs)

    # Enum → reconstruct from value
    if isinstance(cls, type) and issubclass(cls, Enum):
        return cls(data)

    return data


# ---------------------------------------------------------------------------
# Public mapper API
# ---------------------------------------------------------------------------

def figure_to_record(figure: Figure) -> FigureModel:
    """Convert a domain Figure (with Steps) to a FigureModel with child StepModels.

    The returned FigureModel has its ``steps`` relationship pre-populated,
    so a single ``db.add(record)`` will persist everything via cascade.
    """
    record = FigureModel(
        name=figure.name,
        tags=figure.tags,
        total_beats=float(figure.total_beats),
    )
    record.steps = [
        StepModel(
            position=i,
            start_state=_serialize(step.start),
            end_state=_serialize(step.end),
            keyframes=[_serialize(kf) for kf in step.keyframes] if step.keyframes else None,
            duration=step.duration,
        )
        for i, step in enumerate(figure.steps)
    ]
    return record


def record_to_figure(record: FigureModel) -> Figure:
    """Convert a FigureModel (with steps eagerly loaded) back to a domain Figure.

    Steps are ordered by their ``position`` column.
    """
    sorted_steps = sorted(record.steps, key=lambda s: s.position)
    steps = [
        Step(
            start=_deserialize(DancerState, sr.start_state),
            end=_deserialize(DancerState, sr.end_state),
            keyframes=[_deserialize(Keyframe, kf) for kf in sr.keyframes] if sr.keyframes else [],
            duration=sr.duration,
        )
        for sr in sorted_steps
    ]
    return Figure(
        name=record.name,
        steps=steps,
        tags=record.tags,
        total_beats=record.total_beats,
    )
