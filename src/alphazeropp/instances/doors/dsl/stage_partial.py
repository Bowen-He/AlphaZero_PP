"""Partial stage programs for step-by-step hole-filling search.

A PartialStageProgram wraps a StageProgram that may contain GuardHole
or ActionHole placeholders, plus tracking for whether the structure
(number of stages) has been finalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from alphazeropp.instances.doors.dsl.surface_dsl import (
    MoveToGoal, SurfaceAction,
)
from alphazeropp.instances.doors.dsl.stage_dsl import (
    GuardHole, ActionHole, GuardExpr,
    Stage, StageProgram,
)


# ---------------------------------------------------------------------------
# Hole reference types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StructureHole:
    """Decision: add another stage or finalize the structure."""
    pass


@dataclass(frozen=True)
class GuardHoleRef:
    """Fill the guard at stages[stage_idx]."""
    stage_idx: int


@dataclass(frozen=True)
class ActionHoleRef:
    """Fill the action at stages[stage_idx]."""
    stage_idx: int


HoleRef = Union[StructureHole, GuardHoleRef, ActionHoleRef]


# ---------------------------------------------------------------------------
# Fillings for StructureHole
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AddStage:
    """StructureHole filling: append a new empty stage."""
    pass


@dataclass(frozen=True)
class Finalize:
    """StructureHole filling: no more stages will be added."""
    pass


StructureFilling = Union[AddStage, Finalize]


# ---------------------------------------------------------------------------
# Partial stage program
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartialStageProgram:
    """A StageProgram under construction, possibly with holes."""
    stages: tuple[Stage, ...]
    max_stages: int
    structure_finalized: bool = False
    default_action: SurfaceAction = field(default_factory=MoveToGoal)

    def is_terminal(self) -> bool:
        """True if structure is finalized and all stages are complete."""
        return self.structure_finalized and all(s.is_complete() for s in self.stages)

    def to_stage_program(self) -> StageProgram:
        """Convert to a StageProgram. Raises if not terminal."""
        if not self.is_terminal():
            raise ValueError("Cannot convert non-terminal partial to StageProgram")
        return StageProgram(stages=self.stages, default_action=self.default_action)


# ---------------------------------------------------------------------------
# Hole enumeration
# ---------------------------------------------------------------------------

def list_holes(partial: PartialStageProgram) -> list[HoleRef]:
    """List all unfilled holes in the partial program.

    Order: for each stage, guard then action; then StructureHole if applicable.
    """
    holes: list[HoleRef] = []
    for i, stage in enumerate(partial.stages):
        if isinstance(stage.guard, GuardHole):
            holes.append(GuardHoleRef(i))
        if isinstance(stage.action, ActionHole):
            holes.append(ActionHoleRef(i))
    if not partial.structure_finalized:
        # StructureHole exists if we haven't finalized yet
        # But only if all current stages are complete OR we have no stages yet
        all_complete = all(s.is_complete() for s in partial.stages)
        if all_complete:
            holes.append(StructureHole())
    return holes


# ---------------------------------------------------------------------------
# Filling application
# ---------------------------------------------------------------------------

def apply_filling(
    partial: PartialStageProgram,
    hole_ref: HoleRef,
    filling,
) -> PartialStageProgram:
    """Return a new partial with one hole filled.

    - StructureHole + AddStage: append Stage(GuardHole, ActionHole)
    - StructureHole + Finalize: set structure_finalized=True
    - GuardHoleRef + GuardExpr: replace guard at stage_idx
    - ActionHoleRef + SurfaceAction: replace action at stage_idx
    """
    if isinstance(hole_ref, StructureHole):
        if isinstance(filling, AddStage):
            new_stage = Stage(guard=GuardHole(), action=ActionHole())
            return PartialStageProgram(
                stages=partial.stages + (new_stage,),
                max_stages=partial.max_stages,
                structure_finalized=False,
                default_action=partial.default_action,
            )
        if isinstance(filling, Finalize):
            return PartialStageProgram(
                stages=partial.stages,
                max_stages=partial.max_stages,
                structure_finalized=True,
                default_action=partial.default_action,
            )
        raise TypeError(f"Invalid filling for StructureHole: {type(filling)}")

    if isinstance(hole_ref, GuardHoleRef):
        idx = hole_ref.stage_idx
        old = partial.stages[idx]
        new_stage = Stage(guard=filling, action=old.action)
        stages = partial.stages[:idx] + (new_stage,) + partial.stages[idx + 1:]
        return PartialStageProgram(
            stages=stages,
            max_stages=partial.max_stages,
            structure_finalized=partial.structure_finalized,
            default_action=partial.default_action,
        )

    if isinstance(hole_ref, ActionHoleRef):
        idx = hole_ref.stage_idx
        old = partial.stages[idx]
        new_stage = Stage(guard=old.guard, action=filling)
        stages = partial.stages[:idx] + (new_stage,) + partial.stages[idx + 1:]
        return PartialStageProgram(
            stages=stages,
            max_stages=partial.max_stages,
            structure_finalized=partial.structure_finalized,
            default_action=partial.default_action,
        )

    raise TypeError(f"Unknown hole ref type: {type(hole_ref)}")


def partial_depth(partial: PartialStageProgram) -> int:
    """Number of filling decisions made so far."""
    count = 0
    for stage in partial.stages:
        if not isinstance(stage.guard, GuardHole):
            count += 1
        if not isinstance(stage.action, ActionHole):
            count += 1
    # Each stage added was a StructureHole decision
    count += len(partial.stages)
    if partial.structure_finalized:
        count += 1
    return count
