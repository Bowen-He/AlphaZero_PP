"""Hole-selection policies for stage-skeleton program construction.

Each policy implements a different strategy for choosing which hole
to fill next in a PartialStageProgram.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from alphazeropp.instances.doors.dsl.stage_dsl import GuardHole, ActionHole
from alphazeropp.instances.doors.dsl.stage_partial import (
    PartialStageProgram, HoleRef,
    StructureHole, GuardHoleRef, ActionHoleRef,
    list_holes,
)


class HoleSelectionPolicy(ABC):
    """Abstract base class for hole-selection policies."""

    @abstractmethod
    def select_hole(self, partial: PartialStageProgram) -> HoleRef:
        """Choose the next hole to fill. Raises ValueError if no holes."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class LeftmostPolicy(HoleSelectionPolicy):
    """Fill each stage completely (guard then action) before deciding structure.

    Order:
    1. For the earliest incomplete stage: fill guard, then action.
    2. Once all current stages are complete: StructureHole.
    """

    @property
    def name(self) -> str:
        return "leftmost"

    def select_hole(self, partial: PartialStageProgram) -> HoleRef:
        # First, look for unfilled holes in existing stages (left to right)
        for i, stage in enumerate(partial.stages):
            if isinstance(stage.guard, GuardHole):
                return GuardHoleRef(i)
            if isinstance(stage.action, ActionHole):
                return ActionHoleRef(i)

        # All existing stages are complete
        if not partial.structure_finalized:
            return StructureHole()

        raise ValueError("No holes to fill — partial is terminal")


class StructureFirstPolicy(HoleSelectionPolicy):
    """Decide all stage slots first, then fill left-to-right.

    Order:
    1. StructureHole until finalized (decides total stage count upfront).
    2. Once finalized: fill stages left-to-right, guard then action.
    """

    @property
    def name(self) -> str:
        return "structure_first"

    def select_hole(self, partial: PartialStageProgram) -> HoleRef:
        # Structure phase: always pick StructureHole if not finalized
        if not partial.structure_finalized:
            return StructureHole()

        # Fill phase: left-to-right, guard then action
        for i, stage in enumerate(partial.stages):
            if isinstance(stage.guard, GuardHole):
                return GuardHoleRef(i)
            if isinstance(stage.action, ActionHole):
                return ActionHoleRef(i)

        raise ValueError("No holes to fill — partial is terminal")
