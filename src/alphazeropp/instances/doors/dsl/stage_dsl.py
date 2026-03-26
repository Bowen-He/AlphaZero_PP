"""Budget-free typed stage-skeleton DSL for the Doors environment.

A stage program is an ordered sequence of Stage(guard, action) pairs
followed by a default action.  Guards are composed from the same typed
atoms as the surface DSL (AtKeyLoc, KeyAvail, etc.) but may be combined
with And/Not.  No budget appears anywhere in the syntax.

Hierarchy:
  Guard combinators — GuardNot(child), GuardAnd(left, right)
  Holes            — GuardHole(max_depth), ActionHole()
  Stage            — Stage(guard, action)
  Program          — StageProgram(stages, default_action)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from alphazeropp.instances.doors.dsl.surface_dsl import (
    AtKeyLoc, KeyAvail, RoomLocked, PickReady, NeedKey,
    Pick, MoveToKey, MoveToGoal,
    SurfaceCondition, SurfaceAction,
)


# ---------------------------------------------------------------------------
# Guard combinators
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardNot:
    """Negation of a guard expression."""
    child: GuardExpr

    def pretty(self) -> str:
        return f"Not({self.child.pretty()})"


@dataclass(frozen=True)
class GuardAnd:
    """Conjunction of two guard expressions."""
    left: GuardExpr
    right: GuardExpr

    def pretty(self) -> str:
        return f"And({self.left.pretty()}, {self.right.pretty()})"


# Guard type aliases
GuardAtom = Union[AtKeyLoc, KeyAvail, RoomLocked, PickReady, NeedKey]
GuardExpr = Union[AtKeyLoc, KeyAvail, RoomLocked, PickReady, NeedKey, GuardNot, GuardAnd]


# ---------------------------------------------------------------------------
# Holes (budget-free placeholders)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardHole:
    """Placeholder for an unresolved guard.

    *max_depth* is a search constraint hint, NOT part of syntax state.
    """
    max_depth: int = 0

    def pretty(self) -> str:
        return f"GuardHole(d≤{self.max_depth})"


@dataclass(frozen=True)
class ActionHole:
    """Placeholder for an unresolved action."""

    def pretty(self) -> str:
        return "ActionHole"


# ---------------------------------------------------------------------------
# Stage and StageProgram
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage:
    """A single guard → action pair."""
    guard: GuardExpr | GuardHole
    action: SurfaceAction | ActionHole

    def pretty(self) -> str:
        g = self.guard.pretty()
        a = self.action.pretty()
        return f"if {g} then {a}"

    def is_complete(self) -> bool:
        """True if neither guard nor action is a hole."""
        return not isinstance(self.guard, GuardHole) and not isinstance(self.action, ActionHole)


@dataclass(frozen=True)
class StageProgram:
    """Ordered sequence of stages with a default action.

    Semantics: evaluate stages top-to-bottom; first true guard fires.
    If no guard fires, execute default_action.
    """
    stages: tuple[Stage, ...]
    default_action: SurfaceAction = field(default_factory=MoveToGoal)

    def pretty(self) -> str:
        lines = []
        for i, s in enumerate(self.stages):
            prefix = "if" if i == 0 else "elif"
            lines.append(f"{prefix} {s.guard.pretty()} then {s.action.pretty()}")
        lines.append(f"else {self.default_action.pretty()}")
        return "\n".join(lines)

    def is_complete(self) -> bool:
        """True if all stages are complete (no holes)."""
        return all(s.is_complete() for s in self.stages)

    @property
    def num_stages(self) -> int:
        return len(self.stages)
