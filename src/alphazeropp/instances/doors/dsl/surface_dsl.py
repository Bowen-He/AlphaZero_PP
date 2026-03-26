"""Typed surface DSL for the Doors environment — DoorsStageDSL(D).

Defines semantic building blocks parameterised by key index (k) or room
index (r).  These constructs carry *no* raw observation or action indices;
the compiler (surface_compiler.py) resolves them via DoorsGameConfig.

Hierarchy:
  Conditions  — AtKeyLoc(k), KeyAvail(k), RoomLocked(r),
                 PickReady(k), NeedKey(k)
  Actions     — Pick(k), MoveToKey(k), MoveToGoal
  Rules       — PickRule(k), MoveRule(k), GoalRule
  Policy      — SurfacePolicy(rules)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# ---------------------------------------------------------------------------
# Condition atoms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtKeyLoc:
    """Agent is at the location of key *k*."""
    k: int

    def pretty(self) -> str:
        return f"AtKeyLoc({self.k})"


@dataclass(frozen=True)
class KeyAvail:
    """Key *k* is available for pickup."""
    k: int

    def pretty(self) -> str:
        return f"KeyAvail({self.k})"


@dataclass(frozen=True)
class RoomLocked:
    """Room *r* is locked."""
    r: int

    def pretty(self) -> str:
        return f"RoomLocked({self.r})"


@dataclass(frozen=True)
class PickReady:
    """Agent is at key *k* AND key *k* is available."""
    k: int

    def pretty(self) -> str:
        return f"PickReady({self.k})"


@dataclass(frozen=True)
class NeedKey:
    """The room that key *k* unlocks is still locked."""
    k: int

    def pretty(self) -> str:
        return f"NeedKey({self.k})"


SurfaceCondition = Union[AtKeyLoc, KeyAvail, RoomLocked, PickReady, NeedKey]


# ---------------------------------------------------------------------------
# Action atoms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pick:
    """Pick up key *k*."""
    k: int

    def pretty(self) -> str:
        return f"Pick({self.k})"


@dataclass(frozen=True)
class MoveToKey:
    """Move to the location of key *k*."""
    k: int

    def pretty(self) -> str:
        return f"MoveToKey({self.k})"


@dataclass(frozen=True)
class MoveToGoal:
    """Move to the goal location."""

    def pretty(self) -> str:
        return "MoveToGoal"


SurfaceAction = Union[Pick, MoveToKey, MoveToGoal]


# ---------------------------------------------------------------------------
# Rule macros
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PickRule:
    """If PickReady(k) then Pick(k), else continue."""
    k: int

    def pretty(self) -> str:
        return f"PickRule({self.k})"


@dataclass(frozen=True)
class MoveRule:
    """If NeedKey(k) then MoveToKey(k), else continue."""
    k: int

    def pretty(self) -> str:
        return f"MoveRule({self.k})"


@dataclass(frozen=True)
class GoalRule:
    """Default: move to goal."""

    def pretty(self) -> str:
        return "GoalRule"


SurfaceRule = Union[PickRule, MoveRule, GoalRule]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfacePolicy:
    """Ordered sequence of surface rules.

    Invariants:
      - At least one rule.
      - Last rule must be GoalRule.
    """
    rules: tuple[SurfaceRule, ...]

    def __post_init__(self):
        if len(self.rules) == 0:
            raise ValueError("Policy must have at least one rule")
        if not isinstance(self.rules[-1], GoalRule):
            raise ValueError("Last rule must be GoalRule")

    def pretty(self) -> str:
        return " >> ".join(r.pretty() for r in self.rules)
