"""Lifted decision-list DSL for the Doors environment.

Programs in this DSL contain NO grounded key or room indices.  A single
lifted policy expression works for any game size D.  The compiler
(lifted_compiler.py) expands abstract selectors by iterating over
entities from DoorsRelationalRuntime.

Hierarchy:
  Selectors   — NextLockedRoom, GoalRoom, CurrentRoom,
                 KeyFor(RoomSel), LocOf(KeySel), GoalLoc
  Predicates  — Pickable(KeySel), NeedKey(KeySel), GoalReached
  Actions     — LiftedPick(KeySel), GoTo(LocSel), GoToGoal
  Rules       — IfThen(predicate, action), LiftedDefault(action)
  Structure   — ForEachLockedRoom(rules)
  Policy      — LiftedPolicy(body, default)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# ---------------------------------------------------------------------------
# Room selectors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NextLockedRoom:
    """The first lockable room (in canonical order) that is still locked."""

    def pretty(self) -> str:
        return "NextLockedRoom"


@dataclass(frozen=True)
class GoalRoom:
    """The room containing the goal location."""

    def pretty(self) -> str:
        return "GoalRoom"


@dataclass(frozen=True)
class CurrentRoom:
    """Placeholder: the room currently being iterated over.

    Only valid inside a ForEachLockedRoom body.
    """

    def pretty(self) -> str:
        return "CurrentRoom"


RoomSel = Union[NextLockedRoom, GoalRoom, CurrentRoom]


# ---------------------------------------------------------------------------
# Key selectors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyFor:
    """The key that unlocks the selected room."""
    room: RoomSel

    def pretty(self) -> str:
        return f"KeyFor({self.room.pretty()})"


KeySel = KeyFor


# ---------------------------------------------------------------------------
# Location selectors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocOf:
    """The location where the selected key is found."""
    key: KeySel

    def pretty(self) -> str:
        return f"LocOf({self.key.pretty()})"


@dataclass(frozen=True)
class GoalLoc:
    """The goal location."""

    def pretty(self) -> str:
        return "GoalLoc"


LocSel = Union[LocOf, GoalLoc]


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pickable:
    """Key is at agent's location AND available for pickup."""
    key: KeySel

    def pretty(self) -> str:
        return f"Pickable({self.key.pretty()})"


@dataclass(frozen=True)
class NeedKey:
    """The room that this key unlocks is still locked."""
    key: KeySel

    def pretty(self) -> str:
        return f"NeedKey({self.key.pretty()})"


@dataclass(frozen=True)
class GoalReached:
    """Agent is at the goal location."""

    def pretty(self) -> str:
        return "GoalReached"


LiftedPredicate = Union[Pickable, NeedKey, GoalReached]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiftedPick:
    """Pick up the selected key."""
    key: KeySel

    def pretty(self) -> str:
        return f"Pick({self.key.pretty()})"


@dataclass(frozen=True)
class GoTo:
    """Move to the selected location."""
    loc: LocSel

    def pretty(self) -> str:
        return f"GoTo({self.loc.pretty()})"


@dataclass(frozen=True)
class GoToGoal:
    """Move to the goal location."""

    def pretty(self) -> str:
        return "GoToGoal"


LiftedAction = Union[LiftedPick, GoTo, GoToGoal]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IfThen:
    """A single predicate-action rule in the decision list."""
    predicate: LiftedPredicate
    action: LiftedAction

    def pretty(self) -> str:
        return f"if {self.predicate.pretty()} then {self.action.pretty()}"


@dataclass(frozen=True)
class LiftedDefault:
    """The fallthrough action (always executed if no rule matches)."""
    action: LiftedAction

    def pretty(self) -> str:
        return f"default {self.action.pretty()}"


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForEachLockedRoom:
    """A sequence of rules parameterised by CurrentRoom.

    The compiler iterates lockable rooms and emits these rules for
    each room, interleaved: room 1's rules, then room 2's, etc.
    This produces the canonical interleaved order.
    """
    rules: tuple[IfThen, ...]

    def __post_init__(self):
        if len(self.rules) == 0:
            raise ValueError("ForEachLockedRoom must have at least one rule")
        for r in self.rules:
            if not isinstance(r, IfThen):
                raise TypeError(f"Expected IfThen, got {type(r)}")

    def pretty(self) -> str:
        inner = "; ".join(r.pretty() for r in self.rules)
        return f"for each locked room: [{inner}]"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiftedPolicy:
    """A lifted decision list: rules expanded per room, then a default.

    Invariants:
      - body is a ForEachLockedRoom.
      - default is a LiftedDefault.
    """
    body: ForEachLockedRoom
    default: LiftedDefault

    def __post_init__(self):
        if not isinstance(self.body, ForEachLockedRoom):
            raise TypeError(
                f"body must be ForEachLockedRoom, got {type(self.body)}"
            )
        if not isinstance(self.default, LiftedDefault):
            raise TypeError(
                f"default must be LiftedDefault, got {type(self.default)}"
            )

    def pretty(self) -> str:
        return f"{self.body.pretty()}; {self.default.pretty()}"


# ---------------------------------------------------------------------------
# Canonical policy constructor
# ---------------------------------------------------------------------------

def canonical_lifted_policy() -> LiftedPolicy:
    """The universal canonical policy in lifted form.

    Works for any D >= 1.  The compiler expands CurrentRoom over all
    lockable rooms, producing the standard interleaved order:
    PickReady(0), NeedKey(0), PickReady(1), NeedKey(1), ..., GoalRule.

    Structure:
      for each locked room:
        if Pickable(KeyFor(CurrentRoom)) then Pick(KeyFor(CurrentRoom))
        if NeedKey(KeyFor(CurrentRoom)) then GoTo(LocOf(KeyFor(CurrentRoom)))
      default GoToGoal
    """
    k = KeyFor(CurrentRoom())
    return LiftedPolicy(
        body=ForEachLockedRoom(rules=(
            IfThen(Pickable(k), LiftedPick(k)),
            IfThen(NeedKey(k), GoTo(LocOf(k))),
        )),
        default=LiftedDefault(GoToGoal()),
    )
