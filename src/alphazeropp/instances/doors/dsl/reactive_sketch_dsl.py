"""Reactive sketch DSL with behavior-tree semantics for Doors.

Fixed skeleton: While(Not(GoalReached), Fallback(B1, B2, B3, B4))

Four branch schemas (each a Sequence of Check + Do):
  B1: if Pickable(key) then Pick(key)
  B2: if KnownLoc(key) then GoTo(LocOf(key))
  B3: if Reachable(GoalLoc) then GoTo(GoalLoc)
  B4: if ExistsUnlockedFrontier then GoTo(Entrance(NextLockedRoom))

Synthesis scope: branch order only (4! = 24 candidates).
All nodes are frozen dataclasses with pretty() methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


# ---------------------------------------------------------------------------
# Tick status
# ---------------------------------------------------------------------------

class TickStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


# ---------------------------------------------------------------------------
# Room selectors (resolved at tick time against current obs)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NextLockedRoomSel:
    """First lockable room (canonical order) that is still locked."""

    def pretty(self) -> str:
        return "NextLockedRoom"


@dataclass(frozen=True)
class NextUnsearchedRoomSel:
    """First reachable room not yet in memory.searched_rooms."""

    def pretty(self) -> str:
        return "NextUnsearchedRoom"


BTRoomSel = Union[NextLockedRoomSel, NextUnsearchedRoomSel]


# ---------------------------------------------------------------------------
# Key selectors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyForSel:
    """Key that unlocks the selected room."""
    room: BTRoomSel

    def pretty(self) -> str:
        return f"KeyFor({self.room.pretty()})"


BTKeySel = KeyForSel


# ---------------------------------------------------------------------------
# Location selectors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocOfSel:
    """Location where the selected key is found."""
    key: BTKeySel

    def pretty(self) -> str:
        return f"LocOf({self.key.pretty()})"


@dataclass(frozen=True)
class GoalLocSel:
    """The goal location."""

    def pretty(self) -> str:
        return "GoalLoc"


@dataclass(frozen=True)
class EntranceSel:
    """First location in the room just BEFORE the selected room.

    For NextLockedRoom r, this resolves to locations_in_room(r-1)[0],
    i.e. the frontier of reachable space.
    """
    room: BTRoomSel

    def pretty(self) -> str:
        return f"Entrance({self.room.pretty()})"


@dataclass(frozen=True)
class FirstLocInRoomSel:
    """First location INSIDE the selected room.

    Unlike EntranceSel (which goes to room r-1, the frontier before a
    locked room), this goes into the room itself — needed for exploration
    to trigger key discovery.
    """
    room: BTRoomSel

    def pretty(self) -> str:
        return f"FirstLocIn({self.room.pretty()})"


BTLocSel = Union[LocOfSel, GoalLocSel, EntranceSel, FirstLocInRoomSel]


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoalReachedP:
    """Agent is at the goal location."""

    def pretty(self) -> str:
        return "GoalReached"


@dataclass(frozen=True)
class PickableP:
    """Key is at agent's location AND available for pickup."""
    key_sel: BTKeySel

    def pretty(self) -> str:
        return f"Pickable({self.key_sel.pretty()})"


@dataclass(frozen=True)
class KnownLocP:
    """Key location is known (trivially true for known_map=True)."""
    key_sel: BTKeySel

    def pretty(self) -> str:
        return f"KnownLoc({self.key_sel.pretty()})"


@dataclass(frozen=True)
class ReachableP:
    """Room containing the selected location is unlocked."""
    loc_sel: BTLocSel

    def pretty(self) -> str:
        return f"Reachable({self.loc_sel.pretty()})"


@dataclass(frozen=True)
class ExistsUnlockedFrontierP:
    """There exists a locked room whose key is in a reachable room."""

    def pretty(self) -> str:
        return "ExistsUnlockedFrontier"


@dataclass(frozen=True)
class ExistsUnsearchedRoomP:
    """There exists a reachable room not yet in memory.searched_rooms."""

    def pretty(self) -> str:
        return "ExistsUnsearchedRoom"


@dataclass(frozen=True)
class TrueP:
    """Always true. Enables unconditional branches in the typed catalog."""

    def pretty(self) -> str:
        return "True"


BTPredicate = Union[
    GoalReachedP, PickableP, KnownLocP, ReachableP,
    ExistsUnlockedFrontierP, ExistsUnsearchedRoomP, TrueP,
]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PickAction:
    """Pick up the selected key."""
    key: BTKeySel

    def pretty(self) -> str:
        return f"Pick({self.key.pretty()})"


@dataclass(frozen=True)
class GoToAction:
    """Move to the selected location."""
    loc: BTLocSel

    def pretty(self) -> str:
        return f"GoTo({self.loc.pretty()})"


@dataclass(frozen=True)
class NoopAction:
    """Do nothing. Resolves to the environment noop action index."""

    def pretty(self) -> str:
        return "Noop"


BTAction = Union[PickAction, GoToAction, NoopAction]


# ---------------------------------------------------------------------------
# BT structural nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    """Condition check. SUCCESS if predicate true, FAILURE otherwise."""
    predicate: BTPredicate

    def pretty(self) -> str:
        return f"Check({self.predicate.pretty()})"


@dataclass(frozen=True)
class Do:
    """Execute action. Always SUCCESS (with action index)."""
    action: BTAction

    def pretty(self) -> str:
        return f"Do({self.action.pretty()})"


# Forward-reference-safe union for BTNode children
BTLeaf = Union[Check, Do]


@dataclass(frozen=True)
class Sequence:
    """Run children L-to-R. FAILURE on first FAILURE, else SUCCESS."""
    children: tuple

    def __post_init__(self):
        if len(self.children) == 0:
            raise ValueError("Sequence must have at least one child")

    def pretty(self) -> str:
        inner = ", ".join(c.pretty() for c in self.children)
        return f"Sequence({inner})"


@dataclass(frozen=True)
class Fallback:
    """Try children L-to-R. Return first SUCCESS, else FAILURE."""
    children: tuple

    def __post_init__(self):
        if len(self.children) == 0:
            raise ValueError("Fallback must have at least one child")

    def pretty(self) -> str:
        inner = ", ".join(c.pretty() for c in self.children)
        return f"Fallback({inner})"


@dataclass(frozen=True)
class WhileNot:
    """Root loop: tick child each step until predicate becomes true."""
    predicate: BTPredicate
    child: "BTNode"

    def pretty(self) -> str:
        return f"While(Not({self.predicate.pretty()}), {self.child.pretty()})"


BTNode = Union[WhileNot, Fallback, Sequence, Check, Do]


# ---------------------------------------------------------------------------
# Branch constructors (4 fixed schemas)
# ---------------------------------------------------------------------------

def branch_pick_if_ready() -> Sequence:
    """B1: if Pickable(KeyFor(NextLockedRoom)) then Pick(KeyFor(NextLockedRoom))."""
    k = KeyForSel(NextLockedRoomSel())
    return Sequence(children=(
        Check(PickableP(k)),
        Do(PickAction(k)),
    ))


def branch_goto_key() -> Sequence:
    """B2: if KnownLoc(KeyFor(NextLockedRoom)) then GoTo(LocOf(KeyFor(NextLockedRoom)))."""
    k = KeyForSel(NextLockedRoomSel())
    return Sequence(children=(
        Check(KnownLocP(k)),
        Do(GoToAction(LocOfSel(k))),
    ))


def branch_goto_goal() -> Sequence:
    """B3: if Reachable(GoalLoc) then GoTo(GoalLoc)."""
    return Sequence(children=(
        Check(ReachableP(GoalLocSel())),
        Do(GoToAction(GoalLocSel())),
    ))


def branch_explore_frontier() -> Sequence:
    """B4: if ExistsUnlockedFrontier then GoTo(Entrance(NextLockedRoom))."""
    return Sequence(children=(
        Check(ExistsUnlockedFrontierP()),
        Do(GoToAction(EntranceSel(NextLockedRoomSel()))),
    ))


def branch_explore_unsearched() -> Sequence:
    """B5: if ExistsUnsearchedRoom then GoTo(FirstLocIn(NextUnsearchedRoom))."""
    return Sequence(children=(
        Check(ExistsUnsearchedRoomP()),
        Do(GoToAction(FirstLocInRoomSel(NextUnsearchedRoomSel()))),
    ))


BRANCH_BUILDERS = {
    1: branch_pick_if_ready,
    2: branch_goto_key,
    3: branch_goto_goal,
    4: branch_explore_frontier,
    5: branch_explore_unsearched,
}


# ---------------------------------------------------------------------------
# Policy constructor
# ---------------------------------------------------------------------------

def canonical_reactive_policy(
    branch_order: tuple[int, ...] = (1, 2, 3, 4),
) -> WhileNot:
    """Build the fixed-skeleton reactive policy with the given branch order.

    Args:
        branch_order: Permutation of (1, 2, 3, 4) specifying Fallback child order.

    Returns:
        WhileNot root node.
    """
    if sorted(branch_order) != [1, 2, 3, 4]:
        raise ValueError(
            f"branch_order must be a permutation of (1,2,3,4), got {branch_order}"
        )
    branches = tuple(BRANCH_BUILDERS[i]() for i in branch_order)
    return WhileNot(
        predicate=GoalReachedP(),
        child=Fallback(children=branches),
    )


def canonical_partial_map_policy(
    branch_order: tuple[int, ...] = (1, 2, 5, 3, 4),
) -> WhileNot:
    """Build a 5-branch reactive policy for partial-map mode.

    Includes the exploration branch (B5) alongside the original B1–B4.

    Args:
        branch_order: Permutation of (1, 2, 3, 4, 5) specifying Fallback child order.

    Returns:
        WhileNot root node.
    """
    if sorted(branch_order) != [1, 2, 3, 4, 5]:
        raise ValueError(
            f"branch_order must be a permutation of (1,2,3,4,5), got {branch_order}"
        )
    branches = tuple(BRANCH_BUILDERS[i]() for i in branch_order)
    return WhileNot(
        predicate=GoalReachedP(),
        child=Fallback(children=branches),
    )
