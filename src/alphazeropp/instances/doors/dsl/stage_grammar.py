"""Enumeration and counting for the stage-skeleton DSL.

Enumerates all StagePrograms admitted by a SearchCostModel for a given
DoorsGameConfig.  Guard atoms come from the Doors domain vocabulary;
combinators (Not, And) are added up to the allowed depth.
"""

from __future__ import annotations

from itertools import product

from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
from alphazeropp.instances.doors.dsl.surface_dsl import (
    AtKeyLoc, KeyAvail, RoomLocked, PickReady, NeedKey,
    Pick, MoveToKey, MoveToGoal,
    SurfaceAction,
)
from alphazeropp.instances.doors.dsl.stage_dsl import (
    GuardExpr, GuardNot, GuardAnd,
    Stage, StageProgram,
)
from alphazeropp.instances.doors.dsl.stage_search_cost import SearchCostModel


# ---------------------------------------------------------------------------
# Guard atom enumeration
# ---------------------------------------------------------------------------

def enumerate_guard_atoms(cfg: DoorsGameConfig) -> list[GuardExpr]:
    """All typed guard atoms for a given Doors config.

    For D rooms, K = D-1 keys:
      AtKeyLoc(0..K-1), KeyAvail(0..K-1), PickReady(0..K-1), NeedKey(0..K-1),
      RoomLocked(0..D-1)
    """
    atoms: list[GuardExpr] = []
    for k in range(cfg.K):
        atoms.append(AtKeyLoc(k))
        atoms.append(KeyAvail(k))
        atoms.append(PickReady(k))
        atoms.append(NeedKey(k))
    for r in range(cfg.D):
        atoms.append(RoomLocked(r))
    return atoms


# ---------------------------------------------------------------------------
# Guard enumeration (recursive by depth)
# ---------------------------------------------------------------------------

def enumerate_guards(cfg: DoorsGameConfig, max_depth: int) -> list[GuardExpr]:
    """All guard expressions up to *max_depth* combinator nesting.

    depth 0: atoms only
    depth d: atoms + Not(depth d-1) + And(depth d-1, depth d-1)
    """
    if max_depth < 0:
        return []

    atoms = enumerate_guard_atoms(cfg)
    if max_depth == 0:
        return list(atoms)

    # Recursive: get guards at depth d-1
    sub = enumerate_guards(cfg, max_depth - 1)

    guards: list[GuardExpr] = list(sub)  # include all shallower guards

    # Not(g) for each sub-guard
    for g in sub:
        guards.append(GuardNot(g))

    # And(g1, g2) for ordered pairs (not commutative-deduped, for simplicity)
    for g1 in sub:
        for g2 in sub:
            guards.append(GuardAnd(g1, g2))

    return guards


def count_guards(cfg: DoorsGameConfig, max_depth: int) -> int:
    """Count of guard expressions without allocating them all.

    G(0) = A  (number of atoms)
    G(d) = G(d-1) + G(d-1) + G(d-1)^2  =  G(d-1)^2 + 2*G(d-1)
    """
    A = 4 * cfg.K + cfg.D  # atoms count
    g = A
    for _ in range(max_depth):
        g = g * g + 2 * g  # g^2 + 2g = g(g+2)
    return g


# ---------------------------------------------------------------------------
# Action enumeration
# ---------------------------------------------------------------------------

def enumerate_actions(cfg: DoorsGameConfig) -> list[SurfaceAction]:
    """All typed actions for a given Doors config."""
    actions: list[SurfaceAction] = []
    for k in range(cfg.K):
        actions.append(Pick(k))
        actions.append(MoveToKey(k))
    actions.append(MoveToGoal())
    return actions


# ---------------------------------------------------------------------------
# Stage program enumeration
# ---------------------------------------------------------------------------

def enumerate_stage_programs(
    cfg: DoorsGameConfig,
    cost_model: SearchCostModel,
    *,
    max_enumerate: int = 10_000_000,
) -> list[StageProgram]:
    """Enumerate all StagePrograms admitted by *cost_model*.

    Generates programs with 0 to max_stages stages.  Each stage is a
    (guard, action) pair drawn from the Cartesian product of allowed
    guards and actions.  Default action is always MoveToGoal.

    Raises ValueError if the estimated count exceeds *max_enumerate*.
    """
    guards = enumerate_guards(cfg, cost_model.max_guard_depth)
    # Filter guards by cost model (per-guard node cap)
    guards = [g for g in guards if cost_model.admits_guard(g)]
    actions = enumerate_actions(cfg)

    slots = [(g, a) for g in guards for a in actions]
    n_slots = len(slots)

    # Pre-check: total programs = sum_{s=0}^{max_stages} n_slots^s
    total = sum(n_slots ** s for s in range(cost_model.max_stages + 1))
    if total > max_enumerate:
        raise ValueError(
            f"Stage program count {total} exceeds max_enumerate={max_enumerate} "
            f"(guards={len(guards)}, actions={len(actions)}, "
            f"max_stages={cost_model.max_stages})"
        )

    default = MoveToGoal()
    programs: list[StageProgram] = []

    # s=0: just the default
    programs.append(StageProgram(stages=(), default_action=default))

    for s in range(1, cost_model.max_stages + 1):
        for combo in product(slots, repeat=s):
            stages = tuple(Stage(guard=g, action=a) for g, a in combo)
            programs.append(StageProgram(stages=stages, default_action=default))

    return programs


def count_stage_programs(
    cfg: DoorsGameConfig,
    cost_model: SearchCostModel,
) -> int:
    """Count stage programs without allocating them.

    Returns sum_{s=0}^{max_stages} (n_guards * n_actions)^s.
    """
    guards = enumerate_guards(cfg, cost_model.max_guard_depth)
    guards = [g for g in guards if cost_model.admits_guard(g)]
    actions = enumerate_actions(cfg)
    n_slots = len(guards) * len(actions)
    return sum(n_slots ** s for s in range(cost_model.max_stages + 1))
