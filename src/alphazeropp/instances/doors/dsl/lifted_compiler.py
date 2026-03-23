"""Compiler from the lifted DSL to the raw AST layer.

Uses DoorsRelationalRuntime as the SOLE interface to DoorsGameConfig.
No direct cfg field access (cfg.key_loc, cfg.key_unlocks, etc.).

The compiler expands ForEachLockedRoom by iterating lockable rooms
from the relational runtime.  Each room's rules are emitted in order,
producing the canonical interleaved AST structure.
"""

from __future__ import annotations

from alphazeropp.synthesis.ast_nodes import (
    Flip, Ite, Default,
    Condition, Program,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime, KeyId,
)
from alphazeropp.instances.doors.dsl.lifted_dsl import (
    Pickable, NeedKey, GoalReached,
    LiftedPick, GoTo, GoToGoal, LocOf, GoalLoc,
    LiftedDefault, LiftedPolicy,
    LiftedPredicate, LiftedAction,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_lifted_policy(
    policy: LiftedPolicy,
    rt: DoorsRelationalRuntime,
) -> Program:
    """Compile a LiftedPolicy to a raw AST Program.

    Expands ForEachLockedRoom by iterating ``rt.lockable_rooms()``.
    Uses a double-reversed loop so that:
      - Outer: rooms in reverse → room 1's rules are outermost (checked first)
      - Inner: rules in reverse → first rule wraps second (checked first)

    Result for D=3: PickReady(0), NeedKey(0), PickReady(1), NeedKey(1), Default
    This matches ``canonical_policy(3)`` compiled via ``surface_compiler``.
    """
    # Start with the default action
    prog: Program = _compile_default(policy.default, rt)

    # Expand ForEachLockedRoom: for each room, emit all rules
    for r in reversed(rt.lockable_rooms()):
        k = rt.key_for_room(r)
        assert k is not None, f"lockable room {r} has no key"

        for rule in reversed(policy.body.rules):
            cond = _compile_predicate(rule.predicate, rt, k)
            action = _compile_action(rule.action, rt, k)
            assert action is not None, (
                f"Cannot compile action {rule.action!r} for key {k}"
            )
            assert cond is not None, (
                f"Cannot compile predicate {rule.predicate!r} for key {k}"
            )
            prog = Ite(cond, action, prog)

    return prog


# ---------------------------------------------------------------------------
# Predicate compilation
# ---------------------------------------------------------------------------

def _compile_predicate(
    pred: LiftedPredicate,
    rt: DoorsRelationalRuntime,
    k: KeyId,
) -> Condition | None:
    """Compile a lifted predicate with CurrentRoom resolved to key *k*."""
    if isinstance(pred, Pickable):
        return rt.cond_pick_ready(k)

    if isinstance(pred, NeedKey):
        return rt.cond_need_key(k)

    if isinstance(pred, GoalReached):
        return rt.cond_at_loc(rt.goal_location())

    raise TypeError(f"Unknown lifted predicate: {type(pred)}")


# ---------------------------------------------------------------------------
# Action compilation
# ---------------------------------------------------------------------------

def _compile_action(
    action: LiftedAction,
    rt: DoorsRelationalRuntime,
    k: KeyId,
) -> Flip | None:
    """Compile a lifted action with CurrentRoom resolved to key *k*."""
    if isinstance(action, LiftedPick):
        return rt.flip_pick(k)

    if isinstance(action, GoTo):
        if isinstance(action.loc, LocOf):
            return rt.flip_move_to_key(k)
        if isinstance(action.loc, GoalLoc):
            return rt.flip_move_to_goal()
        raise TypeError(f"Unknown LocSel: {type(action.loc)}")

    if isinstance(action, GoToGoal):
        return rt.flip_move_to_goal()

    raise TypeError(f"Unknown lifted action: {type(action)}")


# ---------------------------------------------------------------------------
# Default compilation
# ---------------------------------------------------------------------------

def _compile_default(
    default: LiftedDefault,
    rt: DoorsRelationalRuntime,
) -> Default:
    """Compile the default action."""
    if isinstance(default.action, GoToGoal):
        return Default(rt.flip_move_to_goal())

    if isinstance(default.action, GoTo):
        if isinstance(default.action.loc, GoalLoc):
            return Default(rt.flip_move_to_goal())

    raise TypeError(f"Unsupported default action: {type(default.action)}")
