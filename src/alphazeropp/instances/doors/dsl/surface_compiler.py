"""Compiler from the typed surface DSL to the raw AST layer.

Every function is config-driven via DoorsGameConfig — no hardcoded
observation or action indices.  The output uses only existing AST node
types (Flip, IsZero, Not, And, Ite, Default) and can be evaluated
directly by the existing interpreter.
"""

from __future__ import annotations

from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
    Condition, Program,
)
from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
from alphazeropp.instances.doors.dsl.surface_dsl import (
    AtKeyLoc, KeyAvail, RoomLocked, PickReady, NeedKey,
    Pick, MoveToKey, MoveToGoal,
    PickRule, MoveRule, GoalRule,
    SurfaceCondition, SurfaceAction, SurfaceRule, SurfacePolicy,
)


# ---------------------------------------------------------------------------
# Condition compilation
# ---------------------------------------------------------------------------

def compile_condition(cond: SurfaceCondition, cfg: DoorsGameConfig) -> Condition:
    """Compile a surface condition to a raw AST Condition."""
    if isinstance(cond, AtKeyLoc):
        _check_key(cond.k, cfg)
        return Not(IsZero(cfg.key_loc[cond.k]))

    if isinstance(cond, KeyAvail):
        _check_key(cond.k, cfg)
        return Not(IsZero(cfg.M + cfg.D + cond.k))

    if isinstance(cond, RoomLocked):
        _check_room(cond.r, cfg)
        return IsZero(cfg.M + cond.r)

    if isinstance(cond, PickReady):
        _check_key(cond.k, cfg)
        return And(
            compile_condition(AtKeyLoc(cond.k), cfg),
            compile_condition(KeyAvail(cond.k), cfg),
        )

    if isinstance(cond, NeedKey):
        _check_key(cond.k, cfg)
        return IsZero(cfg.M + cfg.key_unlocks[cond.k])

    raise TypeError(f"Unknown surface condition: {type(cond)}")


# ---------------------------------------------------------------------------
# Action compilation
# ---------------------------------------------------------------------------

def compile_action(act: SurfaceAction, cfg: DoorsGameConfig) -> Flip:
    """Compile a surface action to a raw AST Flip."""
    if isinstance(act, Pick):
        _check_key(act.k, cfg)
        return Flip(cfg.M + act.k)

    if isinstance(act, MoveToKey):
        _check_key(act.k, cfg)
        return Flip(cfg.key_loc[act.k])

    if isinstance(act, MoveToGoal):
        return Flip(cfg.goal_loc)

    raise TypeError(f"Unknown surface action: {type(act)}")


# ---------------------------------------------------------------------------
# Rule compilation
# ---------------------------------------------------------------------------

def compile_rule(
    rule: SurfaceRule,
    else_prog: Program | None,
    cfg: DoorsGameConfig,
) -> Program:
    """Compile a surface rule, chaining to *else_prog*."""
    if isinstance(rule, GoalRule):
        return Default(compile_action(MoveToGoal(), cfg))

    if isinstance(rule, PickRule):
        assert else_prog is not None
        return Ite(
            compile_condition(PickReady(rule.k), cfg),
            compile_action(Pick(rule.k), cfg),
            else_prog,
        )

    if isinstance(rule, MoveRule):
        assert else_prog is not None
        return Ite(
            compile_condition(NeedKey(rule.k), cfg),
            compile_action(MoveToKey(rule.k), cfg),
            else_prog,
        )

    raise TypeError(f"Unknown surface rule: {type(rule)}")


# ---------------------------------------------------------------------------
# Policy compilation
# ---------------------------------------------------------------------------

def compile_policy(policy: SurfacePolicy, cfg: DoorsGameConfig) -> Program:
    """Compile a SurfacePolicy to a complete raw AST Program.

    Chains rules right-to-left: GoalRule becomes the innermost Default,
    then each preceding rule wraps it as an Ite.
    """
    prog = compile_rule(policy.rules[-1], None, cfg)
    for rule in reversed(policy.rules[:-1]):
        prog = compile_rule(rule, prog, cfg)
    return prog


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def pretty_compiled(policy: SurfacePolicy, cfg: DoorsGameConfig) -> str:
    """Return the surface DSL string alongside the compiled raw AST string."""
    prog = compile_policy(policy, cfg)
    return (
        f"Surface: {policy.pretty()}\n"
        f"Compiled:\n{prog.pretty()}"
    )


# ---------------------------------------------------------------------------
# Bounds checking
# ---------------------------------------------------------------------------

def _check_key(k: int, cfg: DoorsGameConfig) -> None:
    if not (0 <= k < cfg.K):
        raise ValueError(f"Key index {k} out of range [0, {cfg.K})")


def _check_room(r: int, cfg: DoorsGameConfig) -> None:
    if not (0 <= r < cfg.D):
        raise ValueError(f"Room index {r} out of range [0, {cfg.D})")
