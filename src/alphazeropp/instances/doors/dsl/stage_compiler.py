"""Compiler from the stage-skeleton DSL to the raw AST layer.

Delegates atom compilation to surface_compiler.compile_condition and
compile_action.  Only adds recursion for GuardNot/GuardAnd combinators.
"""

from __future__ import annotations

from alphazeropp.synthesis.ast_nodes import (
    Not as AstNot, And as AstAnd, Ite, Default,
    Condition, Program,
)
from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
from alphazeropp.instances.doors.dsl.surface_compiler import (
    compile_condition as _compile_atom,
    compile_action,
)
from alphazeropp.instances.doors.dsl.stage_dsl import (
    GuardExpr, GuardNot, GuardAnd, GuardHole, ActionHole,
    Stage, StageProgram,
)
from alphazeropp.instances.doors.dsl.surface_dsl import SurfaceCondition


def compile_guard(guard: GuardExpr, cfg: DoorsGameConfig) -> Condition:
    """Compile a guard expression to a raw AST Condition.

    - Atoms (AtKeyLoc, KeyAvail, etc.) delegate to surface_compiler.
    - GuardNot/GuardAnd recurse, producing ast_nodes.Not/And.
    """
    if isinstance(guard, GuardNot):
        return AstNot(compile_guard(guard.child, cfg))

    if isinstance(guard, GuardAnd):
        return AstAnd(compile_guard(guard.left, cfg), compile_guard(guard.right, cfg))

    if isinstance(guard, GuardHole):
        raise ValueError("Cannot compile a GuardHole — resolve it first")

    # Must be a guard atom (SurfaceCondition)
    return _compile_atom(guard, cfg)  # type: ignore[arg-type]


def compile_stage_program(prog: StageProgram, cfg: DoorsGameConfig) -> Program:
    """Compile a StageProgram to a complete raw AST Program.

    Right-fold: stages chain as nested Ite, with default_action as Default.
    Raises ValueError if any holes remain.
    """
    if not prog.is_complete():
        raise ValueError("Cannot compile a StageProgram with unresolved holes")

    # Start with the default action
    result: Program = Default(compile_action(prog.default_action, cfg))

    # Right-fold: last stage wraps default, then second-to-last wraps that, etc.
    for stage in reversed(prog.stages):
        cond = compile_guard(stage.guard, cfg)
        act = compile_action(stage.action, cfg)  # type: ignore[arg-type]
        result = Ite(cond, act, result)

    return result
