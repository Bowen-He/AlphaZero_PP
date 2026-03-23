"""External search-cost model for the stage-skeleton DSL.

Cost is metadata / search constraint — it never appears in syntax or
observation state.  This module provides cost functions and a cost model
dataclass used by the grammar enumerator.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphazeropp.instances.doors.dsl.stage_dsl import (
    GuardExpr, GuardNot, GuardAnd, GuardHole,
    Stage, StageProgram,
)


# ---------------------------------------------------------------------------
# Guard depth / cost
# ---------------------------------------------------------------------------

def guard_depth(guard: GuardExpr) -> int:
    """Nesting depth of a guard expression (atoms have depth 0)."""
    if isinstance(guard, GuardNot):
        return 1 + guard_depth(guard.child)
    if isinstance(guard, GuardAnd):
        return 1 + max(guard_depth(guard.left), guard_depth(guard.right))
    return 0  # atom


def guard_node_count(guard: GuardExpr) -> int:
    """Number of nodes in a guard expression (each atom/combinator = 1)."""
    if isinstance(guard, GuardNot):
        return 1 + guard_node_count(guard.child)
    if isinstance(guard, GuardAnd):
        return 1 + guard_node_count(guard.left) + guard_node_count(guard.right)
    return 1  # atom


# ---------------------------------------------------------------------------
# Program-level cost
# ---------------------------------------------------------------------------

def stage_program_cost(prog: StageProgram) -> int:
    """Total cost: sum of guard node counts across all stages."""
    return sum(guard_node_count(s.guard) for s in prog.stages
               if not isinstance(s.guard, GuardHole))


# ---------------------------------------------------------------------------
# Search cost model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchCostModel:
    """Constraints on the stage-program search space.

    These are search-time bounds, not part of the DSL syntax.
    """
    max_stages: int
    max_guard_depth: int = 0  # 0 = atoms only, 1 = one level of Not/And
    max_guard_nodes: int | None = None  # per-guard node cap (None = no cap)

    def admits_guard(self, guard: GuardExpr) -> bool:
        """Check if a guard satisfies this cost model."""
        if guard_depth(guard) > self.max_guard_depth:
            return False
        if self.max_guard_nodes is not None and guard_node_count(guard) > self.max_guard_nodes:
            return False
        return True

    def admits_program(self, prog: StageProgram) -> bool:
        """Check if a stage program satisfies this cost model."""
        if prog.num_stages > self.max_stages:
            return False
        for stage in prog.stages:
            if not isinstance(stage.guard, GuardHole) and not self.admits_guard(stage.guard):
                return False
        return True
