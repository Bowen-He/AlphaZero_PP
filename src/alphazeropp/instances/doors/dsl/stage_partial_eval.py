"""Partial semantic signatures for stage programs with holes.

Evaluates what a partial StageProgram would do on each frozen state:
- concrete action index if determined
- UNRESOLVED (-1) if the outcome depends on an unfilled hole
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig, doors_initial_state
from alphazeropp.instances.doors.dsl.stage_dsl import (
    GuardExpr, GuardHole, ActionHole,
)
from alphazeropp.instances.doors.dsl.stage_compiler import compile_guard
from alphazeropp.instances.doors.dsl.surface_compiler import compile_action
from alphazeropp.instances.doors.dsl.stage_partial import PartialStageProgram
from alphazeropp.synthesis.interpreter import eval_condition


UNRESOLVED = -1


def make_rich_frozen_state_suite(cfg: DoorsGameConfig) -> list[np.ndarray]:
    """A richer state suite that distinguishes more guard atoms.

    Extends make_frozen_state_suite with cross-product states:
    - agent at key k's location AND key k already picked (unavailable)
    This distinguishes AtKeyLoc(k) from PickReady(k).
    """
    from alphazeropp.instances.doors.dsl.stage_diagnostics import make_frozen_state_suite
    states = make_frozen_state_suite(cfg)

    # Add: agent at key k location + key k unavailable
    for k in range(cfg.K):
        s = doors_initial_state(cfg).copy()
        s[cfg.start_loc] = 0.0
        s[cfg.key_loc[k]] = 1.0         # agent at key k
        s[cfg.M + cfg.D + k] = 0.0      # key k already picked
        states.append(s)

    return states


# ---------------------------------------------------------------------------
# Guard compilation cache
# ---------------------------------------------------------------------------

class _GuardCache:
    """Cache compiled guard AST conditions keyed by (guard, cfg) identity."""

    def __init__(self):
        self._cache: dict = {}

    def compile(self, guard: GuardExpr, cfg: DoorsGameConfig):
        key = (guard, id(cfg))
        if key not in self._cache:
            self._cache[key] = compile_guard(guard, cfg)
        return self._cache[key]


_global_cache = _GuardCache()


# ---------------------------------------------------------------------------
# Partial semantic signature
# ---------------------------------------------------------------------------

def partial_semantic_signature(
    partial: PartialStageProgram,
    frozen_states: list[np.ndarray],
    cfg: DoorsGameConfig,
    *,
    cache: _GuardCache | None = None,
) -> tuple[int, ...]:
    """Compute the action signature of a partial program on frozen states.

    For each state, walk stages top-to-bottom:
    - Complete guard evaluates true → action (concrete or UNRESOLVED if ActionHole)
    - Complete guard evaluates false → continue to next stage
    - GuardHole → UNRESOLVED
    - Fall through all stages + finalized → default action index
    - Fall through + not finalized → UNRESOLVED
    """
    if cache is None:
        cache = _global_cache

    result = []
    for state in frozen_states:
        action = _eval_partial_on_state(partial, state, cfg, cache)
        result.append(action)
    return tuple(result)


def _eval_partial_on_state(
    partial: PartialStageProgram,
    state: np.ndarray,
    cfg: DoorsGameConfig,
    cache: _GuardCache,
) -> int:
    """Evaluate a partial program on a single state."""
    for stage in partial.stages:
        guard = stage.guard

        if isinstance(guard, GuardHole):
            return UNRESOLVED

        # Guard is complete — compile and evaluate
        compiled_guard = cache.compile(guard, cfg)
        if eval_condition(compiled_guard, state):
            # This stage fires
            if isinstance(stage.action, ActionHole):
                return UNRESOLVED
            compiled_action = compile_action(stage.action, cfg)
            return compiled_action.index
        # Guard is false — fall through to next stage

    # Fell through all stages
    if partial.structure_finalized:
        compiled_default = compile_action(partial.default_action, cfg)
        return compiled_default.index
    else:
        return UNRESOLVED
