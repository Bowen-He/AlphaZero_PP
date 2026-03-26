"""Best-first search over partial stage programs.

Shared search algorithm used by all hole-selection policies.
The only variable across runs is the HoleSelectionPolicy.
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig, doors_initial_state
from alphazeropp.instances.doors.dsl.stage_dsl import StageProgram, GuardHole, ActionHole
from alphazeropp.instances.doors.dsl.stage_compiler import compile_stage_program
from alphazeropp.instances.doors.dsl.stage_search_cost import (
    SearchCostModel, stage_program_cost,
)
from alphazeropp.instances.doors.dsl.stage_grammar import (
    enumerate_guards, enumerate_actions,
)
from alphazeropp.instances.doors.dsl.stage_partial import (
    PartialStageProgram, HoleRef,
    StructureHole, GuardHoleRef, ActionHoleRef,
    AddStage, Finalize, apply_filling, partial_depth,
)
from alphazeropp.instances.doors.dsl.stage_partial_eval import (
    partial_semantic_signature, UNRESOLVED, _GuardCache,
    make_rich_frozen_state_suite,
)
from alphazeropp.instances.doors.dsl.stage_hole_selection import HoleSelectionPolicy
from alphazeropp.synthesis.interpreter import run_policy_episode


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SearchStats:
    candidates_expanded: int = 0
    complete_evaluated: int = 0
    semantically_distinct_partial: int = 0
    semantically_distinct_complete: int = 0
    solving_programs: int = 0
    time_to_first_solver: float | None = None
    time_total: float = 0.0
    dedup_hits: int = 0
    dedup_misses: int = 0
    heap_max_size: int = 0
    branch_factor_by_depth: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class SearchResult:
    policy_name: str
    stats: SearchStats
    solving_programs: list[StageProgram] = field(default_factory=list)
    complete_programs: list[StageProgram] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Search node (for heapq)
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _SearchNode:
    priority: float
    tiebreaker: int
    partial: PartialStageProgram = field(compare=False)
    depth: int = field(compare=False)


# ---------------------------------------------------------------------------
# Best-first search
# ---------------------------------------------------------------------------

def best_first_search(
    cfg: DoorsGameConfig,
    cost_model: SearchCostModel,
    policy: HoleSelectionPolicy,
    frozen_states: list[np.ndarray] | None = None,
    *,
    max_expansions: int = 1_000_000,
    dedup: bool = True,
) -> SearchResult:
    """Run depth-first search with the given hole-selection policy.

    Priority = -depth (explore deeper nodes first) with LIFO tiebreaking.
    This ensures the search reaches complete programs quickly.

    If *frozen_states* is None, uses make_rich_frozen_state_suite(cfg)
    which includes cross-product states for better dedup discrimination.
    """
    if frozen_states is None:
        frozen_states = make_rich_frozen_state_suite(cfg)
    t0 = time.time()
    stats = SearchStats()
    result = SearchResult(policy_name=policy.name, stats=stats)

    # Pre-compute valid fillings
    guards = enumerate_guards(cfg, cost_model.max_guard_depth)
    guards = [g for g in guards if cost_model.admits_guard(g)]
    actions = enumerate_actions(cfg)

    # Guard compilation cache
    cache = _GuardCache()

    # Dedup: (signature, num_stages, structure_finalized) -> best cost seen
    # We include structural info because two partials with the same
    # all-UNRESOLVED signature but different stage counts are NOT equivalent.
    DedupKey = tuple  # (semantic_sig, n_stages, finalized)
    seen: dict[DedupKey, float] = {}

    # Evaluation setup
    x0 = doors_initial_state(cfg)
    n_sites = cfg.obs_size()

    # Initialize
    counter = 0
    initial = PartialStageProgram(
        stages=(),
        max_stages=cost_model.max_stages,
        structure_finalized=False,
    )
    heap: list[_SearchNode] = []
    heapq.heappush(heap, _SearchNode(priority=0, tiebreaker=-counter, partial=initial, depth=0))
    counter += 1

    while heap and stats.candidates_expanded < max_expansions:
        stats.heap_max_size = max(stats.heap_max_size, len(heap))
        node = heapq.heappop(heap)
        partial = node.partial
        depth = node.depth
        stats.candidates_expanded += 1

        # Terminal check
        if partial.is_terminal():
            prog = partial.to_stage_program()
            stats.complete_evaluated += 1
            result.complete_programs.append(prog)

            # Evaluate
            ast = compile_stage_program(prog, cfg)
            env = cfg.make_env(n_sites, frozen_states=[x0])
            ep = run_policy_episode(env, ast, x0=x0, is_solved=cfg.is_solved)
            if ep.solved:
                stats.solving_programs += 1
                result.solving_programs.append(prog)
                if stats.time_to_first_solver is None:
                    stats.time_to_first_solver = time.time() - t0
            continue

        # Select hole
        hole_ref = policy.select_hole(partial)

        # Generate children
        children = _expand(partial, hole_ref, guards, actions, cost_model)
        stats.branch_factor_by_depth[depth].append(len(children))

        for child in children:
            child_cost = _cost(child)

            # Cost model check: number of stages
            if child.stages and len(child.stages) > cost_model.max_stages:
                continue

            # Dedup: only dedup when the signature is fully resolved
            # (no UNRESOLVED entries). Partial programs with UNRESOLVED
            # entries may diverge when holes are filled differently.
            if dedup:
                sig = partial_semantic_signature(child, frozen_states, cfg, cache=cache)
                if UNRESOLVED not in sig:
                    dedup_key = (sig, len(child.stages), child.structure_finalized)
                    if dedup_key in seen and seen[dedup_key] <= child_cost:
                        stats.dedup_hits += 1
                        continue
                    seen[dedup_key] = child_cost
                stats.dedup_misses += 1

            heapq.heappush(heap, _SearchNode(
                priority=-(depth + 1),
                tiebreaker=-counter,
                partial=child,
                depth=depth + 1,
            ))
            counter += 1

    stats.time_total = time.time() - t0
    stats.semantically_distinct_partial = len(seen) if dedup else -1

    # Count distinct complete signatures
    complete_sigs = set()
    for prog in result.complete_programs:
        from alphazeropp.instances.doors.dsl.stage_diagnostics import semantic_signature
        ast = compile_stage_program(prog, cfg)
        sig = semantic_signature(ast, frozen_states)
        complete_sigs.add(sig)
    stats.semantically_distinct_complete = len(complete_sigs)

    return result


def _expand(
    partial: PartialStageProgram,
    hole_ref: HoleRef,
    guards: list,
    actions: list,
    cost_model: SearchCostModel,
) -> list[PartialStageProgram]:
    """Generate all children by filling the given hole."""
    children = []

    if isinstance(hole_ref, StructureHole):
        # Can add a stage (if under max_stages) or finalize
        if len(partial.stages) < partial.max_stages:
            children.append(apply_filling(partial, hole_ref, AddStage()))
        # Can always finalize (even with 0 stages — just the default action)
        children.append(apply_filling(partial, hole_ref, Finalize()))

    elif isinstance(hole_ref, GuardHoleRef):
        for guard in guards:
            children.append(apply_filling(partial, hole_ref, guard))

    elif isinstance(hole_ref, ActionHoleRef):
        for action in actions:
            children.append(apply_filling(partial, hole_ref, action))

    return children


def _cost(partial: PartialStageProgram) -> float:
    """Cost of a partial program (holes count as 0)."""
    total = 0
    for stage in partial.stages:
        if not isinstance(stage.guard, GuardHole):
            from alphazeropp.instances.doors.dsl.stage_search_cost import guard_node_count
            total += guard_node_count(stage.guard)
    return float(total)
