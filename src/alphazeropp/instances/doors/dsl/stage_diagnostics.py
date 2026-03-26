"""Diagnostics comparing Doors program representations.

1. Legacy budgeted AST derivation (budget_grammar)
2. Surface sequence DSL (surface_grammar)
3. Budget-free stage-skeleton DSL (stage_grammar)
4. Reactive sketch BT (reactive_sketch_dsl)

For each representation, reports:
  - search space size (total programs/policies)
  - number of complete candidates
  - number that solve the environment
  - number of semantically distinct programs (by action signature on frozen states)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig, doors_initial_state
from alphazeropp.synthesis.ast_nodes import Program
from alphazeropp.synthesis.interpreter import eval_program, run_policy_episode


# ---------------------------------------------------------------------------
# Semantic signature
# ---------------------------------------------------------------------------

def semantic_signature(
    prog: Program,
    states: list[np.ndarray],
) -> tuple[int, ...]:
    """Compute an action-based signature for a compiled program.

    For each state, record the action index selected.  The resulting tuple
    serves as a semantic fingerprint — programs with the same signature
    behave identically on these states.
    """
    return tuple(eval_program(prog, s) for s in states)


def reactive_semantic_signature(
    policy,
    rt,
    states: list[np.ndarray],
    memory=None,
) -> tuple[int | None, ...]:
    """Compute an action-based signature for a reactive BT policy.

    For each state, tick the BT once and record the selected action index
    (or None if no action).  The resulting tuple serves as a semantic
    fingerprint — policies with the same signature behave identically
    on these states.

    Args:
        memory: Optional ReactiveMemory for partial-map mode.
    """
    from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import (
        ReactiveContext, tick,
    )
    results = []
    for s in states:
        ctx = ReactiveContext(rt, s, memory=memory)
        result = tick(policy, ctx)
        results.append(result.action)
    return tuple(results)


# ---------------------------------------------------------------------------
# Frozen state suite
# ---------------------------------------------------------------------------

def make_frozen_state_suite(cfg: DoorsGameConfig) -> list[np.ndarray]:
    """Generate a suite of representative states for semantic comparison.

    Includes: initial state plus states with various keys picked / rooms unlocked.
    """
    states = [doors_initial_state(cfg)]

    # For each key k, add a state where key k has been picked (avail=0)
    for k in range(cfg.K):
        s = doors_initial_state(cfg).copy()
        s[cfg.M + cfg.D + k] = 0.0  # key k no longer available
        states.append(s)

    # State where agent is at key 0's location
    if cfg.K > 0:
        s = doors_initial_state(cfg).copy()
        s[cfg.start_loc] = 0.0
        s[cfg.key_loc[0]] = 1.0
        states.append(s)

    return states


# ---------------------------------------------------------------------------
# Representation diagnostics
# ---------------------------------------------------------------------------

@dataclass
class RepresentationStats:
    name: str
    total_programs: int
    complete_programs: int
    solving_programs: int
    semantically_distinct: int


def _count_solving(
    programs: list[Program],
    cfg: DoorsGameConfig,
) -> tuple[int, int]:
    """Count solving programs and semantically distinct programs."""
    x0 = doors_initial_state(cfg)
    n_sites = cfg.obs_size()
    states = make_frozen_state_suite(cfg)

    solve_count = 0
    signatures: set[tuple[int, ...]] = set()

    for prog in programs:
        sig = semantic_signature(prog, states)
        signatures.add(sig)
        env = cfg.make_env(n_sites, frozen_states=[x0])
        result = run_policy_episode(env, prog, x0=x0, is_solved=cfg.is_solved)
        if result.solved:
            solve_count += 1

    return solve_count, len(signatures)


def diagnose_surface(cfg: DoorsGameConfig) -> RepresentationStats:
    """Diagnose the surface sequence DSL representation."""
    from alphazeropp.instances.doors.dsl.surface_grammar import (
        count_relaxed_policies, enumerate_relaxed_policies,
    )
    from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy

    D = cfg.D
    total = count_relaxed_policies(D)
    policies = enumerate_relaxed_policies(D)
    programs = [compile_policy(p, cfg) for p in policies]
    solving, distinct = _count_solving(programs, cfg)

    return RepresentationStats(
        name="Surface sequence",
        total_programs=total,
        complete_programs=len(programs),
        solving_programs=solving,
        semantically_distinct=distinct,
    )


def diagnose_stage(
    cfg: DoorsGameConfig,
    max_guard_depth: int = 0,
    max_stages: int | None = None,
) -> RepresentationStats:
    """Diagnose the stage-skeleton DSL representation."""
    from alphazeropp.instances.doors.dsl.stage_grammar import (
        count_stage_programs, enumerate_stage_programs,
    )
    from alphazeropp.instances.doors.dsl.stage_compiler import compile_stage_program
    from alphazeropp.instances.doors.dsl.stage_search_cost import SearchCostModel

    K = cfg.K
    if max_stages is None:
        max_stages = 2 * K + 1

    cost_model = SearchCostModel(
        max_stages=max_stages,
        max_guard_depth=max_guard_depth,
    )
    total = count_stage_programs(cfg, cost_model)

    # Only enumerate if tractable
    if total > 10_000_000:
        return RepresentationStats(
            name=f"Stage skeleton (d≤{max_guard_depth}, s≤{max_stages})",
            total_programs=total,
            complete_programs=total,  # all stage programs are complete (no holes in enumeration)
            solving_programs=-1,  # too large to enumerate
            semantically_distinct=-1,
        )

    stage_progs = enumerate_stage_programs(cfg, cost_model)
    programs = [compile_stage_program(sp, cfg) for sp in stage_progs]
    solving, distinct = _count_solving(programs, cfg)

    return RepresentationStats(
        name=f"Stage skeleton (d≤{max_guard_depth}, s≤{max_stages})",
        total_programs=total,
        complete_programs=len(stage_progs),
        solving_programs=solving,
        semantically_distinct=distinct,
    )


def diagnose_budgeted(cfg: DoorsGameConfig) -> RepresentationStats:
    """Diagnose the legacy budgeted grammar representation.

    Only counts; full enumeration is too expensive for D>=2.
    """
    from alphazeropp.synthesis.budget_grammar import count_canonical_programs
    from alphazeropp.instances.doors.dsl.doors_config import compute_doors_derived_params

    params = compute_doors_derived_params(cfg.D)
    n_sites = params["n_sites"]
    budget = params["budget"]

    total = count_canonical_programs(n_sites, budget)

    return RepresentationStats(
        name=f"Budgeted AST (b={budget})",
        total_programs=total,
        complete_programs=total,
        solving_programs=-1,  # too expensive to enumerate
        semantically_distinct=-1,
    )


def diagnose_reactive_typed(
    cfg: DoorsGameConfig,
    n_branches: int = 4,
    mode: str = "typed",
) -> RepresentationStats:
    """Diagnose the reactive typed grammar representation.

    Args:
        cfg: DoorsGameConfig for the environment.
        n_branches: Number of branches in the Fallback node.
        mode: "typed" (strict pairing) or "raw" (cross-product).
    """
    from alphazeropp.instances.doors.dsl.reactive_branch_catalog import known_map_catalog
    from alphazeropp.instances.doors.dsl.reactive_typed_grammar import (
        count_reactive_typed_policies, enumerate_reactive_typed_policies,
        evaluate_reactive_typed_policies,
    )
    from alphazeropp.instances.doors.dsl.relational_runtime import DoorsRelationalRuntime
    from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import (
        run_reactive_episode,
    )

    catalog = known_map_catalog(mode=mode)
    total = count_reactive_typed_policies(n_branches, catalog)

    if total > 10_000_000:
        return RepresentationStats(
            name=f"Reactive typed ({mode}, N={n_branches})",
            total_programs=total,
            complete_programs=total,
            solving_programs=-1,
            semantically_distinct=-1,
        )

    rt = DoorsRelationalRuntime(cfg)
    x0 = doors_initial_state(cfg)
    states = make_frozen_state_suite(cfg)

    specs_list = enumerate_reactive_typed_policies(n_branches, catalog)
    results = evaluate_reactive_typed_policies(
        specs_list, catalog, cfg, rt, states,
    )

    solve_count = sum(1 for r in results if r["solved"])
    signatures = {tuple(r["signature"]) for r in results}

    return RepresentationStats(
        name=f"Reactive typed ({mode}, N={n_branches})",
        total_programs=total,
        complete_programs=len(specs_list),
        solving_programs=solve_count,
        semantically_distinct=len(signatures),
    )


def diagnose_reactive(cfg: DoorsGameConfig) -> RepresentationStats:
    """Diagnose the reactive sketch BT representation (24 branch permutations)."""
    import itertools
    from alphazeropp.instances.doors.dsl.relational_runtime import DoorsRelationalRuntime
    from alphazeropp.instances.doors.dsl.reactive_sketch_dsl import canonical_reactive_policy
    from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import (
        run_reactive_episode,
    )

    rt = DoorsRelationalRuntime(cfg)
    x0 = doors_initial_state(cfg)
    states = make_frozen_state_suite(cfg)

    solve_count = 0
    signatures: set[tuple[int | None, ...]] = set()

    for perm in itertools.permutations([1, 2, 3, 4]):
        policy = canonical_reactive_policy(perm)
        sig = reactive_semantic_signature(policy, rt, states)
        signatures.add(sig)
        env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
        result = run_reactive_episode(
            env, policy, rt, x0=x0, is_solved=cfg.is_solved,
        )
        if result.solved:
            solve_count += 1

    return RepresentationStats(
        name="Reactive sketch BT",
        total_programs=24,
        complete_programs=24,
        solving_programs=solve_count,
        semantically_distinct=len(signatures),
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(stats_list: list[RepresentationStats]) -> str:
    """Format a comparison table as markdown."""
    lines = [
        "| Representation | Total | Complete | Solving | Distinct |",
        "|----------------|------:|--------:|--------:|---------:|",
    ]
    for s in stats_list:
        solving = str(s.solving_programs) if s.solving_programs >= 0 else "—"
        distinct = str(s.semantically_distinct) if s.semantically_distinct >= 0 else "—"
        lines.append(
            f"| {s.name} | {s.total_programs:,} | {s.complete_programs:,} "
            f"| {solving} | {distinct} |"
        )
    return "\n".join(lines)


def run_diagnostics(D: int, max_guard_depth: int = 0) -> str:
    """Run full diagnostics for D rooms and return markdown report."""
    cfg = DoorsGameConfig(num_rooms=D)

    stats = [
        diagnose_budgeted(cfg),
        diagnose_surface(cfg),
        diagnose_stage(cfg, max_guard_depth=max_guard_depth),
    ]

    header = f"## Representation comparison for D={D}\n\n"
    return header + format_report(stats)


if __name__ == "__main__":
    for D in [2, 3]:
        print(run_diagnostics(D))
        print()
