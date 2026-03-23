#!/usr/bin/env python3
"""Demo script for the reactive sketch DSL.

Shows:
  1. The canonical reactive policy (human-readable BT)
  2. Step-by-step trace for D=2 and D=3
  3. All 24 branch-order permutations: which solve, how many steps
  4. Optimal ordering identification
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_dsl import (
    canonical_reactive_policy,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import (
    run_reactive_episode, format_reactive_trace,
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    # ------------------------------------------------------------------
    # 1. The canonical reactive policy
    # ------------------------------------------------------------------
    section("Canonical Reactive Policy (BT sketch)")
    policy = canonical_reactive_policy()
    print(f"  {policy.pretty()}")
    print()
    print("  Branch order: B1(Pick), B2(GoToKey), B3(GoToGoal), B4(Explore)")
    print("  Search space: 4! = 24 permutations")

    # ------------------------------------------------------------------
    # 2. D=2 trace
    # ------------------------------------------------------------------
    section("D=2 Behavioral Trace (order 1,2,3,4)")
    cfg_d2 = DoorsGameConfig(num_rooms=2, locs_per_room=2)
    rt_d2 = DoorsRelationalRuntime(cfg_d2)
    x0 = doors_initial_state(cfg_d2)
    env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
    result = run_reactive_episode(
        env, policy, rt_d2, x0=x0, is_solved=cfg_d2.is_solved,
    )
    print(format_reactive_trace(result, policy))

    # ------------------------------------------------------------------
    # 3. D=3 trace
    # ------------------------------------------------------------------
    section("D=3 Behavioral Trace (order 1,2,3,4)")
    params_d3 = compute_doors_derived_params(3, 2)
    cfg_d3 = DoorsGameConfig(
        num_rooms=3, locs_per_room=2, horizon=params_d3["horizon"],
    )
    rt_d3 = DoorsRelationalRuntime(cfg_d3)
    x0_d3 = doors_initial_state(cfg_d3)
    env_d3 = cfg_d3.make_env(cfg_d3.obs_size(), frozen_states=[x0_d3])
    result_d3 = run_reactive_episode(
        env_d3, policy, rt_d3, x0=x0_d3, is_solved=cfg_d3.is_solved,
    )
    print(format_reactive_trace(result_d3, policy))

    # ------------------------------------------------------------------
    # 4. All 24 permutations sweep
    # ------------------------------------------------------------------
    for D in [2, 3]:
        section(f"All 24 Permutations — D={D}")
        params = compute_doors_derived_params(D, 2)
        cfg = DoorsGameConfig(
            num_rooms=D, locs_per_room=2, horizon=params["horizon"],
        )
        rt = DoorsRelationalRuntime(cfg)
        x0 = doors_initial_state(cfg)

        print(f"  {'Order':>16}  {'Solved':>6}  {'Steps':>6}")
        print(f"  {'─' * 16}  {'─' * 6}  {'─' * 6}")

        solved_count = 0
        optimal_orders = []
        optimal_steps = 2 * (D - 1) + 1

        for perm in itertools.permutations([1, 2, 3, 4]):
            p = canonical_reactive_policy(perm)
            env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
            r = run_reactive_episode(
                env, p, rt, x0=x0, is_solved=cfg.is_solved,
            )
            status = "yes" if r.solved else "no"
            steps = r.total_env_steps if r.solved else "-"
            print(f"  {str(perm):>16}  {status:>6}  {str(steps):>6}")
            if r.solved:
                solved_count += 1
                if r.total_env_steps == optimal_steps:
                    optimal_orders.append(perm)

        print()
        print(f"  Solved: {solved_count}/24")
        print(f"  Optimal ({optimal_steps} steps): {len(optimal_orders)} orderings")
        for o in optimal_orders:
            print(f"    {o}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    section("Summary")
    print("  The reactive sketch uses BT semantics:")
    print("    While(Not(GoalReached), Fallback(B1..B4))")
    print()
    print("  Each tick evaluates branches left-to-right.")
    print("  The first succeeding branch determines the action.")
    print("  Synthesis searches branch ORDER only (24 candidates).")
    print(f"  Optimal step count: 2*(D-1) + 1 for all D tested.")


if __name__ == "__main__":
    main()
