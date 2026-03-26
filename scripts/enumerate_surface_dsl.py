#!/usr/bin/env python3
"""Enumerate DoorsStageDSL(D) policies and compare against generic grammar.

Reports:
  1. D=2 layout parameters
  2. Surface DSL counts (canonical + relaxed)
  3. Generic budget grammar counts for comparison
  4. Search-space reduction factor
  5. Behavioral trace of compiled D=2 canonical policy
  6. Scaling table for D=2..6
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.surface_compiler import (
    compile_policy, pretty_compiled,
)
from alphazeropp.instances.doors.dsl.surface_grammar import (
    canonical_policy, count_relaxed_policies, enumerate_relaxed_policies,
)
from alphazeropp.synthesis.budget_grammar import (
    count_canonical_programs, count_programs,
)
from alphazeropp.synthesis.interpreter import (
    run_policy_episode, format_trace,
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    # ------------------------------------------------------------------
    # 1. D=2 Layout
    # ------------------------------------------------------------------
    section("D=2 Layout Parameters")
    D = 2
    cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2)
    params = compute_doors_derived_params(D, 2)
    print(f"  D (rooms)       = {D}")
    print(f"  K (keys)        = {cfg.K}")
    print(f"  M (locations)   = {cfg.M}")
    print(f"  n_sites         = {cfg.obs_size()}")
    print(f"  optimal_budget  = {params['optimal_nodes']}")
    print(f"  search_budget   = {params['budget']} (1.5x headroom)")
    print(f"  optimal_steps   = {params['optimal_steps']}")
    print(f"  goal_loc        = {cfg.goal_loc}")
    print(f"  key_loc         = {cfg.key_loc}")
    print(f"  key_unlocks     = {cfg.key_unlocks}")

    # ------------------------------------------------------------------
    # 2. Surface DSL Counts
    # ------------------------------------------------------------------
    section("Surface DSL (Typed Grammar)")
    canon = canonical_policy(D)
    print(f"  Canonical policy:  {canon.pretty()}")
    print(f"  Canonical count:   1")
    relaxed_count = count_relaxed_policies(D)
    print(f"  Relaxed count:     {relaxed_count}")

    # Compile and show the canonical policy
    print(f"\n  Compiled canonical policy:")
    print(f"  {pretty_compiled(canon, cfg)}")

    # ------------------------------------------------------------------
    # 3. Generic Grammar Comparison
    # ------------------------------------------------------------------
    section("Generic Budget Grammar Comparison (D=2)")
    n_sites = cfg.obs_size()
    optimal_budget = params["optimal_nodes"]
    search_budget = params["budget"]

    print(f"  {'Budget':>8}  {'Full':>12}  {'Canonical':>12}")
    print(f"  {'------':>8}  {'----':>12}  {'---------':>12}")
    total_full = 0
    total_canon = 0
    for b in range(2, search_budget + 1):
        full = count_programs(n_sites, b)
        canon_count = count_canonical_programs(n_sites, b)
        total_full += full
        total_canon += canon_count
        if full > 0:
            print(f"  {b:>8}  {full:>12,}  {canon_count:>12,}")

    print(f"  {'':>8}  {'--------':>12}  {'--------':>12}")
    print(f"  {'TOTAL':>8}  {total_full:>12,}  {total_canon:>12,}")

    at_optimal = count_canonical_programs(n_sites, optimal_budget)
    print(f"\n  At optimal budget ({optimal_budget}): {at_optimal:,} canonical programs")

    # ------------------------------------------------------------------
    # 4. Reduction Factor
    # ------------------------------------------------------------------
    section("Search Space Reduction")
    print(f"  Generic canonical (budget 2..{search_budget}): {total_canon:>12,} programs")
    print(f"  Generic canonical (budget {optimal_budget} only):  {at_optimal:>12,} programs")
    print(f"  Surface DSL relaxed:                   {relaxed_count:>12,} programs")
    print(f"  Surface DSL canonical:                 {'1':>12} program")
    if total_canon > 0:
        print(f"\n  Reduction (total → relaxed):  {total_canon:,} → {relaxed_count}")
        print(f"  Reduction (optimal → canon):  {at_optimal:,} → 1")

    # ------------------------------------------------------------------
    # 5. Behavioral Trace
    # ------------------------------------------------------------------
    section("D=2 Canonical Policy — Behavioral Trace")
    prog = compile_policy(canon, cfg)
    x0 = doors_initial_state(cfg)
    env = cfg.make_env(n_sites, frozen_states=[x0])
    result = run_policy_episode(env, prog, x0=x0, is_solved=cfg.is_solved)
    print(format_trace(result, program=prog))
    print(f"  Solved: {result.solved}")
    print(f"  Steps:  {result.total_env_steps}")

    # ------------------------------------------------------------------
    # 6. Scaling Table
    # ------------------------------------------------------------------
    section("Scaling Table: D=2..6")
    print(f"  {'D':>3}  {'K':>3}  {'Canon':>8}  {'Relaxed':>10}  "
          f"{'Generic(opt_budget)':>20}  {'Generic(total)':>16}")
    print(f"  {'---':>3}  {'---':>3}  {'-----':>8}  {'-------':>10}  "
          f"{'-------------------':>20}  {'---------':>16}")
    for d in range(2, 7):
        p = compute_doors_derived_params(d, 2)
        ns = p["n_sites"]
        ob = p["optimal_nodes"]
        sb = p["budget"]
        rc = count_relaxed_policies(d)
        gc_opt = count_canonical_programs(ns, ob)
        gc_total = sum(count_canonical_programs(ns, b) for b in range(2, sb + 1))
        print(f"  {d:>3}  {d-1:>3}  {'1':>8}  {rc:>10,}  "
              f"{gc_opt:>20,}  {gc_total:>16,}")

    print()


if __name__ == "__main__":
    main()
