#!/usr/bin/env python3
"""Reactive Sketch — Exhaustive Permutation Sweep.

Evaluates all 24 branch-order permutations of the reactive sketch DSL
across specified D values.  Optionally compares against all relaxed
surface DSL policies.

Usage:
    python scripts/run_reactive_sketch.py
    python scripts/run_reactive_sketch.py --d-values 2 3 4 5 --compare-surface
    python scripts/run_reactive_sketch.py --d-values 2 3 --metric avg_reward --no-plot
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

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
    run_reactive_episode,
)
from alphazeropp.instances.doors.dsl.surface_grammar import (
    canonical_policy, enumerate_relaxed_policies, count_relaxed_policies,
)
from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
from alphazeropp.synthesis.interpreter import run_policy_episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def build_config(D: int, locs_per_room: int) -> DoorsGameConfig:
    params = compute_doors_derived_params(D, locs_per_room)
    return DoorsGameConfig(
        num_rooms=D, locs_per_room=locs_per_room, horizon=params["horizon"],
    )


# ---------------------------------------------------------------------------
# Reactive sweep
# ---------------------------------------------------------------------------

def run_reactive_sweep(D: int, cfg: DoorsGameConfig, n_frozen: int) -> list[dict]:
    """Evaluate all 24 reactive sketch permutations for a given D."""
    rt = DoorsRelationalRuntime(cfg)
    frozen_states = [doors_initial_state(cfg)] * n_frozen

    results = []
    for perm in itertools.permutations([1, 2, 3, 4]):
        policy = canonical_reactive_policy(perm)
        solved_count = 0
        total_reward = 0.0
        total_steps = 0

        for x0 in frozen_states:
            env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
            r = run_reactive_episode(
                env, policy, rt, x0=x0, is_solved=cfg.is_solved,
            )
            if r.solved:
                solved_count += 1
                total_steps += r.total_env_steps
            total_reward += r.cumulative_reward

        results.append({
            "D": D,
            "dsl": "reactive",
            "order": list(perm),
            "solve_rate": solved_count / n_frozen,
            "avg_reward": round(total_reward / n_frozen, 4),
            "avg_steps": round(total_steps / solved_count, 1) if solved_count > 0 else None,
            "solved": solved_count == n_frozen,
        })

    return results


# ---------------------------------------------------------------------------
# Surface sweep
# ---------------------------------------------------------------------------

def run_surface_sweep(D: int, cfg: DoorsGameConfig, n_frozen: int) -> list[dict]:
    """Evaluate all relaxed surface policies for a given D."""
    policies = enumerate_relaxed_policies(D)
    frozen_states = [doors_initial_state(cfg)] * n_frozen

    results = []
    for i, policy in enumerate(policies):
        prog = compile_policy(policy, cfg)
        solved_count = 0
        total_reward = 0.0
        total_steps = 0

        for x0 in frozen_states:
            env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
            r = run_policy_episode(env, prog, x0=x0, is_solved=cfg.is_solved)
            if r.solved:
                solved_count += 1
                total_steps += r.total_env_steps
            total_reward += r.cumulative_reward

        results.append({
            "D": D,
            "dsl": "surface",
            "policy": policy.pretty(),
            "solve_rate": solved_count / n_frozen,
            "avg_reward": round(total_reward / n_frozen, 4),
            "avg_steps": round(total_steps / solved_count, 1) if solved_count > 0 else None,
            "solved": solved_count == n_frozen,
        })

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plots(
    all_results: dict[int, list[dict]],
    surface_results: dict[int, list[dict]] | None,
    save_path: Path,
) -> None:
    """Generate 2-panel plot: solve rate + reward distribution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plot generation")
        return

    d_values = sorted(all_results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: Solve rate by D ---
    reactive_rates = []
    for D in d_values:
        solved = sum(1 for r in all_results[D] if r["solved"])
        reactive_rates.append(solved / len(all_results[D]))

    x_pos = np.arange(len(d_values))
    width = 0.35

    bars1 = ax1.bar(
        x_pos - width / 2 if surface_results else x_pos,
        reactive_rates, width, label="Reactive Sketch", color="#4C72B0",
    )

    if surface_results:
        surface_rates = []
        for D in d_values:
            if D in surface_results and surface_results[D]:
                solved = sum(1 for r in surface_results[D] if r["solved"])
                surface_rates.append(solved / len(surface_results[D]))
            else:
                surface_rates.append(0)
        ax1.bar(
            x_pos + width / 2, surface_rates, width,
            label="Surface (relaxed)", color="#DD8452",
        )

    ax1.set_xlabel("D (rooms)")
    ax1.set_ylabel("Solve Rate")
    ax1.set_title("Fraction of Policies That Solve")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(d) for d in d_values])
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # --- Panel 2: Reward distribution by D ---
    for i, D in enumerate(d_values):
        rewards = [r["avg_reward"] for r in all_results[D]]
        solved_flags = [r["solved"] for r in all_results[D]]

        # Jitter x positions
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(rewards))
        colors = ["#2ca02c" if s else "#d62728" for s in solved_flags]
        ax2.scatter(
            [i] * len(rewards) + jitter, rewards,
            c=colors, alpha=0.6, s=30, edgecolors="none",
        )

        if surface_results and D in surface_results and surface_results[D]:
            s_rewards = [r["avg_reward"] for r in surface_results[D]]
            s_jitter = np.random.default_rng(99).uniform(-0.15, 0.15, len(s_rewards))
            ax2.scatter(
                [i] * len(s_rewards) + s_jitter, s_rewards,
                marker="D", c="#DD8452", alpha=0.7, s=25, edgecolors="none",
                label="Surface" if i == 0 else None,
            )

    # Optimal reward reference line
    for i, D in enumerate(d_values):
        optimal_steps = 2 * (D - 1) + 1
        # Approximate optimal reward: 1.0 (goal) + (D-1)*0.1 (unlock bonuses) - optimal_steps*0.01
        optimal_reward = 1.0 + (D - 1) * 0.1 - optimal_steps * 0.01

    ax2.set_xlabel("D (rooms)")
    ax2.set_ylabel("Avg Reward")
    ax2.set_title("Reward Distribution (green=solved, red=failed)")
    ax2.set_xticks(range(len(d_values)))
    ax2.set_xticklabels([str(d) for d in d_values])
    ax2.grid(axis="y", alpha=0.3)
    if surface_results:
        ax2.legend()

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reactive Sketch — Exhaustive Permutation Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_reactive_sketch.py\n"
            "  python scripts/run_reactive_sketch.py --d-values 2 3 4 5 --compare-surface\n"
            "  python scripts/run_reactive_sketch.py --d-values 2 3 --metric avg_reward --no-plot\n"
        ),
    )
    parser.add_argument(
        "--d-values", nargs="+", type=int, default=[2, 3, 4],
        help="Room counts to evaluate (default: 2 3 4)",
    )
    parser.add_argument(
        "--locs-per-room", type=int, default=2,
        help="Locations per room (default: 2)",
    )
    parser.add_argument(
        "--metric", choices=["avg_reward", "solve_rate", "weighted"],
        default="weighted",
        help="Evaluation metric for ranking (default: weighted)",
    )
    parser.add_argument(
        "--n-frozen", type=int, default=1,
        help="Number of frozen initial states per evaluation (default: 1)",
    )
    parser.add_argument(
        "--exp-dir", type=str, default=None,
        help="Override experiment directory path",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--compare-surface", action="store_true",
        help="Also evaluate all relaxed surface policies for comparison",
    )
    args = parser.parse_args()

    d_values = sorted(args.d_values)
    t_start = time.time()

    # --- Experiment directory ---
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d_range = f"D{d_values[0]}-{d_values[-1]}"
        exp_dir = Path("experiments") / "reactive_sketch" / f"{timestamp}_{d_range}_{args.metric}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # --- Banner ---
    section("Reactive Sketch \u2014 Exhaustive Permutation Sweep")
    print(f"  D values:        {d_values}")
    print(f"  Locs per room:   {args.locs_per_room}")
    print(f"  Metric:          {args.metric}")
    print(f"  Frozen states:   {args.n_frozen}")
    print(f"  Compare surface: {'yes' if args.compare_surface else 'no'}")
    print(f"  Experiment dir:  {exp_dir}")

    # --- Save config ---
    config = {
        "d_values": d_values,
        "locs_per_room": args.locs_per_room,
        "metric": args.metric,
        "n_frozen": args.n_frozen,
        "compare_surface": args.compare_surface,
        "timestamp": datetime.now().isoformat(),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Run sweeps ---
    all_reactive: dict[int, list[dict]] = {}
    all_surface: dict[int, list[dict]] = {}
    summaries: list[dict] = []

    for D in d_values:
        K = D - 1
        cfg = build_config(D, args.locs_per_room)
        optimal_steps = 2 * K + 1

        section(f"D={D} (K={K}, 24 permutations)")

        # Reactive sweep
        reactive_results = run_reactive_sweep(D, cfg, args.n_frozen)
        all_reactive[D] = reactive_results

        for r in reactive_results:
            order_str = f"({','.join(str(x) for x in r['order'])})"
            solved_str = "yes" if r["solved"] else "no "
            steps_str = f"{r['avg_steps']:.0f}" if r["avg_steps"] is not None else " -"
            print(
                f"  [SWEEP D={D}]  {order_str:>12}  "
                f"solved={solved_str}  steps={steps_str:>3}  "
                f"reward={r['avg_reward']:+.4f}"
            )

        # Reactive summary
        n_solved = sum(1 for r in reactive_results if r["solved"])
        solving = [r for r in reactive_results if r["solved"]]
        n_optimal = sum(
            1 for r in solving
            if r["avg_steps"] is not None and r["avg_steps"] == optimal_steps
        )
        best_reward = max(r["avg_reward"] for r in reactive_results)
        worst_solving_reward = (
            min(r["avg_reward"] for r in solving) if solving else None
        )

        print()
        print(
            f"  [RESULT D={D}] Solved: {n_solved}/24 ({n_solved/24*100:.1f}%) | "
            f"Optimal: {n_optimal}/{n_solved} at {optimal_steps} steps"
        )
        if worst_solving_reward is not None:
            print(
                f"               Best reward: {best_reward:+.4f} | "
                f"Worst solving: {worst_solving_reward:+.4f}"
            )

        summary = {
            "D": D,
            "K": K,
            "reactive_solved": n_solved,
            "reactive_total": 24,
            "reactive_optimal": n_optimal,
            "reactive_best_reward": best_reward,
            "optimal_steps": optimal_steps,
        }

        # Surface sweep (optional)
        if args.compare_surface:
            surface_count = count_relaxed_policies(D)
            if surface_count <= 100_000:
                print(f"\n  [SURFACE D={D}] Evaluating {surface_count} relaxed policies...")
                surface_results = run_surface_sweep(D, cfg, args.n_frozen)
                all_surface[D] = surface_results

                s_solved = sum(1 for r in surface_results if r["solved"])
                s_best = max(r["avg_reward"] for r in surface_results)
                print(
                    f"  [SURFACE D={D}] Solved: {s_solved}/{surface_count} "
                    f"({s_solved/surface_count*100:.1f}%) | "
                    f"Best reward: {s_best:+.4f}"
                )
                summary["surface_solved"] = s_solved
                summary["surface_total"] = surface_count
                summary["surface_best_reward"] = s_best
            else:
                print(
                    f"\n  [SURFACE D={D}] Skipped — {surface_count} policies "
                    f"exceeds enumeration limit"
                )
                summary["surface_solved"] = None
                summary["surface_total"] = surface_count
                summary["surface_best_reward"] = None

        summaries.append(summary)

        # Save per-D results
        with open(exp_dir / "results.jsonl", "a") as f:
            for r in reactive_results:
                f.write(json.dumps(r) + "\n")
            if D in all_surface:
                for r in all_surface[D]:
                    f.write(json.dumps(r) + "\n")

    # --- Summary table ---
    section("Summary")

    if args.compare_surface:
        header = f"  {'D':>3}  {'Reactive':>12}  {'Optimal':>9}  {'Best Reward':>12}  {'Surface':>16}"
        sep = f"  {'─'*3}  {'─'*12}  {'─'*9}  {'─'*12}  {'─'*16}"
    else:
        header = f"  {'D':>3}  {'Reactive':>12}  {'Optimal':>9}  {'Best Reward':>12}"
        sep = f"  {'─'*3}  {'─'*12}  {'─'*9}  {'─'*12}"

    print(header)
    print(sep)

    for s in summaries:
        reactive_str = f"{s['reactive_solved']}/24"
        optimal_str = f"{s['reactive_optimal']}/{s['reactive_solved']}"
        reward_str = f"{s['reactive_best_reward']:+.4f}"

        line = f"  {s['D']:>3}  {reactive_str:>12}  {optimal_str:>9}  {reward_str:>12}"

        if args.compare_surface:
            if s.get("surface_solved") is not None:
                surface_str = f"{s['surface_solved']}/{s['surface_total']}"
            else:
                surface_str = f"?/{s.get('surface_total', '?')}"
            line += f"  {surface_str:>16}"

        print(line)

    # --- Save summary ---
    with open(exp_dir / "summary.jsonl", "w") as f:
        for s in summaries:
            f.write(json.dumps(s) + "\n")

    # --- Plot ---
    if not args.no_plot:
        d_range = f"D{d_values[0]}-{d_values[-1]}"
        plot_name = f"permutation_sweep_{d_range}_{args.metric}.png"
        plot_path = exp_dir / plot_name
        generate_plots(
            all_reactive,
            all_surface if args.compare_surface else None,
            plot_path,
        )

    # --- Timing ---
    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Results:    {exp_dir}")


if __name__ == "__main__":
    main()
