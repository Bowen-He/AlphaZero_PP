#!/usr/bin/env python3
"""Cross-game comparison: Doors derivation vs Surface vs Reactive.

Runs all three derivation games at the same D and produces a side-by-side
comparison CSV, markdown report, and 2x2 plot.

Usage:
    python scripts/compare_derivation_games.py --D 3
    python scripts/compare_derivation_games.py --D 3 5 10 --n-iterations 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np


# ---------------------------------------------------------------------------
# Per-game runners — each returns a result dict
# ---------------------------------------------------------------------------

def run_reactive_game(D: int, n_iterations: int, n_games: int,
                      n_simulations: int) -> dict:
    """Run reactive BT composition game."""
    from alphazeropp.instances.doors.dsl.reactive_training_config import (
        DoorsReactiveDerivationConfig,
    )

    config = DoorsReactiveDerivationConfig(
        num_rooms=D, n_branches=4, catalog_mode="typed",
        n_simulations=n_simulations,
        n_games_per_train=n_games,
        n_iterations=n_iterations,
    )
    game, net, agent, trainer, evaluator = config.build()
    le = game.leaf_evaluator

    t0 = time.time()
    solve_rates = []
    best_rewards = []

    for i in range(n_iterations):
        examples = trainer._collect_training_examples()
        flat = [item for sublist in examples for item in sublist]
        trainer._train_network(flat)

        # Compute iteration metrics
        rewards = [ex[0][1][1] for ex in examples if ex]
        sr = sum(1 for r in rewards if r > 0.5) / len(rewards) if rewards else 0
        avg_r = sum(rewards) / len(rewards) if rewards else 0
        solve_rates.append(sr)
        best_rewards.append(avg_r)

    elapsed = time.time() - t0
    le_stats = le.stats() if le else {}
    n_solving = sum(
        1 for m in (le._full_cache.values() if le else [])
        if m.get("n_solved", 0) > 0
    )

    return {
        "game": "reactive",
        "D": D,
        "iterations": n_iterations,
        "unique_programs": le_stats.get("unique_programs", 0),
        "solving_programs": n_solving,
        "final_solve_rate": solve_rates[-1] if solve_rates else 0,
        "best_avg_reward": max(best_rewards) if best_rewards else 0,
        "wall_clock": round(elapsed, 1),
        "solve_rate_trajectory": solve_rates,
        "reward_trajectory": best_rewards,
    }


def run_surface_game(D: int, n_iterations: int, n_games: int,
                     n_simulations: int) -> dict:
    """Run surface rule sequencing game."""
    from alphazeropp.instances.doors.dsl.surface_derivation_config import (
        DoorsSurfaceDerivationConfig,
    )

    config = DoorsSurfaceDerivationConfig()
    config.game.kwargs["num_rooms"] = D
    K = D - 1
    max_steps = 2 * K + 1
    config.net.kwargs["budget"] = max_steps
    config.net.kwargs["n_sites"] = max_steps
    config.net.kwargs["action_size"] = max_steps
    config.agent.mcts_params["n_simulations"] = n_simulations
    config.trainer.n_games_per_train = n_games
    config.run.n_iterations = n_iterations

    game, net, agent, trainer, evaluator = config.build()
    le = game.leaf_evaluator

    t0 = time.time()
    solve_rates = []
    best_rewards = []

    for i in range(n_iterations):
        examples = trainer._collect_training_examples()
        flat = [item for sublist in examples for item in sublist]
        trainer._train_network(flat)

        rewards = [ex[0][1][1] for ex in examples if ex]
        sr = sum(1 for r in rewards if r > 0.5) / len(rewards) if rewards else 0
        avg_r = sum(rewards) / len(rewards) if rewards else 0
        solve_rates.append(sr)
        best_rewards.append(avg_r)

    elapsed = time.time() - t0
    le_stats = le.stats() if le else {}
    n_solving = sum(
        1 for m in (le._full_cache.values() if le else [])
        if m.get("solve_rate", 0) >= 1.0
    )

    return {
        "game": "surface",
        "D": D,
        "iterations": n_iterations,
        "unique_programs": le_stats.get("unique_programs", 0),
        "solving_programs": n_solving,
        "final_solve_rate": solve_rates[-1] if solve_rates else 0,
        "best_avg_reward": max(best_rewards) if best_rewards else 0,
        "wall_clock": round(elapsed, 1),
        "solve_rate_trajectory": solve_rates,
        "reward_trajectory": best_rewards,
    }


def run_doors_game(D: int, n_iterations: int, n_games: int,
                   n_simulations: int) -> dict:
    """Run doors derivation game (factored macro for D>=10, factored otherwise)."""
    if D >= 10:
        from alphazeropp.instances.doors.dsl.derivation_config import (
            DoorsFactoredD10MacroConfig as ConfigCls,
        )
    else:
        from alphazeropp.instances.doors.dsl.derivation_config import (
            DoorsFactoredDerivationConfig as ConfigCls,
        )

    config = ConfigCls()
    from alphazeropp.instances.doors.dsl.doors_config import (
        compute_doors_derived_params,
    )
    derived = compute_doors_derived_params(D, 2)
    config.game.kwargs["num_rooms"] = D
    config.game.kwargs["n_sites"] = derived["n_sites"]
    config.game.kwargs["budget"] = derived["budget"]
    config.game.kwargs["horizon"] = derived["horizon"]
    config.net.kwargs["n_sites"] = derived["n_sites"]
    config.net.kwargs["budget"] = derived["budget"]
    config.agent.mcts_params["n_simulations"] = n_simulations
    config.trainer.n_games_per_train = n_games
    config.run.n_iterations = n_iterations

    game, net, agent, trainer, evaluator = config.build()
    le = game.leaf_evaluator

    t0 = time.time()
    solve_rates = []
    best_rewards = []

    for i in range(n_iterations):
        examples = trainer._collect_training_examples()
        flat = [item for sublist in examples for item in sublist]
        trainer._train_network(flat)

        rewards = [ex[0][1][1] for ex in examples if ex]
        sr = sum(1 for r in rewards if r > 0.5) / len(rewards) if rewards else 0
        avg_r = sum(rewards) / len(rewards) if rewards else 0
        solve_rates.append(sr)
        best_rewards.append(avg_r)

    elapsed = time.time() - t0
    le_stats = le.stats() if le else {}
    n_solving = sum(
        1 for m in (le._full_cache.values() if le else [])
        if m.get("solve_rate", 0) >= 1.0
    )

    return {
        "game": "doors_factored",
        "D": D,
        "iterations": n_iterations,
        "unique_programs": le_stats.get("unique_programs", 0),
        "solving_programs": n_solving,
        "final_solve_rate": solve_rates[-1] if solve_rates else 0,
        "best_avg_reward": max(best_rewards) if best_rewards else 0,
        "wall_clock": round(elapsed, 1),
        "solve_rate_trajectory": solve_rates,
        "reward_trajectory": best_rewards,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(all_results: dict[int, list[dict]], save_dir: Path):
    """Generate a 2x2 comparison plot across games and D values."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed. Skipping plot.")
        return

    game_colors = {
        "reactive": "#2196F3",
        "surface": "#FF9800",
        "doors_factored": "#4CAF50",
    }

    for D, results in all_results.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Cross-Game Comparison -- D={D}",
                     fontsize=14, fontweight="bold")

        games = [r["game"] for r in results]
        colors = [game_colors.get(g, "#999999") for g in games]

        # Panel 1: Final solve rate
        ax1 = axes[0, 0]
        vals = [r["final_solve_rate"] for r in results]
        ax1.bar(games, vals, color=colors)
        ax1.set_ylabel("Solve Rate")
        ax1.set_title("Final Solve Rate")
        for i, v in enumerate(vals):
            ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

        # Panel 2: Wall clock
        ax2 = axes[0, 1]
        vals = [r["wall_clock"] for r in results]
        ax2.bar(games, vals, color=colors)
        ax2.set_ylabel("Seconds")
        ax2.set_title("Wall Clock")
        for i, v in enumerate(vals):
            ax2.text(i, v + 0.1, f"{v:.1f}s", ha="center", fontsize=9)

        # Panel 3: Unique programs
        ax3 = axes[1, 0]
        vals = [r["unique_programs"] for r in results]
        ax3.bar(games, vals, color=colors)
        ax3.set_ylabel("Unique Programs")
        ax3.set_title("Unique Programs Evaluated")
        for i, v in enumerate(vals):
            ax3.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

        # Panel 4: Reward trajectory
        ax4 = axes[1, 1]
        for r in results:
            traj = r.get("reward_trajectory", [])
            if traj:
                ax4.plot(range(1, len(traj)+1), traj,
                         color=game_colors.get(r["game"], "#999"),
                         label=r["game"], linewidth=2)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Avg Reward")
        ax4.set_title("Reward Trajectory")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        path = save_dir / f"D{D}_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] Saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare doors, surface, and reactive derivation games")
    parser.add_argument("--D", type=int, nargs="+", default=[3],
                        help="Room counts (e.g., --D 3 5 10)")
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--n-games", type=int, default=20)
    parser.add_argument("--n-simulations", type=int, default=40)
    parser.add_argument("--output-dir", type=str,
                        default="results/cross_game_comparison")
    parser.add_argument("--skip-doors", action="store_true",
                        help="Skip doors factored game (slow for large D)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[int, list[dict]] = {}

    for D in args.D:
        print(f"\n{'='*70}")
        print(f"  Cross-Game Comparison: D={D}")
        print(f"  iterations={args.n_iterations}, games/iter={args.n_games}")
        print(f"{'='*70}")

        results = []

        # Reactive
        print(f"\n--- Reactive BT (D={D}) ---")
        r = run_reactive_game(D, args.n_iterations, args.n_games,
                              args.n_simulations)
        print(f"  solve_rate={r['final_solve_rate']:.3f}, "
              f"unique={r['unique_programs']}, "
              f"solving={r['solving_programs']}, "
              f"time={r['wall_clock']}s")
        results.append(r)

        # Surface
        print(f"\n--- Surface Rules (D={D}) ---")
        r = run_surface_game(D, args.n_iterations, args.n_games,
                             args.n_simulations)
        print(f"  solve_rate={r['final_solve_rate']:.3f}, "
              f"unique={r['unique_programs']}, "
              f"solving={r['solving_programs']}, "
              f"time={r['wall_clock']}s")
        results.append(r)

        # Doors (optional)
        if not args.skip_doors:
            print(f"\n--- Doors Factored (D={D}) ---")
            r = run_doors_game(D, args.n_iterations, args.n_games,
                               args.n_simulations)
            print(f"  solve_rate={r['final_solve_rate']:.3f}, "
                  f"unique={r['unique_programs']}, "
                  f"solving={r['solving_programs']}, "
                  f"time={r['wall_clock']}s")
            results.append(r)

        all_results[D] = results

        # Save CSV (without trajectory columns)
        csv_path = output_dir / f"D{D}_comparison.csv"
        csv_fields = ["game", "D", "iterations", "unique_programs",
                      "solving_programs", "final_solve_rate",
                      "best_avg_reward", "wall_clock"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {csv_path}")

        # Save markdown
        md_path = output_dir / f"D{D}_comparison.md"
        with open(md_path, "w") as f:
            f.write(f"# Cross-Game Comparison -- D={D}\n\n")
            f.write(f"Settings: {args.n_iterations} iterations, "
                    f"{args.n_games} games/iter, "
                    f"{args.n_simulations} MCTS sims\n\n")
            f.write("| " + " | ".join(csv_fields) + " |\n")
            f.write("|" + "|".join("---" for _ in csv_fields) + "|\n")
            for r in results:
                f.write("| " + " | ".join(
                    str(r.get(c, "")) for c in csv_fields
                ) + " |\n")
        print(f"Saved: {md_path}")

    # Generate plots
    plot_comparison(all_results, output_dir)

    # Summary table
    print(f"\n{'='*70}")
    print("  OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Game':<20} {'D':>3} {'Solve%':>7} {'Unique':>8} "
          f"{'Solving':>8} {'Time(s)':>8}")
    print(f"  {'-'*18:<20} {'---':>3} {'-----':>7} {'------':>8} "
          f"{'-------':>8} {'-------':>8}")
    for D, results in all_results.items():
        for r in results:
            print(f"  {r['game']:<20} {r['D']:>3} "
                  f"{r['final_solve_rate']:>6.1%} "
                  f"{r['unique_programs']:>8,} "
                  f"{r['solving_programs']:>8} "
                  f"{r['wall_clock']:>8.1f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
