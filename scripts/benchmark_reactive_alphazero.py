#!/usr/bin/env python3
"""Benchmark reactive AlphaZero against baselines.

Compares: random sampling, exhaustive enumeration, AZ from scratch,
AZ from pretrained checkpoint.

Usage:
    python scripts/benchmark_reactive_alphazero.py --D 3
    python scripts/benchmark_reactive_alphazero.py --D 3 5 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.reactive_branch_catalog import (
    known_map_catalog,
)
from alphazeropp.instances.doors.dsl.reactive_leaf_evaluator import (
    ReactiveLeafEvaluator,
)
from alphazeropp.instances.doors.dsl.reactive_typed_grammar import (
    enumerate_reactive_typed_policies,
)
from alphazeropp.instances.doors.dsl.reactive_training_config import (
    DoorsReactiveDerivationConfig,
)
from alphazeropp.instances.doors.dsl.stage_diagnostics import (
    make_frozen_state_suite, reactive_semantic_signature,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime,
)


# ---------------------------------------------------------------------------
# Baseline: Random legal sampling
# ---------------------------------------------------------------------------

def benchmark_random(
    n_samples: int,
    n_branches: int,
    catalog,
    evaluator: ReactiveLeafEvaluator,
) -> dict:
    """Sample random legal policies and evaluate."""
    legal = catalog.legal_pairs()
    rng = np.random.default_rng(42)

    first_solve_idx = None
    n_solved = 0
    rewards = []

    for i in range(n_samples):
        specs = tuple(
            legal[rng.integers(len(legal))] for _ in range(n_branches)
        )
        r = evaluator(specs)
        m = evaluator.get_all_metrics(specs)
        rewards.append(r)
        if m["n_solved"] > 0:
            n_solved += 1
            if first_solve_idx is None:
                first_solve_idx = i

    return {
        "method": "random",
        "candidates": n_samples,
        "solving": n_solved,
        "solve_rate": n_solved / n_samples if n_samples > 0 else 0,
        "first_solve_idx": first_solve_idx,
        "avg_reward": float(np.mean(rewards)) if rewards else 0,
    }


# ---------------------------------------------------------------------------
# Baseline: Exhaustive enumeration
# ---------------------------------------------------------------------------

def benchmark_exhaustive(
    n_branches: int,
    catalog,
    evaluator: ReactiveLeafEvaluator,
) -> dict:
    """Enumerate all policies."""
    specs_list = enumerate_reactive_typed_policies(n_branches, catalog)

    first_solve_idx = None
    n_solved = 0

    for i, specs in enumerate(specs_list):
        m = evaluator.get_all_metrics(specs)
        if m["n_solved"] > 0:
            n_solved += 1
            if first_solve_idx is None:
                first_solve_idx = i

    return {
        "method": "exhaustive",
        "candidates": len(specs_list),
        "solving": n_solved,
        "solve_rate": n_solved / len(specs_list) if specs_list else 0,
        "first_solve_idx": first_solve_idx,
        "avg_reward": 0,  # not computed for efficiency
    }


# ---------------------------------------------------------------------------
# Baseline: AlphaZero (from scratch or pretrained)
# ---------------------------------------------------------------------------

def benchmark_alphazero(
    D: int,
    n_branches: int,
    catalog_mode: str,
    n_iterations: int,
    n_games: int,
    n_simulations: int,
    pretrained: str | None = None,
    label: str = "az_scratch",
) -> dict:
    """Run AlphaZero and measure solve performance."""
    config = DoorsReactiveDerivationConfig(
        num_rooms=D,
        n_branches=n_branches,
        catalog_mode=catalog_mode,
        pretrained_checkpoint=pretrained,
        n_simulations=n_simulations,
        n_games_per_train=n_games,
        n_iterations=n_iterations,
    )

    game, net, agent, trainer, evaluator_obj = config.build()
    le = game.leaf_evaluator

    t0 = time.time()
    total_games = 0
    first_solve_iter = None

    for iteration in range(n_iterations):
        examples = trainer._collect_training_examples()
        total_games += len(examples)

        # Check if any game found a solver
        if le is not None:
            for specs, metrics in le._full_cache.items():
                if metrics.get("n_solved", 0) > 0:
                    if first_solve_iter is None:
                        first_solve_iter = iteration

        flat = [item for sublist in examples for item in sublist]
        trainer._train_network(flat)

    elapsed = time.time() - t0
    le_stats = le.stats() if le else {}

    return {
        "method": label,
        "candidates": le_stats.get("unique_programs", 0),
        "solving": sum(
            1 for m in (le._full_cache.values() if le else [])
            if m.get("n_solved", 0) > 0
        ),
        "first_solve_iter": first_solve_iter,
        "wall_clock": round(elapsed, 1),
        "total_games": total_games,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_benchmark_comparison(results: list[dict], D: int, save_path: str):
    """Generate a 2x2 comparison plot for benchmark results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed. Skipping plot.")
        return

    methods = [r["method"] for r in results]
    colors = {
        "random": "#7fbfff",
        "exhaustive": "#ffb347",
        "az_scratch": "#87de87",
        "az_pretrained": "#ff6961",
    }
    bar_colors = [colors.get(m, "#cccccc") for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Reactive AlphaZero Benchmark — D={D}",
                 fontsize=14, fontweight="bold")

    # Panel 1: Solve rate
    ax1 = axes[0, 0]
    solve_rates = [r.get("solve_rate", 0) for r in results]
    ax1.bar(methods, solve_rates, color=bar_colors)
    ax1.set_ylabel("Solve Rate")
    ax1.set_title("Solve Rate")
    ax1.set_ylim(0, max(solve_rates) * 1.2 if max(solve_rates) > 0 else 1)
    for i, v in enumerate(solve_rates):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Panel 2: Wall clock
    ax2 = axes[0, 1]
    wall_clocks = [r.get("wall_clock", 0) for r in results]
    ax2.bar(methods, wall_clocks, color=bar_colors)
    ax2.set_ylabel("Seconds")
    ax2.set_title("Wall Clock")
    for i, v in enumerate(wall_clocks):
        ax2.text(i, v + 0.1, f"{v:.1f}s", ha="center", fontsize=9)

    # Panel 3: Candidates evaluated
    ax3 = axes[1, 0]
    candidates = [r.get("candidates", 0) for r in results]
    ax3.bar(methods, candidates, color=bar_colors)
    ax3.set_ylabel("Candidates")
    ax3.set_title("Candidates Evaluated")
    for i, v in enumerate(candidates):
        ax3.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    # Panel 4: Solving policies found
    ax4 = axes[1, 1]
    solving = [r.get("solving", 0) for r in results]
    ax4.bar(methods, solving, color=bar_colors)
    ax4.set_ylabel("Solving Policies")
    ax4.set_title("Solving Policies Found")
    for i, v in enumerate(solving):
        ax4.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark reactive AlphaZero vs baselines",
    )
    parser.add_argument("--D", type=int, nargs="+", default=[3],
                        help="Room counts to evaluate (e.g., --D 3 5 10)")
    parser.add_argument("--n-branches", type=int, default=4)
    parser.add_argument("--mode", type=str, default="typed")
    parser.add_argument("--az-iterations", type=int, default=5)
    parser.add_argument("--az-games", type=int, default=10)
    parser.add_argument("--az-simulations", type=int, default=20)
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/reactive_benchmark")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for D in args.D:
        print(f"\n{'='*70}")
        print(f"  Benchmark: D={D}, N={args.n_branches}, mode={args.mode}")
        print(f"{'='*70}")

        # Setup
        params = compute_doors_derived_params(D, 2)
        cfg = DoorsGameConfig(
            num_rooms=D, locs_per_room=2, horizon=params["horizon"],
        )
        catalog = known_map_catalog(mode=args.mode)
        x0 = doors_initial_state(cfg)
        evaluator = ReactiveLeafEvaluator(
            catalog, cfg, [x0], is_solved=cfg.is_solved,
        )

        results = []

        # 1. Random
        print("\n--- Random sampling ---")
        t0 = time.time()
        r = benchmark_random(args.random_samples, args.n_branches, catalog, evaluator)
        r["wall_clock"] = round(time.time() - t0, 1)
        print(f"  {r}")
        results.append(r)

        # 2. Exhaustive
        print("\n--- Exhaustive enumeration ---")
        t0 = time.time()
        r = benchmark_exhaustive(args.n_branches, catalog, evaluator)
        r["wall_clock"] = round(time.time() - t0, 1)
        print(f"  {r}")
        results.append(r)

        # 3. AZ from scratch
        print("\n--- AlphaZero from scratch ---")
        r = benchmark_alphazero(
            D, args.n_branches, args.mode,
            args.az_iterations, args.az_games, args.az_simulations,
            label="az_scratch",
        )
        print(f"  {r}")
        results.append(r)

        # 4. AZ pretrained (if checkpoint provided)
        if args.pretrained:
            print("\n--- AlphaZero pretrained ---")
            r = benchmark_alphazero(
                D, args.n_branches, args.mode,
                args.az_iterations, args.az_games, args.az_simulations,
                pretrained=args.pretrained,
                label="az_pretrained",
            )
            print(f"  {r}")
            results.append(r)

        # Save CSV
        csv_path = output_dir / f"D{D}_comparison.csv"
        all_keys = list(dict.fromkeys(k for r in results for k in r.keys()))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {csv_path}")

        # Save markdown
        md_path = output_dir / f"D{D}_comparison.md"
        with open(md_path, "w") as f:
            f.write(f"# Reactive AlphaZero Benchmark -- D={D}\n\n")
            f.write("| " + " | ".join(all_keys) + " |\n")
            f.write("|" + "|".join("---" for _ in all_keys) + "|\n")
            for r in results:
                f.write("| " + " | ".join(str(r.get(c, "")) for c in all_keys) + " |\n")
        print(f"Saved: {md_path}")

        # Save plot
        plot_path = output_dir / f"D{D}_comparison.png"
        plot_benchmark_comparison(results, D, str(plot_path))


if __name__ == "__main__":
    main()
