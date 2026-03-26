"""Mini sweep: run surface derivation at multiple D values to find difficulty transition.

Usage:
    python scripts/run_surface_mini_sweep.py
    python scripts/run_surface_mini_sweep.py --d 5 8 12 15 20
    python scripts/run_surface_mini_sweep.py --d 5 10 15 --n-iterations 10
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

from alphazeropp.instances.doors.dsl.surface_derivation_config import (
    DoorsSurfaceDerivationConfig,
)
from alphazeropp.instances.doors.dsl.surface_grammar import count_relaxed_policies
from alphazeropp.utils.derivation_utils import run_derivation_training


def compute_optimal_reward(cfg):
    gk = cfg.game.kwargs
    num_rooms = gk["num_rooms"]
    step_penalty = gk.get("step_penalty", 0.01)
    unlock_bonus = gk.get("unlock_bonus", 0.1)
    optimal_steps = 2 * (num_rooms - 1) + 1
    return 1.0 + (num_rooms - 1) * unlock_bonus - optimal_steps * step_penalty


def run_one(D, n_iterations, sweep_dir):
    """Run surface derivation for a single D value."""
    cfg = DoorsSurfaceDerivationConfig(num_rooms=D)
    cfg.run.n_iterations = n_iterations
    cfg.run.plot_every = max(1, n_iterations // 3)

    K = D - 1
    n_policies = count_relaxed_policies(D)
    print(f"\n{'='*60}")
    print(f"  D={D}, K={K}, action_space={2*K+1}, policies={n_policies:,}")
    print(f"  {n_iterations} iterations, mcts={cfg.agent.mcts_params['n_simulations']}, "
          f"games={cfg.trainer.n_games_per_train}")
    print(f"{'='*60}\n")

    run_dir = sweep_dir / f"D{D}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.trainer.checkpoint_dir = str(run_dir / "checkpoints")
    cfg.run.plot_path = str(run_dir / "training_metrics.png")
    cfg.save(str(run_dir / "config.json"))

    optimal_reward = compute_optimal_reward(cfg)
    run_derivation_training(cfg, "surface", optimal_reward, run_dir)

    # Read back program_log to extract summary
    log_path = run_dir / "program_log.jsonl"
    entries = []
    if log_path.exists():
        with open(log_path) as f:
            entries = [json.loads(line) for line in f]

    first_solve = None
    for e in entries:
        if e.get("best_solve_rate", 0) > 0 and first_solve is None:
            first_solve = e["iteration"]

    last = entries[-1] if entries else {}
    return {
        "D": D,
        "K": K,
        "n_policies": n_policies,
        "n_iterations": n_iterations,
        "first_solve_iter": first_solve,
        "final_solve_rate": last.get("best_solve_rate", 0),
        "final_avg_reward": last.get("best_avg_reward", 0),
        "unique_programs": last.get("unique_programs", 0),
        "avg_wall_clock": (
            sum(e.get("iter_wall_clock", 0) for e in entries) / len(entries)
            if entries else 0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Surface derivation mini sweep")
    parser.add_argument("--d", type=int, nargs="+", default=[5, 8, 12, 15, 20],
                        help="D values to sweep (default: 5 8 12 15 20)")
    parser.add_argument("--n-iterations", type=int, default=15,
                        help="Iterations per D (default: 15)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(f"experiments/surface_mini_sweep_{timestamp}")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    with open(sweep_dir / "sweep_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    summaries = []
    for D in args.d:
        summary = run_one(D, args.n_iterations, sweep_dir)
        summaries.append(summary)
        with open(sweep_dir / "sweep_results.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("  MINI SWEEP RESULTS")
    print(f"{'='*80}\n")
    print(f"  {'D':>3}  {'K':>3}  {'Policies':>12}  {'Solve%':>7}  "
          f"{'Reward':>8}  {'1st Solve':>9}  {'Unique':>10}  {'Wall/iter':>9}")
    print(f"  {'---':>3}  {'---':>3}  {'--------':>12}  {'------':>7}  "
          f"{'------':>8}  {'---------':>9}  {'------':>10}  {'---------':>9}")
    for s in summaries:
        fs = s.get("first_solve_iter")
        fs_str = str(fs) if fs else "never"
        print(f"  {s['D']:>3}  {s['K']:>3}  {s['n_policies']:>12,}  "
              f"{s['final_solve_rate']:>6.0%}  "
              f"{s['final_avg_reward']:>8.3f}  "
              f"{fs_str:>9}  "
              f"{s['unique_programs']:>10,}  "
              f"{s['avg_wall_clock']:>8.1f}s")

    print(f"\nResults saved to: {sweep_dir}/")


if __name__ == "__main__":
    main()
