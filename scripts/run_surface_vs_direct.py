"""
Surface Derivation vs Direct Play — Controlled Comparison.

Compares AlphaZero on SurfaceDerivationGame vs DoorsDirectGame across
D in {2, 3, 5, 10} with three variants each: trained, frozen, random.

Usage:
    python scripts/run_surface_vs_direct.py --d 2 3 --variants trained random
    python scripts/run_surface_vs_direct.py --d 2 3 5 10 --variants trained frozen random
    python scripts/run_surface_vs_direct.py --d 2 --n-iterations 5 --n-games 10  # quick smoke
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import copy
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.surface_derivation_game import (
    SurfaceDerivationGame,
)
from alphazeropp.instances.doors.dsl.surface_derivation_config import (
    DoorsSurfaceDerivationConfig,
)
from alphazeropp.instances.doors.config import DoorsDirectConfig
from alphazeropp.instances.doors.game import DoorsDirectGame
from alphazeropp.instances.doors.oracle import optimal_return
from alphazeropp.synthesis.derivation_game import UniformPolicyValueNet
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator
from alphazeropp.synthesis.derivation_network import DerivationPolicyValueNet
from alphazeropp.instances.doors.network import DoorsDirectNet
from alphazeropp.core.agent import Agent
from alphazeropp.training.trainer import Trainer
from alphazeropp.training.evaluator import Evaluator
from alphazeropp.training.gated_trainer import GatedTrainer
from alphazeropp.instances.doors.dsl.surface_grammar import count_relaxed_policies


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_surface_run(D: int, variant: str, n_iterations: int, n_games: int,
                      n_sims: int):
    """Build surface derivation game + training stack for given variant."""
    cfg = DoorsSurfaceDerivationConfig(num_rooms=D)
    cfg.run.n_iterations = n_iterations
    cfg.trainer.n_games_per_train = n_games
    cfg.agent.mcts_params["n_simulations"] = n_sims
    cfg.trainer.n_procs = 1  # Sequential for reproducibility

    game, net, agent, trainer, evaluator = cfg.build()

    if variant == "random":
        # Replace net with uniform policy
        net = UniformPolicyValueNet(game.action_space.n)
        agent = Agent(game=game, net=net,
                      mcts_params=cfg.agent.mcts_params,
                      reward_discount=1.0)
        trainer = Trainer(agent=agent, net=net, game=game,
                          n_games_per_train=n_games,
                          n_past_iterations_to_train=20, n_procs=1,
                          use_tree_reuse=True)
    elif variant == "frozen":
        # Keep initialized net but don't update weights
        pass  # Net is randomly initialized; we skip training below

    return game, net, agent, trainer, evaluator, cfg


def build_direct_run(D: int, variant: str, n_iterations: int, n_games: int,
                     n_sims: int):
    """Build direct play game + training stack for given variant."""
    cfg = DoorsDirectConfig(num_rooms=D, locs_per_room=2)
    cfg.run.n_iterations = n_iterations
    cfg.trainer.n_games_per_train = n_games
    cfg.agent.mcts_params["n_simulations"] = n_sims
    cfg.trainer.n_procs = 1

    game, net, agent, trainer, evaluator = cfg.build()

    if variant == "random":
        net = UniformPolicyValueNet(game.action_space.n)
        agent = Agent(game=game, net=net,
                      mcts_params=cfg.agent.mcts_params,
                      reward_discount=1.0)
        trainer = Trainer(agent=agent, net=net, game=game,
                          n_games_per_train=n_games,
                          n_past_iterations_to_train=20, n_procs=1)
    elif variant == "frozen":
        pass

    return game, net, agent, trainer, evaluator, cfg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_surface_game(game: SurfaceDerivationGame, agent: Agent,
                          n_eval_games: int = 20) -> dict:
    """Play n_eval_games and collect metrics."""
    results = []
    for _ in range(n_eval_games):
        g = game.clone()
        g.reset_wrapper()
        experience, cum_reward, step_infos = agent.play_one_round(
            g, max_moves=100, add_noise=False)
        solved = cum_reward >= 0.99  # solve_rate metric
        results.append({
            "reward": cum_reward,
            "solved": solved,
            "steps": len(experience),
        })

    solve_rate = sum(1 for r in results if r["solved"]) / len(results)
    avg_reward = np.mean([r["reward"] for r in results])
    return {
        "solve_rate": solve_rate,
        "avg_reward": float(avg_reward),
        "n_games": n_eval_games,
    }


def evaluate_direct_game(game: DoorsDirectGame, agent: Agent,
                         n_eval_games: int = 20) -> dict:
    """Play n_eval_games on direct game."""
    results = []
    for _ in range(n_eval_games):
        g = game.clone()
        g.reset_wrapper()
        experience, cum_reward, step_infos = agent.play_one_round(
            g, max_moves=100, add_noise=False)
        # Solved = terminated (reached goal), not truncated (hit horizon)
        solved = g.terminated is True
        results.append({
            "reward": cum_reward,
            "solved": solved,
            "steps": len(experience),
        })

    solve_rate = sum(1 for r in results if r["solved"]) / len(results)
    avg_reward = np.mean([r["reward"] for r in results])
    return {
        "solve_rate": solve_rate,
        "avg_reward": float(avg_reward),
        "n_games": n_eval_games,
    }


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(D: int, game_type: str, variant: str, n_iterations: int,
               n_games: int, n_sims: int, exp_dir: Path) -> dict:
    """Run a single (D, game_type, variant) experiment."""
    label = f"D={D} {game_type} {variant}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")

    if game_type == "surface":
        game, net, agent, trainer, evaluator, cfg = build_surface_run(
            D, variant, n_iterations, n_games, n_sims)
    else:
        game, net, agent, trainer, evaluator, cfg = build_direct_run(
            D, variant, n_iterations, n_games, n_sims)

    gated_trainer = GatedTrainer(trainer, evaluator, cfg.run.accept_threshold)

    # Metrics per iteration
    iter_metrics = []
    best_reward = float("-inf")
    first_solver_iter = None

    for i in range(n_iterations):
        t0 = time.time()

        if variant in ("frozen", "random"):
            # Self-play only — no network training
            trainer._collect_training_examples()
            score = 0.5
        else:
            score, accepted = gated_trainer.train_iteration()

        wall = time.time() - t0

        # Evaluate
        if game_type == "surface":
            eval_result = evaluate_surface_game(game, agent, n_eval_games=10)
            # Also check leaf evaluator cache for best program
            leaf_eval = game.leaf_evaluator
            stats = leaf_eval.stats()
            unique_programs = stats["unique_programs"]
        else:
            eval_result = evaluate_direct_game(game, agent, n_eval_games=10)
            unique_programs = 0

        if eval_result["avg_reward"] > best_reward:
            best_reward = eval_result["avg_reward"]

        if eval_result["solve_rate"] > 0 and first_solver_iter is None:
            first_solver_iter = i + 1

        entry = {
            "iteration": i + 1,
            "D": D,
            "game_type": game_type,
            "variant": variant,
            "solve_rate": eval_result["solve_rate"],
            "avg_reward": eval_result["avg_reward"],
            "best_reward": best_reward,
            "unique_programs": unique_programs,
            "wall_clock_s": round(wall, 2),
        }
        iter_metrics.append(entry)

        sr_str = f"{eval_result['solve_rate']:.0%}"
        print(f"  [{label}] iter {i+1}/{n_iterations}: "
              f"solve={sr_str}, reward={eval_result['avg_reward']:.3f}, "
              f"wall={wall:.1f}s")

    # Write per-iteration metrics
    run_dir = exp_dir / f"{game_type}_{variant}_D{D}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "iterations.jsonl", "w") as f:
        for entry in iter_metrics:
            f.write(json.dumps(entry) + "\n")

    summary = {
        "D": D,
        "game_type": game_type,
        "variant": variant,
        "n_iterations": n_iterations,
        "final_solve_rate": iter_metrics[-1]["solve_rate"] if iter_metrics else 0,
        "final_avg_reward": iter_metrics[-1]["avg_reward"] if iter_metrics else 0,
        "best_reward": best_reward,
        "first_solver_iter": first_solver_iter,
        "unique_programs": iter_metrics[-1].get("unique_programs", 0) if iter_metrics else 0,
    }

    print(f"\n  [{label}] DONE: solve={summary['final_solve_rate']:.0%}, "
          f"best_reward={best_reward:.3f}, "
          f"first_solver={first_solver_iter or 'never'}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Surface vs Direct comparison")
    parser.add_argument("--d", type=int, nargs="+", default=[2, 3],
                        help="Room counts to test (default: 2 3)")
    parser.add_argument("--variants", nargs="+",
                        default=["trained", "random"],
                        choices=["trained", "frozen", "random"],
                        help="Variants to run (default: trained random)")
    parser.add_argument("--n-iterations", type=int, default=10,
                        help="Training iterations per run")
    parser.add_argument("--n-games", type=int, default=20,
                        help="Self-play games per iteration")
    parser.add_argument("--n-sims", type=int, default=40,
                        help="MCTS simulations per move")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/surface_vs_direct_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    with open(exp_dir / "experiment_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    all_summaries = []

    for D in args.d:
        for game_type in ["surface", "direct"]:
            for variant in args.variants:
                summary = run_single(
                    D, game_type, variant,
                    n_iterations=args.n_iterations,
                    n_games=args.n_games,
                    n_sims=args.n_sims,
                    exp_dir=exp_dir,
                )
                all_summaries.append(summary)

                # Append to running results file
                with open(exp_dir / "results.jsonl", "a") as f:
                    f.write(json.dumps(summary) + "\n")

    # Print final comparison table
    print(f"\n\n{'='*80}")
    print("  COMPARISON RESULTS")
    print(f"{'='*80}\n")
    print(f"  {'D':>3}  {'Game':>8}  {'Variant':>8}  {'Solve%':>7}  "
          f"{'Reward':>8}  {'1st Solve':>9}  {'Unique':>8}")
    print(f"  {'---':>3}  {'--------':>8}  {'--------':>8}  {'------':>7}  "
          f"{'--------':>8}  {'---------':>9}  {'------':>8}")
    for s in all_summaries:
        fs = s.get("first_solver_iter")
        fs_str = str(fs) if fs else "never"
        print(f"  {s['D']:>3}  {s['game_type']:>8}  {s['variant']:>8}  "
              f"{s['final_solve_rate']:>6.0%}  "
              f"{s['final_avg_reward']:>8.3f}  "
              f"{fs_str:>9}  "
              f"{s['unique_programs']:>8}")

    print(f"\nResults saved to: {exp_dir}/")


if __name__ == "__main__":
    main()
