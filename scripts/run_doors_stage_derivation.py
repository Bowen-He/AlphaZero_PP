#!/usr/bin/env python3
"""
Doors Stage Derivation -- AlphaZero Hole-Filling Search.

Synthesises Doors navigation policies via stage-skeleton MCTS + AlphaZero.
The stage game fills holes in an if-elif-else skeleton:
  StructureHole → add stage or finalize
  GuardHoleRef  → choose guard atom
  ActionHoleRef → choose typed action

Usage:
    python scripts/run_doors_stage_derivation.py
    python scripts/run_doors_stage_derivation.py --non-interactive
    python scripts/run_doors_stage_derivation.py --d 3 --non-interactive
    python scripts/run_doors_stage_derivation.py --hole-policy structure_first --non-interactive
    python scripts/run_doors_stage_derivation.py --seeds 42 43 --non-interactive
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from alphazeropp.instances.doors.dsl.stage_derivation_config import (
    DoorsStageDerivationConfig,
)
from alphazeropp.instances.doors.dsl.stage_grammar import (
    count_stage_programs, enumerate_guards, enumerate_actions,
)
from alphazeropp.instances.doors.dsl.stage_search_cost import SearchCostModel
from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
from alphazeropp.instances.doors.oracle import optimal_return
from alphazeropp.synthesis.leaf_evaluator import VALID_METRICS
from alphazeropp.utils.interactive_config import (
    build_param_list, interactive_edit, attr_setter, dict_setter,
)
from alphazeropp.utils.derivation_utils import run_derivation_training


# ---------------------------------------------------------------------------
# Config display & interactive editing
# ---------------------------------------------------------------------------

def _build_sections(cfg, seed_state=None):
    """Return sections for Stage DerivationGame config."""
    def _set_num_rooms(val):
        nr = int(val)
        cfg.game.kwargs["num_rooms"] = nr
        new_K = nr - 1
        new_max_stages = 2 * new_K + 1
        cfg.game.kwargs["max_stages"] = new_max_stages
        max_decisions = 3 * new_max_stages + 1
        cfg.game.kwargs["horizon"] = max(15, (2 * new_K + 1) * 5)
        cfg.game.kwargs["n_sites"] = max_decisions
        cfg.net.kwargs["budget"] = max_decisions
        cfg.net.kwargs["n_sites"] = max_decisions

    params = [
        # Problem
        ("num_rooms", cfg.game.kwargs["num_rooms"], _set_num_rooms,
         "Number of rooms (D) -- auto-updates max_stages", None),
        ("max_stages", cfg.game.kwargs["max_stages"],
         dict_setter(cfg.game.kwargs, "max_stages"),
         "Max stages in if-elif chain (default 2K+1)", None),
        ("max_guard_depth", cfg.game.kwargs["max_guard_depth"],
         dict_setter(cfg.game.kwargs, "max_guard_depth"),
         "Max guard nesting (0=atoms, 1=Not/And)", [0, 1]),
        ("hole_policy", cfg.game.kwargs.get("hole_policy", "leftmost"),
         dict_setter(cfg.game.kwargs, "hole_policy"),
         "Hole selection policy", ["leftmost", "structure_first"]),
        # Leaf evaluation
        ("metric", cfg.game.kwargs["metric"],
         dict_setter(cfg.game.kwargs, "metric"),
         "Leaf evaluation metric", list(VALID_METRICS)),
    ]
    if cfg.game.kwargs["metric"] == "penalized_reward":
        params.append(
            ("penalty_lambda", cfg.game.kwargs["penalty_lambda"],
             dict_setter(cfg.game.kwargs, "penalty_lambda"),
             "Penalty weight for interp ops", None))
    if cfg.game.kwargs["metric"] == "weighted":
        params.append(
            ("blend_alpha", cfg.game.kwargs["blend_alpha"],
             dict_setter(cfg.game.kwargs, "blend_alpha"),
             "Weight of solve_rate in blend", None))

    # MCTS
    mcts_descs = {
        "n_simulations": "MCTS rollouts per hole-filling step",
        "temperature": "Exploration temperature for action selection",
        "c_exploration": "UCB exploration constant",
        "dirichlet_alpha": "Dirichlet noise concentration parameter",
        "dirichlet_epsilon": "Weight of Dirichlet noise at root",
        "rollout_n": "Random completions per MCTS leaf (0=disabled)",
        "rollout_mode": "Aggregation: mean or max",
        "rollout_blend": "Blend: (1-b)*rollout + b*nn_value",
        "rollout_budget": "Max total steps for rollouts per leaf",
        "backup_rule": "Q-value backup: mean=avg, max=best, topk=avg top-k, softmax=smooth",
        "backup_topk": "k for topk backup",
        "backup_tau": "tau for softmax backup",
    }
    mcts_choices = {
        "rollout_mode": ["mean", "max"],
        "backup_rule": ["mean", "max", "topk", "softmax"],
    }
    for k in ["n_simulations", "temperature", "c_exploration",
              "dirichlet_alpha", "dirichlet_epsilon",
              "rollout_n", "rollout_mode", "rollout_blend", "rollout_budget",
              "backup_rule", "backup_topk", "backup_tau"]:
        if k in cfg.agent.mcts_params:
            params.append(
                (k, cfg.agent.mcts_params[k],
                 dict_setter(cfg.agent.mcts_params, k),
                 mcts_descs[k], mcts_choices.get(k)))

    # Network
    params.extend([
        ("d_model", cfg.net.kwargs["d_model"],
         dict_setter(cfg.net.kwargs, "d_model"),
         "Transformer embedding dimension", None),
        ("n_heads", cfg.net.kwargs["n_heads"],
         dict_setter(cfg.net.kwargs, "n_heads"),
         "Transformer attention heads", None),
        ("n_layers", cfg.net.kwargs["n_layers"],
         dict_setter(cfg.net.kwargs, "n_layers"),
         "Transformer encoder layers", None),
        ("learning_rate", cfg.net.kwargs["training_params"]["learning_rate"],
         dict_setter(cfg.net.kwargs["training_params"], "learning_rate"),
         "Adam learning rate", None),
        ("batch_size", cfg.net.kwargs["training_params"]["batch_size"],
         dict_setter(cfg.net.kwargs["training_params"], "batch_size"),
         "Training batch size", None),
    ])

    # Agent / Trainer / Evaluator / Run
    params.extend([
        ("reward_discount", cfg.agent.reward_discount,
         attr_setter(cfg.agent, "reward_discount"),
         "Discount factor for future rewards", None),
        ("n_games_per_train", cfg.trainer.n_games_per_train,
         attr_setter(cfg.trainer, "n_games_per_train"),
         "Self-play games per training iteration", None),
        ("n_past_iters", cfg.trainer.n_past_iterations_to_train,
         attr_setter(cfg.trainer, "n_past_iterations_to_train"),
         "Past iterations kept in training buffer", None),
        ("n_procs", cfg.trainer.n_procs,
         attr_setter(cfg.trainer, "n_procs"),
         "Parallel workers for self-play (-1=sequential)", None),
        ("eval_n_games", cfg.evaluator.n_games,
         attr_setter(cfg.evaluator, "n_games"),
         "Games to pit new vs old agent", None),
        ("eval_n_procs", cfg.evaluator.n_procs,
         attr_setter(cfg.evaluator, "n_procs"),
         "Parallel workers for evaluation (-1=sequential)", None),
        ("n_iterations", cfg.run.n_iterations,
         attr_setter(cfg.run, "n_iterations"),
         "Total training iterations", None),
        ("accept_threshold", cfg.run.accept_threshold,
         attr_setter(cfg.run, "accept_threshold"),
         "Win rate to accept new network", None),
        ("plot_every", cfg.run.plot_every,
         attr_setter(cfg.run, "plot_every"),
         "Plot metrics every N iterations", None),
    ])

    if seed_state is not None:
        params.append(
            ("seeds", seed_state["seeds_str"],
             lambda val: seed_state.__setitem__("seeds_str", val),
             "Base seeds (comma-sep for multi-seed CI)", None))

    all_params = build_param_list(params)

    problem_labels = {"num_rooms", "max_stages", "max_guard_depth", "hole_policy"}
    eval_labels = {"metric", "penalty_lambda", "blend_alpha"}
    mcts_labels = {"n_simulations", "temperature", "c_exploration",
                   "dirichlet_alpha", "dirichlet_epsilon",
                   "rollout_n", "rollout_mode", "rollout_blend",
                   "rollout_budget", "backup_rule", "backup_topk",
                   "backup_tau"}
    net_labels = {"d_model", "n_heads", "n_layers", "learning_rate", "batch_size"}
    agent_labels = {"reward_discount"}
    trainer_labels = {"n_games_per_train", "n_past_iters", "n_procs"}
    evaluator_labels = {"eval_n_games", "eval_n_procs"}
    run_labels = {"n_iterations", "accept_threshold", "plot_every", "seeds"}

    return [
        ("Problem",    [p for p in all_params if p[1] in problem_labels]),
        ("Leaf Eval",  [p for p in all_params if p[1] in eval_labels]),
        ("MCTS",       [p for p in all_params if p[1] in mcts_labels]),
        ("Network",    [p for p in all_params if p[1] in net_labels]),
        ("Agent",      [p for p in all_params if p[1] in agent_labels]),
        ("Trainer",    [p for p in all_params if p[1] in trainer_labels]),
        ("Evaluator",  [p for p in all_params if p[1] in evaluator_labels]),
        ("Run",        [p for p in all_params if p[1] in run_labels]),
    ]


# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------

def setup_experiment_dir(cfg):
    """Create and return an experiment directory path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gk = cfg.game.kwargs
    D = gk["num_rooms"]
    K = D - 1
    metric = gk["metric"]
    sim = cfg.agent.mcts_params.get("n_simulations", 0)
    games = cfg.trainer.n_games_per_train
    iters = cfg.run.n_iterations
    policy = gk.get("hole_policy", "leftmost")

    dirname = (f"{timestamp}_stage_D{D}_K{K}_{policy}_{metric}"
               f"_mcts{sim}_games{games}_iter{iters}")

    exp_dir = Path("experiments") / "doors_stage_derivation" / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg.trainer.checkpoint_dir = str(exp_dir / "checkpoints")
    cfg.run.plot_path = str(exp_dir / "training_metrics.png")

    return exp_dir


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def print_banner(cfg, exp_dir):
    """Print startup banner with experiment info."""
    gk = cfg.game.kwargs
    D = gk["num_rooms"]
    K = D - 1
    max_stages = gk["max_stages"]
    max_depth = gk["max_guard_depth"]
    policy = gk.get("hole_policy", "leftmost")
    metric = gk["metric"]
    lpr = gk.get("locs_per_room", 2)

    doors_cfg = DoorsGameConfig(num_rooms=D, locs_per_room=lpr)
    cost_model = SearchCostModel(max_stages=max_stages, max_guard_depth=max_depth)

    n_guards = len([g for g in enumerate_guards(doors_cfg, max_depth)
                    if cost_model.admits_guard(g)])
    n_actions = len(enumerate_actions(doors_cfg))
    total_actions = 2 + n_guards + n_actions
    max_decisions = 3 * max_stages + 1

    try:
        n_programs = count_stage_programs(doors_cfg, cost_model)
    except Exception:
        n_programs = -1

    print()
    print("=" * 80)
    print("  DOORS POLICY SYNTHESIS via Stage-Skeleton Hole-Filling + AlphaZero")
    print("=" * 80)
    print()
    print("  PROBLEM")
    print(f"    Doors environment: D={D} rooms, {lpr} locs/room")
    print(f"    K={K} keys, horizon={gk['horizon']}")
    print()
    print("  STAGE DSL")
    print(f"    max_stages={max_stages}, max_guard_depth={max_depth}")
    print(f"    Guard atoms:  {n_guards}")
    print(f"    Action atoms: {n_actions}")
    print(f"    Action space:  Discrete({total_actions}) = 2 structure + {n_guards} guards + {n_actions} actions")
    if n_programs >= 0:
        print(f"    Total programs: {n_programs:,}")
    else:
        print(f"    Total programs: >10M (too large to count)")
    print(f"    Max episode length: {max_decisions} steps")
    print(f"    Hole selection: {policy}")
    print()
    print("  OBSERVATION")
    print(f"    {max_decisions} slots x 2 floats = {2 * max_decisions} obs dimensions")
    print("    Each slot: (type_id, param)")
    print("    Types: PAD=0, STRUCTURE=1, GUARD=2, ACTION=3")
    print()
    print("  EVALUATION")
    print(f"    Metric: {metric}")
    print(f"    Terminal: compile_stage_program(stages) -> leaf_eval(AST)")
    print()
    print(f"  Experiment dir: {exp_dir}/")
    print()
    print("  Output legend:")
    print("    [TRAIN]  Self-play data collection & network training")
    print("    [EVAL]   Pitting new network vs old network")
    print("    [ITER]   Iteration summary with key metrics")
    print("    [PROG]   Best program discovered so far")
    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# Multi-seed helpers
# ---------------------------------------------------------------------------

def _parse_seeds(seeds_str):
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def _load_jsonl(path):
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _print_seed_summary(logs, seeds):
    if not logs:
        return
    final_solve, final_reward, final_progs = [], [], []
    for seed in seeds:
        if seed not in logs or not logs[seed]:
            continue
        last = logs[seed][-1]
        final_solve.append(last.get("best_solve_rate", 0))
        final_reward.append(last.get("best_avg_reward", 0))
        final_progs.append(last.get("unique_programs", 0))
    n = len(final_solve)
    if n == 0:
        return
    print()
    print("=" * 70)
    print(f"  MULTI-SEED SUMMARY  ({n} seeds: {seeds})")
    print("=" * 70)
    print(f"  Solve rate:    {np.mean(final_solve):.2f} +/- {np.std(final_solve):.2f}")
    print(f"  Avg reward:    {np.mean(final_reward):+.3f} +/- {np.std(final_reward):.3f}")
    print(f"  Unique progs:  {np.mean(final_progs):.0f} +/- {np.std(final_progs):.0f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Doors Stage Derivation -- AlphaZero Hole-Filling")
    parser.add_argument("--d", type=int, default=None,
                        help="Number of rooms (overrides interactive)")
    parser.add_argument("--max-stages", type=int, default=None,
                        help="Max stages (default: 2K+1)")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Max guard depth (default: 0)")
    parser.add_argument("--hole-policy", type=str, default=None,
                        choices=["leftmost", "structure_first"],
                        help="Hole selection policy")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds for multi-seed runs with CI")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Skip interactive config editing")
    return parser.parse_args()


def _compute_optimal_reward(cfg):
    num_rooms = cfg.game.kwargs["num_rooms"]
    step_penalty = cfg.game.kwargs.get("step_penalty", 0.01)
    unlock_bonus = cfg.game.kwargs.get("unlock_bonus", 0.1)
    optimal_steps = 2 * (num_rooms - 1) + 1
    return 1.0 + (num_rooms - 1) * unlock_bonus - optimal_steps * step_penalty


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    # Build config with CLI overrides
    num_rooms = args.d if args.d is not None else 3
    cfg = DoorsStageDerivationConfig(
        num_rooms=num_rooms,
        max_stages=args.max_stages,
        max_guard_depth=args.max_depth if args.max_depth is not None else 0,
        hole_policy=args.hole_policy if args.hole_policy is not None else "leftmost",
    )

    # Seed state for interactive editing
    default_seed = str(args.seeds[0]) if args.seeds else "43"
    seed_state = {"seeds_str": default_seed}

    if not args.non_interactive:
        interactive_edit(
            "Doors Stage Derivation Config",
            lambda: _build_sections(cfg, seed_state),
        )

    # Determine seeds
    if args.seeds and len(args.seeds) > 1:
        seeds = args.seeds
    else:
        seeds = _parse_seeds(seed_state["seeds_str"])

    # Setup experiment directory
    exp_dir = setup_experiment_dir(cfg)

    # Print startup info
    print_banner(cfg, exp_dir)

    optimal_reward = _compute_optimal_reward(cfg)

    if len(seeds) <= 1:
        if seeds:
            s = seeds[0]
            cfg.agent.random_seeds = {
                "mcts": s, "train": s + 1,
                "eval": s + 2, "external_policy": s + 3,
            }
        cfg.save(str(exp_dir / "config.json"))
        run_derivation_training(cfg, "stage", optimal_reward, exp_dir)
    else:
        cfg.save(str(exp_dir / "config.json"))
        seed_dirs = {}
        for seed in seeds:
            print()
            print(f"{'='*70}")
            print(f"  SEED {seed}")
            print(f"{'='*70}")
            cfg.agent.random_seeds = {
                "mcts": seed, "train": seed + 1,
                "eval": seed + 2, "external_policy": seed + 3,
            }
            seed_dir = exp_dir / f"seed_{seed}"
            seed_dirs[seed] = seed_dir
            run_derivation_training(cfg, "stage", optimal_reward, seed_dir)

        logs = {}
        for seed, seed_dir in seed_dirs.items():
            records = _load_jsonl(seed_dir / "program_log.jsonl")
            if records:
                logs[seed] = records

        _print_seed_summary(logs, seeds)

        print()
        print("[Analysis] Multi-seed run complete.")

    print()


if __name__ == "__main__":
    main()
