#!/usr/bin/env python3
"""Doors Reactive Derivation -- AlphaZero Reactive BT Composition.

Synthesises Doors navigation policies via reactive branch-composition
MCTS + AlphaZero.  The reactive game fills N branch slots with
(predicate, action) pairs from a typed catalog; the assembled BT is
evaluated by the tick interpreter on frozen initial states.

Usage:
    python scripts/run_reactive_alphazero.py
    python scripts/run_reactive_alphazero.py --non-interactive
    python scripts/run_reactive_alphazero.py --seeds 42 43 --non-interactive
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

from alphazeropp.instances.doors.dsl.reactive_training_config import (
    DoorsReactiveDerivationConfig,
)
from alphazeropp.instances.doors.dsl.reactive_branch_catalog import (
    known_map_catalog,
)
from alphazeropp.instances.doors.dsl.reactive_typed_grammar import (
    enumerate_reactive_typed_policies,
)
from alphazeropp.instances.doors.oracle import optimal_return
from alphazeropp.utils.interactive_config import (
    build_param_list, interactive_edit, attr_setter, dict_setter,
)
from alphazeropp.utils.derivation_utils import run_derivation_training


# ---------------------------------------------------------------------------
# Config display & interactive editing
# ---------------------------------------------------------------------------

def _build_sections(cfg, seed_state=None):
    """Return sections for ReactiveDerivationGame config.

    Changing num_rooms auto-updates horizon. n_branches changes episode
    length and observation size.
    """
    def _set_num_rooms(val):
        D = int(val)
        cfg.game.kwargs["num_rooms"] = D
        # Auto-scale n_branches: need at least 2K+1 for K=D-1 keys
        K = D - 1
        new_n = max(4, 2 * K + 1)
        cfg.game.kwargs["n_branches"] = new_n
        n_steps = 2 * new_n
        cfg.game.kwargs["n_sites"] = n_steps
        cfg.game.kwargs["budget"] = n_steps
        cfg.net.kwargs["budget"] = n_steps
        cfg.net.kwargs["n_sites"] = n_steps

    def _set_n_branches(val):
        nb = int(val)
        cfg.game.kwargs["n_branches"] = nb
        n_steps = 2 * nb
        cfg.game.kwargs["n_sites"] = n_steps
        cfg.game.kwargs["budget"] = n_steps
        cfg.net.kwargs["budget"] = n_steps
        cfg.net.kwargs["n_sites"] = n_steps

    params = [
        # Problem
        ("num_rooms", cfg.game.kwargs["num_rooms"], _set_num_rooms,
         "Number of rooms (D)", None),
        ("n_branches", cfg.game.kwargs["n_branches"], _set_n_branches,
         "Number of BT branches (N) -- episode = 2N steps", None),
        ("catalog_mode", cfg.game.kwargs["catalog_mode"],
         dict_setter(cfg.game.kwargs, "catalog_mode"),
         "Grammar mode: typed (legal-pair mask) or raw (cross-product)",
         ["typed", "raw"]),
        ("known_map", cfg.game.kwargs["known_map"],
         dict_setter(cfg.game.kwargs, "known_map"),
         "Known map (True) or partial map (False)", [True, False]),
        # Leaf evaluation
        ("metric", cfg.game.kwargs["metric"],
         dict_setter(cfg.game.kwargs, "metric"),
         "Leaf evaluation metric", ["avg_reward", "solve_rate", "weighted"]),
    ]
    if cfg.game.kwargs["metric"] == "weighted":
        params.append(
            ("blend_alpha", cfg.game.kwargs["blend_alpha"],
             dict_setter(cfg.game.kwargs, "blend_alpha"),
             "Weight of solve_rate in blend", None))

    # MCTS
    mcts_descs = {
        "n_simulations": "MCTS rollouts per branch-slot step",
        "temperature": "Exploration temperature for action selection",
        "c_exploration": "UCB exploration constant",
        "dirichlet_alpha": "Dirichlet noise concentration parameter",
        "dirichlet_epsilon": "Weight of Dirichlet noise at root",
        "rollout_n": "Random completions per MCTS leaf (0=disabled)",
        "rollout_mode": "Aggregation: mean or max",
        "rollout_blend": "Blend: (1-b)*rollout + b*nn_value",
        "rollout_budget": "Max total steps for rollouts per leaf",
        "backup_rule": "Q-value backup: mean/max/topk/softmax",
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

    # Network architecture
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

    problem_labels = {"num_rooms", "n_branches", "catalog_mode", "known_map"}
    eval_labels = {"metric", "blend_alpha"}
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
    """Create and return an experiment directory path. Updates cfg paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    D = cfg.game.kwargs["num_rooms"]
    N = cfg.game.kwargs["n_branches"]
    mode = cfg.game.kwargs["catalog_mode"]
    metric = cfg.game.kwargs["metric"]
    sim = cfg.agent.mcts_params.get("n_simulations", 0)
    games = cfg.trainer.n_games_per_train
    iters = cfg.run.n_iterations

    dirname = (f"{timestamp}_reactive_D{D}_N{N}_{mode}_{metric}"
               f"_mcts{sim}_games{games}_iter{iters}")

    exp_dir = Path("experiments") / "doors_reactive_derivation" / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg.trainer.checkpoint_dir = str(exp_dir / "checkpoints")
    cfg.run.plot_path = str(exp_dir / "training_metrics.png")

    return exp_dir


# ---------------------------------------------------------------------------
# Training output helpers
# ---------------------------------------------------------------------------

def print_banner(cfg, exp_dir):
    """Print startup banner with experiment info."""
    gk = cfg.game.kwargs
    D = gk["num_rooms"]
    N = gk["n_branches"]
    mode = gk["catalog_mode"]
    known_map = gk["known_map"]
    metric = gk["metric"]
    episode_len = 2 * N

    catalog = known_map_catalog(mode=mode)
    n_pred = catalog.n_predicates
    n_act = catalog.n_actions
    n_legal = len(catalog.legal_pairs())
    total_policies = n_legal ** N

    print()
    print("=" * 80)
    print("  DOORS POLICY SYNTHESIS via Reactive BT Composition + AlphaZero")
    print("=" * 80)
    print()
    print("  PROBLEM")
    print(f"    Doors PDDL environment: D={D} rooms, 2 locs/room")
    print(f"    Synthesize a reactive BT by composing {N} branches")
    print(f"    from a typed predicate-action catalog.")
    print()
    print("  REACTIVE GRAMMAR")
    print(f"    Catalog mode:    {mode}")
    print(f"    Known map:       {known_map}")
    print(f"    Predicates:      {n_pred}")
    print(f"    Actions:         {n_act}")
    K = D - 1
    min_branches = 2 * K + 1
    print(f"    Legal pairs:     {n_legal} (typed mask blocks absurd combos)")
    if N >= min_branches:
        print(f"    Branches:        {N} (auto: 2K+1 = {min_branches} for K={K} keys)")
    else:
        print(f"    Branches:        {N} (WARNING: <2K+1={min_branches}, may be insufficient)")
    print(f"    Complete policies: {total_policies:,}")
    print()
    print("  BT STRUCTURE")
    print(f"    While(Not(GoalReached), Fallback(B1, ..., B{N}))")
    print(f"    Each Bi = Sequence(Check(predicate_i), Do(action_i))")
    print()
    print("  DERIVATION GAME")
    print(f"    Action space:    Discrete({max(n_pred, n_act)})")
    print(f"    Episode length:  {episode_len} steps (fixed: {N} pred + {N} act)")
    print(f"    Steps alternate: predicate choice (even), action choice (odd)")
    print(f"    Legal mask:      typed pairs enforced at each action step")
    print()
    print("  EVALUATION")
    print(f"    Metric:          {metric}")
    print(f"    Terminal reward:  build_policy(specs) -> tick(BT) on frozen states")
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


def print_architecture(cfg):
    """Print the AlphaZero training loop architecture diagram."""
    D = cfg.game.kwargs["num_rooms"]
    N = cfg.game.kwargs["n_branches"]
    n_sims = cfg.agent.mcts_params.get("n_simulations", "?")
    n_games = cfg.trainer.n_games_per_train
    n_iters = cfg.run.n_iterations
    n_past = cfg.trainer.n_past_iterations_to_train
    d_model = cfg.net.kwargs.get("d_model", 64)
    net_desc = f"Transformer(d={d_model})"

    print("=" * 80)
    print("  Algorithm Architecture: AlphaZero for Reactive BT Composition")
    print("=" * 80)
    print()
    print(f"  Outer loop: {n_iters} training iterations")
    print(f"  Each iteration: {n_games} self-play games -> train net -> evaluate")
    print()
    print("  +----- Iteration i " + "-" * 53 + "+")
    print("  |                                                                      |")
    print(f"  |  STEP 1: Self-Play  ({n_games} reactive BT composition games)")
    print("  |")
    print(f"  |    Each game: fill {N} branch slots = {2*N} decisions")
    print(f"  |      Step 2k:   choose predicate for branch k (typed legal mask)")
    print(f"  |      Step 2k+1: choose action for branch k   (pair legal mask)")
    print(f"  |      At each step: MCTS(partial_BT, {net_desc}, n_sims={n_sims})")
    print("  |        Network reads (type_id, param) token sequence")
    print("  |        -> (pi: slot probs, v: predicted policy quality)")
    print("  |        MCTS refines pi using UCB + backed-up leaf evaluations")
    print("  |      pi_MCTS = visit_counts^(1/tau) / sum")
    print("  |      SAVE (partial_BT, pi_MCTS) as training target")
    print("  |    Terminal: build_policy(specs) -> tick(BT) -> scalar reward")
    print("  |")
    print(f"  |  STEP 2: Train Transformer  (replay buffer: last {n_past} iterations)")
    print("  |")
    print("  |    L = (z - v_theta(seq))^2  -  pi_MCTS * log p_theta(seq)")
    print("  |        value loss              policy loss")
    print("  |    theta <- theta - alpha * grad(L)")
    print("  |")
    print("  |  STEP 3: Evaluate")
    print("  |    Pit new_net vs old_net on fresh composition games.")
    print("  |                                                                      |")
    print("  +" + "-" * 72 + "+")
    print()
    print("  Virtuous cycle:")
    print("    Better Transformer -> better MCTS -> better data -> better Transformer")
    print("    The network learns which partial branch compositions are promising,")
    print("    directing search toward solving reactive BT policies.")
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
    print(f"  {'Seed':>6}  {'Solve':>6}  {'Reward':>8}  {'Programs':>8}")
    print(f"  {'---':>6}  {'---':>6}  {'---':>8}  {'---':>8}")
    for i, seed in enumerate(seeds):
        if seed not in logs:
            continue
        print(f"  {seed:>6}  {final_solve[i]:>5.0%}  "
              f"{final_reward[i]:>+8.3f}  {final_progs[i]:>8}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Doors Reactive Derivation -- AlphaZero BT Composition")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds for multi-seed runs with CI")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Skip interactive config editing")
    return parser.parse_args()


def _compute_optimal_reward(cfg):
    num_rooms = cfg.game.kwargs["num_rooms"]
    step_penalty = 0.01
    unlock_bonus = 0.1
    optimal_steps = 2 * (num_rooms - 1) + 1
    return 1.0 + (num_rooms - 1) * unlock_bonus - optimal_steps * step_penalty


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    # Seed state for interactive editing
    default_seed = str(args.seeds[0]) if args.seeds else "43"
    seed_state = {"seeds_str": default_seed}

    # Config creation
    cfg = DoorsReactiveDerivationConfig()

    if not args.non_interactive:
        interactive_edit(
            "Doors Reactive Derivation Config",
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
    print_architecture(cfg)

    optimal_reward = _compute_optimal_reward(cfg)

    if len(seeds) <= 1:
        # Single-seed run
        if seeds:
            s = seeds[0]
            cfg.agent.random_seeds = {
                "mcts": s, "train": s + 1,
                "eval": s + 2, "external_policy": s + 3,
            }
        cfg.save(str(exp_dir / "config.json"))
        run_derivation_training(cfg, "reactive", optimal_reward, exp_dir)
    else:
        # Multi-seed run
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
            run_derivation_training(cfg, "reactive", optimal_reward, seed_dir)

        # Aggregate across seeds
        logs = {}
        for seed, seed_dir in seed_dirs.items():
            records = _load_jsonl(seed_dir / "program_log.jsonl")
            if records:
                logs[seed] = records

        _print_seed_summary(logs, seeds)

        # Try to plot seed comparison (optional dependency)
        try:
            from run_doors_derivation import (
                _plot_seed_comparison, _plot_seed_deltas, _load_train_stats,
            )
            train_stats = _load_train_stats(seed_dirs)
            _plot_seed_comparison(logs, seeds, exp_dir,
                                  optimal_reward=optimal_reward,
                                  train_stats=train_stats)
            _plot_seed_deltas(logs, seeds, exp_dir)
        except ImportError:
            pass

        print()
        print("[Analysis] Multi-seed run complete.")
        print("  Key things to look for:")
        print("  - Do seeds converge? High variance = unstable training")
        print("  - Flat delta curves = stagnation, consider stopping early")


if __name__ == "__main__":
    main()
