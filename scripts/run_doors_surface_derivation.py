"""
Doors Surface Derivation -- AlphaZero Surface-Rule Sequencing.

Synthesises Doors navigation policies via surface-rule MCTS + AlphaZero.
The surface game places PickRule(k), MoveRule(k), GoalRule in sequence
(no budget, no raw AST holes, no dead ends).

Usage:
    python scripts/run_doors_surface_derivation.py
    python scripts/run_doors_surface_derivation.py --non-interactive
    python scripts/run_doors_surface_derivation.py --seeds 42 43 --non-interactive
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

from alphazeropp.instances.doors.dsl.surface_derivation_config import (
    DoorsSurfaceDerivationConfig,
)
from alphazeropp.instances.doors.dsl.surface_grammar import (
    count_relaxed_policies,
)
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
    """Return sections for Surface DerivationGame config.

    Changing num_rooms auto-updates action space, observation size,
    and horizon.
    """
    def _set_num_rooms(val):
        nr = int(val)
        cfg.game.kwargs["num_rooms"] = nr
        new_K = nr - 1
        new_max_steps = 2 * new_K + 1
        cfg.game.kwargs["horizon"] = max(15, (2 * (nr - 1) + 1) * 5)
        cfg.net.kwargs["budget"] = new_max_steps
        cfg.net.kwargs["n_sites"] = new_max_steps
        cfg.net.kwargs["action_size"] = new_max_steps

    params = [
        # Problem
        ("num_rooms", cfg.game.kwargs["num_rooms"], _set_num_rooms,
         "Number of rooms (D) -- auto-updates action space (2K+1)", None),
        ("allow_early_goal", cfg.game.kwargs.get("allow_early_goal", False),
         dict_setter(cfg.game.kwargs, "allow_early_goal"),
         "Allow GoalRule before all rules placed (ablation)", [True, False]),
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
             "Weight of solve_rate in blend (adaptive: uses raw reward when sr=0)",
             None))

    # MCTS
    mcts_descs = {
        "n_simulations": "MCTS rollouts per rule-placement step",
        "temperature": "Exploration temperature for action selection",
        "c_exploration": "UCB exploration constant",
        "dirichlet_alpha": "Dirichlet noise concentration parameter",
        "dirichlet_epsilon": "Weight of Dirichlet noise at root",
        "rollout_n": "Random completions per MCTS leaf (0=disabled)",
        "rollout_mode": "Aggregation: mean or max",
        "rollout_blend": "Blend: (1-b)*rollout + b*nn_value",
        "rollout_budget": "Max total steps for rollouts per leaf",
        "backup_rule": "Q-value backup: mean=avg all, max=best path, topk=avg top-k, softmax=smooth max",
        "backup_topk": "k for topk backup (ignored unless backup_rule=topk)",
        "backup_tau": "tau for softmax backup; tau->0 = max, tau->inf = mean",
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

    problem_labels = {"num_rooms", "allow_early_goal"}
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
    """Create and return an experiment directory path. Updates cfg paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    D = cfg.game.kwargs["num_rooms"]
    K = D - 1
    metric = cfg.game.kwargs["metric"]
    sim = cfg.agent.mcts_params.get("n_simulations", 0)
    games = cfg.trainer.n_games_per_train
    iters = cfg.run.n_iterations

    dirname = (f"{timestamp}_surface_D{D}_K{K}_{metric}"
               f"_mcts{sim}_games{games}_iter{iters}")

    exp_dir = Path("experiments") / "doors_surface_derivation" / dirname
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
    K = D - 1
    max_steps = 2 * K + 1
    metric = gk["metric"]
    n_policies = count_relaxed_policies(D)
    lpr = gk.get("locs_per_room", 2)
    M = D * lpr

    print()
    print("=" * 80)
    print("  DOORS POLICY SYNTHESIS via Surface Rule Sequencing + AlphaZero")
    print("=" * 80)
    print()
    print("  PROBLEM")
    print(f"    Doors PDDL environment: D={D} rooms, {lpr} locs/room")
    print(f"    M={M} locations, K={K} keys")
    print(f"    Horizon (max steps): {gk['horizon']}")
    print(f"    Synthesize a navigation policy by ordering surface rules.")
    print()
    print("  ROOM LAYOUT & KEYS")
    key_loc = [k * lpr + 1 for k in range(K)]
    for r in range(D):
        locs = list(range(r * lpr, (r + 1) * lpr))
        room_desc = f"    Room {r}: locations {locs}"
        keys_here = [k for k, kl in enumerate(key_loc) if kl // lpr == r]
        for k in keys_here:
            room_desc += f"  [Key {k} at loc {key_loc[k]} -> unlocks Room {k+1}]"
        if r == D - 1:
            room_desc += "  [GOAL]"
        print(room_desc)
    print()
    print("  SURFACE GRAMMAR")
    print(f"    Actions (total {max_steps}):")
    print(f"      0..{K-1}        -> PickRule(k)   : if PickReady(k) then Pick(k)")
    print(f"      {K}..{2*K-1}       -> MoveRule(k)   : if NeedKey(k) then MoveToKey(k)")
    print(f"      {2*K}            -> GoalRule       : default MoveToGoal")
    print(f"    Episode length:   {max_steps} steps (fixed)")
    print(f"    Complete policies: {n_policies:,}")
    if D <= 5:
        from alphazeropp.instances.doors.dsl.doors_config import (
            DoorsGameConfig, compute_doors_derived_params,
        )
        from alphazeropp.instances.doors.dsl.surface_grammar import (
            count_solving_policies,
        )
        params = compute_doors_derived_params(D, lpr)
        dcfg = DoorsGameConfig(num_rooms=D, locs_per_room=lpr,
                               horizon=params["horizon"])
        n_solve = count_solving_policies(D, dcfg)
        print(f"    Solving policies:  {n_solve:,} "
              f"({n_solve/n_policies:.1%} of grammar)")
    print()
    print("  LEGAL MASK (strict mode)")
    print("    PickRule(k): legal iff k not yet picked AND k not yet moved")
    print("    MoveRule(k): legal iff k picked AND k not yet moved")
    print("    GoalRule:    legal iff all 2K non-goal rules placed")
    print("    No dead ends -- at least one action is always legal.")
    print()
    print("  OBSERVATION ENCODING")
    print(f"    {max_steps} slots x 2 floats = {2*max_steps} obs dimensions")
    print("    Each slot: (type_id, param)")
    print("    Tokens: PAD=0, PICK=1, MOVE=2, GOAL=3")
    print("    Param:  key index k for PICK/MOVE, 0 for GOAL/PAD")
    print()
    print("  EVALUATION")
    print(f"    Metric:          {metric}")
    print(f"    Terminal reward:  compile_policy(rules) -> leaf_eval(AST)")
    print(f"    Optimal reward:  ~1.0 (all keys collected, goal reached)")
    print()

    # MCTS backup strategy
    backup_rule = cfg.agent.mcts_params.get("backup_rule", "mean")
    rollout_n = cfg.agent.mcts_params.get("rollout_n", 0)
    rollout_mode = cfg.agent.mcts_params.get("rollout_mode", "mean")
    rollout_blend = cfg.agent.mcts_params.get("rollout_blend", 0.0)
    print("  MCTS SEARCH STRATEGY")
    print(f"    Rollouts:     {rollout_n} random completions per leaf "
          f"(mode={rollout_mode}, blend={rollout_blend})")
    print(f"    Backup rule:  {backup_rule}")
    print()
    print("    Backup rules (how Q-values propagate up the MCTS tree):")
    print("      mean    : Q = average of ALL values. Standard.")
    print("      max     : Q = single best value seen. Optimistic.")
    print("      topk    : Q = average of top-k values. Compromise.")
    print("      softmax : Q = tau*log(mean(exp(v/tau))). Smooth max.")
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
    K = D - 1
    n_sims = cfg.agent.mcts_params.get("n_simulations", "?")
    n_games = cfg.trainer.n_games_per_train
    n_iters = cfg.run.n_iterations
    n_past = cfg.trainer.n_past_iterations_to_train
    d_model = cfg.net.kwargs.get("d_model", 64)
    net_desc = f"Transformer(d={d_model})"

    print("=" * 80)
    print("  Algorithm Architecture: AlphaZero for Surface Rule Sequencing")
    print("=" * 80)
    print()
    print(f"  Outer loop: {n_iters} training iterations")
    print(f"  Each iteration: {n_games} self-play games -> train net -> evaluate")
    print()
    print("  +----- Iteration i " + "-" * 53 + "+")
    print("  |                                                                      |")
    print(f"  |  STEP 1: Self-Play  ({n_games} surface-rule sequencing games)")
    print("  |")
    print(f"  |    Each game: place {2*K+1} rules to build a SurfacePolicy")
    print(f"  |      At each step: MCTS(partial_sequence, {net_desc}, n_sims={n_sims})")
    print("  |        Network reads partial rule sequence as (type_id, param) tokens")
    print("  |        -> (pi: rule probs, v: predicted policy quality)")
    print("  |        MCTS refines pi using UCB + backed-up leaf evaluations")
    print("  |      pi_MCTS = visit_counts^(1/tau) / sum")
    print("  |      SAVE (partial_sequence, pi_MCTS) as training target")
    print("  |    Terminal: compile_policy(rules) -> leaf_eval(AST) -> scalar reward")
    print("  |")
    print(f"  |  STEP 2: Train Transformer  (replay buffer: last {n_past} iterations)")
    print("  |")
    print("  |    L = (z - v_theta(seq))^2  -  pi_MCTS * log p_theta(seq)")
    print("  |        value loss              policy loss")
    print("  |    theta <- theta - alpha * grad(L)")
    print("  |")
    print("  |  STEP 3: Evaluate")
    print("  |    Pit new_net vs old_net on fresh sequencing games.")
    print("  |                                                                      |")
    print("  +" + "-" * 72 + "+")
    print()
    print("  Virtuous cycle:")
    print("    Better Transformer -> better MCTS -> better data -> better Transformer")
    print("    The network learns which partial rule orderings are promising,")
    print("    directing search toward solving policies.")
    print()


# ---------------------------------------------------------------------------
# Multi-seed helpers (reused from run_doors_derivation)
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
        description="Doors Surface Derivation -- AlphaZero Rule Sequencing")
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

    # Seed state for interactive editing
    default_seed = str(args.seeds[0]) if args.seeds else "43"
    seed_state = {"seeds_str": default_seed}

    # Config creation
    cfg = DoorsSurfaceDerivationConfig()

    if not args.non_interactive:
        interactive_edit(
            "Doors Surface Derivation Config",
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
        run_derivation_training(cfg, "surface", optimal_reward, exp_dir)
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
            run_derivation_training(cfg, "surface", optimal_reward, seed_dir)

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
