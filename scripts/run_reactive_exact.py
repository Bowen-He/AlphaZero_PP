#!/usr/bin/env python3
"""Reactive Sketch — Exact Oracle.

Evaluates all 24 branch-order permutations of the reactive sketch BT
for given D values with known_map=True.  Computes behavioral signatures,
equivalence classes, and action traces.

Usage:
    python scripts/run_reactive_exact.py --D 2
    python scripts/run_reactive_exact.py --D 3 --known-map
    python scripts/run_reactive_exact.py --D 2 3 --output-dir results/reactive_exact
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

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
from alphazeropp.instances.doors.dsl.stage_diagnostics import (
    make_frozen_state_suite, reactive_semantic_signature,
)
from alphazeropp.instances.doors.oracle import (
    enumerate_reachable_states, optimal_return,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def build_config(D: int, locs_per_room: int = 2) -> DoorsGameConfig:
    params = compute_doors_derived_params(D, locs_per_room)
    return DoorsGameConfig(
        num_rooms=D, locs_per_room=locs_per_room, horizon=params["horizon"],
    )


def obs_from_reachable_state(
    agent_loc: int, unlock_tuple: tuple[int, ...],
    cfg: DoorsGameConfig,
) -> np.ndarray:
    """Build an observation vector from a reachable (agent_loc, unlock) pair."""
    M = cfg.M
    D = cfg.D
    K = cfg.K
    obs = np.zeros(cfg.obs_size(), dtype=np.float32)
    obs[agent_loc] = 1.0
    for r in range(D):
        if unlock_tuple[r]:
            obs[M + r] = 1.0
    # Keys: available if room is unlocked but key hasn't been used yet
    # In the BFS states, key availability is determined by unlock prefix.
    # Key k unlocks room k+1. Key k is available if room k+1 is still locked
    # AND the room containing key k is unlocked.
    for k in range(K):
        key_room = cfg.loc_room[cfg.key_loc[k]]
        target_room = k + 1  # key k unlocks room k+1
        if unlock_tuple[key_room] and not unlock_tuple[target_room]:
            obs[M + D + k] = 1.0
    return obs


def build_reachable_state_suite(cfg: DoorsGameConfig) -> list[np.ndarray]:
    """Build evaluation states from BFS-enumerated reachable states."""
    env = cfg.make_env(cfg.obs_size())
    reachable = enumerate_reachable_states(env)
    states = []
    for agent_loc, unlock_tuple in sorted(reachable):
        obs = obs_from_reachable_state(agent_loc, unlock_tuple, cfg)
        states.append(obs)
    return states


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_all_permutations(
    D: int,
    cfg: DoorsGameConfig,
    rt: DoorsRelationalRuntime,
    eval_states: list[np.ndarray],
) -> list[dict]:
    """Evaluate all 24 permutations, returning per-policy results."""
    x0 = doors_initial_state(cfg)
    results = []

    for perm in itertools.permutations([1, 2, 3, 4]):
        policy = canonical_reactive_policy(perm)

        # Run episode from initial state
        env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
        episode = run_reactive_episode(
            env, policy, rt, x0=x0, is_solved=cfg.is_solved,
        )

        # Extract action trace
        action_trace = [step.action for step in episode.steps]

        # Compute behavioral signature on evaluation states
        sig = reactive_semantic_signature(policy, rt, eval_states)

        results.append({
            "order": list(perm),
            "solved": episode.solved,
            "reward": round(episode.cumulative_reward, 6),
            "steps": episode.total_env_steps,
            "action_trace": action_trace,
            "signature": list(sig),
        })

    return results


def assign_equivalence_classes(results: list[dict]) -> list[dict]:
    """Assign equivalence class IDs based on signature, return class info."""
    sig_to_class: dict[tuple, int] = {}
    class_members: dict[int, list] = defaultdict(list)
    next_id = 0

    for r in results:
        sig = tuple(r["signature"])
        if sig not in sig_to_class:
            sig_to_class[sig] = next_id
            next_id += 1
        cid = sig_to_class[sig]
        r["equiv_class_id"] = cid
        class_members[cid].append(r)

    # Build class summaries
    classes = []
    for cid in range(next_id):
        members = class_members[cid]
        classes.append({
            "id": cid,
            "member_orders": [m["order"] for m in members],
            "count": len(members),
            "solved": members[0]["solved"],
            "reward": members[0]["reward"],
            "steps": members[0]["steps"],
            "action_trace": members[0]["action_trace"],
            "signature": members[0]["signature"],
        })

    return classes


def format_summary_md(
    D: int,
    results: list[dict],
    classes: list[dict],
    n_eval_states: int,
) -> str:
    """Format a markdown summary report."""
    lines = [
        f"# Reactive Sketch Exact Oracle — D={D}",
        "",
        f"- **Permutations evaluated:** {len(results)}",
        f"- **Evaluation states:** {n_eval_states}",
        f"- **Equivalence classes:** {len(classes)}",
        f"- **Solving permutations:** {sum(1 for r in results if r['solved'])}/24",
        f"- **Solving classes:** {sum(1 for c in classes if c['solved'])}/{len(classes)}",
        f"- **Optimal reward:** {optimal_return(D):.4f}",
        "",
        "## Equivalence Classes",
        "",
        "| Class | Solved | Reward | Steps | Members | Action Trace |",
        "|------:|:------:|-------:|------:|--------:|:-------------|",
    ]

    for c in classes:
        solved_str = "yes" if c["solved"] else "no"
        trace_str = ", ".join(str(a) for a in c["action_trace"])
        members_str = str(c["count"])
        lines.append(
            f"| {c['id']} | {solved_str} | {c['reward']:+.4f} | "
            f"{c['steps']} | {members_str} | [{trace_str}] |"
        )

    lines.extend([
        "",
        "## All Permutations",
        "",
        "| Order | Class | Solved | Reward | Steps | Action Trace |",
        "|:------|------:|:------:|-------:|------:|:-------------|",
    ])

    for r in results:
        order_str = f"({','.join(str(x) for x in r['order'])})"
        solved_str = "yes" if r["solved"] else "no"
        trace_str = ", ".join(str(a) for a in r["action_trace"])
        lines.append(
            f"| {order_str} | {r['equiv_class_id']} | {solved_str} | "
            f"{r['reward']:+.4f} | {r['steps']} | [{trace_str}] |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action name formatting
# ---------------------------------------------------------------------------

def format_action_name(action: int, cfg: DoorsGameConfig) -> str:
    """Convert raw action index to human-readable name."""
    M = cfg.M
    K = cfg.K
    if action < M:
        return f"move_to({action})"
    elif action < M + K:
        return f"pick({action - M})"
    else:
        return "noop"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reactive Sketch — Exact Oracle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--D", nargs="+", type=int, default=[2, 3],
        help="Room counts to evaluate (default: 2 3)",
    )
    parser.add_argument(
        "--known-map", action="store_true", default=True,
        help="Use known_map=True (default, only mode currently supported)",
    )
    parser.add_argument(
        "--use-reachable", action="store_true",
        help="Use BFS-enumerated reachable states instead of frozen state suite",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/reactive_exact",
        help="Output directory (default: results/reactive_exact)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    section("Reactive Sketch — Exact Oracle")
    print(f"  D values:      {args.D}")
    print(f"  known_map:     {args.known_map}")
    print(f"  use_reachable: {args.use_reachable}")
    print(f"  output_dir:    {output_dir}")

    for D in sorted(args.D):
        cfg = build_config(D)
        rt = DoorsRelationalRuntime(cfg, known_map=args.known_map)

        # Build evaluation states
        if args.use_reachable:
            eval_states = build_reachable_state_suite(cfg)
        else:
            eval_states = make_frozen_state_suite(cfg)

        section(f"D={D}  ({len(eval_states)} evaluation states)")

        # Evaluate all 24 permutations
        results = evaluate_all_permutations(D, cfg, rt, eval_states)
        classes = assign_equivalence_classes(results)

        # Terminal output
        n_solved = sum(1 for r in results if r["solved"])
        n_solving_classes = sum(1 for c in classes if c["solved"])
        print(f"  Permutations:       24")
        print(f"  Solving:            {n_solved}/24")
        print(f"  Equivalence classes: {len(classes)}")
        print(f"  Solving classes:    {n_solving_classes}/{len(classes)}")
        print(f"  Optimal reward:     {optimal_return(D):.4f}")
        print()

        # Print equivalence classes
        for c in classes:
            solved_str = "SOLVE" if c["solved"] else "FAIL "
            trace_names = [format_action_name(a, cfg) for a in c["action_trace"]]
            trace_str = ", ".join(trace_names)
            orders_str = " ".join(
                f"({','.join(str(x) for x in o)})" for o in c["member_orders"]
            )
            print(
                f"  Class {c['id']:2d} [{solved_str}] "
                f"reward={c['reward']:+.4f} steps={c['steps']:2d}  "
                f"trace=[{trace_str}]"
            )
            print(f"           members: {orders_str}")

        # Save JSON outputs
        policies_path = output_dir / f"D{D}_policies.json"
        with open(policies_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {policies_path}")

        equiv_path = output_dir / f"D{D}_equivalence.json"
        with open(equiv_path, "w") as f:
            json.dump(classes, f, indent=2)
        print(f"  Saved: {equiv_path}")

        # Save markdown summary
        md = format_summary_md(D, results, classes, len(eval_states))
        md_path = output_dir / f"D{D}_summary.md"
        with open(md_path, "w") as f:
            f.write(md + "\n")
        print(f"  Saved: {md_path}")

    print(f"\n  All results in: {output_dir}")


if __name__ == "__main__":
    main()
