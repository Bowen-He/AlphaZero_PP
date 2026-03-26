#!/usr/bin/env python3
"""Reactive Sketch — Partial-Map Exact Oracle.

Evaluates all 120 branch-order permutations of the 5-branch reactive sketch BT
for given D values with known_map=False.  Computes behavioral signatures,
equivalence classes, and action traces.

Usage:
    python scripts/run_reactive_partial_map_exact.py --D 2
    python scripts/run_reactive_partial_map_exact.py --D 2 3 --output-dir results/reactive_partial_map
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
from alphazeropp.instances.doors.dsl.reactive_memory import ReactiveMemory
from alphazeropp.instances.doors.dsl.reactive_sketch_dsl import (
    canonical_partial_map_policy,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import (
    run_reactive_partial_episode,
)
from alphazeropp.instances.doors.dsl.stage_diagnostics import (
    make_frozen_state_suite, reactive_semantic_signature,
)
from alphazeropp.instances.doors.oracle import optimal_return


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


def format_action_name(action: int, cfg: DoorsGameConfig) -> str:
    M = cfg.M
    K = cfg.K
    if action < M:
        return f"move_to({action})"
    elif action < M + K:
        return f"pick({action - M})"
    else:
        return "noop"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_all_permutations(
    D: int,
    cfg: DoorsGameConfig,
    rt: DoorsRelationalRuntime,
    eval_states: list[np.ndarray],
) -> list[dict]:
    """Evaluate all 120 permutations, returning per-policy results."""
    x0 = doors_initial_state(cfg)

    # Build a memory with starting room discovered (for signature computation)
    base_memory = ReactiveMemory.empty()
    start_room = cfg.loc_room[cfg.start_loc]
    base_memory.discover_keys_in_room(start_room, cfg)
    base_memory.mark_searched(start_room)

    results = []
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        policy = canonical_partial_map_policy(perm)

        # Run episode from initial state
        env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
        episode = run_reactive_partial_episode(
            env, policy, rt, cfg, x0=x0, is_solved=cfg.is_solved,
        )

        action_trace = [step.action for step in episode.steps]

        # Compute behavioral signature on evaluation states with base memory
        sig = reactive_semantic_signature(
            policy, rt, eval_states, memory=base_memory,
        )

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
    """Assign equivalence class IDs based on signature."""
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
        f"# Reactive Sketch Partial-Map Exact Oracle — D={D}",
        "",
        f"- **Permutations evaluated:** {len(results)}",
        f"- **Evaluation states:** {n_eval_states}",
        f"- **Equivalence classes:** {len(classes)}",
        f"- **Solving permutations:** {sum(1 for r in results if r['solved'])}/120",
        f"- **Solving classes:** {sum(1 for c in classes if c['solved'])}/{len(classes)}",
        f"- **Optimal reward (known map):** {optimal_return(D):.4f}",
        "",
        "## Equivalence Classes",
        "",
        "| Class | Solved | Reward | Steps | Members | Action Trace |",
        "|------:|:------:|-------:|------:|--------:|:-------------|",
    ]

    for c in classes:
        solved_str = "yes" if c["solved"] else "no"
        trace_str = ", ".join(str(a) for a in c["action_trace"])
        lines.append(
            f"| {c['id']} | {solved_str} | {c['reward']:+.4f} | "
            f"{c['steps']} | {c['count']} | [{trace_str}] |"
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reactive Sketch — Partial-Map Exact Oracle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--D", nargs="+", type=int, default=[2],
        help="Room counts to evaluate (default: 2)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/reactive_partial_map",
        help="Output directory (default: results/reactive_partial_map)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    section("Reactive Sketch — Partial-Map Exact Oracle")
    print(f"  D values:   {args.D}")
    print(f"  output_dir: {output_dir}")

    for D in sorted(args.D):
        cfg = build_config(D)
        rt = DoorsRelationalRuntime(cfg, known_map=False)
        eval_states = make_frozen_state_suite(cfg)

        section(f"D={D}  ({len(eval_states)} evaluation states)")

        results = evaluate_all_permutations(D, cfg, rt, eval_states)
        classes = assign_equivalence_classes(results)

        n_solved = sum(1 for r in results if r["solved"])
        n_solving_classes = sum(1 for c in classes if c["solved"])
        print(f"  Permutations:        120")
        print(f"  Solving:             {n_solved}/120")
        print(f"  Equivalence classes: {len(classes)}")
        print(f"  Solving classes:     {n_solving_classes}/{len(classes)}")
        print(f"  Optimal (known map): {optimal_return(D):.4f}")
        print()

        for c in classes:
            solved_str = "SOLVE" if c["solved"] else "FAIL "
            trace_names = [format_action_name(a, cfg) for a in c["action_trace"]]
            trace_str = ", ".join(trace_names)
            orders_str = " ".join(
                f"({','.join(str(x) for x in o)})" for o in c["member_orders"][:5]
            )
            if c["count"] > 5:
                orders_str += f" ... (+{c['count'] - 5} more)"
            print(
                f"  Class {c['id']:2d} [{solved_str}] "
                f"reward={c['reward']:+.4f} steps={c['steps']:2d}  "
                f"trace=[{trace_str}]"
            )
            print(f"           members ({c['count']}): {orders_str}")

        # Save outputs
        policies_path = output_dir / f"D{D}_policies.json"
        with open(policies_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {policies_path}")

        equiv_path = output_dir / f"D{D}_equivalence.json"
        with open(equiv_path, "w") as f:
            json.dump(classes, f, indent=2)
        print(f"  Saved: {equiv_path}")

        md = format_summary_md(D, results, classes, len(eval_states))
        md_path = output_dir / f"D{D}_summary.md"
        with open(md_path, "w") as f:
            f.write(md + "\n")
        print(f"  Saved: {md_path}")

    print(f"\n  All results in: {output_dir}")


if __name__ == "__main__":
    main()
