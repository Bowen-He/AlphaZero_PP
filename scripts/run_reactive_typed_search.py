#!/usr/bin/env python3
"""Reactive Typed Grammar — Exact Enumeration and Comparison.

Enumerates all policies from the reactive typed grammar for given D values,
computes behavioral signatures, equivalence classes, and comparison reports.

Usage:
    python scripts/run_reactive_typed_search.py --D 2
    python scripts/run_reactive_typed_search.py --D 2 3 --mode both
    python scripts/run_reactive_typed_search.py --D 3 --n-branches 4 --mode typed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime,
)
from alphazeropp.instances.doors.dsl.reactive_branch_catalog import (
    known_map_catalog,
)
from alphazeropp.instances.doors.dsl.reactive_typed_grammar import (
    count_reactive_typed_policies,
    enumerate_reactive_typed_policies,
    evaluate_reactive_typed_policies,
    assign_equivalence_classes,
)
from alphazeropp.instances.doors.dsl.stage_diagnostics import (
    make_frozen_state_suite,
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


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_mode(
    D: int,
    n_branches: int,
    mode: str,
    output_dir: Path,
) -> dict:
    """Run enumeration for a single (D, mode) pair."""
    cfg = build_config(D)
    rt = DoorsRelationalRuntime(cfg)
    catalog = known_map_catalog(mode=mode)
    eval_states = make_frozen_state_suite(cfg)

    total = count_reactive_typed_policies(n_branches, catalog)
    print(f"  Mode: {mode}")
    print(f"  Legal pairs/branch: {catalog.n_legal_pairs()}")
    print(f"  Total policies: {total:,}")

    t0 = time.time()
    specs_list = enumerate_reactive_typed_policies(n_branches, catalog)
    results = evaluate_reactive_typed_policies(
        specs_list, catalog, cfg, rt, eval_states,
    )
    classes = assign_equivalence_classes(results)
    elapsed = time.time() - t0

    n_solved = sum(1 for r in results if r["solved"])
    n_solving_classes = sum(1 for c in classes if c["solved"])
    distinct_sigs = len({tuple(r["signature"]) for r in results})

    # Find first-solve index
    first_solve_idx = None
    for i, r in enumerate(results):
        if r["solved"]:
            first_solve_idx = i
            break

    print(f"  Solving: {n_solved}/{total}")
    print(f"  Distinct signatures: {distinct_sigs}")
    print(f"  Equivalence classes: {len(classes)}")
    print(f"  Solving classes: {n_solving_classes}/{len(classes)}")
    print(f"  First-solve index: {first_solve_idx}")
    print(f"  Wall clock: {elapsed:.2f}s")
    print(f"  Optimal (known map): {optimal_return(D):.4f}")
    print()

    # Show solving classes
    for c in classes:
        if c["solved"]:
            names = " | ".join(
                f"{n}" for n in c["example_names"]
            )
            print(f"    Class {c['id']:3d} [SOLVE] reward={c['reward']:+.4f} "
                  f"steps={c['steps']:2d} members={c['count']}  [{names}]")

    # Save outputs
    summary = {
        "D": D,
        "n_branches": n_branches,
        "mode": mode,
        "total_policies": total,
        "legal_pairs_per_branch": catalog.n_legal_pairs(),
        "solving": n_solved,
        "distinct_signatures": distinct_sigs,
        "equivalence_classes": len(classes),
        "solving_classes": n_solving_classes,
        "first_solve_index": first_solve_idx,
        "wall_clock_seconds": round(elapsed, 3),
        "optimal_return": round(optimal_return(D), 4),
    }

    # Save summary JSON
    summary_path = output_dir / f"D{D}_{mode}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {summary_path}")

    # Save equivalence classes
    # Convert tuples to lists for JSON serialization
    classes_json = []
    for c in classes:
        c_copy = dict(c)
        c_copy["signature"] = list(c["signature"]) if c["signature"] else []
        c_copy["example_spec"] = [list(s) for s in c["example_spec"]]
        classes_json.append(c_copy)

    equiv_path = output_dir / f"D{D}_{mode}_equivalence.json"
    with open(equiv_path, "w") as f:
        json.dump(classes_json, f, indent=2)
    print(f"  Saved: {equiv_path}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reactive Typed Grammar — Exact Enumeration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--D", nargs="+", type=int, default=[2],
        help="Room counts to evaluate (default: 2)",
    )
    parser.add_argument(
        "--n-branches", type=int, default=4,
        help="Number of branches (default: 4)",
    )
    parser.add_argument(
        "--mode", type=str, default="typed",
        choices=["typed", "raw", "both"],
        help="Legal matrix mode (default: typed)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/reactive_typed",
        help="Output directory (default: results/reactive_typed)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = ["typed", "raw"] if args.mode == "both" else [args.mode]

    section("Reactive Typed Grammar — Exact Enumeration")
    print(f"  D values:     {args.D}")
    print(f"  N branches:   {args.n_branches}")
    print(f"  Modes:        {modes}")
    print(f"  Output dir:   {output_dir}")

    all_summaries = []

    for D in sorted(args.D):
        for mode in modes:
            section(f"D={D}, mode={mode}")
            summary = run_mode(D, args.n_branches, mode, output_dir)
            all_summaries.append(summary)

    # Comparison table if both modes
    if len(modes) > 1:
        section("Comparison: Typed vs Raw")
        print("| D | Mode | Total | Solving | Distinct | Classes | First-Solve |")
        print("|--:|:-----|------:|--------:|---------:|--------:|------------:|")
        for s in all_summaries:
            fs = s["first_solve_index"] if s["first_solve_index"] is not None else "—"
            print(
                f"| {s['D']} | {s['mode']} | {s['total_policies']:,} | "
                f"{s['solving']} | {s['distinct_signatures']} | "
                f"{s['equivalence_classes']} | {fs} |"
            )

    print(f"\n  All results in: {output_dir}")


if __name__ == "__main__":
    main()
