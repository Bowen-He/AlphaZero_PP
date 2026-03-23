#!/usr/bin/env python3
"""Build oracle prefix dataset for reactive typed grammar pretraining.

For each D, exhaustively evaluates all policies, computes prefix oracle
labels (V_max, P_solve, best_next_mask), and saves as JSONL.

Usage:
    python scripts/build_reactive_prefix_dataset.py --D 2 3
    python scripts/build_reactive_prefix_dataset.py --D 2 3 --mode typed --n-branches 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
from alphazeropp.instances.doors.dsl.reactive_prefix_oracle import (
    build_prefix_oracle,
)
from alphazeropp.instances.doors.dsl.reactive_prefix_dataset import (
    build_dataset_from_oracle, ReactivePrefixDataset,
)
from alphazeropp.instances.doors.dsl.reactive_derivation_game import (
    ReactiveDerivationGame,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build oracle prefix dataset for reactive grammar",
    )
    parser.add_argument("--D", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--n-branches", type=int, default=4)
    parser.add_argument("--mode", type=str, default="typed")
    parser.add_argument(
        "--output-dir", type=str,
        default="results/reactive_prefix_dataset",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building prefix dataset: D={args.D}, mode={args.mode}, N={args.n_branches}")

    all_datasets = []

    for D in sorted(args.D):
        t0 = time.time()
        print(f"\n--- D={D} ---")

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

        # Build oracle
        print(f"  Building oracle (evaluating all policies)...")
        oracle = build_prefix_oracle(args.n_branches, catalog, evaluator)
        print(f"  Oracle entries: {len(oracle)}")
        print(f"  Eval stats: {evaluator.stats()}")

        root = oracle[()]
        print(f"  Root: v_max={root.v_max:.4f}, p_solve={root.p_solve:.4f}, "
              f"n_solving={root.n_solving}/{root.n_completions}")

        # Build dataset by replaying through game
        print(f"  Replaying prefixes through game...")
        game = ReactiveDerivationGame(cfg, catalog, n_branches=args.n_branches)
        dataset = build_dataset_from_oracle(
            oracle, game, D, args.n_branches, args.mode,
        )
        print(f"  Dataset entries: {len(dataset)}")

        # Save
        path = output_dir / f"D{D}_{args.mode}.jsonl"
        dataset.save(path)
        elapsed = time.time() - t0
        print(f"  Saved: {path} ({elapsed:.1f}s)")

        all_datasets.append(dataset)

    # Summary
    merged = ReactivePrefixDataset.merge(*all_datasets)
    print(f"\n=== Total: {len(merged)} entries across D={args.D} ===")


if __name__ == "__main__":
    main()
