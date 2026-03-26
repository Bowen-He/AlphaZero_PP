#!/usr/bin/env python3
"""Compare completion orders for the typed stage-skeleton DSL.

Usage:
    python scripts/run_stage_completion_orders.py --d 2
    python scripts/run_stage_completion_orders.py --d 3 --max-stages 3
    python scripts/run_stage_completion_orders.py --d 3 --max-stages 3 --no-dedup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphazeropp.instances.doors.dsl.stage_completion_experiment import (
    run_comparison, format_comparison_table, format_detailed_report,
)


def main():
    parser = argparse.ArgumentParser(description="Compare completion orders")
    parser.add_argument("--d", type=int, default=2, help="Number of rooms (default: 2)")
    parser.add_argument("--max-stages", type=int, default=None,
                        help="Max stages (default: 2K+1)")
    parser.add_argument("--max-depth", type=int, default=0,
                        help="Max guard depth (default: 0)")
    parser.add_argument("--max-expansions", type=int, default=1_000_000,
                        help="Max search expansions (default: 1M)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable semantic dedup")
    parser.add_argument("--output", type=str, default=None,
                        help="Write report to this file")
    args = parser.parse_args()

    D = args.d
    K = D - 1
    max_stages = args.max_stages if args.max_stages is not None else 2 * K + 1
    dedup = not args.no_dedup

    print(f"{'=' * 60}")
    print(f"  Completion-Order Comparison: D={D}")
    print(f"  max_stages={max_stages}, max_guard_depth={args.max_depth}")
    print(f"  max_expansions={args.max_expansions:,}, dedup={'on' if dedup else 'off'}")
    print(f"{'=' * 60}\n")

    results = run_comparison(
        D=D,
        max_stages=max_stages,
        max_guard_depth=args.max_depth,
        max_expansions=args.max_expansions,
        dedup=dedup,
    )

    print(format_comparison_table(results))
    print()

    for name, result in results.items():
        if result.solving_programs:
            print(f"\n  First solver ({name}):")
            print(f"  {result.solving_programs[0].pretty()}")

    if args.output:
        report = format_detailed_report(results, D, max_stages, args.max_depth, dedup)
        Path(args.output).write_text(report)
        print(f"\n  Report written to {args.output}")

    print()


if __name__ == "__main__":
    main()
