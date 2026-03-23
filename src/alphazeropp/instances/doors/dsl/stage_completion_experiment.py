"""Experiment runner: compare completion orders on stage-skeleton DSL."""

from __future__ import annotations

from dataclasses import dataclass

from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
from alphazeropp.instances.doors.dsl.stage_diagnostics import make_frozen_state_suite
from alphazeropp.instances.doors.dsl.stage_search_cost import SearchCostModel
from alphazeropp.instances.doors.dsl.stage_hole_selection import (
    LeftmostPolicy, StructureFirstPolicy,
)
from alphazeropp.instances.doors.dsl.stage_search import SearchResult, best_first_search


def run_comparison(
    D: int,
    max_stages: int = 3,
    max_guard_depth: int = 0,
    max_expansions: int = 1_000_000,
    dedup: bool = True,
) -> dict[str, SearchResult]:
    """Run both completion orders and return results keyed by policy name."""
    cfg = DoorsGameConfig(num_rooms=D)
    cost_model = SearchCostModel(
        max_stages=max_stages,
        max_guard_depth=max_guard_depth,
    )
    frozen_states = make_frozen_state_suite(cfg)

    policies = [LeftmostPolicy(), StructureFirstPolicy()]
    results = {}

    for policy in policies:
        result = best_first_search(
            cfg, cost_model, policy, frozen_states,
            max_expansions=max_expansions,
            dedup=dedup,
        )
        results[policy.name] = result

    return results


def format_comparison_table(results: dict[str, SearchResult]) -> str:
    """Format results as a markdown comparison table."""
    lines = [
        "| Metric | " + " | ".join(results.keys()) + " |",
        "|--------|" + "|".join(["-------:" for _ in results]) + "|",
    ]

    metrics = [
        ("Candidates expanded", lambda r: f"{r.stats.candidates_expanded:,}"),
        ("Complete evaluated", lambda r: f"{r.stats.complete_evaluated:,}"),
        ("Solving programs", lambda r: f"{r.stats.solving_programs:,}"),
        ("Distinct partials", lambda r: f"{r.stats.semantically_distinct_partial:,}" if r.stats.semantically_distinct_partial >= 0 else "—"),
        ("Distinct complete", lambda r: f"{r.stats.semantically_distinct_complete:,}"),
        ("Time to first solver", lambda r: f"{r.stats.time_to_first_solver:.3f}s" if r.stats.time_to_first_solver is not None else "—"),
        ("Total time", lambda r: f"{r.stats.time_total:.3f}s"),
        ("Dedup hits", lambda r: f"{r.stats.dedup_hits:,}"),
        ("Dedup misses", lambda r: f"{r.stats.dedup_misses:,}"),
        ("Max heap size", lambda r: f"{r.stats.heap_max_size:,}"),
    ]

    for label, fn in metrics:
        vals = " | ".join(fn(results[k]) for k in results)
        lines.append(f"| {label} | {vals} |")

    return "\n".join(lines)


def format_detailed_report(
    results: dict[str, SearchResult],
    D: int,
    max_stages: int,
    max_guard_depth: int,
    dedup: bool,
) -> str:
    """Format a full markdown report."""
    header = (
        f"# Completion-Order Comparison: D={D}, max_stages={max_stages}, "
        f"depth≤{max_guard_depth}, dedup={'on' if dedup else 'off'}\n\n"
    )

    table = format_comparison_table(results)

    # Branch factor summary
    bf_lines = ["\n## Branch factor by depth\n"]
    for name, result in results.items():
        bf_lines.append(f"\n### {name}\n")
        bf_lines.append("| Depth | Min | Mean | Max | Count |")
        bf_lines.append("|------:|----:|-----:|----:|------:|")
        for depth in sorted(result.stats.branch_factor_by_depth.keys()):
            bfs = result.stats.branch_factor_by_depth[depth]
            if bfs:
                mn = min(bfs)
                mx = max(bfs)
                mean = sum(bfs) / len(bfs)
                bf_lines.append(f"| {depth} | {mn} | {mean:.1f} | {mx} | {len(bfs)} |")

    return header + table + "\n".join(bf_lines)
