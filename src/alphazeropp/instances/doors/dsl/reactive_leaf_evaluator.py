"""Leaf evaluator for reactive BT policies.

Evaluates complete reactive BT policies (specified as branch_specs tuples)
on frozen initial states, returning a scalar value for MCTS backpropagation.
Results are cached by branch_specs to avoid redundant evaluation.

Unlike the AST LeafEvaluator (which takes Program objects and calls
run_policy_episode), this takes branch_specs tuples and calls
run_reactive_episode via the tick interpreter.

Provides compatibility shims (_program_cache, _surface_labels, game_config,
n_sites) so that the shared run_derivation_training() infrastructure in
derivation_utils.py can consume reactive policies alongside AST programs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
from alphazeropp.instances.doors.dsl.reactive_branch_catalog import (
    ReactiveBranchCatalog,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import (
    run_reactive_episode,
)


VALID_METRICS = ("avg_reward", "solve_rate", "weighted")


# ---------------------------------------------------------------------------
# ReactivePolicy wrapper — duck-types the Program interface for
# derivation_utils.extract_best_program / print_best_program_traces
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReactivePolicy:
    """Lightweight wrapper that gives a reactive BT the .pretty() / .node_count()
    interface expected by derivation_utils.print_best_program_traces."""

    branch_specs: tuple[tuple[int, int], ...]
    branch_names: tuple[str, ...]
    bt_root: object  # WhileNot node (the actual executable BT)

    def pretty(self) -> str:
        lines = ["While(Not(GoalReached), Fallback("]
        for i, name in enumerate(self.branch_names):
            comma = "," if i < len(self.branch_names) - 1 else ""
            # Handle both '->' and unicode arrow '→'
            for sep in ("->", "\u2192"):
                if sep in name:
                    parts = name.split(sep, 1)
                    pred_str = parts[0].strip()
                    act_str = parts[1].strip()
                    break
            else:
                pred_str = name
                act_str = "?"
            lines.append(f"  B{i+1}: Seq(Check({pred_str}), Do({act_str})){comma}")
        lines.append("))")
        return "\n".join(lines)

    def node_count(self) -> int:
        return 2 * len(self.branch_specs)


class ReactiveLeafEvaluator:
    """Evaluates complete reactive BT policies on frozen initial states.

    When MCTS reaches a terminal derivation state (all branches filled),
    this assembles the BT, runs it on all frozen states, and returns a
    scalar value determined by the metric parameter.

    Cache key is the branch_specs tuple (hashable tuple of int pairs),
    which is more efficient than the string-based caching in LeafEvaluator.
    """

    def __init__(
        self,
        catalog: ReactiveBranchCatalog,
        doors_cfg: DoorsGameConfig,
        frozen_states: Sequence[np.ndarray],
        *,
        is_solved: Callable[[np.ndarray], bool],
        metric: str = "weighted",
        blend_alpha: float = 0.7,
        known_map: bool = True,
    ):
        if metric not in VALID_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}, must be one of {VALID_METRICS}"
            )
        self.catalog = catalog
        self.doors_cfg = doors_cfg
        self.game_config = doors_cfg  # alias for derivation_utils compatibility
        self.frozen_states = list(frozen_states)
        self.is_solved = is_solved
        self.metric = metric
        self.blend_alpha = blend_alpha
        self.known_map = known_map
        self._rt = DoorsRelationalRuntime(doors_cfg, known_map=known_map)

        # Caching
        self._cache: dict[tuple, float] = {}
        self._full_cache: dict[tuple, dict] = {}

        # Compatibility shims for derivation_utils.extract_best_program()
        self._program_cache: dict[tuple, ReactivePolicy] = {}
        self._surface_labels: dict[tuple, str] = {}

        # Statistics
        self._eval_count = 0
        self._cache_hits = 0
        self._total_env_steps = 0

        # Baseline for delta exports
        self._base_eval_count = 0
        self._base_cache_hits = 0
        self._base_total_env_steps = 0

    @property
    def n_sites(self) -> int:
        """Observation vector size (for derivation_utils compatibility)."""
        return self.doors_cfg.obs_size()

    def __call__(
        self, branch_specs: tuple[tuple[int, int], ...],
    ) -> float:
        """Evaluate policy, returning cached result if available."""
        if branch_specs in self._cache:
            self._cache_hits += 1
            return self._cache[branch_specs]

        metrics = self._evaluate(branch_specs)
        value = self._compute_metric(metrics)
        self._cache[branch_specs] = value
        self._full_cache[branch_specs] = metrics
        return value

    def get_all_metrics(
        self, branch_specs: tuple[tuple[int, int], ...],
    ) -> dict:
        """Return full metrics dict, using cache."""
        if branch_specs not in self._full_cache:
            self(branch_specs)
        return self._full_cache[branch_specs]

    def _evaluate(
        self, branch_specs: tuple[tuple[int, int], ...],
    ) -> dict:
        """Run the policy on all frozen states and collect metrics."""
        policy = self.catalog.build_policy(branch_specs)
        n_sites = self.doors_cfg.obs_size()

        total_reward = 0.0
        total_steps = 0
        n_solved = 0

        for x0 in self.frozen_states:
            env = self.doors_cfg.make_env(n_sites, frozen_states=[x0])
            result = run_reactive_episode(
                env, policy, self._rt, x0=x0, is_solved=self.is_solved,
            )
            total_reward += result.cumulative_reward
            total_steps += result.total_env_steps
            if result.solved:
                n_solved += 1

        n = len(self.frozen_states)
        self._eval_count += 1
        self._total_env_steps += total_steps

        # Populate compatibility caches for derivation_utils
        if branch_specs not in self._program_cache:
            names = self.catalog.spec_names(branch_specs)
            self._program_cache[branch_specs] = ReactivePolicy(
                branch_specs=branch_specs,
                branch_names=tuple(names),
                bt_root=policy,
            )
            self._surface_labels[branch_specs] = " | ".join(names)

        return {
            "solve_rate": n_solved / n if n > 0 else 0.0,
            "avg_reward": total_reward / n if n > 0 else 0.0,
            "avg_steps": total_steps / n if n > 0 else 0,
            "n_solved": n_solved,
            "n_states": n,
            "n_episodes": n,
        }

    def _compute_metric(self, metrics: dict) -> float:
        """Compute scalar from metrics dict based on self.metric."""
        if self.metric == "avg_reward":
            return metrics["avg_reward"]
        if self.metric == "solve_rate":
            return metrics["solve_rate"]
        if self.metric == "weighted":
            sr = metrics["solve_rate"]
            ar = metrics["avg_reward"]
            if sr > 0:
                return self.blend_alpha * sr + (1 - self.blend_alpha) * ar
            return ar
        raise ValueError(f"Unknown metric: {self.metric}")

    def stats(self) -> dict:
        """Return evaluation statistics."""
        return {
            "eval_count": self._eval_count,
            "cache_hits": self._cache_hits,
            "unique_programs": len(self._cache),
            "total_env_steps": self._total_env_steps,
        }

    # -- Multiprocessing support --

    def export_caches(self) -> dict:
        """Export cache data for cross-process aggregation (delta stats)."""
        return {
            "_cache": dict(self._cache),
            "_full_cache": dict(self._full_cache),
            "_program_cache": dict(self._program_cache),
            "_surface_labels": dict(self._surface_labels),
            "_eval_count": self._eval_count - self._base_eval_count,
            "_cache_hits": self._cache_hits - self._base_cache_hits,
            "_total_env_steps": self._total_env_steps - self._base_total_env_steps,
        }

    def merge_caches(self, other: dict):
        """Merge exported caches from a worker evaluator."""
        for key, value in other["_cache"].items():
            if key not in self._cache:
                self._cache[key] = value
                self._full_cache[key] = other["_full_cache"][key]
        for key, value in other.get("_program_cache", {}).items():
            if key not in self._program_cache:
                self._program_cache[key] = value
        for key, label in other.get("_surface_labels", {}).items():
            if key not in self._surface_labels:
                self._surface_labels[key] = label
        self._eval_count += other.get("_eval_count", 0)
        self._cache_hits += other.get("_cache_hits", 0)
        self._total_env_steps += other.get("_total_env_steps", 0)

    def snapshot_baseline(self):
        """Snapshot current stats as baseline for delta exports."""
        self._base_eval_count = self._eval_count
        self._base_cache_hits = self._cache_hits
        self._base_total_env_steps = self._total_env_steps
