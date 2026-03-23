"""ReactiveDerivationGame: MCTS-guided branch composition for Doors reactive BT.

Casts reactive BT policy construction as a Game (compatible with core MCTS):
  - States are partial branch selections (predicate + action choices)
  - Actions are catalog selections: predicate IDs or action IDs
  - Terminal reward = episode reward of the assembled BT policy

Fixed-length episodes of exactly 2*N steps (N branches, 2 choices each).
Steps alternate: predicate choice (even steps), action choice (odd steps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import gymnasium.spaces as spaces
import numpy as np

from alphazeropp.core.game import Game
from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state,
)
from alphazeropp.instances.doors.dsl.reactive_branch_catalog import (
    ReactiveBranchCatalog,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import (
    run_reactive_episode,
)


# ---------------------------------------------------------------------------
# Observation encoding constants
# ---------------------------------------------------------------------------

REACTIVE_TOKEN_IDS = {
    "PAD": 0,
    "PREDICATE": 1,
    "ACTION": 2,
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReactiveDerivationState:
    """Immutable state for reactive branch composition.

    branch_specs: completed (pred_idx, act_idx) pairs
    pending_pred: predicate chosen for current branch (None if waiting for pred)
    decisions: (type_id, param) history for observation encoding
    """
    branch_specs: tuple[tuple[int, int], ...]
    pending_pred: int | None
    decisions: tuple[tuple[int, int], ...]

    @classmethod
    def initial(cls) -> ReactiveDerivationState:
        return cls(branch_specs=(), pending_pred=None, decisions=())

    def choose_predicate(self, pred_idx: int) -> ReactiveDerivationState:
        """Record a predicate choice for the current branch."""
        return ReactiveDerivationState(
            branch_specs=self.branch_specs,
            pending_pred=pred_idx,
            decisions=self.decisions + ((REACTIVE_TOKEN_IDS["PREDICATE"], pred_idx),),
        )

    def choose_action(self, act_idx: int) -> ReactiveDerivationState:
        """Record an action choice, completing the current branch."""
        assert self.pending_pred is not None
        return ReactiveDerivationState(
            branch_specs=self.branch_specs + ((self.pending_pred, act_idx),),
            pending_pred=None,
            decisions=self.decisions + ((REACTIVE_TOKEN_IDS["ACTION"], act_idx),),
        )

    @property
    def is_predicate_step(self) -> bool:
        """True if the next choice is a predicate (even step)."""
        return self.pending_pred is None


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class ReactiveDerivationGame(Game):
    """Single-player game where actions compose reactive BT branches.

    Action layout: Discrete(max(n_predicates, n_actions)).
    Legal mask alternates between predicate and action masks.

    On even steps (predicate choice): indices 0..n_pred-1 are legal
      for predicates that have at least one compatible action.
    On odd steps (action choice): indices 0..n_act-1 are legal
      where legal_matrix[pending_pred, idx] is True.

    Observation: (type_id, param) pairs, shape (2 * 2 * n_branches,).
    """

    def __init__(
        self,
        doors_cfg: DoorsGameConfig,
        catalog: ReactiveBranchCatalog,
        n_branches: int = 4,
        leaf_evaluator=None,
    ):
        super().__init__()
        self.doors_cfg = doors_cfg
        self.catalog = catalog
        self.n_branches = n_branches
        self.leaf_evaluator = leaf_evaluator  # Optional ReactiveLeafEvaluator

        self._n_pred = catalog.n_predicates
        self._n_act = catalog.n_actions
        self._total_actions = max(self._n_pred, self._n_act)
        self._max_steps = 2 * n_branches
        self._max_decisions = self._max_steps

        self.action_space = spaces.Discrete(self._total_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 * self._max_decisions,), dtype=np.float32,
        )

        # Pre-compute which predicates have at least one legal action
        self._valid_preds = catalog.predicates_with_legal_actions()

        # Inline evaluation setup (used when leaf_evaluator is None)
        self._x0 = doors_initial_state(doors_cfg)
        self._rt = DoorsRelationalRuntime(doors_cfg)

        # Reward cache: branch_specs -> reward (used only for inline eval)
        self._reward_cache: dict[tuple, float] = {}

        self._state: ReactiveDerivationState | None = None

    # -- Game interface --

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self._state = ReactiveDerivationState.initial()
        obs = self._encode_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._state.is_predicate_step:
            self._state = self._state.choose_predicate(action)
            is_terminal = False
            reward = 0.0
            info: dict[str, Any] = {
                "step_type": "predicate",
                "choice": action,
                "predicate_name": self.catalog.predicate_names[action],
            }
        else:
            self._state = self._state.choose_action(action)
            n_completed = len(self._state.branch_specs)
            is_terminal = n_completed == self.n_branches
            info = {
                "step_type": "action",
                "choice": action,
                "action_name": self.catalog.action_names[action],
                "branches_completed": n_completed,
            }

            if is_terminal:
                specs = self._state.branch_specs
                reward = self._evaluate(specs)
                info["branch_specs"] = specs
                info["branch_names"] = self.catalog.spec_names(specs)
                info["leaf_value"] = reward
            else:
                reward = 0.0

        obs = self._encode_obs()
        return obs, reward, is_terminal, False, info

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self._total_actions, dtype=bool)

        if self._state.is_predicate_step:
            # Predicate step: enable all predicates with legal action partners
            for p in self._valid_preds:
                mask[p] = True
        else:
            # Action step: enable actions compatible with pending predicate
            pred_idx = self._state.pending_pred
            legal = self.catalog.legal_actions_for_predicate(pred_idx)
            for a in legal:
                mask[a] = True

        return mask

    # -- Evaluation --

    def _evaluate(self, specs: tuple[tuple[int, int], ...]) -> float:
        """Evaluate a complete policy by running an episode."""
        # Delegate to leaf_evaluator if available
        if self.leaf_evaluator is not None:
            return self.leaf_evaluator(specs)

        # Inline evaluation (backward compatible)
        if specs in self._reward_cache:
            return self._reward_cache[specs]

        policy = self.catalog.build_policy(specs)
        env = self.doors_cfg.make_env(
            self.doors_cfg.obs_size(), frozen_states=[self._x0],
        )
        result = run_reactive_episode(
            env, policy, self._rt,
            x0=self._x0, is_solved=self.doors_cfg.is_solved,
        )
        reward = result.cumulative_reward
        self._reward_cache[specs] = reward
        return reward

    # -- Leaf evaluator cache support (for multiprocessing) --

    def export_leaf_caches(self) -> dict | None:
        """Export leaf evaluator caches for cross-process aggregation."""
        if self.leaf_evaluator is not None:
            return self.leaf_evaluator.export_caches()
        return None

    def merge_leaf_caches(self, other: dict):
        """Merge leaf evaluator caches from a worker."""
        if self.leaf_evaluator is not None and other is not None:
            self.leaf_evaluator.merge_caches(other)

    # -- Observation encoding --

    def _encode_obs(self) -> np.ndarray:
        obs = np.zeros(2 * self._max_decisions, dtype=np.float32)
        for i, (type_id, param) in enumerate(self._state.decisions):
            obs[2 * i] = float(type_id)
            obs[2 * i + 1] = float(param)
        return obs

    # -- Hashable obs --

    @property
    def hashable_obs(self) -> tuple:
        return self._state.decisions

    # -- Stash / Unstash --

    def stash_state(self) -> tuple:
        return (
            self._state,
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
        )

    def unstash_state(self, state: tuple):
        (
            self._state,
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
        ) = state
        return self

    def clone(self) -> ReactiveDerivationGame:
        new = ReactiveDerivationGame(
            self.doors_cfg, self.catalog, self.n_branches,
            leaf_evaluator=self.leaf_evaluator,
        )
        new._reward_cache = self._reward_cache  # share cache
        new.unstash_state(self.stash_state())
        if self.obs is not None:
            new.obs = self.obs.copy()
        return new
