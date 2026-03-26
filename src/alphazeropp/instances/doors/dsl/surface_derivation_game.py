"""SurfaceDerivationGame: MCTS-guided surface-rule sequencing for Doors.

Casts surface DSL policy construction as a Game (compatible with core MCTS):
  - States are partial rule sequences (no budget holes, no raw AST)
  - Actions are surface rule selections: PickRule(k), MoveRule(k), GoalRule
  - Terminal reward = LeafEvaluator score of the compiled raw AST

Fixed-length episodes of exactly 2K+1 steps, no dead ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import gymnasium.spaces as spaces
import numpy as np

from alphazeropp.core.game import Game
from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
from alphazeropp.instances.doors.dsl.surface_dsl import (
    PickRule, MoveRule, GoalRule, SurfaceRule, SurfacePolicy,
)
from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


# ---------------------------------------------------------------------------
# Observation encoding constants
# ---------------------------------------------------------------------------

SURFACE_TOKEN_IDS = {
    "PAD": 0,
    "PICK": 1,
    "MOVE": 2,
    "GOAL": 3,
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceDerivationState:
    """Immutable state for surface rule sequencing.

    rules:       placed rules so far (tuple for immutability)
    picked_mask: bitmask — bit k set iff PickRule(k) has been placed
    moved_mask:  bitmask — bit k set iff MoveRule(k) has been placed
    """
    rules: tuple[SurfaceRule, ...]
    picked_mask: int
    moved_mask: int

    @classmethod
    def initial(cls) -> SurfaceDerivationState:
        return cls(rules=(), picked_mask=0, moved_mask=0)

    def place_rule(self, rule: SurfaceRule) -> SurfaceDerivationState:
        """Return new state with *rule* appended."""
        new_rules = self.rules + (rule,)
        new_picked = self.picked_mask
        new_moved = self.moved_mask
        if isinstance(rule, PickRule):
            new_picked |= (1 << rule.k)
        elif isinstance(rule, MoveRule):
            new_moved |= (1 << rule.k)
        return SurfaceDerivationState(new_rules, new_picked, new_moved)


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class SurfaceDerivationGame(Game):
    """Single-player game where actions are surface rule selections.

    Action layout (total 2K+1 actions):
      0..K-1    → PickRule(k)
      K..2K-1   → MoveRule(k-K)
      2K        → GoalRule

    Legal mask (strict mode, allow_early_goal=False):
      PickRule(k): legal iff k not in picked_mask AND k not in moved_mask
      MoveRule(k): legal iff k in picked_mask AND k not in moved_mask
      GoalRule:    legal iff all 2K non-goal rules have been placed

    Observation: fixed-length (type_id, param) pairs, shape (2*(2K+1),).
    """

    def __init__(
        self,
        num_rooms: int,
        leaf_evaluator: LeafEvaluator,
        doors_cfg: DoorsGameConfig,
        *,
        allow_early_goal: bool = False,
    ):
        super().__init__()
        self.num_rooms = num_rooms
        self.K = num_rooms - 1
        self.leaf_evaluator = leaf_evaluator
        self.doors_cfg = doors_cfg
        self.allow_early_goal = allow_early_goal

        self._max_steps = 2 * self.K + 1
        self._n_actions = self._max_steps  # 2K+1

        self.action_space = spaces.Discrete(self._n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 * self._max_steps,), dtype=np.float32,
        )

        self._state: SurfaceDerivationState | None = None

    # -- Action ↔ Rule mapping --

    def _action_to_rule(self, action: int) -> SurfaceRule:
        if action < self.K:
            return PickRule(action)
        elif action < 2 * self.K:
            return MoveRule(action - self.K)
        else:
            return GoalRule()

    def _rule_to_action(self, rule: SurfaceRule) -> int:
        if isinstance(rule, PickRule):
            return rule.k
        elif isinstance(rule, MoveRule):
            return self.K + rule.k
        else:
            return 2 * self.K

    # -- Game interface --

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self._state = SurfaceDerivationState.initial()
        obs = self._encode_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        rule = self._action_to_rule(action)
        self._state = self._state.place_rule(rule)

        is_terminal = isinstance(rule, GoalRule)
        info: dict[str, Any] = {
            "rule": rule,
            "n_rules_placed": len(self._state.rules),
        }

        if is_terminal:
            policy = SurfacePolicy(self._state.rules)
            program = compile_policy(policy, self.doors_cfg)
            reward = self.leaf_evaluator(program)
            self.leaf_evaluator._surface_labels[program.pretty()] = policy.pretty()
            info["program"] = program
            info["policy"] = policy
            info["leaf_value"] = reward
        else:
            reward = 0.0

        obs = self._encode_obs()
        return obs, reward, is_terminal, False, info

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self._n_actions, dtype=bool)
        K = self.K
        n_placed = len(self._state.rules)

        for k in range(K):
            # PickRule(k): legal iff not yet picked and not yet moved
            if not (self._state.picked_mask & (1 << k)) and \
               not (self._state.moved_mask & (1 << k)):
                mask[k] = True
            # MoveRule(k): legal iff picked but not yet moved
            if (self._state.picked_mask & (1 << k)) and \
               not (self._state.moved_mask & (1 << k)):
                mask[K + k] = True

        # GoalRule: legal iff all 2K non-goal rules placed
        if self.allow_early_goal:
            # Always legal (for future ablation)
            mask[2 * K] = True
        else:
            if n_placed == 2 * K:
                mask[2 * K] = True

        return mask

    # -- Observation encoding --

    def _encode_obs(self) -> np.ndarray:
        """Encode partial rule sequence as (type_id, param) pairs."""
        obs = np.zeros(2 * self._max_steps, dtype=np.float32)
        for i, rule in enumerate(self._state.rules):
            if isinstance(rule, PickRule):
                obs[2 * i] = SURFACE_TOKEN_IDS["PICK"]
                obs[2 * i + 1] = float(rule.k)
            elif isinstance(rule, MoveRule):
                obs[2 * i] = SURFACE_TOKEN_IDS["MOVE"]
                obs[2 * i + 1] = float(rule.k)
            elif isinstance(rule, GoalRule):
                obs[2 * i] = SURFACE_TOKEN_IDS["GOAL"]
                obs[2 * i + 1] = 0.0
        return obs

    # -- Hashable obs --

    @property
    def hashable_obs(self) -> tuple:
        return self._state.rules

    # -- Stash / Unstash (lightweight, all state is immutable) --

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

    def clone(self) -> SurfaceDerivationGame:
        new = SurfaceDerivationGame(
            self.num_rooms, self.leaf_evaluator, self.doors_cfg,
            allow_early_goal=self.allow_early_goal,
        )
        new.unstash_state(self.stash_state())
        if self.obs is not None:
            new.obs = self.obs.copy()
        return new
