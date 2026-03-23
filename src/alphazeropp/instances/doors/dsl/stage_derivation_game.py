"""StageDerivationGame: MCTS-guided hole-filling for Doors stage-skeleton DSL.

Casts stage DSL policy construction as a Game (compatible with core MCTS):
  - States are partial stage programs (PartialStageProgram with holes)
  - Actions are hole fillings: structure decisions, guard atoms, action atoms
  - Terminal reward = LeafEvaluator score of the compiled raw AST

Variable-length episodes (1 to 3*max_stages+1 steps), no dead ends.
A HoleSelectionPolicy deterministically picks which hole to fill;
the agent chooses the filling for that hole.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

import gymnasium.spaces as spaces
import numpy as np

from alphazeropp.core.game import Game
from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
from alphazeropp.instances.doors.dsl.stage_dsl import GuardExpr, StageProgram
from alphazeropp.instances.doors.dsl.stage_compiler import compile_stage_program
from alphazeropp.instances.doors.dsl.stage_grammar import enumerate_guards, enumerate_actions
from alphazeropp.instances.doors.dsl.stage_search_cost import SearchCostModel
from alphazeropp.instances.doors.dsl.stage_partial import (
    PartialStageProgram, HoleRef,
    StructureHole, GuardHoleRef, ActionHoleRef,
    AddStage, Finalize, apply_filling,
)
from alphazeropp.instances.doors.dsl.stage_hole_selection import HoleSelectionPolicy
from alphazeropp.instances.doors.dsl.surface_dsl import SurfaceAction
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


# ---------------------------------------------------------------------------
# Observation encoding constants
# ---------------------------------------------------------------------------

STAGE_TOKEN_IDS = {
    "PAD": 0,
    "STRUCTURE": 1,
    "GUARD": 2,
    "ACTION": 3,
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageDerivationState:
    """Immutable state for stage-skeleton hole-filling.

    partial:   the current partial stage program
    decisions: sequence of (type_id, param) choices made so far
    """
    partial: PartialStageProgram
    decisions: tuple[tuple[int, int], ...]

    @classmethod
    def initial(cls, max_stages: int) -> StageDerivationState:
        return cls(
            partial=PartialStageProgram(
                stages=(), max_stages=max_stages, structure_finalized=False,
            ),
            decisions=(),
        )

    def apply_decision(
        self,
        type_id: int,
        param: int,
        new_partial: PartialStageProgram,
    ) -> StageDerivationState:
        return StageDerivationState(
            partial=new_partial,
            decisions=self.decisions + ((type_id, param),),
        )


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class StageDerivationGame(Game):
    """Single-player game where actions fill holes in a stage-skeleton program.

    Action layout (total 2 + Ng + Na):
      0              → AddStage     (StructureHole)
      1              → Finalize     (StructureHole)
      2..2+Ng-1      → guard atoms  (GuardHoleRef)
      2+Ng..2+Ng+Na-1→ action atoms (ActionHoleRef)

    Legal mask enables only the slice matching the current hole type.

    Observation: (type_id, param) pairs for each decision made,
    shape (2 * max_decisions,) where max_decisions = 3*max_stages + 1.
    """

    def __init__(
        self,
        doors_cfg: DoorsGameConfig,
        cost_model: SearchCostModel,
        leaf_evaluator: LeafEvaluator,
        hole_selection_policy: HoleSelectionPolicy,
    ):
        super().__init__()
        self.doors_cfg = doors_cfg
        self.cost_model = cost_model
        self.leaf_evaluator = leaf_evaluator
        self.hole_selection_policy = hole_selection_policy

        # Pre-compute vocabularies
        all_guards = enumerate_guards(doors_cfg, cost_model.max_guard_depth)
        self._guards = [g for g in all_guards if cost_model.admits_guard(g)]
        self._actions = enumerate_actions(doors_cfg)

        self._n_guards = len(self._guards)
        self._n_actions = len(self._actions)
        self._total_actions = 2 + self._n_guards + self._n_actions

        # Index maps for action↔filling conversion
        self._guard_to_idx = {g: i for i, g in enumerate(self._guards)}
        self._action_to_idx = {a: i for i, a in enumerate(self._actions)}

        # Episode geometry
        self._max_stages = cost_model.max_stages
        self._max_decisions = 3 * self._max_stages + 1

        self.action_space = spaces.Discrete(self._total_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 * self._max_decisions,), dtype=np.float32,
        )

        self._state: StageDerivationState | None = None
        self._current_hole: HoleRef | None = None

    # -- Action ↔ Filling mapping --

    def _action_to_filling(self, action: int):
        """Convert action index to (hole_category, filling)."""
        if action == 0:
            return "structure", AddStage()
        if action == 1:
            return "structure", Finalize()
        if action < 2 + self._n_guards:
            return "guard", self._guards[action - 2]
        return "action", self._actions[action - 2 - self._n_guards]

    def _filling_to_action(self, hole_ref: HoleRef, filling) -> int:
        """Convert filling to action index (for diagnostics)."""
        if isinstance(hole_ref, StructureHole):
            return 0 if isinstance(filling, AddStage) else 1
        if isinstance(hole_ref, GuardHoleRef):
            return 2 + self._guard_to_idx[filling]
        return 2 + self._n_guards + self._action_to_idx[filling]

    # -- Game interface --

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self._state = StageDerivationState.initial(self._max_stages)
        self._current_hole = self.hole_selection_policy.select_hole(
            self._state.partial,
        )
        obs = self._encode_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        category, filling = self._action_to_filling(action)

        new_partial = apply_filling(
            self._state.partial, self._current_hole, filling,
        )

        # Encode decision token
        tid = STAGE_TOKEN_IDS.get(category.upper(), 0)
        if category == "structure":
            param = 0 if isinstance(filling, AddStage) else 1
        elif category == "guard":
            param = self._guard_to_idx[filling]
        else:
            param = self._action_to_idx[filling]

        self._state = self._state.apply_decision(tid, param, new_partial)

        is_terminal = new_partial.is_terminal()
        info: dict[str, Any] = {
            "hole_type": type(self._current_hole).__name__,
            "branching": int(self.get_action_mask().sum()) if not is_terminal else 0,
            "is_complete": is_terminal,
            "is_dead_end": False,
        }

        if is_terminal:
            prog = new_partial.to_stage_program()
            ast = compile_stage_program(prog, self.doors_cfg)
            reward = self.leaf_evaluator(ast)
            self.leaf_evaluator._surface_labels[ast.pretty()] = prog.pretty()
            info["program"] = ast
            info["stage_program"] = prog
            info["leaf_value"] = reward
        else:
            reward = 0.0
            self._current_hole = self.hole_selection_policy.select_hole(
                new_partial,
            )

        obs = self._encode_obs()
        return obs, reward, is_terminal, False, info

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self._total_actions, dtype=bool)
        hole = self._current_hole

        if isinstance(hole, StructureHole):
            n_stages = len(self._state.partial.stages)
            if n_stages < self._max_stages:
                mask[0] = True   # AddStage
            mask[1] = True       # Finalize (always legal)
        elif isinstance(hole, GuardHoleRef):
            mask[2: 2 + self._n_guards] = True
        elif isinstance(hole, ActionHoleRef):
            mask[2 + self._n_guards: 2 + self._n_guards + self._n_actions] = True

        return mask

    # -- Observation encoding --

    def _encode_obs(self) -> np.ndarray:
        obs = np.zeros(2 * self._max_decisions, dtype=np.float32)
        for i, (type_id, param) in enumerate(self._state.decisions):
            obs[2 * i] = float(type_id)
            obs[2 * i + 1] = float(param)
        return obs

    # -- Hashable obs (for MCTS tree nodes) --

    @property
    def hashable_obs(self) -> tuple:
        return self._state.decisions

    # -- Stash / Unstash (lightweight, all state is immutable) --

    def stash_state(self) -> tuple:
        return (
            self._state,
            self._current_hole,
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
            self._current_hole,
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
        ) = state
        return self

    def clone(self) -> StageDerivationGame:
        new = StageDerivationGame(
            self.doors_cfg, self.cost_model,
            self.leaf_evaluator, self.hole_selection_policy,
        )
        new.unstash_state(self.stash_state())
        if self.obs is not None:
            new.obs = self.obs.copy()
        return new
