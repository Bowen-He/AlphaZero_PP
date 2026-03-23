"""Tests for ReactiveDerivationGame D-scaling and leaf_evaluator integration."""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.reactive_branch_catalog import (
    known_map_catalog, PRED_PICKABLE, PRED_KNOWN_LOC, PRED_REACHABLE,
    PRED_EXISTS_FRONTIER, ACT_PICK, ACT_GOTO_KEY, ACT_GOTO_GOAL, ACT_GOTO_ENTRANCE,
)
from alphazeropp.instances.doors.dsl.reactive_derivation_game import (
    ReactiveDerivationGame,
)
from alphazeropp.instances.doors.dsl.reactive_leaf_evaluator import (
    ReactiveLeafEvaluator,
)


@pytest.fixture
def catalog():
    return known_map_catalog(mode="typed")


def _make_game(D, catalog, use_leaf_evaluator=False):
    params = compute_doors_derived_params(D, 2)
    cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2, horizon=params["horizon"])
    le = None
    if use_leaf_evaluator:
        le = ReactiveLeafEvaluator(
            catalog, cfg, [doors_initial_state(cfg)], is_solved=cfg.is_solved,
        )
    return ReactiveDerivationGame(cfg, catalog, n_branches=4, leaf_evaluator=le)


class TestDScaling:

    def test_game_works_d4(self, catalog):
        """Game runs for D=4 without error."""
        game = _make_game(4, catalog)
        obs, _ = game.reset()
        terminated = False
        while not terminated:
            mask = game.get_action_mask()
            action = int(np.where(mask)[0][0])
            obs, reward, terminated, truncated, info = game.step(action)
        assert terminated

    def test_game_works_d5(self, catalog):
        """Game runs for D=5 without error."""
        game = _make_game(5, catalog)
        obs, _ = game.reset()
        terminated = False
        while not terminated:
            mask = game.get_action_mask()
            action = int(np.where(mask)[0][0])
            obs, reward, terminated, truncated, info = game.step(action)
        assert terminated

    def test_action_space_d_independent(self, catalog):
        """Action space is 7 for all D."""
        for D in [2, 3, 4, 5, 8]:
            game = _make_game(D, catalog)
            assert game.action_space.n == 7

    def test_reward_with_leaf_evaluator(self, catalog):
        """Game with ReactiveLeafEvaluator gives consistent reward."""
        game_inline = _make_game(2, catalog, use_leaf_evaluator=False)
        game_le = _make_game(2, catalog, use_leaf_evaluator=True)

        # Play canonical policy through both
        canonical_actions = [
            PRED_PICKABLE, ACT_PICK,
            PRED_KNOWN_LOC, ACT_GOTO_KEY,
            PRED_REACHABLE, ACT_GOTO_GOAL,
            PRED_EXISTS_FRONTIER, ACT_GOTO_ENTRANCE,
        ]

        game_inline.reset()
        for a in canonical_actions:
            _, r_inline, _, _, _ = game_inline.step(a)

        game_le.reset()
        for a in canonical_actions:
            _, r_le, _, _, _ = game_le.step(a)

        # Leaf evaluator uses "weighted" metric by default, inline uses raw reward
        # Both should produce positive reward for the canonical solver
        assert r_inline > 0
        assert r_le > 0
