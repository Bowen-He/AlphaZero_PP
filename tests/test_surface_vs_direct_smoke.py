"""Smoke tests: SurfaceDerivationGame and DoorsDirectGame plug into Agent/Trainer."""

from __future__ import annotations

import pytest
import numpy as np

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.surface_derivation_game import (
    SurfaceDerivationGame,
)
from alphazeropp.instances.doors.game import DoorsDirectGame
from alphazeropp.synthesis.derivation_game import UniformPolicyValueNet
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator
from alphazeropp.core.agent import Agent


def _make_surface_game(D: int) -> SurfaceDerivationGame:
    params = compute_doors_derived_params(D, 2)
    cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2, horizon=params["horizon"])
    n_sites = cfg.obs_size()
    leaf_eval = LeafEvaluator(
        n_sites, [doors_initial_state(cfg)], cfg,
        is_solved=cfg.is_solved, metric="solve_rate",
    )
    return SurfaceDerivationGame(num_rooms=D, leaf_evaluator=leaf_eval, doors_cfg=cfg)


def _make_direct_game(D: int) -> DoorsDirectGame:
    return DoorsDirectGame(num_rooms=D, locs_per_room=2)


class TestSmoke:
    """Both game types plug into Agent with UniformPolicyValueNet."""

    def test_surface_game_plugs_into_agent(self):
        game = _make_surface_game(2)
        net = UniformPolicyValueNet(game.action_space.n)
        agent = Agent(
            game=game, net=net,
            mcts_params={"n_simulations": 5, "temperature": 1.0},
            reward_discount=1.0,
        )
        game.reset_wrapper()
        experience, cum_reward, step_infos = agent.play_one_round(
            game, max_moves=50,
        )
        assert len(experience) > 0
        # Each experience entry: (obs, policy, value)
        obs, policy, value = experience[0]
        assert obs.shape == game.observation_space.shape
        assert len(policy) == game.action_space.n

    def test_direct_game_plugs_into_agent(self):
        game = _make_direct_game(2)
        net = UniformPolicyValueNet(game.action_space.n)
        agent = Agent(
            game=game, net=net,
            mcts_params={"n_simulations": 5, "temperature": 1.0},
            reward_discount=1.0,
        )
        game.reset_wrapper()
        experience, cum_reward, step_infos = agent.play_one_round(
            game, max_moves=50,
        )
        assert len(experience) > 0

    def test_surface_experience_format(self):
        """Verify surface game produces training-compatible examples."""
        game = _make_surface_game(3)
        net = UniformPolicyValueNet(game.action_space.n)
        agent = Agent(
            game=game, net=net,
            mcts_params={"n_simulations": 5, "temperature": 1.0},
            reward_discount=1.0,
        )
        game.reset_wrapper()
        experience, cum_reward, step_infos = agent.play_one_round(
            game, max_moves=50,
        )
        # D=3, K=2: exactly 5 steps
        assert len(experience) == 5
        for obs, policy, value in experience:
            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert isinstance(policy, (list, np.ndarray))
            assert isinstance(value, (int, float, np.floating))
