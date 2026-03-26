"""Smoke tests for reactive AlphaZero training pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.reactive_training_config import (
    DoorsReactiveDerivationConfig,
)


class TestAlphaZeroSmoke:

    def test_config_build(self):
        """DoorsReactiveDerivationConfig.build() succeeds for D=2."""
        config = DoorsReactiveDerivationConfig(
            num_rooms=2, n_branches=4, n_simulations=5, n_iterations=1,
        )
        game, net, agent, trainer, evaluator = config.build()
        assert game.action_space.n == 7
        assert game.observation_space.shape == (16,)
        assert game.leaf_evaluator is not None

    def test_one_mcts_episode(self):
        """Agent plays one full game without error."""
        config = DoorsReactiveDerivationConfig(
            num_rooms=2, n_branches=4, n_simulations=5, n_iterations=1,
        )
        game, net, agent, trainer, evaluator = config.build()
        game.reset_wrapper()  # must reset before play_one_round
        experience, reward, infos = agent.play_one_round(game, max_moves=20)
        assert len(experience) > 0
        # Each experience is (obs, policy, discounted_reward)
        obs, policy, disc_reward = experience[0]
        assert obs.shape == (16,)
        assert len(policy) == 7

    def test_one_training_iteration(self):
        """Collect + train + evaluate runs without error."""
        config = DoorsReactiveDerivationConfig(
            num_rooms=2, n_branches=4, n_simulations=5,
            n_games_per_train=2, n_iterations=1,
        )
        game, net, agent, trainer, evaluator = config.build()

        # Collect experience (returns list of (exp_list, reward, infos, caches))
        raw_results = trainer._collect_training_examples()
        assert len(raw_results) > 0

        # Extract experience lists and process
        experience_lists = [r[0] for r in raw_results]
        flat = trainer._process_training_examples(experience_lists)
        assert len(flat) > 0

        # Train
        trainer._train_network(flat)

        # Network still produces valid predictions
        obs = np.zeros(16, dtype=np.float32)
        policy, value = net.predict(obs)
        assert policy.shape == (7,)
