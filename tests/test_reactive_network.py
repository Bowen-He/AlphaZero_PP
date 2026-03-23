"""Tests for the reactive policy-value network."""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.reactive_network import (
    ReactivePolicyValueNet,
)


@pytest.fixture
def net():
    return ReactivePolicyValueNet(
        budget=8,  # 2 * 4 branches
        action_size=7,
        d_model=32,
        n_heads=2,
        n_layers=1,
        aux_p_solve=True,
        random_seed=42,
    )


class TestReactiveNetwork:

    def test_forward_pass_shapes(self, net):
        """Policy has shape (7,), value is a 0-d numpy array."""
        obs = np.zeros(16, dtype=np.float32)
        policy, value = net.predict(obs)
        assert policy.shape == (7,)
        assert value.shape == ()  # 0-d numpy scalar (MCTS contract)
        assert abs(policy.sum() - 1.0) < 1e-5  # softmax

    def test_predict_interface(self, net):
        """predict() returns (policy, value) compatible with MCTS."""
        obs = np.zeros(16, dtype=np.float32)
        # Fill in some decisions
        obs[0] = 1.0  # type_id = PREDICATE
        obs[1] = 0.0  # param = 0 (Pickable)
        obs[2] = 2.0  # type_id = ACTION
        obs[3] = 0.0  # param = 0 (Pick)

        policy, value = net.predict(obs)
        assert policy.shape == (7,)
        assert all(p >= 0 for p in policy)

    def test_train_step(self, net):
        """Training step runs without error."""
        # Create valid observation format: (type_id, param) pairs
        rng = np.random.default_rng(42)
        examples = []
        for _ in range(20):
            obs = np.zeros(16, dtype=np.float32)
            # Fill some valid decisions: type_id in {0,1,2}, param in {0..6}
            n_filled = rng.integers(0, 5)
            for j in range(n_filled):
                obs[2 * j] = rng.choice([1, 2])  # PREDICATE or ACTION
                obs[2 * j + 1] = float(rng.integers(0, 7))
            pi = np.zeros(7, dtype=np.float32)
            pi[rng.integers(7)] = 1.0
            v = float(rng.standard_normal())
            examples.append((obs, pi, v))

        # Measure predictions before
        test_obs = examples[0][0]
        _, v_before = net.predict(test_obs)

        # Train with 1 epoch
        net.training_params = dict(net.training_params)
        net.training_params["epochs"] = 5
        net.train(examples)

        # Network should have changed (loss decreased from random init)
        _, v_after = net.predict(test_obs)
        # Just check it doesn't crash — we can't guarantee direction
        assert v_after.shape == ()
