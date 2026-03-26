"""Configuration for SurfaceDerivationGame on the Doors environment.

Builds a SurfaceDerivationGame (budget-free surface rule sequencing)
with LeafEvaluator terminal rewards, reusing the existing AlphaZero
training infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphazeropp.core.config import (
    MetaConfig,
    GameConfig as CoreGameConfig,
    NetConfig,
    AgentConfig,
    TrainerConfig,
    EvaluatorConfig,
    RunConfig,
)
from alphazeropp.instances.doors.dsl.surface_derivation_game import (
    SurfaceDerivationGame,
)
from alphazeropp.instances.doors.dsl.derivation_config import DoorsProgressFn
from alphazeropp.synthesis.derivation_network import DerivationPolicyValueNet
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator
from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state,
)
from alphazeropp.core.agent import Agent
from alphazeropp.training.trainer import Trainer
from alphazeropp.training.evaluator import Evaluator


@dataclass
class DoorsSurfaceDerivationConfig(MetaConfig):
    """Configuration for SurfaceDerivationGame on Doors.

    The surface game has a tiny, fixed action space (2K+1 actions)
    and fixed-length episodes (2K+1 steps). No budget, no dead ends.
    """

    def __init__(self, num_rooms: int = 3):
        super().__init__()
        K = num_rooms - 1
        max_steps = 2 * K + 1  # sequence length and action count

        self.game = CoreGameConfig(
            game_cls=SurfaceDerivationGame,
            kwargs={
                "num_rooms": num_rooms,
                "locs_per_room": 2,
                "horizon": max(15, (2 * (num_rooms - 1) + 1) * 5),
                "step_penalty": 0.01,
                "unlock_bonus": 0.1,
                "allow_early_goal": False,
                "n_sites": max_steps,  # needed by rename_plot_with_stats
                # LeafEvaluator sub-config
                "metric": "weighted",
                "penalty_lambda": 0.1,
                "blend_alpha": 0.7,
            },
        )
        self.net = NetConfig(
            net_cls=DerivationPolicyValueNet,
            kwargs={
                "budget": max_steps,  # sequence length for transformer
                "n_sites": max_steps,  # token vocabulary context
                "action_size": max_steps,
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
                "dropout": 0.1,
                "training_params": {
                    "epochs": 5,
                    "batch_size": 32,
                    "learning_rate": 3e-4,
                    "weight_decay": 1e-4,
                    "policy_weight": 2.0,
                },
            },
        )
        self.agent = AgentConfig(
            mcts_params={
                "n_simulations": 80,
                "temperature": 1.0,
                "c_exploration": 1.5,
                "dirichlet_alpha": 0.25,
                "dirichlet_epsilon": 0.40,
                "rollout_n": 4,
                "rollout_mode": "max",
                "rollout_blend": 0.3,
                "rollout_budget": 200,
                "backup_rule": "max",
                "backup_topk": 3,
                "backup_tau": 0.1,
            },
            reward_discount=1.0,
            random_seeds={
                "mcts": 43,
                "train": 47,
                "eval": 23,
                "external_policy": 68,
            },
        )
        self.trainer = TrainerConfig(
            n_games_per_train=30,
            n_past_iterations_to_train=20,
            n_procs=8,
            checkpoint_dir="checkpoints",
        )
        self.evaluator = EvaluatorConfig(
            n_games=20,
            n_procs=8,
        )
        self.run = RunConfig(
            n_iterations=30,
            accept_threshold=0.40,
            plot_every=5,
            plot_path="doors_surface_training_metrics.png",
        )

    def build(self):
        """Build SurfaceDerivationGame, network, agent, trainer, evaluator."""
        gk = self.game.kwargs

        # Build DoorsGameConfig for environment / LeafEvaluator
        doors_cfg = DoorsGameConfig(
            num_rooms=gk["num_rooms"],
            locs_per_room=gk.get("locs_per_room", 2),
            horizon=gk["horizon"],
            step_penalty=gk["step_penalty"],
            unlock_bonus=gk["unlock_bonus"],
        )
        n_sites = doors_cfg.obs_size()

        # LeafEvaluator (reuses existing infrastructure)
        leaf_eval = LeafEvaluator(
            n_sites,
            [doors_initial_state(doors_cfg)],
            doors_cfg,
            metric=gk["metric"],
            penalty_lambda=gk["penalty_lambda"],
            blend_alpha=gk["blend_alpha"],
            is_solved=doors_cfg.is_solved,
            progress_fn=DoorsProgressFn(doors_cfg),
        )

        # SurfaceDerivationGame
        game = SurfaceDerivationGame(
            num_rooms=gk["num_rooms"],
            leaf_evaluator=leaf_eval,
            doors_cfg=doors_cfg,
            allow_early_goal=gk.get("allow_early_goal", False),
        )

        # Network (reuse DerivationPolicyValueNet with surface dimensions)
        nk = dict(self.net.kwargs)
        nk["action_size"] = game.action_space.n
        net = DerivationPolicyValueNet(**nk)

        # Training stack
        agent = Agent(
            game=game,
            net=net,
            mcts_params=self.agent.mcts_params,
            reward_discount=self.agent.reward_discount,
            external_policy=self.agent.external_policy,
            random_seeds=self.agent.random_seeds,
        )
        trainer = Trainer(
            agent=agent,
            net=net,
            game=game,
            n_games_per_train=self.trainer.n_games_per_train,
            n_past_iterations_to_train=self.trainer.n_past_iterations_to_train,
            n_procs=self.trainer.n_procs,
            checkpoint_dir=self.trainer.checkpoint_dir,
            use_tree_reuse=True,
        )
        evaluator = Evaluator(
            n_games=self.evaluator.n_games,
            n_procs=self.evaluator.n_procs,
        )

        return game, net, agent, trainer, evaluator
