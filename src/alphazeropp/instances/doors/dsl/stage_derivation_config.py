"""Configuration for StageDerivationGame on the Doors environment.

Builds a StageDerivationGame (stage-skeleton hole-filling) with
LeafEvaluator terminal rewards, reusing the existing AlphaZero
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
from alphazeropp.instances.doors.dsl.stage_derivation_game import StageDerivationGame
from alphazeropp.instances.doors.dsl.stage_search_cost import SearchCostModel
from alphazeropp.instances.doors.dsl.stage_hole_selection import (
    LeftmostPolicy, StructureFirstPolicy,
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
class DoorsStageDerivationConfig(MetaConfig):
    """Configuration for StageDerivationGame on Doors.

    The stage game fills holes in an if-elif-else skeleton:
      - StructureHole: add another stage or finalize
      - GuardHoleRef:  choose a guard atom (or composite)
      - ActionHoleRef: choose a typed action

    Action space = Discrete(2 + N_guards + N_actions) with masking.
    Episode length = 1 to 3*max_stages + 1 steps.
    """

    def __init__(
        self,
        num_rooms: int = 3,
        max_stages: int | None = None,
        max_guard_depth: int = 0,
        hole_policy: str = "leftmost",
    ):
        super().__init__()
        K = num_rooms - 1
        if max_stages is None:
            max_stages = 2 * K + 1

        max_decisions = 3 * max_stages + 1

        self.game = CoreGameConfig(
            game_cls=StageDerivationGame,
            kwargs={
                "num_rooms": num_rooms,
                "locs_per_room": 2,
                "horizon": max(15, (2 * (num_rooms - 1) + 1) * 5),
                "step_penalty": 0.01,
                "unlock_bonus": 0.1,
                # Stage-specific
                "max_stages": max_stages,
                "max_guard_depth": max_guard_depth,
                "hole_policy": hole_policy,
                "n_sites": max_decisions,  # for plot naming
                # LeafEvaluator sub-config
                "metric": "weighted",
                "penalty_lambda": 0.1,
                "blend_alpha": 0.7,
            },
        )
        self.net = NetConfig(
            net_cls=DerivationPolicyValueNet,
            kwargs={
                "budget": max_decisions,
                "n_sites": max_decisions,
                "action_size": 0,  # computed in build()
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
            plot_path="doors_stage_training_metrics.png",
        )

    def build(self):
        """Build StageDerivationGame, network, agent, trainer, evaluator."""
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

        # LeafEvaluator
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

        # Cost model + hole policy
        cost_model = SearchCostModel(
            max_stages=gk["max_stages"],
            max_guard_depth=gk["max_guard_depth"],
        )
        policy_name = gk.get("hole_policy", "leftmost")
        if policy_name == "structure_first":
            hole_policy = StructureFirstPolicy()
        else:
            hole_policy = LeftmostPolicy()

        # StageDerivationGame
        game = StageDerivationGame(
            doors_cfg=doors_cfg,
            cost_model=cost_model,
            leaf_evaluator=leaf_eval,
            hole_selection_policy=hole_policy,
        )

        # Network (action_size from game)
        max_decisions = 3 * gk["max_stages"] + 1
        nk = dict(self.net.kwargs)
        nk["budget"] = max_decisions
        nk["n_sites"] = max_decisions
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
