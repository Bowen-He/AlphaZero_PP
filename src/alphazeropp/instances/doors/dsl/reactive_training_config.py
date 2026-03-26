"""Configuration for ReactiveDerivationGame on the Doors environment.

Builds a ReactiveDerivationGame with ReactiveLeafEvaluator terminal rewards,
reusing the existing AlphaZero training infrastructure.

Follows DoorsSurfaceDerivationConfig pattern.
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
from alphazeropp.instances.doors.dsl.reactive_derivation_game import (
    ReactiveDerivationGame,
)
from alphazeropp.instances.doors.dsl.reactive_branch_catalog import (
    known_map_catalog,
)
from alphazeropp.instances.doors.dsl.reactive_leaf_evaluator import (
    ReactiveLeafEvaluator,
)
from alphazeropp.instances.doors.dsl.reactive_network import (
    ReactivePolicyValueNet,
)
from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.core.agent import Agent
from alphazeropp.training.trainer import Trainer
from alphazeropp.training.evaluator import Evaluator


@dataclass
class DoorsReactiveDerivationConfig(MetaConfig):
    """Configuration for ReactiveDerivationGame on Doors.

    The reactive game has a tiny, fixed action space (7 actions)
    and fixed-length episodes (2*n_branches steps). D-independent.
    """

    def __init__(
        self,
        num_rooms: int = 3,
        n_branches: int | None = None,
        catalog_mode: str = "typed",
        known_map: bool = True,
        pretrained_checkpoint: str | None = None,
        n_simulations: int = 40,
        n_games_per_train: int = 30,
        n_iterations: int = 20,
    ):
        super().__init__()
        self.num_rooms = num_rooms
        # Auto-scale: need at least 2K+1 branches for K=D-1 keys
        if n_branches is None:
            K = num_rooms - 1
            n_branches = max(4, 2 * K + 1)
        self.n_branches = n_branches
        self.catalog_mode = catalog_mode
        self.known_map = known_map
        self.pretrained_checkpoint = pretrained_checkpoint

        n_steps = 2 * n_branches  # episode length

        self.game = CoreGameConfig(
            game_cls=ReactiveDerivationGame,
            kwargs={
                "num_rooms": num_rooms,
                "n_branches": n_branches,
                "n_sites": n_steps,  # for derivation_utils title/plot naming
                "budget": n_steps,   # no AST budget, but needed for plot filename
                "catalog_mode": catalog_mode,
                "known_map": known_map,
                "metric": "weighted",
                "blend_alpha": 0.7,
            },
        )
        self.net = NetConfig(
            net_cls=ReactivePolicyValueNet,
            kwargs={
                "budget": n_steps,
                "n_sites": n_steps,
                "action_size": 7,  # max(n_predicates, n_actions)
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
                "dropout": 0.1,
                "aux_p_solve": True,
                "aux_best_action": False,
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
                "n_simulations": n_simulations,
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
            n_games_per_train=n_games_per_train,
            n_past_iterations_to_train=20,
            n_procs=3,  # 3 parallel self-play workers (use -1 for sequential debug)
            checkpoint_dir="checkpoints/reactive_stage4",
        )
        self.evaluator = EvaluatorConfig(
            n_games=10,
            n_procs=3,  # 3 parallel eval workers
        )
        self.run = RunConfig(
            n_iterations=n_iterations,
            accept_threshold=0.40,
            plot_every=5,
            plot_path="reactive_stage4_training_metrics.png",
        )

    def build(self):
        """Build ReactiveDerivationGame, network, agent, trainer, evaluator."""
        gk = self.game.kwargs

        # Build DoorsGameConfig
        params = compute_doors_derived_params(gk["num_rooms"], 2)
        doors_cfg = DoorsGameConfig(
            num_rooms=gk["num_rooms"],
            locs_per_room=2,
            horizon=params["horizon"],
        )

        # Build catalog
        catalog = known_map_catalog(mode=gk["catalog_mode"])

        # Build ReactiveLeafEvaluator
        x0 = doors_initial_state(doors_cfg)
        leaf_eval = ReactiveLeafEvaluator(
            catalog=catalog,
            doors_cfg=doors_cfg,
            frozen_states=[x0],
            is_solved=doors_cfg.is_solved,
            metric=gk.get("metric", "weighted"),
            blend_alpha=gk.get("blend_alpha", 0.7),
            known_map=gk.get("known_map", True),
        )

        # Build game
        game = ReactiveDerivationGame(
            doors_cfg=doors_cfg,
            catalog=catalog,
            n_branches=gk["n_branches"],
            leaf_evaluator=leaf_eval,
        )

        # Build network
        nk = dict(self.net.kwargs)
        nk["action_size"] = game.action_space.n
        # Strip keys not accepted by ReactivePolicyValueNet
        nk.pop("n_sites", None)
        net = ReactivePolicyValueNet(**nk)

        # Load pretrained checkpoint if provided
        if self.pretrained_checkpoint is not None:
            net.load_checkpoint(self.pretrained_checkpoint)

        # Build training stack
        agent = Agent(
            game=game,
            net=net,
            mcts_params=self.agent.mcts_params,
            reward_discount=self.agent.reward_discount,
            external_policy=getattr(self.agent, "external_policy", None),
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
