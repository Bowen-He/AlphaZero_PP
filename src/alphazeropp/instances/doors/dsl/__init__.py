"""Doors-specific DSL components (game config, derivation config, surface DSL).

This package currently exports the Surface DSL layer — typed guard/action
vocabulary with macro-level derivation game for AlphaZero MCTS training.

TODO: The following modules exist on this branch but are not yet exported
here.  They will be added in subsequent commits once validated:
  - stage_dsl / stage_compiler / stage_grammar  (stage-skeleton DSL)
  - relational_runtime  (typed relational query layer)
  - lifted_dsl / lifted_compiler  (D-agnostic lifted decision lists)
  - reactive_sketch_dsl / reactive_sketch_interpreter  (BT-style reactive policies)
"""

# ---------------------------------------------------------------------------
# Surface DSL — typed guards, actions, macros, grammar, derivation game
# ---------------------------------------------------------------------------
from alphazeropp.instances.doors.dsl.surface_dsl import (  # noqa: F401
    AtKeyLoc, KeyAvail, RoomLocked, PickReady, NeedKey,
    Pick, MoveToKey, MoveToGoal,
    PickRule, MoveRule, GoalRule,
    SurfacePolicy,
)
from alphazeropp.instances.doors.dsl.surface_compiler import (  # noqa: F401
    compile_condition, compile_action, compile_rule, compile_policy,
)
from alphazeropp.instances.doors.dsl.surface_grammar import (  # noqa: F401
    canonical_policy, count_relaxed_policies, enumerate_relaxed_policies,
)
from alphazeropp.instances.doors.dsl.surface_derivation_game import (  # noqa: F401
    SurfaceDerivationState, SurfaceDerivationGame,
)
from alphazeropp.instances.doors.dsl.surface_derivation_config import (  # noqa: F401
    DoorsSurfaceDerivationConfig,
)
