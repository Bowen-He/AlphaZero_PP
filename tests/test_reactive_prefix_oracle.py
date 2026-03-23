"""Tests for the reactive prefix oracle."""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state,
)
from alphazeropp.instances.doors.dsl.reactive_branch_catalog import (
    known_map_catalog,
)
from alphazeropp.instances.doors.dsl.reactive_leaf_evaluator import (
    ReactiveLeafEvaluator,
)
from alphazeropp.instances.doors.dsl.reactive_prefix_oracle import (
    build_prefix_oracle,
)


@pytest.fixture
def oracle_d2():
    cfg = DoorsGameConfig(num_rooms=2, locs_per_room=2)
    catalog = known_map_catalog(mode="typed")
    evaluator = ReactiveLeafEvaluator(
        catalog, cfg, [doors_initial_state(cfg)], is_solved=cfg.is_solved,
    )
    return build_prefix_oracle(4, catalog, evaluator), evaluator, catalog


class TestPrefixOracle:

    def test_complete_policies_evaluated(self, oracle_d2):
        """All 4096 complete policies are in the oracle."""
        oracle, evaluator, catalog = oracle_d2
        root = oracle[()]
        assert root.n_completions == 4096

    def test_v_max_at_root(self, oracle_d2):
        """V_max at root equals best complete policy reward."""
        oracle, evaluator, catalog = oracle_d2
        root = oracle[()]
        # Evaluate the canonical solving policy directly
        from alphazeropp.instances.doors.dsl.reactive_branch_catalog import (
            PRED_PICKABLE, PRED_KNOWN_LOC, PRED_REACHABLE, PRED_EXISTS_FRONTIER,
            ACT_PICK, ACT_GOTO_KEY, ACT_GOTO_GOAL, ACT_GOTO_ENTRANCE,
        )
        canonical = ((PRED_PICKABLE, ACT_PICK), (PRED_KNOWN_LOC, ACT_GOTO_KEY),
                      (PRED_REACHABLE, ACT_GOTO_GOAL), (PRED_EXISTS_FRONTIER, ACT_GOTO_ENTRANCE))
        direct = evaluator(canonical)
        assert root.v_max == pytest.approx(direct, abs=1e-6)

    def test_p_solve_at_root(self, oracle_d2):
        """P_solve at root equals n_solving / 4096."""
        oracle, evaluator, catalog = oracle_d2
        root = oracle[()]
        assert root.p_solve == pytest.approx(root.n_solving / 4096)
        assert root.n_solving == 104

    def test_best_next_mask_nonempty(self, oracle_d2):
        """Root has at least one best-next action."""
        oracle, evaluator, catalog = oracle_d2
        root = oracle[()]
        assert root.best_next_mask.any()

    def test_prefix_v_max_monotonicity(self, oracle_d2):
        """V_max never increases as prefix grows (child <= parent)."""
        oracle, evaluator, catalog = oracle_d2
        violations = 0
        for key, entry in oracle.items():
            if len(key) == 0:
                continue
            parent_key = key[:-1]
            if parent_key in oracle:
                if entry.v_max > oracle[parent_key].v_max + 1e-9:
                    violations += 1
        assert violations == 0

    def test_terminal_prefix_matches_eval(self, oracle_d2):
        """Complete prefix V_max equals direct evaluator result."""
        oracle, evaluator, catalog = oracle_d2
        # Find a terminal entry (complete policy)
        terminal_entries = [
            (k, e) for k, e in oracle.items()
            if e.n_completions == 1 and len(e.prefix_specs) == 4
        ]
        assert len(terminal_entries) > 0
        key, entry = terminal_entries[0]
        direct = evaluator(entry.prefix_specs)
        assert entry.v_max == pytest.approx(direct, abs=1e-6)
