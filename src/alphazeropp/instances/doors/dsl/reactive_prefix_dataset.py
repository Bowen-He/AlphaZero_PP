"""Dataset for oracle-supervised pretraining of reactive derivation networks.

Converts prefix oracle entries into (obs, policy_target, value_target) triples
compatible with DerivationPolicyValueNet.train().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from alphazeropp.instances.doors.dsl.reactive_prefix_oracle import (
    PrefixOracleEntry,
)


@dataclass
class PrefixDatasetEntry:
    """One training example derived from a prefix oracle entry."""
    obs: np.ndarray               # game observation at this prefix
    legal_mask: np.ndarray        # legal action mask
    policy_target: np.ndarray     # normalized best_next_mask (probability distribution)
    value_target: float           # v_max from oracle
    p_solve: float                # oracle solve probability
    d: int                        # number of rooms
    n_branches: int
    mode: str                     # "typed" or "raw"
    decision_prefix: tuple[int, ...]  # decision sequence key


class ReactivePrefixDataset:
    """Collection of prefix dataset entries with save/load and conversion."""

    def __init__(self, entries: list[PrefixDatasetEntry]):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def save(self, path: Path | str):
        """Save as JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in self.entries:
                record = {
                    "obs": entry.obs.tolist(),
                    "legal_mask": entry.legal_mask.tolist(),
                    "policy_target": entry.policy_target.tolist(),
                    "value_target": entry.value_target,
                    "p_solve": entry.p_solve,
                    "d": entry.d,
                    "n_branches": entry.n_branches,
                    "mode": entry.mode,
                    "decision_prefix": list(entry.decision_prefix),
                }
                f.write(json.dumps(record) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> ReactivePrefixDataset:
        """Load from JSONL."""
        entries = []
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                entries.append(PrefixDatasetEntry(
                    obs=np.array(record["obs"], dtype=np.float32),
                    legal_mask=np.array(record["legal_mask"], dtype=bool),
                    policy_target=np.array(record["policy_target"], dtype=np.float32),
                    value_target=record["value_target"],
                    p_solve=record["p_solve"],
                    d=record["d"],
                    n_branches=record["n_branches"],
                    mode=record["mode"],
                    decision_prefix=tuple(record["decision_prefix"]),
                ))
        return cls(entries)

    def to_training_examples(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Convert to (state, policy, value) triples for net.train().

        Follows the format expected by DerivationPolicyValueNet.train():
        each example is (state, pi, v) where pi is a probability vector
        and v is a scalar value target.
        """
        examples = []
        for entry in self.entries:
            examples.append((
                entry.obs,
                entry.policy_target,
                entry.value_target,
            ))
        return examples

    @classmethod
    def merge(cls, *datasets: ReactivePrefixDataset) -> ReactivePrefixDataset:
        """Merge multiple datasets."""
        entries = []
        for ds in datasets:
            entries.extend(ds.entries)
        return cls(entries)


def build_dataset_from_oracle(
    oracle: dict[tuple, PrefixOracleEntry],
    game,
    d: int,
    n_branches: int,
    mode: str,
) -> ReactivePrefixDataset:
    """Convert oracle entries into dataset entries by replaying through the game.

    For each prefix in the oracle, replays the decision sequence through the
    game to obtain the observation vector and legal mask at that state.

    Args:
        oracle: Dict mapping decision_prefix → PrefixOracleEntry.
        game: ReactiveDerivationGame instance (will be reset for each entry).
        d: Number of rooms.
        n_branches: Number of branches.
        mode: Catalog mode ("typed" or "raw").

    Returns:
        ReactivePrefixDataset with one entry per non-terminal oracle node.
    """
    entries = []

    for decision_prefix, oracle_entry in oracle.items():
        # Skip terminal entries (complete policies — no next action to predict)
        if oracle_entry.n_completions == 1 and oracle_entry.pending_pred is None:
            # This is a complete policy if prefix_specs has n_branches entries
            if len(oracle_entry.prefix_specs) == n_branches:
                continue

        # Replay decision sequence through the game to get obs + legal_mask
        obs, _ = game.reset()
        for action in decision_prefix:
            obs, _, terminated, _, _ = game.step(action)
            if terminated:
                break

        if terminated:
            continue

        legal_mask = game.get_action_mask()

        # Build policy target: normalize best_next_mask to probability distribution
        best_mask = oracle_entry.best_next_mask.copy().astype(np.float32)
        # Intersect with legal mask (safety check)
        best_mask *= legal_mask.astype(np.float32)
        mask_sum = best_mask.sum()
        if mask_sum > 0:
            policy_target = best_mask / mask_sum
        else:
            # Fallback: uniform over legal actions
            legal_float = legal_mask.astype(np.float32)
            policy_target = legal_float / legal_float.sum() if legal_float.sum() > 0 else legal_float

        entries.append(PrefixDatasetEntry(
            obs=obs.copy(),
            legal_mask=legal_mask.copy(),
            policy_target=policy_target,
            value_target=oracle_entry.v_max,
            p_solve=oracle_entry.p_solve,
            d=d,
            n_branches=n_branches,
            mode=mode,
            decision_prefix=decision_prefix,
        ))

    return ReactivePrefixDataset(entries)
